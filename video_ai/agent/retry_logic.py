"""
Retry Logic - Fault-tolerant generation with automatic recovery

This module handles:
- Automatic retry on failures
- Dynamic parameter adjustment on OOM
- Fallback strategies
- Error classification and handling
"""

import time
import logging
import traceback
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of generation errors"""
    OOM = auto()              # Out of memory
    MODEL_LOAD = auto()       # Failed to load model
    INFERENCE = auto()        # Inference failure
    ENCODING = auto()         # Video encoding failure
    TIMEOUT = auto()          # Operation timed out
    INVALID_INPUT = auto()    # Invalid input parameters
    UNKNOWN = auto()          # Unknown error


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 5
    retry_on_oom: bool = True
    reduce_resolution_on_retry: bool = True
    reduce_frames_on_retry: bool = True
    resolution_reduction_factor: float = 0.65   # legacy; adjust_parameters_for_oom uses its own
    frame_reduction_factor: float = 0.60        # legacy; adjust_parameters_for_oom uses its own
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    exponential_backoff: bool = True


@dataclass
class RetryAttempt:
    """Record of a single retry attempt"""
    attempt_number: int
    success: bool
    error_type: Optional[ErrorType]
    error_message: str
    duration_seconds: float
    parameters_used: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    result: Any
    attempts: List[RetryAttempt]
    total_duration: float
    final_error: Optional[str]
    parameters_adjusted: bool


class RetryManager:
    """
    Manages retry logic with intelligent parameter adjustment.
    
    Handles automatic retries with:
    - Exponential backoff
    - Resolution/frame reduction on OOM
    - Error classification
    - Detailed logging
    """
    
    # Error patterns for classification
    OOM_PATTERNS = [
        'out of memory',
        'oom',
        'cuda out of memory',
        'cudaerrormemoryal',
        'directml',
        'memory allocation',
        'insufficient memory',
        'cannot allocate',
        'cudnn_status_internal_error',
        'host_allocation_failed',
        'cuda error',
        'cudnn error',
    ]
    
    MODEL_LOAD_PATTERNS = [
        'failed to load',
        'model not found',
        'invalid model',
        'corrupted',
        'onnx'
    ]
    
    TIMEOUT_PATTERNS = [
        'timeout',
        'timed out',
        'deadline exceeded'
    ]
    
    def __init__(self, config: Optional[RetryConfig] = None, config_dir: Optional[Path] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
            config_dir: Path to configuration directory
        """
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_dir)
        
        self.attempt_history: List[RetryAttempt] = []
    
    def _load_config(self, config_dir: Optional[Path]) -> RetryConfig:
        """Load retry configuration from YAML"""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        else:
            config_dir = Path(config_dir)
        
        defaults_path = config_dir / "defaults.yaml"
        
        if defaults_path.exists():
            with open(defaults_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                retry_config = data.get('retry', {})
                return RetryConfig(
                    max_attempts=retry_config.get('max_attempts', 3),
                    retry_on_oom=retry_config.get('retry_on_oom', True),
                    reduce_resolution_on_retry=retry_config.get('reduce_resolution_on_retry', True),
                    reduce_frames_on_retry=retry_config.get('reduce_frames_on_retry', True)
                )
        
        return RetryConfig()
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an error into an ErrorType.
        
        Args:
            error: The exception to classify
            
        Returns:
            Classified ErrorType
        """
        error_str = str(error).lower()
        
        for pattern in self.OOM_PATTERNS:
            if pattern in error_str:
                return ErrorType.OOM
        
        for pattern in self.MODEL_LOAD_PATTERNS:
            if pattern in error_str:
                return ErrorType.MODEL_LOAD
        
        for pattern in self.TIMEOUT_PATTERNS:
            if pattern in error_str:
                return ErrorType.TIMEOUT
        
        return ErrorType.UNKNOWN
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """
        Determine if we should retry based on error type and attempt count.
        
        Args:
            error_type: Type of error encountered
            attempt: Current attempt number
            
        Returns:
            True if should retry
        """
        if attempt >= self.config.max_attempts:
            return False
        
        # Always retry OOM if configured
        if error_type == ErrorType.OOM and self.config.retry_on_oom:
            return True
        
        # Retry unknown errors
        if error_type == ErrorType.UNKNOWN:
            return True
        
        # Retry inference errors
        if error_type == ErrorType.INFERENCE:
            return True
        
        # Don't retry invalid input or model load failures
        if error_type in [ErrorType.INVALID_INPUT, ErrorType.MODEL_LOAD]:
            return False
        
        return True
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry.
        
        Args:
            attempt: Current attempt number (1-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.config.exponential_backoff:
            delay = self.config.base_delay_seconds * (2 ** (attempt - 1))
        else:
            delay = self.config.base_delay_seconds
        
        return min(delay, self.config.max_delay_seconds)
    
    def adjust_parameters_for_oom(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggressively adjust generation parameters after an OOM crash.

        Wan2.1 VRAM usage scales roughly as  W × H × num_frames.
        A gentle 0.75× is rarely enough; we cut the pixel budget in
        half and trim frames significantly so the next attempt has a
        realistic chance of fitting in VRAM.
        """
        adjusted = params.copy()

        # ── 1. Free leaked CUDA memory from the failed attempt ──
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache after OOM")
        except Exception:
            pass

        # ── 2. Aggressively reduce resolution (≈ halve pixel count) ──
        OOM_SCALE = 0.65   # ~42 % fewer pixels per retry
        if self.config.reduce_resolution_on_retry:
            if 'width' in adjusted:
                w = int(adjusted['width'] * OOM_SCALE)
                adjusted['width'] = max(256, (w // 16) * 16)   # Wan2.1: multiples of 16
            if 'height' in adjusted:
                h = int(adjusted['height'] * OOM_SCALE)
                adjusted['height'] = max(256, (h // 16) * 16)

        # ── 3. Reduce frame count (keep 4k+1 for Wan2.1) ──
        FRAME_SCALE = 0.60
        if self.config.reduce_frames_on_retry and 'num_frames' in adjusted:
            nf = int(adjusted['num_frames'] * FRAME_SCALE)
            nf = max(5, ((nf - 1) // 4) * 4 + 1)   # Wan2.1: 4k+1
            adjusted['num_frames'] = nf

        # ── 4. Cap at known-safe maximums for ≤16 GB VRAM ──
        MAX_PIXELS = 832 * 480     # ~400 K — safe for 16 GB
        cur_pixels = adjusted.get('width', 832) * adjusted.get('height', 480)
        if cur_pixels > MAX_PIXELS:
            ratio = (MAX_PIXELS / cur_pixels) ** 0.5
            adjusted['width']  = max(256, (int(adjusted['width']  * ratio) // 16) * 16)
            adjusted['height'] = max(256, (int(adjusted['height'] * ratio) // 16) * 16)

        if adjusted.get('num_frames', 33) > 49:
            adjusted['num_frames'] = 49   # 3 sec max after OOM

        # ── 5. Reduce batch size if present ──
        if 'batch_size' in adjusted and adjusted['batch_size'] > 1:
            adjusted['batch_size'] = 1

        logger.info(f"Adjusted parameters for OOM: {adjusted}")
        return adjusted
    
    def execute_with_retry(
        self,
        func: Callable,
        params: Dict[str, Any],
        description: str = "operation"
    ) -> RetryResult:
        """
        Execute a function with automatic retry logic.
        
        Args:
            func: Function to execute (should accept **kwargs)
            params: Parameters to pass to function
            description: Description for logging
            
        Returns:
            RetryResult with outcome details
        """
        attempts: List[RetryAttempt] = []
        current_params = params.copy()
        start_time = time.time()
        result = None
        final_error = None
        parameters_adjusted = False
        
        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            logger.info(f"Attempt {attempt}/{self.config.max_attempts} for {description}")
            
            try:
                result = func(**current_params)
                
                # Success!
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    success=True,
                    error_type=None,
                    error_message="",
                    duration_seconds=time.time() - attempt_start,
                    parameters_used=current_params.copy()
                )
                attempts.append(attempt_record)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_duration=time.time() - start_time,
                    final_error=None,
                    parameters_adjusted=parameters_adjusted
                )
                
            except Exception as e:
                error_type = self.classify_error(e)
                error_msg = str(e)
                
                logger.warning(
                    f"Attempt {attempt} failed with {error_type.name}: {error_msg}"
                )
                
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    success=False,
                    error_type=error_type,
                    error_message=error_msg,
                    duration_seconds=time.time() - attempt_start,
                    parameters_used=current_params.copy()
                )
                attempts.append(attempt_record)
                final_error = error_msg
                
                # Check if we should retry
                if not self.should_retry(error_type, attempt):
                    logger.error(f"Not retrying: {error_type.name}")
                    break
                
                # Adjust parameters for OOM
                if error_type == ErrorType.OOM:
                    current_params = self.adjust_parameters_for_oom(current_params)
                    parameters_adjusted = True

                # On any CUDA error, try to recover GPU state
                if error_type in (ErrorType.OOM, ErrorType.UNKNOWN):
                    try:
                        import torch, gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                
                # Wait before retry — longer after OOM to let GPU settle
                delay = self.get_delay(attempt)
                if error_type == ErrorType.OOM:
                    delay = max(delay, 5.0)   # minimum 5 s after OOM
                logger.info(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
        
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_duration=time.time() - start_time,
            final_error=final_error,
            parameters_adjusted=parameters_adjusted
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics from history"""
        if not self.attempt_history:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'avg_attempts_per_success': 0.0,
                'error_breakdown': {}
            }
        
        successful = [a for a in self.attempt_history if a.success]
        failed = [a for a in self.attempt_history if not a.success]
        
        error_breakdown = {}
        for attempt in failed:
            if attempt.error_type:
                error_name = attempt.error_type.name
                error_breakdown[error_name] = error_breakdown.get(error_name, 0) + 1
        
        return {
            'total_attempts': len(self.attempt_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.attempt_history) if self.attempt_history else 0,
            'error_breakdown': error_breakdown
        }
    
    def clear_history(self):
        """Clear attempt history"""
        self.attempt_history.clear()


class FallbackChain:
    """
    Chain of fallback strategies for generation.
    
    Tries progressively simpler generation strategies
    when more complex ones fail.
    """
    
    def __init__(self):
        self.strategies: List[Dict[str, Any]] = []
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Set up default fallback strategies"""
        self.strategies = [
            {
                'name': 'full_quality',
                'description': 'Full quality DirectML generation',
                'params': {
                    'backend': 'directml',
                    'precision': 'fp16',
                    'batch_size': 1
                }
            },
            {
                'name': 'reduced_resolution',
                'description': 'Reduced resolution DirectML',
                'params': {
                    'backend': 'directml',
                    'precision': 'fp16',
                    'resolution_scale': 0.75,
                    'batch_size': 1
                }
            },
            {
                'name': 'cpu_full',
                'description': 'CPU inference at full resolution',
                'params': {
                    'backend': 'cpu',
                    'precision': 'fp32',
                    'batch_size': 1
                }
            },
            {
                'name': 'cpu_reduced',
                'description': 'CPU inference at reduced resolution',
                'params': {
                    'backend': 'cpu',
                    'precision': 'fp32',
                    'resolution_scale': 0.5,
                    'batch_size': 1
                }
            }
        ]
    
    def execute_with_fallback(
        self,
        func: Callable,
        base_params: Dict[str, Any]
    ) -> RetryResult:
        """
        Execute with fallback chain.
        
        Args:
            func: Function to execute
            base_params: Base parameters
            
        Returns:
            RetryResult
        """
        attempts = []
        start_time = time.time()
        
        for strategy in self.strategies:
            logger.info(f"Trying strategy: {strategy['name']}")
            
            # Merge strategy params with base params
            params = base_params.copy()
            params.update(strategy['params'])
            
            # Apply resolution scaling if specified
            if 'resolution_scale' in strategy['params']:
                scale = strategy['params']['resolution_scale']
                if 'width' in params:
                    params['width'] = int(params['width'] * scale)
                if 'height' in params:
                    params['height'] = int(params['height'] * scale)
            
            attempt_start = time.time()
            
            try:
                result = func(**params)
                
                attempts.append(RetryAttempt(
                    attempt_number=len(attempts) + 1,
                    success=True,
                    error_type=None,
                    error_message="",
                    duration_seconds=time.time() - attempt_start,
                    parameters_used=params
                ))
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_duration=time.time() - start_time,
                    final_error=None,
                    parameters_adjusted=strategy['name'] != 'full_quality'
                )
                
            except Exception as e:
                logger.warning(f"Strategy {strategy['name']} failed: {e}")
                
                attempts.append(RetryAttempt(
                    attempt_number=len(attempts) + 1,
                    success=False,
                    error_type=ErrorType.UNKNOWN,
                    error_message=str(e),
                    duration_seconds=time.time() - attempt_start,
                    parameters_used=params
                ))
        
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_duration=time.time() - start_time,
            final_error="All fallback strategies exhausted",
            parameters_adjusted=True
        )


def main():
    """Test retry logic"""
    logging.basicConfig(level=logging.INFO)
    
    # Create a function that fails sometimes
    fail_count = [0]
    
    def flaky_function(**kwargs):
        fail_count[0] += 1
        if fail_count[0] < 3:
            raise MemoryError("Out of memory!")
        return f"Success with params: {kwargs}"
    
    # Test retry manager
    manager = RetryManager()
    
    result = manager.execute_with_retry(
        flaky_function,
        {'width': 1280, 'height': 720, 'num_frames': 100},
        description="test generation"
    )
    
    print(f"\nResult: {'Success' if result.success else 'Failed'}")
    print(f"Total attempts: {len(result.attempts)}")
    print(f"Duration: {result.total_duration:.2f}s")
    print(f"Parameters adjusted: {result.parameters_adjusted}")
    if result.success:
        print(f"Result: {result.result}")


if __name__ == "__main__":
    main()
