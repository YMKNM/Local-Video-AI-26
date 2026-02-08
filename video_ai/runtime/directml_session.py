"""
DirectML Session Manager - GPU acceleration for AMD on Windows

This module handles:
- DirectML execution provider setup
- Session configuration and optimization
- GPU device management
- Memory management
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

logger = logging.getLogger(__name__)


class DirectMLSession:
    """
    Manages ONNX Runtime sessions with DirectML execution provider.
    
    DirectML is the only stable GPU backend for AMD GPUs on Windows.
    This class abstracts the session creation and configuration.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize DirectML session manager.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.hardware_config = self._load_hardware_config()
        
        self._ort = None
        self._available_providers = None
        self._active_sessions: Dict[str, Any] = {}
        
        # Initialize ONNX Runtime
        self._init_onnxruntime()
    
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware configuration"""
        config_path = self.config_dir / "hardware.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _init_onnxruntime(self):
        """Initialize ONNX Runtime and check providers"""
        try:
            import onnxruntime as ort
            self._ort = ort
            self._available_providers = ort.get_available_providers()
            
            logger.info(f"ONNX Runtime version: {ort.__version__}")
            logger.info(f"Available providers: {self._available_providers}")
            
            if 'DmlExecutionProvider' not in self._available_providers:
                logger.warning(
                    "DirectML not available! Install onnxruntime-directml: "
                    "pip install onnxruntime-directml"
                )
            
        except ImportError:
            logger.error(
                "ONNX Runtime not installed! Install with: "
                "pip install onnxruntime-directml"
            )
            raise RuntimeError("ONNX Runtime not available")
    
    @property
    def is_directml_available(self) -> bool:
        """Check if DirectML is available"""
        return 'DmlExecutionProvider' in (self._available_providers or [])
    
    @property
    def available_providers(self) -> List[str]:
        """Get list of available execution providers"""
        return self._available_providers or []
    
    def get_session_options(self, optimize: bool = True) -> Any:
        """
        Create optimized session options.
        
        Args:
            optimize: Whether to enable optimizations
            
        Returns:
            ONNX Runtime SessionOptions
        """
        opts = self._ort.SessionOptions()
        
        if optimize:
            # Enable optimizations
            opts.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable memory optimization
            opts.enable_mem_pattern = True
            opts.enable_cpu_mem_arena = True
            
            # Set execution mode
            opts.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Logging level
            opts.log_severity_level = 2  # Warning
        
        # Set number of threads for CPU operations
        cpu_config = self.hardware_config.get('cpu_fallback', {})
        num_threads = cpu_config.get('num_threads', 8)
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        
        return opts
    
    def get_provider_options(self) -> List[tuple]:
        """
        Get execution providers with options.
        
        Returns priority-ordered list of providers with their options.
        DirectML is preferred, with CPU as fallback.
        """
        providers = []
        
        # DirectML provider (primary for AMD)
        if self.is_directml_available:
            dml_config = self.hardware_config.get('directml', {})
            gpu_config = self.hardware_config.get('gpu', {})
            
            dml_options = {
                'device_id': gpu_config.get('device_id', 0),
            }
            
            # Add optional DirectML settings
            if not dml_config.get('disable_metacommands', False):
                dml_options['disable_metacommands'] = False
            
            providers.append(('DmlExecutionProvider', dml_options))
            logger.info(f"DirectML provider configured: {dml_options}")
        
        # CPU provider (fallback)
        cpu_options = {}
        providers.append(('CPUExecutionProvider', cpu_options))
        
        return providers
    
    def create_session(
        self,
        model_path: str,
        session_name: str = "default",
        force_cpu: bool = False
    ) -> Any:
        """
        Create an ONNX Runtime inference session.
        
        Args:
            model_path: Path to ONNX model file
            session_name: Name for tracking the session
            force_cpu: Force CPU execution (skip DirectML)
            
        Returns:
            ONNX Runtime InferenceSession
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Creating session for: {model_path.name}")
        
        # Get session options
        sess_options = self.get_session_options()
        
        # Get providers
        if force_cpu:
            providers = [('CPUExecutionProvider', {})]
        else:
            providers = self.get_provider_options()
        
        # Create session
        try:
            session = self._ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Log which provider is being used
            active_provider = session.get_providers()[0] if session.get_providers() else "Unknown"
            logger.info(f"Session created with provider: {active_provider}")
            
            # Track session
            self._active_sessions[session_name] = {
                'session': session,
                'model_path': str(model_path),
                'provider': active_provider
            }
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            
            # Try CPU fallback
            if not force_cpu and self.hardware_config.get('cpu_fallback', {}).get('enabled', True):
                logger.warning("Attempting CPU fallback...")
                return self.create_session(model_path, session_name, force_cpu=True)
            
            raise
    
    def get_session(self, session_name: str) -> Optional[Any]:
        """Get an existing session by name"""
        session_info = self._active_sessions.get(session_name)
        return session_info['session'] if session_info else None
    
    def close_session(self, session_name: str):
        """Close and remove a session"""
        if session_name in self._active_sessions:
            del self._active_sessions[session_name]
            logger.info(f"Closed session: {session_name}")
    
    def close_all_sessions(self):
        """Close all active sessions"""
        session_names = list(self._active_sessions.keys())
        for name in session_names:
            self.close_session(name)
        logger.info("All sessions closed")
    
    def get_session_info(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        session_info = self._active_sessions.get(session_name)
        if not session_info:
            return None
        
        session = session_info['session']
        
        return {
            'name': session_name,
            'model_path': session_info['model_path'],
            'provider': session_info['provider'],
            'inputs': [
                {
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                }
                for inp in session.get_inputs()
            ],
            'outputs': [
                {
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                }
                for out in session.get_outputs()
            ]
        }
    
    def run_inference(
        self,
        session_name: str,
        inputs: Dict[str, Any],
        output_names: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Run inference on a session.
        
        Args:
            session_name: Name of the session
            inputs: Dictionary of input name -> numpy array
            output_names: Optional list of output names (None = all)
            
        Returns:
            List of output arrays
        """
        session = self.get_session(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        
        return session.run(output_names, inputs)
    
    def benchmark_session(
        self,
        session_name: str,
        inputs: Dict[str, Any],
        num_iterations: int = 10,
        warmup: int = 2
    ) -> Dict[str, float]:
        """
        Benchmark a session's inference performance.
        
        Args:
            session_name: Session to benchmark
            inputs: Input data
            num_iterations: Number of iterations
            warmup: Warmup iterations (not counted)
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        import numpy as np
        
        session = self.get_session(session_name)
        if session is None:
            raise ValueError(f"Session not found: {session_name}")
        
        times = []
        
        # Warmup
        for _ in range(warmup):
            session.run(None, inputs)
        
        # Benchmark
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean_ms': float(times.mean() * 1000),
            'std_ms': float(times.std() * 1000),
            'min_ms': float(times.min() * 1000),
            'max_ms': float(times.max() * 1000),
            'iterations': num_iterations
        }


def main():
    """Test DirectML session"""
    logging.basicConfig(level=logging.INFO)
    
    session_manager = DirectMLSession()
    
    print("\n=== DirectML Session Manager ===")
    print(f"DirectML available: {session_manager.is_directml_available}")
    print(f"Providers: {session_manager.available_providers}")
    
    # Show provider configuration
    providers = session_manager.get_provider_options()
    print("\nConfigured providers:")
    for provider, options in providers:
        print(f"  {provider}: {options}")


if __name__ == "__main__":
    main()
