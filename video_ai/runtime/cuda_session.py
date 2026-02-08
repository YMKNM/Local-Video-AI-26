"""
CUDA Session Manager - NVIDIA GPU Runtime for Video AI

This module handles:
- CUDA device management and memory tracking
- TensorRT optimization and inference
- Mixed precision (FP16/BF16/INT8) support
- Memory-efficient attention (xformers/flash-attention)
- Dynamic batching and streaming
"""

import os
import gc
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import yaml
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CUDADeviceInfo:
    """CUDA device information container"""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    is_available: bool
    driver_version: str
    cuda_version: str
    supports_fp16: bool
    supports_bf16: bool
    supports_tf32: bool
    supports_int8: bool


@dataclass
class TensorRTConfig:
    """TensorRT optimization configuration"""
    enabled: bool = True
    fp16_mode: bool = True
    int8_mode: bool = False
    bf16_mode: bool = False
    workspace_size_gb: float = 4.0
    max_batch_size: int = 4
    optimization_level: int = 3
    timing_cache_path: Optional[str] = None
    engine_cache_path: Optional[str] = None


@dataclass
class MemoryConfig:
    """Memory management configuration"""
    max_memory_fraction: float = 0.9
    enable_memory_pool: bool = True
    enable_gradient_checkpointing: bool = True
    enable_attention_slicing: bool = True
    attention_slice_size: Optional[int] = None
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_cpu_offload: bool = False
    enable_sequential_offload: bool = False


class CUDASession:
    """
    CUDA Session Manager for NVIDIA GPU inference.
    
    Provides optimized GPU execution with:
    - Automatic device selection and memory management
    - TensorRT acceleration for production inference
    - Mixed precision support (FP16/BF16/TF32/INT8)
    - Memory-efficient attention mechanisms
    - Dynamic resource allocation
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize CUDA session manager.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.hardware_config = self._load_hardware_config()
        
        # Session state
        self._initialized = False
        self._device_id = 0
        self._device_info: Optional[CUDADeviceInfo] = None
        self._tensorrt_config: Optional[TensorRTConfig] = None
        self._memory_config: Optional[MemoryConfig] = None
        
        # Loaded components
        self._torch = None
        self._cuda_available = False
        self._tensorrt_available = False
        self._xformers_available = False
        
        # Memory tracking
        self._memory_lock = threading.Lock()
        self._allocated_tensors: Dict[str, int] = {}
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        self._compiled_cache: Dict[str, Any] = {}
        
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware configuration from YAML"""
        config_path = self.config_dir / "hardware.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def initialize(self, device_id: int = 0) -> bool:
        """
        Initialize CUDA session.
        
        Args:
            device_id: CUDA device ID to use
            
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            # Import PyTorch
            import torch
            self._torch = torch
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self._cuda_available = False
                self._initialized = True
                return True
            
            self._cuda_available = True
            self._device_id = device_id
            
            # Set device
            torch.cuda.set_device(device_id)
            
            # Get device info
            self._device_info = self._get_device_info(device_id)
            logger.info(f"CUDA initialized: {self._device_info.name}")
            logger.info(f"  VRAM: {self._device_info.total_memory_gb:.1f}GB total, "
                       f"{self._device_info.free_memory_gb:.1f}GB free")
            logger.info(f"  Compute Capability: {self._device_info.compute_capability}")
            
            # Configure memory
            self._configure_memory()
            
            # Check TensorRT availability
            self._check_tensorrt()
            
            # Check xformers availability
            self._check_xformers()
            
            # Configure precision
            self._configure_precision()
            
            self._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import PyTorch: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return False
    
    def _get_device_info(self, device_id: int) -> CUDADeviceInfo:
        """Get CUDA device information"""
        torch = self._torch
        
        props = torch.cuda.get_device_properties(device_id)
        
        total_memory = props.total_memory / (1024**3)
        free_memory = torch.cuda.mem_get_info(device_id)[0] / (1024**3)
        used_memory = total_memory - free_memory
        
        compute_cap = (props.major, props.minor)
        
        return CUDADeviceInfo(
            device_id=device_id,
            name=props.name,
            compute_capability=compute_cap,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            used_memory_gb=used_memory,
            is_available=True,
            driver_version=torch.version.cuda or "Unknown",
            cuda_version=torch.version.cuda or "Unknown",
            supports_fp16=compute_cap >= (5, 3),
            supports_bf16=compute_cap >= (8, 0),
            supports_tf32=compute_cap >= (8, 0),
            supports_int8=compute_cap >= (6, 1)
        )
    
    def _configure_memory(self):
        """Configure CUDA memory management"""
        torch = self._torch
        
        # Memory configuration from hardware config
        mem_config = self.hardware_config.get('memory_optimization', {})
        
        self._memory_config = MemoryConfig(
            max_memory_fraction=0.9,
            enable_memory_pool=True,
            enable_gradient_checkpointing=mem_config.get('gradient_checkpointing', True),
            enable_attention_slicing=mem_config.get('attention_slicing', True),
            enable_vae_slicing=mem_config.get('vae_slicing', True),
            enable_vae_tiling=mem_config.get('vae_tiling', True),
            enable_cpu_offload=mem_config.get('sequential_cpu_offload', False),
            enable_sequential_offload=mem_config.get('model_cpu_offload', False)
        )
        
        # Set memory allocation strategy
        cuda_settings = self.hardware_config.get('cuda_settings', {})
        if cuda_settings.get('memory_pool') == 'cuda_malloc_async':
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self._memory_config.max_memory_fraction,
                    self._device_id
                )
                logger.info("Configured CUDA memory pool")
            except Exception as e:
                logger.warning(f"Failed to configure memory pool: {e}")
        
        # Enable TF32 for RTX 30 series
        if cuda_settings.get('enable_tf32', True) and self._device_info.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for matrix operations")
        
        # Enable cuDNN benchmark
        if cuda_settings.get('enable_cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled")
    
    def _check_tensorrt(self):
        """Check TensorRT availability"""
        trt_config = self.hardware_config.get('tensorrt_settings', {})
        
        if not trt_config.get('enabled', True):
            self._tensorrt_available = False
            return
        
        try:
            import tensorrt
            self._tensorrt_available = True
            logger.info(f"TensorRT available: v{tensorrt.__version__}")
            
            self._tensorrt_config = TensorRTConfig(
                enabled=True,
                fp16_mode=trt_config.get('precision', 'fp16') == 'fp16',
                int8_mode=trt_config.get('precision', 'fp16') == 'int8',
                workspace_size_gb=trt_config.get('workspace_size_gb', 4.0),
                max_batch_size=trt_config.get('max_batch_size', 4),
                optimization_level=trt_config.get('optimization_level', 3),
                timing_cache_path=trt_config.get('cache_dir', 'cache/tensorrt'),
                engine_cache_path=trt_config.get('cache_dir', 'cache/tensorrt')
            )
        except ImportError:
            self._tensorrt_available = False
            logger.info("TensorRT not available")
    
    def _check_xformers(self):
        """Check xformers availability"""
        mem_opt = self.hardware_config.get('memory_optimization', {})
        
        if not mem_opt.get('enable_xformers', True):
            self._xformers_available = False
            return
        
        try:
            import xformers
            import xformers.ops
            self._xformers_available = True
            logger.info(f"xformers available: v{xformers.__version__}")
        except ImportError:
            self._xformers_available = False
            logger.info("xformers not available, using standard attention")
    
    def _configure_precision(self):
        """Configure default precision settings"""
        torch = self._torch
        
        quant_config = self.hardware_config.get('quantization', {})
        default_precision = quant_config.get('default_precision', 'fp16')
        
        # Set default dtype
        if default_precision == 'fp16' and self._device_info.supports_fp16:
            self._default_dtype = torch.float16
        elif default_precision == 'bf16' and self._device_info.supports_bf16:
            self._default_dtype = torch.bfloat16
        else:
            self._default_dtype = torch.float32
        
        logger.info(f"Default precision: {default_precision}")
    
    @property
    def device(self):
        """Get current CUDA device"""
        if self._cuda_available:
            return self._torch.device(f'cuda:{self._device_id}')
        return self._torch.device('cpu')
    
    @property
    def dtype(self):
        """Get default data type"""
        return getattr(self, '_default_dtype', self._torch.float32)
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory information in GB
        """
        if not self._cuda_available:
            return {
                'total': 0,
                'free': 0,
                'used': 0,
                'allocated': 0,
                'cached': 0
            }
        
        torch = self._torch
        
        total = torch.cuda.get_device_properties(self._device_id).total_memory
        free, total_free = torch.cuda.mem_get_info(self._device_id)
        allocated = torch.cuda.memory_allocated(self._device_id)
        cached = torch.cuda.memory_reserved(self._device_id)
        
        return {
            'total': total / (1024**3),
            'free': free / (1024**3),
            'used': (total - free) / (1024**3),
            'allocated': allocated / (1024**3),
            'cached': cached / (1024**3)
        }
    
    def clear_cache(self):
        """Clear CUDA memory cache"""
        if self._cuda_available:
            self._torch.cuda.empty_cache()
            gc.collect()
            logger.info("CUDA cache cleared")
    
    @contextmanager
    def inference_mode(self):
        """Context manager for inference mode with memory optimization"""
        torch = self._torch
        
        if self._cuda_available:
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    yield
        else:
            with torch.inference_mode():
                yield
    
    def optimize_for_inference(self, model, use_tensorrt: bool = True) -> Any:
        """
        Optimize model for inference.
        
        Args:
            model: PyTorch model to optimize
            use_tensorrt: Whether to use TensorRT optimization
            
        Returns:
            Optimized model
        """
        torch = self._torch
        
        # Move to device
        model = model.to(self.device, dtype=self.dtype)
        model.eval()
        
        # Enable memory-efficient attention if available
        if self._xformers_available and hasattr(model, 'enable_xformers_memory_efficient_attention'):
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory-efficient attention")
        
        # Apply torch.compile for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            perf_config = self.hardware_config.get('performance', {}).get('inference', {})
            compile_mode = perf_config.get('compile_mode', 'reduce-overhead')
            
            try:
                model = torch.compile(model, mode=compile_mode)
                logger.info(f"Applied torch.compile with mode={compile_mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # TensorRT optimization
        if use_tensorrt and self._tensorrt_available:
            model = self._apply_tensorrt(model)
        
        return model
    
    def _apply_tensorrt(self, model) -> Any:
        """Apply TensorRT optimization to model"""
        try:
            import torch_tensorrt
            
            trt_config = self._tensorrt_config
            
            # Determine precision
            enabled_precisions = {self._torch.float32}
            if trt_config.fp16_mode:
                enabled_precisions.add(self._torch.float16)
            if trt_config.int8_mode:
                enabled_precisions.add(self._torch.int8)
            
            # Compile with TensorRT
            # Note: This requires example inputs and is model-specific
            logger.info("TensorRT optimization available (requires model-specific setup)")
            
            return model
            
        except ImportError:
            logger.warning("torch_tensorrt not available")
            return model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return model
    
    def create_onnx_session(
        self,
        model_path: str,
        use_tensorrt: bool = True
    ):
        """
        Create ONNX Runtime session with CUDA/TensorRT execution provider.
        
        Args:
            model_path: Path to ONNX model
            use_tensorrt: Whether to use TensorRT execution provider
            
        Returns:
            ONNX Runtime InferenceSession
        """
        try:
            import onnxruntime as ort
            
            # Configure providers
            providers = []
            
            if use_tensorrt and self._tensorrt_available:
                trt_options = {
                    'device_id': self._device_id,
                    'trt_fp16_enable': self._tensorrt_config.fp16_mode,
                    'trt_int8_enable': self._tensorrt_config.int8_mode,
                    'trt_max_workspace_size': int(self._tensorrt_config.workspace_size_gb * 1024**3),
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': self._tensorrt_config.engine_cache_path
                }
                providers.append(('TensorrtExecutionProvider', trt_options))
            
            if self._cuda_available:
                cuda_options = {
                    'device_id': self._device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': int(self._device_info.total_memory_gb * 0.8 * 1024**3),
                    'cudnn_conv_algo_search': 'EXHAUSTIVE'
                }
                providers.append(('CUDAExecutionProvider', cuda_options))
            
            providers.append('CPUExecutionProvider')
            
            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Created ONNX session with providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get CUDA capabilities and recommendations.
        
        Returns:
            Dictionary with capabilities and recommendations
        """
        if not self._initialized:
            self.initialize()
        
        if not self._cuda_available:
            return {
                'cuda_available': False,
                'backend': 'cpu',
                'max_resolution': (640, 360),
                'max_frames': 60,
                'max_fps': 24,
                'recommended_model': 'zeroscope-v2'
            }
        
        mem_info = self.get_memory_info()
        free_vram = mem_info['free']
        
        # Determine capabilities based on available VRAM
        res_scaling = self.hardware_config.get('resolution_scaling', {})
        
        if free_vram >= 9:
            tier = res_scaling.get('ultra_vram', {})
            model = 'ltx-video-2'
        elif free_vram >= 7:
            tier = res_scaling.get('high_vram', {})
            model = 'ltx-video-2'
        elif free_vram >= 5:
            tier = res_scaling.get('medium_vram', {})
            model = 'ltx-video-2b'
        elif free_vram >= 3:
            tier = res_scaling.get('low_vram', {})
            model = 'accvideo'
        else:
            tier = res_scaling.get('critical_vram', {})
            model = 'zeroscope-v2'
        
        return {
            'cuda_available': True,
            'tensorrt_available': self._tensorrt_available,
            'xformers_available': self._xformers_available,
            'device_name': self._device_info.name,
            'compute_capability': self._device_info.compute_capability,
            'total_vram_gb': self._device_info.total_memory_gb,
            'free_vram_gb': free_vram,
            'backend': 'cuda' if not self._tensorrt_available else 'tensorrt',
            'max_resolution': (tier.get('max_width', 854), tier.get('max_height', 480)),
            'max_frames': tier.get('max_frames', 144),
            'max_fps': tier.get('max_fps', 24),
            'chunk_size': tier.get('chunk_size', 16),
            'recommended_model': model,
            'supports_fp16': self._device_info.supports_fp16,
            'supports_bf16': self._device_info.supports_bf16,
            'supports_tf32': self._device_info.supports_tf32
        }
    
    def __repr__(self) -> str:
        if not self._initialized:
            return "CUDASession(not initialized)"
        if not self._cuda_available:
            return "CUDASession(CPU mode)"
        return f"CUDASession(device={self._device_info.name}, VRAM={self._device_info.total_memory_gb:.1f}GB)"
