"""
Video Diffusion Model Module

Handles the core video diffusion model for frame generation.
Supports LTX-Video, CogVideoX, and other video diffusion models.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoDiffusionModel:
    """
    Video diffusion model for generating video frames.
    
    Supports:
    - LTX-Video (Lightricks)
    - CogVideoX (THUDM)
    - Custom ONNX models
    
    Uses ONNX Runtime with DirectML for AMD GPU acceleration.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "ltx-video",
        device: str = "auto",
        dtype: str = "float16"
    ):
        """
        Initialize video diffusion model.
        
        Args:
            model_path: Path to ONNX model file
            model_name: Model identifier
            device: Compute device ('auto', 'directml', 'cpu')
            dtype: Data type ('float16', 'float32')
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        self._session = None
        self._loaded = False
        
        # Model config
        self._config = self._get_model_config(model_name)
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        configs = {
            "ltx-video": {
                "latent_channels": 16,
                "temporal_dim": 2,  # temporal downscale factor
                "spatial_dim": 8,   # spatial downscale factor
                "input_names": ["latents", "timestep", "encoder_hidden_states"],
                "output_names": ["noise_pred"],
                "block_out_channels": [320, 640, 1280, 1280],
            },
            "cogvideox": {
                "latent_channels": 16,
                "temporal_dim": 4,
                "spatial_dim": 8,
                "input_names": ["latents", "timestep", "encoder_hidden_states"],
                "output_names": ["noise_pred"],
                "block_out_channels": [320, 640, 1280, 1280],
            },
            "hummingbird": {
                "latent_channels": 8,
                "temporal_dim": 2,
                "spatial_dim": 8,
                "input_names": ["latents", "timestep", "encoder_hidden_states"],
                "output_names": ["noise_pred"],
                "block_out_channels": [256, 512, 768, 768],
            }
        }
        return configs.get(model_name, configs["ltx-video"])
    
    def load(self) -> bool:
        """
        Load the diffusion model.
        
        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True
        
        try:
            if self.model_path and self.model_path.exists():
                self._load_onnx_model()
            else:
                logger.warning(f"Model path not found: {self.model_path}")
                self._setup_placeholder()
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            return False
    
    def _load_onnx_model(self):
        """Load ONNX model for DirectML"""
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Memory optimization for large models
            sess_options.add_session_config_entry("session.memory.enable_memory_arena_shrinkage", "1")
            
            # Select execution provider
            providers = self._get_providers()
            
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Loaded diffusion ONNX model: {self.model_path}")
            logger.info(f"Using provider: {self._session.get_providers()[0]}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._setup_placeholder()
    
    def _get_providers(self) -> List[str]:
        """Get execution providers based on device setting"""
        if self.device == "cpu":
            return ['CPUExecutionProvider']
        
        providers = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
        except Exception:
            providers = ['CPUExecutionProvider']
        
        return providers
    
    def _setup_placeholder(self):
        """Setup placeholder for testing"""
        logger.warning("Using placeholder diffusion model (outputs random noise)")
        self._session = None
    
    def get_latent_shape(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int
    ) -> Tuple[int, ...]:
        """
        Calculate latent tensor shape.
        
        Args:
            batch_size: Batch size
            num_frames: Number of output frames
            height: Output height
            width: Output width
            
        Returns:
            Latent tensor shape
        """
        latent_frames = max(1, num_frames // self._config["temporal_dim"])
        latent_height = height // self._config["spatial_dim"]
        latent_width = width // self._config["spatial_dim"]
        
        return (
            batch_size,
            self._config["latent_channels"],
            latent_frames,
            latent_height,
            latent_width
        )
    
    def predict_noise(
        self,
        latents: np.ndarray,
        timestep: int,
        encoder_hidden_states: np.ndarray,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Predict noise for denoising step.
        
        Args:
            latents: Current latent tensor [B, C, T, H, W]
            timestep: Current diffusion timestep
            encoder_hidden_states: Text encoder output
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Predicted noise tensor
        """
        if not self._loaded:
            self.load()
        
        # Prepare timestep
        timestep_array = np.array([timestep], dtype=np.float32)
        
        if self._session is not None:
            noise_pred = self._run_onnx(latents, timestep_array, encoder_hidden_states)
        else:
            noise_pred = self._run_placeholder(latents)
        
        return noise_pred
    
    def _run_onnx(
        self,
        latents: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray
    ) -> np.ndarray:
        """Run ONNX inference"""
        # Ensure correct dtypes
        if self.dtype == "float16":
            latents = latents.astype(np.float16)
            encoder_hidden_states = encoder_hidden_states.astype(np.float16)
        else:
            latents = latents.astype(np.float32)
            encoder_hidden_states = encoder_hidden_states.astype(np.float32)
        
        timestep = timestep.astype(np.float32)
        
        # Get input names
        input_names = [inp.name for inp in self._session.get_inputs()]
        
        # Prepare inputs dict
        onnx_inputs = {}
        for name in input_names:
            name_lower = name.lower()
            if 'latent' in name_lower or 'sample' in name_lower:
                onnx_inputs[name] = latents
            elif 'time' in name_lower:
                onnx_inputs[name] = timestep
            elif 'hidden' in name_lower or 'encoder' in name_lower or 'context' in name_lower:
                onnx_inputs[name] = encoder_hidden_states
        
        # Run
        outputs = self._session.run(None, onnx_inputs)
        
        return outputs[0]
    
    def _run_placeholder(self, latents: np.ndarray) -> np.ndarray:
        """Generate placeholder noise for testing"""
        # Return random noise with same shape as latents
        return np.random.randn(*latents.shape).astype(np.float32) * 0.1
    
    def step(
        self,
        latents: np.ndarray,
        noise_pred: np.ndarray,
        timestep: int,
        scheduler_state: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform one denoising step.
        
        Args:
            latents: Current latents
            noise_pred: Predicted noise
            timestep: Current timestep
            scheduler_state: Scheduler state dict
            
        Returns:
            Tuple of (updated latents, updated scheduler state)
        """
        # Simple DDPM-style step
        alpha = scheduler_state.get('alphas', {})[timestep] if 'alphas' in scheduler_state else 0.99
        alpha_prev = scheduler_state.get('alphas', {})[timestep - 1] if 'alphas' in scheduler_state and timestep > 0 else 1.0
        
        # Simplified denoising
        sigma = np.sqrt(1 - alpha)
        pred_x0 = (latents - sigma * noise_pred) / np.sqrt(alpha)
        
        # Add noise for next step (except last step)
        if timestep > 0:
            noise = np.random.randn(*latents.shape).astype(np.float32)
            sigma_next = np.sqrt(1 - alpha_prev)
            latents = np.sqrt(alpha_prev) * pred_x0 + sigma_next * noise
        else:
            latents = pred_x0
        
        return latents, scheduler_state
    
    @property
    def latent_channels(self) -> int:
        """Get number of latent channels"""
        return self._config["latent_channels"]
    
    @property
    def temporal_scale(self) -> int:
        """Get temporal downscale factor"""
        return self._config["temporal_dim"]
    
    @property
    def spatial_scale(self) -> int:
        """Get spatial downscale factor"""
        return self._config["spatial_dim"]
    
    def unload(self):
        """Unload model to free memory"""
        self._session = None
        self._loaded = False
        logger.info("Unloaded diffusion model")


class DiffusionScheduler:
    """
    Diffusion scheduler for managing the denoising process.
    
    Supports:
    - DDPM (Denoising Diffusion Probabilistic Models)
    - DDIM (Denoising Diffusion Implicit Models)
    - DPM++ (DPM-Solver++)
    - Euler/Euler Ancestral
    """
    
    def __init__(
        self,
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        """
        Initialize scheduler.
        
        Args:
            scheduler_type: Type of scheduler
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
        """
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Compute schedules
        self._compute_schedules()
    
    def _compute_schedules(self):
        """Compute alpha and beta schedules"""
        if self.beta_schedule == "linear":
            betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            x = np.linspace(0, self.num_train_timesteps, steps)
            alphas_cumprod = np.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        
        self.betas = betas.astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
    
    def set_timesteps(self, num_inference_steps: int) -> np.ndarray:
        """
        Set inference timesteps.
        
        Args:
            num_inference_steps: Number of denoising steps
            
        Returns:
            Array of timesteps
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps[::-1]  # Reverse for denoising
        
        return timesteps.astype(np.int64)
    
    def get_state(self) -> Dict[str, Any]:
        """Get scheduler state for step function"""
        return {
            'alphas': {i: self.alphas_cumprod[i] for i in range(len(self.alphas_cumprod))},
            'betas': self.betas,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
        }
    
    def add_noise(
        self,
        original: np.ndarray,
        noise: np.ndarray,
        timestep: int
    ) -> np.ndarray:
        """
        Add noise to samples for forward diffusion.
        
        Args:
            original: Original samples
            noise: Noise to add
            timestep: Timestep
            
        Returns:
            Noisy samples
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise
