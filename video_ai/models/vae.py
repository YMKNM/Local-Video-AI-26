"""
Video VAE Module

Variational Autoencoder for video latent space encoding/decoding.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoVAE:
    """
    Video VAE for encoding/decoding between pixel and latent space.
    
    The VAE compresses video frames into a lower-dimensional latent space
    for efficient diffusion, then decodes the denoised latents back to pixels.
    """
    
    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        model_name: str = "ltx-video-vae",
        device: str = "auto",
        dtype: str = "float16"
    ):
        """
        Initialize Video VAE.
        
        Args:
            encoder_path: Path to ONNX encoder model
            decoder_path: Path to ONNX decoder model
            model_name: Model identifier
            device: Compute device
            dtype: Data type
        """
        self.encoder_path = Path(encoder_path) if encoder_path else None
        self.decoder_path = Path(decoder_path) if decoder_path else None
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        self._encoder_session = None
        self._decoder_session = None
        self._loaded = False
        
        # VAE config
        self._config = self._get_vae_config(model_name)
    
    def _get_vae_config(self, model_name: str) -> Dict[str, Any]:
        """Get VAE-specific configuration"""
        configs = {
            "ltx-video-vae": {
                "latent_channels": 16,
                "temporal_scale": 2,
                "spatial_scale": 8,
                "scaling_factor": 0.18215,
                "shift_factor": 0.0,
            },
            "cogvideox-vae": {
                "latent_channels": 16,
                "temporal_scale": 4,
                "spatial_scale": 8,
                "scaling_factor": 0.18215,
                "shift_factor": 0.0,
            },
            "standard-vae": {
                "latent_channels": 4,
                "temporal_scale": 1,
                "spatial_scale": 8,
                "scaling_factor": 0.18215,
                "shift_factor": 0.0,
            }
        }
        return configs.get(model_name, configs["ltx-video-vae"])
    
    def load(self) -> bool:
        """
        Load VAE models.
        
        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True
        
        try:
            # Load decoder (always needed for generation)
            if self.decoder_path and self.decoder_path.exists():
                self._load_decoder()
            else:
                logger.warning("VAE decoder not found, using placeholder")
                self._setup_placeholder()
            
            # Encoder is optional (only needed for img2vid)
            if self.encoder_path and self.encoder_path.exists():
                self._load_encoder()
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            return False
    
    def _load_decoder(self):
        """Load ONNX decoder model"""
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = self._get_providers()
            
            self._decoder_session = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Loaded VAE decoder: {self.decoder_path}")
            
        except Exception as e:
            logger.error(f"Failed to load decoder: {e}")
            self._setup_placeholder()
    
    def _load_encoder(self):
        """Load ONNX encoder model"""
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = self._get_providers()
            
            self._encoder_session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Loaded VAE encoder: {self.encoder_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load encoder: {e}")
    
    def _get_providers(self) -> List[str]:
        """Get execution providers"""
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
        logger.warning("Using placeholder VAE (outputs test patterns)")
        self._decoder_session = None
    
    def encode(self, video: np.ndarray) -> np.ndarray:
        """
        Encode video frames to latent space.
        
        Args:
            video: Video tensor [B, C, T, H, W] in range [0, 1]
            
        Returns:
            Latent tensor [B, latent_C, T//ts, H//ss, W//ss]
        """
        if not self._loaded:
            self.load()
        
        if self._encoder_session is not None:
            return self._run_encoder(video)
        else:
            return self._encode_placeholder(video)
    
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode latents to video frames.
        
        Args:
            latents: Latent tensor [B, C, T, H, W]
            
        Returns:
            Video tensor [B, 3, T*ts, H*ss, W*ss] in range [0, 1]
        """
        if not self._loaded:
            self.load()
        
        # Unscale latents
        latents = self._unscale_latents(latents)
        
        if self._decoder_session is not None:
            return self._run_decoder(latents)
        else:
            return self._decode_placeholder(latents)
    
    def _scale_latents(self, latents: np.ndarray) -> np.ndarray:
        """Scale latents for diffusion"""
        return (latents - self._config["shift_factor"]) * self._config["scaling_factor"]
    
    def _unscale_latents(self, latents: np.ndarray) -> np.ndarray:
        """Unscale latents for decoding"""
        return latents / self._config["scaling_factor"] + self._config["shift_factor"]
    
    def _run_encoder(self, video: np.ndarray) -> np.ndarray:
        """Run ONNX encoder"""
        if self.dtype == "float16":
            video = video.astype(np.float16)
        else:
            video = video.astype(np.float32)
        
        # Normalize to [-1, 1] range
        video = video * 2.0 - 1.0
        
        input_name = self._encoder_session.get_inputs()[0].name
        outputs = self._encoder_session.run(None, {input_name: video})
        
        latents = outputs[0]
        return self._scale_latents(latents)
    
    def _run_decoder(self, latents: np.ndarray) -> np.ndarray:
        """Run ONNX decoder"""
        if self.dtype == "float16":
            latents = latents.astype(np.float16)
        else:
            latents = latents.astype(np.float32)
        
        input_name = self._decoder_session.get_inputs()[0].name
        outputs = self._decoder_session.run(None, {input_name: latents})
        
        video = outputs[0].astype(np.float32)
        
        # Convert from [-1, 1] to [0, 1]
        video = (video + 1.0) / 2.0
        video = np.clip(video, 0.0, 1.0)
        
        return video
    
    def _encode_placeholder(self, video: np.ndarray) -> np.ndarray:
        """Generate placeholder latents from video"""
        B, C, T, H, W = video.shape
        
        latent_t = max(1, T // self._config["temporal_scale"])
        latent_h = H // self._config["spatial_scale"]
        latent_w = W // self._config["spatial_scale"]
        latent_c = self._config["latent_channels"]
        
        # Generate deterministic latents based on video content
        seed = int(np.mean(video) * 1000) % 2**31
        rng = np.random.RandomState(seed)
        
        latents = rng.randn(B, latent_c, latent_t, latent_h, latent_w).astype(np.float32)
        
        return self._scale_latents(latents)
    
    def _decode_placeholder(self, latents: np.ndarray) -> np.ndarray:
        """Generate placeholder video from latents"""
        B, C, T, H, W = latents.shape
        
        out_t = T * self._config["temporal_scale"]
        out_h = H * self._config["spatial_scale"]
        out_w = W * self._config["spatial_scale"]
        
        # Generate a gradient test pattern based on latents
        video = np.zeros((B, 3, out_t, out_h, out_w), dtype=np.float32)
        
        for t in range(out_t):
            # Time-varying gradient
            time_factor = t / max(1, out_t - 1)
            
            # Create RGB gradient pattern
            for y in range(out_h):
                for x in range(out_w):
                    video[0, 0, t, y, x] = x / out_w  # Red gradient
                    video[0, 1, t, y, x] = y / out_h  # Green gradient
                    video[0, 2, t, y, x] = time_factor  # Blue time gradient
        
        # Add some variation based on latent values
        latent_influence = np.mean(latents, axis=(1, 2, 3, 4), keepdims=True)
        video = video * 0.8 + np.abs(latent_influence.reshape(B, 1, 1, 1, 1)) * 0.2
        
        video = np.clip(video, 0.0, 1.0)
        
        return video
    
    def sample_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample random latents for generation.
        
        Args:
            batch_size: Batch size
            num_frames: Number of video frames
            height: Output height
            width: Output width
            seed: Random seed
            
        Returns:
            Random latent tensor
        """
        latent_t = max(1, num_frames // self._config["temporal_scale"])
        latent_h = height // self._config["spatial_scale"]
        latent_w = width // self._config["spatial_scale"]
        latent_c = self._config["latent_channels"]
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        latents = rng.randn(batch_size, latent_c, latent_t, latent_h, latent_w)
        
        return latents.astype(np.float32)
    
    def get_latent_shape(
        self,
        num_frames: int,
        height: int,
        width: int
    ) -> Tuple[int, int, int, int]:
        """
        Get latent shape for given video dimensions.
        
        Args:
            num_frames: Number of video frames
            height: Video height
            width: Video width
            
        Returns:
            Tuple of (channels, temporal_frames, height, width)
        """
        return (
            self._config["latent_channels"],
            max(1, num_frames // self._config["temporal_scale"]),
            height // self._config["spatial_scale"],
            width // self._config["spatial_scale"]
        )
    
    @property
    def latent_channels(self) -> int:
        """Get number of latent channels"""
        return self._config["latent_channels"]
    
    @property
    def temporal_scale(self) -> int:
        """Get temporal downscale factor"""
        return self._config["temporal_scale"]
    
    @property
    def spatial_scale(self) -> int:
        """Get spatial downscale factor"""
        return self._config["spatial_scale"]
    
    def unload(self):
        """Unload models to free memory"""
        self._encoder_session = None
        self._decoder_session = None
        self._loaded = False
        logger.info("Unloaded VAE")
