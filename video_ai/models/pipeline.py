"""
Diffusion Pipeline Module

Complete pipeline for text-to-video generation.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np

from .text_encoder import TextEncoder
from .video_diffusion import VideoDiffusionModel, DiffusionScheduler
from .vae import VideoVAE

logger = logging.getLogger(__name__)


class DiffusionPipeline:
    """
    Complete text-to-video diffusion pipeline.
    
    This pipeline orchestrates:
    1. Text encoding
    2. Latent sampling
    3. Iterative denoising
    4. VAE decoding
    
    Usage:
        pipeline = DiffusionPipeline(model_dir="/path/to/models")
        pipeline.load()
        
        frames = pipeline.generate(
            prompt="A beautiful sunset",
            num_frames=144,
            height=480,
            width=854
        )
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16"
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_dir: Directory containing model files
            device: Compute device ('auto', 'directml', 'cpu')
            dtype: Data type ('float16', 'float32')
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.dtype = dtype
        
        # Initialize components (lazy loading)
        self._text_encoder: Optional[TextEncoder] = None
        self._diffusion_model: Optional[VideoDiffusionModel] = None
        self._vae: Optional[VideoVAE] = None
        self._scheduler: Optional[DiffusionScheduler] = None
        
        self._loaded = False
        
        # Progress callback
        self._progress_callback: Optional[Callable[[float, str], None]] = None
    
    def on_progress(self, callback: Callable[[float, str], None]):
        """Set progress callback"""
        self._progress_callback = callback
        return self
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.debug(f"Progress {progress:.1%}: {message}")
    
    def load(self) -> bool:
        """
        Load all pipeline components.
        
        Returns:
            True if all components loaded successfully
        """
        if self._loaded:
            return True
        
        self._report_progress(0.0, "Loading models...")
        
        try:
            # Initialize text encoder
            self._report_progress(0.1, "Loading text encoder...")
            self._text_encoder = TextEncoder(
                model_path=self._get_model_path("text_encoder.onnx"),
                device=self.device
            )
            self._text_encoder.load()
            
            # Initialize diffusion model
            self._report_progress(0.3, "Loading diffusion model...")
            self._diffusion_model = VideoDiffusionModel(
                model_path=self._get_model_path("unet.onnx"),
                device=self.device,
                dtype=self.dtype
            )
            self._diffusion_model.load()
            
            # Initialize VAE
            self._report_progress(0.5, "Loading VAE...")
            self._vae = VideoVAE(
                decoder_path=self._get_model_path("vae_decoder.onnx"),
                encoder_path=self._get_model_path("vae_encoder.onnx"),
                device=self.device,
                dtype=self.dtype
            )
            self._vae.load()
            
            # Initialize scheduler
            self._report_progress(0.7, "Initializing scheduler...")
            self._scheduler = DiffusionScheduler(
                scheduler_type="ddpm",
                num_train_timesteps=1000
            )
            
            self._loaded = True
            self._report_progress(1.0, "Models loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def _get_model_path(self, filename: str) -> Optional[str]:
        """Get path to model file"""
        if self.model_dir is None:
            return None
        
        path = self.model_dir / filename
        if path.exists():
            return str(path)
        
        return None
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 144,
        height: int = 480,
        width: int = 854,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate video frames from text prompt.
        
        Args:
            prompt: Text description of the video
            negative_prompt: Negative prompt for CFG
            num_frames: Number of frames to generate
            height: Frame height
            width: Frame width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            Video frames array [T, H, W, C] in range [0, 255] uint8
        """
        if not self._loaded:
            self.load()
        
        start_time = time.time()
        
        # Step 1: Encode text
        self._report_progress(0.0, "Encoding text prompt...")
        text_embeddings = self._encode_prompt(prompt, negative_prompt, guidance_scale)
        
        # Step 2: Sample initial latents
        self._report_progress(0.05, "Sampling latents...")
        latents = self._sample_latents(num_frames, height, width, seed)
        
        # Step 3: Run denoising loop
        self._report_progress(0.1, "Starting denoising...")
        latents = self._denoise(
            latents=latents,
            text_embeddings=text_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Step 4: Decode latents to video
        self._report_progress(0.95, "Decoding video...")
        video = self._decode_latents(latents)
        
        # Step 5: Post-process
        frames = self._postprocess(video, num_frames)
        
        elapsed = time.time() - start_time
        self._report_progress(1.0, f"Generation complete in {elapsed:.1f}s")
        
        return frames
    
    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float
    ) -> np.ndarray:
        """Encode prompts for conditioning"""
        # Encode positive prompt
        positive_result = self._text_encoder.encode(prompt)
        positive_embeddings = positive_result['embeddings']
        
        if guidance_scale > 1.0 and negative_prompt:
            # Encode negative prompt
            negative_result = self._text_encoder.encode(negative_prompt)
            negative_embeddings = negative_result['embeddings']
            
            # Concatenate for CFG
            text_embeddings = np.concatenate([negative_embeddings, positive_embeddings], axis=0)
        else:
            text_embeddings = positive_embeddings
        
        return text_embeddings
    
    def _sample_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> np.ndarray:
        """Sample initial random latents"""
        return self._vae.sample_latents(
            batch_size=1,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed
        )
    
    def _denoise(
        self,
        latents: np.ndarray,
        text_embeddings: np.ndarray,
        num_inference_steps: int,
        guidance_scale: float
    ) -> np.ndarray:
        """Run the denoising loop"""
        # Get timesteps
        timesteps = self._scheduler.set_timesteps(num_inference_steps)
        scheduler_state = self._scheduler.get_state()
        
        # Scale latents
        latents = latents * self._scheduler.sqrt_one_minus_alphas_cumprod[timesteps[0]]
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            progress = 0.1 + 0.85 * (i / len(timesteps))
            self._report_progress(progress, f"Denoising step {i+1}/{len(timesteps)}")
            
            # Expand latents for CFG if needed
            if guidance_scale > 1.0 and text_embeddings.shape[0] > 1:
                latent_model_input = np.concatenate([latents, latents], axis=0)
            else:
                latent_model_input = latents
            
            # Predict noise
            noise_pred = self._diffusion_model.predict_noise(
                latents=latent_model_input,
                timestep=int(t),
                encoder_hidden_states=text_embeddings,
                guidance_scale=guidance_scale
            )
            
            # Apply CFG
            if guidance_scale > 1.0 and noise_pred.shape[0] > 1:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents, scheduler_state = self._diffusion_model.step(
                latents=latents,
                noise_pred=noise_pred,
                timestep=int(t),
                scheduler_state=scheduler_state
            )
        
        return latents
    
    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents to video tensor"""
        return self._vae.decode(latents)
    
    def _postprocess(self, video: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Post-process video tensor to frames.
        
        Args:
            video: Video tensor [B, C, T, H, W] in range [0, 1]
            target_frames: Target number of frames
            
        Returns:
            Frames array [T, H, W, C] in range [0, 255] uint8
        """
        # Remove batch dimension and transpose
        # From [B, C, T, H, W] to [T, H, W, C]
        video = video[0]  # Remove batch
        video = np.transpose(video, (1, 2, 3, 0))  # C, T, H, W -> T, H, W, C
        
        # Adjust frame count if needed
        current_frames = video.shape[0]
        if current_frames != target_frames:
            # Simple frame interpolation
            indices = np.linspace(0, current_frames - 1, target_frames)
            frames = []
            for idx in indices:
                lower = int(idx)
                upper = min(lower + 1, current_frames - 1)
                alpha = idx - lower
                frame = (1 - alpha) * video[lower] + alpha * video[upper]
                frames.append(frame)
            video = np.stack(frames)
        
        # Convert to uint8
        frames = (video * 255).clip(0, 255).astype(np.uint8)
        
        return frames
    
    def unload(self):
        """Unload all models to free memory"""
        if self._text_encoder:
            self._text_encoder.unload()
        if self._diffusion_model:
            self._diffusion_model.unload()
        if self._vae:
            self._vae.unload()
        
        self._loaded = False
        logger.info("Pipeline unloaded")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get estimated memory usage"""
        return {
            'text_encoder': "~2 GB",
            'diffusion_model': "~8-12 GB",
            'vae': "~1-2 GB",
            'total': "~12-16 GB"
        }


class PipelineConfig:
    """Configuration for the diffusion pipeline"""
    
    def __init__(
        self,
        model_name: str = "ltx-video",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        enable_cfg: bool = True,
        scheduler: str = "ddpm"
    ):
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.enable_cfg = enable_cfg
        self.scheduler = scheduler
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
            'enable_cfg': self.enable_cfg,
            'scheduler': self.scheduler
        }
