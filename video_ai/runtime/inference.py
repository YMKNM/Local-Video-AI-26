"""
Inference Engine - Orchestrates model inference for video generation

This module handles:
- Text encoding (CLIP/T5)
- Video diffusion model inference
- VAE decoding
- Frame generation pipeline
- Multi-step diffusion process
"""

import os
import gc
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import yaml

from .onnx_loader import ONNXModelLoader
from .directml_session import DirectMLSession

logger = logging.getLogger(__name__)


class DiffusionScheduler:
    """
    Implements diffusion scheduling for iterative denoising.
    
    Supports DDIM, Euler, and Euler Ancestral schedulers.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear"
    ):
        """
        Initialize scheduler.
        
        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Compute betas
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
        # Compute other values needed for diffusion
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Timesteps for inference
        self.timesteps = None
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of denoising steps
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = np.flip(self.timesteps).copy()
    
    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0
    ) -> np.ndarray:
        """
        Perform a single denoising step (DDIM).
        
        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample
            eta: DDIM eta parameter (0 = deterministic)
            
        Returns:
            Denoised sample
        """
        # Get alpha values for current and previous timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        # Find previous timestep
        prev_timestep = max(timestep - self.num_train_timesteps // len(self.timesteps), 0)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        
        # Compute predicted original sample
        beta_prod_t = 1 - alpha_prod_t
        pred_original = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        
        # Clip predicted original sample (optional, helps stability)
        pred_original = np.clip(pred_original, -1.0, 1.0)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = eta * np.sqrt(variance)
        
        # Compute direction pointing to x_t
        pred_sample_direction = np.sqrt(1 - alpha_prod_t_prev - std_dev ** 2) * model_output
        
        # Compute previous sample
        prev_sample = np.sqrt(alpha_prod_t_prev) * pred_original + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0:
            noise = np.random.randn(*sample.shape).astype(np.float32)
            prev_sample = prev_sample + std_dev * noise
        
        return prev_sample
    
    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray
    ) -> np.ndarray:
        """
        Add noise to samples for a given timestep.
        
        Args:
            original_samples: Clean samples
            noise: Random noise
            timesteps: Timesteps to add noise for
            
        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples


class InferenceEngine:
    """
    Main inference engine for video generation.
    
    Coordinates:
    - Text encoding
    - Diffusion model inference
    - VAE decoding
    - Frame generation
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the inference engine.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        
        # Initialize components
        self.session_manager = DirectMLSession(config_dir)
        self.model_loader = ONNXModelLoader(config_dir)
        
        # Load configuration
        self.model_config = self._load_model_config()
        self.hardware_config = self._load_hardware_config()
        
        # Active sessions
        self._text_encoder_session = None
        self._diffusion_session = None
        self._vae_session = None
        
        # Scheduler
        self._scheduler = None
        
        # Placeholder mode (when ONNX models are not downloaded)
        self._placeholder_mode = False
        
        # Real diffusers pipeline (when model weights are available)
        self._diffusers_pipeline = None
        self._diffusers_mode = False
        
        # Generation state
        self._is_generating = False
        self._progress_callback: Optional[Callable] = None
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_path = self.config_dir / "models.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware configuration"""
        config_path = self.config_dir / "hardware.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        Set a callback for progress updates.
        
        Args:
            callback: Function(current_step, total_steps, message)
        """
        self._progress_callback = callback
    
    def _report_progress(self, current: int, total: int, message: str = ""):
        """Report generation progress"""
        if self._progress_callback:
            self._progress_callback(current, total, message)
        logger.info(f"Progress: {current}/{total} - {message}")
    
    def load_models(self, model_id: Optional[str] = None):
        """Load all required models into memory.
        
        Args:
            model_id: Specific model ID from model_registry to load.
                      If None, uses DEFAULT_MODEL (wan2.1-t2v-1.3b).
        
        Priority order:
        1. Real diffusers models (Wan2.1, CogVideoX, LTX-Video, etc.) — best quality
        2. ONNX models — fast inference
        3. Placeholder mode — synthetic frames for pipeline validation
        """
        # ── Try real diffusers models first ──
        try:
            from .diffusers_pipeline import DiffusersPipeline, DEFAULT_MODEL
            models_dir = Path(__file__).parent.parent.parent / "models"

            model_to_load = model_id if model_id else DEFAULT_MODEL
            dp = DiffusersPipeline(
                model_name=model_to_load,
                models_dir=models_dir,
                progress_callback=self._progress_callback,
            )
            if dp.is_model_downloaded():
                self._report_progress(0, 3, f"Loading {dp.spec.display_name}...")
                dp.load()
                self._diffusers_pipeline = dp
                self._diffusers_mode = True
                self._placeholder_mode = False
                # Set sentinel values so None-checks elsewhere pass
                self._text_encoder_session = "diffusers"
                self._diffusion_session = "diffusers"
                self._vae_session = "diffusers"
                self._report_progress(3, 3, f"{dp.spec.display_name} loaded")
                logger.info(f"Loaded diffusers model: {dp.spec.display_name} — AI video generation enabled")
                return
            else:
                # If a specific model was requested, fail loudly instead of
                # silently falling back to placeholder mode.
                if model_id:
                    raise RuntimeError(
                        f"Model '{dp.spec.display_name}' is not downloaded. "
                        f"Please download it first via the System tab or run: "
                        f"python download_models.py --model {model_to_load}"
                    )
                logger.info(
                    f"Diffusers model '{model_to_load}' not downloaded yet "
                    f"(checked {dp.model_local_path()}). "
                    f"Falling back to ONNX / placeholder."
                )
        except Exception as e:
            logger.warning(f"Could not initialise diffusers pipeline: {e}")
        
        # ── Fall back to ONNX models ──
        active_models = self.model_config.get('active_models', {})
        missing_models = []
        
        self._report_progress(0, 4, "Loading text encoder...")
        
        # Load text encoder
        text_encoder_name = active_models.get('text_encoder', 'clip-vit-large')
        self._text_encoder_session = self.model_loader.load_model(
            'text_encoder', text_encoder_name, self.session_manager
        )
        if self._text_encoder_session is None:
            missing_models.append(f"text_encoder/{text_encoder_name}")
        
        self._report_progress(1, 4, "Loading diffusion model...")
        
        # Load diffusion model
        diffusion_name = active_models.get('video_diffusion', 'ltx-video-2b')
        self._diffusion_session = self.model_loader.load_model(
            'video_diffusion', diffusion_name, self.session_manager
        )
        if self._diffusion_session is None:
            missing_models.append(f"video_diffusion/{diffusion_name}")
        
        self._report_progress(2, 4, "Loading VAE...")
        
        # Load VAE
        vae_name = active_models.get('vae', 'ltx-vae')
        self._vae_session = self.model_loader.load_model(
            'vae', vae_name, self.session_manager
        )
        if self._vae_session is None:
            missing_models.append(f"vae/{vae_name}")
        
        self._report_progress(3, 4, "Initializing scheduler...")
        
        # Initialize scheduler
        scheduler_name = active_models.get('scheduler', 'ddim')
        scheduler_config = self.model_config.get('schedulers', {}).get(scheduler_name, {})
        
        self._scheduler = DiffusionScheduler(
            num_train_timesteps=scheduler_config.get('num_train_timesteps', 1000),
            beta_start=scheduler_config.get('beta_start', 0.00085),
            beta_end=scheduler_config.get('beta_end', 0.012),
            beta_schedule=scheduler_config.get('beta_schedule', 'scaled_linear')
        )
        
        # If any models are missing, switch to placeholder mode
        if missing_models:
            self._placeholder_mode = True
            logger.warning(
                f"ONNX models not found: {missing_models}. "
                f"Switching to placeholder mode (synthetic frames). "
                f"Run 'python download_models.py' to download real models."
            )
            # Set sentinel values so None-checks pass
            if self._text_encoder_session is None:
                self._text_encoder_session = "placeholder"
            if self._diffusion_session is None:
                self._diffusion_session = "placeholder"
            if self._vae_session is None:
                self._vae_session = "placeholder"
        else:
            self._placeholder_mode = False
        
        self._report_progress(4, 4, "Models loaded")
        logger.info(
            f"Model loading complete "
            f"({'placeholder mode - no ONNX models' if self._placeholder_mode else 'all models loaded'})"
        )
    
    def unload_models(self):
        """Unload all models to free memory"""
        # Unload diffusers pipeline
        if self._diffusers_pipeline is not None:
            self._diffusers_pipeline.unload()
            self._diffusers_pipeline = None
            self._diffusers_mode = False
        
        self._text_encoder_session = None
        self._diffusion_session = None
        self._vae_session = None
        self._scheduler = None
        self._placeholder_mode = False
        
        self.model_loader.unload_all_models()
        self.session_manager.close_all_sessions()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("All models unloaded")

    def set_model(self, model_id: str):
        """
        Switch to a different model, unloading the current one first.
        
        Args:
            model_id: Model ID from model_registry (e.g. 'cogvideox-2b')
        """
        current = self.get_current_model_id()
        if current == model_id and self._diffusers_mode:
            logger.info(f"Model '{model_id}' is already loaded")
            return
        
        logger.info(f"Switching model: {current} → {model_id}")
        self.unload_models()
        self.load_models(model_id=model_id)

    def get_current_model_id(self) -> Optional[str]:
        """Return the ID of the currently loaded model, or None."""
        if self._diffusers_pipeline is not None:
            return self._diffusers_pipeline.model_name
        return None
    
    def encode_text(self, prompt: str, max_length: int = 77) -> np.ndarray:
        """
        Encode text prompt to embeddings.
        
        Args:
            prompt: Text prompt
            max_length: Maximum token length
            
        Returns:
            Text embeddings array
        """
        if self._text_encoder_session is None:
            raise RuntimeError("Text encoder not loaded. Call load_models() first.")
        
        # Get embedding dimension from config
        text_encoder_config = self.model_config.get('text_encoders', {})
        active_encoder = self.model_config.get('active_models', {}).get('text_encoder', 'clip-vit-large')
        embedding_dim = text_encoder_config.get(active_encoder, {}).get('embedding_dim', 768)
        
        if self._placeholder_mode:
            # Generate deterministic placeholder embeddings from prompt text
            seed = sum(ord(c) for c in prompt) % (2**31)
            rng = np.random.RandomState(seed)
            embeddings = rng.randn(1, max_length, embedding_dim).astype(np.float32)
            logger.info(f"Placeholder text encoding for: '{prompt[:60]}...'")
            return embeddings
        
        # Real tokenization + encoding (when models are loaded)
        logger.warning("Using placeholder text encoding - implement real tokenizer")
        embeddings = np.random.randn(1, max_length, embedding_dim).astype(np.float32)
        return embeddings
    
    def _create_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Create initial random latents for diffusion.
        
        Args:
            batch_size: Batch size
            num_frames: Number of frames
            height: Frame height
            width: Frame width
            seed: Random seed
            
        Returns:
            Random latent tensor
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Latent dimensions (typically downscaled by VAE factor)
        latent_channels = 4  # Standard for SD-style models
        latent_height = height // 8
        latent_width = width // 8
        
        # Create random latents
        latents = np.random.randn(
            batch_size, num_frames, latent_channels, latent_height, latent_width
        ).astype(np.float32)
        
        return latents
    
    def _run_diffusion_step(
        self,
        latents: np.ndarray,
        text_embeddings: np.ndarray,
        timestep: int,
        guidance_scale: float = 7.5
    ) -> np.ndarray:
        """
        Run a single diffusion step with classifier-free guidance.
        
        Args:
            latents: Current latent state
            text_embeddings: Text condition embeddings
            timestep: Current timestep
            guidance_scale: CFG scale
            
        Returns:
            Updated latents
        """
        if self._diffusion_session is None:
            raise RuntimeError("Diffusion model not loaded")
        
        # Prepare inputs
        # Note: Actual input names depend on the specific model
        
        # For classifier-free guidance, we need unconditional + conditional
        if guidance_scale > 1.0:
            # Create unconditional embeddings (zeros or learned null embedding)
            uncond_embeddings = np.zeros_like(text_embeddings)
            
            # Concatenate for batch processing
            latent_input = np.concatenate([latents, latents], axis=0)
            embedding_input = np.concatenate([uncond_embeddings, text_embeddings], axis=0)
        else:
            latent_input = latents
            embedding_input = text_embeddings
        
        # Prepare timestep input
        timestep_input = np.array([timestep], dtype=np.int64)
        
        # This is a placeholder - actual inference depends on model architecture
        # For demonstration, we simulate the diffusion step
        logger.debug(f"Running diffusion step at t={timestep}")
        
        # Simulate model output (replace with actual session.run())
        noise_pred = np.random.randn(*latent_input.shape).astype(np.float32) * 0.1
        
        # Apply classifier-free guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_cond = np.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Apply scheduler step
        latents = self._scheduler.step(noise_pred, timestep, latents)
        
        return latents
    
    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode latents to pixel space using VAE.
        
        Args:
            latents: Latent tensor [B, F, C, H, W]
            
        Returns:
            Decoded frames [B, F, 3, H*8, W*8]
        """
        if self._vae_session is None:
            raise RuntimeError("VAE not loaded")
        
        # VAE scaling factor (typical for SD-style VAEs)
        scaling_factor = 0.18215
        latents = latents / scaling_factor
        
        # Decode each frame
        batch_size, num_frames, channels, height, width = latents.shape
        
        decoded_frames = []
        for f in range(num_frames):
            frame_latent = latents[:, f]
            
            # Placeholder for actual VAE decoding
            # In practice: frame = self._vae_session.run(None, {'latent': frame_latent})[0]
            
            # Simulate decoded frame
            decoded_frame = np.random.randn(
                batch_size, 3, height * 8, width * 8
            ).astype(np.float32)
            decoded_frame = np.clip(decoded_frame * 0.5 + 0.5, 0, 1)  # Scale to [0, 1]
            
            decoded_frames.append(decoded_frame)
        
        # Stack frames
        frames = np.stack(decoded_frames, axis=1)
        
        return frames
    
    def generate_frames(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 854,
        height: int = 480,
        num_frames: int = 144,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate video frames from a text prompt.
        
        Args:
            prompt: Positive text prompt
            negative_prompt: Negative text prompt
            width: Frame width
            height: Frame height
            num_frames: Number of frames to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            output_dir: Directory to save frames
            
        Returns:
            List of paths to generated frame images
        """
        self._is_generating = True
        start_time = time.time()
        
        try:
            logger.info(f"Starting generation: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
            
            # Ensure models are loaded
            if self._text_encoder_session is None:
                self.load_models()
            
            # Set seed
            if seed is not None:
                np.random.seed(seed)
            
            # ── Real AI generation via diffusers ──
            if self._diffusers_mode and self._diffusers_pipeline is not None:
                return self._diffusers_pipeline.generate_frames(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    output_dir=output_dir,
                )
            
            # ── Placeholder mode: generate synthetic frames directly ──
            if self._placeholder_mode:
                return self._generate_placeholder_frames(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=24,
                    seed=seed,
                    output_dir=output_dir,
                    num_inference_steps=num_inference_steps,
                )
            
            # ── Real model inference path ──
            # 1. Encode text
            self._report_progress(0, num_inference_steps + 2, "Encoding text...")
            text_embeddings = self.encode_text(prompt)
            
            # 2. Create initial latents
            self._report_progress(1, num_inference_steps + 2, "Creating latents...")
            latents = self._create_latents(1, num_frames, height, width, seed)
            
            # 3. Set up scheduler
            self._scheduler.set_timesteps(num_inference_steps)
            
            # 4. Diffusion loop
            for i, timestep in enumerate(self._scheduler.timesteps):
                self._report_progress(
                    i + 2, num_inference_steps + 2,
                    f"Denoising step {i+1}/{num_inference_steps}"
                )
                
                latents = self._run_diffusion_step(
                    latents,
                    text_embeddings,
                    int(timestep),
                    guidance_scale
                )
            
            # 5. Decode latents to frames
            self._report_progress(
                num_inference_steps + 1, num_inference_steps + 2,
                "Decoding frames..."
            )
            frames = self._decode_latents(latents)
            
            # 6. Save frames
            frame_paths = []
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Convert and save each frame
                for f in range(num_frames):
                    frame = frames[0, f]  # [3, H, W]
                    frame = np.transpose(frame, (1, 2, 0))  # [H, W, 3]
                    frame = (frame * 255).astype(np.uint8)
                    
                    frame_path = output_path / f"frame_{f:05d}.png"
                    
                    # Save using PIL or cv2
                    try:
                        from PIL import Image
                        Image.fromarray(frame).save(frame_path)
                    except ImportError:
                        import cv2
                        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    frame_paths.append(str(frame_path))
            
            self._report_progress(
                num_inference_steps + 2, num_inference_steps + 2,
                "Generation complete"
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.2f}s")
            
            return frame_paths
            
        finally:
            self._is_generating = False
    
    def _generate_placeholder_frames(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int = 24,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        num_inference_steps: int = 30,
    ) -> List[str]:
        """
        Generate synthetic placeholder frames when ONNX models are not available.

        Creates an animated gradient video with the prompt text overlaid,
        so the full pipeline (frame write → assemble → MP4) can be validated.
        """
        from PIL import Image, ImageDraw, ImageFont
        import math

        logger.info(
            f"Generating {num_frames} placeholder frames "
            f"({width}x{height}) for prompt: '{prompt[:80]}'"
        )

        if seed is not None:
            np.random.seed(seed)

        frame_paths: List[str] = []
        if not output_dir:
            return frame_paths

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_steps = num_frames
        for idx in range(num_frames):
            self._report_progress(idx, total_steps, f"Rendering placeholder frame {idx + 1}/{num_frames}")

            t = idx / max(num_frames - 1, 1)  # 0 → 1

            # Animated gradient background
            r = int(128 + 127 * math.sin(2 * math.pi * t))
            g = int(128 + 127 * math.sin(2 * math.pi * t + 2.094))
            b = int(128 + 127 * math.sin(2 * math.pi * t + 4.189))

            img = Image.new("RGB", (width, height), (r, g, b))
            draw = ImageDraw.Draw(img)

            # Try to use a readable font, fall back to default
            try:
                font_large = ImageFont.truetype("arial.ttf", max(24, height // 16))
                font_small = ImageFont.truetype("arial.ttf", max(16, height // 24))
            except (IOError, OSError):
                font_large = ImageFont.load_default()
                font_small = font_large

            # Header
            header = "VIDEO AI - Placeholder Mode"
            bbox = draw.textbbox((0, 0), header, font=font_large)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) // 2, height // 6), header, fill="white", font=font_large)

            # Prompt text (wrapped)
            prompt_display = prompt[:120] + ("..." if len(prompt) > 120 else "")
            bbox2 = draw.textbbox((0, 0), prompt_display, font=font_small)
            tw2 = bbox2[2] - bbox2[0]
            draw.text(
                ((width - tw2) // 2, height // 2 - 10),
                prompt_display,
                fill="white",
                font=font_small,
            )

            # Frame counter
            counter_text = f"Frame {idx + 1}/{num_frames}"
            bbox3 = draw.textbbox((0, 0), counter_text, font=font_small)
            tw3 = bbox3[2] - bbox3[0]
            draw.text(((width - tw3) // 2, height * 3 // 4), counter_text, fill="white", font=font_small)

            # Note about models
            note = "Download ONNX models for real AI generation"
            bbox4 = draw.textbbox((0, 0), note, font=font_small)
            tw4 = bbox4[2] - bbox4[0]
            draw.text(((width - tw4) // 2, height * 7 // 8), note, fill=(220, 220, 220), font=font_small)

            frame_file = output_path / f"frame_{idx:05d}.png"
            img.save(frame_file)
            frame_paths.append(str(frame_file))

        self._report_progress(total_steps, total_steps, "Placeholder generation complete")
        elapsed = time.time() - time.time()  # near-zero
        logger.info(f"Placeholder generation complete — {num_frames} frames written to {output_dir}")
        return frame_paths

    def is_generating(self) -> bool:
        """Check if generation is in progress"""
        return self._is_generating
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        active_models = self.model_config.get('active_models', {})
        
        info = {
            'active_models': active_models,
            'text_encoder_loaded': self._text_encoder_session is not None,
            'diffusion_loaded': self._diffusion_session is not None,
            'vae_loaded': self._vae_session is not None,
            'scheduler_ready': self._scheduler is not None,
            'placeholder_mode': self._placeholder_mode,
            'diffusers_mode': self._diffusers_mode,
            'diffusers_model': self._diffusers_pipeline.model_name if self._diffusers_pipeline else None,
        }
        
        return info


def main():
    """Test inference engine"""
    logging.basicConfig(level=logging.INFO)
    
    engine = InferenceEngine()
    
    print("\n=== Inference Engine ===")
    print(f"DirectML available: {engine.session_manager.is_directml_available}")
    
    # Show model info
    info = engine.get_model_info()
    print(f"\nActive models: {info['active_models']}")
    
    # Check if models are available
    all_available, missing = engine.model_loader.check_all_models_available()
    
    if not all_available:
        print(f"\nMissing models: {missing}")
        print("Please download the required models first.")
    else:
        print("\nAll models available!")
        
        # Optionally run a test generation
        # engine.load_models()
        # frames = engine.generate_frames("Test prompt", output_dir="test_output")


if __name__ == "__main__":
    main()
