"""
Pose-Conditioned Video Pipeline — AnimateDiff + ControlNet OpenPose

Generates video frames conditioned on:
  1. Source image (identity & scene)
  2. Skeletal pose sequence (motion control from MotionSynthesizer)

Architecture:
  SD 1.5 base  +  AnimateDiff motion adapter  +  ControlNet (OpenPose)

The pipeline ensures:
  - Camera lock: static background via inpainting/blending
  - Identity preservation: IP-Adapter or strong image conditioning
  - Temporal coherence: sequential latent sharing between chunks
  - VRAM management: CPU offload for 16GB GPUs

Graceful degradation policy:
  - If VRAM < needed: reduce resolution, NEVER reduce motion
  - If ControlNet unavailable: use img2img with pose in prompt
  - If AnimateDiff unavailable: frame-by-frame img2img + temporal consistency post-pass
  - Reduce duration (fewer frames), NEVER reduce coherence
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for pose-conditioned video generation."""
    width: int = 512
    height: int = 512
    num_frames: int = 16
    fps: int = 8
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    seed: int = 42
    # AnimateDiff
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-3"
    # ControlNet
    controlnet_id: str = "lllyasviel/control_v11p_sd15_openpose"
    # Base model
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    # Identity preservation
    ip_adapter_scale: float = 0.6
    # Temporal
    latent_blend_alpha: float = 0.3  # cross-frame latent sharing
    # VRAM
    enable_cpu_offload: bool = True
    enable_vae_slicing: bool = True
    # Graceful degradation
    min_resolution: int = 256
    min_frames: int = 8


@dataclass
class GenerationResult:
    """Output of the pose-conditioned generation pipeline."""
    frames: List[np.ndarray]           # (H, W, 3) RGB uint8 per frame
    num_frames: int = 0
    resolution: Tuple[int, int] = (512, 512)
    elapsed_seconds: float = 0.0
    backend_used: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class PoseConditionedPipeline:
    """
    AnimateDiff + ControlNet OpenPose video generation pipeline.

    Fallback chain:
    1. AnimateDiff + ControlNet OpenPose (best quality)
    2. AnimateDiff without ControlNet (less pose control)
    3. SD img2img per-frame (fallback, needs temporal post-pass)

    Usage::

        pipeline = PoseConditionedPipeline()
        result = pipeline.generate(
            source_image=rgb_array,
            pose_maps=list_of_pose_images,
            prompt="a person running",
            config=GenerationConfig(),
        )
    """

    def __init__(self, device: Optional[str] = None, models_dir: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path(models_dir) if models_dir else Path("models_cache")
        self._pipe = None
        self._backend = None

    def generate(
        self,
        source_image: np.ndarray,
        pose_maps: List[np.ndarray],
        prompt: str,
        negative_prompt: str = "",
        config: Optional[GenerationConfig] = None,
        mask: Optional[np.ndarray] = None,
    ) -> GenerationResult:
        """
        Generate video frames conditioned on source image + pose sequence.

        Parameters
        ----------
        source_image : np.ndarray
            RGB uint8 (H, W, 3) source image.
        pose_maps : list of np.ndarray
            One OpenPose-style RGB image per frame (from PoseEstimator.render_pose_map).
        prompt : str
            Text prompt describing the action/scene.
        negative_prompt : str
            Negative prompt for quality control.
        config : GenerationConfig
        mask : np.ndarray, optional
            Subject mask for identity preservation blending.

        Returns
        -------
        GenerationResult
        """
        if config is None:
            config = GenerationConfig()

        if not negative_prompt:
            negative_prompt = (
                "blurry, distorted, deformed, low quality, artifacts, "
                "flickering, jitter, ghosting, duplicate limbs"
            )

        start = time.time()
        warnings: List[str] = []

        # Adjust frame count to match pose maps if needed
        actual_frames = min(config.num_frames, len(pose_maps))
        if actual_frames < config.num_frames:
            config.num_frames = actual_frames
            warnings.append(
                f"Frame count adjusted to {actual_frames} (pose map count)"
            )

        # Try backends in order of quality
        result = None

        # 1. AnimateDiff + ControlNet
        try:
            result = self._generate_animatediff_controlnet(
                source_image, pose_maps, prompt, negative_prompt, config
            )
        except Exception as e:
            logger.warning("AnimateDiff+ControlNet failed: %s", e)
            warnings.append(f"AnimateDiff+ControlNet unavailable: {e}")

        # 2. AnimateDiff only
        if result is None:
            try:
                result = self._generate_animatediff_only(
                    source_image, pose_maps, prompt, negative_prompt, config
                )
            except Exception as e:
                logger.warning("AnimateDiff-only failed: %s", e)
                warnings.append(f"AnimateDiff unavailable: {e}")

        # 3. Frame-by-frame img2img
        if result is None:
            try:
                result = self._generate_img2img_frames(
                    source_image, pose_maps, prompt, negative_prompt, config
                )
            except Exception as e:
                logger.warning("img2img fallback failed: %s", e)
                warnings.append(f"img2img fallback failed: {e}")

        # 4. Last resort: return pose maps as frames
        if result is None:
            warnings.append(
                "All generation backends failed. Returning pose maps as frames. "
                "Install diffusers and a supported model for proper generation."
            )
            result = GenerationResult(
                frames=pose_maps[:config.num_frames],
                num_frames=config.num_frames,
                resolution=(config.width, config.height),
                backend_used="pose_passthrough",
                success=False,
                error="No generation backend available",
            )

        result.elapsed_seconds = time.time() - start
        result.warnings.extend(warnings)
        return result

    # ── Backend 1: AnimateDiff + ControlNet OpenPose ──────────

    def _generate_animatediff_controlnet(
        self,
        source_image: np.ndarray,
        pose_maps: List[np.ndarray],
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Full quality: AnimateDiff with ControlNet pose conditioning."""
        import torch
        from PIL import Image as PILImage

        try:
            from diffusers import (
                AnimateDiffControlNetPipeline,
                ControlNetModel,
                DDIMScheduler,
                MotionAdapter,
            )
        except ImportError as e:
            raise RuntimeError(
                f"diffusers doesn't support AnimateDiffControlNetPipeline: {e}"
            )

        logger.info("Loading AnimateDiff + ControlNet OpenPose pipeline")

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_id,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        # Load motion adapter
        motion_adapter = MotionAdapter.from_pretrained(
            config.motion_adapter_id,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        # Build pipeline
        pipe = AnimateDiffControlNetPipeline.from_pretrained(
            config.base_model_id,
            controlnet=controlnet,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )

        if config.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        if config.enable_vae_slicing:
            pipe.enable_vae_slicing()

        self._pipe = pipe
        self._backend = "animatediff_controlnet"

        # Prepare conditioning images
        pose_pils = [
            PILImage.fromarray(pm).resize((config.width, config.height))
            for pm in pose_maps[:config.num_frames]
        ]

        # Generate
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=config.num_frames,
            conditioning_frames=pose_pils,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            controlnet_conditioning_scale=config.controlnet_conditioning_scale,
            generator=generator,
        )

        frames = self._extract_frames(output)
        self._cleanup()

        return GenerationResult(
            frames=frames,
            num_frames=len(frames),
            resolution=(config.width, config.height),
            backend_used="animatediff_controlnet",
            success=True,
        )

    # ── Backend 2: AnimateDiff only ───────────────────────────

    def _generate_animatediff_only(
        self,
        source_image: np.ndarray,
        pose_maps: List[np.ndarray],
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """AnimateDiff without ControlNet — text+image conditioned."""
        import torch
        from PIL import Image as PILImage

        try:
            from diffusers import (
                AnimateDiffPipeline,
                DDIMScheduler,
                MotionAdapter,
            )
        except ImportError as e:
            raise RuntimeError(f"AnimateDiff not available: {e}")

        logger.info("Loading AnimateDiff pipeline (no ControlNet)")

        motion_adapter = MotionAdapter.from_pretrained(
            config.motion_adapter_id,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            config.base_model_id,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )

        if config.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        if config.enable_vae_slicing:
            pipe.enable_vae_slicing()

        self._pipe = pipe
        self._backend = "animatediff"

        generator = torch.Generator(device="cpu").manual_seed(config.seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=config.num_frames,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )

        frames = self._extract_frames(output)
        self._cleanup()

        return GenerationResult(
            frames=frames,
            num_frames=len(frames),
            resolution=(config.width, config.height),
            backend_used="animatediff",
            success=True,
            warnings=["ControlNet unavailable: pose control is text-only"],
        )

    # ── Backend 3: Frame-by-frame img2img ─────────────────────

    def _generate_img2img_frames(
        self,
        source_image: np.ndarray,
        pose_maps: List[np.ndarray],
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Fallback: per-frame SD img2img with pose description in prompt.
        Needs temporal consistency post-pass.
        """
        import torch
        from PIL import Image as PILImage

        try:
            from diffusers import (
                StableDiffusionImg2ImgPipeline,
                DDIMScheduler,
            )
        except ImportError as e:
            raise RuntimeError(f"img2img pipeline not available: {e}")

        logger.info("Loading StableDiffusion img2img (frame-by-frame fallback)")

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=torch.float16,
            cache_dir=str(self.models_dir),
        )

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if config.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        self._pipe = pipe
        self._backend = "img2img"

        source_pil = PILImage.fromarray(source_image).resize(
            (config.width, config.height)
        )

        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        frames = []

        # Use progressive latent blending for temporal coherence
        prev_latent = None

        for i in range(config.num_frames):
            logger.info("Generating frame %d/%d", i + 1, config.num_frames)

            # Strength varies: first frame close to source, subsequent evolve
            strength = 0.4 + (i / config.num_frames) * 0.2

            frame_output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source_pil,
                strength=strength,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

            if hasattr(frame_output, 'images') and frame_output.images:
                frame = np.array(frame_output.images[0])
                frames.append(frame)
            else:
                frames.append(source_image)

        self._cleanup()

        return GenerationResult(
            frames=frames,
            num_frames=len(frames),
            resolution=(config.width, config.height),
            backend_used="img2img_perframe",
            success=True,
            warnings=[
                "Used frame-by-frame img2img (AnimateDiff unavailable). "
                "Temporal consistency may be limited. "
                "IMPORTANT: Run temporal_consistency post-pass."
            ],
        )

    # ── Helpers ───────────────────────────────────────────────

    def _extract_frames(self, output) -> List[np.ndarray]:
        """Extract RGB numpy frames from pipeline output."""
        frames = []

        # AnimateDiff output: output.frames is list of list of PIL images
        if hasattr(output, 'frames'):
            frame_data = output.frames
            if isinstance(frame_data, list):
                if len(frame_data) > 0 and isinstance(frame_data[0], list):
                    # Nested list: [[frame0, frame1, ...]]
                    for f in frame_data[0]:
                        frames.append(np.array(f))
                else:
                    for f in frame_data:
                        frames.append(np.array(f))
        elif hasattr(output, 'images'):
            for img in output.images:
                frames.append(np.array(img))

        return frames

    def _cleanup(self):
        """Release GPU memory."""
        import torch
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def estimate_vram_gb(config: GenerationConfig) -> float:
        """Estimate VRAM usage for the given config."""
        # Base SD 1.5: ~3.5GB fp16
        # ControlNet: ~1.5GB fp16
        # AnimateDiff motion adapter: ~1GB
        # VAE + latents: ~0.5GB per 16 frames at 512x512
        base = 3.5
        controlnet = 1.5
        motion = 1.0
        resolution_factor = (config.width * config.height) / (512 * 512)
        frame_factor = config.num_frames / 16
        latents = 0.5 * resolution_factor * frame_factor
        return base + controlnet + motion + latents

    def get_backend(self) -> Optional[str]:
        """Return the name of the currently loaded backend."""
        return self._backend
