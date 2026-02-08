"""
Diffusers-based Video Generation Pipeline

Uses HuggingFace diffusers to run real AI video generation models.
Supports all models listed in model_registry.py:
  - Wan2.1-T2V-1.3B  (default, fits 16GB VRAM with bf16)
  - CogVideoX-2B / 5B
  - LTX-Video 2B distilled

This module is used by InferenceEngine when real model weights are available,
replacing the placeholder frame generator.
"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import torch
from PIL import Image

from .model_registry import MODEL_REGISTRY, ModelSpec, get_model

logger = logging.getLogger(__name__)

# Keep legacy constant for backwards-compat
DEFAULT_MODEL = "wan2.1-t2v-1.3b"
# Re-export DIFFUSERS_MODELS as a thin wrapper so old callers work
DIFFUSERS_MODELS = {k: {"repo_id": v.repo_id, "local_subdir": v.local_subdir}
                    for k, v in MODEL_REGISTRY.items()}


class DiffusersPipeline:
    """
    Wraps a HuggingFace diffusers video-generation pipeline so the rest of
    the Video AI codebase can call it through a simple interface.

    Supports dynamic model switching via set_model().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        models_dir: Optional[Path] = None,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.model_name = model_name
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent.parent / "models"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback

        self._pipe = None
        self._spec: Optional[ModelSpec] = get_model(model_name)
        if self._spec is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

    # ── public helpers ──────────────────────────────────────────

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def model_local_path(self) -> Path:
        return self.models_dir / self._spec.local_subdir

    def is_model_downloaded(self) -> bool:
        """Check whether model weights exist on disk."""
        p = self.model_local_path()
        if not p.exists():
            return False
        has_model_index = (p / "model_index.json").exists()
        has_transformer = (p / "transformer").exists() or (p / "unet").exists()
        return has_model_index and has_transformer

    def set_model(self, model_id: str):
        """Switch to a different model (unloads current one first)."""
        if model_id == self.model_name and self._pipe is not None:
            return  # already loaded
        spec = get_model(model_id)
        if spec is None:
            raise ValueError(f"Unknown model: {model_id}")
        self.unload()
        self.model_name = model_id
        self._spec = spec

    # ── loading / unloading ────────────────────────────────────

    def load(self):
        """Load the pipeline into GPU memory."""
        if self._pipe is not None:
            logger.info("Pipeline already loaded")
            return

        spec = self._spec
        local_path = self.model_local_path()

        # Decide where to load from: local path first, then HF repo
        if self.is_model_downloaded():
            source = str(local_path)
            logger.info(f"Loading diffusers model from local: {source}")
        else:
            source = spec.repo_id
            logger.info(f"Model not cached locally — loading from HuggingFace: {source}")

        self._report(0, 4, f"Loading {spec.display_name}...")

        # Import the correct pipeline class
        import diffusers
        PipeClass = getattr(diffusers, spec.pipeline_cls)

        # Check available VRAM to decide loading strategy
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Available VRAM: {vram_gb:.1f} GB")

        # For GPUs with <= 20GB VRAM, use CPU offload
        if vram_gb <= 20 or spec.needs_cpu_offload:
            logger.info("Using CPU offload strategy (VRAM <= 20GB)")
            self._report(1, 4, "Loading to CPU first...")
            self._pipe = PipeClass.from_pretrained(
                source,
                torch_dtype=spec.dtype,
            )
            self._report(2, 4, "Enabling model CPU offload...")
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = PipeClass.from_pretrained(
                source,
                torch_dtype=spec.dtype,
            )
            self._report(2, 4, "Moving to GPU...")
            self._pipe.to(self.device)

        # Enable memory-efficient attention if available
        self._report(3, 4, "Optimizing...")

        # Configure scheduler if specified
        if spec.scheduler_cls:
            try:
                SchedClass = getattr(diffusers, spec.scheduler_cls)
                self._pipe.scheduler = SchedClass.from_config(
                    self._pipe.scheduler.config, **spec.scheduler_kwargs
                )
                logger.info(f"Scheduler: {spec.scheduler_cls} {spec.scheduler_kwargs}")
            except Exception as e:
                logger.warning(f"Could not set scheduler: {e} — using default")

        try:
            self._pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            self._pipe.enable_vae_tiling()
        except Exception:
            pass

        self._report(4, 4, "Pipeline ready")
        logger.info(f"Diffusers pipeline loaded: {spec.display_name} on {self.device}")

    def unload(self):
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Diffusers pipeline unloaded")

    # ── generation ─────────────────────────────────────────────

    def generate_frames(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        num_frames: int = 33,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Generate video frames from a text prompt using the real AI model.

        Returns a list of file paths to the saved PNG frames.
        """
        if self._pipe is None:
            self.load()

        spec = self._spec

        # Clamp params to model limits
        num_frames = min(num_frames, spec.max_num_frames)
        # Snap frames to model rule (4k+1 for Wan, 8k+1 for LTX, etc.)
        num_frames = spec.snap_frames(num_frames)

        # Ensure dimensions match model requirements
        width, height = spec.snap_dims(width, height)

        logger.info(
            f"Real AI generation: {width}x{height}, {num_frames} frames, "
            f"{num_inference_steps} steps, guidance={guidance_scale}"
        )

        # ── VRAM budget check ─────────────────────────────────
        # Refuse to start if the config will almost certainly OOM.
        # Formula calibrated against real RTX 5080 measurements.
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            estimated_peak = spec.estimate_peak_vram(width, height, num_frames)
            logger.info(
                f"VRAM budget: estimated peak {estimated_peak:.1f} GB "
                f"/ {vram_total:.1f} GB total"
            )
            if estimated_peak > vram_total * 0.95:
                safe_frames = int(((vram_total * 0.95 / 1.25 - spec.vram_base_gb)
                                   / (width * height * spec.vram_per_pixel_frame)))
                safe_frames = max(5, spec.snap_frames(safe_frames))
                raise RuntimeError(
                    f"OOM: estimated {estimated_peak:.1f} GB exceeds "
                    f"{vram_total:.1f} GB VRAM. "
                    f"Try ≤{safe_frames} frames at {width}×{height}, "
                    f"or reduce resolution to 832×480."
                )

        # Build generator for reproducibility
        # Use "cpu" device for generator — required when using CPU offload
        # (Wan2.1 official example uses torch.Generator("cpu"))
        generator = None
        if seed is not None and seed > 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # Step callback for progress
        total_steps = num_inference_steps
        def step_callback(pipe, step, timestep, callback_kwargs):
            self._report(
                step + 1, total_steps,
                f"Denoising step {step + 1}/{total_steps}"
            )
            return callback_kwargs

        start = time.time()

        with torch.inference_mode():
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback,
            )

        elapsed = time.time() - start
        logger.info(f"Diffusion completed in {elapsed:.1f}s")

        # ── extract frames from output ────────────────────────
        # output.frames is typically List[List[PIL.Image]]
        frames_list = output.frames[0]  # first (and only) batch item

        # ── save frames to disk ───────────────────────────────
        frame_paths: List[str] = []
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            for idx, frame in enumerate(frames_list):
                if not isinstance(frame, Image.Image):
                    # numpy array → PIL
                    if isinstance(frame, np.ndarray):
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        frame = Image.fromarray(frame)
                    else:
                        continue

                fp = out / f"frame_{idx:05d}.png"
                frame.save(fp)
                frame_paths.append(str(fp))

            logger.info(f"Saved {len(frame_paths)} frames to {output_dir}")

        self._report(total_steps, total_steps, "Generation complete")
        return frame_paths

    # ── helpers ────────────────────────────────────────────────

    def _report(self, current: int, total: int, msg: str):
        if self.progress_callback:
            self.progress_callback(current, total, msg)
        logger.info(f"Progress: {current}/{total} — {msg}")
