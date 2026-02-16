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


# ── quanto + group-offload compatibility patch ─────────────────
#
# Root cause (confirmed empirically):
#   diffusers' group offloading moves parameters between CPU and GPU using
#       param.data = param.data.to(device)
#   For quanto INT8 WeightQBytesTensor (a torch wrapper-subclass), the
#   C-level `.data` setter replaces the parameter's outer storage pointer
#   but does NOT transfer the internal `._data` (int8) and `._scale`
#   (bf16) components.  The parameter *appears* to be on CUDA but the
#   weight matrix read by mm is still on CPU → "mat2 is on cpu" error.
#
# Fix:
#   Use `module.to(device)` instead, which goes through `Module._apply()`
#   and — with the __future__ flag — creates brand-new Parameter objects
#   that properly wrap the moved quanto tensor.  Confirmed working across
#   repeated onload/offload cycles.
# ────────────────────────────────────────────────────────────────

_QUANTO_PATCH_APPLIED = False


def _patch_group_offload_for_quanto():
    """Monkey-patch diffusers ModuleGroup so device transfers work with quanto."""
    global _QUANTO_PATCH_APPLIED
    if _QUANTO_PATCH_APPLIED:
        return

    from diffusers.hooks.group_offloading import ModuleGroup

    _orig_onload = ModuleGroup._onload_from_memory
    _orig_offload = ModuleGroup._offload_to_memory

    def _quanto_safe_onload(self):
        """Onload via module.to() — keeps quanto INT8 wrappers intact."""
        old = torch.__future__.get_overwrite_module_params_on_conversion()
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        try:
            for m in self.modules:
                m.to(self.onload_device, non_blocking=False)
            # Standalone params/buffers (e.g. scale_shift_table) are plain
            # tensors — param.data assignment is fine for those.
            for p in self.parameters:
                p.data = p.data.to(self.onload_device, non_blocking=False)
            for b in self.buffers:
                b.data = b.data.to(self.onload_device, non_blocking=False)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(old)

    def _quanto_safe_offload(self):
        """Offload via module.to() with __future__ flag for quanto compat."""
        if self.stream is not None:
            return _orig_offload(self)          # stream path unchanged
        old = torch.__future__.get_overwrite_module_params_on_conversion()
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        try:
            for m in self.modules:
                m.to(self.offload_device, non_blocking=False)
            for p in self.parameters:
                p.data = p.data.to(self.offload_device, non_blocking=False)
            for b in self.buffers:
                b.data = b.data.to(self.offload_device, non_blocking=False)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(old)

    ModuleGroup._onload_from_memory = _quanto_safe_onload
    ModuleGroup._offload_to_memory = _quanto_safe_offload
    _QUANTO_PATCH_APPLIED = True
    logger.info("Patched group offloading for quanto INT8 compatibility")

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
        self._using_cpu_offload = False
        self._step_times: list = []
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
        try:
            PipeClass = getattr(diffusers, spec.pipeline_cls)
        except AttributeError:
            raise RuntimeError(
                f"Pipeline class '{spec.pipeline_cls}' not found in "
                f"diffusers {diffusers.__version__}. "
                f"{'Install diffusers from source: pip install git+https://github.com/huggingface/diffusers' if spec.min_diffusers_version == 'main' else f'Upgrade diffusers to >= {spec.min_diffusers_version}'}"
            ) from None

        # Check available VRAM to decide loading strategy
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Available VRAM: {vram_gb:.1f} GB")

        # For GPUs with <= 20GB VRAM, use CPU offload
        if vram_gb <= 20 or spec.needs_cpu_offload:
            logger.info("Using CPU offload strategy (VRAM <= 20GB)")
            self._using_cpu_offload = True
            self._report(1, 4, "Loading to CPU first...")

            # LTX-2 family: use NF4 quantisation when system RAM < 64 GB
            if spec.family == "ltx2":
                load_kwargs = self._ltx2_load_kwargs(source, spec)
                self._pipe = PipeClass.from_pretrained(source, **load_kwargs)
            else:
                self._pipe = PipeClass.from_pretrained(
                    source,
                    torch_dtype=spec.dtype,
                )

            self._report(2, 4, "Enabling CPU offload...")
            # LTX-2 with quanto INT8: use group offloading (block-level)
            # with our monkey-patch to fix the param.data assignment bug.
            #
            # WHY NOT enable_sequential_cpu_offload()?
            #   accelerate's AlignDevicesHook calls
            #     set_module_tensor_to_device(module, name, "meta")
            #   which internally does  param_cls(new_value, requires_grad=...)
            #   → WeightQBytesTensor.__new__() is missing 6 required args
            #   → crashes immediately (known diffusers bug #10526).
            #
            # WHY NOT enable_model_cpu_offload()?
            #   Moves whole components to GPU one at a time, but the INT8
            #   transformer (~18 GB) and text_encoder (~23 GB) each exceed
            #   16 GB VRAM.
            #
            # WHY does vanilla group offloading fail with quanto INT8?
            #   The onload path does  param.data = param.data.to(device)
            #   For quanto WeightQBytesTensor (a torch wrapper-subclass),
            #   the C-level .data setter replaces the outer storage pointer
            #   but leaves the internal ._data (int8) and ._scale (bf16)
            #   on the OLD device → "mat2 is on cpu" error.
            #   _patch_group_offload_for_quanto() fixes this by using
            #   module.to(device) which goes through Module._apply() to
            #   create proper new Parameters wrapping the moved tensors.
            #
            # BLOCK-LEVEL (num_blocks_per_group=1):
            #   Each LTX-2 transformer block is ~400 MB INT8 — fits
            #   easily in 16 GB VRAM.  Small components (VAE, connectors,
            #   audio_vae, vocoder ~5.2 GB total) are excluded from
            #   offloading and kept permanently on GPU.
            if spec.family == "ltx2":
                _patch_group_offload_for_quanto()
                self._pipe.enable_group_offload(
                    onload_device=torch.device(self.device),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                    use_stream=False,
                    exclude_modules=["vae", "audio_vae", "vocoder", "connectors"],
                )
                # VAE tiling: official LTX-2 docs recommend this to avoid
                # OOM during VAE decoding (121 frames × 512×768 is heavy).
                if hasattr(self._pipe, 'vae') and hasattr(self._pipe.vae, 'enable_tiling'):
                    self._pipe.vae.enable_tiling()
                    logger.info("Enabled VAE tiling for LTX-2")
                logger.info("Using group offload (block-level, quanto-patched) for LTX-2")
            else:
                self._pipe.enable_model_cpu_offload()
        else:
            self._using_cpu_offload = False
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

        if hasattr(self._pipe, 'enable_vae_slicing'):
            try:
                self._pipe.enable_vae_slicing()
            except Exception:
                pass
        if hasattr(self._pipe, 'enable_vae_tiling'):
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

        # Step callback for progress with ETA tracking
        total_steps = num_inference_steps
        self._step_times = []
        step_start = [time.time()]  # mutable ref for closure

        if self._using_cpu_offload:
            logger.warning(
                f"CPU offload active — each denoising step may take 1-3 minutes. "
                f"Total estimated time: {num_inference_steps * 2:.0f}-{num_inference_steps * 3:.0f} min. "
                f"Reduce steps or use a smaller model for faster results."
            )
            self._report(0, total_steps, f"Starting denoising (CPU offload, step 1 may take a few minutes)...")

        def step_callback(pipe, step, timestep, callback_kwargs):
            now = time.time()
            step_elapsed = now - step_start[0]
            self._step_times.append(step_elapsed)
            step_start[0] = now

            avg_step = sum(self._step_times) / len(self._step_times)
            remaining = (total_steps - step - 1) * avg_step
            if remaining >= 60:
                eta_str = f"~{remaining / 60:.0f} min left"
            else:
                eta_str = f"~{remaining:.0f}s left"

            msg = f"Step {step + 1}/{total_steps} ({step_elapsed:.0f}s/step, {eta_str})"
            logger.info(msg)
            self._report(step + 1, total_steps, msg)
            return callback_kwargs

        start = time.time()

        # Build call kwargs — LTX-2 needs frame_rate
        call_kwargs = dict(
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
        if spec.family == "ltx2":
            call_kwargs["frame_rate"] = float(spec.native_fps)

        # ── Diagnostic logging — prompt & conditioning ────
        logger.info(f"[PROMPT] Positive: {prompt[:200]}")
        logger.info(f"[PROMPT] Negative: {(negative_prompt or 'None')[:200]}")
        logger.info(
            f"[PARAMS] guidance={guidance_scale}, steps={num_inference_steps}, "
            f"frames={num_frames}, size={width}x{height}"
        )

        # ── Embedding validation hook (LTX-2 only) ───────
        # Monkey-patch encode_prompt to validate embeddings
        # after text encoding, catching NaN/inf/all-zeros that
        # would produce video unrelated to the prompt.
        _orig_encode = None
        if spec.family == "ltx2" and hasattr(self._pipe, 'encode_prompt'):
            _orig_encode = self._pipe.encode_prompt

            def _validated_encode(*args, **kwargs):
                result = _orig_encode(*args, **kwargs)
                prompt_embeds = result[0]
                if prompt_embeds is not None:
                    _emb = prompt_embeds.float()
                    has_nan = torch.isnan(_emb).any().item()
                    has_inf = torch.isinf(_emb).any().item()
                    all_zero = (_emb.abs().sum().item() == 0.0)
                    emb_mean = _emb.mean().item()
                    emb_std = _emb.std().item()
                    emb_min = _emb.min().item()
                    emb_max = _emb.max().item()
                    logger.info(
                        f"[EMBED] shape={list(prompt_embeds.shape)}, "
                        f"dtype={prompt_embeds.dtype}, "
                        f"mean={emb_mean:.4f}, std={emb_std:.4f}, "
                        f"min={emb_min:.4f}, max={emb_max:.4f}"
                    )
                    if has_nan:
                        logger.error(
                            "[EMBED] ⚠ NaN detected in prompt embeddings! "
                            "Text encoder output is corrupt. This will "
                            "cause the video to be unrelated to the prompt."
                        )
                    if has_inf:
                        logger.error(
                            "[EMBED] ⚠ Inf detected in prompt embeddings! "
                            "Text encoder output has overflow."
                        )
                    if all_zero:
                        logger.error(
                            "[EMBED] ⚠ All-zero prompt embeddings! "
                            "Text encoder produced no conditioning signal. "
                            "The video will be unconditional (random)."
                        )
                return result

            self._pipe.encode_prompt = _validated_encode

        # ── Connector output validation hook (LTX-2) ─────
        _connector_hook_handle = None
        if spec.family == "ltx2" and hasattr(self._pipe, 'connectors'):
            def _connector_post_hook(module, input, output):
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    vid_emb, aud_emb = output[0], output[1]
                    for label, emb in [("video", vid_emb), ("audio", aud_emb)]:
                        if emb is None:
                            continue
                        _e = emb.float()
                        logger.info(
                            f"[CONNECTOR-{label}] shape={list(emb.shape)}, "
                            f"dtype={emb.dtype}, "
                            f"mean={_e.mean().item():.4f}, "
                            f"std={_e.std().item():.4f}, "
                            f"NaN={torch.isnan(_e).any().item()}, "
                            f"Inf={torch.isinf(_e).any().item()}"
                        )
                        if torch.isnan(_e).any():
                            logger.error(
                                f"[CONNECTOR-{label}] ⚠ NaN in connector "
                                f"output! Conditioning is corrupt."
                            )
            _connector_hook_handle = self._pipe.connectors.register_forward_hook(
                _connector_post_hook
            )

        with torch.inference_mode():
            output = self._pipe(**call_kwargs)

        # Clean up hooks
        if _connector_hook_handle is not None:
            _connector_hook_handle.remove()
        if _orig_encode is not None:
            self._pipe.encode_prompt = _orig_encode

        elapsed = time.time() - start
        logger.info(f"Diffusion completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # ── extract frames from output ────────────────────────
        # LTX-2 returns LTX2PipelineOutput(frames=..., audio=...)
        # Other pipelines return output.frames[0]
        audio_tensor = None
        if hasattr(output, 'audio') and output.audio is not None:
            audio_tensor = output.audio
            logger.info("Audio output captured from LTX-2 pipeline")

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

            # Save audio if LTX-2 produced it
            if audio_tensor is not None:
                try:
                    audio_path = out / "audio.pt"
                    torch.save(audio_tensor.cpu(), audio_path)
                    logger.info(f"Saved audio tensor to {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not save audio: {e}")

        self._report(total_steps, total_steps, "Generation complete")
        return frame_paths

    # ── LTX-2 quantisation helpers ─────────────────────────────

    def _ltx2_load_kwargs(self, source: str, spec: ModelSpec) -> dict:
        """Build from_pretrained kwargs for LTX-2 with INT8 quantisation.

        Uses **quanto INT8** for the transformer and text_encoder:
          - 2× better precision than NF4 → much better prompt adherence
            and temporal coherence.
          - Compatible with group offloading (plain int8 tensors).
          - Total model ~46 GB INT8: fits in 64 GB RAM without swap.

        CRITICAL NOTES:
        - Only quantise transformer + text_encoder.  The VAE, connectors,
          audio_vae and vocoder MUST stay in BF16 — quantising the VAE
          destroys decoded pixel quality.
        - The transformer is a diffusers model → needs diffusers QuantoConfig
          (parameter: ``weights_dtype``).
        - The text_encoder is a transformers model (Gemma3) → needs
          **transformers** QuantoConfig (parameter: ``weights``).
          Using the wrong class causes silent failure / garbage output.
        """
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        kwargs: dict = {"torch_dtype": spec.dtype}

        if ram_gb < 96:
            try:
                from diffusers import (
                    QuantoConfig as DiffusersQuantoConfig,
                    PipelineQuantizationConfig,
                )
                from transformers import QuantoConfig as TransformersQuantoConfig

                # PipelineQuantizationConfig.quant_mapping with per-component
                # configs.  Must use the CORRECT QuantoConfig class per
                # component: diffusers' for diffusers models, transformers'
                # for transformers models.  They have different __init__
                # signatures (weights_dtype vs weights) and are NOT
                # interchangeable.
                quant_cfg = PipelineQuantizationConfig(
                    quant_mapping={
                        "transformer": DiffusersQuantoConfig(
                            weights_dtype="int8",
                        ),
                        "text_encoder": TransformersQuantoConfig(
                            weights="int8",
                        ),
                    }
                )
                kwargs["quantization_config"] = quant_cfg
                swap_note = "swap will be used" if ram_gb < 48 else "fits in RAM"
                logger.info(
                    f"LTX-2: system RAM {ram_gb:.0f} GB — "
                    f"quanto INT8 for transformer + text_encoder "
                    f"(~46 GB total, {swap_note})"
                )
            except ImportError:
                logger.warning(
                    "PipelineQuantizationConfig, QuantoConfig, or optimum-quanto "
                    "not available — cannot INT8-quantise LTX-2. "
                    "Install with: pip install optimum-quanto"
                )
            except Exception as e:
                logger.warning(f"INT8 quantisation setup failed: {e}")
        else:
            logger.info(f"LTX-2: {ram_gb:.0f} GB RAM available — loading full BF16 (no quantisation)")

        return kwargs

    # ── helpers ────────────────────────────────────────────────

    def _report(self, current: int, total: int, msg: str):
        if self.progress_callback:
            self.progress_callback(current, total, msg)
        logger.info(f"Progress: {current}/{total} — {msg}")
