"""
Model Registry — centralised catalogue of supported video generation models.

Each entry stores everything needed to:
  • decide if the model will fit on the current GPU
  • download the weights from HuggingFace
  • instantiate the correct diffusers pipeline class
  • set sane default generation parameters

To add a new model, append an entry to MODEL_REGISTRY and, if it uses a new
pipeline class, add a loader in DiffusersPipeline.load().
"""

from __future__ import annotations

import logging
import torch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Compatibility(Enum):
    COMPATIBLE = "compatible"
    CONDITIONAL = "conditional"
    INCOMPATIBLE = "incompatible"


@dataclass
class ModelSpec:
    """Full specification for one video-generation model."""

    # ── identity ────────────────────────────────────────────
    id: str                             # e.g. "wan2.1-t2v-1.3b"
    display_name: str                   # pretty name for the UI
    family: str                         # "wan", "cogvideo", "ltx" …
    version: str                        # e.g. "2.1"
    parameters: str                     # "1.3B", "2B", etc.
    description: str                    # one-liner for dropdown tooltip

    # ── HuggingFace / disk ──────────────────────────────────
    repo_id: str                        # HF repo  (diffusers format)
    local_subdir: str                   # folder under  models/
    disk_gb: float                      # approximate download size

    # ── pipeline plumbing ───────────────────────────────────
    pipeline_cls: str                   # diffusers class name
    dtype: torch.dtype = torch.bfloat16
    scheduler_cls: str = "UniPCMultistepScheduler"
    scheduler_kwargs: Dict = field(default_factory=dict)
    needs_cpu_offload: bool = True      # True for VRAM ≤ 20 GB

    # ── generation defaults ─────────────────────────────────
    default_width: int = 832
    default_height: int = 480
    default_num_frames: int = 33
    max_num_frames: int = 81
    frame_rule: str = "4k+1"           # or "8k+1", "any"
    dim_multiple: int = 16             # width/height must be multiples
    default_steps: int = 30
    default_guidance: float = 5.0
    native_fps: int = 16

    # ── VRAM model ──────────────────────────────────────────
    vram_base_gb: float = 5.0          # weights on GPU during inference
    vram_per_pixel_frame: float = 1.4e-7  # W*H*F scaling constant

    # ── licensing / provenance ──────────────────────────────
    license: str = "Apache-2.0"
    source_url: str = ""
    modalities: List[str] = field(default_factory=lambda: ["text-to-video"])
    quality_tier: str = "standard"      # "entry", "standard", "high"
    notes: str = ""

    # ── helpers ─────────────────────────────────────────────

    def estimate_peak_vram(self, w: int, h: int, f: int) -> float:
        """Estimated peak VRAM (GB) for a given generation config."""
        return (self.vram_base_gb + w * h * f * self.vram_per_pixel_frame) * 1.25

    def snap_frames(self, n: int) -> int:
        """Snap frame count to the model's rule."""
        n = max(5, min(n, self.max_num_frames))
        if self.frame_rule == "4k+1":
            return ((n - 1) // 4) * 4 + 1
        elif self.frame_rule == "8k+1":
            return ((n - 1) // 8) * 8 + 1
        return n

    def snap_dims(self, w: int, h: int) -> Tuple[int, int]:
        m = self.dim_multiple
        return max(m, (w // m) * m), max(m, (h // m) * m)

    def check_compatibility(self, vram_gb: float, disk_free_gb: float) -> Compatibility:
        """Quick compatibility check against actual hardware."""
        if disk_free_gb < self.disk_gb:
            return Compatibility.INCOMPATIBLE
        # Estimate VRAM at default resolution
        peak = self.estimate_peak_vram(
            self.default_width, self.default_height, self.default_num_frames
        )
        if peak > vram_gb * 0.95:
            return Compatibility.INCOMPATIBLE
        if peak > vram_gb * 0.75:
            return Compatibility.CONDITIONAL
        return Compatibility.COMPATIBLE

    def ui_label(self) -> str:
        """Label shown in the model-selection dropdown."""
        return f"{self.display_name}  ({self.parameters}, {self.quality_tier})"


# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec):
    MODEL_REGISTRY[spec.id] = spec


# ── 1. Wan2.1-T2V-1.3B (already installed) ────────────────────────
_register(ModelSpec(
    id="wan2.1-t2v-1.3b",
    display_name="Wan2.1 T2V 1.3B",
    family="wan",
    version="2.1",
    parameters="1.3B",
    description="Compact but powerful; SOTA among sub-2B models. 480p, 5 sec, 16 fps.",
    repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    local_subdir="wan2.1-t2v-1.3b",
    disk_gb=27,
    pipeline_cls="WanPipeline",
    scheduler_cls="UniPCMultistepScheduler",
    scheduler_kwargs={"flow_shift": 3.0},
    default_width=832, default_height=480,
    default_num_frames=33, max_num_frames=81,
    frame_rule="4k+1", dim_multiple=16,
    default_steps=30, default_guidance=5.0,
    native_fps=16,
    vram_base_gb=5.0, vram_per_pixel_frame=1.4e-7,
    license="Apache-2.0",
    source_url="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B",
    modalities=["text-to-video"],
    quality_tier="standard",
    notes="Best quality-per-VRAM ratio. Requires 4k+1 frames, dims÷16.",
))

# ── 2. CogVideoX-2B ───────────────────────────────────────────────
_register(ModelSpec(
    id="cogvideox-2b",
    display_name="CogVideoX 2B",
    family="cogvideo",
    version="1.0",
    parameters="2B",
    description="6-sec videos at 720×480, 8 fps. Apache-2.0.",
    repo_id="THUDM/CogVideoX-2b",
    local_subdir="cogvideox-2b",
    disk_gb=11,
    pipeline_cls="CogVideoXPipeline",
    dtype=torch.float16,
    scheduler_cls="",  # uses built-in
    scheduler_kwargs={},
    default_width=720, default_height=480,
    default_num_frames=49, max_num_frames=49,
    frame_rule="any", dim_multiple=16,
    default_steps=20, default_guidance=6.0,
    native_fps=8,
    vram_base_gb=4.0, vram_per_pixel_frame=1.5e-7,
    license="Apache-2.0",
    source_url="https://huggingface.co/THUDM/CogVideoX-2b",
    modalities=["text-to-video"],
    quality_tier="entry",
    notes="Fixed 720×480 res, 6s clips. Very low VRAM with offload (~4 GB). Dims must be ÷16.",
))

# ── 3. CogVideoX-5B ───────────────────────────────────────────────
_register(ModelSpec(
    id="cogvideox-5b",
    display_name="CogVideoX 5B",
    family="cogvideo",
    version="1.0",
    parameters="5B",
    description="Higher quality, 6-sec videos at 720×480, 8 fps.",
    repo_id="THUDM/CogVideoX-5b",
    local_subdir="cogvideox-5b",
    disk_gb=20,
    pipeline_cls="CogVideoXPipeline",
    dtype=torch.bfloat16,
    scheduler_cls="",
    scheduler_kwargs={},
    default_width=720, default_height=480,
    default_num_frames=49, max_num_frames=49,
    frame_rule="any", dim_multiple=16,
    default_steps=20, default_guidance=6.0,
    native_fps=8,
    vram_base_gb=5.0, vram_per_pixel_frame=1.6e-7,
    license="CogVideoX (custom, research-OK)",
    source_url="https://huggingface.co/THUDM/CogVideoX-5b",
    modalities=["text-to-video"],
    quality_tier="standard",
    notes="Better quality than 2B; CogVideoX custom license.",
))

# ── 4. LTX-Video 2B (original, diffusers-native) ─────────────────
# NOTE: The HF repo (Lightricks/LTX-Video) is ~27 GB because of the full
# unquantized T5-XXL text encoder (~18 GB).  Needs ~28 GB system RAM for
# CPU-offload loading.  Use `allow_patterns` when downloading to avoid
# pulling the /media folder (demo GIFs).
_register(ModelSpec(
    id="ltx-video-2b",
    display_name="LTX-Video 2B",
    family="ltx",
    version="0.9",
    parameters="2B",
    description="Real-time DiT model, 768p 24 fps. Large download (~27 GB) due to T5-XXL encoder.",
    repo_id="Lightricks/LTX-Video",
    local_subdir="ltx-video-2b",
    disk_gb=27,
    pipeline_cls="LTXPipeline",
    dtype=torch.bfloat16,
    scheduler_cls="",
    scheduler_kwargs={},
    default_width=768, default_height=512,
    default_num_frames=97, max_num_frames=257,
    frame_rule="8k+1", dim_multiple=32,
    default_steps=50, default_guidance=7.5,
    native_fps=24,
    vram_base_gb=5.5, vram_per_pixel_frame=1.2e-7,
    license="LTX-Video Open Weights",
    source_url="https://huggingface.co/Lightricks/LTX-Video",
    modalities=["text-to-video", "image-to-video"],
    quality_tier="entry",
    notes="Diffusers-native. ~27 GB disk (T5-XXL is ~18 GB). Needs ~28 GB RAM. 24 fps output.",
))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_compatible_models(
    vram_gb: Optional[float] = None,
    disk_free_gb: Optional[float] = None,
) -> List[ModelSpec]:
    """Return models compatible with the current hardware, sorted best-first."""
    if vram_gb is None:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            vram_gb = 16.0  # assume mid-range

    if disk_free_gb is None:
        import shutil
        try:
            disk_free_gb = shutil.disk_usage(Path(__file__).anchor).free / (1024**3)
        except Exception:
            disk_free_gb = 100.0

    out = []
    for spec in MODEL_REGISTRY.values():
        compat = spec.check_compatibility(vram_gb, disk_free_gb)
        if compat != Compatibility.INCOMPATIBLE:
            out.append(spec)

    # Sort: compatible before conditional, then by quality tier
    tier_order = {"high": 0, "standard": 1, "entry": 2}
    out.sort(key=lambda s: (
        0 if s.check_compatibility(vram_gb, disk_free_gb) == Compatibility.COMPATIBLE else 1,
        tier_order.get(s.quality_tier, 9),
    ))
    return out


def get_model(model_id: str) -> Optional[ModelSpec]:
    return MODEL_REGISTRY.get(model_id)


def get_all_models() -> Dict[str, ModelSpec]:
    return dict(MODEL_REGISTRY)


def dropdown_choices(
    vram_gb: Optional[float] = None,
    disk_free_gb: Optional[float] = None,
) -> List[str]:
    """Return a list of (label) strings suitable for a Gradio Dropdown."""
    models = get_compatible_models(vram_gb, disk_free_gb)
    return [m.ui_label() for m in models]


def model_from_label(label: str) -> Optional[ModelSpec]:
    """Reverse-lookup a ModelSpec from its UI label."""
    for spec in MODEL_REGISTRY.values():
        if spec.ui_label() == label:
            return spec
    return None
