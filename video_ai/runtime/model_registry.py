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

    # ── RAM / diffusers constraints ─────────────────────────
    min_ram_gb: float = 0.0            # min system RAM required (0 = no check)
    min_diffusers_version: str = ""    # e.g. "0.37.0" or "main" if unreleased

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

    def check_compatibility(
        self,
        vram_gb: float,
        disk_free_gb: float,
        ram_gb: float = 0.0,
    ) -> Compatibility:
        """Quick compatibility check against actual hardware."""
        if disk_free_gb < self.disk_gb:
            return Compatibility.INCOMPATIBLE
        # RAM gate — models that need more system RAM than available
        if self.min_ram_gb > 0 and ram_gb > 0 and ram_gb < self.min_ram_gb:
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

# ── 5. LTX-2 19B (audio-video DiT, requires diffusers main) ──────
# This is a 47B-total-parameter model (19B transformer + 27B Gemma3 text
# encoder).  The diffusers format weighs ~150 GB on disk.  BF16 loading
# requires ≥90 GB system RAM.  Even aggressive NF4 quantization (~26 GB)
# barely fits 32 GB RAM and produces degraded quality.
# Requires: diffusers ≥0.37.0 (or install from source / "main" branch)
_register(ModelSpec(
    id="ltx-2-19b",
    display_name="LTX-2 19B",
    family="ltx2",
    version="1.0",
    parameters="19B",
    description="DiT audio-video foundation model. 768p 24fps. Auto-NF4 on <64 GB RAM.",
    repo_id="Lightricks/LTX-2",
    local_subdir="ltx-2-19b",
    disk_gb=95,                # diffusers-format download (~38 GB transformer + ~50 GB text encoder + extras)
    pipeline_cls="LTX2Pipeline",
    dtype=torch.bfloat16,
    scheduler_cls="",          # uses built-in FlowMatchEulerDiscreteScheduler
    scheduler_kwargs={},
    needs_cpu_offload=True,
    default_width=768, default_height=512,
    default_num_frames=121, max_num_frames=257,
    frame_rule="8k+1", dim_multiple=32,
    default_steps=40, default_guidance=4.0,
    native_fps=24,
    vram_base_gb=5.0,          # sequential CPU offload keeps VRAM low (~3-5 GB)
    vram_per_pixel_frame=1.2e-7,
    min_ram_gb=28.0,           # NF4 quantised: ~25 GB total in RAM
    min_diffusers_version="main",
    license="ltx-2-community-license-agreement",
    source_url="https://huggingface.co/Lightricks/LTX-2",
    modalities=["text-to-video", "image-to-video", "text-to-audio"],
    quality_tier="high",
    notes=(
        "47B total params (19B transformer + 27B Gemma3 text encoder). "
        "~95 GB disk. Uses NF4 quantisation on ≤64 GB RAM systems. "
        "Requires diffusers from source (LTX2Pipeline). "
        "Output includes synchronised audio. 24 fps."
    ),
))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_compatible_models(
    vram_gb: Optional[float] = None,
    disk_free_gb: Optional[float] = None,
    ram_gb: Optional[float] = None,
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

    if ram_gb is None:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = 32.0  # conservative default

    out = []
    for spec in MODEL_REGISTRY.values():
        compat = spec.check_compatibility(vram_gb, disk_free_gb, ram_gb)
        if compat != Compatibility.INCOMPATIBLE:
            out.append(spec)

    # Sort: compatible before conditional, then by quality tier
    tier_order = {"high": 0, "standard": 1, "entry": 2}
    out.sort(key=lambda s: (
        0 if s.check_compatibility(vram_gb, disk_free_gb, ram_gb) == Compatibility.COMPATIBLE else 1,
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
    ram_gb: Optional[float] = None,
    include_all: bool = True,
) -> List[str]:
    """Return a list of (label) strings suitable for a Gradio Dropdown.

    When *include_all* is True (default), models that are INCOMPATIBLE with
    the detected hardware are still shown but prefixed with ⚠️.
    """
    if not include_all:
        models = get_compatible_models(vram_gb, disk_free_gb, ram_gb)
        return [m.ui_label() for m in models]

    # Auto-detect hardware once
    if vram_gb is None:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            vram_gb = 16.0
    if disk_free_gb is None:
        import shutil
        try:
            disk_free_gb = shutil.disk_usage(Path(__file__).anchor).free / (1024**3)
        except Exception:
            disk_free_gb = 100.0
    if ram_gb is None:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = 32.0

    labels: List[str] = []
    for spec in MODEL_REGISTRY.values():
        compat = spec.check_compatibility(vram_gb, disk_free_gb, ram_gb)
        prefix = "⚠️ " if compat == Compatibility.INCOMPATIBLE else ""
        labels.append(f"{prefix}{spec.ui_label()}")
    return labels


def model_from_label(label: str) -> Optional[ModelSpec]:
    """Reverse-lookup a ModelSpec from its UI label.

    Handles the optional ⚠️ prefix added by dropdown_choices().
    """
    clean = label.lstrip("⚠️ ").strip()
    for spec in MODEL_REGISTRY.values():
        if spec.ui_label() == clean or spec.ui_label() == label:
            return spec
    return None
