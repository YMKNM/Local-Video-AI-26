"""
Video AI — Professional Web UI

Modern Gradio-based interface for AI video generation.
Supports multiple models via model_registry:
  - Wan2.1-T2V-1.3B  (default)
  - CogVideoX-2B
  - CogVideoX-5B
  - LTX-Video 2B (distilled)

Features:
  - Dynamic model selection with per-model presets
  - Duration control mapped to real frame counts
  - Resolution presets with proper aspect ratios
  - Real-time GPU/VRAM monitoring
  - Live progress bar with denoising step tracking
  - Generation history
  - Professional white theme
"""

import gc
import os
import sys
import logging
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not installed. Install with: pip install gradio")

from .log_handler import UILogHandler, LogCapture

# Import Aggressive Generator tab
try:
    from .aggressive_generator_tab import create_aggressive_generator_tab
    AGGRESSIVE_TAB_AVAILABLE = True
except ImportError:
    AGGRESSIVE_TAB_AVAILABLE = False

# Import Image-to-Video Animation tab
try:
    from .image_motion_tab import create_image_motion_tab
    IMAGE_MOTION_TAB_AVAILABLE = True
except ImportError:
    IMAGE_MOTION_TAB_AVAILABLE = False

# Import DeepSeek (Offline LLM) tab
try:
    from .deepseek_tab import create_deepseek_tab
    DEEPSEEK_TAB_AVAILABLE = True
except ImportError:
    DEEPSEEK_TAB_AVAILABLE = False

# Import model registry
try:
    from ..runtime.model_registry import (
        MODEL_REGISTRY, get_model, get_compatible_models,
        dropdown_choices, model_from_label, ModelSpec,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    logger.warning("model_registry not available — using hardcoded Wan2.1 presets only")

# ─── Per-Model Presets ────────────────────────────────────────────────────────

def _build_model_presets() -> Dict[str, Dict]:
    """Build resolution / duration / quality presets from the registry."""
    presets: Dict[str, Dict] = {}

    if REGISTRY_AVAILABLE:
        for spec in MODEL_REGISTRY.values():
            dw, dh = spec.default_width, spec.default_height
            fps = spec.native_fps
            max_f = spec.max_num_frames

            # --- resolutions ---
            resolutions: Dict[str, Tuple[int, int]] = {}
            resolutions[f"{dh}p — {dw}×{dh} (default)"] = (dw, dh)

            # Add a "fast / small" option at ~60 % scale
            fw = spec.snap_dims(int(dw * 0.6), int(dh * 0.6))[0]
            fh = spec.snap_dims(int(dw * 0.6), int(dh * 0.6))[1]
            resolutions[f"{fh}p — {fw}×{fh} (fast)"] = (fw, fh)

            # Portrait
            pw, ph = spec.snap_dims(dh, dw)
            if pw != dw or ph != dh:
                resolutions[f"{ph}p — {pw}×{ph} (portrait)"] = (pw, ph)

            resolutions["Custom"] = (0, 0)

            # --- durations ---
            durations: Dict[str, int] = {}
            for sec in [1, 2, 3, 4, 5]:
                frames = spec.snap_frames(int(sec * fps))
                if frames > max_f:
                    break
                dur_label = f"{sec} sec — {frames} frames"
                durations[dur_label] = frames

            if not durations:
                durations[f"default — {spec.default_num_frames} frames"] = spec.default_num_frames

            presets[spec.id] = {
                "resolutions": resolutions,
                "durations": durations,
                "fps": fps,
                "max_frames": max_f,
                "default_steps": spec.default_steps,
                "default_guidance": spec.default_guidance,
                "display_name": spec.display_name,
            }

    # Fallback if registry not available
    if not presets:
        presets["wan2.1-t2v-1.3b"] = {
            "resolutions": {
                "480p — 832×480 (default)": (832, 480),
                "480p — 624×352 (fast)": (624, 352),
                "480p — 480×832 (portrait)": (480, 832),
                "720p — 1280×720 (⚠ may OOM)": (1280, 720),
                "Custom": (0, 0),
            },
            "durations": {
                "1 sec — 17 frames": 17,
                "2 sec — 33 frames": 33,
                "3 sec — 49 frames": 49,
                "4 sec — 65 frames": 65,
                "5 sec — 81 frames": 81,
            },
            "fps": 16,
            "max_frames": 81,
            "default_steps": 30,
            "default_guidance": 5.0,
            "display_name": "Wan2.1-T2V-1.3B",
        }

    return presets

MODEL_PRESETS = _build_model_presets()

# Quality / speed presets (model-independent step counts)
QUALITY_PRESETS = {
    "Draft (10 steps)": 10,
    "Balanced (20 steps)": 20,
    "Quality (30 steps)": 30,
    "Maximum (50 steps)": 50,
}

# ─── CSS ─────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Root Variables ───────────────────────────────────── */
:root {
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --primary-light: #eef2ff;
    --surface: #ffffff;
    --surface-1: #f8fafc;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --text: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --text-faint: #94a3b8;
    --success: #16a34a;
    --success-bg: #f0fdf4;
    --warning: #d97706;
    --warning-bg: #fffbeb;
    --danger: #dc2626;
    --danger-bg: #fef2f2;
    --accent-gradient: linear-gradient(135deg, #4f46e5, #7c3aed);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 30px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-xl: 0 20px 50px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.05);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-lg: 16px;
}

/* ── Global Typography ────────────────────────────────── */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

/* ── Header ───────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 32px 0 16px;
}
.app-header h1 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 0;
    line-height: 1.2;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 450;
    margin: 6px 0 0;
    letter-spacing: 0.2px;
}

/* ── Cards / Panels ───────────────────────────────────── */
.gr-panel, .gr-box, .gr-form {
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Status Bar ───────────────────────────────────────── */
.status-ready    { color: var(--success);  font-weight: 600; }
.status-running  { color: var(--warning);  font-weight: 600; }
.status-error    { color: var(--danger);   font-weight: 600; }

/* ── Compact settings rows ────────────────────────────── */
.settings-row { gap: 14px !important; }

/* ── Section headers in controls ──────────────────────── */
.controls-section h3 {
    font-size: 0.85rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin: 0 0 4px;
}

/* ── Log Console ──────────────────────────────────────── */
.log-console textarea {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 11.5px !important;
    line-height: 1.6 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06) !important;
}

/* ── Generate Button ──────────────────────────────────── */
.generate-btn {
    background: var(--accent-gradient) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 0 !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35), var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.generate-btn:hover {
    box-shadow: 0 6px 20px rgba(79,70,229,0.45), var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
.generate-btn:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
}

/* ── Progress Bar ─────────────────────────────────────── */
.progress-container {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    box-shadow: var(--shadow-sm);
}
.progress-bar-outer {
    width: 100%;
    height: 10px;
    background: var(--surface-2);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.progress-bar-inner {
    height: 100%;
    background: var(--accent-gradient);
    border-radius: 6px;
    transition: width 0.4s ease;
    box-shadow: 0 0 8px rgba(79,70,229,0.3);
}
.progress-step-text {
    font-size: 0.82rem;
    font-weight: 550;
    color: var(--text-secondary);
    margin-top: 8px;
    font-variant-numeric: tabular-nums;
}
.progress-pct {
    font-weight: 700;
    color: var(--primary);
}

/* ── Video Output Area ────────────────────────────────── */
.video-output-card {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-md) !important;
    overflow: hidden;
}

/* ── Result Status Card ───────────────────────────────── */
.result-status {
    border-radius: var(--radius) !important;
    padding: 4px !important;
}

/* ── Footer ───────────────────────────────────────────── */
.app-footer {
    text-align: center;
    color: var(--text-faint);
    font-size: 0.8rem;
    font-weight: 450;
    padding: 12px 0;
    letter-spacing: 0.15px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}

/* ── Tab styling ──────────────────────────────────────── */
.tabs > .tab-nav > button {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1px !important;
}

/* ── Input fields ─────────────────────────────────────── */
textarea, input[type="text"], input[type="number"] {
    font-size: 0.92rem !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* ── Labels ───────────────────────────────────────────── */
label span {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

/* ── Dropdown shadow ──────────────────────────────────── */
.gr-dropdown {
    box-shadow: var(--shadow-sm) !important;
}
"""


class WebUI:
    """Professional Web UI for Video AI generation."""

    def __init__(
        self,
        config_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        share: bool = False,
        port: int = 7860,
    ):
        self.config_dir = Path(config_dir) if config_dir else self._default_config_dir()
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.share = share
        self.port = port
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_handler = UILogHandler(max_entries=500)
        self._setup_logging()

        # State
        self._generator = None
        self._is_generating = False
        self._cancel_requested = False
        self._current_progress = 0.0
        self._current_status = "Ready"
        self._progress_step = 0
        self._progress_total_steps = 0
        self._progress_phase = "idle"   # idle | loading | denoising | encoding | done
        self._generation_history: List[Dict[str, Any]] = []

        # Load config
        self._load_config()

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _default_config_dir() -> Path:
        return Path(__file__).parent.parent / "configs"

    def _setup_logging(self):
        root = logging.getLogger()
        root.addHandler(self.log_handler)
        for ns in ("video_ai", "video_ai.agent", "video_ai.runtime", "video_ai.models"):
            logging.getLogger(ns).addHandler(self.log_handler)

    def _load_config(self):
        self.models_config: dict = {}
        self.hardware_config: dict = {}
        self.defaults_config: dict = {}
        try:
            import yaml
            for name, attr in [("models.yaml", "models_config"),
                               ("hardware.yaml", "hardware_config"),
                               ("defaults.yaml", "defaults_config")]:
                p = self.config_dir / name
                if p.exists():
                    with open(p) as f:
                        setattr(self, attr, yaml.safe_load(f) or {})
        except Exception as e:
            logger.warning(f"Config load: {e}")

    # ── GPU / System info ──────────────────────────────────────────

    @staticmethod
    def _gpu_info() -> Dict[str, Any]:
        """Return GPU name, VRAM total / used / free in GB."""
        import torch
        info = {"name": "No GPU", "total": 0, "used": 0, "free": 0, "available": False}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            used = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - used
            info.update(
                name=props.name,
                total=round(total, 1),
                used=round(used, 1),
                free=round(free, 1),
                available=True,
            )
        return info

    def _format_gpu_badge(self) -> str:
        g = self._gpu_info()
        if not g["available"]:
            return "🔴 No GPU detected"
        return f"🟢 **{g['name']}** — {g['total']} GB VRAM ({g['free']} GB free)"

    # ── Progress bar ───────────────────────────────────────────────

    @staticmethod
    def _render_progress_bar(step: int, total: int, message: str) -> str:
        """Render an HTML progress bar with step-level accuracy."""
        if total <= 0:
            pct = 0
        else:
            pct = min(int((step / total) * 100), 100)

        # Colour varies by phase
        if pct == 0:
            bar_colour = "#e2e8f0"       # idle — grey
            pct_colour = "#94a3b8"
        elif pct >= 100:
            bar_colour = "linear-gradient(135deg, #16a34a, #22c55e)"  # done — green
            pct_colour = "#16a34a"
        else:
            bar_colour = "linear-gradient(135deg, #4f46e5, #7c3aed)"  # active — indigo
            pct_colour = "#4f46e5"

        step_label = f"Step {step}/{total}" if total > 0 else ""
        msg_safe = (message or "").replace("<", "&lt;").replace(">", "&gt;")

        return f"""
        <div class="progress-container">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
            <span class="progress-step-text">{msg_safe}</span>
            <span class="progress-step-text" style="color:{pct_colour};font-weight:700;">{pct}%</span>
          </div>
          <div class="progress-bar-outer">
            <div class="progress-bar-inner" style="width:{pct}%;background:{bar_colour};"></div>
          </div>
          <div style="margin-top:4px;text-align:right;">
            <span class="progress-step-text" style="font-size:0.78rem;color:#94a3b8;">{step_label}</span>
          </div>
        </div>"""

    def _step_callback(self, current: int, total: int, message: str):
        """Called by the diffusers pipeline on each denoising step."""
        self._progress_step = current
        self._progress_total_steps = total
        self._current_status = message

    def _format_system_status(self) -> str:
        g = self._gpu_info()
        # RAM
        ram_total = ram_avail = 0
        try:
            import psutil
            m = psutil.virtual_memory()
            ram_total = round(m.total / (1024**3), 1)
            ram_avail = round(m.available / (1024**3), 1)
        except ImportError:
            pass

        # FFmpeg
        ffmpeg_ok = False
        try:
            import imageio_ffmpeg
            ffmpeg_ok = bool(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass

        # Model status — check all registry models
        model_lines = []
        models_dir = Path(__file__).parent.parent.parent / "models"
        if REGISTRY_AVAILABLE:
            for spec in MODEL_REGISTRY.values():
                p = models_dir / spec.local_subdir
                downloaded = p.exists() and (p / "model_index.json").exists()
                icon = "✅" if downloaded else "❌"
                model_lines.append(f"| **{spec.display_name}** | {icon} {'Downloaded' if downloaded else 'Not found'} |")
        else:
            model_path = models_dir / "wan2.1-t2v-1.3b"
            if model_path.exists() and (model_path / "model_index.json").exists():
                model_lines.append("| **Wan2.1-T2V-1.3B** | ✅ Downloaded |")
            else:
                model_lines.append("| **Wan2.1-T2V-1.3B** | ❌ Not found |")

        # PyTorch
        import torch
        pt_ver = torch.__version__
        cuda_ver = torch.version.cuda or "N/A"

        model_table = "\n".join(model_lines)

        return f"""## System Status

| Component | Status |
|-----------|--------|
| **GPU** | {g['name'] if g['available'] else '❌ Not detected'} |
| **VRAM** | {g['total']} GB total / {g['free']} GB free |
| **RAM** | {ram_total} GB total / {ram_avail} GB available |
| **PyTorch** | {pt_ver} (CUDA {cuda_ver}) |
| **FFmpeg** | {'✅ Available' if ffmpeg_ok else '❌ Not found'} |

## AI Models

| Model | Status |
|-------|--------|
{model_table}
"""

    # ── frame / duration logic ─────────────────────────────────────

    @staticmethod
    def _seconds_to_frames(seconds: float, fps: int = 16) -> int:
        """Convert duration to frame count (generic — model rules in registry)."""
        raw = int(seconds * fps)
        return max(5, raw)

    @staticmethod
    def _frames_to_seconds(frames: int, fps: int = 16) -> float:
        return round(frames / fps, 2)

    # ── model-change callbacks ────────────────────────────────────

    def _on_model_change(self, model_label: str):
        """When user picks a different model, update resolution / duration dropdowns."""
        model_id = None
        if REGISTRY_AVAILABLE:
            spec_obj = model_from_label(model_label)
            model_id = spec_obj.id if spec_obj else None
        if not model_id:
            model_id = list(MODEL_PRESETS.keys())[0]

        p = MODEL_PRESETS.get(model_id, list(MODEL_PRESETS.values())[0])

        res_choices = list(p["resolutions"].keys())
        dur_choices = list(p["durations"].keys())
        default_guidance = p.get("default_guidance", 5.0)

        # Return updates for: resolution dropdown, duration dropdown,
        # guidance slider, model-info markdown
        spec = get_model(model_id) if REGISTRY_AVAILABLE else None
        info_md = ""
        if spec:
            info_md = (
                f"**{spec.display_name}** — {spec.parameters} · "
                f"{spec.native_fps} fps · max {spec.max_num_frames} frames · "
                f"~{spec.estimate_peak_vram(spec.default_width, spec.default_height, spec.default_num_frames):.0f} GB VRAM"
            )

        return (
            gr.update(choices=res_choices, value=res_choices[0]),
            gr.update(choices=dur_choices, value=dur_choices[0]),
            gr.update(value=default_guidance),
            info_md,
        )

    # ── generation ────────────────────────────────────────────────

    def _generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        model_label: str,
        duration_preset: str,
        resolution_preset: str,
        custom_width: int,
        custom_height: int,
        quality_preset: str,
        guidance: float,
        seed: int,
        progress=None,
    ) -> Tuple[Optional[str], str, str]:
        """Core generation function."""
        if not prompt or not prompt.strip():
            return None, "⚠️ Please enter a prompt.", self._get_logs()
        if self._is_generating:
            return None, "⚠️ Generation already in progress.", self._get_logs()

        self._is_generating = True
        self._cancel_requested = False
        self._progress_step = 0
        self._progress_total_steps = 0
        self._current_status = "Initialising…"
        video_path = None
        status = ""

        def prog(val, desc=""):
            if progress is not None:
                progress(val, desc=desc)
            self._current_progress = val
            self._current_status = desc

        try:
            # ── Resolve model ID ─────────────────────────────
            model_id = None
            if REGISTRY_AVAILABLE:
                spec_obj = model_from_label(model_label)
                model_id = spec_obj.id if spec_obj else None
            if not model_id:
                model_id = list(MODEL_PRESETS.keys())[0]

            p = MODEL_PRESETS.get(model_id, list(MODEL_PRESETS.values())[0])

            # ── Resolve parameters ───────────────────────────
            num_frames = p["durations"].get(duration_preset, list(p["durations"].values())[0])

            if resolution_preset == "Custom":
                # Snap to model's dim_multiple via registry
                if REGISTRY_AVAILABLE:
                    spec = get_model(model_id)
                    if spec:
                        custom_width, custom_height = spec.snap_dims(custom_width, custom_height)
                    else:
                        custom_width = (custom_width // 16) * 16
                        custom_height = (custom_height // 16) * 16
                else:
                    custom_width = (custom_width // 16) * 16
                    custom_height = (custom_height // 16) * 16
                width, height = custom_width, custom_height
            else:
                width, height = p["resolutions"].get(
                    resolution_preset, list(p["resolutions"].values())[0]
                )

            steps = QUALITY_PRESETS.get(quality_preset, 30)
            actual_seed = seed if seed >= 0 else -1

            fps = p["fps"]
            duration_sec = round(num_frames / fps, 2)

            logger.info("=" * 60)
            logger.info("NEW GENERATION REQUEST")
            logger.info(f"  Model    : {model_id}")
            logger.info(f"  Prompt   : {prompt[:80]}...")
            logger.info(f"  Frames   : {num_frames} ({duration_sec}s @ {fps} fps)")
            logger.info(f"  Size     : {width}×{height}")
            logger.info(f"  Steps    : {steps}  |  Guidance: {guidance}  |  Seed: {actual_seed}")
            logger.info("=" * 60)

            prog(0.0, "Initialising…")

            # ── Create / reuse VideoAI ──────────────────────
            from video_ai import VideoAI

            if self._generator is None:
                self._generator = VideoAI(
                    config_dir=str(self.config_dir),
                    output_dir=str(self.output_dir),
                )

            if self._cancel_requested:
                return None, "⚠️ Cancelled.", self._get_logs()

            self._current_status = "Loading models (may take several minutes on first run)…"

            # ── Wire up step-level progress callback ────────
            try:
                engine = self._generator.planner._get_inference_engine()
                engine.set_progress_callback(self._step_callback)
                # Switch model if needed
                current_model = engine.get_current_model_id()
                if current_model != model_id:
                    engine.set_model(model_id)
            except Exception:
                pass  # non-fatal if callback wiring fails

            self._progress_total_steps = steps
            self._progress_step = 0
            self._current_status = f"Starting denoising (0/{steps}) - first step may take 1-3 min with CPU offload..."

            # ── Run generation ──────────────────────────────
            result = self._generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                duration_seconds=duration_sec,
                width=width,
                height=height,
                fps=fps,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=actual_seed if actual_seed >= 0 else None,
                model_name=model_id,
            )

            prog(0.95, "Encoding video…")
            self._current_status = "Encoding video…"
            self._progress_step = steps
            self._progress_total_steps = steps

            if result.success:
                video_path = result.output_path
                status = (
                    f"### ✅ Generation Complete\n\n"
                    f"- **Duration**: {duration_sec}s ({num_frames} frames)\n"
                    f"- **Resolution**: {width}×{height}\n"
                    f"- **Steps**: {steps}  |  **Guidance**: {guidance}\n"
                    f"- **File**: `{Path(video_path).name}`\n"
                )
                # Add to history
                self._generation_history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prompt": prompt[:60],
                    "frames": num_frames,
                    "resolution": f"{width}×{height}",
                    "path": video_path,
                })
                # Keep last 20
                self._generation_history = self._generation_history[:20]
            else:
                status = f"### ❌ Generation Failed\n\n{result.error}"
                logger.error(f"Failed: {result.error}")

            prog(1.0, "Done")
            self._current_status = "Complete ✓"

        except Exception as e:
            logger.exception(f"Generation error: {e}")
            status = f"### ❌ Error\n\n```\n{e}\n```"
        finally:
            self._is_generating = False

        return video_path, status, self._get_logs()

    def _cancel_generation(self) -> str:
        if self._is_generating:
            self._cancel_requested = True
            logger.warning("Cancel requested")
            return "⚠️ Cancellation requested…"
        return "Nothing to cancel."

    def _get_logs(self, max_lines: int = 200) -> str:
        entries = self.log_handler.get_logs(count=max_lines)
        return "\n".join(e.format() for e in entries)

    def _clear_logs(self) -> str:
        self.log_handler.clear()
        return ""

    def _format_history(self) -> str:
        if not self._generation_history:
            return "*No videos generated yet.*"
        lines = []
        for h in self._generation_history:
            lines.append(
                f"**{h['time']}** — {h['resolution']} / {h['frames']}f — "
                f"`{h['prompt']}…`"
            )
        return "\n\n".join(lines)

    def _on_resolution_change(self, preset: str):
        """Show / hide custom size inputs based on preset selection."""
        if preset == "Custom":
            return gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)

    # ── build interface ────────────────────────────────────────────

    def create_interface(self) -> "gr.Blocks":
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required: pip install gradio")

        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "-apple-system", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Cascadia Code", "Consolas", "monospace"],
        ).set(
            body_background_fill="#ffffff",
            body_background_fill_dark="#ffffff",
            block_background_fill="#ffffff",
            block_background_fill_dark="#ffffff",
            block_border_color="#e2e8f0",
            block_border_color_dark="#e2e8f0",
            block_border_width="1px",
            block_label_text_color="#64748b",
            block_title_text_color="#0f172a",
            block_shadow="0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)",
            input_background_fill="#f8fafc",
            input_background_fill_dark="#f8fafc",
            input_border_color="#e2e8f0",
            input_border_color_dark="#e2e8f0",
            button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
            button_primary_text_color="#ffffff",
            button_primary_shadow="0 4px 14px rgba(79,70,229,0.3)",
            shadow_spread="0px",
        )

        # Store for launch() — Gradio 6.0 moved theme/css to launch()
        self._theme = theme
        self._custom_css = CUSTOM_CSS

        with gr.Blocks(
            title="Video AI Studio",
        ) as app:

            # ── Header ──────────────────────────────────────
            gr.HTML("""
            <div class="app-header">
                <h1>🎬 Video AI Studio</h1>
                <p>AI Video Generation &nbsp;·&nbsp; Multi-Model &nbsp;·&nbsp; NVIDIA CUDA</p>
            </div>
            """)

            # GPU badge
            gpu_md = gr.Markdown(value=self._format_gpu_badge())

            # Hidden state for progress tracking
            progress_pct = gr.State(value=0.0)
            progress_msg = gr.State(value="")

            with gr.Tabs():

                # ━━━━━━━━━━━━━━━━  GENERATE TAB  ━━━━━━━━━━━━━━━━
                with gr.TabItem("🎬 Generate", id="generate"):
                    with gr.Row(equal_height=False):

                        # ── LEFT: Controls ──────────────────────
                        with gr.Column(scale=1, min_width=380):

                            # ── Model Selection ─────────────────
                            _model_choices = (
                                dropdown_choices() if REGISTRY_AVAILABLE
                                else ["Wan2.1-T2V-1.3B"]
                            )
                            _default_model_label = _model_choices[0]
                            _default_model_id = (
                                (model_from_label(_default_model_label).id
                                 if model_from_label(_default_model_label) else "wan2.1-t2v-1.3b")
                                if REGISTRY_AVAILABLE
                                else "wan2.1-t2v-1.3b"
                            )
                            _default_presets = MODEL_PRESETS.get(
                                _default_model_id, list(MODEL_PRESETS.values())[0]
                            )

                            model_selector = gr.Dropdown(
                                choices=_model_choices,
                                value=_default_model_label,
                                label="🤖 Model",
                                info="Select AI model (download first via System tab)",
                            )

                            model_info_md = gr.Markdown(
                                value="",
                                elem_classes=["model-info"],
                            )

                            prompt = gr.Textbox(
                                label="✏️ Prompt",
                                placeholder="Describe your video… e.g. 'A golden retriever running through a sunlit meadow, cinematic slow motion, golden hour lighting'",
                                lines=4,
                                max_lines=8,
                            )

                            negative_prompt = gr.Textbox(
                                label="🚫 Negative Prompt (optional)",
                                placeholder="Blurry, low quality, watermark, static…",
                                lines=3,
                                value="worst quality, inconsistent motion, blurry, jittery, distorted, shaky, glitchy, deformed, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static, Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, ugly, deformed, blurry, jitter, flicker, camera shake, sudden zoom, scene cuts, distorted face, extra limbs, broken anatomy, melting textures, low resolution, noisy artifacts, inconsistent character design",
                            )

                            gr.Markdown("### ⏱️ Duration & Resolution")

                            with gr.Row(elem_classes=["settings-row"]):
                                duration_preset = gr.Dropdown(
                                    choices=list(_default_presets["durations"].keys()),
                                    value=list(_default_presets["durations"].keys())[0],
                                    label="Duration",
                                    info=f"{_default_presets['fps']} fps",
                                )
                                resolution_preset = gr.Dropdown(
                                    choices=list(_default_presets["resolutions"].keys()),
                                    value=list(_default_presets["resolutions"].keys())[0],
                                    label="Resolution",
                                )

                            with gr.Row(elem_classes=["settings-row"]):
                                custom_width = gr.Slider(
                                    minimum=256, maximum=1280,
                                    value=832, step=16,
                                    label="Width",
                                    visible=False,
                                )
                                custom_height = gr.Slider(
                                    minimum=256, maximum=720,
                                    value=480, step=16,
                                    label="Height",
                                    visible=False,
                                )

                            resolution_preset.change(
                                fn=self._on_resolution_change,
                                inputs=[resolution_preset],
                                outputs=[custom_width, custom_height],
                            )

                            gr.Markdown("### 🎛️ Quality & Sampling")

                            with gr.Row(elem_classes=["settings-row"]):
                                quality_preset = gr.Dropdown(
                                    choices=list(QUALITY_PRESETS.keys()),
                                    value="Balanced (20 steps)",
                                    label="Quality",
                                )
                                guidance = gr.Slider(
                                    minimum=1.0, maximum=15.0,
                                    value=_default_presets.get("default_guidance", 5.0),
                                    step=0.5,
                                    label="Guidance Scale",
                                    info="Model-specific (see model info above)",
                                )

                            # Update presets when model changes (after guidance is defined)
                            model_selector.change(
                                fn=self._on_model_change,
                                inputs=[model_selector],
                                outputs=[resolution_preset, duration_preset, guidance, model_info_md],
                            )

                            seed = gr.Number(
                                value=-1,
                                label="🎲 Seed",
                                info="-1 = random, any positive number = reproducible",
                                precision=0,
                            )

                            with gr.Row():
                                generate_btn = gr.Button(
                                    "🎬  Generate Video",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["generate-btn"],
                                )
                                cancel_btn = gr.Button(
                                    "⏹ Cancel", variant="stop", size="sm",
                                )

                        # ── RIGHT: Output ───────────────────────
                        with gr.Column(scale=1, min_width=420):

                            video_output = gr.Video(
                                label="Generated Video",
                                interactive=False,
                                height=400,
                                elem_classes=["video-output-card"],
                            )

                            # ── Accurate Progress Bar ─────────
                            progress_html = gr.HTML(
                                value=self._render_progress_bar(0, 0, "Ready"),
                                visible=True,
                            )

                            status_output = gr.Markdown(
                                value="*Ready — enter a prompt and click Generate.*",
                                elem_classes=["result-status"],
                            )

                    # ── Event wiring ─────────────────────────────
                    logs_state = gr.State("")

                    # Progress polling timer — fires every 1s while generating
                    progress_timer = gr.Timer(value=1, active=False)

                    def _poll_progress():
                        """Return current progress bar HTML."""
                        return self._render_progress_bar(
                            self._progress_step,
                            self._progress_total_steps,
                            self._current_status,
                        )

                    progress_timer.tick(
                        fn=_poll_progress,
                        outputs=[progress_html],
                    )

                    def _on_generate_start():
                        """Activate progress timer when generation starts."""
                        return gr.Timer(active=True)

                    def _on_generate_end(video, status, logs):
                        """Deactivate progress timer when generation ends."""
                        final_bar = self._render_progress_bar(
                            self._progress_step,
                            self._progress_total_steps,
                            self._current_status,
                        )
                        return video, status, logs, final_bar, gr.Timer(active=False)

                    # Start timer, then run generation, then stop timer
                    generate_btn.click(
                        fn=_on_generate_start,
                        outputs=[progress_timer],
                    ).then(
                        fn=self._generate_video,
                        inputs=[
                            prompt, negative_prompt, model_selector,
                            duration_preset, resolution_preset,
                            custom_width, custom_height,
                            quality_preset, guidance, seed,
                        ],
                        outputs=[video_output, status_output, logs_state],
                    ).then(
                        fn=lambda v, s, l: _on_generate_end(v, s, l),
                        inputs=[video_output, status_output, logs_state],
                        outputs=[video_output, status_output, logs_state, progress_html, progress_timer],
                    )

                    cancel_btn.click(
                        fn=self._cancel_generation,
                        outputs=[status_output],
                    )

                # ━━━━━━━━━━━━━━━━  LOGS TAB  ━━━━━━━━━━━━━━━━━━━
                with gr.TabItem("📋 Logs", id="logs"):
                    with gr.Row():
                        refresh_logs = gr.Button("🔄 Refresh", size="sm")
                        clear_logs = gr.Button("🗑️ Clear", size="sm")

                    logs_output = gr.Textbox(
                        label="Generation Logs",
                        lines=30, max_lines=60,
                        interactive=False,
                        elem_classes=["log-console"],
                    )

                    refresh_logs.click(fn=self._get_logs, outputs=[logs_output])
                    clear_logs.click(fn=self._clear_logs, outputs=[logs_output])

                # ━━━━━━━━━━━━━━━━  SYSTEM TAB  ━━━━━━━━━━━━━━━━━━
                with gr.TabItem("💻 System", id="system"):
                    refresh_sys = gr.Button("🔄 Refresh", size="sm")
                    system_md = gr.Markdown(value=self._format_system_status())
                    refresh_sys.click(fn=self._format_system_status, outputs=[system_md])

                    gr.Markdown("---")

                    # Build model info table dynamically
                    _model_info_rows = ""
                    if REGISTRY_AVAILABLE:
                        for spec in MODEL_REGISTRY.values():
                            _model_info_rows += (
                                f"| **{spec.display_name}** | {spec.parameters} | "
                                f"{spec.native_fps} | {spec.max_num_frames} | "
                                f"{spec.default_width}×{spec.default_height} | "
                                f"~{spec.estimate_peak_vram(spec.default_width, spec.default_height, spec.default_num_frames):.0f} GB | "
                                f"{spec.license} |\n"
                            )
                    else:
                        _model_info_rows = "| Wan2.1-T2V-1.3B | 1.3B | 16 | 81 | 832×480 | ~8 GB | Apache-2.0 |\n"

                    gr.Markdown(f"""### 🤖 Supported Models

| Model | Params | FPS | Max Frames | Default Res | Est. VRAM | License |
|-------|--------|-----|------------|-------------|-----------|---------|
{_model_info_rows}

> Select a model from the **Generate** tab dropdown. Models must be downloaded first.
> Use `python download_models.py` to download model weights.
""")

                # ━━━━━━━━━━━━━━━━  HISTORY TAB  ━━━━━━━━━━━━━━━━━
                with gr.TabItem("📜 History", id="history"):
                    refresh_hist = gr.Button("🔄 Refresh", size="sm")
                    history_md = gr.Markdown(value=self._format_history())
                    refresh_hist.click(fn=self._format_history, outputs=[history_md])

                # ━━━━━━━━━━━━━━━━  HELP TAB  ━━━━━━━━━━━━━━━━━━━
                with gr.TabItem("❓ Help", id="help"):
                    gr.Markdown("""## How to Use Video AI Studio

### Quick Start
1. **Select a model** from the dropdown (must be downloaded first)
2. Enter a detailed prompt describing your video
3. Pick a **duration** and **resolution** (auto-adjusted per model)
4. Click **🎬 Generate Video**
5. First run loads the model (~5 min); subsequent runs are faster

### Writing Good Prompts
- **Be specific**: *"A cat sitting on a windowsill watching rain, soft indoor lighting, shallow depth of field"*
- **Add motion**: *"camera slowly pans left"*, *"wind blowing through trees"*
- **Add style**: *"cinematic"*, *"anime"*, *"photorealistic"*, *"oil painting"*
- **Use negative prompt** to avoid: *"blurry, low quality, watermark, static"*

### Model Guide

| Model | Best For | Speed | Notes |
|-------|----------|-------|-------|
| **Wan2.1-T2V-1.3B** | General purpose | Medium | Default, great quality/speed balance |
| **CogVideoX-2B** | Short clips | Fast | 720×480, 6 sec max |
| **CogVideoX-5B** | Higher quality | Slow | Needs CPU offload |
| **LTX-Video 2B** | Fast preview | Very Fast | 8 steps only, 24 fps |

### Tips
- Start with **Wan2.1-T2V-1.3B** for best quality on 16 GB VRAM
- **CogVideoX-2B** is great for quick drafts (~4 GB with offload)
- **LTX-Video** is the fastest (8 steps!) but lower quality
- Longer videos need more VRAM — use lower resolution if OOM
- Set a **seed** to reproduce the exact same video
- The model loads once and stays in memory for faster subsequent generations
- Switching models unloads the current one first
""")

                # ━━━━━━━━━━━━━━  AGGRESSIVE TAB (if available)  ━━
                if AGGRESSIVE_TAB_AVAILABLE:
                    with gr.TabItem("🔥 Aggressive", id="aggressive"):
                        create_aggressive_generator_tab()

                # ━━━━━━━━━━━━━━  IMAGE-TO-VIDEO TAB (if available)  ━━
                if IMAGE_MOTION_TAB_AVAILABLE:
                    with gr.TabItem("Image to Video", id="image_motion"):
                        create_image_motion_tab()

                # ━━━━━━━━━━━━━━  DEEPSEEK LLM TAB (if available)  ━━
                if DEEPSEEK_TAB_AVAILABLE:
                    with gr.TabItem("DeepSeek (Offline LLM)", id="deepseek_llm"):
                        create_deepseek_tab()

            # ── Footer ──────────────────────────────────────
            gr.HTML("""
            <div class="app-footer">
                Video AI Studio &nbsp;·&nbsp; Multi-Model &nbsp;·&nbsp; Local AI Generation &nbsp;·&nbsp; Your data never leaves your machine
            </div>
            """)

        return app

    # ── launch ────────────────────────────────────────────────────

    def launch(self):
        app = self.create_interface()
        logger.info(f"Launching Video AI Studio on port {self.port}")
        logger.info(f"Output directory: {self.output_dir}")
        app.launch(
            server_name="0.0.0.0",
            server_port=self.port,
            share=self.share,
            show_error=True,
            inbrowser=True,
            theme=self._theme,
            css=self._custom_css,
        )


# ─── Public launcher ─────────────────────────────────────────────────────────

def launch_ui(
    config_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    share: bool = False,
    port: int = 7860,
):
    """Launch the Video AI web interface."""
    ui = WebUI(
        config_dir=config_dir,
        output_dir=output_dir,
        share=share,
        port=port,
    )
    ui.launch()


if __name__ == "__main__":
    launch_ui()
