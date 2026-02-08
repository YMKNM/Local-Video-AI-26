"""
Image-to-Video Animation Tab

Gradio UI tab that exposes the image_motion pipeline.  Users upload a
still image, type an action prompt (e.g. "Make the boy run forward"),
and receive an animated video clip.

This tab follows the same pattern as ``aggressive_generator_tab.py``
and is imported conditionally by ``web_ui.py``.
"""

from __future__ import annotations

import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import the pipeline so that this file can be safely imported
# even when heavy deps are missing.
_pipeline = None
_PIPELINE_AVAILABLE = True

try:
    from ..image_motion import (
        ObjectVideoPipeline,
        PipelineConfig,
        PipelineResult,
    )
except ImportError:
    _PIPELINE_AVAILABLE = False
    ObjectVideoPipeline = None  # type: ignore
    PipelineConfig = None       # type: ignore
    PipelineResult = None       # type: ignore

# Attempt to import model registry for dropdown
try:
    from ..runtime.model_registry import dropdown_choices, model_from_label
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False
    dropdown_choices = None
    model_from_label = None


# ── Defaults ──────────────────────────────────────────────────────────

_DEFAULT_PROMPT = "Make the subject walk forward slowly"
_DEFAULT_INTENSITY = 0.7
_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 6.0
_DEFAULT_FRAMES = 24
_DEFAULT_FPS = 8


# ── Generation logic ─────────────────────────────────────────────────

def _run_image_to_video(
    image: Optional[np.ndarray],
    action_prompt: str,
    intensity: float,
    num_frames: int,
    fps: int,
    steps: int,
    guidance: float,
    model_label: str,
    seed: int,
    stabilize: bool,
) -> tuple:
    """
    Called by the Generate button.
    Returns (video_path_or_None, status_text).
    """
    global _pipeline

    if not _PIPELINE_AVAILABLE:
        return None, "Error: image_motion module not available. Check logs."

    if image is None:
        return None, "Error: Please upload an image first."

    if not action_prompt.strip():
        return None, "Error: Please enter an action prompt."

    # Save uploaded image to a temp file
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(image)
    tmp_dir = Path(tempfile.mkdtemp(prefix="img2vid_"))
    img_path = tmp_dir / "input.png"
    img_pil.save(str(img_path))

    # Resolve model
    model_name = "wan2.1-t2v-1.3b"
    if _REGISTRY_AVAILABLE and model_from_label and model_label:
        spec = model_from_label(model_label)
        if spec:
            model_name = spec.id

    # Build config
    cfg = PipelineConfig(
        output_dir=str(tmp_dir / "output"),
        width=512,
        height=512,
        num_frames=int(num_frames),
        fps=int(fps),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        seed=int(seed) if seed and seed > 0 else None,
        motion_intensity=float(intensity),
        stabilize=bool(stabilize),
        model_name=model_name,
    )

    try:
        if _pipeline is None:
            _pipeline = ObjectVideoPipeline(config=cfg)
        result = _pipeline.run(
            image_path=str(img_path),
            action_prompt=action_prompt,
            config=cfg,
        )
    except Exception as e:
        logger.error("Image-to-video failed: %s", traceback.format_exc())
        return None, f"Error: {e}"

    if result.success and result.video_path:
        warn_text = ""
        if result.warnings:
            warn_text = "\nWarnings:\n- " + "\n- ".join(result.warnings)
        seg_info = ""
        if result.segmentation:
            retries = getattr(result.segmentation, '_retries_used', 0)
            seg_info = (
                f"Seg: {result.segmentation.quality.value} "
                f"({result.segmentation.confidence:.0%} conf"
                f"{', ' + str(retries) + ' retries' if retries > 0 else ''})"
            )
        status = (
            f"Done in {result.elapsed_seconds:.1f}s | "
            f"{len(result.frame_paths)} frames | "
            f"Action: {result.intent.action.value if result.intent else '?'} | "
            f"{seg_info}"
            f"{warn_text}"
        )
        return result.video_path, status
    else:
        err = result.error or "Unknown error"
        return None, f"Generation failed: {err}"


# ── Tab builder ───────────────────────────────────────────────────────

def create_image_motion_tab() -> gr.Tab:
    """Create the Image-to-Video Animation tab for the main UI."""

    with gr.Tab("Image to Video") as tab:
        gr.Markdown("""
        ## Image-to-Video Animation (v2 -- Pose-Conditioned)

        Upload a still image and describe the motion you want.
        The pipeline uses SAM 2 segmentation (with retry), DWPose skeletal
        extraction, procedural motion synthesis, and AnimateDiff + ControlNet
        for pose-conditioned video generation.

        **Key features:**
        - SAM 2 segmentation with retry (never reduces motion on low confidence)
        - Skeletal pose detection (DWPose / OpenPose / MediaPipe)
        - Biomechanically correct walk/run/jump/dance cycles
        - Temporal consistency with optical flow + anti-ghosting

        **Example prompts:**
        - *Make the boy run forward*
        - *Make the woman walk slowly*
        - *Make the dog jump*
        - *Make the car drive away*
        """)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                action_prompt = gr.Textbox(
                    label="Action Prompt",
                    placeholder="Make the boy run forward",
                    value="",
                    lines=2,
                )

                with gr.Accordion("Settings", open=False):
                    intensity_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=_DEFAULT_INTENSITY,
                        step=0.05,
                        label="Motion Intensity",
                        info="Higher = more dramatic motion",
                    )
                    num_frames_slider = gr.Slider(
                        minimum=8,
                        maximum=81,
                        value=_DEFAULT_FRAMES,
                        step=1,
                        label="Number of Frames",
                    )
                    fps_slider = gr.Slider(
                        minimum=4,
                        maximum=24,
                        value=_DEFAULT_FPS,
                        step=1,
                        label="FPS",
                    )
                    steps_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=_DEFAULT_STEPS,
                        step=1,
                        label="Inference Steps",
                        info="More steps = higher quality, slower",
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=_DEFAULT_GUIDANCE,
                        step=0.5,
                        label="Guidance Scale",
                    )

                    # Model dropdown
                    if _REGISTRY_AVAILABLE and dropdown_choices:
                        try:
                            choices = dropdown_choices()
                        except Exception:
                            choices = []
                    else:
                        choices = []
                    model_dropdown = gr.Dropdown(
                        choices=choices if choices else ["Default"],
                        value=choices[0] if choices else "Default",
                        label="Model",
                    )

                    seed_input = gr.Number(
                        value=0,
                        label="Seed (0 = random)",
                        precision=0,
                    )
                    stabilize_check = gr.Checkbox(
                        value=True,
                        label="Temporal Stabilisation",
                        info="Smooth jitter in output frames",
                    )

                generate_btn = gr.Button(
                    "Generate Video",
                    variant="primary",
                    size="lg",
                )

            # ── Right column: outputs ────────────────────────
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Generated Video",
                    autoplay=True,
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                )

        # ── Wire events ──────────────────────────────────────
        generate_btn.click(
            fn=_run_image_to_video,
            inputs=[
                image_input,
                action_prompt,
                intensity_slider,
                num_frames_slider,
                fps_slider,
                steps_slider,
                guidance_slider,
                model_dropdown,
                seed_input,
                stabilize_check,
            ],
            outputs=[video_output, status_text],
        )

    return tab
