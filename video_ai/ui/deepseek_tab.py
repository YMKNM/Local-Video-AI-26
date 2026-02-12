"""
DeepSeek (Offline LLM) — Gradio Tab
====================================
Adds a "DeepSeek (Offline LLM)" tab to Video AI Studio.

Features:
  - Model selector dropdown (3 validated models)
  - Prompt input, max-tokens, temperature, top-p controls
  - Context-length display (auto-updates per model)
  - Load / Unload / Generate buttons
  - Streaming text output
  - System status panel (VRAM, RAM, backend)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import gradio as gr

logger = logging.getLogger(__name__)

# Lazy import backend so the file stays importable even when heavy deps
# are missing.
_BACKEND_AVAILABLE = True
try:
    from ..deepseek import (
        DEEPSEEK_MODELS,
        DeepSeekModelSpec,
        deepseek_dropdown_choices,
        spec_from_label,
        get_gpu_info,
        get_manager,
    )
except ImportError:
    _BACKEND_AVAILABLE = False
    logger.warning("DeepSeek backend not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_info_md(spec: Optional[DeepSeekModelSpec]) -> str:
    """Build a Markdown status block for a model spec."""
    if spec is None:
        return "_Select a model above._"
    return (
        f"**{spec.display_name}**\n\n"
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| Parameters | {spec.parameters} |\n"
        f"| Quantisation | {spec.quant.value} |\n"
        f"| Context length | {spec.context_length:,} tokens |\n"
        f"| Disk size | ~{spec.disk_gb} GB |\n"
        f"| Est. VRAM | ~{spec.vram_gb} GB |\n"
        f"| Est. RAM | ~{spec.ram_gb} GB |\n"
        f"| License | {spec.license} |\n"
        f"| HuggingFace | [{spec.repo_id}]({spec.hf_url}) |\n\n"
        f"_{spec.notes}_"
    )


def _status_md() -> str:
    """Return system status as Markdown."""
    if not _BACKEND_AVAILABLE:
        return "Backend not available."
    mgr = get_manager()
    return f"```\n{mgr.status_text()}\n```"


# ---------------------------------------------------------------------------
# Tab Constructor
# ---------------------------------------------------------------------------

def create_deepseek_tab() -> None:
    """Create the full DeepSeek (Offline LLM) tab contents.

    Call this inside a ``gr.TabItem`` context.
    """
    if not _BACKEND_AVAILABLE:
        gr.Markdown("## DeepSeek backend is not available.\n\nPlease install dependencies:\n```\npip install transformers accelerate bitsandbytes psutil\n```")
        return

    choices = deepseek_dropdown_choices()

    gr.Markdown("### DeepSeek — Local Offline LLM Inference")

    with gr.Row(equal_height=False):
        # ── Left column: controls ───────────────────────────
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=choices,
                value=choices[0] if choices else None,
                label="Model",
                info="Select a DeepSeek model to load",
                interactive=True,
            )

            model_info = gr.Markdown(
                value=_spec_info_md(spec_from_label(choices[0]) if choices else None),
                label="Model Details",
            )

            with gr.Row():
                load_btn = gr.Button("Load Model", variant="primary", scale=2)
                unload_btn = gr.Button("Unload", variant="secondary", scale=1)

            load_status = gr.Textbox(
                label="Load Status",
                interactive=False,
                lines=3,
                max_lines=5,
            )

            gr.Markdown("---")
            gr.Markdown("#### Generation Settings")

            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here ...",
                lines=5,
                max_lines=15,
            )

            max_tokens = gr.Slider(
                minimum=16,
                maximum=4096,
                value=512,
                step=16,
                label="Max New Tokens",
                info="Maximum number of tokens to generate",
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="Temperature",
                info="0 = deterministic, higher = more creative",
            )

            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p (nucleus sampling)",
            )

            generate_btn = gr.Button(
                "Generate",
                variant="primary",
                interactive=True,
            )

        # ── Right column: output + status ───────────────────
        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="Output",
                interactive=False,
                lines=22,
                max_lines=40,
                show_copy_button=True,
            )

            gr.Markdown("#### System Status")
            status_box = gr.Textbox(
                label="System Status",
                interactive=False,
                lines=7,
                max_lines=10,
                value="No model loaded.",
            )

            refresh_btn = gr.Button("Refresh Status", size="sm")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_model_change(label: str):
        """When the dropdown changes, update the info panel and defaults."""
        spec = spec_from_label(label)
        info = _spec_info_md(spec)
        defaults = {}
        if spec:
            defaults = {
                max_tokens: gr.update(value=spec.default_max_tokens),
                temperature: gr.update(value=spec.default_temperature),
                top_p: gr.update(value=spec.default_top_p),
            }
        return info, defaults.get(max_tokens, gr.update()), defaults.get(temperature, gr.update()), defaults.get(top_p, gr.update())

    model_dd.change(
        fn=_on_model_change,
        inputs=[model_dd],
        outputs=[model_info, max_tokens, temperature, top_p],
    )

    def _load_model(label: str, progress=gr.Progress(track_tqdm=True)):
        """Load the selected model."""
        spec = spec_from_label(label)
        if spec is None:
            return "Error: unknown model selection.", _status_md()

        mgr = get_manager()
        messages = []

        def _cb(msg: str):
            messages.append(msg)

        try:
            result = mgr.load(spec, progress_cb=_cb)
            return result, _status_md()
        except RuntimeError as e:
            return f"FAILED: {e}", _status_md()
        except Exception as e:
            logger.exception("Unexpected error loading model")
            return f"FAILED: {e}\n{traceback.format_exc()}", _status_md()

    load_btn.click(
        fn=_load_model,
        inputs=[model_dd],
        outputs=[load_status, status_box],
    )

    def _unload():
        mgr = get_manager()
        mgr.unload()
        return "Model unloaded.", _status_md()

    unload_btn.click(
        fn=_unload,
        inputs=[],
        outputs=[load_status, status_box],
    )

    def _generate(prompt: str, tokens: int, temp: float, tp: float):
        """Run inference with streaming output."""
        if not prompt or not prompt.strip():
            yield "Error: please enter a prompt."
            return

        mgr = get_manager()
        if not mgr.is_loaded:
            yield "Error: no model loaded. Click 'Load Model' first."
            return

        try:
            for partial in mgr.generate(
                prompt=prompt.strip(),
                max_new_tokens=int(tokens),
                temperature=temp,
                top_p=tp,
                stream=True,
            ):
                yield partial
        except Exception as e:
            logger.exception("Generation error")
            yield f"ERROR: {e}\n{traceback.format_exc()}"

    generate_btn.click(
        fn=_generate,
        inputs=[prompt_input, max_tokens, temperature, top_p],
        outputs=[output_box],
    )

    def _refresh():
        return _status_md()

    refresh_btn.click(fn=_refresh, inputs=[], outputs=[status_box])
