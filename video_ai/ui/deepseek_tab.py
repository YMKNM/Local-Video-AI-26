"""
DeepSeek (Offline LLM) — Gradio Tab  (Copilot-style layout)
=============================================================
Two-panel design:
  Left  – Model selector, info card, load/unload, generation sliders, system status
  Right – Chat conversation history (gr.Chatbot), prompt input, Generate + Stop buttons

Features preserved from the original tab:
  - 3 validated model dropdown with auto-defaults
  - VRAM / RAM checks, quantisation info
  - Streaming generation
  - System status panel

New:
  - Conversation history with user / assistant distinction
  - Stop Generation button (safe interruption via threading.Event)
  - Tokens-per-second metric
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Generator, List, Optional, Tuple

import gradio as gr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
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

def _spec_info_md(spec: Optional["DeepSeekModelSpec"] = None) -> str:
    """Markdown card showing model details."""
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
    """Return system status as Markdown code block."""
    if not _BACKEND_AVAILABLE:
        return "Backend not available."
    mgr = get_manager()
    return f"```\n{mgr.status_text()}\n```"


# ---------------------------------------------------------------------------
# Tab Constructor
# ---------------------------------------------------------------------------

def create_deepseek_tab() -> None:  # noqa: C901
    """Create the Copilot-style DeepSeek chat tab.

    Call inside a ``gr.TabItem`` context.
    """
    if not _BACKEND_AVAILABLE:
        gr.Markdown(
            "## DeepSeek backend is not available.\n\n"
            "Install dependencies:\n```\npip install transformers accelerate bitsandbytes psutil\n```"
        )
        return

    choices = deepseek_dropdown_choices()

    gr.Markdown("### DeepSeek — Local Offline LLM")

    with gr.Row(equal_height=False):
        # ================================================================
        # LEFT PANEL – settings & status
        # ================================================================
        with gr.Column(scale=1, min_width=320):
            # -- Model selector -----------------------------------------
            gr.Markdown("#### Model")
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
                lines=2,
                max_lines=4,
            )

            # -- Generation sliders -------------------------------------
            gr.Markdown("---")
            gr.Markdown("#### Generation Settings")

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
                label="Top-p (nucleus)",
            )

            # -- System status ------------------------------------------
            gr.Markdown("---")
            gr.Markdown("#### System Status")
            status_box = gr.Markdown(value=_status_md())
            refresh_btn = gr.Button("Refresh Status", size="sm")

        # ================================================================
        # RIGHT PANEL – chat area
        # ================================================================
        with gr.Column(scale=2, min_width=480):
            chatbot = gr.Chatbot(
                value=[],
                height=520,
                label="Conversation",
                render_markdown=True,
            )

            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Type your message and press Enter or click Generate …",
                lines=3,
                max_lines=8,
            )

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", scale=3)
                stop_btn = gr.Button("Stop", variant="stop", scale=1, interactive=False)
                clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

    # Hidden state: whether generation is running (for button interactivity)
    is_generating = gr.State(False)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_model_change(label: str):
        """Dropdown change → update info card + slider defaults."""
        spec = spec_from_label(label)
        info = _spec_info_md(spec)
        if spec:
            return (
                info,
                gr.update(value=spec.default_max_tokens),
                gr.update(value=spec.default_temperature),
                gr.update(value=spec.default_top_p),
            )
        return info, gr.update(), gr.update(), gr.update()

    model_dd.change(
        fn=_on_model_change,
        inputs=[model_dd],
        outputs=[model_info, max_tokens, temperature, top_p],
    )

    # -- Load / Unload --------------------------------------------------

    def _load_model(label: str, progress=gr.Progress(track_tqdm=True)):
        spec = spec_from_label(label)
        if spec is None:
            return "Error: unknown model selection.", _status_md()
        mgr = get_manager()
        try:
            result = mgr.load(spec, progress_cb=lambda m: None)
            return result, _status_md()
        except Exception as e:
            logger.exception("Error loading model")
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

    unload_btn.click(fn=_unload, inputs=[], outputs=[load_status, status_box])

    # -- Generate (streaming) -------------------------------------------

    def _generate(
        prompt: str,
        history: list,
        tokens: int,
        temp: float,
        tp: float,
    ):
        """Yield updated chat history as tokens stream in."""
        if not prompt or not prompt.strip():
            yield history, "", gr.update(interactive=False), gr.update(interactive=False), _status_md()
            return

        mgr = get_manager()
        if not mgr.is_loaded:
            # Append an error message from assistant
            history = history + [
                gr.ChatMessage(role="user", content=prompt.strip()),
                gr.ChatMessage(role="assistant", content="Error: no model loaded. Click **Load Model** first."),
            ]
            yield history, "", gr.update(interactive=True), gr.update(interactive=False), _status_md()
            return

        # Append user message
        history = history + [gr.ChatMessage(role="user", content=prompt.strip())]
        # Placeholder for assistant
        history = history + [gr.ChatMessage(role="assistant", content="")]
        # Enable Stop, disable Generate while running
        yield history, "", gr.update(interactive=False), gr.update(interactive=True), _status_md()

        try:
            for partial in mgr.generate(
                prompt=prompt.strip(),
                max_new_tokens=int(tokens),
                temperature=temp,
                top_p=tp,
                stream=True,
            ):
                # Update the last assistant message in-place
                history[-1] = gr.ChatMessage(role="assistant", content=partial)
                yield history, "", gr.update(interactive=False), gr.update(interactive=True), _status_md()

            # If stopped early, append a note
            if mgr._stop_event.is_set():
                current = history[-1].content if hasattr(history[-1], "content") else str(history[-1])
                history[-1] = gr.ChatMessage(
                    role="assistant",
                    content=current + "\n\n_(generation stopped by user)_",
                )
        except Exception as e:
            logger.exception("Generation error")
            history[-1] = gr.ChatMessage(
                role="assistant",
                content=f"ERROR: {e}\n```\n{traceback.format_exc()}\n```",
            )

        # Re-enable Generate, disable Stop
        yield history, "", gr.update(interactive=True), gr.update(interactive=False), _status_md()

    gen_click_event = generate_btn.click(
        fn=_generate,
        inputs=[prompt_input, chatbot, max_tokens, temperature, top_p],
        outputs=[chatbot, prompt_input, generate_btn, stop_btn, status_box],
    )

    # Submit on Enter as well
    gen_submit_event = prompt_input.submit(
        fn=_generate,
        inputs=[prompt_input, chatbot, max_tokens, temperature, top_p],
        outputs=[chatbot, prompt_input, generate_btn, stop_btn, status_box],
    )

    # -- Stop Generation -------------------------------------------------

    def _stop():
        """Signal the backend to stop after the current token."""
        if _BACKEND_AVAILABLE:
            get_manager().request_stop()
        return gr.update(interactive=True), gr.update(interactive=False)

    stop_btn.click(
        fn=_stop,
        inputs=[],
        outputs=[generate_btn, stop_btn],
        cancels=[gen_click_event, gen_submit_event],
    )

    # -- Clear Chat ------------------------------------------------------

    def _clear_chat():
        return [], _status_md()

    clear_btn.click(fn=_clear_chat, inputs=[], outputs=[chatbot, status_box])

    # -- Refresh Status --------------------------------------------------

    def _refresh():
        return _status_md()

    refresh_btn.click(fn=_refresh, inputs=[], outputs=[status_box])
