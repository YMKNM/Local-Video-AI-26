"""
DeepSeek Offline LLM Backend
=============================
Manages downloading, loading, inference, and memory for DeepSeek models
running fully offline via HuggingFace Transformers.

Supported models (validated for RTX 5080, 16 GB VRAM):
  A) DeepSeek-R1-Distill-Qwen-1.5B  — BF16, ~4 GB VRAM
  B) DeepSeek-R1-Distill-Qwen-7B    — 4-bit NF4, ~6 GB VRAM
  C) DeepSeek-R1-Distill-Qwen-14B   — 4-bit NF4, ~10 GB VRAM
"""

from __future__ import annotations

import gc
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – keeps module importable even when deps are missing
# ---------------------------------------------------------------------------
_torch = None
_transformers = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers as _tf
        _transformers = _tf
    return _transformers


# ---------------------------------------------------------------------------
# Model Specification
# ---------------------------------------------------------------------------

class QuantMode(Enum):
    BF16 = "bf16"
    NF4 = "nf4"          # bitsandbytes 4-bit


@dataclass
class DeepSeekModelSpec:
    """Everything needed to download, load, and run one DeepSeek model."""
    model_id: str                  # internal key
    display_name: str              # human-readable
    repo_id: str                   # HuggingFace repo
    parameters: str                # e.g. "1.5B"
    quant: QuantMode               # quantisation strategy
    disk_gb: float                 # approximate download size (GB)
    vram_gb: float                 # estimated VRAM at inference (GB)
    ram_gb: float                  # estimated system RAM needed (GB)
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    context_length: int = 32768    # model's native ctx window
    license: str = "MIT"
    hf_url: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry of the 3 validated models
# ---------------------------------------------------------------------------

DEEPSEEK_MODELS: Dict[str, DeepSeekModelSpec] = {}

def _build_registry():
    specs = [
        DeepSeekModelSpec(
            model_id="deepseek-r1-1.5b",
            display_name="DeepSeek-R1 Distill Qwen 1.5B",
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            parameters="1.5B",
            quant=QuantMode.BF16,
            disk_gb=3.6,
            vram_gb=4.0,
            ram_gb=6.0,
            default_max_tokens=512,
            context_length=32768,
            hf_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            notes="Fastest. Native BF16, no quantisation needed.",
        ),
        DeepSeekModelSpec(
            model_id="deepseek-r1-7b",
            display_name="DeepSeek-R1 Distill Qwen 7B",
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            parameters="7B",
            quant=QuantMode.NF4,
            disk_gb=15.2,
            vram_gb=6.0,
            ram_gb=10.0,
            default_max_tokens=512,
            context_length=32768,
            hf_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            notes="Balanced. 4-bit NF4 via bitsandbytes.",
        ),
        DeepSeekModelSpec(
            model_id="deepseek-r1-14b",
            display_name="DeepSeek-R1 Distill Qwen 14B",
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            parameters="14B",
            quant=QuantMode.NF4,
            disk_gb=29.5,
            vram_gb=10.0,
            ram_gb=16.0,
            default_max_tokens=512,
            context_length=32768,
            hf_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            notes="Best quality. 4-bit NF4 via bitsandbytes. AIME 69.7%, MATH-500 93.9%.",
        ),
    ]
    for s in specs:
        DEEPSEEK_MODELS[s.model_id] = s


_build_registry()


def deepseek_dropdown_choices() -> List[str]:
    """Return list of display labels for the UI dropdown."""
    return [
        f"{s.display_name}  [{s.parameters}, {s.quant.value}]"
        for s in DEEPSEEK_MODELS.values()
    ]


def spec_from_label(label: str) -> Optional[DeepSeekModelSpec]:
    """Resolve a dropdown label back to its spec."""
    for s in DEEPSEEK_MODELS.values():
        tag = f"{s.display_name}  [{s.parameters}, {s.quant.value}]"
        if tag == label:
            return s
    return None


# ---------------------------------------------------------------------------
# GPU / Memory helpers
# ---------------------------------------------------------------------------

def get_gpu_info() -> Dict[str, Any]:
    """Return current GPU memory stats (in GB)."""
    torch = _get_torch()
    info: Dict[str, Any] = {"available": False}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["available"] = True
        info["name"] = torch.cuda.get_device_name(idx)
        info["total_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 2)
        info["allocated_gb"] = round(torch.cuda.memory_allocated(idx) / 1e9, 2)
        info["reserved_gb"] = round(torch.cuda.memory_reserved(idx) / 1e9, 2)
        info["free_gb"] = round(info["total_gb"] - info["allocated_gb"], 2)
    return info


def _check_vram_fits(spec: DeepSeekModelSpec) -> Tuple[bool, str]:
    """Check if a model will fit in available VRAM.  Returns (ok, reason)."""
    info = get_gpu_info()
    if not info["available"]:
        return False, "No CUDA GPU detected."
    free = info["free_gb"]
    needed = spec.vram_gb
    # Leave 1 GB headroom for OS/drivers/KV-cache
    if free < needed + 1.0:
        return False, (
            f"Insufficient VRAM: {free:.1f} GB free, "
            f"model needs ~{needed:.1f} GB + 1 GB headroom."
        )
    return True, f"OK — {free:.1f} GB free, model needs ~{needed:.1f} GB."


# ---------------------------------------------------------------------------
# Model Manager (singleton)
# ---------------------------------------------------------------------------

class DeepSeekManager:
    """Thread-safe manager that loads at most one DeepSeek model at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._streamer = None
        self._loaded_spec: Optional[DeepSeekModelSpec] = None
        self._load_time: float = 0.0

    # ── properties ──────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_spec(self) -> Optional[DeepSeekModelSpec]:
        return self._loaded_spec

    # ── unload ──────────────────────────────────────────────

    def unload(self) -> None:
        """Release current model and free VRAM/RAM."""
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._model is not None:
            logger.info("Unloading DeepSeek model: %s", self._loaded_spec.model_id if self._loaded_spec else "?")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._streamer = None
            self._loaded_spec = None
            self._load_time = 0.0
            torch = _get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            logger.info("Model unloaded, VRAM freed.")

    # ── load ────────────────────────────────────────────────

    def load(
        self,
        spec: DeepSeekModelSpec,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Download (if needed) and load the specified model.

        Returns a status string.  Raises RuntimeError on failure.
        """
        with self._lock:
            # Already loaded?
            if self._loaded_spec and self._loaded_spec.model_id == spec.model_id:
                return f"Model already loaded: {spec.display_name}"

            # Unload any previous model first
            self._unload_locked()

            # VRAM check
            fits, reason = _check_vram_fits(spec)
            if not fits:
                raise RuntimeError(reason)

            torch = _get_torch()
            transformers = _get_transformers()

            def _msg(m: str):
                logger.info(m)
                if progress_cb:
                    progress_cb(m)

            t0 = time.time()
            _msg(f"Downloading / caching {spec.display_name} from HuggingFace ...")

            try:
                # Tokenizer
                _msg("Loading tokenizer ...")
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    spec.repo_id,
                    trust_remote_code=True,
                )

                # Model loading strategy based on quant mode
                if spec.quant == QuantMode.NF4:
                    _msg(f"Loading model with 4-bit NF4 quantisation (bitsandbytes) ...")
                    from transformers import BitsAndBytesConfig
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        spec.repo_id,
                        quantization_config=bnb_cfg,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    # BF16 native
                    _msg("Loading model in BF16 ...")
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        spec.repo_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )

                model.eval()
                elapsed = time.time() - t0
                self._model = model
                self._tokenizer = tokenizer
                self._loaded_spec = spec
                self._load_time = elapsed

                gpu = get_gpu_info()
                vram_used = gpu.get("allocated_gb", 0)
                _msg(
                    f"Loaded {spec.display_name} in {elapsed:.1f}s  |  "
                    f"VRAM: {vram_used:.1f} / {gpu.get('total_gb', 0):.1f} GB"
                )
                return (
                    f"Loaded {spec.display_name} in {elapsed:.1f}s\n"
                    f"VRAM used: {vram_used:.1f} GB / {gpu.get('total_gb', 0):.1f} GB\n"
                    f"Quantisation: {spec.quant.value}"
                )

            except Exception as exc:
                # Clean up partial state
                self._unload_locked()
                logger.exception("Failed to load %s", spec.display_name)
                raise RuntimeError(f"Failed to load {spec.display_name}: {exc}") from exc

    # ── generate ────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate text from the loaded model.

        Yields partial strings when *stream=True*, or a single final string.
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Load a model first.")

        torch = _get_torch()
        transformers = _get_transformers()

        spec = self._loaded_spec
        tokenizer = self._tokenizer
        model = self._model

        # Clamp params
        temperature = max(0.01, min(temperature, 1.5))
        top_p = max(0.0, min(top_p, 1.0))
        max_new_tokens = max(1, min(max_new_tokens, spec.context_length))

        # Tokenise
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback if chat template not available
            input_text = f"<|begin▁of▁sentence|>User: {prompt}\n\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        if stream:
            # Streaming with TextIteratorStreamer
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.01),
                streamer=streamer,
            )

            thread = threading.Thread(
                target=model.generate, kwargs=gen_kwargs, daemon=True,
            )
            thread.start()

            generated = ""
            for chunk in streamer:
                generated += chunk
                yield generated       # yield cumulative text

            thread.join(timeout=120)
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0.01),
                )
            new_tokens = outputs[0][input_len:]
            yield tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── status ──────────────────────────────────────────────

    def status_dict(self) -> Dict[str, Any]:
        """Return a dict summarising current state (for the UI panel)."""
        gpu = get_gpu_info()
        import psutil
        ram = psutil.virtual_memory()
        result = {
            "model_loaded": self.is_loaded,
            "model_name": self._loaded_spec.display_name if self._loaded_spec else "None",
            "model_params": self._loaded_spec.parameters if self._loaded_spec else "-",
            "quant": self._loaded_spec.quant.value if self._loaded_spec else "-",
            "backend": "Transformers + bitsandbytes" if (self._loaded_spec and self._loaded_spec.quant == QuantMode.NF4) else "Transformers BF16",
            "load_time_s": round(self._load_time, 1),
            "vram_allocated_gb": gpu.get("allocated_gb", 0),
            "vram_total_gb": gpu.get("total_gb", 0),
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_total_gb": round(ram.total / 1e9, 1),
            "gpu_name": gpu.get("name", "N/A"),
        }
        return result

    def status_text(self) -> str:
        """Human-readable status string for the UI."""
        s = self.status_dict()
        lines = [
            f"Model:   {s['model_name']}  ({s['model_params']}, {s['quant']})",
            f"Backend: {s['backend']}",
            f"Load:    {s['load_time_s']}s",
            f"VRAM:    {s['vram_allocated_gb']:.1f} / {s['vram_total_gb']:.1f} GB",
            f"RAM:     {s['ram_used_gb']:.1f} / {s['ram_total_gb']:.1f} GB",
            f"GPU:     {s['gpu_name']}",
        ]
        if not s["model_loaded"]:
            lines = ["No model loaded."]
            lines.append(f"VRAM:  {s['vram_allocated_gb']:.1f} / {s['vram_total_gb']:.1f} GB")
            lines.append(f"RAM:   {s['ram_used_gb']:.1f} / {s['ram_total_gb']:.1f} GB")
            lines.append(f"GPU:   {s['gpu_name']}")
        return "\n".join(lines)


# Global singleton
_manager: Optional[DeepSeekManager] = None


def get_manager() -> DeepSeekManager:
    """Return the global DeepSeekManager singleton."""
    global _manager
    if _manager is None:
        _manager = DeepSeekManager()
    return _manager
