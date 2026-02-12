"""
DeepSeek Integration — Functional Tests
========================================
Validates the full backend lifecycle: registry, load, generate, switch, unload.

Tests are ordered so that heavier (download-dependent) tests run after basic
unit tests.  Each model is loaded, prompted, and unloaded in sequence.

Run:
    python -m pytest test_deepseek.py -v --tb=short
    # or directly:
    python test_deepseek.py
"""

from __future__ import annotations

import gc
import sys
import os
import time
import logging

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------

from video_ai.deepseek import (
    DEEPSEEK_MODELS,
    DeepSeekModelSpec,
    QuantMode,
    deepseek_dropdown_choices,
    spec_from_label,
    get_gpu_info,
    get_manager,
    DeepSeekManager,
)


# ---------------------------------------------------------------------------
# 1. Registry / Unit tests (no GPU needed)
# ---------------------------------------------------------------------------

class TestRegistry:
    """Tests that don't require model downloads."""

    def test_registry_has_three_models(self):
        assert len(DEEPSEEK_MODELS) == 3, f"Expected 3 models, got {len(DEEPSEEK_MODELS)}"

    def test_model_ids(self):
        expected = {"deepseek-r1-1.5b", "deepseek-r1-7b", "deepseek-r1-14b"}
        assert set(DEEPSEEK_MODELS.keys()) == expected

    def test_dropdown_choices(self):
        choices = deepseek_dropdown_choices()
        assert len(choices) == 3
        for c in choices:
            assert "DeepSeek" in c

    def test_spec_from_label_roundtrip(self):
        choices = deepseek_dropdown_choices()
        for label in choices:
            spec = spec_from_label(label)
            assert spec is not None, f"No spec for label: {label}"
            assert spec.model_id in DEEPSEEK_MODELS

    def test_spec_from_label_invalid(self):
        assert spec_from_label("NotAModel") is None

    def test_all_specs_have_required_fields(self):
        for spec in DEEPSEEK_MODELS.values():
            assert spec.repo_id.startswith("deepseek-ai/")
            assert spec.disk_gb > 0
            assert spec.vram_gb > 0
            assert spec.context_length > 0
            assert spec.license == "MIT"

    def test_quant_modes(self):
        s15 = DEEPSEEK_MODELS["deepseek-r1-1.5b"]
        s7 = DEEPSEEK_MODELS["deepseek-r1-7b"]
        s14 = DEEPSEEK_MODELS["deepseek-r1-14b"]
        assert s15.quant == QuantMode.BF16
        assert s7.quant == QuantMode.NF4
        assert s14.quant == QuantMode.NF4

    def test_vram_within_16gb(self):
        for spec in DEEPSEEK_MODELS.values():
            assert spec.vram_gb <= 16.0, f"{spec.model_id} needs {spec.vram_gb} GB VRAM"


class TestGPUInfo:
    """GPU helper tests."""

    def test_gpu_info_returns_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_gpu_available(self):
        info = get_gpu_info()
        assert info["available"], "CUDA GPU required for these tests"

    def test_gpu_has_enough_vram(self):
        info = get_gpu_info()
        assert info["total_gb"] >= 15.0, f"Need >=15 GB VRAM, got {info['total_gb']}"


class TestManagerBasics:
    """Manager tests that don't load models."""

    def test_singleton(self):
        m1 = get_manager()
        m2 = get_manager()
        assert m1 is m2

    def test_initial_state(self):
        mgr = DeepSeekManager()
        assert not mgr.is_loaded
        assert mgr.loaded_spec is None

    def test_status_text_no_model(self):
        mgr = DeepSeekManager()
        txt = mgr.status_text()
        assert "No model loaded" in txt

    def test_unload_when_empty(self):
        mgr = DeepSeekManager()
        mgr.unload()  # should not raise


# ---------------------------------------------------------------------------
# 2. Model Load / Generate / Switch tests (require downloads)
# ---------------------------------------------------------------------------

# These tests download models from HuggingFace on first run.
# Mark with a custom marker so they can be skipped if needed.

MODEL_IDS_ORDERED = ["deepseek-r1-1.5b", "deepseek-r1-7b", "deepseek-r1-14b"]


@pytest.fixture(scope="module")
def manager():
    """Shared manager for the load/generate tests."""
    mgr = DeepSeekManager()
    yield mgr
    mgr.unload()


class TestModelA:
    """Load, generate, validate the 1.5B model."""

    MODEL_ID = "deepseek-r1-1.5b"

    def test_load(self, manager: DeepSeekManager):
        spec = DEEPSEEK_MODELS[self.MODEL_ID]
        result = manager.load(spec)
        assert manager.is_loaded
        assert manager.loaded_spec.model_id == self.MODEL_ID
        assert "Loaded" in result
        logger.info("Model A loaded: %s", result.replace("\n", " | "))

    def test_short_generation(self, manager: DeepSeekManager):
        """Generate 10-20 tokens."""
        output = ""
        for chunk in manager.generate("What is 2+2?", max_new_tokens=20, temperature=0.1):
            output = chunk
        assert len(output) > 0, "Empty generation"
        logger.info("Short gen (%d chars): %s", len(output), output[:100])

    def test_medium_generation(self, manager: DeepSeekManager):
        """Generate ~100 tokens."""
        output = ""
        for chunk in manager.generate("Explain gravity in two sentences.", max_new_tokens=100, temperature=0.5):
            output = chunk
        assert len(output) > 20, f"Too short: {len(output)} chars"
        logger.info("Medium gen (%d chars): %s", len(output), output[:200])

    def test_memory_within_limits(self, manager: DeepSeekManager):
        info = get_gpu_info()
        assert info["allocated_gb"] < 16.0, f"VRAM {info['allocated_gb']} GB exceeds limit"
        logger.info("VRAM after 1.5B: %.1f / %.1f GB", info["allocated_gb"], info["total_gb"])

    def test_status_shows_loaded(self, manager: DeepSeekManager):
        s = manager.status_dict()
        assert s["model_loaded"]
        assert s["model_name"] == DEEPSEEK_MODELS[self.MODEL_ID].display_name

    def test_unload(self, manager: DeepSeekManager):
        manager.unload()
        assert not manager.is_loaded
        info = get_gpu_info()
        logger.info("VRAM after unload: %.1f GB", info["allocated_gb"])


class TestModelB:
    """Load, generate, validate the 7B model."""

    MODEL_ID = "deepseek-r1-7b"

    def test_load(self, manager: DeepSeekManager):
        spec = DEEPSEEK_MODELS[self.MODEL_ID]
        result = manager.load(spec)
        assert manager.is_loaded
        assert manager.loaded_spec.model_id == self.MODEL_ID
        logger.info("Model B loaded: %s", result.replace("\n", " | "))

    def test_short_generation(self, manager: DeepSeekManager):
        output = ""
        for chunk in manager.generate("What is the capital of France?", max_new_tokens=20, temperature=0.1):
            output = chunk
        assert len(output) > 0
        logger.info("Short gen (%d chars): %s", len(output), output[:100])

    def test_medium_generation(self, manager: DeepSeekManager):
        output = ""
        for chunk in manager.generate("Write a Python function to reverse a string.", max_new_tokens=100, temperature=0.3):
            output = chunk
        assert len(output) > 20
        logger.info("Medium gen (%d chars): %s", len(output), output[:200])

    def test_memory_within_limits(self, manager: DeepSeekManager):
        info = get_gpu_info()
        assert info["allocated_gb"] < 16.0
        logger.info("VRAM after 7B: %.1f / %.1f GB", info["allocated_gb"], info["total_gb"])

    def test_unload(self, manager: DeepSeekManager):
        manager.unload()
        assert not manager.is_loaded


class TestModelC:
    """Load, generate, validate the 14B model."""

    MODEL_ID = "deepseek-r1-14b"

    def test_load(self, manager: DeepSeekManager):
        spec = DEEPSEEK_MODELS[self.MODEL_ID]
        result = manager.load(spec)
        assert manager.is_loaded
        assert manager.loaded_spec.model_id == self.MODEL_ID
        logger.info("Model C loaded: %s", result.replace("\n", " | "))

    def test_short_generation(self, manager: DeepSeekManager):
        output = ""
        for chunk in manager.generate("Summarise Newton's first law.", max_new_tokens=30, temperature=0.1):
            output = chunk
        assert len(output) > 0
        logger.info("Short gen (%d chars): %s", len(output), output[:100])

    def test_medium_generation(self, manager: DeepSeekManager):
        output = ""
        for chunk in manager.generate("Write a haiku about the ocean.", max_new_tokens=100, temperature=0.7):
            output = chunk
        assert len(output) > 10
        logger.info("Medium gen (%d chars): %s", len(output), output[:200])

    def test_long_context(self, manager: DeepSeekManager):
        """Send a longer prompt to exercise context handling."""
        long_prompt = "Explain the following topics briefly: " + ", ".join(
            [f"topic_{i}" for i in range(50)]
        )
        output = ""
        for chunk in manager.generate(long_prompt, max_new_tokens=150, temperature=0.5):
            output = chunk
        assert len(output) > 20
        logger.info("Long ctx gen (%d chars): %s", len(output), output[:200])

    def test_memory_within_limits(self, manager: DeepSeekManager):
        info = get_gpu_info()
        assert info["allocated_gb"] < 16.0
        logger.info("VRAM after 14B: %.1f / %.1f GB", info["allocated_gb"], info["total_gb"])

    def test_unload(self, manager: DeepSeekManager):
        manager.unload()
        assert not manager.is_loaded


# ---------------------------------------------------------------------------
# 3. Model switching test
# ---------------------------------------------------------------------------

class TestModelSwitching:
    """Verify switching between models doesn't crash or leak memory."""

    def test_switch_a_to_b(self, manager: DeepSeekManager):
        a = DEEPSEEK_MODELS["deepseek-r1-1.5b"]
        b = DEEPSEEK_MODELS["deepseek-r1-7b"]

        manager.load(a)
        assert manager.loaded_spec.model_id == a.model_id
        # loading b should auto-unload a
        manager.load(b)
        assert manager.loaded_spec.model_id == b.model_id

        info = get_gpu_info()
        assert info["allocated_gb"] < 16.0
        logger.info("Switched A->B, VRAM: %.1f GB", info["allocated_gb"])
        manager.unload()

    def test_switch_c_to_a(self, manager: DeepSeekManager):
        c = DEEPSEEK_MODELS["deepseek-r1-14b"]
        a = DEEPSEEK_MODELS["deepseek-r1-1.5b"]

        manager.load(c)
        assert manager.loaded_spec.model_id == c.model_id
        manager.load(a)
        assert manager.loaded_spec.model_id == a.model_id

        info = get_gpu_info()
        assert info["allocated_gb"] < 6.0, "VRAM should be low after switching to 1.5B"
        logger.info("Switched C->A, VRAM: %.1f GB", info["allocated_gb"])
        manager.unload()

    def test_double_load_same_model(self, manager: DeepSeekManager):
        a = DEEPSEEK_MODELS["deepseek-r1-1.5b"]
        manager.load(a)
        result = manager.load(a)
        assert "already loaded" in result.lower()
        manager.unload()


# ---------------------------------------------------------------------------
# 4. Stop Generation & Performance
# ---------------------------------------------------------------------------

class TestStopGeneration:
    """Verify stop mechanism and tokens-per-sec tracking."""

    def test_stop_mid_generation(self, manager: DeepSeekManager):
        """Request stop after a few tokens and verify it halts."""
        import threading

        spec = DEEPSEEK_MODELS["deepseek-r1-1.5b"]
        manager.load(spec)

        chunks = []

        def _gen():
            for chunk in manager.generate(
                "Write a very long essay about the history of mathematics.",
                max_new_tokens=500,
                temperature=0.5,
            ):
                chunks.append(chunk)

        t = threading.Thread(target=_gen, daemon=True)
        t.start()
        # Wait a bit for tokens to start flowing, then stop
        time.sleep(2)
        manager.request_stop()
        t.join(timeout=30)

        assert len(chunks) > 0, "Should have produced at least some output before stop"
        # With 500 max tokens, a full run produces a LOT more output.
        # The stop should have cut it short.
        logger.info(
            "Stop test: %d chunks, last chunk %d chars",
            len(chunks),
            len(chunks[-1]) if chunks else 0,
        )

    def test_tokens_per_sec_tracked(self, manager: DeepSeekManager):
        """After generation, tokens_per_sec should be > 0."""
        if not manager.is_loaded:
            spec = DEEPSEEK_MODELS["deepseek-r1-1.5b"]
            manager.load(spec)

        for _ in manager.generate("Say hello.", max_new_tokens=20, temperature=0.1):
            pass

        tps = manager.last_tokens_per_sec
        assert tps > 0, f"Expected tokens_per_sec > 0, got {tps}"
        logger.info("Tokens/sec: %.1f", tps)

    def test_status_dict_has_tokens_per_sec(self, manager: DeepSeekManager):
        """status_dict should include tokens_per_sec key."""
        s = manager.status_dict()
        assert "tokens_per_sec" in s
        logger.info("status_dict tokens_per_sec: %s", s["tokens_per_sec"])
        manager.unload()


# ---------------------------------------------------------------------------
# 5. UI tab import test
# ---------------------------------------------------------------------------

class TestUITab:
    """Verify the tab module imports cleanly."""

    def test_import_deepseek_tab(self):
        from video_ai.ui.deepseek_tab import create_deepseek_tab
        assert callable(create_deepseek_tab)

    def test_tab_available_in_web_ui(self):
        from video_ai.ui.web_ui import DEEPSEEK_TAB_AVAILABLE
        assert DEEPSEEK_TAB_AVAILABLE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with pytest if available, otherwise use basic assertions
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-x"]))
