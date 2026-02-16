# Codebase Audit Report

**Date:** 2025-07-24  
**Scope:** Full repository analysis of `Local-Video-AI-26`

---

## 1. Architecture Overview

The application follows a layered architecture:

```
User Interfaces (Gradio UI / CLI / REST API)
        ↓
  Agent Layer (planner.py, prompt_engine.py, retry_logic.py)
        ↓
  Runtime Layer (inference.py → diffusers_pipeline.py → model_registry.py)
        ↓
  Output Layer (video/assembler.py → ffmpeg_wrapper.py)
```

**Core generation flow:**
1. User submits prompt via UI/CLI/API
2. `PromptEngine` expands prompt with model-specific quality tags
3. `GenerationPlanner` selects model, estimates VRAM, validates params
4. `InferenceEngine` loads model via `DiffusersPipeline`
5. Diffusion inference generates video frames
6. `FFmpegWrapper` encodes frames to H.264 MP4
7. `RetryManager` catches OOM and retries with reduced params

---

## 2. Code Statistics

| Package | Files | Est. Lines | Purpose |
|---------|-------|-----------|---------|
| `agent/` | 5 | ~2,800 | Planning, prompts, resources, retry, temporal |
| `runtime/` | 7 | ~3,200 | Model registry, diffusers pipeline, inference, CUDA, ONNX |
| `ui/` | 5 | ~3,500 | Gradio web UI, tabs, logging |
| `video/` | 3 | ~800 | Frame writing, FFmpeg, assembly |
| `generators/` | 3 | ~1,700 | Aggressive image gen, image-to-motion, video models |
| `image_motion/` | 11 | ~5,000 | SAM2-based image animation pipeline |
| `deepseek/` | 1 | ~510 | Offline LLM backend |
| `api/` | 1 | ~500 | FastAPI REST server |
| `sdk/` | 2 | ~400 | Python + JavaScript client SDKs |
| `models/` | 4 | ~600 | Legacy ONNX pipeline modules |
| `configs/` | 4 | ~600 | YAML configuration |
| `examples/` | 3 | ~200 | Usage examples |
| Root scripts | 5 | ~1,100 | run_ui, generate, api, download, test |
| **Total** | **~54** | **~18,000+** | |

---

## 3. Strengths

- **Clean model registry pattern** — `model_registry.py` is well-designed with `ModelSpec` dataclass, VRAM estimation, compatibility checking, and UI integration
- **Robust retry system** — OOM recovery with progressive resolution/frame reduction
- **Lazy imports throughout** — Package-level `__init__.py` and `__getattr__` prevent import-chain crashes
- **Comprehensive prompt expansion** — Model-family-aware prompt engineering with quality tags
- **Hardware-aware planning** — VRAM estimation, disk checks, RAM-gated model selection
- **Graceful degradation** — Image-motion pipeline falls back when optional deps (SAM2, pose) are missing
- **INT8 quantization** — Automatic quanto quantization for LTX-2 19B on systems with <96 GB RAM
- **Device safety** — Monkey-patched group offloading to fix diffusers device-mismatch bugs

---

## 4. Issues Found

### Critical

| Issue | Location | Impact |
|-------|----------|--------|
| **Dual model catalog** | `model_registry.py` (5 models) vs `generators/video_models.py` (6 different models) | Confusion about which catalog is authoritative. `model_registry.py` is the active one. |
| **CORS allows all origins** | `api/server.py` — `allow_origins=["*"]` | Security vulnerability if API is exposed to network |

### High

| Issue | Location | Impact |
|-------|----------|--------|
| **Legacy ONNX pipeline** | `models/`, `runtime/onnx_loader.py`, `runtime/directml_session.py`, `runtime/inference.py` (partial) | Dead code path — all generation now uses `diffusers_pipeline.py`. ~2000 lines of unreachable code. |
| **`video_models.py` stubs** | `generators/video_models.py` — 5 of 6 model classes are stubs | 900 lines with only `LTXVideo2Model` partially implemented |
| **`temporal_prompt.py` unintegrated** | `agent/temporal_prompt.py` — 848 lines | Never imported or called by any other module |

### Medium

| Issue | Location | Impact |
|-------|----------|--------|
| **`gpu_scheduler.py` simulated** | `runtime/gpu_scheduler.py` | Always falls through to simulated execution — never actually schedules real GPU work |
| **`models.yaml` references nonexistent models** | `configs/models.yaml` — HunyuanVideo, Genmo Mochi, AccVideo | Config entries for models that were never implemented |
| **Outdated banner text** | `generate.py` — "AMD GPUs", "DirectML Accelerated" | Cosmetic but misleading |
| **`LivePortraitModel` placeholder** | `generators/video_models.py` | Placeholder class with no implementation |

### Low

| Issue | Location | Impact |
|-------|----------|--------|
| **`hardware.yaml` TensorRT section** | `configs/hardware.yaml` — TensorRT optimization settings | TensorRT is not used; config section is dead |
| **Example scripts reference DirectML** | `examples/directml_demo.py` | Outdated example that won't work |
| **SDK clients untested** | `sdk/python_client.py`, `sdk/javascript/` | No tests or integration verification |

---

## 5. Dead Code Summary

| File/Module | Lines | Status | Recommendation |
|-------------|-------|--------|----------------|
| `runtime/directml_session.py` | ~400 | Dead — DirectML is no longer used | Archive or remove |
| `runtime/onnx_loader.py` | ~300 | Dead — ONNX path superseded by diffusers | Archive or remove |
| `models/pipeline.py` | ~150 | Dead — ONNX pipeline | Archive or remove |
| `models/text_encoder.py` | ~150 | Dead — ONNX text encoder | Archive or remove |
| `models/vae.py` | ~150 | Dead — ONNX VAE | Archive or remove |
| `models/video_diffusion.py` | ~200 | Dead — ONNX diffusion model | Archive or remove |
| `agent/temporal_prompt.py` | ~850 | Unintegrated — never imported | Integrate or archive |
| `examples/directml_demo.py` | ~50 | Dead — DirectML example | Remove |
| `generators/video_models.py` | ~900 | Mostly stubs (5/6 classes) | Consolidate with model_registry or remove stubs |
| **Total dead/stub code** | **~3,150** | | ~17% of codebase |

---

## 6. Performance Observations

- **Model loading** is the primary bottleneck — LTX-2 19B takes 3-5 minutes to load with INT8 quantization
- **Group offloading** (block-level) is correctly configured for RTX 5080's 16 GB VRAM
- **Monkey-patching** of `ModuleGroup._onload_from_memory` works around diffusers bug #10526
- **No model caching between generations** — every generation reloads the full pipeline (high overhead)
- **FFmpeg encoding** is single-threaded via subprocess — adequate for short clips

---

## 7. Security Considerations

| Finding | Severity | Location |
|---------|----------|----------|
| CORS `allow_origins=["*"]` | Medium | `api/server.py` |
| No authentication on REST API | Low | `api/server.py` |
| No input validation on file paths | Low | Multiple locations |
| Model downloads over HTTPS (HuggingFace) | OK | Default behavior |
| All processing is local | Strength | No data leaves the machine |

---

## 8. Recommended Improvements (Priority Order)

1. **Cache loaded models** — Avoid reloading the pipeline between generations. Keep the last-used model in memory.
2. **Remove or archive dead ONNX/DirectML code** — ~3,150 lines of unreachable code adds maintenance burden.
3. **Consolidate model catalogs** — Merge `generators/video_models.py` into `model_registry.py` or remove stubs.
4. **Integrate `temporal_prompt.py`** — 848 lines of temporal prompt scheduling could enable multi-scene videos.
5. **Fix API CORS** — Restrict to `localhost` or configurable origins.
6. **Add model caching/warm start** — Detect if same model is already loaded and skip reload.
7. **Clean up `configs/models.yaml`** — Remove entries for nonexistent models (HunyuanVideo, Genmo Mochi, AccVideo).
8. **Add upper bounds to dependency versions** — Prevent future breaking changes from unbounded `>=` pins.
9. **Add integration tests** — Current tests only verify setup; no end-to-end generation tests.
10. **Fix `generate.py` banner** — Still says "AMD GPUs" and "DirectML Accelerated".
