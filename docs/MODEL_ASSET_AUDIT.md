# Model & Asset Audit Report

**Date:** 2025-07-24  
**Scope:** All model weights and cached assets in `Local-Video-AI-26`

---

## 1. Model Directory Summary

| Directory | Size | Status | Used By |
|-----------|------|--------|---------|
| `models/wan2.1-t2v-1.3b/` | 26.95 GB | **Active** | `model_registry.py` — default model |
| `models/cogvideox-2b/` | 12.83 GB | **Active** | `model_registry.py` — entry-tier |
| `models/cogvideox-5b/` | 20.05 GB | **Active** | `model_registry.py` — standard-tier |
| `models/ltx-video-2b/` | 26.47 GB | **Active** | `model_registry.py` — LTX v0.9 |
| `models/ltx-2-19b/` | 134.75 GB | **Active** | `model_registry.py` — highest quality |
| `models/text_encoder/` | 6.38 GB | **Legacy** | Old ONNX pipeline — likely orphaned |
| `models/vae/` | 0.62 GB | **Legacy** | Old ONNX pipeline — likely orphaned |
| `models/video_diffusion/` | 236.66 GB | **Redundant** | See analysis below |
| `models/sam_vit_b_01ec64.pth` | 0.36 GB | **Active** | SAM segmentation for image-to-video |
| **Total models/** | **~465 GB** | | |

### Other Cached Assets

| Directory | Size | Status | Contents |
|-----------|------|--------|----------|
| `models_cache/` | 8.01 GB | **Legacy** | AnimateDiff v1.5-3 (1.56 GB), ControlNet OpenPose (1.35 GB), SD 1.5 (5.11 GB) |
| `cache/huggingface/` | 31.31 GB | **Active** | HuggingFace hub cache (model downloads, tokenizer configs) |
| `outputs/` | 265.5 MB | **User data** | Generated video files |
| `test_output/` | 3.9 MB | **Temp** | Test artifacts |
| `test_proof_videos/` | 18.6 MB | **Temp** | Test proof videos |

### Grand Total: ~505 GB

---

## 2. `models/video_diffusion/` Analysis (236.66 GB — Redundant)

This directory contains the **raw Lightricks/LTX-Video HuggingFace repository** downloaded in its entirety, including every version checkpoint ever released:

| File | Size |
|------|------|
| `ltx-video-2b-v0.9.safetensors` | 8.9 GB |
| `ltx-video-2b-v0.9.1.safetensors` | 5.5 GB |
| `ltx-video-2b-v0.9.5.safetensors` | 6.0 GB |
| `ltxv-2b-0.9.6-dev-04-25.safetensors` | 6.0 GB |
| `ltxv-2b-0.9.6-distilled-04-25.safetensors` | 6.0 GB |
| `ltxv-2b-0.9.8-distilled.safetensors` | 6.0 GB |
| `ltxv-2b-0.9.8-distilled-fp8.safetensors` | 4.3 GB |
| `ltxv-13b-0.9.7-dev.safetensors` | 27.3 GB |
| `ltxv-13b-0.9.7-dev-fp8.safetensors` | 15.0 GB |
| `ltxv-13b-0.9.7-distilled.safetensors` | 27.3 GB |
| `ltxv-13b-0.9.7-distilled-fp8.safetensors` | 15.0 GB |
| `ltxv-13b-0.9.7-distilled-lora128.safetensors` | 1.3 GB |
| `ltxv-13b-0.9.8-dev.safetensors` | 27.3 GB |
| `ltxv-13b-0.9.8-dev-fp8.safetensors` | 15.0 GB |
| `ltxv-13b-0.9.8-distilled.safetensors` | 27.3 GB |
| `ltxv-13b-0.9.8-distilled-fp8.safetensors` | 15.0 GB |
| `ltxv-spatial-upscaler-0.9.7.safetensors` | 0.5 GB |
| `ltxv-spatial-upscaler-0.9.8.safetensors` | 0.5 GB |
| `ltxv-temporal-upscaler-0.9.7.safetensors` | 0.5 GB |
| `ltxv-temporal-upscaler-0.9.8.safetensors` | 0.5 GB |
| + subdirectories (scheduler, text_encoder, tokenizer, transformer, vae) | varies |

**None of these files are used by the current pipeline.** The active diffusers-format LTX-Video 2B is at `models/ltx-video-2b/` (26.47 GB), and LTX-2 19B is at `models/ltx-2-19b/` (134.75 GB).

**Recommendation:** Delete `models/video_diffusion/` entirely — reclaims **236.66 GB** of disk space.

---

## 3. `models_cache/` Analysis (8.01 GB — Legacy)

| Model | Size | Status |
|-------|------|--------|
| AnimateDiff Motion Adapter v1.5-3 | 1.56 GB | Not used — no AnimateDiff code path |
| ControlNet v11p SD15 OpenPose | 1.35 GB | Used by `image_motion/` pipeline |
| Stable Diffusion v1.5 | 5.11 GB | Used by `image_motion/` pipeline (AnimateDiff base) |

**Recommendation:** Keep ControlNet + SD 1.5 if image-to-video is active. AnimateDiff adapter (1.56 GB) can potentially be removed if not used.

---

## 4. Legacy ONNX Model Directories

| Directory | Size | Notes |
|-----------|------|-------|
| `models/text_encoder/` | 6.38 GB | Legacy ONNX text encoder — not used by diffusers pipeline |
| `models/vae/` | 0.62 GB | Legacy ONNX VAE — not used by diffusers pipeline |

**Recommendation:** These are from the original ONNX/DirectML architecture. They can be deleted once the ONNX code path is formally deprecated — reclaims **~7 GB**.

---

## 5. Model Compatibility Matrix

| Model | VRAM (est. peak) | RAM Required | Quantization | Status |
|-------|------------------|-------------|--------------|--------|
| Wan2.1 T2V 1.3B | ~7 GB | 16 GB | None (BF16) | Tested ✓ |
| CogVideoX 2B | ~6 GB | 16 GB | None (FP16) | Tested ✓ |
| CogVideoX 5B | ~8 GB | 24 GB | None (BF16) | Tested ✓ |
| LTX-Video 2B | ~7 GB | 28 GB | None (BF16) | Tested ✓ |
| LTX-2 19B | ~5 GB* | 64 GB | INT8 (quanto) | Tested ✓ |

*LTX-2 uses sequential CPU offload — only ~3-5 GB VRAM on GPU at any time, but loads ~46 GB into system RAM.

All models are compatible with the RTX 5080 (16 GB VRAM) + 68.5 GB RAM hardware.

---

## 6. Disk Space Recovery Opportunities

| Action | Space Recovered | Risk |
|--------|----------------|------|
| Delete `models/video_diffusion/` | **236.66 GB** | None — completely redundant with `models/ltx-video-2b/` |
| Delete `models/text_encoder/` | 6.38 GB | Low — only used by dead ONNX path |
| Delete `models/vae/` | 0.62 GB | Low — only used by dead ONNX path |
| Delete AnimateDiff adapter in `models_cache/` | 1.56 GB | Low — verify image-to-video doesn't need it |
| Clear `test_output/` + `test_proof_videos/` | 22.5 MB | None — test artifacts |
| **Total recoverable** | **~245 GB** | |

---

## 7. Recommendations

1. **Immediately delete `models/video_diffusion/`** — 236 GB of unused raw checkpoints. This is by far the largest waste.
2. **Plan deprecation of ONNX models** — `text_encoder/` and `vae/` (7 GB) are dead weight once ONNX code is archived.
3. **Monitor `cache/huggingface/`** — 31 GB of HF cache can grow over time. Consider periodic cleanup with `huggingface-cli cache`.
4. **Keep all 5 registered models** — Each serves a different quality/speed tradeoff and all are tested on current hardware.
5. **Add `.gitignore` entries** — Ensure `models_cache/`, `cache/`, `test_output/`, `test_proof_videos/` are all gitignored (they already are).
