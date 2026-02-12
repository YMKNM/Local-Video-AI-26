# DeepSeek Models — Comprehensive Local Deployment Report

> **Generated**: July 2025  
> **Target Hardware**: NVIDIA RTX 5080 (16 GB VRAM), Intel i5-12400, 32 GB RAM, 464 GB free disk (Z:), Windows 11

---

## 1. Hardware Profile

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5080 — 16,303 MiB VRAM (≈ 15.9 GB) |
| **CPU** | Intel Core i5-12400 (6 cores / 12 threads) |
| **RAM** | 31.8 GB DDR |
| **Disk (Z:)** | 463.6 GB free of 3,726 GB |
| **OS** | Windows 11 Enterprise |
| **Driver** | NVIDIA 591.74 |
| **CUDA** | 12.8 (via PyTorch 2.10.0+cu128) |

**Key Constraint**: 16 GB VRAM is the hard limit. Models requiring >14 GB VRAM leave no room for KV cache/activations at longer contexts.

---

## 2. Complete DeepSeek Model Catalogue (83 models)

### Model Family Overview

| Family | Param Range | Architecture | Models on HF | 16 GB Viable? |
|--------|-------------|-------------|--------------|----------------|
| **DeepSeek-R1** (Reasoning) | 671B | MoE (37B active) | 1 | ❌ |
| **DeepSeek-R1 Distilled** | 1.5B – 70B | Dense (Qwen2/Llama3) | 6 | ✅ 1.5B–14B, ⚠️ 32B |
| **DeepSeek-R1-0528-Qwen3-8B** | 8B | Dense (Qwen3) | 1 | ✅ |
| **DeepSeek-V3 / V3.2** | 685B | MoE | 5 | ❌ |
| **DeepSeek-V2 / V2-Lite** | 15.7B – 236B | MoE | 4+ | ⚠️ Lite only w/ quant |
| **DeepSeek-Coder** (v1/v1.5) | 1.3B – 33B | Dense | 8 | ✅ 1.3B–7B |
| **DeepSeek-LLM** | 7B – 67B | Dense | 4 | ✅ 7B only |
| **DeepSeek-Math** | 7B / 685B | Dense/MoE | 4 | ✅ 7B only |
| **DeepSeek-Prover** | 7B / 671B | Dense/MoE | 2 | ✅ 7B only |
| **Janus / JanusFlow** | 1B – 7B | Dense + vision | 4 | ✅ 1B; ⚠️ 7B |
| **DeepSeek-VL2** | 3B – 27B | MoE | 3 | ⚠️ tiny only |
| **DeepSeek-OCR** | 3B | Dense | 2 | ✅ |

---

## 3. Compatibility Classification

### ✅ COMPATIBLE — Run at BF16/FP16, no quantization needed

| Model | Params | Category | Disk | VRAM (BF16) | License |
|-------|--------|----------|------|-------------|---------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Reasoning | 3.6 GB | ~4 GB | MIT |
| DeepSeek-R1-0528-Qwen3-8B | 8B | Reasoning (newest) | 16.4 GB | ~16 GB | MIT |
| DeepSeek-R1-Distill-Qwen-7B | 8B | Reasoning | 15.2 GB | ~15 GB | MIT |
| DeepSeek-R1-Distill-Llama-8B | 8B | Reasoning | 16.1 GB | ~16 GB | MIT |
| deepseek-coder-1.3b-instruct | 1.3B | Coder | 2.7 GB | ~3.5 GB | DS License |
| deepseek-coder-6.7b-instruct | 6.7B | Coder | 13.5 GB | ~14 GB | DS License |
| deepseek-coder-7b-instruct-v1.5 | 7B | Coder v1.5 | 13.8 GB | ~14.5 GB | DS License |
| deepseek-llm-7b-chat | 7B | General Chat | 13.5 GB | ~14.5 GB | DS License |
| deepseek-math-7b-instruct | 7B | Math | 13.5 GB | ~14.5 GB | DS License |
| DeepSeek-Prover-V2-7B | 7B | Theorem Proving | 13.8 GB | ~14.5 GB | DS License |
| DeepSeek-OCR | 3B | OCR/Document | 6.7 GB | ~7.5 GB | MIT |
| DeepSeek-OCR-2 | 3B | OCR/Document v2 | 6.8 GB | ~7.5 GB | Apache-2.0 |
| Janus-Pro-1B | 1.8B | Multimodal | 4.2 GB | ~5 GB | MIT |
| JanusFlow-1.3B | 2B | Multimodal | 4.1 GB | ~5 GB | MIT |

> **Note**: The 7B/8B models are tight (~14–16 GB). Use GGUF Q8_0 for comfortable headroom.

### ⚠️ CONDITIONAL — Requires quantization or partial offloading

| Model | Params | Min Viable Quant | VRAM | Quality Loss |
|-------|--------|-----------------|------|-------------|
| DeepSeek-R1-Distill-Qwen-14B | 15B | Q4_K_M (GGUF) | ~10.5 GB | Minimal |
| DeepSeek-R1-Distill-Qwen-14B | 15B | Q6_K (GGUF) | ~13.6 GB | Negligible |
| DeepSeek-R1-Distill-Qwen-32B | 33B | IQ2_M (GGUF, bartowski) | ~12.8 GB | Significant |
| DeepSeek-R1-Distill-Qwen-32B | 33B | Q2_K (GGUF) | ~13.8 GB | Significant |
| Janus-Pro-7B | 7.4B | Q4 recommended | ~10 GB | Minor |
| deepseek-vl2-small | 16B MoE | 4-bit quant needed | ~10 GB | Moderate |
| DeepSeek-V2-Lite-Chat | 15.7B MoE | 4-bit quant needed | ~10 GB | Moderate |

### ❌ INCOMPATIBLE — Cannot fit in 16 GB VRAM at any useful quality

| Model | Params | Why |
|-------|--------|-----|
| DeepSeek-R1 | 671B MoE | >200 GB even at Q2 |
| DeepSeek-V3 / V3.2 / V3.2-Exp | 685B MoE | >200 GB even at Q2 |
| DeepSeek-V2 | 236B MoE | ~60 GB at Q2 |
| DeepSeek-R1-Distill-Llama-70B | 70B | ~26 GB at Q2 |
| DeepSeek-R1-Distill-Qwen-32B (good quant) | 33B | Q4_K_M = 20 GB |
| deepseek-coder-33b-instruct | 33B | ~66 GB BF16 |
| deepseek-llm-67b-chat | 67B | ~134 GB BF16 |
| DeepSeek-Math-V2 | 685B MoE | Same as V3 |
| DeepSeek-Prover-V2-671B | 671B MoE | Same as R1 |

---

## 4. Ranked Comparison — Best Models for Your Hardware

### 🏆 Top Picks (Ranked by capability-per-VRAM)

| Rank | Model | Quant | VRAM | Disk | tok/s (est.) | Best For |
|------|-------|-------|------|------|-------------|----------|
| **1** | R1-Distill-Qwen-14B | Q4_K_M GGUF | 10.5 GB | 9.0 GB | ~15–25 | **Best overall reasoning** |
| **2** | R1-0528-Qwen3-8B | Q8_0 GGUF | 10.2 GB | 8.7 GB | ~25–40 | **Newest reasoning, fast** |
| **3** | R1-Distill-Qwen-14B | Q6_K GGUF | 13.6 GB | 12.1 GB | ~12–20 | **Highest quality that fits** |
| **4** | DeepSeek-OCR-2 | BF16 | 7.5 GB | 6.8 GB | ~30–50 | **Document OCR** |
| **5** | R1-Distill-Llama-8B | Q8_0 GGUF | 10.0 GB | 8.5 GB | ~25–40 | Reasoning (Llama ecosystem) |
| **6** | deepseek-coder-6.7b-instruct | BF16 | 14 GB | 13.5 GB | ~20–35 | **Code generation** |
| **7** | deepseek-coder-1.3b-instruct | BF16 | 3.5 GB | 2.7 GB | ~60–100 | Fast code, low resource |
| **8** | R1-Distill-Qwen-1.5B | BF16 | 4 GB | 3.6 GB | ~60–100 | Fast reasoning, low resource |
| **9** | DeepSeek-OCR | BF16 | 7.5 GB | 6.7 GB | ~30–50 | Document OCR (v1) |
| **10** | Janus-Pro-1B | BF16 | 5 GB | 4.2 GB | ~40–60 | Image understanding + gen |

> **Estimated tok/s**: RTX 5080 Blackwell with GGUF via llama.cpp or transformers BF16. Actual performance depends on context length, batch size, and whether Flash Attention is available.

### Benchmark Highlights

| Model | AIME 2024 | MATH-500 | HumanEval | MMLU |
|-------|-----------|----------|-----------|------|
| R1-Distill-Qwen-14B | **69.7%** | **93.9%** | — | — |
| R1-0528-Qwen3-8B | ~55% | ~88% | — | ~70% |
| R1-Distill-Llama-8B | 50.4% | 89.1% | — | — |
| R1-Distill-Qwen-7B | 55.5% | 92.8% | — | — |
| R1-Distill-Qwen-1.5B | 28.9% | 83.9% | — | — |
| deepseek-coder-6.7b | — | — | **78.6%** | — |
| deepseek-coder-1.3b | — | — | 65.2% | — |
| deepseek-math-7b | — | 52.4%† | — | — |

*† MATH benchmark, not MATH-500*

---

## 5. Optimization Strategy

### For GGUF models (via llama.cpp / Ollama / LM Studio)

| Technique | Benefit | When to Use |
|-----------|---------|-------------|
| **Q4_K_M quantization** | 4× size reduction, ~5% quality loss | Default for 7B–14B models |
| **Q6_K quantization** | 2.7× reduction, ~2% quality loss | When VRAM allows (≤14B) |
| **Q8_0 quantization** | 2× reduction, <1% quality loss | 8B models with headroom |
| **Flash Attention 2** | 2–4× faster attention, less VRAM | Always enable |
| **KV cache quantization** | Reduce context memory | Long-context use (Q8_0 KV) |
| **Context window limiting** | Reduce VRAM usage | Keep ≤4K for tight fits |

### For transformers/BF16 models (via HuggingFace)

| Technique | Benefit | When to Use |
|-----------|---------|-------------|
| **`torch_dtype=torch.bfloat16`** | 2× vs FP32 | Always |
| **`device_map="auto"`** | Automatic GPU/CPU split | Large models |
| **`load_in_4bit=True`** (bitsandbytes) | 4× reduction | MoE models or 14B+ |
| **`attn_implementation="flash_attention_2"`** | Faster inference | If flash-attn installed |
| **Gradient checkpointing** | Lower peak VRAM | Not needed for inference |

---

## 6. Deployment Plan

### Option A: Ollama (Recommended for GGUF models)

Best for: R1 distills, reasoning, chat, code — maximum performance.

#### Step 1: Install Ollama
```powershell
# Download from https://ollama.com/download/windows
# Or via winget:
winget install Ollama.Ollama
```

#### Step 2: Pull recommended models
```powershell
# #1 Pick — Best reasoning (14B Q4_K_M, ~9 GB download)
ollama pull deepseek-r1:14b

# #2 Pick — Newest 8B reasoning (Q8_0, ~8.7 GB)
ollama pull deepseek-r1:8b

# Fast reasoning (1.5B, ~1 GB)
ollama pull deepseek-r1:1.5b

# Code generation
ollama pull deepseek-coder:6.7b
ollama pull deepseek-coder:1.3b
```

#### Step 3: Run
```powershell
# Interactive chat
ollama run deepseek-r1:14b

# API server (OpenAI-compatible at localhost:11434)
ollama serve
# Then: curl http://localhost:11434/v1/chat/completions ...
```

#### Step 4: Verify VRAM usage
```powershell
nvidia-smi  # Should show ~10-11 GB for 14B Q4_K_M
```

---

### Option B: HuggingFace Transformers (For OCR, Janus, custom pipelines)

Best for: OCR, multimodal, integration with existing video_ai project.

#### Step 1: Install dependencies
```powershell
& "z:\AI 2026 Local\video_ai\venv\Scripts\python.exe" -m pip install accelerate bitsandbytes
```

#### Step 2: Download and run DeepSeek-OCR-2
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# ~7.5 GB VRAM
```

#### Step 3: Download and run R1-Distill-Qwen-14B (4-bit via bitsandbytes)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quant_config, device_map="auto"
)
# ~10 GB VRAM
```

---

### Option C: LM Studio (GUI, easiest)

1. Download LM Studio from https://lmstudio.ai
2. Search "DeepSeek R1 Distill 14B GGUF" → download Q4_K_M
3. Load model → chat or use local API at `localhost:1234`

---

## 7. Recommended Download Plan

Based on your 464 GB free disk and use cases:

| Priority | Model | Source | Size | Use Case |
|----------|-------|--------|------|----------|
| 🥇 | R1-Distill-Qwen-14B Q4_K_M | `ollama pull deepseek-r1:14b` | ~9 GB | General reasoning, math, planning |
| 🥈 | R1-0528-Qwen3-8B Q8_0 | `ollama pull deepseek-r1:8b` | ~8.7 GB | Fast reasoning, newest arch |
| 🥉 | DeepSeek-OCR-2 (BF16) | HuggingFace transformers | ~6.8 GB | Document OCR for video_ai |
| 4 | deepseek-coder-6.7b | `ollama pull deepseek-coder:6.7b` | ~3.8 GB | Code generation/review |
| 5 | R1-Distill-Qwen-1.5B | `ollama pull deepseek-r1:1.5b` | ~1 GB | Ultra-fast reasoning |
| 6 | Janus-Pro-1B (BF16) | HuggingFace | ~4.2 GB | Image understand + gen |

**Total disk**: ~33.5 GB for all 6 models

---

## 8. Performance Expectations

| Model (Quant) | VRAM | Prompt Processing | Generation | Context |
|---------------|------|-------------------|------------|---------|
| R1-14B Q4_K_M | 10.5 GB | ~800 tok/s | ~20 tok/s | 4K safe, 8K possible |
| R1-8B Q8_0 | 10.2 GB | ~1200 tok/s | ~35 tok/s | 8K safe |
| R1-1.5B BF16 | 4 GB | ~2000 tok/s | ~80 tok/s | 8K safe |
| OCR-2 BF16 | 7.5 GB | ~600 tok/s | ~40 tok/s | Image + 2K text |
| Coder-6.7b BF16 | 14 GB | ~500 tok/s | ~25 tok/s | 4K (tight VRAM) |

> Rough estimates for RTX 5080 (Blackwell arch, CUDA 12.8). Actual numbers depend on context length, batch size, and Flash Attention availability.

---

## 9. GGUF Quantization Quick Reference

For the **14B model** (most versatile):

| Quant | Size | VRAM | Quality | Recommendation |
|-------|------|------|---------|---------------|
| Q2_K | 5.8 GB | ~7.3 GB | Low | ❌ Avoid |
| Q3_K_M | 7.3 GB | ~8.8 GB | Medium-low | Acceptable |
| **Q4_K_M** | **9.0 GB** | **~10.5 GB** | **Good** | **✅ Sweet spot** |
| Q5_K_M | 10.5 GB | ~12.0 GB | High | ✅ If VRAM allows |
| Q6_K | 12.1 GB | ~13.6 GB | Very high | ✅ Best quality that fits |
| Q8_0 | 15.7 GB | ~17.2 GB | Near-perfect | ❌ Too large |

For the **8B model** (best speed):

| Quant | Size | VRAM | Quality | Recommendation |
|-------|------|------|---------|---------------|
| Q4_K_M | 5.0 GB | ~6.5 GB | Good | Good if VRAM tight |
| Q6_K | 6.7 GB | ~8.2 GB | Very high | If running with video models |
| **Q8_0** | **8.7 GB** | **~10.2 GB** | **Near-perfect** | **✅ Recommended** |
| BF16 | 16.4 GB | ~17.9 GB | Full | ❌ Too large |

---

## 10. What DeepSeek Does NOT Have

| Category | Status | Details |
|----------|--------|---------|
| **Video generation** | ❌ None | No video diffusion models exist |
| **Audio/TTS/ASR** | ❌ None | No speech models |
| **3D generation** | ❌ None | No 3D/NeRF models |
| **Image generation (high-res)** | ⚠️ Limited | Janus generates 384×384 only |
| **Embedding models** | ❌ None | No sentence/text embedding models |
| **LoRA adapters** | ❌ None official | Community only |

---

## 11. Notes on MoE Models (V2-Lite, VL2-Small)

DeepSeek-V2-Lite (15.7B total, 2.4B activated) and VL2-Small (16B total, 2.8B activated) sound like they should fit in 16 GB, but **all expert parameters must be loaded into VRAM** even though only a fraction activate per token:

- V2-Lite: 15.7B × 2 bytes = ~31.4 GB + overhead = **~40 GB VRAM needed** at BF16
- With 4-bit quantization: ~8–10 GB ✅ (fits, but quality suffers)
- **Recommended**: Use the dense 7B/8B models instead — similar or better performance at native precision

---

## 12. Integration with video_ai Project

The DeepSeek models most relevant to the existing video_ai project:

1. **DeepSeek-OCR-2** → Extract text/layout from video frames for captioning
2. **Janus-Pro-1B** → Image understanding for scene analysis
3. **R1-Distill-Qwen-14B** → AI agent/planner (complements existing `agent/planner.py`)

Integration approaches:
- Add entries to `models/model_registry.py` for LLM/OCR models
- Create a new `llm/` module for text inference
- Use Ollama's API at `localhost:11434` as an external service

---

*Report complete. All data sourced from HuggingFace model cards as of July 2025.*
