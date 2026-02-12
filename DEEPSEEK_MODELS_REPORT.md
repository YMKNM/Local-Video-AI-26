# DeepSeek Models on HuggingFace — Comprehensive Research Report

**Organization:** [deepseek-ai](https://huggingface.co/deepseek-ai) | **Total Models:** 83 | **Team Members:** 31  
**Report Date:** February 2026  
**Purpose:** Structured comparison table for local deployment evaluation (16GB VRAM constraint)

---

## Table of Contents

1. [Executive Summary — 16GB VRAM Viability](#1-executive-summary--16gb-vram-viability)
2. [Complete Model Catalog by Family](#2-complete-model-catalog-by-family)
3. [VRAM Estimation Methodology](#3-vram-estimation-methodology)
4. [Framework Compatibility Matrix](#4-framework-compatibility-matrix)
5. [Community Quantization Ecosystem (GGUF)](#5-community-quantization-ecosystem-gguf)
6. [Benchmark Highlights](#6-benchmark-highlights)

---

## 1. Executive Summary — 16GB VRAM Viability

### Models That FIT in 16GB VRAM (native BF16 or with quantization)

| Model | Total Params | VRAM (BF16) | VRAM (Q4) | Fits 16GB? | Best Use |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~3 GB | ~1.5 GB | **YES** | Lightweight reasoning |
| DeepSeek-R1-Distill-Qwen-7B | 7B (8B on HF) | ~16 GB | ~5 GB | **YES (Q4/Q8)** | Strong reasoning |
| DeepSeek-R1-Distill-Llama-8B | 8B | ~16 GB | ~5 GB | **YES (Q4/Q8)** | Reasoning (Llama ecosystem) |
| deepseek-coder-1.3b-instruct | 1.3B (1B on HF) | ~3 GB | ~1 GB | **YES** | Code completion |
| deepseek-coder-6.7b-instruct | 6.7B (7B on HF) | ~14 GB | ~4.5 GB | **YES** | Strong coding |
| deepseek-coder-7b-instruct-v1.5 | 7B | ~14 GB | ~4.5 GB | **YES** | Improved coding |
| deepseek-llm-7b-chat | 7B | ~14 GB | ~4.5 GB | **YES** | General chat |
| deepseek-math-7b-instruct | 7B | ~14 GB | ~4.5 GB | **YES** | Math reasoning |
| DeepSeek-Prover-V2-7B | 7B | ~14 GB | ~4.5 GB | **YES** | Formal theorem proving (Lean 4) |
| DeepSeek-Prover-V1.5-RL | 7B | ~14 GB | ~4.5 GB | **YES** | Formal theorem proving |
| DeepSeek-OCR | 3B | ~6 GB | ~2.5 GB | **YES** | Document OCR |
| DeepSeek-OCR-2 | 3B | ~6 GB | ~2.5 GB | **YES** | Improved OCR |
| Janus-Pro-1B | 1.5B (LLM) | ~4 GB | ~2 GB | **YES** | Multimodal understand+gen |
| JanusFlow-1.3B | 1.3B (LLM, 2B total) | ~4 GB | ~2 GB | **YES** | Multimodal (rectified flow) |
| deepseek-vl-1.3b-base | ~2B | ~4 GB | ~2 GB | **YES** | Vision-language |

### Models That MAYBE Fit (with aggressive quantization)

| Model | Total Params | VRAM (BF16) | VRAM (Q4) | Notes |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B | 14B (15B on HF) | ~30 GB | ~9 GB | **Fits Q4 only** |
| DeepSeek-V2-Lite / Chat | 16B total, 2.4B active | ~32 GB | ~10 GB | MoE — needs full weights in VRAM |
| deepseek-vl2-small | 16B (2.8B active) | ~32 GB | ~10 GB | MoE VL model, tight with Q4 |
| Janus-Pro-7B | 7B (LLM) + vision | ~16 GB | ~5 GB | **Tight at BF16** due to vision encoder overhead |

### Models That DO NOT Fit in 16GB

| Model | Total Params | VRAM (BF16) | Why |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-32B | 32B (33B on HF) | ~66 GB | Too large even at Q4 (~20 GB) |
| DeepSeek-R1-Distill-Llama-70B | 70B (71B on HF) | ~142 GB | Needs multi-GPU |
| deepseek-coder-33b-instruct | 33B | ~66 GB | Too large |
| deepseek-llm-67b-* | 67B | ~134 GB | Too large |
| DeepSeek-V2 / V2-Chat | 236B total, 21B active | ~472 GB | Massive MoE |
| DeepSeek-V3 / V3-Base | 671B (685B on HF) | ~1,370 GB | Multi-node cluster required |
| DeepSeek-V3.2 / variants | 685B on HF | ~1,370 GB | Multi-node cluster required |
| DeepSeek-R1 / R1-Zero | 671B (685B on HF) | ~1,370 GB | Multi-node cluster required |
| DeepSeek-Prover-V2-671B | 671B | ~1,370 GB | Multi-node cluster required |
| DeepSeek-Math-V2 | 685B | ~1,370 GB | Multi-node cluster required |

---

## 2. Complete Model Catalog by Family

### 2A. Flagship Models (V3/R1/V3.2) — MoE Architecture

All flagship models use **DeepSeekMoE + Multi-head Latent Attention (MLA)**. Total params 671B, activated params 37B per token. Context length 128K. NOT supported by HuggingFace Transformers directly — use SGLang (recommended), vLLM, LMDeploy, or TensorRT-LLM.

| Model | HF Params | Activated | Context | License | Tensor Type | Downloads/mo | Type |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 685B | 37B | 128K | DeepSeek Model License | BF16, F8_E4M3, F32 | 1,210K | Base chat model |
| DeepSeek-V3-Base | 685B | 37B | 128K | DeepSeek Model License | BF16 | — | Pre-trained base |
| DeepSeek-V3-0324 | 685B | 37B | 128K | MIT | BF16 | — | March 2024 checkpoint |
| DeepSeek-V3.2 | 685B | 37B | — | MIT | BF16, F8_E4M3, F32 | 301K | Latest flagship (DSA attention) |
| DeepSeek-V3.2-Speciale | 685B | 37B | — | MIT | BF16 | — | Specialized variant |
| DeepSeek-V3.2-Exp | 685B | 37B | — | — | BF16 | — | Experimental |
| DeepSeek-V3.2-Exp-Base | 685B | 37B | — | — | BF16 | 694 | Experimental base |
| DeepSeek-R1 | 685B | 37B | 128K | MIT | BF16, F8_E4M3, F32 | 343K | Reasoning model (o1-class) |
| DeepSeek-R1-Zero | 685B | 37B | 128K | MIT | BF16 | — | RL-only (no SFT) reasoning |
| DeepSeek-Prover-V2-671B | 685B | 37B | — | MIT | BF16 | — | Formal theorem proving |
| DeepSeek-Math-V2 | 685B | 37B | — | — | BF16 | — | Math specialist |

**16GB Feasibility:** NONE — all require multi-node GPU clusters (minimum ~8×80GB GPUs per node, 2+ nodes recommended).

---

### 2B. R1-Distill Models — Dense Reasoning

Distilled from DeepSeek-R1 using 800K curated reasoning samples. All use **dense transformer** architectures (Qwen2 or Llama). Support: transformers, vLLM, SGLang, TGI, llama.cpp (GGUF).

| Model | HF Params | Base Model | Architecture | Context | License | Tensor | DL/mo |
|---|---|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 2B | Qwen2.5-Math-1.5B | qwen2 (dense) | 32K | MIT (Apache 2.0 base) | BF16 | 879K |
| R1-Distill-Qwen-7B | 8B | Qwen2.5-Math-7B | qwen2 (dense) | 32K | MIT (Apache 2.0 base) | BF16 | 715K |
| R1-Distill-Llama-8B | 8B | Llama-3.1-8B | llama (dense) | 32K | MIT (Llama 3.1 base) | BF16 | 444K |
| R1-Distill-Qwen-14B | 15B | Qwen2.5-14B | qwen2 (dense) | 32K | MIT (Apache 2.0 base) | BF16 | 413K |
| R1-Distill-Qwen-32B | 33B | Qwen2.5-32B | qwen2 (dense) | 32K | MIT (Apache 2.0 base) | BF16 | 1,451K |
| R1-Distill-Llama-70B | 71B | Llama-3.3-70B-Instruct | llama (dense) | 32K | MIT (Llama 3.3 base) | BF16 | 302K |

**Disk Size Estimates (BF16 safetensors):**
- 1.5B → ~3 GB | 7B → ~14 GB | 8B → ~16 GB | 14B → ~28 GB | 32B → ~64 GB | 70B → ~140 GB

**Community Quantizations:** 1.5B → 237 | 7B → 172 | 8B → 189 | 14B → 135 | 32B → 144 | 70B → 61

---

### 2C. DeepSeek-V2 Family — MoE

Uses **DeepSeekMoE + MLA** architecture. The "Lite" variant is a smaller model for researchers.

| Model | HF Params | Total/Active | Context | Architecture | License | Tensor | DL/mo |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2 | 236B | 236B / 21B | 128K | deepseek_v2 (MoE + MLA) | DeepSeek Model License | BF16 | — |
| DeepSeek-V2-Chat | 236B | 236B / 21B | 128K | deepseek_v2 (MoE + MLA) | DeepSeek Model License | BF16 | — |
| DeepSeek-V2-Lite | 16B | 15.7B / 2.4B | 32K | deepseek_v2 (MoE + MLA) | DeepSeek Model License | BF16 | 149K |
| DeepSeek-V2-Lite-Chat | 16B | 15.7B / 2.4B | 32K | deepseek_v2 (MoE + MLA) | DeepSeek Model License | BF16 | 248K |

**Architecture detail (V2-Lite):** 27 layers, hidden dim 2048, 16 attention heads, KV compression dim 512, 2 shared + 64 routed experts per MoE layer, 6 experts activated per token.  
**VRAM:** V2-Lite needs 40GB×1 GPU (BF16). V2 full needs multi-GPU setup.  
**16GB Feasibility:** V2-Lite *might* fit with Q4 quantization (~10 GB), but MoE quantization is more complex than dense.

---

### 2D. DeepSeek-Coder Family — Code Models

All trained from scratch on **2T tokens (87% code, 13% natural language)**, using a **LLaMA-based dense architecture** with 16K context window. Available in base + instruct variants.

| Model | HF Params | Context | Architecture | License | Tensor | DL/mo |
|---|---|---|---|---|---|---|
| deepseek-coder-1.3b-base | 1B | 16K | llama (dense) | DeepSeek Model License | BF16 | 165K (instruct) |
| deepseek-coder-1.3b-instruct | 1B | 16K | llama (dense) | DeepSeek Model License | BF16 | 166K |
| deepseek-coder-5.7bmqa-base | ~6B | 16K | llama (dense) | DeepSeek Model License | BF16 | — |
| deepseek-coder-6.7b-base | 7B | 16K | llama (dense) | DeepSeek Model License | BF16 | — |
| deepseek-coder-6.7b-instruct | 7B | 16K | llama (dense) | DeepSeek Model License | BF16 | 96K |
| deepseek-coder-7b-base-v1.5 | 7B | 4K | llama (dense) | DeepSeek Model License | BF16 | — |
| deepseek-coder-7b-instruct-v1.5 | 7B | 4K | llama (dense) | DeepSeek Model License | BF16 | 9K |
| deepseek-coder-33b-base | 33B | 16K | llama (dense) | DeepSeek Model License | BF16 | — |
| deepseek-coder-33b-instruct | 33B | 16K | llama (dense) | DeepSeek Model License | BF16 | 22K |

**Note:** v1.5 models are continue-pretrained from DeepSeek-LLM 7B (not from coder-6.7b). The 5.7b "mqa" variant uses Multi-Query Attention.  
**Community Quantizations:** 1.3b → 17 | 6.7b → 34 | 7b-v1.5 → 11 | 33b → 11

---

### 2E. DeepSeek-LLM Family — General Language Models

Dense LLaMA-based transformers trained on **2T tokens** of English + Chinese data.

| Model | HF Params | Context | Architecture | License | Tensor | DL/mo |
|---|---|---|---|---|---|---|
| deepseek-llm-7b-base | 7B | 4K | llama (dense) | DeepSeek Model License | — | — |
| deepseek-llm-7b-chat | 7B | 4K | llama (dense) | DeepSeek Model License | — | 40K |
| deepseek-llm-67b-base | 67B | 4K | llama (dense) | DeepSeek Model License | — | — |
| deepseek-llm-67b-chat | 67B | 4K | llama (dense) | DeepSeek Model License | — | — |

**Community Quantizations:** 7b → 16 | 67b → likely similar

---

### 2F. DeepSeek-Math Family — Math Specialists

Based on DeepSeek-LLM 7B, further trained on **120B math tokens** from Common Crawl + curated sources.

| Model | HF Params | Context | Architecture | License | DL/mo |
|---|---|---|---|---|---|
| deepseek-math-7b-base | 7B | — | llama (dense) | DeepSeek Model License | — |
| deepseek-math-7b-instruct | 7B | — | llama (dense) | DeepSeek Model License | 4K |
| deepseek-math-7b-rl | 7B | — | llama (dense) | DeepSeek Model License | — |

**Community Quantizations:** 7b-instruct → 19

---

### 2G. DeepSeek-Prover Family — Formal Theorem Proving (Lean 4)

| Model | HF Params | Base | Architecture | License | DL/mo |
|---|---|---|---|---|---|
| DeepSeek-Prover-V1.5-Base | 7B | — | llama (dense) | DeepSeek Model License | — |
| DeepSeek-Prover-V1.5-SFT | 7B | V1.5-Base | llama (dense) | DeepSeek Model License | — |
| DeepSeek-Prover-V1.5-RL | 7B | V1.5-SFT | llama (dense) | DeepSeek Model License | — |
| DeepSeek-Prover-V2-7B | 7B | V1.5-Base (ext. 32K ctx) | llama (dense) | MIT | 33K |
| DeepSeek-Prover-V2-671B | 685B | DeepSeek-V3-Base | MoE (DeepSeekMoE) | MIT | — |

**Benchmarks (V2-7B):** MiniF2F-test 88.9% pass ratio (671B size), 49/658 PutnamBench solved  
**Community Quantizations:** V2-7B → 14

---

### 2H. DeepSeek-VL / VL2 Family — Vision-Language Models

**VL2** uses MoE architecture with separate vision and language encoders. Three sizes with different activated params.

| Model | HF Params | Activated | Vision Encoder | LLM Base | Architecture | License | DL/mo |
|---|---|---|---|---|---|---|---|
| deepseek-vl-1.3b-base | 2B | 2B (dense) | — | DeepSeek-LLM-1.3B | dense | DeepSeek Model License | — |
| deepseek-vl-1.3b-chat | 2B | 2B | — | DeepSeek-LLM-1.3B | dense | DeepSeek Model License | — |
| deepseek-vl-7b-base | — | — | — | DeepSeek-LLM-7B | dense | DeepSeek Model License | — |
| deepseek-vl-7b-chat | — | — | — | DeepSeek-LLM-7B | dense | DeepSeek Model License | — |
| deepseek-vl2-tiny | — | 1.0B | SigLIP | DeepSeekMoE-3B | MoE (deepseek_vl_v2) | DeepSeek Model License | — |
| deepseek-vl2-small | 16B | 2.8B | SigLIP | DeepSeekMoE-16B | MoE (deepseek_vl_v2) | DeepSeek Model License | 113K |
| deepseek-vl2 | — | 4.5B | SigLIP | DeepSeekMoE-27B | MoE (deepseek_vl_v2) | DeepSeek Model License | — |

**Capabilities:** Visual QA, OCR, document/table/chart understanding, visual grounding.

---

### 2I. Janus / JanusFlow Family — Unified Multimodal Understanding + Generation

Autoregressive framework that unifies image understanding and image generation in ONE model. Uses decoupled visual encoding (SigLIP for understanding, tokenizer/SDXL-VAE for generation).

| Model | HF Params | LLM Base | Vision Understanding | Image Generation | License | DL/mo |
|---|---|---|---|---|---|---|
| Janus-1.3B | 2B | DeepSeek-LLM-1.3B | SigLIP-L (384×384) | VQ tokenizer (384×384) | DeepSeek Model License | — |
| Janus-7B | — | DeepSeek-LLM-7B | SigLIP-L | VQ tokenizer | DeepSeek Model License | — |
| Janus-Pro-1B | 1.5B LLM | DeepSeek-LLM-1.5B | SigLIP-L (384×384) | LlamaGen tokenizer (16× downsample) | MIT | 10K |
| Janus-Pro-7B | 7B LLM | DeepSeek-LLM-7B | SigLIP-L (384×384) | LlamaGen tokenizer | MIT | 19K |
| JanusFlow-1.3B | 2B total | DeepSeek-LLM-1.3B | SigLIP-L (384×384) | Rectified flow + SDXL-VAE (384×384) | MIT | 347 |

**Pipeline:** Any-to-Any (text↔image understanding↔image generation)

---

### 2J. DeepSeek-OCR Family — Document OCR

Specialized vision-language models for optical character recognition and document understanding.

| Model | HF Params | Architecture | License | Tensor | DL/mo | Capabilities |
|---|---|---|---|---|---|---|
| DeepSeek-OCR | 3B | deepseek_vl_v2 | MIT | BF16 | 3,016K | Document→Markdown, tables, charts |
| DeepSeek-OCR-2 | 3B | deepseek_vl_v2 | Apache 2.0 | BF16 | 888K | Improved OCR, visual causal flow |

**Support:** transformers (flash-attn), vLLM (officially supported upstream)  
**Modes:** Dynamic resolution up to (0-6)×768×768 + 1×1024×1024  
**16GB Feasibility:** YES — easily fits at ~6 GB BF16

---

### 2K. Other / Specialized Models

| Model | HF Params | Description | License | DL/mo |
|---|---|---|---|---|
| ESFT-vanilla-lite | 16B | Expert Specialized Fine-Tuning variant of V2-Lite | DeepSeek Model License | — |
| DeepSeek-V3.1-Base | 685B | V3.1 base model | — | — |
| DeepSeek-V3.1 | 685B | V3.1 chat model | — | — |

---

## 3. VRAM Estimation Methodology

**Formula:** VRAM ≈ (Number of Parameters × Bytes per Parameter) + KV Cache + Overhead

| Precision | Bytes/Param | 7B Model | 14B Model | 33B Model | 70B Model |
|---|---|---|---|---|---|
| FP32 | 4.0 | ~28 GB | ~56 GB | ~132 GB | ~280 GB |
| BF16/FP16 | 2.0 | ~14 GB | ~28 GB | ~66 GB | ~140 GB |
| INT8 (Q8) | 1.0 | ~7 GB | ~14 GB | ~33 GB | ~70 GB |
| INT4 (Q4/GPTQ/AWQ) | 0.5 | ~4.5 GB* | ~9 GB* | ~20 GB* | ~40 GB* |
| Q4_K_M (GGUF) | ~0.55 | ~5 GB | ~10 GB | ~22 GB | ~44 GB |

*Includes ~10-15% overhead for activations, KV cache, and framework buffers.

**MoE Models Note:** For MoE architectures (V2-Lite, VL2-small, V3), ALL expert weights must reside in VRAM even though only a subset is activated per token. VRAM is determined by **total params**, not activated params.

---

## 4. Framework Compatibility Matrix

| Framework | Flagship (V3/R1/V3.2) | R1-Distill | Coder | LLM/Math | Prover V2-7B | VL2 | Janus/JanusPro | OCR |
|---|---|---|---|---|---|---|---|---|
| **transformers** | NO (V3.2 partial) | YES | YES | YES | YES | YES (custom_code) | YES | YES |
| **vLLM** | YES | YES | — | — | — | — | — | YES |
| **SGLang** (recommended for MoE) | YES (primary) | YES | — | — | — | — | — | — |
| **LMDeploy** | YES | — | — | — | — | — | — | — |
| **TensorRT-LLM** | YES | — | — | — | — | — | — | — |
| **llama.cpp / GGUF** | Community | Community | Community | Community | Community | — | — | — |
| **text-generation-inference** | — | YES | YES | YES | — | — | — | — |

**GGUF/llama.cpp ecosystem:** 3,139 community GGUF quantizations exist for DeepSeek models on HuggingFace. Top providers: MaziyarPanahi, unsloth, lmstudio-community, bartowski.

---

## 5. Community Quantization Ecosystem (GGUF)

For 16GB VRAM users, GGUF quantizations via llama.cpp or Ollama are the most practical path. Key quantization levels:

| Quant Level | Bits/Weight | Quality Loss | 7B Size | 14B Size | 32B Size |
|---|---|---|---|---|---|
| Q8_0 | 8.5 | Minimal | ~7.7 GB | ~15.4 GB | ~35 GB |
| Q6_K | 6.6 | Very low | ~5.9 GB | ~11.8 GB | ~27 GB |
| Q5_K_M | 5.7 | Low | ~5.1 GB | ~10.3 GB | ~23 GB |
| Q4_K_M | 4.8 | Moderate | ~4.4 GB | ~8.7 GB | ~20 GB |
| Q3_K_M | 3.9 | Noticeable | ~3.5 GB | ~7.0 GB | ~16 GB |
| Q2_K | 3.4 | Significant | ~3.0 GB | ~6.0 GB | ~14 GB |
| IQ2_XS | 2.4 | High | ~2.2 GB | ~4.3 GB | ~10 GB |

**Best picks for 16GB VRAM via GGUF:**
- R1-Distill-Qwen-14B at Q4_K_M (~8.7 GB) — excellent reasoning
- R1-Distill-Qwen-32B at Q2_K (~14 GB) or IQ2_XS (~10 GB) — best reasoning at aggressive quant
- R1-Distill-Qwen-7B at Q8_0 (~7.7 GB) — near-lossless quality

---

## 6. Benchmark Highlights

### R1-Distill Models (AIME 2024 / MATH-500 / GPQA Diamond / LiveCodeBench / Codeforces Rating)

| Model | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench | CF Rating |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 28.9 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Llama-8B | 50.4 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Qwen-14B | 69.7 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-70B | 70.0 | 94.5 | 65.2 | 57.5 | 1633 |
| *o1-mini (ref)* | *63.6* | *90.0* | *60.0* | *53.8* | *1820* |

**Key insight:** R1-Distill-Qwen-14B (fits 16GB at Q4) outperforms o1-mini on AIME and MATH-500. R1-Distill-Qwen-7B (easily fits 16GB) achieves 92.8% on MATH-500.

### V2-Lite Chat Benchmarks (vs 7B dense references)

| Benchmark | DeepSeek-V2-Lite-Chat | 7B Dense Ref | 16B MoE Ref |
|---|---|---|---|
| MMLU | 55.7 | 49.7 | 47.2 |
| HumanEval | 57.3 | 45.1 | 45.7 |
| GSM8K | 72.0 | 62.6 | 62.2 |
| MATH | 27.9 | 14.7 | 15.2 |

---

## License Summary

| License | Models |
|---|---|
| **MIT** | DeepSeek-R1, R1-Zero, all R1-Distill, V3-0324, V3.2, V3.2-Speciale, DeepSeek-OCR, Janus-Pro-1B/7B, JanusFlow-1.3B, Prover-V2 |
| **Apache 2.0** | DeepSeek-OCR-2 |
| **DeepSeek Model License** (commercial allowed) | V3, V3-Base, V2, V2-Lite, all Coder, LLM, Math, VL/VL2 |
| **Base model licenses also apply** | R1-Distill-Llama-* (Llama 3.x license), R1-Distill-Qwen-* (Apache 2.0 base) |

All DeepSeek models support **commercial use**.

---

## Quick Reference: Top Recommendations for 16GB VRAM

| Use Case | Recommended Model | Quant | Est. VRAM | Why |
|---|---|---|---|---|
| **Best reasoning** | R1-Distill-Qwen-14B | Q4_K_M GGUF | ~9 GB | Beats o1-mini on math benchmarks |
| **Best reasoning (no quant)** | R1-Distill-Qwen-7B | BF16 | ~16 GB | Near-native quality, strong reasoning |
| **Best coding** | deepseek-coder-6.7b-instruct | Q8 or BF16 | ~14 GB | SOTA open-source code model at this size |
| **Document OCR** | DeepSeek-OCR-2 | BF16 | ~6 GB | Latest OCR, tons of headroom |
| **Math** | deepseek-math-7b-rl | Q8 or BF16 | ~14 GB | Specialized math solver |
| **Formal proofs** | DeepSeek-Prover-V2-7B | BF16 | ~14 GB | Lean 4 theorem proving |
| **Multimodal (understand+generate)** | Janus-Pro-1B | BF16 | ~4 GB | Understand images + generate images |
| **Lightweight general** | R1-Distill-Qwen-1.5B | BF16 | ~3 GB | Fits anywhere, decent reasoning |
| **Aggressive: best possible quality** | R1-Distill-Qwen-32B | IQ2_XS GGUF | ~10 GB | Significant quality loss but still usable |
