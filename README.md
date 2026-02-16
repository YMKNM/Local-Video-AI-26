# Local Video AI

**Fully offline, local AI video generation for NVIDIA CUDA GPUs on Windows.**

Generate videos from text prompts using state-of-the-art diffusion models — no cloud, no API keys, no internet required after initial model download. Runs entirely on consumer hardware with an RTX 5080 or similar.

---

## Features

- **Multi-Model Support** — 5 registered models from 1.3B to 19B parameters, selectable per generation
- **Gradio Web UI** — Tabbed interface for video generation, aggressive generation, image-to-video, and DeepSeek chat
- **CLI & Python API** — Generate videos from the command line or integrate into scripts
- **REST API** — FastAPI server with WebSocket support for programmatic access
- **INT8 Quantization** — Automatic quanto INT8 quantization for large models (LTX-2 19B) on systems with < 96 GB RAM
- **Intelligent Prompt Expansion** — Model-family-aware prompt engineering with quality tag injection
- **VRAM-Aware Planning** — Automatic hardware detection, VRAM estimation, and model compatibility checking
- **CPU Offloading** — Block-level group offloading keeps VRAM usage under control
- **Retry Logic** — Automatic OOM recovery with resolution/frame reduction
- **FFmpeg Video Assembly** — H.264 MP4 output with configurable quality
- **DeepSeek LLM Chat** — Offline DeepSeek-R1 inference (1.5B/7B/14B) in a dedicated UI tab
- **Image-to-Video** — SAM2-based image animation pipeline with pose detection and motion estimation

---

## Supported Models

| Model | Parameters | Resolution | FPS | Duration | Disk | Quality |
|-------|-----------|-----------|-----|----------|------|---------|
| **Wan2.1 T2V 1.3B** (default) | 1.3B | 832×480 | 16 | ~2 s | ~27 GB | Standard |
| **CogVideoX 2B** | 2B | 720×480 | 8 | 6 s | ~11 GB | Entry |
| **CogVideoX 5B** | 5B | 720×480 | 8 | 6 s | ~20 GB | Standard |
| **LTX-Video 2B** | 2B | 768×512 | 24 | ~4 s | ~27 GB | Entry |
| **LTX-2 19B** | 19B (47B total) | 768×512 | 24 | ~5 s | ~135 GB | High |

All models are downloaded from HuggingFace in diffusers format and stored locally under `models/`.

LTX-2 19B includes a 27B Gemma3 text encoder and supports text-to-video, image-to-video, and text-to-audio with synchronized output.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA with 8+ GB VRAM, CUDA support | RTX 5080 (16 GB VRAM) |
| **System RAM** | 32 GB | 64+ GB (required for LTX-2 19B with INT8) |
| **Disk** | 50 GB (smallest model) | 500+ GB (all models) |
| **CUDA** | 12.0+ | 12.8 |
| **OS** | Windows 10/11 | Windows 11 |
| **Python** | 3.10+ | 3.11 |
| **FFmpeg** | Required | Add to PATH |

### Current Development Hardware

- NVIDIA GeForce RTX 5080 — 16 GB VRAM, CUDA 12.8, Compute Capability 12.0 (Blackwell)
- 68.5 GB DDR5 RAM
- PyTorch 2.10.0+cu128

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YMKNM/Local-Video-AI-26.git
cd Local-Video-AI-26
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install diffusers from Source

Required for LTX-2 19B support (`LTX2Pipeline` is not yet in a stable release):

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install FFmpeg

Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.

### 7. Download Models

Models are automatically downloaded from HuggingFace on first use. To pre-download:

```bash
python download_models.py
```

Or use the HuggingFace CLI:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir models/wan2.1-t2v-1.3b
```

---

## Usage

### Web UI (Gradio)

```bash
python run_ui.py
```

Opens at `http://localhost:7860`. Options:

```bash
python run_ui.py --port 8080          # Custom port
python run_ui.py --share              # Public Gradio link
python run_ui.py --debug              # Debug logging
```

**UI Tabs:**
1. **Video Generation** — Select model, enter prompt, configure resolution/frames/steps/guidance, generate
2. **Aggressive Generator** — Batch generation with aggressive memory management
3. **Image-to-Video** — Upload an image, animate it using SAM2-based motion pipeline
4. **DeepSeek Chat** — Offline LLM chat using DeepSeek-R1-Distill models

### Command Line

```bash
python generate.py --prompt "A cinematic drone shot over mountains at sunset"
python generate.py --prompt "A cat playing" --seconds 4
python generate.py --prompt "Ocean waves" --width 1280 --height 720 --seed 42
```

### Python API

```python
from video_ai import VideoAI

ai = VideoAI()
result = ai.generate("A sunset over the ocean")
print(result.output_path)
```

### REST API (FastAPI)

```bash
uvicorn video_ai.api.server:app --host 0.0.0.0 --port 8000
```

```python
import httpx

response = httpx.post("http://localhost:8000/generate", json={
    "prompt": "A golden retriever running on a beach",
    "model": "wan2.1-t2v-1.3b"
})
```

---

## Configuration

Configuration files are in `video_ai/configs/`:

| File | Purpose |
|------|---------|
| `defaults.yaml` | Default generation parameters (steps, guidance, resolution) |
| `hardware.yaml` | GPU/RAM detection settings, VRAM thresholds, offloading strategy |
| `models.yaml` | Legacy ONNX model paths (superseded by `model_registry.py`) |
| `prompt_templates.yaml` | Prompt expansion templates and quality tags |

Key runtime settings are in `defaults.yaml`:

```yaml
generation:
  steps: 30
  guidance_scale: 5.0
  width: 832
  height: 480
  fps: 24
  duration_seconds: 6
```

---

## Project Structure

```
Local-Video-AI-26/
├── run_ui.py                  # Web UI launcher
├── generate.py                # CLI entry point
├── api.py                     # Python API wrapper
├── download_models.py         # Model downloader
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installer
│
├── video_ai/                  # Main package
│   ├── __init__.py            # VideoAI class, lazy imports
│   │
│   ├── agent/                 # Planning & orchestration
│   │   ├── planner.py         # GenerationPlanner — central orchestrator
│   │   ├── prompt_engine.py   # Model-aware prompt expansion
│   │   ├── resource_monitor.py# GPU/RAM/disk monitoring
│   │   ├── retry_logic.py     # OOM recovery with parameter reduction
│   │   └── temporal_prompt.py # Temporal prompt scheduling (experimental)
│   │
│   ├── runtime/               # Model loading & inference
│   │   ├── model_registry.py  # Canonical model catalog (5 models)
│   │   ├── diffusers_pipeline.py # HuggingFace Diffusers pipeline wrapper
│   │   ├── inference.py       # Inference engine (bridges planner → pipeline)
│   │   ├── cuda_session.py    # CUDA session management
│   │   └── gpu_scheduler.py   # Multi-job GPU scheduling
│   │
│   ├── ui/                    # Web interface
│   │   ├── web_ui.py          # Gradio UI (4 tabs)
│   │   ├── deepseek_tab.py    # DeepSeek chat tab
│   │   ├── image_motion_tab.py# Image-to-video tab
│   │   ├── aggressive_generator_tab.py # Batch generation tab
│   │   └── log_handler.py     # UI logging integration
│   │
│   ├── video/                 # Video output pipeline
│   │   ├── assembler.py       # Frame → video assembly
│   │   ├── ffmpeg_wrapper.py  # FFmpeg process management
│   │   └── frame_writer.py    # Frame I/O
│   │
│   ├── generators/            # Specialized generators
│   │   ├── aggressive_image.py# Memory-aggressive image generation
│   │   ├── image_to_motion.py # Image animation generator
│   │   └── video_models.py    # Extended model definitions
│   │
│   ├── image_motion/          # SAM2-based image animation
│   │   ├── animator.py        # Core animation engine
│   │   ├── sam2_segment.py    # SAM2 segmentation
│   │   ├── motion_estimator.py# Optical flow & motion
│   │   ├── pose_detector.py   # Pose estimation
│   │   └── ...                # Supporting modules
│   │
│   ├── deepseek/              # Offline DeepSeek LLM
│   │   └── __init__.py        # DeepSeek-R1-Distill (1.5B/7B/14B)
│   │
│   ├── api/                   # REST API
│   │   └── server.py          # FastAPI application
│   │
│   ├── sdk/                   # Client SDKs
│   │   ├── python_client.py   # Python SDK
│   │   └── javascript/        # JavaScript SDK
│   │
│   ├── models/                # Legacy ONNX pipeline modules
│   │
│   ├── configs/               # YAML configuration
│   │   ├── defaults.yaml
│   │   ├── hardware.yaml
│   │   ├── models.yaml
│   │   └── prompt_templates.yaml
│   │
│   └── examples/              # Usage examples
│       ├── basic_generation.py
│       ├── advanced_generation.py
│       └── directml_demo.py
│
├── models/                    # Downloaded model weights (not in git)
│   ├── wan2.1-t2v-1.3b/      # ~27 GB
│   ├── cogvideox-2b/          # ~13 GB
│   ├── cogvideox-5b/          # ~20 GB
│   ├── ltx-video-2b/          # ~27 GB
│   ├── ltx-2-19b/             # ~135 GB
│   └── ...
│
├── outputs/                   # Generated videos (not in git)
├── docs/                      # Documentation
└── deploy/                    # Deployment configs
```

---

## How It Works

1. **Prompt** → `PromptEngine` expands short prompts with model-specific quality tags and cinematic descriptors
2. **Planning** → `GenerationPlanner` selects model, estimates VRAM, snaps resolution/frames to model constraints
3. **Loading** → `DiffusersPipeline` loads the HuggingFace pipeline with CPU offloading and optional INT8 quantization
4. **Inference** → Diffusion model generates frames (latent space → pixel space via VAE)
5. **Assembly** → `FFmpegWrapper` encodes frames to H.264 MP4
6. **Retry** → If OOM occurs, `RetryManager` reduces resolution/frames and retries automatically

---

## Troubleshooting

### Out of Memory (OOM)

The retry system automatically reduces resolution and frame count on OOM. To reduce VRAM usage manually:

- Use a smaller model (Wan2.1 1.3B or CogVideoX 2B)
- Reduce resolution (e.g., 512×320)
- Reduce frame count
- Ensure no other GPU-heavy applications are running

### Model Download Issues

Models are large (11–135 GB). If downloads fail:

- Check disk space (`models/` can exceed 400 GB with all models)
- Use `huggingface-cli download` with `--resume-download` for resumable downloads
- Set `HF_HOME` environment variable to control cache location

### CUDA / PyTorch Issues

- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- RTX 5080 requires PyTorch with CUDA 12.8+ (`cu128`)
- Install from: `https://download.pytorch.org/whl/cu128`

### diffusers Version

LTX-2 19B requires diffusers installed from source (the `LTX2Pipeline` class). If you get import errors:

```bash
pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

### FFmpeg Not Found

Ensure `ffmpeg` is on your system PATH:

```bash
ffmpeg -version
```

If not installed, download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` directory to your PATH.

---

## Development

### Running Tests

```bash
python -m pytest test_setup.py -v
```

### Project Dependencies

Core stack:
- **PyTorch 2.10+** (CUDA 12.8) — tensor computation and GPU inference
- **diffusers** (from source) — HuggingFace diffusion model pipelines
- **transformers** — text encoder models (T5, Gemma3)
- **optimum-quanto** — INT8 weight quantization
- **accelerate** — model loading and device management
- **Gradio** — web UI framework
- **FastAPI** — REST API server
- **FFmpeg** — video encoding

---

## Known Limitations

- Windows-only (tested on Windows 11)
- NVIDIA GPUs only (CUDA required — the original DirectML/AMD path is no longer active)
- LTX-2 19B requires 64+ GB system RAM for INT8 quantization
- No TensorRT acceleration (planned)
- Single-GPU only
- Video length is model-dependent (typically 2–6 seconds per generation)

---

## License

This project is for personal/research use. Individual models have their own licenses:

| Model | License |
|-------|---------|
| Wan2.1 T2V 1.3B | Apache 2.0 |
| CogVideoX 2B | Apache 2.0 |
| CogVideoX 5B | CogVideoX (custom, research-OK) |
| LTX-Video 2B | LTX-Video Open Weights |
| LTX-2 19B | LTX-2 Community License |
| DeepSeek-R1 | DeepSeek License |
