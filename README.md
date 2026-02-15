# 🎬 Video AI Platform

**Enterprise-Ready AI Video Generation for NVIDIA GPUs**

A production-grade, scalable AI video generation platform optimized for NVIDIA RTX GPUs using CUDA and TensorRT acceleration. Features REST/WebSocket APIs, Docker/Kubernetes deployment, and comprehensive SDKs.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ Features

### Core Capabilities
- 🎬 **Text-to-Video Generation**: Generate high-quality videos up to 4K@60fps
- 🖼️ **Image-to-Video**: Animate still images with motion control
- 🎵 **Audio Synchronization**: Integrated AudioLDM2/MusicGen support
- 🔄 **Long-form Video**: 10+ second clips with temporal consistency

### State-of-the-Art Models
- 📦 **LTX-Video-2**: 5B parameter, 4K support, 257 frames
- 🎭 **HunyuanVideo**: 13B parameter flagship model
- 🔥 **Genmo Mochi**: 10B parameter with text-motion alignment
- ⚡ **AccVideo**: 1.5B fast preview (8-step inference)

### Enterprise Features
- 🌐 **REST/WebSocket API**: FastAPI-based microservices architecture
- 📦 **Python/JavaScript SDKs**: Full automation support
- 🐳 **Docker/Kubernetes**: Production deployment templates
- 📊 **GPU Scheduler**: Priority queue with dynamic quantization
- 🛡️ **Safety Pipeline**: Content filtering, bias detection, watermarking
- 📋 **Audit Logging**: Compliance-ready provenance tracking

### Performance
- 🔥 **CUDA/TensorRT Optimized**: Full hardware acceleration
- 💾 **Smart Memory Management**: Tiled processing for 4K
- ⚡ **Dynamic Quantization**: FP16/BF16/INT8/NVFP8 support
- 🔄 **Progressive Preview**: Fast preview during generation

---

## 📋 System Requirements

### Hardware
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 3060 (12GB) | RTX 3080 (10GB) | RTX 4090 (24GB) |
| **CPU** | 6-core (i5-10400) | 6-core (i5-12400) | 8+ core |
| **RAM** | 16 GB | 64 GB DDR5 | 128 GB |
| **Storage** | 100 GB SSD | 500 GB NVMe | 1 TB NVMe |

### Software
- **OS**: Windows 11/10 or Ubuntu 22.04+
- **Drivers**: NVIDIA Driver 525+ with CUDA 11.8+
- **Python**: 3.10 or later
- **FFmpeg**: Required for video encoding
- **Docker**: Optional, for containerized deployment

---

## 🚀 Quick Start

### Step 1: Clone and Setup Environment

```powershell
# Clone the repository
git clone https://github.com/your-org/video-ai.git
cd video_ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Install FFmpeg

**Windows:**
```powershell
# Using winget
winget install FFmpeg

# Or download from https://github.com/BtbN/FFmpeg-Builds/releases
# Extract to C:\ffmpeg and add to PATH
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Step 3: Download Models

```powershell
# Download models (~50GB for all)
python download_models.py --all

# Or download specific models
python download_models.py --model ltx-video
```

### Step 4: Verify Installation

```powershell
# Run the test suite
python test_setup.py

# Check system info
python generate.py --info
```

### Step 5: Launch the Web UI 🌐

```powershell
# Start the web interface
python run_ui.py
```

The UI will open automatically at **http://localhost:7860**

---

## 🖥️ Using the Web UI

### Launching the UI

```powershell
# Basic launch
python run_ui.py

# Custom port
python run_ui.py --port 8080

# Create public shareable link
python run_ui.py --share

# Enable debug logging
python run_ui.py --debug
```

### UI Features

| Tab | Description |
|-----|-------------|
| **🎥 Generate** | Enter prompts, select models, adjust settings, generate videos |
| **📋 Logs** | View real-time generation logs for troubleshooting |
| **💻 System** | Check GPU, RAM, and dependency status |
| **🔧 Troubleshooting** | Automatic issue detection with solutions |
| **❓ Help** | Usage guide and tips |

### Generation Workflow

1. **Enter Prompt**: Describe the video you want (be specific!)
2. **Select Model**: Choose from available models
3. **Choose Quality**: Fast (quick), Balanced (default), Quality (best)
4. **Adjust Settings** (optional): Duration, resolution, FPS, seed
5. **Click Generate**: Watch progress in real-time
6. **View Result**: Video plays automatically when complete

### Writing Good Prompts

| ❌ Bad | ✅ Good |
|--------|---------|
| "A dog" | "A golden retriever running through a sunny meadow, slow motion, cinematic" |
| "Water" | "Crystal clear ocean waves crashing on a tropical beach at sunset, aerial view" |
| "City" | "Neon-lit Tokyo streets at night, rain reflections, cyberpunk style, panning shot" |

---

## ⌨️ Command Line Interface

### Basic Commands

```powershell
# Generate a video
python generate.py --prompt "A sunset over mountains"

# Specify duration and quality
python generate.py --prompt "Ocean waves" --seconds 8 --quality high

# Custom resolution
python generate.py --prompt "City at night" --width 1280 --height 720

# Reproducible generation with seed
python generate.py --prompt "Forest scene" --seed 42

# Preview prompt expansion without generating
python generate.py --prompt "Cat playing" --show-prompt --dry-run

# Show system information
python generate.py --info
```

### All CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Text description of the video | Required |
| `--seconds` | Video duration in seconds | 6 |
| `--width` | Frame width in pixels | 854 |
| `--height` | Frame height in pixels | 480 |
| `--fps` | Frames per second | 24 |
| `--quality` | Quality preset (fast/balanced/quality) | balanced |
| `--steps` | Number of inference steps | 30 |
| `--guidance` | Guidance scale | 7.5 |
| `--seed` | Random seed (-1 for random) | -1 |
| `--output` | Output directory | ./outputs |
| `--model` | Model to use | ltx-video |
| `--info` | Show system information | - |
| `--dry-run` | Plan without executing | - |

---

## 🐍 Python API

### Basic Usage

```python
from video_ai import VideoAI, generate

# Quick generation
result = generate("A beautiful sunset over mountains")
print(f"Video saved to: {result.output_path}")

# Using VideoAI class
ai = VideoAI()

# Check system capabilities
capabilities = ai.get_capabilities()
print(f"GPU: {capabilities['gpu_name']}")
print(f"VRAM: {capabilities['vram_total_gb']} GB")

# Generate with options
result = ai.generate(
    prompt="A colorful coral reef with tropical fish",
    duration_seconds=8,
    quality_preset="balanced",
    seed=42
)
```

### Python SDK

```python
from video_ai.sdk import VideoAIClient

# Connect to API server
client = VideoAIClient(api_url="http://localhost:8000")

# Generate video
job = await client.generate(
    prompt="A majestic eagle soaring over mountains",
    width=1280,
    height=720,
    quality_preset="quality"
)

# Wait for completion with progress callback
def on_progress(progress, status):
    print(f"{progress}% - {status}")

result = await client.wait_for_completion(job.job_id, on_progress=on_progress)

# Download result
await client.download(result.job_id, "eagle_video.mp4")
```

### JavaScript SDK

```typescript
import { VideoAIClient } from '@video-ai/sdk';

const client = new VideoAIClient({ apiUrl: 'http://localhost:8000' });

// Generate video
const job = await client.generate({
  prompt: 'A sunset over the ocean',
  qualityPreset: 'balanced'
});

// Wait with progress
const result = await client.waitForCompletion(job.jobId, {
  onProgress: (progress) => console.log(`${progress}%`)
});

// Download
await client.download(result.jobId, 'sunset.mp4');
```

---

## 🌐 REST API

### Start the API Server

```powershell
# Start API server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# With auto-reload for development
python -m uvicorn api.server:app --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Start video generation |
| `GET` | `/jobs/{job_id}` | Get job status |
| `DELETE` | `/jobs/{job_id}` | Cancel job |
| `GET` | `/jobs/{job_id}/output` | Download video |
| `GET` | `/models` | List available models |
| `GET` | `/status` | System status |
| `GET` | `/health` | Health check |
| `WS` | `/ws/{client_id}` | WebSocket for real-time updates |

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A spaceship flying through a nebula",
    "width": 1280,
    "height": 720,
    "duration_seconds": 6,
    "quality_preset": "balanced"
  }'
```

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Build the image
docker build -t video-ai:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 video-ai:latest

# Or use docker-compose
docker-compose up -d
```

### Development Mode

```bash
docker-compose -f docker-compose.dev.yml up
```

---

## ☸️ Kubernetes Deployment

### Using Helm

```bash
# Add the repo (if hosted)
helm repo add video-ai https://charts.video-ai.dev

# Install
helm install video-ai ./deploy/helm/video-ai \
  --namespace video-ai \
  --create-namespace \
  --set api.ingress.hosts[0].host=video-ai.example.com
```

### Custom Values

```yaml
# custom-values.yaml
api:
  replicaCount: 2
  resources:
    requests:
      nvidia.com/gpu: 1

persistence:
  models:
    size: 200Gi
```

```bash
helm install video-ai ./deploy/helm/video-ai -f custom-values.yaml
```

---

## 📁 Project Structure

```
video_ai/
├── 📂 api/                   # REST/WebSocket API
│   └── server.py            # FastAPI application
│
├── 📂 sdk/                   # Client SDKs
│   ├── python_sdk.py        # Python client
│   └── javascript/          # TypeScript/JS client
│
├── 📂 agent/                 # AI Agent (orchestration)
│   ├── planner.py           # Central orchestrator
│   ├── prompt_engine.py     # Prompt expansion
│   ├── temporal_prompt.py   # Shot list generator
│   ├── resource_monitor.py  # GPU/RAM monitoring
│   └── retry_logic.py       # Fault tolerance
│
├── 📂 runtime/               # Execution layer
│   ├── cuda_session.py      # CUDA/TensorRT setup
│   ├── gpu_scheduler.py     # Job scheduling
│   ├── onnx_loader.py       # Model loading
│   └── inference.py         # Inference engine
│
├── 📂 safety/                # Compliance
│   └── compliance.py        # Filters, watermarking, audit
│
├── 📂 models/                # Model definitions
│   ├── text_encoder.py      # Text encoding
│   ├── video_diffusion.py   # Diffusion model
│   ├── vae.py               # Video VAE
│   └── pipeline.py          # Full pipeline
│
├── 📂 video/                 # Video assembly
│   ├── frame_writer.py      # Frame output
│   ├── assembler.py         # Video encoding
│   └── ffmpeg_wrapper.py    # FFmpeg interface
│
├── 📂 configs/               # Configuration files
│   ├── hardware.yaml        # GPU settings (RTX 3080)
│   ├── models.yaml          # Model configs
│   └── defaults.yaml        # Default params
│
├── 📂 deploy/                # Deployment
│   ├── helm/                # Kubernetes Helm charts
│   ├── nginx/               # Nginx config
│   └── prometheus/          # Monitoring config
│
├── 🐳 Dockerfile            # Container image
├── 🐳 docker-compose.yml    # Multi-container setup
├── 🎯 run_ui.py             # Launch web UI
├── 🎯 generate.py           # CLI entry point
├── 📋 requirements.txt      # Dependencies
└── 📖 README.md             # This file
```

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ "CUDA not available"
```powershell
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### ❌ "TensorRT not found"
```powershell
pip install tensorrt>=8.6.0
```

#### ❌ "FFmpeg not found"
```powershell
# Windows
winget install FFmpeg

# Linux
sudo apt install ffmpeg
```

#### ❌ "Out of memory"
- Use "fast" quality preset
- Enable quantization: Use INT8 or FP16 mode
- Reduce resolution: `--width 640 --height 360`
- Reduce duration: `--seconds 4`
- Enable tiled processing in hardware.yaml

#### ❌ "Model not found"
```powershell
python download_models.py --model ltx-video-2
```

### Debug Mode

```powershell
# Enable verbose logging
LOG_LEVEL=DEBUG python -m uvicorn api.server:app
```

---

## 📊 Supported Models

| Model | Parameters | VRAM | Max Resolution | Quality |
|-------|------------|------|----------------|---------|
| LTX-Video-2 | 5B | 8 GB | 4K (3840x2160) | ⭐⭐⭐⭐⭐ |
| HunyuanVideo | 13B | 16 GB | 1920x1080 | ⭐⭐⭐⭐⭐ |
| Genmo Mochi | 10B | 12 GB | 1920x1080 | ⭐⭐⭐⭐ |
| AccVideo | 1.5B | 4 GB | 1280x720 | ⭐⭐⭐ (Fast) |
| CogVideoX-2B | 2B | 6 GB | 1280x720 | ⭐⭐⭐ |
| ZeroScope V2 | 1.7B | 4 GB | 576x320 | ⭐⭐ (Fast) |

---

## ⚙️ Configuration

### Hardware Settings (`configs/hardware.yaml`)
```yaml
gpu:
  device_id: 0
  name: "NVIDIA GeForce RTX 3080"
  vram_gb: 10.0
  compute_capability: "8.6"

cuda:
  version_minimum: "11.8"
  cudnn_version: "8.9"
  tensorrt_enabled: true

memory:
  vram_buffer_gb: 1.5
  enable_tiled_processing: true
```

### Quality Presets (`configs/defaults.yaml`)
```yaml
quality_presets:
  fast:
    num_inference_steps: 15
    width: 640
    height: 360
  balanced:
    num_inference_steps: 30
    width: 1280
    height: 720
  quality:
    num_inference_steps: 50
    width: 1920
    height: 1080
  ultra:
    num_inference_steps: 75
    width: 3840
    height: 2160
```

---

## 🛡️ Safety & Compliance

### Content Filtering
- Automatic NSFW/violence detection
- Customizable blocked keyword lists
- Bias detection and mitigation

### Provenance
- C2PA-compliant digital watermarking
- AI-generated content metadata
- Audit logging for all generations

### Configuration
```yaml
# In configs/defaults.yaml
safety:
  content_filter:
    enabled: true
    threshold: 0.7
  watermark:
    enabled: true
    strength: 0.3
  audit_logging:
    enabled: true
```

---

## 🗺️ Roadmap

- [x] CUDA/TensorRT integration
- [x] REST/WebSocket API
- [x] Python & JavaScript SDKs
- [x] Docker/Kubernetes deployment
- [x] GPU scheduler with quantization
- [x] Safety/compliance pipeline
- [x] Temporal prompt generator
- [ ] Multi-GPU distributed inference
- [ ] Frame interpolation (RIFE)
- [ ] Upscaling (Real-ESRGAN)
- [ ] Audio-visual sync
- [ ] Multi-shot timeline editing

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

---

<p align="center">
  Made with ❤️ for the AI community<br>
  Optimized for NVIDIA RTX GPUs
</p>
