# Video AI - Installation Guide

## ✅ Installation Complete

All core requirements have been successfully installed! The platform is ready to use.

## Quick Start

### Easy Mode (Automatic Setup)
```batch
# This will automatically:
# 1. Create virtual environment
# 2. Install all requirements
# 3. Launch the application

start.bat              # Start everything (API + UI)
start.bat --ui         # Start only UI
start.bat --ui --share # Start UI with public link
```

### Manual Mode (If Already Set Up)
```bash
# Activate virtual environment first
venv\Scripts\activate

# Then start the platform
run.bat --all

# Or use Python directly
python run_ui.py
```

## Installed Dependencies

### Core Framework ✅
- PyTorch 2.7.1 + CUDA 11.8
- Diffusers 0.36.0
- Transformers 4.57.6
- Accelerate 1.12.0
- Gradio 6.4.0

### NVIDIA CUDA/TensorRT ✅
- ONNX Runtime GPU 1.23.2
- TensorRT 10.14.1.48
- CUDA Toolkit 13.1.1

### Memory Optimization ✅
- xformers 0.0.31
- bitsandbytes 0.49.1

### Video Processing ✅
- FFmpeg Python
- OpenCV 4.11.0
- imageio 2.37.2

### API Server ✅
- FastAPI 0.128.0
- Uvicorn 0.40.0
- Celery 5.6.2

### Aggressive Generator ✅
- ControlNet Aux 0.0.10
- FaceXLib 0.3.0
- RealESRGAN 0.3.0

## ⚠️ Optional Dependencies (Require C++ Build Tools)

The following packages require **Microsoft Visual C++ 14.0 or greater** to install:

### 1. TTS (Coqui TTS)
- **Purpose**: Text-to-speech for lip sync audio generation
- **Workaround**: Use external TTS services or pre-generated audio files
- **Install**: Uncomment `TTS>=0.21.0` in requirements.txt after installing VS Build Tools

### 2. InsightFace
- **Purpose**: Face detection and analysis for Live Portrait
- **Workaround**: Use alternative face detection libraries (FaceXLib already installed)
- **Install**: Uncomment `insightface>=0.7.3` in requirements.txt after installing VS Build Tools

### 3. Triton
- **Purpose**: GPU kernel optimization (Linux only)
- **Status**: Not available on Windows - already commented out

## Installing C++ Build Tools (Optional)

If you want to install TTS and InsightFace:

### Option 1: Visual Studio Build Tools (Recommended)
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select "Desktop development with C++"
4. Install (requires ~6GB disk space)
5. Restart terminal
6. Uncomment packages in requirements.txt
7. Run: `pip install TTS insightface`

### Option 2: Visual Studio Community
1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install Visual Studio Community
3. Select "Desktop development with C++" workload
4. Follow steps 5-7 above

## Current Status

### ✅ Fully Functional (No C++ Build Tools Required)
- Image generation with SDXL
- Video generation with all 6 models (LTX-2, HunyuanVideo, Mochi, etc.)
- Facial animation with Live Portrait
- Expression engine
- UI/API servers
- All core features

### ⚠️ Limited Functionality (Requires C++ Build Tools)
- TTS-based lip sync (can use external audio files instead)
- InsightFace-based face detection (FaceXLib alternative available)

## Verification

Test the installation:

```bash
# Check Python environment
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check Gradio
python -c "import gradio; print(f'Gradio {gradio.__version__}')"

# Check diffusers
python -c "import diffusers; print(f'Diffusers {diffusers.__version__}')"

# Launch UI
python run_ui.py
```

## Troubleshooting

### Import Errors
If you see module not found errors:
```bash
pip install -r requirements.txt
```

### CUDA Errors
Ensure you have:
- NVIDIA GPU with CUDA 11.8+ support
- Latest NVIDIA drivers installed
- Windows GPU acceleration enabled

### Memory Errors
- Close other GPU applications
- Reduce batch sizes in generation settings
- Use lower resolution outputs

## Next Steps

1. **Test the UI**: `python run_ui.py`
2. **Generate a video**: Use the basic generator tab
3. **Try Aggressive Generator**: Explore the 🔥 tab for advanced image-to-motion
4. **Optional**: Install C++ Build Tools for TTS/InsightFace

## Support

For issues or questions:
1. Check `INSTALLATION.md` (this file)
2. Review `README.md` for usage examples
3. Check logs in `logs/` directory
