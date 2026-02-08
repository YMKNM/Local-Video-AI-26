# Video AI — Multi-Model Guide

## Supported Models

| Model | Parameters | Default Resolution | FPS | Max Duration | Est. VRAM | Disk | License |
|-------|------------|-------------------|-----|-------------|-----------|------|---------|
| **Wan2.1-T2V-1.3B** | 1.3B | 832×480 | 16 | 5.0 sec (81 frames) | ~9 GB | 27 GB | Apache-2.0 |
| **CogVideoX 2B** | 2B | 720×480 | 8 | 6.1 sec (49 frames) | ~8 GB | 11 GB | Apache-2.0 |
| **CogVideoX 5B** | 5B | 720×480 | 8 | 6.1 sec (49 frames) | ~10 GB | 20 GB | CogVideoX (custom) |
| **LTX-Video 2B** | 2B | 768×512 | 24 | ~10.7 sec (257 frames) | ~11 GB | 27 GB | LTX-Video Open |

### RTX 5080 (16 GB VRAM) Compatibility

All four models are **compatible** with the RTX 5080 at default settings using CPU offload.

### Download & Load Test Status

| Model | Downloaded | Load Tested | Load Time | Notes |
|-------|-----------|-------------|-----------|-------|
| **Wan2.1-T2V-1.3B** | ✅ 26.9 GB | ✅ | ~60s | Default model |
| **CogVideoX 2B** | ✅ 13 GB | ✅ | ~69s | CogVideoXPipeline, CPU offload |
| **CogVideoX 5B** | ❌ | — | — | Custom license, download with `--model cogvideox-5b` |
| **LTX-Video 2B** | ✅ 26.5 GB | ✅ | ~283s | LTXPipeline, T5-XXL encoder (17.75 GB), CPU offload |

---

## Model Details

### 1. Wan2.1-T2V-1.3B ⭐ (Recommended Default)

- **Best for**: General-purpose video generation with best quality-per-VRAM ratio
- **Architecture**: DiT (Diffusion Transformer)
- **Repo**: [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- **Precision**: bfloat16
- **Scheduler**: UniPCMultistepScheduler (flow_shift=3.0)
- **Frame Rule**: 4k+1 (i.e. 5, 9, 13, 17, 21, …, 81)
- **Dimension Rule**: Divisible by 16
- **Recommended Settings**: 832×480, guidance=5.0, 25-30 steps
- **Typical Generation Time**: ~45-90s for 33 frames at 832×480

### 2. CogVideoX 2B

- **Best for**: Short 6-second clips with low VRAM usage
- **Architecture**: 3D VAE + DiT
- **Repo**: [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- **Precision**: float16
- **Fixed Resolution**: 720×480 (model is trained on this resolution)
- **Fixed Duration**: 49 frames (~6.1 sec at 8 fps)
- **Guidance**: 6.0
- **Steps**: 50
- **Very low VRAM with CPU offload**: ~4-5 GB

### 3. CogVideoX 5B

- **Best for**: Higher quality than 2B at the same resolution
- **Architecture**: Same as 2B but larger transformer
- **Repo**: [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- **Precision**: bfloat16
- **License**: CogVideoX custom license (research-friendly)
- **Note**: Needs CPU offload on 16 GB cards

### 4. LTX-Video 2B

- **Best for**: Fast previews, longer videos (up to 10 sec)
- **Architecture**: DiT, real-time capable
- **Repo**: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- **Precision**: bfloat16
- **Frame Rule**: 8k+1 (i.e. 9, 17, 25, 33, …, 257)
- **Dimension Rule**: Divisible by 32
- **FPS**: 24 (smooth video)
- **Steps**: 50 (base model)
- **Guidance**: 7.5
- **Disk**: ~27 GB (T5-XXL encoder is ~18 GB, 4 shards)
- **RAM**: Needs ~28 GB system RAM during loading
- **Download Note**: The HuggingFace repo is a monorepo containing both diffusers and single-file variants. Use `download_models.py` to get only the diffusers format. Disable xet protocol (`HF_HUB_DISABLE_XET=1`) if downloads stall.

---

## Downloading Models

### Using the download utility

```bash
# List all models with download status
python download_models.py --list

# Check what's already downloaded
python download_models.py --check

# Download a specific model
python download_models.py --model wan2.1-t2v-1.3b
python download_models.py --model cogvideox-2b
python download_models.py --model cogvideox-5b
python download_models.py --model ltx-video-2b

# Download all compatible models
python download_models.py --all

# Force re-download
python download_models.py --model cogvideox-2b --force
```

### Manual download with huggingface-cli

```bash
# Example: download CogVideoX-2B
huggingface-cli download THUDM/CogVideoX-2b --local-dir models/cogvideox-2b
```

---

## Using in the Web UI

1. **Start the UI**: `python run_ui.py`
2. **Select a model** from the "🤖 Model" dropdown on the Generate tab
3. **Resolution and duration** presets automatically update per-model
4. **Guidance scale** auto-adjusts to the model's recommended value
5. **Click Generate** — the model loads on first use (~2-5 min), then stays in memory

### Model Switching

When you select a different model in the dropdown:
- The current model is unloaded from GPU memory
- Resolution/duration presets update to match the new model
- The new model loads on the next generation

### Checking Model Status

Go to the **💻 System** tab to see which models are downloaded.

---

## Adding New Models

Models are defined in `video_ai/runtime/model_registry.py`. To add a new model:

1. **Create a `ModelSpec`** entry with all required fields
2. **Register it** with `_register(ModelSpec(...))`
3. **Ensure the diffusers Pipeline class** is available (e.g., `CogVideoXPipeline`)
4. **Set accurate VRAM coefficients** (run a test generation and measure peak VRAM)
5. **Download the model** and verify `model_index.json` exists in the local subdir

### ModelSpec Key Fields

```python
ModelSpec(
    id="my-model",                    # Unique ID (used in code and CLI)
    display_name="My Model",          # Shown in UI
    repo_id="org/model-name",         # HuggingFace repo
    local_subdir="my-model",          # Subdirectory under models/
    pipeline_cls="WanPipeline",       # diffusers pipeline class name
    dtype=torch.bfloat16,             # Model precision
    default_width=832, default_height=480,
    default_num_frames=33,
    max_num_frames=81,
    frame_rule="4k+1",               # Frame count constraint
    dim_multiple=16,                  # Width/height must be divisible by this
    vram_base_gb=5.0,                 # Base VRAM for model loading
    vram_per_pixel_frame=1.4e-7,      # VRAM per pixel×frame
)
```

---

## Hardware Requirements

### Minimum
- **GPU**: NVIDIA GPU with 8+ GB VRAM (RTX 3060, etc.)
- **RAM**: 16 GB system RAM
- **Disk**: 10-30 GB per model (LTX-Video is ~27 GB due to T5-XXL encoder)
- **CUDA**: 11.8+

### Recommended (RTX 5080)
- **GPU**: 16 GB VRAM — runs all 4 models comfortably with CPU offload
- **RAM**: 32 GB for smooth model loading
- **Disk**: 60+ GB for multiple models
- **CUDA**: 12.8

### VRAM-Constrained Tips
- Use **CPU offload** (enabled automatically for ≤20 GB cards)
- Start with smaller models: CogVideoX-2B (~4 GB) or Wan2.1-1.3B (~8 GB)
- Reduce resolution or frame count if you get OOM errors
- The system automatically suggests safe parameters on OOM

---

## Troubleshooting

### OOM (Out of Memory) Errors
- The system detects OOM and retries with reduced parameters
- Try: fewer frames, lower resolution, or a smaller model
- CogVideoX-2B uses the least VRAM (~4 GB with offload)

### Model Not Found
- Ensure the model is downloaded: `python download_models.py --check`
- The system falls back to placeholder mode if no models are found
- Check the System tab in the UI for model status

### Slow First Generation
- First generation loads the model into memory (~2-5 minutes)
- Subsequent generations reuse the loaded model and are much faster
- Switching models requires reloading (~2-5 minutes again)
