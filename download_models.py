#!/usr/bin/env python3
"""
Model Download Utility

Downloads AI video generation models for Video AI.
Uses the centralised model_registry for model metadata.
Supports downloading from Hugging Face Hub with resume capability.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from video_ai.runtime.model_registry import (
        MODEL_REGISTRY, get_model, get_compatible_models, get_all_models,
        ModelSpec, Compatibility,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    logger.warning("model_registry not available — using legacy model list")

# Legacy model registry (fallback when model_registry.py can't be imported)
LEGACY_MODEL_REGISTRY = {
    'clip-vit-large': {
        'type': 'text_encoder',
        'source': 'openai/clip-vit-large-patch14',
        'size_gb': 1.2,
        'description': 'CLIP ViT-L/14 text encoder (ONNX path only)',
        'local_subdir': 'text_encoder/clip-vit-large',
    },
    'sd-vae': {
        'type': 'vae',
        'source': 'stabilityai/sd-vae-ft-mse',
        'size_gb': 0.3,
        'description': 'Stable Diffusion VAE (ONNX path only)',
        'local_subdir': 'vae/sd-vae',
    },
}


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        logger.error("huggingface_hub not installed!")
        logger.error("Install with: pip install huggingface_hub")
        return False


def list_models():
    """List all available models with download status."""
    models_dir = Path(__file__).parent / "models"

    print("\n🤖 AI Video Generation Models (from model_registry)")
    print("=" * 70)

    if REGISTRY_AVAILABLE:
        for spec in MODEL_REGISTRY.values():
            local = models_dir / spec.local_subdir
            downloaded = local.exists() and (local / "model_index.json").exists()
            icon = "✅" if downloaded else "⬜"
            compat = spec.check_compatibility(vram_gb=16.0, disk_free_gb=400.0)
            compat_icon = {"COMPATIBLE": "🟢", "CONDITIONAL": "🟡", "INCOMPATIBLE": "🔴"}.get(compat.name, "⚪")
            print(f"\n  {icon} {spec.display_name}  {compat_icon}")
            print(f"      ID        : {spec.id}")
            print(f"      Repo      : {spec.repo_id}")
            print(f"      Params    : {spec.parameters}")
            print(f"      Disk      : ~{spec.disk_gb} GB")
            print(f"      VRAM (est): ~{spec.estimate_peak_vram(spec.default_width, spec.default_height, spec.default_num_frames):.0f} GB")
            print(f"      License   : {spec.license}")
            if downloaded:
                print(f"      Path      : {local}")
    else:
        for name, info in LEGACY_MODEL_REGISTRY.items():
            print(f"\n  ⬜ {name}")
            print(f"      Source: {info['source']}")
            print(f"      Size:   ~{info['size_gb']} GB")

    print()


def download_model(
    model_id: str,
    models_dir: Path,
    force: bool = False,
) -> bool:
    """
    Download a model from Hugging Face Hub.

    Args:
        model_id: Model ID (e.g. 'wan2.1-t2v-1.3b', 'cogvideox-2b')
        models_dir: Base models directory
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    if not check_huggingface_hub():
        return False

    from huggingface_hub import snapshot_download

    # Look up in registry
    if REGISTRY_AVAILABLE:
        spec = get_model(model_id)
        if spec is None:
            logger.error(f"Unknown model: {model_id}")
            logger.info(f"Available: {[s.id for s in MODEL_REGISTRY.values()]}")
            return False
        repo_id = spec.repo_id
        local_subdir = spec.local_subdir
        disk_gb = spec.disk_gb
        display_name = spec.display_name
    elif model_id in LEGACY_MODEL_REGISTRY:
        info = LEGACY_MODEL_REGISTRY[model_id]
        repo_id = info['source']
        local_subdir = info['local_subdir']
        disk_gb = info['size_gb']
        display_name = model_id
    else:
        logger.error(f"Unknown model: {model_id}")
        return False

    output_dir = models_dir / local_subdir

    # Check if already downloaded
    if output_dir.exists() and (output_dir / "model_index.json").exists() and not force:
        logger.info(f"✅ {display_name} already downloaded at {output_dir}")
        logger.info("   Use --force to re-download")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n📥 Downloading {display_name}")
    logger.info(f"   Repo      : {repo_id}")
    logger.info(f"   Est. Size : ~{disk_gb} GB")
    logger.info(f"   Dest      : {output_dir}")
    logger.info("")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"\n✅ Download complete: {display_name}")
        return True
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        return False


def download_compatible(models_dir: Path, force: bool = False) -> Dict[str, bool]:
    """Download all COMPATIBLE models (not CONDITIONAL or INCOMPATIBLE)."""
    results = {}

    if not REGISTRY_AVAILABLE:
        logger.error("model_registry not available — cannot determine compatible models")
        return results

    compatible = get_compatible_models(vram_gb=16.0)
    logger.info(f"\n📦 Downloading {len(compatible)} compatible models...")
    total_gb = sum(s.disk_gb for s in compatible)
    logger.info(f"   Total disk: ~{total_gb:.0f} GB")
    logger.info("")

    for spec in compatible:
        results[spec.id] = download_model(spec.id, models_dir, force)

    # Summary
    ok = sum(results.values())
    logger.info(f"\n📊 Summary: {ok}/{len(results)} downloaded")
    for mid, status in results.items():
        icon = "✅" if status else "❌"
        logger.info(f"   {icon} {mid}")

    return results


def check_models(models_dir: Path):
    """Check which models are installed."""
    print("\n📊 Model Status")
    print("=" * 60)

    if REGISTRY_AVAILABLE:
        for spec in MODEL_REGISTRY.values():
            p = models_dir / spec.local_subdir
            if p.exists() and (p / "model_index.json").exists():
                # Calculate actual size
                total_bytes = sum(
                    f.stat().st_size for f in p.rglob("*") if f.is_file()
                )
                size_gb = total_bytes / (1024**3)
                print(f"  ✅ {spec.display_name:<30s} ({size_gb:.1f} GB)")
            else:
                print(f"  ❌ {spec.display_name:<30s} (not downloaded)")
    else:
        for name, info in LEGACY_MODEL_REGISTRY.items():
            p = models_dir / info['local_subdir']
            status = "✅ Installed" if p.exists() else "❌ Not installed"
            print(f"  {status}: {name}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download AI video generation models for Video AI"
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model ID to download (e.g. wan2.1-t2v-1.3b, cogvideox-2b)',
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all compatible models',
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models',
    )
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check which models are installed',
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(Path(__file__).parent / "models"),
        help='Models directory (default: models/)',
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download',
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    if args.list:
        list_models()
        return 0

    if args.check:
        check_models(models_dir)
        return 0

    if args.all:
        results = download_compatible(models_dir, args.force)
        return 0 if all(results.values()) else 1

    if args.model:
        success = download_model(args.model, models_dir, args.force)
        return 0 if success else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
