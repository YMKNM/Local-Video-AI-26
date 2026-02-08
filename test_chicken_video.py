#!/usr/bin/env python3
"""
Test script for generating a 10-second chicken dancing video
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def check_system():
    """Check system status"""
    print("=" * 60)
    print("VIDEO AI - SYSTEM STATUS")
    print("=" * 60)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.1f} GB")
    
    from video_ai.generators import MODEL_SPECS, VideoModelType
    
    print("\nAvailable Video Generation Models:")
    for model_type, specs in MODEL_SPECS.items():
        name = specs['name']
        vram = specs['vram_required_gb']
        max_res = specs['max_resolution']
        max_fps = specs['max_fps']
        max_dur = specs['max_duration']
        modes = [m.value for m in specs['modes']]
        
        fit = "✓" if vram <= vram_gb else "✗"
        print(f"  [{fit}] {model_type.value}: {name}")
        print(f"      VRAM: {vram}GB | Max: {max_res} @ {max_fps}fps, {max_dur}s")
        print(f"      Modes: {modes}")
    
    return vram_gb


def generate_chicken_video(vram_gb: float):
    """Generate a 10-second chicken dancing video"""
    print("\n" + "=" * 60)
    print("GENERATING VIDEO: 10-second chicken dancing")
    print("=" * 60)
    
    from video_ai.generators import (
        VideoModelOrchestrator,
        VideoModelType,
        VideoGenerationConfig,
        MODEL_SPECS,
        GenerationMode,
        generate_video
    )
    
    # Select best model that fits in VRAM
    selected_model = None
    for model_type, specs in MODEL_SPECS.items():
        if specs['vram_required_gb'] <= vram_gb:
            if GenerationMode.TEXT_TO_VIDEO in specs['modes']:
                selected_model = model_type
                print(f"\nSelected model: {specs['name']} ({specs['vram_required_gb']}GB VRAM)")
                break
    
    if selected_model is None:
        print("ERROR: No suitable model fits in available VRAM")
        return None
    
    # Create orchestrator
    orchestrator = VideoModelOrchestrator()
    
    def progress_callback(progress, message):
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  [{bar}] {progress*100:.0f}% - {message}", end='', flush=True)
        if progress >= 1.0:
            print()
    
    orchestrator.set_progress_callback(progress_callback)
    
    # Configure generation
    config = VideoGenerationConfig(
        model_type=selected_model,
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        width=768,
        height=512,
        duration_seconds=10.0,  # 10 seconds
        fps=24,
        num_inference_steps=30,
        guidance_scale=7.5,
    )
    
    prompt = """A cute chicken dancing happily in a sunny farm, 
    vibrant colors, smooth motion, high quality, 4K, 
    professional cinematography, golden hour lighting"""
    
    negative_prompt = "blurry, distorted, low quality, static, frozen"
    
    print(f"\nPrompt: {prompt[:80]}...")
    print(f"Resolution: {config.width}x{config.height}")
    print(f"Duration: {config.duration_seconds}s at {config.fps}fps")
    print(f"Steps: {config.num_inference_steps}")
    print()
    
    try:
        result = orchestrator.generate(
            prompt=prompt,
            config=config,
            negative_prompt=negative_prompt
        )
        
        print(f"\n✓ Video generated successfully!")
        print(f"  Output: {result.video_path}")
        print(f"  Duration: {result.duration:.1f}s")
        print(f"  Time: {result.generation_time:.1f}s")
        
        return result.video_path
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        orchestrator.unload_all()


def main():
    """Main test function"""
    try:
        vram_gb = check_system()
        output_path = generate_chicken_video(vram_gb)
        
        print("\n" + "=" * 60)
        if output_path and Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / 1024 / 1024
            print(f"✓ TEST PASSED")
            print(f"  Video: {output_path}")
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print("✗ TEST FAILED - No output generated")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
