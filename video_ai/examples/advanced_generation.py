"""
Example: Advanced Video Generation

This example demonstrates advanced features including:
- Quality presets
- Prompt expansion
- Resource monitoring
- Error handling with retries
"""

import logging
from pathlib import Path

from video_ai import VideoAI
from video_ai.api import VideoGenerator, VideoGenerationConfig
from video_ai.agent import PromptEngine, ResourceMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_quality_presets():
    """Show different quality preset options"""
    print("\n" + "=" * 60)
    print("Quality Presets Demonstration")
    print("=" * 60)
    
    ai = VideoAI()
    prompt = "A butterfly landing on a flower in a garden"
    
    presets = ["fast", "balanced", "quality"]
    
    for preset in presets:
        print(f"\nGenerating with '{preset}' preset...")
        
        result = ai.generate(
            prompt=prompt,
            duration_seconds=4,
            quality_preset=preset
        )
        
        print(f"  Output: {result.output_path}")
        print(f"  Steps: {result.metadata.get('num_inference_steps', 'N/A')}")
        print(f"  Time: {result.metadata.get('generation_time', 'N/A'):.1f}s")


def demonstrate_prompt_expansion():
    """Show how prompts are expanded internally"""
    print("\n" + "=" * 60)
    print("Prompt Expansion Demonstration")
    print("=" * 60)
    
    engine = PromptEngine()
    
    prompts = [
        "A dog running",
        "zoom in on a flower",
        "sunset over the ocean",
        "cinematic shot of a mountain"
    ]
    
    for prompt in prompts:
        expanded = engine.expand(prompt)
        
        print(f"\nOriginal: '{prompt}'")
        print(f"Expanded: '{expanded.positive[:100]}...'")
        print(f"Camera: {expanded.camera_motion or 'auto'}")
        print(f"Lighting: {expanded.lighting or 'auto'}")
        print(f"Style: {expanded.style or 'auto'}")


def demonstrate_resource_monitoring():
    """Show resource monitoring capabilities"""
    print("\n" + "=" * 60)
    print("Resource Monitoring Demonstration")
    print("=" * 60)
    
    monitor = ResourceMonitor()
    status = monitor.get_resource_status()
    
    print(f"\nGPU Information:")
    print(f"  Name: {status.gpu.name}")
    print(f"  VRAM Total: {status.gpu.vram_total_gb:.1f} GB")
    print(f"  VRAM Free: {status.gpu.vram_free_gb:.1f} GB")
    print(f"  Backend: {status.gpu.backend}")
    
    print(f"\nSystem Information:")
    print(f"  RAM Total: {status.system.ram_total_gb:.1f} GB")
    print(f"  RAM Available: {status.system.ram_available_gb:.1f} GB")
    print(f"  CPU Cores: {status.system.cpu_count}")
    
    print(f"\nRecommendations:")
    print(f"  Resolution: {status.recommended_resolution}")
    print(f"  Max Frames: {status.recommended_max_frames}")
    print(f"  High Quality: {'Yes' if status.can_run_high_quality else 'No'}")
    
    if status.warnings:
        print(f"\nWarnings:")
        for warning in status.warnings:
            print(f"  - {warning}")
    
    # Test generation feasibility
    print(f"\nFeasibility Tests:")
    resolutions = [(854, 480), (1280, 720), (1920, 1080)]
    frames = 144
    
    for width, height in resolutions:
        can_generate, message = monitor.can_generate(width, height, frames)
        status_str = "✓" if can_generate else "✗"
        print(f"  {width}x{height} @ {frames} frames: {status_str} - {message}")


def demonstrate_advanced_generation():
    """Show advanced generation with callbacks and config"""
    print("\n" + "=" * 60)
    print("Advanced Generation with Callbacks")
    print("=" * 60)
    
    # Create custom configuration
    config = VideoGenerationConfig(
        duration_seconds=8,
        width=854,
        height=480,
        fps=24,
        quality_preset="balanced",
        num_inference_steps=30,
        guidance_scale=7.5,
        save_frames=True,  # Also save individual frames
        save_metadata=True
    )
    
    # Create generator with config
    generator = VideoGenerator(config=config)
    
    # Set up progress callback
    def on_progress(progress: float, message: str):
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {progress*100:5.1f}% - {message[:40]:<40}", end="", flush=True)
    
    def on_complete(result):
        print(f"\n  ✓ Generation complete!")
        print(f"    Output: {result.output_path}")
    
    generator.on_progress(on_progress)
    generator.on_complete(on_complete)
    
    # Generate
    print("\nGenerating video with progress tracking...")
    result = generator.generate(
        prompt="A spacecraft traveling through a nebula with colorful gas clouds",
        seed=12345
    )
    
    print(f"\nMetadata saved: {result.metadata.get('metadata_saved', False)}")


def demonstrate_batch_generation():
    """Show batch generation of multiple videos"""
    print("\n" + "=" * 60)
    print("Batch Generation Demonstration")
    print("=" * 60)
    
    ai = VideoAI()
    
    prompts = [
        "Waves crashing on a rocky shore",
        "Snow falling in a quiet forest",
        "Fireworks exploding in the night sky"
    ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: '{prompt[:50]}...'")
        
        result = ai.generate(
            prompt=prompt,
            duration_seconds=4,
            quality_preset="fast",  # Use fast for batch
            seed=i * 1000  # Different seed for each
        )
        
        results.append(result)
        print(f"  Saved: {result.output_path}")
    
    print(f"\n✓ Batch complete! Generated {len(results)} videos.")


def main():
    """Run all demonstrations"""
    try:
        demonstrate_resource_monitoring()
        demonstrate_prompt_expansion()
        
        # Uncomment to run actual generation demos
        # (requires models to be downloaded)
        # demonstrate_quality_presets()
        # demonstrate_advanced_generation()
        # demonstrate_batch_generation()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
