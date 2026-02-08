"""
Example: Basic Video Generation

This example demonstrates the simplest way to generate a video
using the Video AI system.
"""

from video_ai import VideoAI, generate

def main():
    # Method 1: Quick generation using the convenience function
    print("Method 1: Quick generation")
    print("-" * 40)
    
    result = generate(
        prompt="A majestic eagle soaring over snow-capped mountains at sunset",
        duration_seconds=6
    )
    
    print(f"Video saved to: {result.output_path}")
    print(f"Duration: {result.metadata.get('duration_seconds', 'N/A')}s")
    print(f"Resolution: {result.metadata.get('width', 'N/A')}x{result.metadata.get('height', 'N/A')}")
    print()
    
    # Method 2: Using the VideoAI class
    print("Method 2: Using VideoAI class")
    print("-" * 40)
    
    ai = VideoAI()
    
    # Check capabilities first
    capabilities = ai.get_capabilities()
    print(f"GPU: {capabilities.get('gpu_name', 'Unknown')}")
    print(f"VRAM: {capabilities.get('vram_total_gb', 'N/A'):.1f} GB")
    print(f"Recommended resolution: {capabilities.get('recommended_resolution', 'N/A')}")
    print()
    
    # Generate with custom settings
    result = ai.generate(
        prompt="A colorful coral reef with tropical fish swimming peacefully",
        duration_seconds=8,
        width=854,
        height=480,
        fps=24,
        quality_preset="balanced",
        seed=42  # For reproducibility
    )
    
    print(f"Video saved to: {result.output_path}")
    print(f"Success: {result.success}")
    print()
    
    # Method 3: Plan then execute (for inspection)
    print("Method 3: Plan then execute")
    print("-" * 40)
    
    # Plan the generation job
    job = ai.plan(
        prompt="A time-lapse of clouds moving over a city skyline",
        duration_seconds=10
    )
    
    print(f"Job ID: {job.job_id}")
    print(f"Planned steps: {len(job.pipeline_steps)}")
    print(f"Estimated VRAM: {job.estimated_vram_gb:.1f} GB")
    print()
    
    # Execute the planned job
    result = ai.execute(job)
    print(f"Video saved to: {result.output_path}")


if __name__ == "__main__":
    main()
