#!/usr/bin/env python3
"""
Video AI - Command Line Interface

Generate AI videos from text prompts using AMD GPU acceleration.

Usage:
    python generate.py --prompt "Your description here"
    python generate.py --prompt "A sunset" --duration 10 --quality high
    python generate.py --config generation.yaml
    
Examples:
    python generate.py --prompt "A cinematic drone shot over mountains at sunset"
    python generate.py --prompt "A cat playing" --seconds 4 --fps 30
    python generate.py --prompt "Ocean waves" --width 1280 --height 720 --seed 42
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_ai import VideoAI, GenerationPlanner
from video_ai.agent import ResourceMonitor, PromptEngine


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                     VIDEO AI GENERATOR                         ║
║         Local AI Video Generation for AMD GPUs                 ║
║                    DirectML Accelerated                        ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_info():
    """Print system information"""
    monitor = ResourceMonitor()
    status = monitor.get_resource_status()
    
    print("\n📊 System Information:")
    print(f"   GPU: {status.gpu.name}")
    print(f"   VRAM: {status.gpu.vram_free_gb:.1f}/{status.gpu.vram_total_gb:.1f} GB free")
    print(f"   Backend: {status.gpu.backend}")
    print(f"   DirectML: {'✓ Available' if status.gpu.is_available else '✗ Not available'}")
    print(f"   System RAM: {status.system.ram_available_gb:.1f}/{status.system.ram_total_gb:.1f} GB free")
    
    if status.warnings:
        print("\n⚠️  Warnings:")
        for warning in status.warnings:
            print(f"   - {warning}")
    
    print(f"\n📐 Recommended Settings:")
    print(f"   Resolution: {status.recommended_resolution[0]}x{status.recommended_resolution[1]}")
    print(f"   Max Frames: {status.recommended_max_frames}")
    print()


def expand_prompt_interactive():
    """Interactive prompt expansion"""
    engine = PromptEngine()
    
    print("\n🎬 Interactive Prompt Expansion")
    print("-" * 40)
    
    prompt = input("Enter your prompt: ").strip()
    
    if not prompt:
        print("Error: Empty prompt")
        return
    
    # Expand the prompt
    expanded = engine.expand(prompt)
    
    print("\n📝 Prompt Analysis:")
    print(f"   Original: {expanded.original}")
    print(f"   Subject: {expanded.subject}")
    print(f"   Camera: {expanded.camera_motion}")
    print(f"   Lighting: {expanded.lighting}")
    print(f"   Style: {expanded.style}")
    
    print(f"\n✨ Expanded Prompt:")
    print(f"   {expanded.expanded}")
    
    print(f"\n🚫 Negative Prompt:")
    print(f"   {expanded.negative}")
    
    print(f"\n🎲 Seed: {expanded.seed}")


def list_models():
    """List available models"""
    from video_ai.runtime import ONNXModelLoader
    
    loader = ONNXModelLoader()
    
    print("\n📦 Model Status:")
    print("-" * 50)
    
    # Check required models
    required = loader.get_required_models()
    
    for model in required:
        status = "✓" if model['available'] else "✗"
        print(f"   {status} {model['type']}: {model['name']}")
    
    # List available models
    available = loader.list_available_models()
    
    if available:
        print("\n📚 Available Models:")
        for mtype, models in available.items():
            print(f"   {mtype}:")
            for m in models:
                print(f"     - {m}")
    else:
        print("\n⚠️  No models found. Please download models first.")
        print("   See README.md for model setup instructions.")


def run_generation(args):
    """Run video generation"""
    logger = logging.getLogger(__name__)
    
    # Validate prompt
    if not args.prompt:
        print("Error: --prompt is required")
        return 1
    
    print(f"\n🎬 Generating Video")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.seconds}s")
    print(f"   Resolution: {args.width}x{args.height}")
    print(f"   FPS: {args.fps}")
    print(f"   Quality: {args.quality}")
    if args.seed:
        print(f"   Seed: {args.seed}")
    print()
    
    # Initialize generator
    ai = VideoAI(output_dir=args.output_dir)
    
    # Set up progress callback
    def progress_callback(current, total, message):
        progress = current / total if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r   [{bar}] {progress*100:.0f}% - {message}", end="", flush=True)
        if current == total:
            print()
    
    try:
        # Plan the generation
        job = ai.plan(
            prompt=args.prompt,
            duration_seconds=args.seconds,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
            quality_preset=args.quality,
            num_inference_steps=args.steps
        )
        
        print(f"   Job ID: {job.id}")
        print(f"   Frames: {job.num_frames}")
        print(f"   Steps: {job.num_inference_steps}")
        print()
        
        # Show expanded prompt
        if args.show_prompt and job.expanded_prompt:
            print(f"   Expanded: {job.expanded_prompt.expanded[:100]}...")
            print()
        
        # Estimate time
        estimated = ai.planner.estimate_generation_time(
            job.width, job.height, job.num_frames, job.num_inference_steps
        )
        print(f"   Estimated time: {estimated/60:.1f} minutes")
        print()
        
        if args.dry_run:
            print("   [Dry run - not generating]")
            return 0
        
        # Execute
        start_time = time.time()
        result = ai.execute(job)
        elapsed = time.time() - start_time
        
        if result.success:
            print(f"\n✅ Generation Complete!")
            print(f"   Output: {result.output_path}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Frames: {len(result.frame_paths)}")
            
            # Save metadata
            if args.save_metadata:
                metadata_path = Path(result.output_path).with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(result.metadata, f, indent=2, default=str)
                print(f"   Metadata: {metadata_path}")
            
            return 0
        else:
            print(f"\n❌ Generation Failed!")
            print(f"   Error: {result.error}")
            return 1
            
    except Exception as e:
        logger.exception("Generation failed")
        print(f"\n❌ Error: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Video AI - Generate AI videos from text prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "A sunset over the ocean"
  python generate.py --prompt "A cat playing" --seconds 4 --quality high
  python generate.py --prompt "Mountains" --width 1280 --height 720 --seed 42
  python generate.py --info
  python generate.py --models
  python generate.py --expand
        """
    )
    
    # Main options
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt describing the video to generate'
    )
    
    # Video settings
    parser.add_argument(
        '--seconds', '-s',
        type=float,
        default=6.0,
        help='Video duration in seconds (default: 6)'
    )
    
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=854,
        help='Frame width (default: 854)'
    )
    
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=480,
        help='Frame height (default: 480)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Quality settings
    parser.add_argument(
        '--quality', '-q',
        choices=['fast', 'balanced', 'quality'],
        default='balanced',
        help='Quality preset (default: balanced)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of diffusion steps (default: 30)'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (default: outputs/)'
    )
    
    parser.add_argument(
        '--save-metadata',
        action='store_true',
        default=True,
        help='Save generation metadata to JSON'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Keep intermediate frame files'
    )
    
    # Utility options
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show system information and exit'
    )
    
    parser.add_argument(
        '--models', '-m',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--expand', '-e',
        action='store_true',
        help='Interactive prompt expansion mode'
    )
    
    parser.add_argument(
        '--show-prompt',
        action='store_true',
        help='Show expanded prompt before generation'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Plan generation but don\'t execute'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Save logs to file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Print banner
    print_banner()
    
    # Handle utility modes
    if args.info:
        print_system_info()
        return 0
    
    if args.models:
        list_models()
        return 0
    
    if args.expand:
        expand_prompt_interactive()
        return 0
    
    # Require prompt for generation
    if not args.prompt:
        parser.print_help()
        print("\nError: --prompt is required for generation")
        print("Use --info to check system status")
        print("Use --models to list available models")
        return 1
    
    # Run generation
    return run_generation(args)


if __name__ == "__main__":
    sys.exit(main())
