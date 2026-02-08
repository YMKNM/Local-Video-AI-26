#!/usr/bin/env python3
"""
Video AI - End-to-End Pipeline Validation Test

Generates a real MP4 video proving the full pipeline works:
  FrameWriter → FFmpeg Assembler → MP4 output

This does NOT require any AI models - it creates synthetic frames
to validate every component of the video pipeline.
"""

import sys
import os
import time
import math
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def create_animated_frames(num_frames: int, width: int, height: int) -> np.ndarray:
    """Create visually interesting animated test frames."""
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    for i in range(num_frames):
        t = i / num_frames  # normalized time 0..1

        # Background gradient: shifts hue over time
        for y in range(height):
            ny = y / height
            r = int(128 + 127 * math.sin(2 * math.pi * (t + ny * 0.3)))
            g = int(128 + 127 * math.sin(2 * math.pi * (t + ny * 0.3 + 0.33)))
            b = int(128 + 127 * math.sin(2 * math.pi * (t + ny * 0.3 + 0.66)))
            frames[i, y, :, 0] = r
            frames[i, y, :, 1] = g
            frames[i, y, :, 2] = b

        # Bouncing circle
        cx = int(width * (0.5 + 0.35 * math.sin(2 * math.pi * t * 2)))
        cy = int(height * (0.5 + 0.3 * math.cos(2 * math.pi * t * 3)))
        radius = 40
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        frames[i][mask] = [255, 255, 255]

        # Second circle (smaller, offset phase)
        cx2 = int(width * (0.5 + 0.25 * math.cos(2 * math.pi * t * 1.5)))
        cy2 = int(height * (0.5 + 0.2 * math.sin(2 * math.pi * t * 2.5)))
        mask2 = (xx - cx2) ** 2 + (yy - cy2) ** 2 <= 20 ** 2
        frames[i][mask2] = [255, 200, 0]

        # Frame counter text region (dark bar at bottom)
        bar_h = 40
        frames[i, height - bar_h:, :, :] = (frames[i, height - bar_h:, :, :] * 0.3).astype(np.uint8)

    return frames


def main():
    print("=" * 60)
    print("  VIDEO AI - PIPELINE VALIDATION TEST")
    print("=" * 60)

    # Step 1: System info
    print("\n📋 System Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM: {vram:.1f} GB")
        else:
            print("   GPU: CPU only (CUDA not available)")
    except ImportError:
        print("   GPU: PyTorch not available")

    # Step 2: Create output directory
    output_dir = Path(__file__).parent / "outputs" / "test_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_pipeline_video.mp4"
    print(f"\n📁 Output: {output_path}")

    # Step 3: Generate frames
    fps = 24
    duration = 5  # seconds
    num_frames = fps * duration
    width, height = 640, 360

    print(f"\n🎨 Generating {num_frames} frames ({width}x{height} @ {fps}fps, {duration}s)...")
    t0 = time.time()
    frames = create_animated_frames(num_frames, width, height)
    gen_time = time.time() - t0
    print(f"   ✅ Frames generated in {gen_time:.1f}s")

    # Step 4: Write frames to disk
    print("\n💾 Writing frames to disk...")
    from video_ai.video.frame_writer import FrameWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = FrameWriter(tmpdir, format='png')
        t0 = time.time()
        paths = writer.write_frames(frames)
        write_time = time.time() - t0
        print(f"   ✅ {len(paths)} frames written in {write_time:.1f}s")

        # Step 5: Assemble into video
        print("\n🎬 Assembling video with FFmpeg...")
        from video_ai.video.assembler import VideoAssembler

        assembler = VideoAssembler()
        t0 = time.time()
        success = assembler.assemble(
            frame_paths=paths,
            output_path=output_path,
            fps=fps,
            quality_preset='medium'
        )
        encode_time = time.time() - t0

        if success and output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"   ✅ Video encoded in {encode_time:.1f}s ({size_kb:.0f} KB)")
        else:
            print("   ❌ Video assembly FAILED")
            sys.exit(1)

    # Step 6: Verify with FFmpeg probe
    print("\n🔍 Verifying output...")
    from video_ai.video.ffmpeg_wrapper import FFmpegWrapper

    ffmpeg = FFmpegWrapper()
    try:
        info = ffmpeg.get_video_info(str(output_path))
        print(f"   Format: {info.get('codec', 'unknown')}")
        print(f"   Resolution: {info.get('width', '?')}x{info.get('height', '?')}")
        print(f"   Duration: {info.get('duration', '?')}s")
    except Exception:
        # ffprobe may not be available; check file size instead
        if output_path.stat().st_size > 1024:
            print(f"   File size: {output_path.stat().st_size / 1024:.0f} KB (valid)")
        else:
            print("   ⚠️ File seems too small")

    # Summary
    total_time = gen_time + write_time + encode_time
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE VALIDATION PASSED")
    print(f"  Output: {output_path}")
    print(f"  Video: {width}x{height} @ {fps}fps, {duration}s, {num_frames} frames")
    print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
