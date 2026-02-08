#!/usr/bin/env python3
"""
Video AI - Test Suite

Verifies that all components are working correctly.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    print("\n📦 Testing Imports...")
    
    tests = [
        ("video_ai", "Main package"),
        ("video_ai.agent", "Agent package"),
        ("video_ai.agent.planner", "Generation Planner"),
        ("video_ai.agent.prompt_engine", "Prompt Engine"),
        ("video_ai.agent.resource_monitor", "Resource Monitor"),
        ("video_ai.agent.retry_logic", "Retry Logic"),
        ("video_ai.runtime", "Runtime package"),
        ("video_ai.runtime.directml_session", "DirectML Session"),
        ("video_ai.runtime.onnx_loader", "ONNX Loader"),
        ("video_ai.runtime.inference", "Inference Engine"),
        ("video_ai.video", "Video package"),
        ("video_ai.video.frame_writer", "Frame Writer"),
        ("video_ai.video.ffmpeg_wrapper", "FFmpeg Wrapper"),
        ("video_ai.video.assembler", "Video Assembler"),
    ]
    
    passed = 0
    failed = 0
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✅ {description}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {description}: {e}")
            failed += 1
    
    return failed == 0


def test_prompt_engine():
    """Test prompt expansion"""
    print("\n🎬 Testing Prompt Engine...")
    
    from video_ai.agent.prompt_engine import PromptEngine
    
    engine = PromptEngine()
    
    test_prompts = [
        "A sunset over the ocean",
        "A drone shot of mountains at golden hour",
        "Neon lights in a cyberpunk city at night"
    ]
    
    for prompt in test_prompts:
        try:
            expanded = engine.expand(prompt)
            print(f"  ✅ '{prompt[:30]}...'")
            print(f"     Camera: {expanded.camera_motion}, Style: {expanded.style}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False
    
    return True


def test_resource_monitor():
    """Test resource monitoring"""
    print("\n📊 Testing Resource Monitor...")
    
    from video_ai.agent.resource_monitor import ResourceMonitor
    
    try:
        monitor = ResourceMonitor()
        status = monitor.get_resource_status()
        
        print(f"  ✅ GPU: {status.gpu.name}")
        print(f"     VRAM: {status.gpu.vram_free_gb:.1f}/{status.gpu.vram_total_gb:.1f} GB")
        print(f"     Backend: {status.gpu.backend}")
        print(f"     RAM: {status.system.ram_available_gb:.1f}/{status.system.ram_total_gb:.1f} GB")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_directml():
    """Test DirectML availability"""
    print("\n🔥 Testing DirectML...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        print(f"  Available providers: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print(f"  ✅ DirectML is available!")
            return True
        else:
            print(f"  ⚠️ DirectML not available (CPU fallback will be used)")
            return True  # Not a failure, just a warning
            
    except ImportError:
        print(f"  ❌ ONNX Runtime not installed")
        print(f"     Install with: pip install onnxruntime-directml")
        return False


def test_ffmpeg():
    """Test FFmpeg availability"""
    print("\n🎥 Testing FFmpeg...")
    
    from video_ai.video.ffmpeg_wrapper import FFmpegWrapper
    
    try:
        ffmpeg = FFmpegWrapper()
        print(f"  ✅ FFmpeg found: {ffmpeg.ffmpeg_path}")
        print(f"     Version: {ffmpeg._version}")
        
        encoders = ffmpeg.get_supported_encoders()
        h264 = 'libx264' in encoders
        h265 = 'libx265' in encoders
        
        print(f"     H.264: {'✓' if h264 else '✗'}")
        print(f"     H.265: {'✓' if h265 else '✗'}")
        
        return True
        
    except RuntimeError as e:
        print(f"  ❌ FFmpeg not found!")
        print(f"     Please install FFmpeg and add to PATH")
        return False


def test_frame_writer():
    """Test frame writing"""
    print("\n🖼️ Testing Frame Writer...")
    
    import numpy as np
    from video_ai.video.frame_writer import FrameWriter
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = FrameWriter(tmpdir, format='png')
            
            # Create test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            path = writer.write_frame(frame)
            
            if path.exists():
                print(f"  ✅ Frame written: {path.name}")
                return True
            else:
                print(f"  ❌ Frame not created")
                return False
                
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_video_assembly():
    """Test video assembly (requires FFmpeg)"""
    print("\n🎬 Testing Video Assembly...")
    
    import numpy as np
    from video_ai.video.frame_writer import FrameWriter
    from video_ai.video.assembler import VideoAssembler
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test frames
            writer = FrameWriter(tmpdir / "frames", format='png')
            frames = np.random.randint(0, 255, (24, 240, 320, 3), dtype=np.uint8)
            paths = writer.write_frames(frames)
            
            print(f"  Created {len(paths)} test frames")
            
            # Assemble video
            assembler = VideoAssembler()
            output_path = tmpdir / "test_video.mp4"
            
            success = assembler.assemble(
                frame_paths=paths,
                output_path=output_path,
                fps=24,
                quality_preset='draft'
            )
            
            if success and output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"  ✅ Video created: {size_kb:.1f} KB")
                return True
            else:
                print(f"  ❌ Video not created")
                return False
                
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_generation_planner():
    """Test generation planning"""
    print("\n🧠 Testing Generation Planner...")
    
    from video_ai.agent.planner import GenerationPlanner
    
    try:
        planner = GenerationPlanner()
        
        # Plan a job
        job = planner.plan_generation(
            prompt="A beautiful sunset over the ocean",
            duration_seconds=4,
            quality_preset="fast"
        )
        
        print(f"  ✅ Job planned: {job.id}")
        print(f"     Resolution: {job.width}x{job.height}")
        print(f"     Frames: {job.num_frames}")
        print(f"     Steps: {job.num_inference_steps}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("               VIDEO AI - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Prompt Engine", test_prompt_engine),
        ("Resource Monitor", test_resource_monitor),
        ("DirectML", test_directml),
        ("FFmpeg", test_ffmpeg),
        ("Frame Writer", test_frame_writer),
        ("Video Assembly", test_video_assembly),
        ("Generation Planner", test_generation_planner),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("                    TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    print(f"\n  Results: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Video AI is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
