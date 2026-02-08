#!/usr/bin/env python3
"""
Image-to-Video Animation -- Test Suite

Verifies that all new image_motion components work correctly
without touching existing code.  Uses the same assertion-based
pattern as test_setup.py.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Imports
# ═══════════════════════════════════════════════════════════════════

def test_image_motion_imports():
    """All image_motion sub-modules can be imported."""
    print("\n[TEST] Testing image_motion imports...")

    modules = [
        ("video_ai.image_motion", "Package init"),
        ("video_ai.image_motion.action_parser", "Action Parser"),
        ("video_ai.image_motion.segmenter", "Object Segmenter"),
        ("video_ai.image_motion.motion_planner", "Motion Planner"),
        ("video_ai.image_motion.object_video_pipeline", "Object Video Pipeline"),
        ("video_ai.image_motion.temporal_stabilizer", "Temporal Stabilizer"),
    ]

    passed = 0
    for module, desc in modules:
        try:
            __import__(module)
            print(f"  [OK] {desc}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")

    assert passed == len(modules), f"Only {passed}/{len(modules)} imports succeeded"
    print(f"  All {passed} imports OK")


# ═══════════════════════════════════════════════════════════════════
# 2. Action Parser
# ═══════════════════════════════════════════════════════════════════

def test_action_parser_basic():
    """ActionParser handles common action prompts."""
    print("\n[TEST] Testing ActionParser basic prompts...")

    from video_ai.image_motion.action_parser import (
        ActionParser, ActionVerb, Direction, Speed, SubjectCategory,
    )

    parser = ActionParser()

    # ── Human walk ───────────────────────────────────────────
    intent = parser.parse("Make the boy walk forward slowly")
    assert intent.action == ActionVerb.WALK, f"Expected WALK, got {intent.action}"
    assert intent.direction == Direction.FORWARD
    assert intent.speed == Speed.SLOW
    assert intent.subject_category == SubjectCategory.HUMAN
    print("  [OK] 'Make the boy walk forward slowly'")

    # ── Human run ────────────────────────────────────────────
    intent = parser.parse("Make the woman run right quickly")
    assert intent.action == ActionVerb.RUN, f"Expected RUN, got {intent.action}"
    assert intent.direction == Direction.RIGHT
    assert intent.speed == Speed.FAST
    print("  [OK] 'Make the woman run right quickly'")

    # ── Animal jump ──────────────────────────────────────────
    intent = parser.parse("Make the dog jump")
    assert intent.action == ActionVerb.JUMP, f"Expected JUMP, got {intent.action}"
    assert intent.subject_category == SubjectCategory.ANIMAL
    print("  [OK] 'Make the dog jump'")

    # ── Vehicle drive ────────────────────────────────────────
    intent = parser.parse("Make the car drive away")
    assert intent.action == ActionVerb.DRIVE, f"Expected DRIVE, got {intent.action}"
    assert intent.subject_category == SubjectCategory.VEHICLE
    print("  [OK] 'Make the car drive away'")

    print("  All parser basic tests passed")


def test_action_parser_safety():
    """ActionParser applies safety clamps (vehicles can't dance, etc.)."""
    print("\n[TEST] Testing ActionParser safety clamps...")

    from video_ai.image_motion.action_parser import (
        ActionParser, ActionVerb, SubjectCategory,
    )

    parser = ActionParser()

    # Vehicle + dance should fall back to drive
    intent = parser.parse("Make the truck dance")
    assert intent.action == ActionVerb.DRIVE, (
        f"Vehicle+dance should clamp to DRIVE, got {intent.action}"
    )
    print("  [OK] Vehicle+dance -> DRIVE")

    # Low-confidence prompts get capped intensity
    intent = parser.parse("xyz abc")
    assert intent.intensity <= 0.35, (
        f"Low-confidence prompt should cap intensity, got {intent.intensity}"
    )
    print("  [OK] Low-confidence -> capped intensity")

    print("  All parser safety tests passed")


def test_action_parser_edge_cases():
    """ActionParser handles empty/weird inputs gracefully."""
    print("\n[TEST] Testing ActionParser edge cases...")

    from video_ai.image_motion.action_parser import ActionParser, ActionVerb

    parser = ActionParser()

    # Empty string
    intent = parser.parse("")
    assert intent.action == ActionVerb.IDLE
    print("  [OK] Empty string -> IDLE")

    # No verb
    intent = parser.parse("the boy")
    assert intent.action == ActionVerb.IDLE
    print("  [OK] No verb -> IDLE")

    # Only verb, no subject
    intent = parser.parse("run")
    assert intent.action == ActionVerb.RUN
    print("  [OK] Bare verb 'run'")

    print("  All parser edge-case tests passed")


# ═══════════════════════════════════════════════════════════════════
# 3. Segmenter
# ═══════════════════════════════════════════════════════════════════

def test_segmenter_fallback():
    """ObjectSegmenter centre-heuristic works on a synthetic image."""
    print("\n[TEST] Testing ObjectSegmenter centre heuristic...")

    from video_ai.image_motion.segmenter import (
        ObjectSegmenter, SegmentationQuality,
    )

    # Create a simple 100x100 RGB image with a bright centre object
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 200   # bright square in centre

    seg = ObjectSegmenter(device="cpu")

    # This will likely use the fallback path (GrabCut) or centre heuristic
    result = seg.segment(img, subject_hint="object")

    assert result.mask.shape == (100, 100), f"Mask shape mismatch: {result.mask.shape}"
    assert result.mask.dtype == np.float32, f"Mask dtype: {result.mask.dtype}"
    assert 0 <= result.confidence <= 1.0
    assert result.quality != SegmentationQuality.FAILED
    assert result.object_area_ratio > 0
    print(f"  [OK] Mask shape={result.mask.shape}, quality={result.quality.value}, "
          f"area={result.object_area_ratio:.1%}")

    # Background mask should exist and be inverse
    assert result.background_mask is not None
    print("  [OK] Background mask present")

    print("  Segmenter fallback test passed")


# ═══════════════════════════════════════════════════════════════════
# 4. Motion Planner
# ═══════════════════════════════════════════════════════════════════

def test_motion_planner_basic():
    """MotionPlanner produces a valid plan for a walk action."""
    print("\n[TEST] Testing MotionPlanner basic plan...")

    from video_ai.image_motion.action_parser import ActionParser
    from video_ai.image_motion.segmenter import (
        SegmentationResult, SegmentationQuality,
    )
    from video_ai.image_motion.motion_planner import MotionPlanner

    parser = ActionParser()
    intent = parser.parse("Make the boy walk forward")

    # Build a synthetic segmentation result
    mask = np.ones((100, 100), dtype=np.float32) * 0.0
    mask[20:80, 30:70] = 1.0
    seg = SegmentationResult(
        mask=mask,
        bbox=(30, 20, 70, 80),
        label="boy",
        confidence=0.8,
        quality=SegmentationQuality.HIGH,
        object_area_ratio=0.24,
        background_mask=1.0 - mask,
    )

    planner = MotionPlanner()
    plan = planner.plan(intent, seg, total_frames=16, fps=8,
                        image_shape=(100, 100))

    assert plan.total_frames == 16, f"Expected 16 frames, got {plan.total_frames}"
    assert plan.fps == 8
    assert len(plan.keyframes) == 16
    print(f"  [OK] Plan: {plan.total_frames} frames, {plan.fps} fps")

    # Conditioning arrays
    arrays = plan.to_conditioning_arrays()
    assert "dx" in arrays and arrays["dx"].shape == (16,)
    assert "dy" in arrays and arrays["dy"].shape == (16,)
    assert "limb_phase" in arrays
    print("  [OK] Conditioning arrays valid")

    # Duration
    assert abs(plan.duration_seconds - 2.0) < 0.01
    print(f"  [OK] Duration: {plan.duration_seconds}s")

    print("  Motion planner basic test passed")


def test_motion_planner_jump():
    """MotionPlanner adds a parabolic arc for jump actions."""
    print("\n[TEST] Testing MotionPlanner jump arc...")

    from video_ai.image_motion.action_parser import ActionParser
    from video_ai.image_motion.segmenter import (
        SegmentationResult, SegmentationQuality,
    )
    from video_ai.image_motion.motion_planner import MotionPlanner

    parser = ActionParser()
    intent = parser.parse("Make the cat jump")

    mask = np.ones((200, 200), dtype=np.float32)
    seg = SegmentationResult(
        mask=mask, bbox=(0, 0, 200, 200), label="cat",
        confidence=0.7, quality=SegmentationQuality.MEDIUM,
        object_area_ratio=1.0, background_mask=np.zeros_like(mask),
    )

    planner = MotionPlanner()
    plan = planner.plan(intent, seg, total_frames=24, fps=8,
                        image_shape=(200, 200))

    arrays = plan.to_conditioning_arrays()
    dy = arrays["dy"]
    # Mid-clip should have negative dy (upward = negative in screen coords)
    mid = len(dy) // 2
    # The jump arc adds negative dy to some frames
    min_dy = float(dy.min())
    assert min_dy < 0, f"Jump should produce negative dy, min_dy={min_dy}"
    print(f"  [OK] Jump arc: min dy = {min_dy:.2f}")

    print("  Motion planner jump test passed")


# ═══════════════════════════════════════════════════════════════════
# 5. Temporal Stabilizer
# ═══════════════════════════════════════════════════════════════════

def test_temporal_stabilizer():
    """TemporalStabilizer processes a short synthetic sequence."""
    print("\n[TEST] Testing TemporalStabilizer...")

    from video_ai.image_motion.temporal_stabilizer import TemporalStabilizer

    stab = TemporalStabilizer(flow_smoothing=0.5, denoise_strength=0.3)

    # Create 5 synthetic frames with slight jitter
    tmp = Path(tempfile.mkdtemp(prefix="stab_test_"))
    import cv2

    frame_paths = []
    for i in range(5):
        frame = np.full((64, 64, 3), fill_value=128, dtype=np.uint8)
        # Add a moving bright patch with jitter
        offset = i * 5 + np.random.randint(-2, 3)
        x = max(0, min(44, 10 + offset))
        frame[20:40, x:x+20] = 255
        fp = tmp / f"frame_{i:03d}.png"
        cv2.imwrite(str(fp), frame)
        frame_paths.append(str(fp))

    out_dir = str(tmp / "stabilised")
    result = stab.stabilize_sequence(frame_paths, output_dir=out_dir)

    assert len(result) == 5, f"Expected 5 frames, got {len(result)}"
    for fp in result:
        assert Path(fp).exists(), f"Output frame missing: {fp}"
    print(f"  [OK] Stabilised {len(result)} frames")

    print("  Temporal stabilizer test passed")


# ═══════════════════════════════════════════════════════════════════
# 6. Pipeline Config & Result
# ═══════════════════════════════════════════════════════════════════

def test_pipeline_config():
    """PipelineConfig and PipelineResult data classes are usable."""
    print("\n[TEST] Testing PipelineConfig & PipelineResult...")

    from video_ai.image_motion import PipelineConfig, PipelineResult

    cfg = PipelineConfig(
        output_dir="test_out",
        width=256,
        height=256,
        num_frames=16,
        fps=8,
        motion_intensity=0.5,
    )
    assert cfg.width == 256
    assert cfg.motion_intensity == 0.5
    print("  [OK] PipelineConfig")

    result = PipelineResult(
        success=True, video_path="/tmp/test.mp4",
        frame_paths=["/tmp/f0.png", "/tmp/f1.png"],
        elapsed_seconds=5.0,
    )
    d = result.to_dict()
    assert d["success"] is True
    assert d["num_frames"] == 2
    print("  [OK] PipelineResult.to_dict()")

    print("  Pipeline config test passed")


# ═══════════════════════════════════════════════════════════════════
# 7. UI Tab Import
# ═══════════════════════════════════════════════════════════════════

def test_ui_tab_import():
    """The image_motion_tab module imports successfully."""
    print("\n[TEST] Testing UI tab import...")

    from video_ai.ui.image_motion_tab import create_image_motion_tab

    assert callable(create_image_motion_tab)
    print("  [OK] create_image_motion_tab importable")

    print("  UI tab import test passed")


# ═══════════════════════════════════════════════════════════════════
# 8. Regression: existing tests still pass
# ═══════════════════════════════════════════════════════════════════

def test_existing_imports_unchanged():
    """Existing package imports still work after adding image_motion."""
    print("\n[TEST] Regression: existing imports...")

    modules = [
        "video_ai",
        "video_ai.agent.planner",
        "video_ai.agent.prompt_engine",
        "video_ai.agent.resource_monitor",
        "video_ai.runtime.inference",
        "video_ai.video.assembler",
        "video_ai.video.frame_writer",
    ]
    for mod in modules:
        __import__(mod)
        print(f"  [OK] {mod}")

    print("  Regression test passed")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Image-to-Video Animation -- Test Suite")
    print("=" * 60)

    tests = [
        test_image_motion_imports,
        test_action_parser_basic,
        test_action_parser_safety,
        test_action_parser_edge_cases,
        test_segmenter_fallback,
        test_motion_planner_basic,
        test_motion_planner_jump,
        test_temporal_stabilizer,
        test_pipeline_config,
        test_ui_tab_import,
        test_existing_imports_unchanged,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed ({passed + failed} total)")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
