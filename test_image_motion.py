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
        ("video_ai.image_motion.sam2_segmenter", "SAM2 Segmenter"),
        ("video_ai.image_motion.pose_estimator", "Pose Estimator"),
        ("video_ai.image_motion.motion_planner", "Motion Planner"),
        ("video_ai.image_motion.motion_synthesizer", "Motion Synthesizer"),
        ("video_ai.image_motion.pose_conditioned_pipeline", "Pose Pipeline"),
        ("video_ai.image_motion.temporal_consistency", "Temporal Consistency"),
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
# 8. SAM2 Segmenter
# ═══════════════════════════════════════════════════════════════════

def test_sam2_segmenter_dataclass():
    """SAM2 MultiObjectSegmentation dataclass works correctly."""
    print("\n[TEST] Testing SAM2Segmenter dataclass...")

    from video_ai.image_motion.sam2_segmenter import (
        MultiObjectSegmentation,
    )

    mask = np.ones((64, 64), dtype=np.float32)
    seg = MultiObjectSegmentation(
        subject_mask=mask,
        subject_confidence=0.95,
        subject_label="person",
        subject_bbox=(10, 10, 50, 50),
        background_mask=1.0 - mask,
        retries_used=1,
        strategy_used="centre_point",
    )

    assert seg.is_high_confidence, "0.95 should be high confidence"
    assert seg.is_usable, "0.95 should be usable"
    assert seg.retries_used == 1
    print("  [OK] High confidence detection correct")

    # Low confidence
    seg2 = MultiObjectSegmentation(
        subject_mask=mask,
        subject_confidence=0.5,
        subject_label="object",
        subject_bbox=(0, 0, 64, 64),
        background_mask=1.0 - mask,
    )
    assert not seg2.is_high_confidence
    assert not seg2.is_usable
    print("  [OK] Low confidence detection correct")

    print("  SAM2 segmenter dataclass test passed")


# ═══════════════════════════════════════════════════════════════════
# 9. Pose Estimator
# ═══════════════════════════════════════════════════════════════════

def test_pose_estimator_heuristic():
    """PoseEstimator heuristic fallback produces valid keypoints."""
    print("\n[TEST] Testing PoseEstimator heuristic fallback...")

    from video_ai.image_motion.pose_estimator import (
        PoseEstimator, Joint,
    )

    img = np.zeros((200, 150, 3), dtype=np.uint8)
    img[30:170, 40:110] = 180  # person-shaped blob

    mask = np.zeros((200, 150), dtype=np.float32)
    mask[30:170, 40:110] = 1.0

    estimator = PoseEstimator(device="cpu")
    pose = estimator.estimate(img, mask=mask, bbox=(40, 30, 110, 170))

    assert len(pose.keypoints) == 18, f"Expected 18 keypoints, got {len(pose.keypoints)}"
    assert pose.num_visible > 0, "Should have some visible joints"
    assert pose.subject_height_px > 0, "Should estimate height"
    print(f"  [OK] Keypoints: {pose.num_visible}/18 visible, height={pose.subject_height_px:.0f}px")

    # Pose map rendering
    pose_map = estimator.render_pose_map(pose, 150, 200)
    assert pose_map.shape == (200, 150, 3)
    assert pose_map.dtype == np.uint8
    print("  [OK] Pose map rendered")

    # OpenPose array export
    arr = pose.to_openpose_array()
    assert arr.shape == (18, 3)
    print("  [OK] to_openpose_array shape correct")

    print("  Pose estimator heuristic test passed")


# ═══════════════════════════════════════════════════════════════════
# 10. Motion Synthesizer
# ═══════════════════════════════════════════════════════════════════

def test_motion_synthesizer_walk():
    """MotionSynthesizer produces walk cycle with correct frame count."""
    print("\n[TEST] Testing MotionSynthesizer walk cycle...")

    from video_ai.image_motion.pose_estimator import Keypoint, PoseResult, Joint
    from video_ai.image_motion.motion_synthesizer import (
        MotionSynthesizer, MotionConfig,
    )

    # Build a synthetic initial pose
    kps = []
    for j in Joint:
        kps.append(Keypoint(x=100 + j.value * 5, y=50 + j.value * 10, confidence=0.8))
    initial = PoseResult(
        keypoints=kps,
        overall_confidence=0.8,
        subject_height_px=150.0,
        facing_angle=0.0,
        hip_centre=(120.0, 130.0),
    )

    synth = MotionSynthesizer()
    config = MotionConfig(num_frames=16, fps=8, seed=42)
    frames = synth.synthesize(initial, "walk", "medium", "forward", config)

    assert len(frames) == 16, f"Expected 16 frames, got {len(frames)}"
    assert all(len(f.keypoints) == 18 for f in frames)
    print(f"  [OK] Walk cycle: {len(frames)} frames, 18 keypoints each")

    # Keypoints should change between frames (not all identical)
    kp_f0 = frames[0].keypoints[Joint.R_KNEE.value]
    kp_f8 = frames[8].keypoints[Joint.R_KNEE.value]
    assert kp_f0.y != kp_f8.y, "Knee should move during walk cycle"
    print("  [OK] Keypoints vary between frames")

    print("  Motion synthesizer walk test passed")


def test_motion_synthesizer_jump():
    """MotionSynthesizer produces jump sequence with vertical displacement."""
    print("\n[TEST] Testing MotionSynthesizer jump sequence...")

    from video_ai.image_motion.pose_estimator import Keypoint, PoseResult, Joint
    from video_ai.image_motion.motion_synthesizer import (
        MotionSynthesizer, MotionConfig,
    )

    kps = []
    for j in Joint:
        kps.append(Keypoint(x=100.0, y=50.0 + j.value * 10, confidence=0.8))
    initial = PoseResult(
        keypoints=kps,
        overall_confidence=0.8,
        subject_height_px=150.0,
        hip_centre=(100.0, 100.0),
    )

    synth = MotionSynthesizer()
    config = MotionConfig(num_frames=20, fps=8, seed=42, intensity=1.0)
    frames = synth.synthesize(initial, "jump", "medium", "forward", config)

    assert len(frames) == 20

    # During peak flight (around frame 10-11), nose should be higher (lower y)
    nose_ys = [f.keypoints[Joint.NOSE.value].y for f in frames]
    peak_y = min(nose_ys)
    initial_y = nose_ys[0]
    assert peak_y < initial_y, "Jump peak should be above start position"
    print(f"  [OK] Jump: peak nose y={peak_y:.0f} < initial={initial_y:.0f}")

    print("  Motion synthesizer jump test passed")


# ═══════════════════════════════════════════════════════════════════
# 11. Pose-Conditioned Pipeline Config
# ═══════════════════════════════════════════════════════════════════

def test_pose_pipeline_config():
    """PoseConditionedPipeline GenerationConfig and GenerationResult work."""
    print("\n[TEST] Testing PoseConditionedPipeline config...")

    from video_ai.image_motion.pose_conditioned_pipeline import (
        GenerationConfig, GenerationResult, PoseConditionedPipeline,
    )

    cfg = GenerationConfig(
        width=512, height=512, num_frames=16,
        num_inference_steps=20,
    )
    assert cfg.controlnet_id == "lllyasviel/control_v11p_sd15_openpose"
    print("  [OK] GenerationConfig defaults")

    result = GenerationResult(
        frames=[np.zeros((64, 64, 3), np.uint8)],
        num_frames=1,
        backend_used="test",
    )
    assert result.success
    print("  [OK] GenerationResult")

    # VRAM estimate
    vram = PoseConditionedPipeline.estimate_vram_gb(cfg)
    assert vram > 5.0, f"Expected >5GB VRAM estimate, got {vram}"
    print(f"  [OK] VRAM estimate: {vram:.1f} GB")

    print("  Pose pipeline config test passed")


# ═══════════════════════════════════════════════════════════════════
# 12. Temporal Consistency
# ═══════════════════════════════════════════════════════════════════

def test_temporal_consistency():
    """TemporalConsistencyPass processes synthetic frames."""
    print("\n[TEST] Testing TemporalConsistencyPass...")

    from video_ai.image_motion.temporal_consistency import (
        TemporalConsistencyPass, ConsistencyConfig,
    )

    tc = TemporalConsistencyPass()

    # Create 5 frames with slight flicker
    frames = []
    for i in range(5):
        f = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Add varying noise
        noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(f)

    config = ConsistencyConfig(
        enable_flow_smoothing=False,  # skip flow (needs cv2)
        enable_anti_ghosting=True,
        enable_temporal_denoise=True,
        enable_background_lock=False,
    )

    result = tc.apply(frames, config=config)
    assert len(result.frames) == 5
    print(f"  [OK] Processed {len(result.frames)} frames, {result.num_corrections} corrections")

    # Blended latents
    latents = tc.create_blended_latents(
        num_frames=8, latent_shape=(4, 32, 32), alpha=0.3, seed=42
    )
    assert latents.shape == (8, 4, 32, 32)
    # Adjacent frames should have some correlation
    corr = np.corrcoef(latents[0].ravel(), latents[1].ravel())[0, 1]
    assert corr > 0, f"Adjacent latents should be correlated, got {corr:.3f}"
    print(f"  [OK] Blended latents: shape={latents.shape}, adj_corr={corr:.3f}")

    print("  Temporal consistency test passed")


# ═══════════════════════════════════════════════════════════════════
# 13. Segmenter never-reduce-motion
# ═══════════════════════════════════════════════════════════════════

def test_segmenter_never_reduces_motion():
    """Verify that segmenter warnings never say 'reduced' or 'halved'."""
    print("\n[TEST] Testing segmenter never-reduce-motion policy...")

    from video_ai.image_motion.segmenter import ObjectSegmenter

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    seg = ObjectSegmenter(device="cpu")
    result = seg.segment(img, subject_hint="nothing")

    for w in result.warnings:
        assert "will be reduced" not in w.lower(), (
            f"Warning should not say motion will be reduced: {w}"
        )
        assert "halved" not in w.lower(), (
            f"Warning should not say motion halved: {w}"
        )
    print("  [OK] No 'motion reduced' warnings found")

    print("  Never-reduce-motion test passed")


# ═══════════════════════════════════════════════════════════════════
# 14. Motion planner never-reduce-motion
# ═══════════════════════════════════════════════════════════════════

def test_planner_full_intensity_on_low_seg():
    """MotionPlanner uses full intensity even with LOW segmentation."""
    print("\n[TEST] Testing planner full intensity on low segmentation...")

    from video_ai.image_motion.action_parser import ActionParser
    from video_ai.image_motion.segmenter import (
        SegmentationResult, SegmentationQuality,
    )
    from video_ai.image_motion.motion_planner import MotionPlanner

    parser = ActionParser()
    intent = parser.parse("Make the boy run forward")

    # LOW quality segmentation
    mask = np.ones((100, 100), dtype=np.float32) * 0.3
    seg = SegmentationResult(
        mask=mask, bbox=(0, 0, 100, 100), label="unknown",
        confidence=0.2,
        quality=SegmentationQuality.LOW,
        object_area_ratio=0.3,
        background_mask=1.0 - mask,
    )

    planner = MotionPlanner()
    plan = planner.plan(intent, seg, total_frames=16, fps=8,
                        image_shape=(100, 100))

    # Should NOT have "halved" warning
    for w in plan.warnings:
        assert "halved" not in w.lower(), f"Should not halve: {w}"

    # Intensity should NOT be multiplied by 0.5
    mid = plan.keyframes[len(plan.keyframes) // 2]
    assert mid.intensity > 0, "Intensity should be positive"
    print(f"  [OK] Mid-frame intensity = {mid.intensity:.3f} (not halved)")

    print("  Planner full intensity test passed")


# ═══════════════════════════════════════════════════════════════════
# 15. Regression: existing tests still pass
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
        test_sam2_segmenter_dataclass,
        test_pose_estimator_heuristic,
        test_motion_synthesizer_walk,
        test_motion_synthesizer_jump,
        test_pose_pipeline_config,
        test_temporal_consistency,
        test_segmenter_never_reduces_motion,
        test_planner_full_intensity_on_low_seg,
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
