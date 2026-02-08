"""
Object-Aware Video Pipeline -- Architecture v2

Orchestrates the full image-to-video animation workflow using
a pose-conditioned generation pipeline:

    1. **Segment** the subject using SAM 2 with retry logic
       (falls back to SAM 1 / GrabCut, NEVER reduces motion).
    2. **Parse** the action prompt into a structured ``ActionIntent``.
    3. **Detect pose** via DWPose/OpenPose skeletal extraction.
    4. **Synthesize motion** -- procedural biomechanical pose sequence.
    5. **Generate video** via AnimateDiff + ControlNet OpenPose
       (conditioned on pose maps, not just text).
    6. **Temporal consistency** -- latent reuse + optical flow + anti-ghosting.
    7. **Assemble** frames into output video.

Graceful degradation:
  - If segmentation confidence < 90%: RETRY (not reduce motion)
  - If VRAM insufficient: reduce resolution, NEVER reduce motion
  - If ControlNet unavailable: fall back to AnimateDiff / img2img
  - If duration too long: reduce frame count, NEVER reduce coherence

Usage::

    from video_ai.image_motion import ObjectVideoPipeline, PipelineConfig
    pipe = ObjectVideoPipeline()
    result = pipe.run(image_path="photo.jpg",
                      action_prompt="Make the boy run forward")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .action_parser import ActionIntent, ActionParser
from .motion_planner import MotionPlan, MotionPlanner
from .segmenter import ObjectSegmenter, SegmentationResult
from .temporal_stabilizer import TemporalStabilizer

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """End-user-facing knobs for image-to-video generation."""

    # Output
    output_dir: str = "outputs/image_motion"
    width: int = 512
    height: int = 512
    num_frames: int = 24
    fps: int = 8
    num_inference_steps: int = 20
    guidance_scale: float = 6.0
    seed: Optional[int] = None

    # Motion
    motion_intensity: float = 0.7      # 0.0 .. 1.0
    stabilize: bool = True             # apply temporal stabilizer?

    # Model selection
    model_name: str = "wan2.1-t2v-1.3b"   # any key in MODEL_REGISTRY

    # Advanced
    negative_prompt: str = (
        "blurry, distorted, low quality, watermark, text, "
        "static, frozen, jittery, morphing"
    )
    prompt_prefix: str = ""            # prepended to the enriched prompt
    prompt_suffix: str = ""            # appended to the enriched prompt


# ── Result ────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Outcome of one image-to-video generation."""

    success: bool
    video_path: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)
    segmentation: Optional[SegmentationResult] = None
    intent: Optional[ActionIntent] = None
    motion_plan: Optional[MotionPlan] = None
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "video_path": self.video_path,
            "num_frames": len(self.frame_paths),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "segmentation": self.segmentation.to_dict() if self.segmentation else None,
            "intent": {
                "action": self.intent.action.value,
                "subject": self.intent.subject,
                "direction": self.intent.direction.value,
                "speed": self.intent.speed.value,
                "confidence": self.intent.confidence,
            } if self.intent else None,
            "warnings": self.warnings,
            "error": self.error,
        }


# ── Pipeline ──────────────────────────────────────────────────────────

# Stage index -> (start_fraction, end_fraction) of overall progress.
# Generation (stage 5) dominates wall-clock time.
_STAGE_RANGES: List[Tuple[float, float]] = [
    (0.00, 0.02),   # 0: Load image
    (0.02, 0.14),   # 1: SAM segmentation (may retry)
    (0.14, 0.17),   # 2: Parse action
    (0.17, 0.24),   # 3: Detect pose
    (0.24, 0.29),   # 4: Synthesize motion
    (0.29, 0.85),   # 5: Generate video (heavy)
    (0.85, 0.92),   # 6: Temporal consistency
    (0.92, 0.96),   # 7: Stabilise
    (0.96, 1.00),   # 8: Assemble
]


class ObjectVideoPipeline:
    """
    End-to-end image-to-video animation with pose-conditioned generation.

    Pipeline stages:
      SAM 2 (retry) -> DWPose -> MotionSynthesizer -> AnimateDiff+ControlNet
      -> TemporalConsistency -> VideoAssembly
    """

    # Make the class-level ranges accessible on the instance
    _STAGE_RANGES = _STAGE_RANGES

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback

        # Sub-modules (lazy-initialised)
        self._sam2_segmenter = None
        self._legacy_segmenter: Optional[ObjectSegmenter] = None
        self._parser: Optional[ActionParser] = None
        self._planner: Optional[MotionPlanner] = None
        self._stabilizer: Optional[TemporalStabilizer] = None
        self._pose_estimator = None
        self._motion_synthesizer = None
        self._pose_pipeline = None
        self._temporal_consistency = None
        self._diffusion_pipeline = None  # legacy fallback

    # ── public entry ──────────────────────────────────────────

    def run(
        self,
        image_path: str,
        action_prompt: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Run the full pose-conditioned image-to-video pipeline.

        Parameters
        ----------
        image_path : str
            Path to the input still image.
        action_prompt : str
            Natural-language action, e.g. "Make the boy run forward".
        config : PipelineConfig, optional
            Overrides the instance-level config for this run only.

        Returns
        -------
        PipelineResult
        """
        cfg = config or self.config
        start = time.time()
        warnings: List[str] = []
        job_id = uuid.uuid4().hex[:8]

        try:
            # ── 0. Load image ────────────────────────────────
            self._report(0, "Loading image...")
            image_rgb = self._load_image(image_path)
            h, w = image_rgb.shape[:2]
            logger.info("[%s] Image loaded: %dx%d", job_id, w, h)

            # ── 1. Segment with SAM 2 + retry ───────────────
            self._report(1, "Segmenting subject (SAM 2 with retry)...")
            seg = self._segment_with_retry(
                image_rgb, self._extract_subject_hint(action_prompt)
            )
            warnings.extend(seg.warnings)
            logger.info(
                "[%s] Segmentation: %s (%.0f%% conf, quality=%s, retries=%d)",
                job_id, seg.label, seg.confidence * 100,
                seg.quality.value,
                getattr(seg, '_retries_used', 0),
            )

            # ── 2. Parse action ──────────────────────────────
            self._report(2, "Parsing action prompt...")
            intent = self._get_parser().parse(action_prompt)
            intent.intensity *= cfg.motion_intensity
            warnings.extend(intent.warnings)
            logger.info(
                "[%s] Intent: %s %s %s (conf=%.2f)",
                job_id, intent.action.value, intent.subject,
                intent.direction.value, intent.confidence,
            )

            # ── 3. Detect pose (DWPose/OpenPose) ────────────
            self._report(3, "Extracting skeletal pose...")
            pose_result = self._detect_pose(image_rgb, seg)
            if pose_result.warnings:
                warnings.extend(pose_result.warnings)
            logger.info(
                "[%s] Pose: %d visible joints, conf=%.2f, height=%.0fpx",
                job_id, pose_result.num_visible,
                pose_result.overall_confidence,
                pose_result.subject_height_px,
            )

            # ── 4. Synthesize motion (procedural poses) ─────
            self._report(4, "Synthesizing motion sequence...")
            pose_sequence = self._synthesize_motion(
                pose_result, intent, cfg
            )
            logger.info(
                "[%s] Motion synthesized: %d frames",
                job_id, len(pose_sequence),
            )

            # ── 5. Plan motion (legacy, for metadata) ───────
            plan = self._get_planner().plan(
                intent=intent,
                seg=seg,
                total_frames=cfg.num_frames,
                fps=cfg.fps,
                image_shape=(h, w),
            )
            warnings.extend(plan.warnings)

            # ── 6. Generate video (pose-conditioned) ─────────
            self._report(5, "Generating video (pose-conditioned)...")
            frame_paths, gen_warnings = self._generate_pose_conditioned(
                image_rgb=image_rgb,
                pose_sequence=pose_sequence,
                intent=intent,
                seg=seg,
                cfg=cfg,
            )
            warnings.extend(gen_warnings)

            # ── 7. Temporal consistency + stabilisation ──────
            if len(frame_paths) > 1:
                self._report(6, "Applying temporal consistency...")
                frame_paths = self._apply_temporal_consistency(
                    frame_paths, seg, cfg
                )

                if cfg.stabilize:
                    self._report(7, "Stabilising output...")
                    frame_paths = self._get_stabilizer().stabilize_sequence(
                        frame_paths
                    )

            # ── 8. Assemble video ────────────────────────────
            self._report(8, "Assembling final video...")
            out_dir = Path(cfg.output_dir) / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = self._assemble_video(frame_paths, out_dir, cfg.fps)

            elapsed = time.time() - start
            logger.info(
                "[%s] Pipeline complete in %.1fs -> %s",
                job_id, elapsed, video_path,
            )

            return PipelineResult(
                success=True,
                video_path=video_path,
                frame_paths=frame_paths,
                segmentation=seg,
                intent=intent,
                motion_plan=plan,
                elapsed_seconds=elapsed,
                warnings=warnings,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "[%s] Pipeline failed after %.1fs: %s",
                job_id, elapsed, e, exc_info=True,
            )
            return PipelineResult(
                success=False,
                elapsed_seconds=elapsed,
                warnings=warnings,
                error=str(e),
            )

    # ── Stage 1: Segmentation with retry ──────────────────────

    def _segment_with_retry(
        self, image: np.ndarray, subject_hint: str
    ) -> SegmentationResult:
        """
        Segment using SAM 2 with retry, falling back to legacy segmenter.
        NEVER reduces motion intensity on low confidence.
        """
        # Try SAM 2 segmenter first
        try:
            sam2 = self._get_sam2_segmenter()
            multi_seg = sam2.segment(image, subject_hint=subject_hint)

            # Convert MultiObjectSegmentation -> SegmentationResult
            from .segmenter import SegmentationQuality
            if multi_seg.is_high_confidence:
                quality = SegmentationQuality.HIGH
            elif multi_seg.is_usable:
                quality = SegmentationQuality.MEDIUM
            else:
                quality = SegmentationQuality.LOW

            seg = SegmentationResult(
                mask=multi_seg.subject_mask,
                bbox=multi_seg.subject_bbox,
                label=multi_seg.subject_label,
                confidence=multi_seg.subject_confidence,
                quality=quality,
                object_area_ratio=float(multi_seg.subject_mask.sum())
                / max(multi_seg.subject_mask.size, 1),
                background_mask=multi_seg.background_mask,
                warnings=multi_seg.warnings,
            )
            # Attach retry metadata
            seg._retries_used = multi_seg.retries_used  # type: ignore
            return seg

        except Exception as e:
            logger.warning("SAM 2 segmenter failed (%s), using legacy", e)

        # Legacy fallback (ObjectSegmenter)
        seg = self._get_legacy_segmenter().segment(image, subject_hint)
        seg._retries_used = 0  # type: ignore
        return seg

    # ── Stage 3: Pose detection ───────────────────────────────

    def _detect_pose(
        self, image: np.ndarray, seg: SegmentationResult
    ):
        """Extract skeletal pose using DWPose/OpenPose/MediaPipe/heuristic."""
        estimator = self._get_pose_estimator()
        return estimator.estimate(
            image, mask=seg.mask, bbox=seg.bbox
        )

    # ── Stage 4: Motion synthesis ─────────────────────────────

    def _synthesize_motion(self, pose_result, intent, cfg):
        """Generate procedural pose sequence from initial pose + action."""
        from .motion_synthesizer import MotionConfig
        synth = self._get_motion_synthesizer()

        motion_cfg = MotionConfig(
            num_frames=cfg.num_frames,
            fps=cfg.fps,
            intensity=cfg.motion_intensity,
            seed=cfg.seed or 42,
        )

        return synth.synthesize(
            initial_pose=pose_result,
            verb=intent.action.value,
            speed=intent.speed.value,
            direction=intent.direction.value,
            config=motion_cfg,
        )

    # ── Stage 5: Pose-conditioned generation ──────────────────

    def _generate_pose_conditioned(
        self,
        image_rgb: np.ndarray,
        pose_sequence,
        intent: ActionIntent,
        seg: SegmentationResult,
        cfg: PipelineConfig,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate frames using AnimateDiff + ControlNet OpenPose.
        Falls back to legacy text-to-video if pose pipeline unavailable.
        """
        warnings: List[str] = []

        # Render pose maps from the synthesized pose sequence
        try:
            estimator = self._get_pose_estimator()
            pose_maps = [
                estimator.render_pose_map(p, cfg.width, cfg.height)
                for p in pose_sequence
            ]
        except Exception as e:
            logger.warning("Pose map rendering failed: %s", e)
            warnings.append(f"Pose map rendering failed: {e}")
            pose_maps = None

        # Try pose-conditioned pipeline
        if pose_maps:
            try:
                from .pose_conditioned_pipeline import (
                    GenerationConfig,
                    PoseConditionedPipeline,
                )
                pipe = self._get_pose_pipeline()
                gen_cfg = GenerationConfig(
                    width=cfg.width,
                    height=cfg.height,
                    num_frames=cfg.num_frames,
                    fps=cfg.fps,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    seed=cfg.seed or 42,
                )

                prompt = self._build_enriched_prompt(intent, None, cfg)
                result = pipe.generate(
                    source_image=image_rgb,
                    pose_maps=pose_maps,
                    prompt=prompt,
                    negative_prompt=cfg.negative_prompt,
                    config=gen_cfg,
                    mask=seg.mask,
                )

                if result.success and result.frames:
                    # Save frames to disk
                    out_dir = Path(cfg.output_dir) / "frames"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    frame_paths = self._save_frames(result.frames, out_dir)
                    warnings.extend(result.warnings)
                    return frame_paths, warnings

                warnings.extend(result.warnings)
                logger.warning(
                    "Pose pipeline returned no frames, falling back to legacy"
                )

            except Exception as e:
                logger.warning("Pose-conditioned pipeline failed: %s", e)
                warnings.append(
                    f"Pose-conditioned pipeline unavailable ({e}), "
                    "using legacy text-to-video"
                )

        # Legacy fallback: text-to-video with enriched prompt
        logger.info("Using legacy text-to-video generation")
        prompt = self._build_enriched_prompt(intent, None, cfg)
        frame_paths = self._generate_with_diffusion(
            image_rgb=image_rgb, prompt=prompt, cfg=cfg, seg=seg
        )
        warnings.append(
            "Used text-to-video fallback. Install controlnet_aux and "
            "AnimateDiff for pose-conditioned generation."
        )
        return frame_paths, warnings

    # ── Stage 6: Temporal consistency ─────────────────────────

    def _apply_temporal_consistency(
        self,
        frame_paths: List[str],
        seg: SegmentationResult,
        cfg: PipelineConfig,
    ) -> List[str]:
        """Apply post-generation temporal consistency pass."""
        try:
            from .temporal_consistency import (
                ConsistencyConfig,
                TemporalConsistencyPass,
            )

            # Load frames as arrays
            frames = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                frames.append(np.array(img))

            if len(frames) < 2:
                return frame_paths

            tc = self._get_temporal_consistency()
            tc_cfg = ConsistencyConfig(
                enable_background_lock=True,
                enable_flow_smoothing=True,
                enable_anti_ghosting=True,
                enable_temporal_denoise=True,
            )

            # Background from the original source (first frame as proxy)
            background = frames[0].copy()

            # Subject masks for background lock
            masks = [seg.mask] * len(frames)

            result = tc.apply(
                frames=frames,
                subject_masks=masks,
                background=background,
                config=tc_cfg,
            )

            if result.num_corrections > 0:
                logger.info(
                    "Temporal consistency: %d corrections, %d ghost fixes",
                    result.num_corrections, result.ghost_frames_fixed,
                )

            # Save corrected frames back
            out_dir = Path(frame_paths[0]).parent
            return self._save_frames(result.frames, out_dir)

        except Exception as e:
            logger.warning("Temporal consistency pass failed: %s", e)
            return frame_paths

    # ── Prompt enrichment ─────────────────────────────────────

    def _build_enriched_prompt(
        self,
        intent: ActionIntent,
        plan: Optional[MotionPlan],
        cfg: PipelineConfig,
    ) -> str:
        """
        Build a diffusion-model prompt enriched with motion cues.
        """
        parts: List[str] = []

        if cfg.prompt_prefix:
            parts.append(cfg.prompt_prefix)

        subject = intent.subject or "the subject"
        verb_map = {
            "walk": "walking",
            "run": "running",
            "jump": "jumping",
            "drive": "driving",
            "fly": "flying",
            "dance": "dancing",
            "swim": "swimming",
            "wave": "waving",
            "turn": "turning around",
            "sit": "sitting down",
            "stand": "standing up",
            "idle": "standing still",
        }
        action_ing = verb_map.get(intent.action.value, intent.action.value + "ing")
        parts.append(f"{subject} {action_ing}")

        if intent.direction.value not in ("in_place", "forward", "stationary"):
            parts.append(intent.direction.value.replace("_", " "))

        speed_adj = {
            "very_slow": "very slowly",
            "slow": "slowly",
            "medium": "",
            "fast": "quickly",
            "very_fast": "very fast",
        }
        spd = speed_adj.get(intent.speed.value, "")
        if spd:
            parts.append(spd)

        parts.append(
            "smooth continuous motion, high quality, photorealistic, "
            "natural movement, consistent lighting, temporal coherence"
        )

        if cfg.prompt_suffix:
            parts.append(cfg.prompt_suffix)

        enriched = ", ".join(p for p in parts if p)
        logger.info("Enriched prompt: %s", enriched)
        return enriched

    # ── Legacy diffusion generation (fallback) ────────────────

    def _generate_with_diffusion(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        cfg: PipelineConfig,
        seg: SegmentationResult,
    ) -> List[str]:
        """
        Generate video frames using the existing DiffusersPipeline.
        This is the legacy fallback when pose-conditioned pipeline
        is unavailable.
        """
        from ..runtime.diffusers_pipeline import DiffusersPipeline
        from ..runtime.model_registry import get_model

        spec = get_model(cfg.model_name)
        if spec is None:
            raise ValueError(f"Unknown model: {cfg.model_name}")

        gen_w, gen_h = spec.snap_dims(cfg.width, cfg.height)
        gen_frames = spec.snap_frames(cfg.num_frames)

        out_dir = Path(cfg.output_dir) / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._diffusion_pipeline is None or self._diffusion_pipeline.model_name != cfg.model_name:
            # Create a sub-progress callback that maps diffusers steps into
            # the generation stage (stage 5) of the overall pipeline progress.
            def _gen_sub_cb(step: int, total: int, msg: str):
                sub_frac = step / max(total, 1)
                self._report(5, msg, sub_fraction=sub_frac)

            self._diffusion_pipeline = DiffusersPipeline(
                model_name=cfg.model_name,
                progress_callback=_gen_sub_cb,
            )

        frame_paths = self._diffusion_pipeline.generate_frames(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=gen_w,
            height=gen_h,
            num_frames=gen_frames,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
            output_dir=str(out_dir),
        )

        return frame_paths

    # ── Frame I/O ─────────────────────────────────────────────

    @staticmethod
    def _save_frames(
        frames: List[np.ndarray], output_dir: Path
    ) -> List[str]:
        """Save numpy RGB frames to PNG files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame in enumerate(frames):
            path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(str(path))
            paths.append(str(path))
        return paths

    # ── Video assembly ────────────────────────────────────────

    def _assemble_video(
        self,
        frame_paths: List[str],
        output_dir: Path,
        fps: int,
    ) -> str:
        """Assemble frames into a video file (mp4)."""
        video_path = str(output_dir / "output.mp4")

        try:
            from ..video.assembler import VideoAssembler
            assembler = VideoAssembler()
            assembler.assemble(
                frame_paths=frame_paths,
                output_path=video_path,
                fps=fps,
            )
        except Exception as e:
            logger.warning("VideoAssembler failed (%s), trying ffmpeg", e)
            try:
                from ..video.ffmpeg_wrapper import FFmpegWrapper
                ff = FFmpegWrapper()
                ff.frames_to_video(
                    frame_dir=str(Path(frame_paths[0]).parent),
                    output_path=video_path,
                    fps=fps,
                )
            except Exception as e2:
                logger.error("FFmpeg fallback also failed: %s", e2)
                raise RuntimeError(
                    f"Could not assemble video: {e}; ffmpeg: {e2}"
                ) from e2

        logger.info("Video assembled: %s", video_path)
        return video_path

    # ── Image loading ─────────────────────────────────────────

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load an image file as RGB uint8 numpy array."""
        img = Image.open(path).convert("RGB")
        return np.array(img)

    @staticmethod
    def _extract_subject_hint(prompt: str) -> str:
        """Pull a rough subject noun from the action prompt."""
        lower = prompt.lower()
        for prefix in ("make the ", "make a ", "animate the ", "animate a ",
                       "let the ", "have the "):
            if lower.startswith(prefix):
                lower = lower[len(prefix):]
                break

        words = lower.split()
        hint_words = []
        _verbs = {"walk", "run", "jump", "drive", "fly", "dance",
                  "swim", "wave", "turn", "sit", "stand", "move",
                  "go", "spin", "lean", "crawl", "sprint", "jog",
                  "ride", "gallop", "trot", "glide", "soar", "dash"}
        for w in words:
            clean = w.strip(".,!?;:")
            if clean in _verbs:
                break
            hint_words.append(clean)

        return " ".join(hint_words[:3]) if hint_words else ""

    # ── Lazy sub-module getters ───────────────────────────────

    def _get_sam2_segmenter(self):
        if self._sam2_segmenter is None:
            from .sam2_segmenter import SAM2Segmenter
            self._sam2_segmenter = SAM2Segmenter()
        return self._sam2_segmenter

    def _get_legacy_segmenter(self) -> ObjectSegmenter:
        if self._legacy_segmenter is None:
            self._legacy_segmenter = ObjectSegmenter()
        return self._legacy_segmenter

    def _get_parser(self) -> ActionParser:
        if self._parser is None:
            self._parser = ActionParser()
        return self._parser

    def _get_planner(self) -> MotionPlanner:
        if self._planner is None:
            self._planner = MotionPlanner()
        return self._planner

    def _get_stabilizer(self) -> TemporalStabilizer:
        if self._stabilizer is None:
            self._stabilizer = TemporalStabilizer()
        return self._stabilizer

    def _get_pose_estimator(self):
        if self._pose_estimator is None:
            from .pose_estimator import PoseEstimator
            self._pose_estimator = PoseEstimator()
        return self._pose_estimator

    def _get_motion_synthesizer(self):
        if self._motion_synthesizer is None:
            from .motion_synthesizer import MotionSynthesizer
            self._motion_synthesizer = MotionSynthesizer()
        return self._motion_synthesizer

    def _get_pose_pipeline(self):
        if self._pose_pipeline is None:
            from .pose_conditioned_pipeline import PoseConditionedPipeline
            self._pose_pipeline = PoseConditionedPipeline()
        return self._pose_pipeline

    def _get_temporal_consistency(self):
        if self._temporal_consistency is None:
            from .temporal_consistency import TemporalConsistencyPass
            self._temporal_consistency = TemporalConsistencyPass()
        return self._temporal_consistency

    # ── Progress reporting ────────────────────────────────────

    def _report(self, stage: int, msg: str, sub_fraction: float = 0.0):
        """
        Report progress as an overall fraction (0.0 -- 1.0).

        Parameters
        ----------
        stage : int
            Pipeline stage index (0-8).
        msg : str
            Human-readable description of the current work.
        sub_fraction : float
            Progress *within* the current stage (0.0 -- 1.0).
            Used for sub-step reporting inside heavy stages.
        """
        if stage < len(self._STAGE_RANGES):
            start, end = self._STAGE_RANGES[stage]
            fraction = start + (end - start) * min(sub_fraction, 1.0)
        else:
            fraction = 1.0
        if self.progress_callback:
            self.progress_callback(fraction, msg)
        logger.info("Pipeline [%.0f%%] %s", fraction * 100, msg)
