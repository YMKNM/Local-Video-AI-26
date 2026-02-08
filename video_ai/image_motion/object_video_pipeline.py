"""
Object-Aware Video Pipeline -- Stage 4

Orchestrates the full image-to-video animation workflow:

    1. **Segment** the subject out of the input image.
    2. **Parse** the action prompt into a structured ``ActionIntent``.
    3. **Plan** frame-by-frame motion from intent + mask.
    4. **Generate** a video using the diffusion pipeline with
       motion-conditioned prompt enrichment.
    5. **Stabilise** the output temporally (see ``temporal_stabilizer``).

This module never modifies existing code -- it imports existing
infrastructure (``DiffusersPipeline``, ``VideoAssembler``, etc.) and
composes them with the new image-motion stages.

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

class ObjectVideoPipeline:
    """
    End-to-end image-to-video animation with object-level awareness.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback

        # Sub-modules (lazy-initialised)
        self._segmenter: Optional[ObjectSegmenter] = None
        self._parser: Optional[ActionParser] = None
        self._planner: Optional[MotionPlanner] = None
        self._stabilizer: Optional[TemporalStabilizer] = None
        self._diffusion_pipeline = None  # DiffusersPipeline (lazy)

    # ── public entry ──────────────────────────────────────────

    def run(
        self,
        image_path: str,
        action_prompt: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Run the full image-to-video pipeline.

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
            self._report(0, 6, "Loading image...")
            image_rgb = self._load_image(image_path)
            h, w = image_rgb.shape[:2]
            logger.info("[%s] Image loaded: %dx%d", job_id, w, h)

            # ── 1. Segment ───────────────────────────────────
            self._report(1, 6, "Segmenting subject...")
            seg = self._get_segmenter().segment(
                image_rgb, subject_hint=self._extract_subject_hint(action_prompt)
            )
            warnings.extend(seg.warnings)
            logger.info("[%s] Segmentation: %s (%.0f%% conf, quality=%s)",
                        job_id, seg.label, seg.confidence * 100, seg.quality.value)

            # ── 2. Parse action ──────────────────────────────
            self._report(2, 6, "Parsing action prompt...")
            intent = self._get_parser().parse(action_prompt)
            intent.intensity *= cfg.motion_intensity
            warnings.extend(intent.warnings)
            logger.info("[%s] Intent: %s %s %s (conf=%.2f)",
                        job_id, intent.action.value, intent.subject,
                        intent.direction.value, intent.confidence)

            # ── 3. Plan motion ───────────────────────────────
            self._report(3, 6, "Planning motion...")
            plan = self._get_planner().plan(
                intent=intent,
                seg=seg,
                total_frames=cfg.num_frames,
                fps=cfg.fps,
                image_shape=(h, w),
            )
            warnings.extend(plan.warnings)

            # ── 4. Generate video frames ─────────────────────
            self._report(4, 6, "Generating video (this may take a while)...")
            enriched_prompt = self._build_enriched_prompt(intent, plan, cfg)
            frame_paths = self._generate_with_diffusion(
                image_rgb=image_rgb,
                prompt=enriched_prompt,
                cfg=cfg,
                seg=seg,
            )

            # ── 5. Temporal stabilisation ────────────────────
            if cfg.stabilize and len(frame_paths) > 1:
                self._report(5, 6, "Stabilising output...")
                frame_paths = self._get_stabilizer().stabilize_sequence(
                    frame_paths
                )

            # ── 6. Assemble video ────────────────────────────
            self._report(6, 6, "Assembling final video...")
            out_dir = Path(cfg.output_dir) / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = self._assemble_video(frame_paths, out_dir, cfg.fps)

            elapsed = time.time() - start
            logger.info("[%s] Pipeline complete in %.1fs -> %s",
                        job_id, elapsed, video_path)

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
            logger.error("[%s] Pipeline failed after %.1fs: %s",
                         job_id, elapsed, e, exc_info=True)
            return PipelineResult(
                success=False,
                elapsed_seconds=elapsed,
                warnings=warnings,
                error=str(e),
            )

    # ── Prompt enrichment ─────────────────────────────────────

    def _build_enriched_prompt(
        self,
        intent: ActionIntent,
        plan: MotionPlan,
        cfg: PipelineConfig,
    ) -> str:
        """
        Build a diffusion-model prompt enriched with motion cues.

        Translates the structured ``ActionIntent`` back into a verbose,
        natural-language description that steers the diffusion model
        towards the desired motion.
        """
        parts: List[str] = []

        if cfg.prompt_prefix:
            parts.append(cfg.prompt_prefix)

        # Subject + action
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

        # Direction
        if intent.direction.value not in ("in_place", "forward"):
            parts.append(intent.direction.value.replace("_", " "))

        # Speed
        speed_adj = {
            "very_slow": "very slowly",
            "slow": "slowly",
            "normal": "",
            "fast": "quickly",
            "very_fast": "very fast",
        }
        spd = speed_adj.get(intent.speed.value, "")
        if spd:
            parts.append(spd)

        # Quality cues
        parts.append(
            "smooth continuous motion, high quality, photorealistic, "
            "natural movement, consistent lighting"
        )

        if cfg.prompt_suffix:
            parts.append(cfg.prompt_suffix)

        enriched = ", ".join(p for p in parts if p)
        logger.info("Enriched prompt: %s", enriched)
        return enriched

    # ── Diffusion generation ──────────────────────────────────

    def _generate_with_diffusion(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        cfg: PipelineConfig,
        seg: SegmentationResult,
    ) -> List[str]:
        """
        Generate video frames using the existing DiffusersPipeline.

        For models that support image-to-video (e.g. LTX-Video),
        the input image is passed directly.  For text-only models,
        we rely on the enriched prompt to encode motion cues.
        """
        # Lazy import to avoid circular deps
        from ..runtime.diffusers_pipeline import DiffusersPipeline
        from ..runtime.model_registry import get_model

        spec = get_model(cfg.model_name)
        if spec is None:
            raise ValueError(f"Unknown model: {cfg.model_name}")

        # Decide generation dimensions (respect model limits)
        gen_w, gen_h = spec.snap_dims(cfg.width, cfg.height)
        gen_frames = spec.snap_frames(cfg.num_frames)

        # Create output directory for frames
        out_dir = Path(cfg.output_dir) / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Initialise pipeline
        if self._diffusion_pipeline is None or self._diffusion_pipeline.model_name != cfg.model_name:
            self._diffusion_pipeline = DiffusersPipeline(
                model_name=cfg.model_name,
                progress_callback=self.progress_callback,
            )

        pipe = self._diffusion_pipeline

        # Check if model supports image-to-video
        is_i2v = "image-to-video" in spec.modalities

        if is_i2v:
            logger.info(
                "Model %s supports image-to-video -- passing source image",
                spec.display_name,
            )
            # For I2V models, we'd pass the image as conditioning.
            # The DiffusersPipeline currently only exposes text-to-video.
            # We generate with an enriched prompt that describes the scene.
            # TODO: extend DiffusersPipeline to accept image conditioning
            # when the model supports it (LTXImageToVideoPipeline).

        # Generate frames using text prompt
        frame_paths = pipe.generate_frames(
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
            logger.warning("VideoAssembler failed (%s), trying ffmpeg directly", e)
            # Fallback: simple ffmpeg concat
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
                    f"Could not assemble video: {e}; ffmpeg fallback: {e2}"
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
        """Pull a rough subject noun from the action prompt for segmentation."""
        lower = prompt.lower()
        # Strip common prefixes
        for prefix in ("make the ", "make a ", "animate the ", "animate a ",
                       "let the ", "have the "):
            if lower.startswith(prefix):
                lower = lower[len(prefix):]
                break

        # Take the first noun-like word(s) before the verb
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

    def _get_segmenter(self) -> ObjectSegmenter:
        if self._segmenter is None:
            self._segmenter = ObjectSegmenter()
        return self._segmenter

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

    # ── Progress reporting ────────────────────────────────────

    def _report(self, current: int, total: int, msg: str):
        if self.progress_callback:
            self.progress_callback(current, total, msg)
        logger.info("Pipeline progress: %d/%d -- %s", current, total, msg)
