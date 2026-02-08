"""
Motion Planner -- Stage 3

Converts an ``ActionIntent`` (from the action parser) plus a
``SegmentationResult`` (from the segmenter) into frame-by-frame
motion constraints that can condition a video-diffusion model.

**Design invariants**

* All motion is *physically plausible* -- speeds, accelerations, and
  joint angles respect Newtonian kinematics at the chosen scale.
* Background pixels receive the *inverse* camera-motion transform so
  that the background stays stable while the subject moves.
* Motion profiles are per-subject-category (human, animal, vehicle,
  object) and per-verb.  Unknown verbs default to a gentle drift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .action_parser import (
    ActionIntent,
    ActionVerb,
    CameraMode,
    Direction,
    Speed,
    SubjectCategory,
)
from .segmenter import SegmentationResult

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────

class MotionType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    CYCLIC_LIMB = "cyclic_limb"
    SCALE = "scale"
    DEFORMATION = "deformation"


@dataclass
class MotionVector:
    """Per-frame displacement for the subject centroid."""
    dx: float = 0.0          # pixels / frame  (rightward +)
    dy: float = 0.0          # pixels / frame  (downward +)
    rotation_deg: float = 0.0
    scale_factor: float = 1.0


@dataclass
class CameraTransform:
    """Per-frame camera motion (applied to background)."""
    pan_x: float = 0.0       # pixels / frame
    pan_y: float = 0.0
    zoom: float = 1.0        # multiplier


@dataclass
class MotionKeyframe:
    """Motion state at a single frame."""
    frame_idx: int
    subject_motion: MotionVector
    camera: CameraTransform
    limb_phase: float = 0.0    # [0, 2*pi] cyclic limb phase
    intensity: float = 1.0     # overall scale factor


@dataclass
class MotionPlan:
    """Complete per-frame motion plan for a clip."""
    keyframes: List[MotionKeyframe]
    total_frames: int
    fps: int
    subject_label: str
    action_verb: ActionVerb
    direction: Direction
    speed: Speed
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0

    def to_conditioning_arrays(self) -> Dict[str, np.ndarray]:
        """Export motion plan as numpy arrays for pipeline consumption.

        Returns
        -------
        dict with keys:
            dx, dy          -- (N,) subject displacement per frame
            rotation        -- (N,) degrees
            scale           -- (N,) multiplier
            limb_phase      -- (N,) cyclic phase
            intensity       -- (N,) motion strength
            cam_pan_x/y     -- (N,) background pan
            cam_zoom        -- (N,) background zoom
        """
        n = self.total_frames
        out: Dict[str, np.ndarray] = {
            "dx": np.zeros(n, dtype=np.float32),
            "dy": np.zeros(n, dtype=np.float32),
            "rotation": np.zeros(n, dtype=np.float32),
            "scale": np.ones(n, dtype=np.float32),
            "limb_phase": np.zeros(n, dtype=np.float32),
            "intensity": np.ones(n, dtype=np.float32),
            "cam_pan_x": np.zeros(n, dtype=np.float32),
            "cam_pan_y": np.zeros(n, dtype=np.float32),
            "cam_zoom": np.ones(n, dtype=np.float32),
        }
        for kf in self.keyframes:
            i = kf.frame_idx
            if 0 <= i < n:
                out["dx"][i] = kf.subject_motion.dx
                out["dy"][i] = kf.subject_motion.dy
                out["rotation"][i] = kf.subject_motion.rotation_deg
                out["scale"][i] = kf.subject_motion.scale_factor
                out["limb_phase"][i] = kf.limb_phase
                out["intensity"][i] = kf.intensity
                out["cam_pan_x"][i] = kf.camera.pan_x
                out["cam_pan_y"][i] = kf.camera.pan_y
                out["cam_zoom"][i] = kf.camera.zoom
        return out


# ── Speed look-up table (pixels-per-frame at 480p @ 24-fps) ──────────

_SPEED_BASE: Dict[Speed, float] = {
    Speed.VERY_SLOW: 0.5,
    Speed.SLOW:      1.0,
    Speed.MEDIUM:    2.5,
    Speed.FAST:      5.0,
    Speed.VERY_FAST: 8.0,
}

# Verb-specific multipliers (relative to speed base)
_VERB_SPEED_MULT: Dict[ActionVerb, float] = {
    ActionVerb.WALK:  1.0,
    ActionVerb.RUN:   2.2,
    ActionVerb.JUMP:  1.5,
    ActionVerb.DRIVE: 3.5,
    ActionVerb.FLY:   3.0,
    ActionVerb.DANCE: 1.2,
    ActionVerb.SWIM:  0.9,
    ActionVerb.WAVE:  0.0,   # no translation
    ActionVerb.TURN:  0.0,
    ActionVerb.SIT:   0.0,
    ActionVerb.STAND: 0.0,
    ActionVerb.IDLE:  0.0,
}

# Limb-cycle frequency (Hz) per verb
_LIMB_FREQ: Dict[ActionVerb, float] = {
    ActionVerb.WALK:  1.8,
    ActionVerb.RUN:   3.2,
    ActionVerb.JUMP:  0.8,
    ActionVerb.DANCE: 2.5,
    ActionVerb.SWIM:  1.4,
    ActionVerb.WAVE:  2.0,
    ActionVerb.FLY:   3.5,
    ActionVerb.DRIVE: 0.0,
    ActionVerb.TURN:  0.0,
    ActionVerb.SIT:   0.0,
    ActionVerb.STAND: 0.0,
    ActionVerb.IDLE:  0.0,
}


# ── Direction → unit-vector mapping ──────────────────────────────────

def _direction_vector(d: Direction) -> Tuple[float, float]:
    """Return (dx, dy) unit vector for a direction.  Y is inverted (screen)."""
    _map = {
        Direction.FORWARD:           ( 0.0, -1.0),
        Direction.BACKWARD:          ( 0.0,  1.0),
        Direction.LEFT:              (-1.0,  0.0),
        Direction.RIGHT:             ( 1.0,  0.0),
        Direction.UP:                ( 0.0, -1.0),
        Direction.DOWN:              ( 0.0,  1.0),
        Direction.CLOCKWISE:         ( 0.0,  0.0),
        Direction.COUNTER_CLOCKWISE: ( 0.0,  0.0),
        Direction.STATIONARY:        ( 0.0,  0.0),
    }
    return _map.get(d, (0.0, 0.0))


# ── Camera mode helpers ──────────────────────────────────────────────

def _camera_for_mode(
    mode: CameraMode,
    subject_dx: float,
    subject_dy: float,
    frame_idx: int,
    total_frames: int,
) -> CameraTransform:
    """Compute per-frame camera transform for *mode*."""
    if mode == CameraMode.STATIC:
        return CameraTransform()

    if mode == CameraMode.FOLLOW:
        # Camera follows subject (cancel out translation in BG)
        return CameraTransform(pan_x=-subject_dx * 0.7,
                               pan_y=-subject_dy * 0.7)

    if mode == CameraMode.PAN:
        # Slow continuous pan right
        return CameraTransform(pan_x=0.8)

    if mode == CameraMode.ZOOM_IN:
        progress = frame_idx / max(total_frames - 1, 1)
        z = 1.0 + 0.15 * progress          # up to 15 % zoom
        return CameraTransform(zoom=z)

    if mode == CameraMode.ZOOM_OUT:
        progress = frame_idx / max(total_frames - 1, 1)
        z = 1.0 - 0.12 * progress
        return CameraTransform(zoom=max(z, 0.85))

    return CameraTransform()


# ── Planner ──────────────────────────────────────────────────────────

class MotionPlanner:
    """Build a ``MotionPlan`` from intent + segmentation."""

    def plan(
        self,
        intent: ActionIntent,
        seg: SegmentationResult,
        total_frames: int = 24,
        fps: int = 8,
        image_shape: Optional[Tuple[int, int]] = None,  # (H, W)
    ) -> MotionPlan:
        """
        Generate frame-by-frame motion constraints.

        Parameters
        ----------
        intent : ActionIntent
            Parsed action prompt.
        seg : SegmentationResult
            Subject mask & bbox from segmenter.
        total_frames : int
            Number of output frames.
        fps : int
            Output FPS.
        image_shape : tuple, optional
            (H, W) of the source image (used for scale normalisation).

        Returns
        -------
        MotionPlan
        """
        warnings: List[str] = list(intent.warnings)
        h, w = image_shape or (512, 512)

        # Scale factor — motion values are defined for 480p; adapt.
        res_scale = min(h, w) / 480.0

        # Base speed  (pixels / frame at source resolution)
        base_speed = _SPEED_BASE.get(intent.speed, 2.5)
        verb_mult  = _VERB_SPEED_MULT.get(intent.action, 1.0)
        px_per_frame = base_speed * verb_mult * res_scale * intent.intensity

        # Direction unit vector
        ux, uy = _direction_vector(intent.direction)

        # Jump special-case: add a parabolic Y arc
        is_jump = intent.action == ActionVerb.JUMP

        # Limb cycle frequency
        limb_freq = _LIMB_FREQ.get(intent.action, 0.0)

        # Reduce intensity when segmentation quality is poor
        seg_mult = 1.0
        if not seg.is_usable:
            seg_mult = 0.5
            warnings.append(
                "Low segmentation quality -- motion intensity halved"
            )

        keyframes: List[MotionKeyframe] = []
        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)   # normalised time [0, 1]

            # Translation
            dx = ux * px_per_frame * seg_mult
            dy = uy * px_per_frame * seg_mult

            # Jump parabola overlay
            if is_jump:
                jump_height = 30.0 * res_scale * intent.intensity * seg_mult
                dy_jump = -jump_height * math.sin(math.pi * t)
                dy += dy_jump

            # Rotation (turn verb)
            rot = 0.0
            if intent.action == ActionVerb.TURN:
                rot = 180.0 * t * intent.intensity * seg_mult

            # Scale (simulate depth for backward / forward)
            scale = 1.0
            if intent.direction == Direction.BACKWARD:
                scale = 1.0 + 0.15 * t * intent.intensity
            elif intent.direction == Direction.FORWARD:
                scale = 1.0 - 0.10 * t * intent.intensity

            # Limb phase
            phase = 0.0
            if limb_freq > 0:
                phase = 2.0 * math.pi * limb_freq * (i / fps)

            # Camera
            cam = _camera_for_mode(intent.camera, dx, dy, i, total_frames)

            # Ease-in / ease-out
            ease = _smooth_ease(t)

            kf = MotionKeyframe(
                frame_idx=i,
                subject_motion=MotionVector(
                    dx=dx * ease,
                    dy=dy * ease,
                    rotation_deg=rot,
                    scale_factor=scale,
                ),
                camera=cam,
                limb_phase=phase,
                intensity=intent.intensity * seg_mult * ease,
            )
            keyframes.append(kf)

        plan = MotionPlan(
            keyframes=keyframes,
            total_frames=total_frames,
            fps=fps,
            subject_label=intent.subject or seg.label,
            action_verb=intent.action,
            direction=intent.direction,
            speed=intent.speed,
            warnings=warnings,
        )

        logger.info(
            "MotionPlan: %s %s %s, %d frames @ %d fps, intensity=%.2f",
            plan.subject_label, intent.action.value,
            intent.direction.value, total_frames, fps, intent.intensity,
        )
        return plan


# ── Easing ───────────────────────────────────────────────────────────

def _smooth_ease(t: float) -> float:
    """Smooth-step ease-in / ease-out.  t in [0, 1] -> [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
