"""
Motion Synthesizer — Procedural Biomechanical Pose Sequences

Generates anatomically plausible per-frame skeletal poses for actions
like walking, running, jumping, dancing, etc.

**Design principles:**

1. Physics-based: inertia, momentum, ground contact, gravity
2. Biomechanically correct: joint angle limits, limb counterswing,
   hip-shoulder counter-rotation
3. Smooth: cubic interpolation between keyframes, ease-in/out for
   acceleration/deceleration phases
4. Deterministic per seed: same (action, num_frames, seed) = same output

Takes as input:
  - Initial ``PoseResult`` from pose_estimator
  - ``ActionIntent`` from action_parser (verb, direction, speed)
  - Frame count and FPS

Outputs:
  - List of ``PoseResult`` (one per frame) representing the full motion
  - Each PoseResult has modified keypoint positions from the initial pose
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose_estimator import Joint, Keypoint, PoseResult

logger = logging.getLogger(__name__)


# ── Physics constants ────────────────────────────────────────────

GRAVITY = 9.81           # m/s^2
DEFAULT_STRIDE_RATIO = 0.4  # stride length relative to subject height


# ── Cycle definitions (phase-based) ─────────────────────────────

@dataclass
class LimbCycleDef:
    """
    Defines sinusoidal oscillation for one joint in a locomotion cycle.

    phase_offset : fraction of cycle [0, 1)
    amplitude    : max displacement in pixels (relative to subject height)
    axis         : 'x' or 'y'
    """
    joint: Joint
    phase_offset: float
    amplitude: float     # fraction of subject height
    axis: str = "y"      # 'x' or 'y'


# Walk cycle limb phases — bilateral counter-motion
WALK_CYCLE: List[LimbCycleDef] = [
    # Right leg: hip forward/back, knee lift, ankle ground contact
    LimbCycleDef(Joint.R_HIP,   0.0,   0.02, "x"),
    LimbCycleDef(Joint.R_HIP,   0.0,   0.01, "y"),
    LimbCycleDef(Joint.R_KNEE,  0.1,   0.05, "y"),
    LimbCycleDef(Joint.R_ANKLE, 0.15,  0.04, "y"),

    # Left leg: 180 degrees out of phase
    LimbCycleDef(Joint.L_HIP,   0.5,   0.02, "x"),
    LimbCycleDef(Joint.L_HIP,   0.5,   0.01, "y"),
    LimbCycleDef(Joint.L_KNEE,  0.6,   0.05, "y"),
    LimbCycleDef(Joint.L_ANKLE, 0.65,  0.04, "y"),

    # Arms counter-swing: opposite to corresponding leg
    LimbCycleDef(Joint.R_ELBOW, 0.5,   0.03, "x"),
    LimbCycleDef(Joint.R_WRIST, 0.55,  0.04, "x"),
    LimbCycleDef(Joint.L_ELBOW, 0.0,   0.03, "x"),
    LimbCycleDef(Joint.L_WRIST, 0.05,  0.04, "x"),

    # Shoulder bob (vertical)
    LimbCycleDef(Joint.R_SHOULDER, 0.25, 0.008, "y"),
    LimbCycleDef(Joint.L_SHOULDER, 0.75, 0.008, "y"),

    # Head bob (subtle)
    LimbCycleDef(Joint.NOSE,  0.25,  0.005, "y"),
    LimbCycleDef(Joint.NECK,  0.25,  0.003, "y"),
]

# Run cycle — more aggressive, higher amplitude
RUN_CYCLE: List[LimbCycleDef] = [
    # Legs: greater knee lift, faster stride
    LimbCycleDef(Joint.R_HIP,   0.0,   0.04, "x"),
    LimbCycleDef(Joint.R_HIP,   0.0,   0.02, "y"),
    LimbCycleDef(Joint.R_KNEE,  0.1,   0.10, "y"),
    LimbCycleDef(Joint.R_ANKLE, 0.15,  0.08, "y"),

    LimbCycleDef(Joint.L_HIP,   0.5,   0.04, "x"),
    LimbCycleDef(Joint.L_HIP,   0.5,   0.02, "y"),
    LimbCycleDef(Joint.L_KNEE,  0.6,   0.10, "y"),
    LimbCycleDef(Joint.L_ANKLE, 0.65,  0.08, "y"),

    # Arms: forceful drive
    LimbCycleDef(Joint.R_ELBOW, 0.5,   0.06, "x"),
    LimbCycleDef(Joint.R_ELBOW, 0.5,   0.03, "y"),
    LimbCycleDef(Joint.R_WRIST, 0.55,  0.08, "x"),
    LimbCycleDef(Joint.L_ELBOW, 0.0,   0.06, "x"),
    LimbCycleDef(Joint.L_ELBOW, 0.0,   0.03, "y"),
    LimbCycleDef(Joint.L_WRIST, 0.05,  0.08, "x"),

    # Torso lean forward (running posture)
    LimbCycleDef(Joint.NECK,  0.0,  0.010, "x"),
    LimbCycleDef(Joint.NOSE,  0.0,  0.012, "x"),

    # Shoulder/head bob (more pronounced)
    LimbCycleDef(Joint.R_SHOULDER, 0.25, 0.015, "y"),
    LimbCycleDef(Joint.L_SHOULDER, 0.75, 0.015, "y"),
    LimbCycleDef(Joint.NOSE,  0.25,  0.010, "y"),
]

# Dance cycle — lateral sway, arm raise, hip sway
DANCE_CYCLE: List[LimbCycleDef] = [
    LimbCycleDef(Joint.R_HIP,   0.0,   0.03, "x"),
    LimbCycleDef(Joint.L_HIP,   0.0,   0.03, "x"),
    LimbCycleDef(Joint.R_SHOULDER, 0.25, 0.04, "x"),
    LimbCycleDef(Joint.L_SHOULDER, 0.25, 0.04, "x"),
    LimbCycleDef(Joint.R_WRIST, 0.0,   0.10, "y"),
    LimbCycleDef(Joint.L_WRIST, 0.5,   0.10, "y"),
    LimbCycleDef(Joint.R_ELBOW, 0.0,   0.06, "y"),
    LimbCycleDef(Joint.L_ELBOW, 0.5,   0.06, "y"),
    LimbCycleDef(Joint.R_KNEE,  0.0,   0.02, "y"),
    LimbCycleDef(Joint.L_KNEE,  0.5,   0.02, "y"),
    LimbCycleDef(Joint.NOSE,    0.25,  0.02, "x"),
    LimbCycleDef(Joint.NECK,    0.25,  0.015, "x"),
]

# Wave cycle — one arm oscillation
WAVE_CYCLE: List[LimbCycleDef] = [
    LimbCycleDef(Joint.R_WRIST,  0.0,  0.08, "x"),
    LimbCycleDef(Joint.R_WRIST,  0.25, 0.04, "y"),
    LimbCycleDef(Joint.R_ELBOW,  0.0,  0.04, "x"),
    LimbCycleDef(Joint.R_SHOULDER, 0.0, 0.01, "y"),
]

# Map verb strings to cycles
VERB_CYCLES: Dict[str, List[LimbCycleDef]] = {
    "walk": WALK_CYCLE,
    "run": RUN_CYCLE,
    "dance": DANCE_CYCLE,
    "wave": WAVE_CYCLE,
}

# Cycle frequency multipliers by speed
SPEED_FREQ: Dict[str, float] = {
    "very_slow": 0.3,
    "slow": 0.6,
    "medium": 1.0,
    "fast": 1.5,
    "very_fast": 2.0,
}

# Motion intensity multipliers by speed
SPEED_INTENSITY: Dict[str, float] = {
    "very_slow": 0.5,
    "slow": 0.7,
    "medium": 1.0,
    "fast": 1.3,
    "very_fast": 1.6,
}


@dataclass
class MotionConfig:
    """Configuration for motion synthesis."""
    num_frames: int = 16
    fps: float = 8.0
    intensity: float = 1.0
    seed: int = 42
    # Translational motion
    enable_hip_translation: bool = True
    hip_translation_px_per_frame: float = 0.0  # auto-computed
    # Physics
    enable_inertia: bool = True
    enable_ground_contact: bool = True
    # Joint limits
    enable_joint_limits: bool = True


class MotionSynthesizer:
    """
    Generate procedural skeletal pose sequences for video animation.

    Usage::

        synth = MotionSynthesizer()
        frames = synth.synthesize(initial_pose, "run", "fast", config)
        # frames is List[PoseResult] with one entry per frame
    """

    def __init__(self):
        self._rng = np.random.default_rng(42)

    def synthesize(
        self,
        initial_pose: PoseResult,
        verb: str,
        speed: str = "medium",
        direction: str = "forward",
        config: Optional[MotionConfig] = None,
    ) -> List[PoseResult]:
        """
        Generate a full frame sequence of poses.

        Parameters
        ----------
        initial_pose : PoseResult
            Starting skeletal pose from PoseEstimator.
        verb : str
            Action verb (walk, run, jump, dance, wave, etc.)
        speed : str
            Speed enum name (very_slow / slow / medium / fast / very_fast)
        direction : str
            Movement direction (forward / backward / left / right)
        config : MotionConfig, optional

        Returns
        -------
        List[PoseResult]
            Per-frame skeletal poses. Length == config.num_frames.
        """
        if config is None:
            config = MotionConfig()

        self._rng = np.random.default_rng(config.seed)
        verb_lower = verb.lower().strip()
        speed_lower = speed.lower().strip()

        logger.info(
            "Synthesizing %d frames: verb=%s speed=%s dir=%s",
            config.num_frames, verb_lower, speed_lower, direction,
        )

        # Special handling for jump
        if verb_lower == "jump":
            return self._synthesize_jump(initial_pose, speed_lower, config)

        # Cyclic verb
        cycle = VERB_CYCLES.get(verb_lower)
        if cycle is None:
            # Default to walk for locomotion verbs, idle for others
            if verb_lower in ("drive", "fly", "swim"):
                cycle = WALK_CYCLE
            else:
                cycle = WALK_CYCLE  # fallback
                logger.info("No specific cycle for '%s', using walk cycle", verb_lower)

        return self._synthesize_cyclic(
            initial_pose, cycle, speed_lower, direction, config
        )

    # ── Cyclic motion (walk, run, dance, wave) ────────────────

    def _synthesize_cyclic(
        self,
        initial: PoseResult,
        cycle: List[LimbCycleDef],
        speed: str,
        direction: str,
        config: MotionConfig,
    ) -> List[PoseResult]:
        """Generate cyclic locomotion/gesture poses."""
        freq_mult = SPEED_FREQ.get(speed, 1.0)
        int_mult = SPEED_INTENSITY.get(speed, 1.0) * config.intensity
        h_ref = max(initial.subject_height_px, 100.0)

        # Hip translation for locomotion
        dir_vec = self._direction_vector(direction)
        translate_per_frame = h_ref * DEFAULT_STRIDE_RATIO * freq_mult / config.fps
        if config.hip_translation_px_per_frame > 0:
            translate_per_frame = config.hip_translation_px_per_frame

        frames: List[PoseResult] = []
        cycle_duration = config.fps / max(freq_mult, 0.1)  # frames per cycle

        for f in range(config.num_frames):
            phase = (f / max(cycle_duration, 1.0)) % 1.0
            t = f / max(config.num_frames - 1, 1)  # normalised time [0, 1]

            # Ease-in at start, ease-out at end
            ease = self._ease_in_out(t)

            new_kps = self._apply_cycle(
                initial.keypoints, cycle, phase, h_ref, int_mult * ease
            )

            # Hip translation
            if config.enable_hip_translation and dir_vec != (0, 0):
                tx = dir_vec[0] * translate_per_frame * f * ease
                ty = dir_vec[1] * translate_per_frame * f * ease
                new_kps = self._translate_all(new_kps, tx, ty)

            # Ground contact constraint
            if config.enable_ground_contact:
                new_kps = self._enforce_ground_contact(
                    new_kps, initial.keypoints
                )

            # Joint angle limits
            if config.enable_joint_limits:
                new_kps = self._enforce_joint_limits(new_kps)

            # Add micro-noise for naturalness
            new_kps = self._add_micro_noise(new_kps, h_ref * 0.002)

            frames.append(self._make_frame_pose(new_kps, initial))

        return frames

    # ── Jump motion ───────────────────────────────────────────

    def _synthesize_jump(
        self,
        initial: PoseResult,
        speed: str,
        config: MotionConfig,
    ) -> List[PoseResult]:
        """
        Jump sequence with biomechanically accurate phases:
        1. Preparation (crouch) — bend knees, lower CoM
        2. Launch — explosive extension, arms drive upward
        3. Flight — parabolic arc, limbs tuck
        4. Landing — knees absorb, settle
        """
        h_ref = max(initial.subject_height_px, 100.0)
        int_mult = SPEED_INTENSITY.get(speed, 1.0) * config.intensity
        n = config.num_frames

        # Phase boundaries (fraction of total frames)
        prep_end = 0.2
        launch_end = 0.35
        peak = 0.55
        land_start = 0.75

        jump_height = h_ref * 0.25 * int_mult  # max vertical displacement

        frames: List[PoseResult] = []

        for f in range(n):
            t = f / max(n - 1, 1)
            new_kps = [Keypoint(kp.x, kp.y, kp.confidence)
                       for kp in initial.keypoints]

            if t <= prep_end:
                # Preparation: crouch
                crouch_t = t / prep_end
                crouch_amt = math.sin(crouch_t * math.pi / 2)
                self._apply_crouch(new_kps, h_ref, crouch_amt * 0.1 * int_mult)

            elif t <= launch_end:
                # Launch: explosive extension
                launch_t = (t - prep_end) / (launch_end - prep_end)
                rise = math.sin(launch_t * math.pi / 2)
                self._apply_extension(new_kps, h_ref, rise * 0.05 * int_mult)
                # Vertical displacement starts
                dy = -jump_height * rise * 0.3
                new_kps = self._translate_all(new_kps, 0, dy)

            elif t <= land_start:
                # Flight: parabolic arc
                flight_t = (t - launch_end) / (land_start - launch_end)
                arc = 1.0 - (2 * flight_t - 1) ** 2  # parabola peaks at 0.5
                dy = -jump_height * arc
                new_kps = self._translate_all(new_kps, 0, dy)
                # Tuck legs slightly
                self._apply_tuck(new_kps, h_ref, 0.03 * int_mult * arc)

            else:
                # Landing: absorb
                land_t = (t - land_start) / (1.0 - land_start)
                absorb = 1.0 - land_t
                dy = -jump_height * 0.05 * absorb
                new_kps = self._translate_all(new_kps, 0, dy)
                self._apply_crouch(new_kps, h_ref, absorb * 0.08 * int_mult)

            if config.enable_joint_limits:
                new_kps = self._enforce_joint_limits(new_kps)
            new_kps = self._add_micro_noise(new_kps, h_ref * 0.001)

            frames.append(self._make_frame_pose(new_kps, initial))

        return frames

    # ── Pose manipulation helpers ─────────────────────────────

    def _apply_cycle(
        self,
        base_kps: List[Keypoint],
        cycle: List[LimbCycleDef],
        phase: float,
        h_ref: float,
        intensity: float,
    ) -> List[Keypoint]:
        """Apply sinusoidal cycle offsets to base keypoints."""
        kps = [Keypoint(kp.x, kp.y, kp.confidence) for kp in base_kps]

        for c in cycle:
            joint_idx = c.joint.value
            if joint_idx >= len(kps):
                continue

            angle = 2 * math.pi * (phase + c.phase_offset)
            displacement = math.sin(angle) * c.amplitude * h_ref * intensity

            if c.axis == "x":
                kps[joint_idx] = Keypoint(
                    kps[joint_idx].x + displacement,
                    kps[joint_idx].y,
                    kps[joint_idx].confidence,
                )
            else:
                kps[joint_idx] = Keypoint(
                    kps[joint_idx].x,
                    kps[joint_idx].y + displacement,
                    kps[joint_idx].confidence,
                )

        return kps

    def _apply_crouch(
        self, kps: List[Keypoint], h_ref: float, amount: float
    ) -> None:
        """Bend knees and lower hips (in-place)."""
        for j in [Joint.R_HIP, Joint.L_HIP]:
            kps[j.value] = Keypoint(
                kps[j.value].x,
                kps[j.value].y + h_ref * amount,
                kps[j.value].confidence,
            )
        for j in [Joint.R_KNEE, Joint.L_KNEE]:
            kps[j.value] = Keypoint(
                kps[j.value].x,
                kps[j.value].y + h_ref * amount * 0.5,
                kps[j.value].confidence,
            )

    def _apply_extension(
        self, kps: List[Keypoint], h_ref: float, amount: float
    ) -> None:
        """Extend body upward (launch phase)."""
        # Raise arms
        for j in [Joint.R_WRIST, Joint.L_WRIST, Joint.R_ELBOW, Joint.L_ELBOW]:
            kps[j.value] = Keypoint(
                kps[j.value].x,
                kps[j.value].y - h_ref * amount,
                kps[j.value].confidence,
            )

    def _apply_tuck(
        self, kps: List[Keypoint], h_ref: float, amount: float
    ) -> None:
        """Tuck legs during flight."""
        for j in [Joint.R_KNEE, Joint.L_KNEE]:
            kps[j.value] = Keypoint(
                kps[j.value].x,
                kps[j.value].y - h_ref * amount,
                kps[j.value].confidence,
            )

    def _translate_all(
        self, kps: List[Keypoint], dx: float, dy: float
    ) -> List[Keypoint]:
        """Translate all visible keypoints."""
        return [
            Keypoint(kp.x + dx, kp.y + dy, kp.confidence)
            if kp.is_visible else kp
            for kp in kps
        ]

    def _enforce_ground_contact(
        self,
        kps: List[Keypoint],
        base_kps: List[Keypoint],
    ) -> List[Keypoint]:
        """Ensure feet don't go below original ankle height."""
        max_ankle_y = max(
            base_kps[Joint.R_ANKLE.value].y,
            base_kps[Joint.L_ANKLE.value].y,
        )
        for j in [Joint.R_ANKLE, Joint.L_ANKLE]:
            kp = kps[j.value]
            if kp.is_visible and kp.y > max_ankle_y:
                kps[j.value] = Keypoint(kp.x, max_ankle_y, kp.confidence)
        return kps

    def _enforce_joint_limits(
        self, kps: List[Keypoint]
    ) -> List[Keypoint]:
        """
        Constrain joint angles to anatomically plausible ranges.

        This prevents hyperextension / impossible poses.
        """
        # Knee cannot extend past 180 degrees (no hyperextension)
        for hip_j, knee_j, ankle_j in [
            (Joint.R_HIP, Joint.R_KNEE, Joint.R_ANKLE),
            (Joint.L_HIP, Joint.L_KNEE, Joint.L_ANKLE),
        ]:
            hip = kps[hip_j.value]
            knee = kps[knee_j.value]
            ankle = kps[ankle_j.value]

            if hip.is_visible and knee.is_visible and ankle.is_visible:
                angle = self._joint_angle(hip, knee, ankle)
                # Knee should be between 30 and 180 degrees
                if angle < 30:
                    # Pull ankle to minimum bend position
                    kps[ankle_j.value] = self._constrain_angle(
                        hip, knee, ankle, 30
                    )

        # Elbow: 10 to 180 degrees
        for shoulder_j, elbow_j, wrist_j in [
            (Joint.R_SHOULDER, Joint.R_ELBOW, Joint.R_WRIST),
            (Joint.L_SHOULDER, Joint.L_ELBOW, Joint.L_WRIST),
        ]:
            sh = kps[shoulder_j.value]
            el = kps[elbow_j.value]
            wr = kps[wrist_j.value]

            if sh.is_visible and el.is_visible and wr.is_visible:
                angle = self._joint_angle(sh, el, wr)
                if angle < 10:
                    kps[wrist_j.value] = self._constrain_angle(
                        sh, el, wr, 10
                    )

        return kps

    def _add_micro_noise(
        self, kps: List[Keypoint], magnitude: float
    ) -> List[Keypoint]:
        """Add small per-joint noise for natural movement feel."""
        return [
            Keypoint(
                kp.x + self._rng.normal(0, magnitude) if kp.is_visible else 0,
                kp.y + self._rng.normal(0, magnitude) if kp.is_visible else 0,
                kp.confidence,
            )
            for kp in kps
        ]

    def _make_frame_pose(
        self, kps: List[Keypoint], template: PoseResult
    ) -> PoseResult:
        """Build PoseResult from modified keypoints, inheriting template metadata."""
        visible = [kp for kp in kps if kp.is_visible]
        overall_conf = (
            sum(kp.confidence for kp in visible) / len(visible)
            if visible else 0.0
        )

        r_hip = kps[Joint.R_HIP.value]
        l_hip = kps[Joint.L_HIP.value]
        if r_hip.is_visible and l_hip.is_visible:
            hip_cx = (r_hip.x + l_hip.x) / 2
            hip_cy = (r_hip.y + l_hip.y) / 2
        else:
            hip_cx, hip_cy = template.hip_centre

        nose = kps[Joint.NOSE.value]
        r_ankle = kps[Joint.R_ANKLE.value]
        l_ankle = kps[Joint.L_ANKLE.value]
        height_px = 0.0
        if nose.is_visible:
            ankles_vis = [a for a in [r_ankle, l_ankle] if a.is_visible]
            if ankles_vis:
                avg_y = sum(a.y for a in ankles_vis) / len(ankles_vis)
                height_px = abs(avg_y - nose.y)

        return PoseResult(
            keypoints=kps,
            overall_confidence=overall_conf,
            subject_height_px=height_px or template.subject_height_px,
            facing_angle=template.facing_angle,
            hip_centre=(hip_cx, hip_cy),
            is_human=template.is_human,
        )

    # ── Math utilities ────────────────────────────────────────

    @staticmethod
    def _direction_vector(direction: str) -> Tuple[float, float]:
        """Map direction string to normalised (dx, dy) vector."""
        d = direction.lower().strip()
        _map = {
            "forward": (0, 0),       # no 2D translation for depth
            "backward": (0, 0),
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1),
            "stationary": (0, 0),
        }
        return _map.get(d, (0, 0))

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Smoothstep ease-in-out: 0 at t=0, 1 at t=0.15, 1 through t=0.85, 0 at t=1."""
        if t < 0.15:
            return t / 0.15
        elif t > 0.85:
            return (1.0 - t) / 0.15
        return 1.0

    @staticmethod
    def _joint_angle(
        parent: Keypoint, joint: Keypoint, child: Keypoint
    ) -> float:
        """Angle at *joint* between parent→joint and joint→child (degrees)."""
        v1 = (parent.x - joint.x, parent.y - joint.y)
        v2 = (child.x - joint.x, child.y - joint.y)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _constrain_angle(
        parent: Keypoint, joint: Keypoint, child: Keypoint,
        min_degrees: float,
    ) -> Keypoint:
        """Return adjusted child position so joint angle >= min_degrees."""
        # Direction from joint to child
        dx = child.x - joint.x
        dy = child.y - joint.y
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            return child

        # Direction from joint to parent
        pdx = parent.x - joint.x
        pdy = parent.y - joint.y
        parent_angle = math.atan2(pdy, pdx)

        # Rotate child to ensure minimum angle
        child_angle = math.atan2(dy, dx)
        current_angle = abs(child_angle - parent_angle)

        if math.degrees(current_angle) < min_degrees:
            target_rad = math.radians(min_degrees)
            new_angle = parent_angle + target_rad
            new_x = joint.x + length * math.cos(new_angle)
            new_y = joint.y + length * math.sin(new_angle)
            return Keypoint(new_x, new_y, child.confidence)

        return child
