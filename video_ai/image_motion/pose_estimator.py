"""
Pose Estimator — Skeletal Pose Extraction

Extracts full-body skeletal keypoints from a single image using
DWPose (preferred) or OpenPose via the ``controlnet_aux`` package.

The 18-point COCO body model provides the joint hierarchy needed for
biomechanically correct animation:

    hips → knees → ankles    (locomotion)
    shoulders → elbows → wrists  (arm swing / counterbalance)
    neck → nose / eyes / ears    (head stability)

This module outputs ``PoseResult`` which includes:
  - 2D keypoint coordinates with per-joint confidence
  - Limb connections and angles
  - Body orientation estimate (facing direction)
  - Subject height in pixels (for stride length calibration)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── COCO 18-point body model ─────────────────────────────────────────

class Joint(IntEnum):
    """COCO 18-keypoint body model indices."""
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    R_HIP = 8
    R_KNEE = 9
    R_ANKLE = 10
    L_HIP = 11
    L_KNEE = 12
    L_ANKLE = 13
    R_EYE = 14
    L_EYE = 15
    R_EAR = 16
    L_EAR = 17


# Skeleton connections for rendering and biomechanics
SKELETON_CONNECTIONS = [
    (Joint.NOSE, Joint.NECK),
    (Joint.NECK, Joint.R_SHOULDER), (Joint.R_SHOULDER, Joint.R_ELBOW),
    (Joint.R_ELBOW, Joint.R_WRIST),
    (Joint.NECK, Joint.L_SHOULDER), (Joint.L_SHOULDER, Joint.L_ELBOW),
    (Joint.L_ELBOW, Joint.L_WRIST),
    (Joint.NECK, Joint.R_HIP), (Joint.R_HIP, Joint.R_KNEE),
    (Joint.R_KNEE, Joint.R_ANKLE),
    (Joint.NECK, Joint.L_HIP), (Joint.L_HIP, Joint.L_KNEE),
    (Joint.L_KNEE, Joint.L_ANKLE),
    (Joint.NOSE, Joint.R_EYE), (Joint.NOSE, Joint.L_EYE),
    (Joint.R_EYE, Joint.R_EAR), (Joint.L_EYE, Joint.L_EAR),
]


@dataclass
class Keypoint:
    """Single joint keypoint."""
    x: float             # pixel x
    y: float             # pixel y
    confidence: float    # [0, 1]

    @property
    def is_visible(self) -> bool:
        return self.confidence >= 0.3

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PoseResult:
    """Complete skeletal pose for one detected person."""
    keypoints: List[Keypoint]           # 18 keypoints (COCO order)
    overall_confidence: float           # average of visible joints
    subject_height_px: float = 0.0      # nose-to-ankle distance in pixels
    facing_angle: float = 0.0           # degrees, 0 = facing camera
    hip_centre: Tuple[float, float] = (0.0, 0.0)  # midpoint of hips
    is_human: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def num_visible(self) -> int:
        return sum(1 for kp in self.keypoints if kp.is_visible)

    def get_joint(self, joint: Joint) -> Keypoint:
        return self.keypoints[joint.value]

    def limb_angle(self, j1: Joint, j2: Joint) -> float:
        """Angle of the limb from j1 to j2 (degrees, 0 = rightward)."""
        kp1, kp2 = self.get_joint(j1), self.get_joint(j2)
        if not (kp1.is_visible and kp2.is_visible):
            return 0.0
        dx = kp2.x - kp1.x
        dy = kp2.y - kp1.y
        return math.degrees(math.atan2(dy, dx))

    def to_openpose_array(self) -> np.ndarray:
        """Convert to (18, 3) array [x, y, conf] for ControlNet."""
        arr = np.zeros((18, 3), dtype=np.float32)
        for i, kp in enumerate(self.keypoints):
            arr[i] = [kp.x, kp.y, kp.confidence]
        return arr

    def to_dict(self) -> Dict:
        return {
            "num_visible": self.num_visible,
            "overall_confidence": round(self.overall_confidence, 3),
            "subject_height_px": round(self.subject_height_px, 1),
            "facing_angle": round(self.facing_angle, 1),
            "hip_centre": (round(self.hip_centre[0], 1),
                           round(self.hip_centre[1], 1)),
            "is_human": self.is_human,
            "warnings": self.warnings,
        }


class PoseEstimator:
    """
    Extract skeletal pose from a single image.

    Tries DWPose first (preferred for animation), falls back to OpenPose,
    then to a heuristic joint estimator based on the segmentation mask.

    Usage::

        estimator = PoseEstimator()
        pose = estimator.estimate(image_rgb, mask=subject_mask)
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dwpose = None
        self._openpose = None
        self._backend = None

    # ── Public API ────────────────────────────────────────────

    def estimate(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> PoseResult:
        """
        Estimate full-body pose from *image* (H, W, 3 RGB uint8).

        Parameters
        ----------
        image : np.ndarray
            Input image.
        mask : np.ndarray, optional
            Subject segmentation mask (H, W) float32.
            Used to crop & focus detection on the subject.
        bbox : tuple, optional
            (x1, y1, x2, y2) bounding box of subject.

        Returns
        -------
        PoseResult
        """
        # Try DWPose
        try:
            return self._estimate_dwpose(image, bbox)
        except Exception as e:
            logger.warning("DWPose unavailable (%s), trying OpenPose", e)

        # Try OpenPose
        try:
            return self._estimate_openpose(image, bbox)
        except Exception as e:
            logger.warning("OpenPose unavailable (%s), trying MediaPipe", e)

        # Try MediaPipe
        try:
            return self._estimate_mediapipe(image, bbox)
        except Exception as e:
            logger.warning("MediaPipe unavailable (%s), using heuristic", e)

        # Heuristic fallback from mask/bbox
        return self._estimate_from_mask(image, mask, bbox)

    # ── DWPose backend ────────────────────────────────────────

    def _estimate_dwpose(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> PoseResult:
        """Use DWPose via controlnet_aux for high-quality pose."""
        if self._dwpose is None:
            try:
                from controlnet_aux import DWposeDetector
                self._dwpose = DWposeDetector.from_pretrained(
                    "yzd-v/DWPose",
                    det_config="yolox_l_8xb8-300e_coco.py",
                    pose_config="dwpose-l_384x288.py",
                )
                self._backend = "dwpose"
                logger.info("DWPose loaded")
            except Exception as e:
                raise RuntimeError(f"DWPose not available: {e}")

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)

        # DWPose returns an image + optionally keypoints
        pose_output = self._dwpose(
            pil_img,
            detect_resolution=512,
            image_resolution=image.shape[1],
            output_type="np",
        )

        # DWPose from controlnet_aux returns drawn pose image
        # We need to extract keypoints — use the internal detector
        if hasattr(self._dwpose, 'pose_estimation'):
            # Some versions expose raw keypoints
            candidate, subset = self._dwpose.pose_estimation(image)
            if len(subset) > 0:
                return self._parse_openpose_output(candidate, subset[0], image.shape)

        # If we only got the pose image, do heuristic extraction
        logger.info("DWPose returned image only, extracting keypoints from pose map")
        return self._keypoints_from_pose_image(pose_output, image.shape)

    # ── OpenPose backend ──────────────────────────────────────

    def _estimate_openpose(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> PoseResult:
        """Use OpenPose via controlnet_aux."""
        if self._openpose is None:
            try:
                from controlnet_aux import OpenposeDetector
                self._openpose = OpenposeDetector.from_pretrained(
                    "lllyasviel/ControlNet"
                )
                self._backend = "openpose"
                logger.info("OpenPose loaded")
            except Exception as e:
                raise RuntimeError(f"OpenPose not available: {e}")

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)

        result = self._openpose(
            pil_img,
            detect_resolution=512,
            image_resolution=image.shape[1],
            hand_and_face=False,
            output_type="np",
        )

        # Extract keypoints from the pose renderer
        if hasattr(self._openpose, 'detect_poses'):
            poses = self._openpose.detect_poses(image)
            if poses:
                body = poses[0].body
                return self._parse_body_keypoints(body, image.shape)

        return self._keypoints_from_pose_image(result, image.shape)

    # ── MediaPipe backend ─────────────────────────────────────

    def _estimate_mediapipe(
        self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> PoseResult:
        """Use MediaPipe Pose for lightweight pose detection."""
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(image)

        if not results.pose_landmarks:
            raise RuntimeError("MediaPipe detected no pose")

        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # Map MediaPipe 33 landmarks → COCO 18
        # MediaPipe indices: 0=nose, 11=l_shoulder, 12=r_shoulder, etc.
        mp_to_coco = {
            Joint.NOSE: 0,
            Joint.L_EYE: 2, Joint.R_EYE: 5,
            Joint.L_EAR: 7, Joint.R_EAR: 8,
            Joint.L_SHOULDER: 11, Joint.R_SHOULDER: 12,
            Joint.L_ELBOW: 13, Joint.R_ELBOW: 14,
            Joint.L_WRIST: 15, Joint.R_WRIST: 16,
            Joint.L_HIP: 23, Joint.R_HIP: 24,
            Joint.L_KNEE: 25, Joint.R_KNEE: 26,
            Joint.L_ANKLE: 27, Joint.R_ANKLE: 28,
        }

        keypoints = []
        for j in Joint:
            mp_idx = mp_to_coco.get(j)
            if mp_idx is not None and mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                keypoints.append(Keypoint(
                    x=lm.x * w,
                    y=lm.y * h,
                    confidence=lm.visibility,
                ))
            elif j == Joint.NECK:
                # Synthesize neck as midpoint of shoulders
                ls = landmarks[mp_to_coco[Joint.L_SHOULDER]]
                rs = landmarks[mp_to_coco[Joint.R_SHOULDER]]
                keypoints.append(Keypoint(
                    x=(ls.x + rs.x) / 2 * w,
                    y=(ls.y + rs.y) / 2 * h,
                    confidence=min(ls.visibility, rs.visibility),
                ))
            else:
                keypoints.append(Keypoint(0, 0, 0))

        self._backend = "mediapipe"
        return self._finalize_pose(keypoints, image.shape)

    # ── Heuristic fallback ────────────────────────────────────

    def _estimate_from_mask(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> PoseResult:
        """
        Estimate approximate pose from mask/bbox geometry.

        Assumes an upright standing human with proportional limb positions.
        This is FAR less accurate but allows the pipeline to proceed.
        """
        h, w = image.shape[:2]
        warnings = [
            "Using heuristic pose estimation (no DWPose/OpenPose/MediaPipe). "
            "Install controlnet_aux or mediapipe for accurate skeletal poses."
        ]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
        elif mask is not None:
            ys, xs = np.where(mask > 0.5)
            if len(xs) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            else:
                x1, y1, x2, y2 = w // 4, h // 4, w * 3 // 4, h * 3 // 4
        else:
            x1, y1, x2, y2 = w // 4, h // 4, w * 3 // 4, h * 3 // 4

        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2

        # Standard human body proportions (head = 1/8 of height)
        head_top = y1
        head_h = bh * 0.125
        neck_y = head_top + head_h
        shoulder_y = neck_y + bh * 0.05
        hip_y = y1 + bh * 0.5
        knee_y = y1 + bh * 0.75
        ankle_y = y2

        shoulder_spread = bw * 0.35
        hip_spread = bw * 0.2

        # Build approximate keypoints
        kps = [Keypoint(0, 0, 0)] * 18
        kps[Joint.NOSE] = Keypoint(cx, head_top + head_h * 0.6, 0.5)
        kps[Joint.NECK] = Keypoint(cx, neck_y, 0.5)
        kps[Joint.R_SHOULDER] = Keypoint(cx - shoulder_spread, shoulder_y, 0.4)
        kps[Joint.R_ELBOW] = Keypoint(cx - shoulder_spread * 1.1, shoulder_y + bh * 0.15, 0.3)
        kps[Joint.R_WRIST] = Keypoint(cx - shoulder_spread * 1.0, hip_y, 0.3)
        kps[Joint.L_SHOULDER] = Keypoint(cx + shoulder_spread, shoulder_y, 0.4)
        kps[Joint.L_ELBOW] = Keypoint(cx + shoulder_spread * 1.1, shoulder_y + bh * 0.15, 0.3)
        kps[Joint.L_WRIST] = Keypoint(cx + shoulder_spread * 1.0, hip_y, 0.3)
        kps[Joint.R_HIP] = Keypoint(cx - hip_spread, hip_y, 0.4)
        kps[Joint.R_KNEE] = Keypoint(cx - hip_spread * 0.8, knee_y, 0.3)
        kps[Joint.R_ANKLE] = Keypoint(cx - hip_spread * 0.5, ankle_y, 0.3)
        kps[Joint.L_HIP] = Keypoint(cx + hip_spread, hip_y, 0.4)
        kps[Joint.L_KNEE] = Keypoint(cx + hip_spread * 0.8, knee_y, 0.3)
        kps[Joint.L_ANKLE] = Keypoint(cx + hip_spread * 0.5, ankle_y, 0.3)
        kps[Joint.R_EYE] = Keypoint(cx - bw * 0.05, head_top + head_h * 0.45, 0.3)
        kps[Joint.L_EYE] = Keypoint(cx + bw * 0.05, head_top + head_h * 0.45, 0.3)
        kps[Joint.R_EAR] = Keypoint(cx - bw * 0.1, head_top + head_h * 0.5, 0.2)
        kps[Joint.L_EAR] = Keypoint(cx + bw * 0.1, head_top + head_h * 0.5, 0.2)

        self._backend = "heuristic"
        result = self._finalize_pose(kps, image.shape)
        result.warnings = warnings
        return result

    # ── Shared helpers ────────────────────────────────────────

    def _parse_openpose_output(
        self, candidate: np.ndarray, subset: np.ndarray, shape: Tuple
    ) -> PoseResult:
        """Parse raw OpenPose output arrays into PoseResult."""
        keypoints = []
        for j in Joint:
            idx = int(subset[j.value])
            if idx >= 0 and idx < len(candidate):
                x, y, conf = candidate[idx][0], candidate[idx][1], candidate[idx][2]
                keypoints.append(Keypoint(float(x), float(y), float(conf)))
            else:
                keypoints.append(Keypoint(0, 0, 0))

        return self._finalize_pose(keypoints, shape)

    def _parse_body_keypoints(self, body, shape: Tuple) -> PoseResult:
        """Parse controlnet_aux body keypoints."""
        keypoints = []
        for j in Joint:
            if hasattr(body, 'keypoints') and j.value < len(body.keypoints):
                kp = body.keypoints[j.value]
                if kp is not None:
                    keypoints.append(Keypoint(float(kp.x), float(kp.y),
                                              float(getattr(kp, 'score', 0.8))))
                else:
                    keypoints.append(Keypoint(0, 0, 0))
            else:
                keypoints.append(Keypoint(0, 0, 0))

        return self._finalize_pose(keypoints, shape)

    def _keypoints_from_pose_image(
        self, pose_image: np.ndarray, shape: Tuple
    ) -> PoseResult:
        """Extract approximate keypoint positions from a rendered pose image."""
        # This is a fallback when we only get the drawn skeleton
        # Use colour detection to find joint positions
        warnings = ["Keypoints extracted from pose image (approximate)"]
        logger.warning("Extracting keypoints from pose image — accuracy limited")

        # For now, return heuristic from image centre
        h, w = shape[:2]
        return self._estimate_from_mask(
            np.zeros((h, w, 3), dtype=np.uint8), None, None
        )

    def _finalize_pose(
        self, keypoints: List[Keypoint], shape: Tuple
    ) -> PoseResult:
        """Compute derived metrics and build PoseResult."""
        h, w = shape[:2]

        # Overall confidence
        visible = [kp for kp in keypoints if kp.is_visible]
        overall_conf = (
            sum(kp.confidence for kp in visible) / len(visible)
            if visible else 0.0
        )

        # Subject height (nose to avg ankle)
        nose = keypoints[Joint.NOSE]
        r_ankle = keypoints[Joint.R_ANKLE]
        l_ankle = keypoints[Joint.L_ANKLE]
        height_px = 0.0
        if nose.is_visible:
            ankles = [a for a in [r_ankle, l_ankle] if a.is_visible]
            if ankles:
                avg_ankle_y = sum(a.y for a in ankles) / len(ankles)
                height_px = abs(avg_ankle_y - nose.y)

        # Hip centre
        r_hip = keypoints[Joint.R_HIP]
        l_hip = keypoints[Joint.L_HIP]
        if r_hip.is_visible and l_hip.is_visible:
            hip_cx = (r_hip.x + l_hip.x) / 2
            hip_cy = (r_hip.y + l_hip.y) / 2
        else:
            hip_cx, hip_cy = w / 2, h / 2

        # Facing angle estimate (from shoulder width perspective)
        r_sh = keypoints[Joint.R_SHOULDER]
        l_sh = keypoints[Joint.L_SHOULDER]
        facing = 0.0
        if r_sh.is_visible and l_sh.is_visible:
            sh_width = abs(l_sh.x - r_sh.x)
            max_width = w * 0.4  # expected full-frontal shoulder width
            ratio = min(sh_width / max(max_width, 1), 1.0)
            facing = math.degrees(math.acos(ratio))

        return PoseResult(
            keypoints=keypoints,
            overall_confidence=overall_conf,
            subject_height_px=height_px,
            facing_angle=facing,
            hip_centre=(hip_cx, hip_cy),
            is_human=overall_conf > 0.2,
        )

    def render_pose_map(
        self, pose: PoseResult, width: int, height: int
    ) -> np.ndarray:
        """
        Render an OpenPose-style skeleton image for ControlNet conditioning.

        Returns an RGB uint8 (H, W, 3) image with coloured limbs.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Limb colours (OpenPose standard)
        colours = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170), (255, 0, 85),
        ]

        try:
            import cv2

            # Draw limbs
            for idx, (j1, j2) in enumerate(SKELETON_CONNECTIONS):
                kp1 = pose.get_joint(j1)
                kp2 = pose.get_joint(j2)
                if kp1.is_visible and kp2.is_visible:
                    pt1 = (int(kp1.x), int(kp1.y))
                    pt2 = (int(kp2.x), int(kp2.y))
                    colour = colours[idx % len(colours)]
                    cv2.line(canvas, pt1, pt2, colour, 2)

            # Draw joints
            for i, kp in enumerate(pose.keypoints):
                if kp.is_visible:
                    centre = (int(kp.x), int(kp.y))
                    cv2.circle(canvas, centre, 4, colours[i % len(colours)], -1)

        except ImportError:
            # Pure numpy fallback — draw simple dots
            for kp in pose.keypoints:
                if kp.is_visible:
                    x, y = int(kp.x), int(kp.y)
                    if 0 <= x < width and 0 <= y < height:
                        r = 3
                        y1 = max(0, y - r)
                        y2 = min(height, y + r)
                        x1 = max(0, x - r)
                        x2 = min(width, x + r)
                        canvas[y1:y2, x1:x2] = (255, 255, 255)

        return canvas
