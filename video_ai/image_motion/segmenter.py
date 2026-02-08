"""
Object Segmenter — Stage 1

Detects and segments the main subject in a single image using
text-guided or automatic saliency detection.

Two backends are supported (auto-selected at runtime):

 1. **SAM-based** — uses ``segment-anything`` + ``grounding-dino`` for
    text-guided detection.  High accuracy, requires GPU.
 2. **Fallback** — uses OpenCV GrabCut + saliency maps.  Works on CPU,
    lower accuracy.

All heavy model imports are deferred so that importing this module is
fast and safe even when the ML libraries are not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────

class SegmentationQuality(Enum):
    HIGH = "high"           # clean mask, single object, high confidence
    MEDIUM = "medium"       # usable but may have artefacts
    LOW = "low"             # fallback / guessed
    FAILED = "failed"       # could not segment


@dataclass
class SegmentationResult:
    """Result of object detection + segmentation."""

    mask: np.ndarray                        # (H, W) float32 in [0, 1]
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2)
    label: str                              # detected class label
    confidence: float                       # detection confidence
    quality: SegmentationQuality
    object_area_ratio: float = 0.0          # mask area / image area
    background_mask: Optional[np.ndarray] = None  # inverse mask
    warnings: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.quality in (SegmentationQuality.HIGH,
                                SegmentationQuality.MEDIUM)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "quality": self.quality.value,
            "bbox": list(self.bbox),
            "object_area_ratio": self.object_area_ratio,
            "warnings": self.warnings,
        }


# ── Quality thresholds ────────────────────────────────────────────────

MIN_OBJECT_AREA_RATIO = 0.005      # object must be ≥ 0.5 % of image
MAX_OBJECT_AREA_RATIO = 0.95       # … and ≤ 95 %
MIN_DETECTION_CONFIDENCE = 0.25


# ── Segmenter ─────────────────────────────────────────────────────────

class ObjectSegmenter:
    """
    Detects and segments the primary object in an image.

    Usage::

        seg = ObjectSegmenter()
        result = seg.segment(image_rgb, subject_hint="boy")
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sam_loaded = False
        self._sam_model = None
        self._sam_predictor = None

    # ── public API ────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        subject_hint: str = "",
    ) -> SegmentationResult:
        """
        Segment the main subject in *image* (H, W, 3 uint8 RGB).

        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format, dtype uint8.
        subject_hint : str
            Optional text hint (e.g. "boy", "car") that guides detection.

        Returns
        -------
        SegmentationResult
        """
        h, w = image.shape[:2]

        # Try SAM-based segmentation first
        try:
            return self._segment_sam(image, subject_hint)
        except Exception as e:
            logger.warning("SAM segmentation unavailable (%s), using fallback", e)

        # Fallback: OpenCV saliency + GrabCut
        try:
            return self._segment_fallback(image, subject_hint)
        except Exception as e:
            logger.error("Fallback segmentation failed: %s", e)

        # Last resort: centre-crop heuristic
        return self._segment_centre_heuristic(image)

    # ── SAM-based path ────────────────────────────────────────

    def _segment_sam(
        self, image: np.ndarray, hint: str
    ) -> SegmentationResult:
        """Use Segment Anything Model (SAM) for high-quality masking."""
        # Lazy import — heavy deps
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise RuntimeError("segment-anything not installed")

        if not self._sam_loaded:
            self._load_sam()

        assert self._sam_predictor is not None
        self._sam_predictor.set_image(image)

        h, w = image.shape[:2]

        # If we have a hint, try to use GroundingDINO for a box
        bbox = self._grounding_detect(image, hint) if hint else None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            input_box = np.array([x1, y1, x2, y2])
            masks, scores, _ = self._sam_predictor.predict(
                box=input_box[None, :],
                multimask_output=True,
            )
        else:
            # Automatic: use centre point
            cx, cy = w // 2, h // 2
            input_point = np.array([[cx, cy]])
            input_label = np.array([1])
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        # Pick best mask
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)
        conf = float(scores[best_idx])

        return self._build_result(image, mask, conf, hint or "object",
                                  source="sam")

    def _load_sam(self):
        """Load SAM model weights (lazy)."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch

            # Try to find SAM checkpoint
            models_dir = Path(__file__).parent.parent.parent / "models"
            ckpt = models_dir / "sam_vit_h_4b8939.pth"
            if not ckpt.exists():
                ckpt = models_dir / "sam_vit_b_01ec64.pth"
                model_type = "vit_b"
            else:
                model_type = "vit_h"

            if not ckpt.exists():
                raise FileNotFoundError(
                    f"No SAM checkpoint found in {models_dir}. "
                    "Download from https://github.com/facebookresearch/segment-anything"
                )

            sam = sam_model_registry[model_type](checkpoint=str(ckpt))
            sam.to(self.device)
            self._sam_model = sam
            self._sam_predictor = SamPredictor(sam)
            self._sam_loaded = True
            logger.info("SAM loaded: %s on %s", model_type, self.device)
        except Exception as e:
            logger.warning("Could not load SAM: %s", e)
            raise

    def _grounding_detect(
        self, image: np.ndarray, text: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """Attempt GroundingDINO-based text-guided detection for a bounding box."""
        try:
            from groundingdino.util.inference import load_model, predict
            # This would need a GroundingDINO checkpoint — optional dep
            logger.debug("GroundingDINO detection for '%s'", text)
        except ImportError:
            pass
        return None  # fallback to centre-point SAM

    # ── OpenCV fallback ───────────────────────────────────────

    def _segment_fallback(
        self, image: np.ndarray, hint: str
    ) -> SegmentationResult:
        """CPU-only segmentation using OpenCV GrabCut + saliency."""
        import cv2

        h, w = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Saliency detection for initial seed
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, sal_map = saliency.computeSaliency(bgr)

        if not ok or sal_map is None:
            raise RuntimeError("Saliency computation failed")

        sal_map = (sal_map * 255).astype(np.uint8)
        _, thresh = cv2.threshold(sal_map, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours to build a rough bbox
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No salient region found")

        biggest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(biggest)

        # Expand bbox by 10 %
        pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        # GrabCut
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[:] = cv2.GC_PR_BGD
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (x1, y1, x2 - x1, y2 - y1)
        cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model,
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

        binary = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0
        ).astype(np.float32)

        # Smooth edges
        binary = cv2.GaussianBlur(binary, (5, 5), 0)

        return self._build_result(image, binary, 0.55,
                                  hint or "object", source="grabcut")

    # ── Centre heuristic (last resort) ────────────────────────

    def _segment_centre_heuristic(self, image: np.ndarray) -> SegmentationResult:
        """Generate an elliptical mask centred on the image."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        cx, cy = w // 2, h // 2
        rx, ry = w // 3, h // 3

        try:
            import cv2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
        except ImportError:
            # Pure numpy fallback
            Y, X = np.ogrid[:h, :w]
            mask = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0).astype(np.float32)

        return self._build_result(image, mask, 0.3, "object",
                                  source="heuristic")

    # ── shared helpers ────────────────────────────────────────

    def _build_result(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        confidence: float,
        label: str,
        source: str,
    ) -> SegmentationResult:
        """Build a SegmentationResult, computing bbox and quality."""
        h, w = image.shape[:2]
        binary = (mask > 0.5).astype(np.uint8)
        area = float(binary.sum())
        total = float(h * w)
        area_ratio = area / total if total > 0 else 0.0

        # Bounding box from mask
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            bbox = (0, 0, w, h)

        # Quality assessment
        warnings: List[str] = []
        if area_ratio < MIN_OBJECT_AREA_RATIO:
            quality = SegmentationQuality.LOW
            warnings.append(
                f"Object too small ({area_ratio:.1%} of image). "
                "Motion may be inaccurate."
            )
        elif area_ratio > MAX_OBJECT_AREA_RATIO:
            quality = SegmentationQuality.LOW
            warnings.append(
                f"Object fills most of the image ({area_ratio:.1%}). "
                "Background stabilisation may be limited."
            )
        elif confidence >= 0.7 and source == "sam":
            quality = SegmentationQuality.HIGH
        elif confidence >= 0.4:
            quality = SegmentationQuality.MEDIUM
        else:
            quality = SegmentationQuality.LOW
            warnings.append(
                f"Low segmentation confidence ({confidence:.0%}). "
                "Full motion intensity maintained -- consider re-segmenting "
                "with SAM2Segmenter for higher accuracy."
            )

        bg_mask = 1.0 - mask

        logger.info(
            "Segmentation [%s]: label=%s conf=%.2f quality=%s area=%.1f%%",
            source, label, confidence, quality.value, area_ratio * 100,
        )

        return SegmentationResult(
            mask=mask,
            bbox=bbox,
            label=label,
            confidence=confidence,
            quality=quality,
            object_area_ratio=area_ratio,
            background_mask=bg_mask,
            warnings=warnings,
        )
