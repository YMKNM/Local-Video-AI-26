"""
SAM 2 High-Confidence Segmenter

Provides subject + secondary-object + background segmentation using
Segment Anything Model 2 (SAM 2) with automatic retry logic.

**CRITICAL DESIGN RULE**: If segmentation confidence < 90 %, the system
RE-SEGMENTS with different parameters instead of reducing motion.

Retry strategy (up to ``max_retries``):
  1. Original point/box prompt
  2. Multi-point grid prompt (3x3 grid inside detected region)
  3. Iterative mask refinement (feed previous mask back as input)
  4. Expanded bounding box with contour-based seed

Only after exhausting all retries does this module report LOW quality.
Motion intensity is NEVER reduced here — that is the caller's failure
mode, not ours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum acceptable confidence — below this we RETRY, not degrade
HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class MultiObjectSegmentation:
    """Result containing subject + secondary objects + background."""
    subject_mask: np.ndarray          # (H, W) float32 [0, 1] — primary subject
    subject_confidence: float
    subject_label: str
    subject_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    secondary_masks: List[np.ndarray] = field(default_factory=list)
    secondary_labels: List[str] = field(default_factory=list)
    secondary_confidences: List[float] = field(default_factory=list)

    background_mask: Optional[np.ndarray] = None  # (H, W) float32 — locked BG

    retries_used: int = 0
    strategy_used: str = "initial"
    warnings: List[str] = field(default_factory=list)

    @property
    def is_high_confidence(self) -> bool:
        return self.subject_confidence >= HIGH_CONFIDENCE_THRESHOLD

    @property
    def is_usable(self) -> bool:
        return self.subject_confidence >= MEDIUM_CONFIDENCE_THRESHOLD

    @property
    def object_area_ratio(self) -> float:
        if self.subject_mask is None:
            return 0.0
        binary = (self.subject_mask > 0.5).astype(np.float32)
        return float(binary.sum()) / max(1.0, float(self.subject_mask.size))

    def to_dict(self) -> Dict:
        return {
            "subject_label": self.subject_label,
            "subject_confidence": round(self.subject_confidence, 3),
            "high_confidence": self.is_high_confidence,
            "subject_bbox": list(self.subject_bbox),
            "object_area_ratio": round(self.object_area_ratio, 4),
            "secondary_objects": len(self.secondary_masks),
            "retries_used": self.retries_used,
            "strategy_used": self.strategy_used,
            "warnings": self.warnings,
        }


class SAM2Segmenter:
    """
    High-confidence segmentation using SAM 2 with retry logic.

    Usage::

        seg = SAM2Segmenter()
        result = seg.segment(image_rgb, subject_hint="player",
                             secondary_hints=["ball"])
    """

    def __init__(
        self,
        device: Optional[str] = None,
        max_retries: int = 3,
        models_dir: Optional[str] = None,
    ):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_retries = max_retries
        self.models_dir = Path(models_dir) if models_dir else (
            Path(__file__).parent.parent.parent / "models"
        )

        self._sam2_loaded = False
        self._sam2_predictor = None
        self._model_cfg = None

    # ── Public API ────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        subject_hint: str = "",
        secondary_hints: Optional[List[str]] = None,
    ) -> MultiObjectSegmentation:
        """
        Segment subject (and optionally secondary objects) from *image*.

        If confidence < 90% after the first attempt, re-segments with
        different strategies.  NEVER reduces motion intensity.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 (H, W, 3).
        subject_hint : str
            Text hint for the primary subject (e.g. "player", "boy").
        secondary_hints : list of str, optional
            Labels for secondary objects to segment (e.g. ["ball"]).

        Returns
        -------
        MultiObjectSegmentation
        """
        h, w = image.shape[:2]

        # Attempt SAM 2
        try:
            self._ensure_loaded()
            return self._segment_with_retry(image, subject_hint,
                                            secondary_hints or [])
        except Exception as e:
            logger.warning("SAM 2 unavailable (%s), trying SAM 1 fallback", e)

        # SAM 1 fallback
        try:
            return self._segment_sam1(image, subject_hint)
        except Exception as e:
            logger.warning("SAM 1 unavailable (%s), trying GrabCut", e)

        # GrabCut fallback (CPU-only)
        try:
            return self._segment_grabcut_retry(image, subject_hint)
        except Exception as e:
            logger.error("All segmentation methods failed: %s", e)

        # Absolute last resort — but mark as low confidence, NOT motion-reduced
        return self._centre_heuristic(image, subject_hint)

    # ── SAM 2 with retry ──────────────────────────────────────

    def _segment_with_retry(
        self,
        image: np.ndarray,
        subject_hint: str,
        secondary_hints: List[str],
    ) -> MultiObjectSegmentation:
        """Multi-strategy SAM 2 segmentation with retry loop."""
        h, w = image.shape[:2]
        best_mask = None
        best_conf = 0.0
        strategy_used = "initial"
        retries = 0

        strategies = [
            ("centre_point", self._strategy_centre_point),
            ("grid_points", self._strategy_grid_points),
            ("mask_refinement", self._strategy_mask_refinement),
            ("contour_box", self._strategy_contour_box),
        ]

        for name, strategy_fn in strategies:
            if best_conf >= HIGH_CONFIDENCE_THRESHOLD:
                break  # already good enough

            try:
                mask, conf = strategy_fn(image, best_mask)
                if conf > best_conf:
                    best_mask = mask
                    best_conf = conf
                    strategy_used = name
                    logger.info(
                        "Segmentation attempt [%s]: conf=%.2f%% (%s threshold)",
                        name, conf * 100,
                        "ABOVE" if conf >= HIGH_CONFIDENCE_THRESHOLD else "below"
                    )
            except Exception as e:
                logger.warning("Strategy %s failed: %s", name, e)

            retries += 1
            if retries > self.max_retries:
                break

        if best_mask is None:
            raise RuntimeError("All SAM 2 strategies failed")

        # Build background mask
        bg_mask = 1.0 - best_mask

        # Segment secondary objects
        sec_masks, sec_labels, sec_confs = [], [], []
        for hint in secondary_hints:
            try:
                sm, sc = self._segment_secondary(image, best_mask, hint)
                sec_masks.append(sm)
                sec_labels.append(hint)
                sec_confs.append(sc)
            except Exception as e:
                logger.warning("Secondary segmentation for '%s' failed: %s",
                               hint, e)

        # Build bbox
        binary = (best_mask > 0.5).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            bbox = (0, 0, w, h)

        warnings = []
        if best_conf < HIGH_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Segmentation confidence {best_conf:.0%} is below 90%% "
                f"after {retries} retries.  Generation will proceed at "
                f"FULL motion intensity.  Consider a clearer input image."
            )

        return MultiObjectSegmentation(
            subject_mask=best_mask,
            subject_confidence=best_conf,
            subject_label=subject_hint or "subject",
            subject_bbox=bbox,
            secondary_masks=sec_masks,
            secondary_labels=sec_labels,
            secondary_confidences=sec_confs,
            background_mask=bg_mask,
            retries_used=retries,
            strategy_used=strategy_used,
            warnings=warnings,
        )

    # ── Retry strategies ──────────────────────────────────────

    def _strategy_centre_point(
        self, image: np.ndarray, prev_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Prompt SAM 2 with a single centre point."""
        h, w = image.shape[:2]
        self._sam2_predictor.set_image(image)

        point = np.array([[w // 2, h // 2]])
        label = np.array([1])
        masks, scores, _ = self._sam2_predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.float32), float(scores[best_idx])

    def _strategy_grid_points(
        self, image: np.ndarray, prev_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Prompt SAM 2 with a 3x3 grid of points inside the detected region."""
        h, w = image.shape[:2]
        self._sam2_predictor.set_image(image)

        if prev_mask is not None:
            # Sample points inside previous mask
            ys, xs = np.where(prev_mask > 0.5)
            if len(xs) > 9:
                indices = np.linspace(0, len(xs) - 1, 9, dtype=int)
                points = np.stack([xs[indices], ys[indices]], axis=1)
            else:
                points = self._make_grid(w, h)
        else:
            points = self._make_grid(w, h)

        labels = np.ones(len(points), dtype=np.int32)
        masks, scores, _ = self._sam2_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.float32), float(scores[best_idx])

    def _strategy_mask_refinement(
        self, image: np.ndarray, prev_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Feed previous mask back as a prompt for iterative refinement."""
        if prev_mask is None:
            raise RuntimeError("No previous mask to refine")

        self._sam2_predictor.set_image(image)

        # Use the low-res mask as input
        mask_input = prev_mask[None, :, :]  # (1, H, W)

        masks, scores, _ = self._sam2_predictor.predict(
            mask_input=mask_input,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.float32), float(scores[best_idx])

    def _strategy_contour_box(
        self, image: np.ndarray, prev_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Find salient contour via edge detection, use as box prompt."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found")

        biggest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(biggest)

        h, w = image.shape[:2]
        pad = int(max(bw, bh) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        self._sam2_predictor.set_image(image)
        box = np.array([x1, y1, x2, y2])
        masks, scores, _ = self._sam2_predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.float32), float(scores[best_idx])

    def _segment_secondary(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        hint: str,
    ) -> Tuple[np.ndarray, float]:
        """Segment a secondary object, excluding the subject region."""
        h, w = image.shape[:2]
        self._sam2_predictor.set_image(image)

        # Use negative points from subject + positive points from remaining area
        bg_area = (subject_mask < 0.3)
        ys, xs = np.where(bg_area)
        if len(xs) < 5:
            raise RuntimeError("Not enough background area for secondary object")

        # Sample some background points
        idx = np.linspace(0, len(xs) - 1, 5, dtype=int)
        points = np.stack([xs[idx], ys[idx]], axis=1)
        labels = np.ones(5, dtype=np.int32)

        masks, scores, _ = self._sam2_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)
        # Zero out subject region
        mask = mask * (1.0 - subject_mask)
        return mask, float(scores[best_idx])

    # ── SAM 2 model loading ───────────────────────────────────

    def _ensure_loaded(self):
        if self._sam2_loaded:
            return

        try:
            # Try sam2 package first (official Meta release)
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2
            self._load_sam2_official()
            return
        except ImportError:
            pass

        try:
            # Fallback: segment_anything_2 (community packaging)
            from segment_anything import sam_model_registry, SamPredictor
            self._load_sam1_as_fallback()
            return
        except ImportError:
            pass

        raise RuntimeError(
            "Neither sam2 nor segment-anything is installed. "
            "Install with: pip install segment-anything-2  OR  "
            "pip install segment-anything"
        )

    def _load_sam2_official(self):
        """Load official SAM 2 from Meta."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        import torch

        # Check for SAM 2 checkpoints
        ckpt_names = [
            ("sam2.1_hiera_large.pt", "sam2.1_hiera_l"),
            ("sam2_hiera_large.pt", "sam2_hiera_l"),
            ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+"),
            ("sam2_hiera_base_plus.pt", "sam2_hiera_b+"),
            ("sam2.1_hiera_small.pt", "sam2.1_hiera_s"),
            ("sam2_hiera_small.pt", "sam2_hiera_s"),
            ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t"),
            ("sam2_hiera_tiny.pt", "sam2_hiera_t"),
        ]

        for ckpt_name, cfg_name in ckpt_names:
            ckpt_path = self.models_dir / ckpt_name
            if ckpt_path.exists():
                logger.info("Loading SAM 2: %s on %s", ckpt_name, self.device)
                sam2 = build_sam2(cfg_name, str(ckpt_path), device=self.device)
                self._sam2_predictor = SAM2ImagePredictor(sam2)
                self._sam2_loaded = True
                return

        raise FileNotFoundError(
            f"No SAM 2 checkpoint in {self.models_dir}. "
            "Download from https://github.com/facebookresearch/sam2"
        )

    def _load_sam1_as_fallback(self):
        """Load SAM 1 as a compatible fallback."""
        from segment_anything import sam_model_registry, SamPredictor
        import torch

        ckpt_names = [
            ("sam_vit_h_4b8939.pth", "vit_h"),
            ("sam_vit_l_0b3195.pth", "vit_l"),
            ("sam_vit_b_01ec64.pth", "vit_b"),
        ]

        for ckpt_name, model_type in ckpt_names:
            ckpt_path = self.models_dir / ckpt_name
            if ckpt_path.exists():
                logger.info("Loading SAM 1: %s on %s", ckpt_name, self.device)
                sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
                sam.to(self.device)
                self._sam2_predictor = SamPredictor(sam)
                self._sam2_loaded = True
                return

        raise FileNotFoundError(
            f"No SAM checkpoint in {self.models_dir}. "
            "Download from https://github.com/facebookresearch/segment-anything"
        )

    # ── SAM 1 direct fallback ─────────────────────────────────

    def _segment_sam1(
        self, image: np.ndarray, hint: str
    ) -> MultiObjectSegmentation:
        """Direct SAM 1 segmentation as a module-level fallback."""
        from segment_anything import sam_model_registry, SamPredictor

        if not self._sam2_loaded:
            self._load_sam1_as_fallback()

        return self._segment_with_retry(image, hint, [])

    # ── GrabCut fallback with retry ───────────────────────────

    def _segment_grabcut_retry(
        self, image: np.ndarray, hint: str
    ) -> MultiObjectSegmentation:
        """GrabCut with multiple iteration counts as retry strategy."""
        import cv2

        h, w = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        best_mask = None
        best_conf = 0.0

        # Try different starting rectangles
        rects = [
            (w // 6, h // 6, w * 4 // 6, h * 4 // 6),       # centre 66%
            (w // 8, h // 8, w * 6 // 8, h * 6 // 8),       # centre 75%
            (w // 4, h // 4, w * 2 // 4, h * 2 // 4),       # centre 50%
        ]

        for rect in rects:
            if best_conf >= HIGH_CONFIDENCE_THRESHOLD:
                break

            gc_mask = np.zeros((h, w), np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)

            for iters in [5, 10, 15]:
                try:
                    cv2.grabCut(bgr, gc_mask, rect, bgd, fgd,
                                iterCount=iters, mode=cv2.GC_INIT_WITH_RECT)
                    mask = np.where(
                        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                        1.0, 0.0
                    ).astype(np.float32)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)

                    # Estimate confidence from mask edge sharpness
                    edge_sharpness = self._mask_edge_sharpness(mask)
                    area_ratio = float((mask > 0.5).sum()) / (h * w)
                    conf = min(0.85, edge_sharpness * 0.5 + area_ratio * 0.3 + 0.2)

                    if conf > best_conf:
                        best_conf = conf
                        best_mask = mask
                except Exception:
                    continue

        if best_mask is None:
            raise RuntimeError("GrabCut segmentation failed")

        bg_mask = 1.0 - best_mask
        binary = (best_mask > 0.5).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) > 0 else (0, 0, w, h)

        warnings = []
        if best_conf < HIGH_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"GrabCut confidence {best_conf:.0%} below 90%%. "
                f"Full motion intensity maintained. "
                f"For better results, use an image with clear subject."
            )

        return MultiObjectSegmentation(
            subject_mask=best_mask,
            subject_confidence=best_conf,
            subject_label=hint or "subject",
            subject_bbox=bbox,
            background_mask=bg_mask,
            retries_used=len(rects),
            strategy_used="grabcut_retry",
            warnings=warnings,
        )

    # ── Centre heuristic (absolute last resort) ───────────────

    def _centre_heuristic(
        self, image: np.ndarray, hint: str
    ) -> MultiObjectSegmentation:
        """Elliptical mask centred on image. Low confidence, full motion."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2
        rx, ry = w // 3, h // 3

        try:
            import cv2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
        except ImportError:
            Y, X = np.ogrid[:h, :w]
            mask = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0).astype(np.float32)

        return MultiObjectSegmentation(
            subject_mask=mask,
            subject_confidence=0.30,
            subject_label=hint or "subject",
            subject_bbox=(cx - rx, cy - ry, cx + rx, cy + ry),
            background_mask=1.0 - mask,
            retries_used=0,
            strategy_used="heuristic",
            warnings=[
                "Using centre-heuristic segmentation (confidence 30%%). "
                "Motion will proceed at FULL intensity. "
                "For better quality, install segment-anything or sam2."
            ],
        )

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _make_grid(w: int, h: int, n: int = 3) -> np.ndarray:
        """Generate an NxN grid of points inside the image."""
        xs = np.linspace(w * 0.2, w * 0.8, n)
        ys = np.linspace(h * 0.2, h * 0.8, n)
        grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        return grid

    @staticmethod
    def _mask_edge_sharpness(mask: np.ndarray) -> float:
        """Estimate mask quality from edge gradient magnitude."""
        try:
            import cv2
            grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_pixels = (mag > 0.1).sum()
            total = mask.size
            # Sharp edges = good segmentation
            return min(1.0, edge_pixels / max(1, total) * 50)
        except ImportError:
            return 0.5
