"""
Temporal Consistency — Pre- and Post-Generation Frame Coherence

Two-stage approach to temporal consistency:

**Stage 1: Pre-generation (latent-level)**
  - Cross-frame latent initialization: each frame's starting noise
    is partially blended from the previous frame's latent
  - Deterministic noise schedule per frame for reproducibility

**Stage 2: Post-generation (pixel-level)**
  - Optical flow estimation between consecutive frames
  - Flow-guided pixel blending to reduce flicker
  - Object identity tracking via mask propagation
  - Anti-ghosting: detect and suppress temporal artefacts
  - Background lock: static background compositing

Design principle: reduce duration, reduce speed — NEVER introduce artefacts
or degrade frame quality. If consistency cannot be achieved, warn but
preserve individual frame quality.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyConfig:
    """Tuning parameters for temporal consistency."""
    # Pre-generation latent blending
    latent_blend_alpha: float = 0.3     # [0, 1] how much of prev frame latent to reuse
    noise_schedule: str = "linear"      # linear, cosine, constant

    # Post-generation flow smoothing
    enable_flow_smoothing: bool = True
    flow_blend_strength: float = 0.4    # [0, 1]
    flow_window_size: int = 3           # frames to consider

    # Background lock
    enable_background_lock: bool = True
    background_blend_alpha: float = 0.8  # how much to keep original background

    # Anti-ghosting
    enable_anti_ghosting: bool = True
    ghost_threshold: float = 0.15       # pixel diff threshold for ghost detection

    # Identity tracking
    enable_identity_tracking: bool = True
    identity_match_threshold: float = 0.7

    # Temporal denoising
    enable_temporal_denoise: bool = True
    denoise_strength: float = 0.5       # [0, 1]
    temporal_kernel_size: int = 3       # frames for temporal averaging

    # Graceful degradation
    max_processing_time_per_frame: float = 2.0  # seconds


@dataclass
class ConsistencyResult:
    """Output from the temporal consistency pass."""
    frames: List[np.ndarray]
    num_corrections: int = 0
    ghost_frames_fixed: int = 0
    flow_smoothed_frames: int = 0
    warnings: List[str] = field(default_factory=list)


class TemporalConsistencyPass:
    """
    Apply temporal consistency to a sequence of generated video frames.

    This is a post-generation pass that improves frame-to-frame coherence
    WITHOUT reducing motion or quality.

    Usage::

        tc = TemporalConsistencyPass()
        result = tc.apply(
            frames=generated_frames,
            subject_masks=per_frame_masks,
            background=original_background,
            config=ConsistencyConfig(),
        )
    """

    def __init__(self):
        self._flow_model = None

    def apply(
        self,
        frames: List[np.ndarray],
        subject_masks: Optional[List[np.ndarray]] = None,
        background: Optional[np.ndarray] = None,
        config: Optional[ConsistencyConfig] = None,
    ) -> ConsistencyResult:
        """
        Apply full temporal consistency pipeline to frame sequence.

        Parameters
        ----------
        frames : list of np.ndarray
            Generated RGB uint8 frames (H, W, 3).
        subject_masks : list of np.ndarray, optional
            Per-frame binary masks for subject region.
        background : np.ndarray, optional
            Original background image for background lock.
        config : ConsistencyConfig

        Returns
        -------
        ConsistencyResult
        """
        if config is None:
            config = ConsistencyConfig()

        if len(frames) < 2:
            return ConsistencyResult(frames=frames)

        result_frames = [f.copy() for f in frames]
        warnings: List[str] = []
        num_corrections = 0
        ghost_fixes = 0
        flow_smoothed = 0

        # Step 1: Background lock
        if config.enable_background_lock and background is not None:
            result_frames, n = self._apply_background_lock(
                result_frames, subject_masks, background, config
            )
            num_corrections += n
            logger.info("Background lock applied (%d frames adjusted)", n)

        # Step 2: Optical flow smoothing
        if config.enable_flow_smoothing:
            try:
                result_frames, n = self._apply_flow_smoothing(
                    result_frames, config
                )
                flow_smoothed = n
                num_corrections += n
                logger.info("Flow smoothing applied (%d frames)", n)
            except Exception as e:
                logger.warning("Flow smoothing failed: %s", e)
                warnings.append(f"Flow smoothing skipped: {e}")

        # Step 3: Anti-ghosting
        if config.enable_anti_ghosting:
            result_frames, n = self._apply_anti_ghosting(
                result_frames, config
            )
            ghost_fixes = n
            num_corrections += n
            if n > 0:
                logger.info("Anti-ghosting fixed %d frames", n)

        # Step 4: Temporal denoising
        if config.enable_temporal_denoise:
            result_frames = self._apply_temporal_denoise(
                result_frames, config
            )
            num_corrections += 1  # batch operation

        return ConsistencyResult(
            frames=result_frames,
            num_corrections=num_corrections,
            ghost_frames_fixed=ghost_fixes,
            flow_smoothed_frames=flow_smoothed,
            warnings=warnings,
        )

    # ── Pre-generation: Latent Initialization ─────────────────

    def create_blended_latents(
        self,
        num_frames: int,
        latent_shape: Tuple[int, ...],
        alpha: float = 0.3,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Create temporally-coherent initial noise for all frames.

        Each frame's noise is a blend of:
          - Fresh random noise (1 - alpha)
          - Previous frame's noise (alpha)

        This makes adjacent frames share structure in latent space,
        improving temporal coherence FROM THE START.

        Parameters
        ----------
        num_frames : int
        latent_shape : tuple
            Shape of a single latent (C, H, W).
        alpha : float
            Blending strength [0, 1].
        seed : int

        Returns
        -------
        np.ndarray
            Shape (num_frames, *latent_shape), float32.
        """
        rng = np.random.default_rng(seed)
        all_latents = np.zeros((num_frames, *latent_shape), dtype=np.float32)

        # First frame: pure noise
        all_latents[0] = rng.standard_normal(latent_shape).astype(np.float32)

        # Subsequent frames: blend with previous
        for i in range(1, num_frames):
            fresh = rng.standard_normal(latent_shape).astype(np.float32)
            all_latents[i] = alpha * all_latents[i - 1] + (1 - alpha) * fresh

            # Re-normalise to unit variance
            std = all_latents[i].std()
            if std > 1e-6:
                all_latents[i] /= std

        return all_latents

    # ── Post-generation: Background Lock ──────────────────────

    def _apply_background_lock(
        self,
        frames: List[np.ndarray],
        masks: Optional[List[np.ndarray]],
        background: np.ndarray,
        config: ConsistencyConfig,
    ) -> Tuple[List[np.ndarray], int]:
        """Replace background regions with original background."""
        if masks is None:
            return frames, 0

        bg = background.copy()
        h, w = frames[0].shape[:2]

        # Resize background to match frames
        if bg.shape[:2] != (h, w):
            try:
                import cv2
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LANCZOS4)
            except ImportError:
                from PIL import Image as PILImage
                bg = np.array(
                    PILImage.fromarray(bg).resize((w, h), PILImage.LANCZOS)
                )

        alpha = config.background_blend_alpha
        count = 0

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            if mask is None:
                continue

            # Ensure mask is correct shape
            if mask.shape[:2] != (h, w):
                try:
                    import cv2
                    mask = cv2.resize(mask.astype(np.float32), (w, h))
                except ImportError:
                    continue

            mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask

            # Subject region: keep generated frame
            # Background region: blend with original
            bg_mask = 1.0 - mask_3d
            blended = (
                mask_3d * frame.astype(np.float32) +
                bg_mask * (alpha * bg.astype(np.float32) +
                           (1 - alpha) * frame.astype(np.float32))
            )
            frames[i] = np.clip(blended, 0, 255).astype(np.uint8)
            count += 1

        return frames, count

    # ── Post-generation: Optical Flow Smoothing ───────────────

    def _apply_flow_smoothing(
        self,
        frames: List[np.ndarray],
        config: ConsistencyConfig,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Use optical flow to warp and blend adjacent frames for smoothness.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available for flow smoothing")
            return frames, 0

        count = 0
        strength = config.flow_blend_strength
        window = config.flow_window_size

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            # Dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Warp previous frame using flow
            h, w = frames[i].shape[:2]
            flow_map = np.column_stack([
                (np.arange(w)[np.newaxis, :] + flow[:, :, 0]).ravel(),
                (np.arange(h)[:, np.newaxis] + flow[:, :, 1]).ravel(),
            ]).reshape(h, w, 2).astype(np.float32)

            warped = cv2.remap(
                frames[i - 1],
                flow_map[:, :, 0],
                flow_map[:, :, 1],
                cv2.INTER_LINEAR,
            )

            # Blend current frame with flow-warped previous
            frames[i] = np.clip(
                (1 - strength) * frames[i].astype(np.float32) +
                strength * warped.astype(np.float32),
                0, 255
            ).astype(np.uint8)
            count += 1

        return frames, count

    # ── Post-generation: Anti-Ghosting ────────────────────────

    def _apply_anti_ghosting(
        self,
        frames: List[np.ndarray],
        config: ConsistencyConfig,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Detect and suppress ghosting artefacts (semi-transparent duplicates).

        Ghost detection: if a region appears in frame N that wasn't in N-1
        or N+1, and has low contrast, it's likely a ghost.
        """
        if len(frames) < 3:
            return frames, 0

        threshold = config.ghost_threshold * 255
        fixes = 0

        for i in range(1, len(frames) - 1):
            prev_f = frames[i - 1].astype(np.float32)
            curr_f = frames[i].astype(np.float32)
            next_f = frames[i + 1].astype(np.float32)

            # Temporal difference
            diff_prev = np.abs(curr_f - prev_f).mean(axis=2)
            diff_next = np.abs(curr_f - next_f).mean(axis=2)

            # Ghost candidates: high diff to both neighbours
            ghost_mask = (diff_prev > threshold) & (diff_next > threshold)

            # But prev and next should be similar (ghost is transient)
            neighbour_diff = np.abs(prev_f - next_f).mean(axis=2)
            ghost_mask = ghost_mask & (neighbour_diff < threshold * 0.7)

            ghost_pixels = ghost_mask.sum()
            total_pixels = ghost_mask.size

            if ghost_pixels > total_pixels * 0.001:  # >0.1% ghosting
                # Replace ghost regions with average of neighbours
                ghost_3d = ghost_mask[:, :, np.newaxis]
                replacement = ((prev_f + next_f) / 2).astype(np.uint8)
                frames[i] = np.where(ghost_3d, replacement, frames[i])
                fixes += 1
                logger.debug("Fixed ghosting in frame %d (%d pixels)",
                             i, ghost_pixels)

        return frames, fixes

    # ── Post-generation: Temporal Denoising ───────────────────

    def _apply_temporal_denoise(
        self,
        frames: List[np.ndarray],
        config: ConsistencyConfig,
    ) -> List[np.ndarray]:
        """
        Temporal median / mean filtering across adjacent frames.

        Reduces high-frequency flicker while preserving motion.
        """
        k = config.temporal_kernel_size
        strength = config.denoise_strength

        if k < 2 or len(frames) < k:
            return frames

        half_k = k // 2
        result = []

        for i in range(len(frames)):
            # Gather window
            start = max(0, i - half_k)
            end = min(len(frames), i + half_k + 1)
            window = np.stack(
                [f.astype(np.float32) for f in frames[start:end]], axis=0
            )

            # Temporal mean
            temporal_mean = window.mean(axis=0)

            # Blend with current frame
            blended = (
                (1 - strength) * frames[i].astype(np.float32) +
                strength * temporal_mean
            )
            result.append(np.clip(blended, 0, 255).astype(np.uint8))

        return result

    # ── Identity Tracking ─────────────────────────────────────

    def propagate_masks(
        self,
        initial_mask: np.ndarray,
        frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Propagate subject mask across frames using optical flow.

        This tracks the subject across frames so background lock and
        identity preservation can operate per-frame.
        """
        masks = [initial_mask.copy()]

        try:
            import cv2
        except ImportError:
            # Without cv2, duplicate the initial mask
            return [initial_mask.copy() for _ in frames]

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            h, w = initial_mask.shape[:2]
            flow_map_x = np.arange(w)[np.newaxis, :] + flow[:, :, 0]
            flow_map_y = np.arange(h)[:, np.newaxis] + flow[:, :, 1]

            warped_mask = cv2.remap(
                masks[-1].astype(np.float32),
                flow_map_x.astype(np.float32),
                flow_map_y.astype(np.float32),
                cv2.INTER_LINEAR,
            )

            # Threshold to keep it binary
            warped_mask = (warped_mask > 0.5).astype(np.float32)
            masks.append(warped_mask)

            prev_gray = curr_gray

        return masks
