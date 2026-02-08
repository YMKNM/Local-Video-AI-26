"""
Temporal Stabilizer -- Stage 5

Post-processes a sequence of generated video frames to improve
temporal coherence:

    1. **Optical-flow smoothing** -- uses Farneback dense optical flow
       (OpenCV) to detect jitter and dampen it across adjacent frames.
    2. **Temporal denoising** -- fast weighted average of neighbouring
       frames to suppress flicker.
    3. **Optional frame interpolation** -- doubles the frame rate via
       flow-warped blending (cheap but effective).

All processing is OpenCV-only (no extra ML models required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TemporalStabilizer:
    """
    Post-process generated frames for temporal coherence.

    Usage::

        stab = TemporalStabilizer()
        smoothed = stab.stabilize_sequence(frame_paths)
    """

    def __init__(
        self,
        flow_smoothing: float = 0.6,
        denoise_strength: float = 0.3,
        interpolate: bool = False,
    ):
        """
        Parameters
        ----------
        flow_smoothing : float
            Strength of optical-flow-based smoothing [0, 1].
            0 = no smoothing, 1 = heavy smoothing.
        denoise_strength : float
            Temporal denoising blend weight [0, 1].
        interpolate : bool
            If True, insert interpolated frames between each pair
            (doubles the frame count).
        """
        self.flow_smoothing = max(0.0, min(1.0, flow_smoothing))
        self.denoise_strength = max(0.0, min(1.0, denoise_strength))
        self.interpolate = interpolate

    # ── public API ────────────────────────────────────────────

    def stabilize_sequence(
        self,
        frame_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Stabilise a sequence of frame images.

        Parameters
        ----------
        frame_paths : list of str
            Ordered paths to PNG frames.
        output_dir : str, optional
            Where to write stabilised frames.  If ``None``, frames are
            written next to the originals with a ``_stab`` suffix.

        Returns
        -------
        list of str
            Paths to the stabilised (and optionally interpolated) frames.
        """
        if len(frame_paths) < 2:
            logger.info("Only %d frame(s) -- nothing to stabilise", len(frame_paths))
            return list(frame_paths)

        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available -- skipping stabilisation")
            return list(frame_paths)

        # Load all frames into memory
        frames = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                logger.warning("Could not read frame: %s", fp)
                continue
            frames.append(img)

        if len(frames) < 2:
            return list(frame_paths)

        logger.info(
            "Stabilising %d frames (flow=%.2f, denoise=%.2f, interp=%s)",
            len(frames), self.flow_smoothing, self.denoise_strength,
            self.interpolate,
        )

        # ── Step 1: Temporal denoising ────────────────────────
        if self.denoise_strength > 0:
            frames = self._temporal_denoise(frames)

        # ── Step 2: Optical-flow smoothing ────────────────────
        if self.flow_smoothing > 0:
            frames = self._flow_smooth(frames)

        # ── Step 3: Frame interpolation (optional) ────────────
        if self.interpolate:
            frames = self._interpolate_frames(frames)

        # ── Write output ──────────────────────────────────────
        if output_dir:
            out = Path(output_dir)
        else:
            out = Path(frame_paths[0]).parent / "stabilised"
        out.mkdir(parents=True, exist_ok=True)

        out_paths: List[str] = []
        for idx, frm in enumerate(frames):
            fp = out / f"frame_{idx:05d}.png"
            cv2.imwrite(str(fp), frm)
            out_paths.append(str(fp))

        logger.info("Wrote %d stabilised frames to %s", len(out_paths), out)
        return out_paths

    # ── Temporal denoising ────────────────────────────────────

    def _temporal_denoise(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Weighted average of each frame with its neighbours.

        Kernel  [0.15, 0.70, 0.15]  for denoise_strength=0.3.
        The centre weight decreases with higher strength.
        """
        n = len(frames)
        alpha = self.denoise_strength * 0.5   # neighbour weight per side
        centre = 1.0 - 2 * alpha

        out: List[np.ndarray] = []
        for i in range(n):
            blended = frames[i].astype(np.float32) * centre

            if i > 0:
                blended += frames[i - 1].astype(np.float32) * alpha
            else:
                blended += frames[i].astype(np.float32) * alpha  # repeat first

            if i < n - 1:
                blended += frames[i + 1].astype(np.float32) * alpha
            else:
                blended += frames[i].astype(np.float32) * alpha  # repeat last

            out.append(np.clip(blended, 0, 255).astype(np.uint8))

        logger.debug("Temporal denoise applied (alpha=%.3f)", alpha)
        return out

    # ── Optical-flow smoothing ────────────────────────────────

    def _flow_smooth(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Use Farneback dense optical flow to smooth jittery motion.

        For each frame pair (i, i+1), compute the forward flow and
        warp frame i towards frame i+1, then blend.
        """
        import cv2

        n = len(frames)
        out = [frames[0].copy()]
        strength = self.flow_smoothing * 0.4  # moderate blend

        for i in range(1, n):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            # Build remap coords from flow
            h, w = curr_gray.shape
            flow_map = np.column_stack((
                np.tile(np.arange(w), h),
                np.repeat(np.arange(h), w),
            )).reshape(h, w, 2).astype(np.float32)

            # Warp previous frame towards current using half the flow
            warp_map = flow_map + flow * 0.5
            warped_prev = cv2.remap(
                frames[i - 1],
                warp_map[:, :, 0],
                warp_map[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Blend
            blended = cv2.addWeighted(
                frames[i], 1.0 - strength,
                warped_prev, strength,
                0,
            )
            out.append(blended)

        logger.debug("Optical flow smoothing applied (strength=%.3f)", strength)
        return out

    # ── Frame interpolation ───────────────────────────────────

    def _interpolate_frames(
        self, frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Insert an interpolated frame between each adjacent pair
        (flow-warped 50/50 blend).  Doubles the frame count.
        """
        import cv2

        n = len(frames)
        out: List[np.ndarray] = [frames[0].copy()]

        for i in range(n - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            h, w = prev_gray.shape
            flow_map = np.column_stack((
                np.tile(np.arange(w), h),
                np.repeat(np.arange(h), w),
            )).reshape(h, w, 2).astype(np.float32)

            # Warp prev -> mid (half flow)
            mid_map = flow_map + flow * 0.5
            warped = cv2.remap(
                frames[i],
                mid_map[:, :, 0],
                mid_map[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Blend with next frame at 50/50
            interp = cv2.addWeighted(warped, 0.5, frames[i + 1], 0.5, 0)
            out.append(interp)
            out.append(frames[i + 1].copy())

        logger.debug("Interpolated %d -> %d frames", n, len(out))
        return out
