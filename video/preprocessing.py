"""Frame preprocessing.

Resizing happens once, here, and the scale factor is recorded so that every
coordinate the pipeline produces can be mapped back to original pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameGeometry:
    """Relationship between the processed frame and the source frame."""
    src_width: int
    src_height: int
    proc_width: int
    proc_height: int
    scale: float          # proc = src * scale

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        if self.scale <= 0:
            return x, y
        return x / self.scale, y / self.scale


class Preprocessor:
    """Downscale frames to a target long edge and convert colour spaces.

    Downscaling is the single biggest lever on runtime: MediaPipe and YOLO both
    cost roughly O(pixels), and wrist localisation is not meaningfully better at
    4K than at 960px for this task.
    """

    def __init__(self, target_long_edge: Optional[int] = 960) -> None:
        self.target_long_edge = target_long_edge
        self._geometry: Optional[FrameGeometry] = None

    @property
    def geometry(self) -> Optional[FrameGeometry]:
        return self._geometry

    def _compute_geometry(self, frame: np.ndarray) -> FrameGeometry:
        h, w = frame.shape[:2]
        scale = 1.0
        if self.target_long_edge is not None:
            long_edge = max(h, w)
            if long_edge > self.target_long_edge:
                scale = self.target_long_edge / float(long_edge)
        return FrameGeometry(
            src_width=w,
            src_height=h,
            proc_width=int(round(w * scale)),
            proc_height=int(round(h * scale)),
            scale=scale,
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(bgr_resized, rgb_resized)``.

        MediaPipe needs RGB, OpenCV drawing needs BGR, so both are returned to
        avoid a second colour conversion per frame.
        """
        if self._geometry is None:
            self._geometry = self._compute_geometry(frame)
        g = self._geometry
        if g.scale != 1.0:
            bgr = cv2.resize(
                frame, (g.proc_width, g.proc_height), interpolation=cv2.INTER_AREA
            )
        else:
            bgr = frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr, rgb
