"""Video ingestion and metadata probing.

Nothing downstream assumes a fixed duration, frame rate or resolution; every
consumer reads those from :class:`VideoMeta`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoLoadError(RuntimeError):
    """Raised when a file cannot be opened or contains no decodable frames."""


@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    duration_s: float
    width: int
    height: int
    #: True when ``n_frames`` came from the container header rather than a
    #: full decode pass, which some AVI/MOV files report inaccurately.
    frame_count_is_estimate: bool = False

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["resolution"] = self.resolution
        return d


def probe(path: str | Path) -> VideoMeta:
    """Read fps / frame count / duration / resolution without decoding fully."""
    path = Path(path)
    if not path.exists():
        raise VideoLoadError(f"File not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoLoadError(
            f"OpenCV could not open '{path.name}'. The container or codec may "
            f"be unsupported by this build of OpenCV."
        )
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoLoadError(f"'{path.name}' contains no decodable frames.")
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
    finally:
        cap.release()

    estimate = False
    if not np.isfinite(fps) or fps <= 1e-3:
        # Some webcam-recorded MOV files report fps == 0.
        fps = 30.0
        estimate = True
    if n_frames <= 0:
        n_frames = _count_frames_by_decode(path)
        estimate = True

    duration = n_frames / fps if fps > 0 else 0.0
    return VideoMeta(
        path=str(path),
        fps=fps,
        n_frames=n_frames,
        duration_s=duration,
        width=width,
        height=height,
        frame_count_is_estimate=estimate,
    )


def _count_frames_by_decode(path: Path, hard_cap: int = 200_000) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while count < hard_cap:
            ok = cap.grab()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


class VideoReader:
    """Iterate frames as ``(frame_index, timestamp_seconds, bgr_image)``.

    ``stride`` skips frames for speed while timestamps stay in real seconds, so
    every extracted feature remains anchored to the original video clock.
    """

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> None:
        self.path = str(path)
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.meta = probe(self.path)

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise VideoLoadError(f"Could not reopen {self.path}")
        emitted = 0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if idx % self.stride == 0:
                    yield idx, idx / self.meta.fps, frame
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()

    @property
    def expected_frames(self) -> int:
        """How many frames iteration will emit, for progress reporting."""
        n = self.meta.n_frames // self.stride
        if self.max_frames is not None:
            n = min(n, self.max_frames)
        return max(1, n)


def plan_processing(meta: VideoMeta, stride: int, max_frames: Optional[int]) -> Dict:
    """Describe what will actually be processed, including truncation.

    Returned to the UI so the user is told when a long video is being clipped
    rather than silently analysing only the beginning.
    """
    total_after_stride = max(1, meta.n_frames // max(1, stride))
    processed = total_after_stride
    truncated = False
    if max_frames is not None and processed > max_frames:
        processed = max_frames
        truncated = True
    covered_s = processed * stride / meta.fps if meta.fps > 0 else 0.0
    return {
        "frames_to_process": processed,
        "stride": stride,
        "truncated": truncated,
        "covered_duration_s": round(covered_s, 3),
        "effective_sample_rate_hz": round(meta.fps / max(1, stride), 3),
    }
