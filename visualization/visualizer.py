"""Annotated video rendering (Step 7/8 of the pipeline).

Draws, for every frame: the arm skeleton, the hand, the tracked object, a fading
wrist trail, a HUD with the current phase and timestamp, and a colour-coded
segmentation timeline with a playhead.

Frames that were skipped by ``frame_stride`` reuse the nearest processed
sample's annotations, so the overlay stays smooth while the underlying data
remains honest about its sampling rate.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from segmentation.base import Segment
from utils.config import PHASE_COLORS_BGR, VizConfig
from utils.signal_utils import format_timestamp

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (32, 30, 28)


class SegmentVideoRenderer:
    """Writes ``processed_video.mp4`` with overlays and a phase timeline."""

    def __init__(self, cfg: VizConfig) -> None:
        self.cfg = cfg

    # -- public ----------------------------------------------------------- #
    def render(
        self,
        video_path: str,
        out_path: Path,
        df: pd.DataFrame,
        segments: List[Segment],
        total_duration: float,
        render_long_edge: int = 960,
        object_available: bool = False,
        task_label: str = "UNKNOWN",
        task_confidence: float = 0.0,
        progress_cb=None,
        max_frames: Optional[int] = None,
    ) -> Optional[Path]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        scale = 1.0
        if max(src_w, src_h) > render_long_edge:
            scale = render_long_edge / float(max(src_w, src_h))
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
        # Even dimensions are required by H.264 / yuv420p.
        out_w -= out_w % 2
        out_h -= out_h % 2
        timeline_h = self.cfg.timeline_height_px
        canvas_h = out_h + timeline_h

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer, codec = self._open_writer(out_path, src_fps, out_w, canvas_h)
        if writer is None:
            cap.release()
            return None

        frame_lookup = df["frame_index"].to_numpy(dtype=int)
        seg_of_row = self._segment_per_row(df, segments)
        trail: Deque[Tuple[int, int]] = deque()
        trail_len = max(2, int(round(self.cfg.draw_trail_seconds * src_fps)))

        i = 0
        written = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if max_frames is not None and written >= max_frames:
                    break

                if scale != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h),
                                       interpolation=cv2.INTER_AREA)
                else:
                    frame = frame[:out_h, :out_w]

                row_i = self._nearest_row(frame_lookup, i)
                t = i / src_fps
                seg = seg_of_row[row_i] if row_i is not None else None

                if row_i is not None:
                    self._draw_frame(
                        frame, df.iloc[row_i], scale, trail, trail_len,
                        object_available
                    )
                self._draw_hud(frame, t, seg, task_label, task_confidence,
                               df.iloc[row_i] if row_i is not None else None)

                canvas = np.full((canvas_h, out_w, 3), PANEL, dtype=np.uint8)
                canvas[:out_h] = frame
                self._draw_timeline(
                    canvas[out_h:], segments, t, max(total_duration, 1e-6)
                )
                writer.write(canvas)

                i += 1
                written += 1
                if progress_cb is not None and i % 15 == 0 and n_src > 0:
                    progress_cb(i, n_src, "render")
        finally:
            writer.release()
            cap.release()

        if progress_cb is not None:
            progress_cb(written, max(n_src, written), "render")

        final = out_path
        if self.cfg.remux_with_ffmpeg:
            remuxed = self._remux(out_path)
            if remuxed is not None:
                final = remuxed
        return final

    # -- writer ----------------------------------------------------------- #
    def _open_writer(self, path: Path, fps: float, w: int, h: int):
        for tag in self.cfg.codec_preference:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*tag), fps, (w, h)
            )
            if writer.isOpened():
                return writer, tag
            writer.release()
        return None, None

    @staticmethod
    def _remux(path: Path) -> Optional[Path]:
        """Re-encode to baseline H.264 + yuv420p so browsers can play it.

        OpenCV's mp4 output is frequently not playable in Chrome/Edge; ffmpeg,
        when present, fixes that. Failure here is non-fatal.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None
        tmp = path.with_name(path.stem + "_h264.mp4")
        cmd = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=900)
        except Exception:
            return None
        if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
            tmp.unlink(missing_ok=True)
            return None
        path.unlink(missing_ok=True)
        tmp.replace(path)
        return path

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _segment_per_row(
        df: pd.DataFrame, segments: List[Segment]
    ) -> List[Optional[Segment]]:
        out: List[Optional[Segment]] = [None] * len(df)
        for s in segments:
            for r in range(s.start_index, min(s.end_index, len(df))):
                out[r] = s
        return out

    @staticmethod
    def _nearest_row(frame_lookup: np.ndarray, frame_index: int) -> Optional[int]:
        if frame_lookup.size == 0:
            return None
        pos = int(np.searchsorted(frame_lookup, frame_index))
        if pos >= frame_lookup.size:
            return int(frame_lookup.size - 1)
        if pos > 0 and abs(frame_lookup[pos - 1] - frame_index) <= abs(
            frame_lookup[pos] - frame_index
        ):
            return pos - 1
        return pos

    @staticmethod
    def _pt(row, xk: str, yk: str, scale: float) -> Optional[Tuple[int, int]]:
        x, y = row.get(xk), row.get(yk)
        if x is None or y is None:
            return None
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if np.isnan(fx) or np.isnan(fy):
            return None
        return int(round(fx * scale)), int(round(fy * scale))

    def _draw_frame(
        self,
        frame: np.ndarray,
        row,
        scale: float,
        trail: Deque[Tuple[int, int]],
        trail_len: int,
        object_available: bool,
    ) -> None:
        sh = self._pt(row, "shoulder_x", "shoulder_y", scale)
        el = self._pt(row, "elbow_x", "elbow_y", scale)
        wr = self._pt(row, "wrist_x", "wrist_y", scale)
        hd = self._pt(row, "hand_x", "hand_y", scale)

        if object_available:
            ox, oy = row.get("object_x"), row.get("object_y")
            ow, oh = row.get("object_w"), row.get("object_h")
            if not any(
                v is None or np.isnan(float(v)) for v in (ox, oy, ow, oh)
            ):
                x1 = int(round((float(ox) - float(ow) / 2) * scale))
                y1 = int(round((float(oy) - float(oh) / 2) * scale))
                x2 = int(round((float(ox) + float(ow) / 2) * scale))
                y2 = int(round((float(oy) + float(oh) / 2) * scale))
                contact = bool(row.get("contact", False))
                col = (60, 200, 90) if contact else (200, 200, 60)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                           4, col, -1)
                cv2.putText(frame, "OBJECT" + (" (contact)" if contact else ""),
                            (x1, max(14, y1 - 6)), FONT, 0.45, col, 1, cv2.LINE_AA)
                if wr is not None:
                    cv2.line(frame, wr,
                             (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                             (120, 120, 200), 1, cv2.LINE_AA)

        for a, b in ((sh, el), (el, wr)):
            if a is not None and b is not None:
                cv2.line(frame, a, b, (245, 245, 245), 3, cv2.LINE_AA)
        for p, r, col in ((sh, 5, (255, 190, 90)), (el, 5, (255, 150, 120))):
            if p is not None:
                cv2.circle(frame, p, r, col, -1, cv2.LINE_AA)

        if wr is not None:
            trail.append(wr)
            while len(trail) > trail_len:
                trail.popleft()
            pts = list(trail)
            for k in range(1, len(pts)):
                alpha = k / max(len(pts) - 1, 1)
                col = (int(60 + 120 * alpha), int(90 + 110 * alpha), 255)
                cv2.line(frame, pts[k - 1], pts[k], col, 2, cv2.LINE_AA)
            closed = str(row.get("hand_state", "UNKNOWN")) == "CLOSED"
            cv2.circle(frame, wr, 8, (40, 60, 240) if closed else (80, 220, 240),
                       -1, cv2.LINE_AA)
            cv2.circle(frame, wr, 8, BLACK, 1, cv2.LINE_AA)

        if hd is not None and (wr is None or abs(hd[0] - wr[0]) + abs(hd[1] - wr[1]) > 6):
            cv2.circle(frame, hd, 5, (180, 255, 180), 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        t: float,
        seg: Optional[Segment],
        task: str,
        task_conf: float,
        row,
    ) -> None:
        h, w = frame.shape[:2]
        fs = self.cfg.font_scale
        lines = [
            f"t = {format_timestamp(t)}",
            f"TASK: {task}  ({task_conf:.2f})",
        ]
        if seg is not None:
            lines.append(
                f"PHASE {seg.id}: {seg.label}  ({seg.confidence:.2f})"
            )
            lines.append(
                f"  {format_timestamp(seg.start_time)} - "
                f"{format_timestamp(seg.end_time)}"
            )
        else:
            lines.append("PHASE: (not analysed)")
        if row is not None:
            state = str(row.get("hand_state", "UNKNOWN"))
            spd = row.get("wrist_speed")
            spd_s = "n/a" if spd is None or np.isnan(float(spd)) else f"{float(spd):.2f}"
            lines.append(f"HAND: {state}   |v|: {spd_s} bw/s")

        pad = 8
        box_w = 0
        for ln in lines:
            box_w = max(box_w, cv2.getTextSize(ln, FONT, fs, 1)[0][0])
        box_w = min(box_w + 2 * pad, w - 12)
        box_h = int(len(lines) * 22 * fs / 0.6) + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        colour = PHASE_COLORS_BGR.get(seg.label if seg else "UNKNOWN", WHITE)
        y = 8 + pad + int(14 * fs / 0.6)
        for k, ln in enumerate(lines):
            cv2.putText(frame, ln, (8 + pad, y), FONT, fs,
                        colour if k == 2 else WHITE, 1, cv2.LINE_AA)
            y += int(22 * fs / 0.6)

    def _draw_timeline(
        self,
        strip: np.ndarray,
        segments: List[Segment],
        t: float,
        total: float,
    ) -> None:
        h, w = strip.shape[:2]
        bar_top, bar_bot = 26, h - 22
        cv2.rectangle(strip, (0, 0), (w, h), PANEL, -1)
        cv2.putText(strip, "SEGMENTATION TIMELINE", (8, 17), FONT, 0.42,
                    (190, 190, 190), 1, cv2.LINE_AA)

        for s in segments:
            x1 = int(round(s.start_time / total * (w - 2))) + 1
            x2 = int(round(s.end_time / total * (w - 2))) + 1
            x2 = max(x2, x1 + 1)
            col = PHASE_COLORS_BGR.get(s.label, PHASE_COLORS_BGR["UNKNOWN"])
            cv2.rectangle(strip, (x1, bar_top), (x2, bar_bot), col, -1)
            cv2.rectangle(strip, (x1, bar_top), (x2, bar_bot), (20, 20, 20), 1)
            label = s.label
            tw = cv2.getTextSize(label, FONT, 0.36, 1)[0][0]
            if tw < (x2 - x1) - 4:
                cv2.putText(strip, label, (x1 + 3, bar_bot - 6), FONT, 0.36,
                            BLACK, 1, cv2.LINE_AA)

        px = int(round(t / total * (w - 2))) + 1
        cv2.line(strip, (px, bar_top - 6), (px, bar_bot + 6), WHITE, 2, cv2.LINE_AA)
        cv2.putText(strip, format_timestamp(t), (max(2, px - 26), h - 6), FONT,
                    0.38, WHITE, 1, cv2.LINE_AA)
        cv2.putText(strip, format_timestamp(total), (w - 58, 17), FONT, 0.38,
                    (170, 170, 170), 1, cv2.LINE_AA)
