"""Human pose and hand extraction (Step 2 of the pipeline).

Design notes
------------
* Coordinates are reported in **pixels of the source frame** and explicitly
  tagged ``2D_image``. MediaPipe also emits a ``z`` channel, but it is a
  torso-normalised *relative* depth with no metric meaning and no calibration
  against this camera, so it is stored separately as ``z_relative`` and the
  representation records ``depth_metric_available = False``. No 3D coordinate
  is ever fabricated.
* Landmarks below the configured visibility are written as NaN rather than
  being carried forward, so downstream code can distinguish "not moving" from
  "not observed".
* Two MediaPipe backends are supported. ``solutions`` is the legacy
  ``mp.solutions.pose``/``.hands`` API (no model download). ``tasks`` is the
  newer ``mediapipe.tasks`` API, which is the only one present in mediapipe
  >= 0.10.30 builds; it fetches ``.task`` bundles into ``models/`` on first use.
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.config import MODEL_DIR, PoseConfig

# MediaPipe BlazePose landmark indices used here.
POSE_IDX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
}

# MediaPipe Hands landmark indices.
HAND_WRIST = 0
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_MCP = 9
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20
FINGERTIPS = (HAND_INDEX_TIP, HAND_MIDDLE_TIP, HAND_RING_TIP, HAND_PINKY_TIP)

_TASK_MODEL_URLS = {
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


@dataclass
class Keypoint:
    x: float = float("nan")          # source-frame pixels
    y: float = float("nan")
    visibility: float = 0.0
    z_relative: float = float("nan")  # unitless, NOT metric depth

    @property
    def valid(self) -> bool:
        return not (math.isnan(self.x) or math.isnan(self.y))


@dataclass
class PoseFrame:
    """Per-frame human state. All spatial values are 2D source-frame pixels."""
    frame_index: int
    timestamp: float
    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    pose_detected: bool = False
    pose_confidence: float = 0.0
    # Raw hand aperture (mean fingertip-to-palm distance / palm length).
    # Normalised into a 0..1 grip ratio later, using this video's own range.
    hand_aperture: Dict[str, float] = field(default_factory=dict)
    hand_center: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    hand_confidence: Dict[str, float] = field(default_factory=dict)
    # "hands" when derived from the 21-landmark hand model, "pose" when it fell
    # back to the coarse pose thumb/index landmarks, "" when unavailable.
    aperture_source: Dict[str, str] = field(default_factory=dict)

    def kp(self, name: str) -> Keypoint:
        return self.keypoints.get(name, Keypoint())


@dataclass
class PoseSequence:
    frames: List[PoseFrame]
    active_arm: str                # "left" | "right" | "none"
    coordinate_space: str = "2D_image"
    depth_metric_available: bool = False
    backend: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.pose_detected for f in self.frames) / len(self.frames)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _BaseBackend:
    name = "base"

    def pose(self, rgb: np.ndarray, timestamp_ms: int):
        raise NotImplementedError

    def hands(self, rgb: np.ndarray, timestamp_ms: int):
        raise NotImplementedError

    def close(self) -> None:
        pass


class _SolutionsBackend(_BaseBackend):
    """Legacy ``mp.solutions`` API. Returns normalised-landmark lists."""

    name = "mediapipe.solutions"

    def __init__(self, cfg: PoseConfig) -> None:
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=cfg.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=min(1, cfg.model_complexity),
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def pose(self, rgb, timestamp_ms):
        res = self._pose.process(rgb)
        if res.pose_landmarks is None:
            return None
        return [
            (lm.x, lm.y, getattr(lm, "z", float("nan")),
             getattr(lm, "visibility", 1.0))
            for lm in res.pose_landmarks.landmark
        ]

    def hands(self, rgb, timestamp_ms):
        res = self._hands.process(rgb)
        if res.multi_hand_landmarks is None:
            return []
        out = []
        handed = res.multi_handedness or []
        for i, lms in enumerate(res.multi_hand_landmarks):
            label, score = "unknown", 0.0
            if i < len(handed) and handed[i].classification:
                cl = handed[i].classification[0]
                label, score = cl.label.lower(), float(cl.score)
            pts = [(lm.x, lm.y) for lm in lms.landmark]
            out.append({"label": label, "score": score, "points": pts})
        return out

    def close(self) -> None:
        try:
            self._pose.close()
            self._hands.close()
        except Exception:
            pass


class _TasksBackend(_BaseBackend):
    """``mediapipe.tasks`` API for builds where ``mp.solutions`` was removed."""

    name = "mediapipe.tasks"

    def __init__(self, cfg: PoseConfig) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        self._mp = mp
        self._vision = vision
        pose_model = _ensure_task_model("pose")
        hand_model = _ensure_task_model("hand")

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=cfg.max_num_hands,
                min_hand_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
            )
        )
        self._last_ts = -1

    def _image(self, rgb):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def _mono(self, timestamp_ms: int) -> int:
        # The VIDEO running mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def pose(self, rgb, timestamp_ms):
        res = self._pose.detect_for_video(self._image(rgb), self._mono(timestamp_ms))
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        return [
            (lm.x, lm.y, getattr(lm, "z", float("nan")),
             getattr(lm, "visibility", 1.0) or 1.0)
            for lm in lms
        ]

    def hands(self, rgb, timestamp_ms):
        res = self._hands.detect_for_video(self._image(rgb), self._mono(timestamp_ms))
        if not res.hand_landmarks:
            return []
        out = []
        for i, lms in enumerate(res.hand_landmarks):
            label, score = "unknown", 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                c = res.handedness[i][0]
                label = str(c.category_name or "").lower()
                score = float(c.score or 0.0)
            out.append({
                "label": label,
                "score": score,
                "points": [(lm.x, lm.y) for lm in lms],
            })
        return out

    def close(self) -> None:
        for obj in (getattr(self, "_pose", None), getattr(self, "_hands", None)):
            try:
                obj.close()
            except Exception:
                pass


def _ensure_task_model(kind: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = _TASK_MODEL_URLS[kind]
    dest = MODEL_DIR / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed Google CDN URL
    return dest


def _make_backend(cfg: PoseConfig) -> _BaseBackend:
    want = (cfg.backend or "auto").lower()
    errors: List[str] = []
    order = {
        "auto": ["solutions", "tasks"],
        "solutions": ["solutions"],
        "tasks": ["tasks"],
    }.get(want, ["solutions", "tasks"])

    for kind in order:
        try:
            if kind == "solutions":
                import mediapipe as mp
                if not hasattr(mp, "solutions"):
                    raise RuntimeError(
                        "this mediapipe build has no 'solutions' module"
                    )
                return _SolutionsBackend(cfg)
            return _TasksBackend(cfg)
        except Exception as exc:  # noqa: BLE001 - report all attempts together
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No usable MediaPipe backend. Attempts -> " + " | ".join(errors)
    )


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class PoseExtractor:
    """Turns frames into a :class:`PoseSequence`.

    Input:  iterable of ``(frame_index, timestamp_s, bgr_frame)`` plus a
            preprocessor that yields the RGB frame actually fed to the model.
    Output: :class:`PoseSequence` with per-frame 2D keypoints in source pixels.
    """

    def __init__(self, cfg: PoseConfig) -> None:
        self.cfg = cfg
        self.backend = _make_backend(cfg)
        self.notes: List[str] = []

    # -- helpers ---------------------------------------------------------- #
    def _landmark_to_kp(
        self, lm: Tuple[float, float, float, float], w: int, h: int, scale: float
    ) -> Keypoint:
        x, y, z, vis = lm
        if vis < self.cfg.min_landmark_visibility:
            return Keypoint(visibility=float(vis))
        # Normalised -> processed px -> source px.
        return Keypoint(
            x=float(x * w / scale),
            y=float(y * h / scale),
            visibility=float(vis),
            z_relative=float(z) if z is not None and not math.isnan(z) else float("nan"),
        )

    @staticmethod
    def _aperture_from_hand(points: List[Tuple[float, float]], w: int, h: int) -> float:
        """Mean fingertip-to-palm distance divided by palm length.

        Scale-free by construction, so it does not change when the hand moves
        towards or away from the camera.
        """
        px = np.array([(p[0] * w, p[1] * h) for p in points], dtype=float)
        palm = float(np.linalg.norm(px[HAND_MIDDLE_MCP] - px[HAND_WRIST]))
        if palm < 1e-6:
            return float("nan")
        tips = [float(np.linalg.norm(px[t] - px[HAND_WRIST])) for t in FINGERTIPS]
        pinch = float(np.linalg.norm(px[HAND_THUMB_TIP] - px[HAND_INDEX_TIP]))
        # Fingertip extension dominates; the thumb-index pinch adds sensitivity
        # to precision grasps where the other fingers stay partly extended.
        return (0.75 * float(np.mean(tips)) + 0.25 * pinch) / palm

    @staticmethod
    def _aperture_from_pose(f: PoseFrame, side: str) -> float:
        """Coarse fallback using the pose model's thumb/index/pinky landmarks."""
        wrist = f.kp(f"{side}_wrist")
        index = f.kp(f"{side}_index")
        thumb = f.kp(f"{side}_thumb")
        pinky = f.kp(f"{side}_pinky")
        if not (wrist.valid and index.valid and thumb.valid):
            return float("nan")
        span = math.dist((wrist.x, wrist.y), (index.x, index.y))
        if span < 1e-6:
            return float("nan")
        pinch = math.dist((thumb.x, thumb.y), (index.x, index.y))
        spread = (
            math.dist((pinky.x, pinky.y), (index.x, index.y)) if pinky.valid else pinch
        )
        return (0.6 * pinch + 0.4 * spread) / span

    # -- main ------------------------------------------------------------- #
    def process_frame(
        self, frame_index: int, timestamp: float, bgr: np.ndarray, preprocessor
    ) -> Tuple[PoseFrame, np.ndarray]:
        """Extract one frame's human state.

        Returns the :class:`PoseFrame` together with the preprocessed BGR frame,
        so the caller can reuse it (for object detection and for overlays)
        without resizing twice.
        """
        proc_bgr, rgb = preprocessor.process(bgr)
        h, w = proc_bgr.shape[:2]
        scale = preprocessor.geometry.scale if preprocessor.geometry else 1.0

        pf = PoseFrame(frame_index=frame_index, timestamp=float(timestamp))

        lms = self.backend.pose(rgb, int(timestamp * 1000))
        if lms is not None:
            pf.pose_detected = True
            vis_of_interest = []
            for name, idx in POSE_IDX.items():
                if idx < len(lms):
                    pf.keypoints[name] = self._landmark_to_kp(lms[idx], w, h, scale)
                    vis_of_interest.append(lms[idx][3])
            pf.pose_confidence = (
                float(np.mean(vis_of_interest)) if vis_of_interest else 0.0
            )

        for hand in self.backend.hands(rgb, int(timestamp * 1000)):
            side = self._assign_hand_side(hand, pf, w, h, scale)
            if side is None:
                continue
            ap = self._aperture_from_hand(hand["points"], w, h)
            if math.isnan(ap):
                continue
            pts = np.array(
                [(p[0] * w / scale, p[1] * h / scale) for p in hand["points"]],
                dtype=float,
            )
            pf.hand_aperture[side] = ap
            pf.hand_center[side] = (
                float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            )
            pf.hand_confidence[side] = float(hand["score"])
            pf.aperture_source[side] = "hands"

        # Fall back to pose-derived aperture only where the hand model failed.
        for side in ("left", "right"):
            if side not in pf.hand_aperture:
                ap = self._aperture_from_pose(pf, side)
                if not math.isnan(ap):
                    pf.hand_aperture[side] = ap
                    pf.hand_confidence[side] = 0.35   # coarse source, low trust
                    pf.aperture_source[side] = "pose"
            wr = pf.kp(f"{side}_wrist")
            if side not in pf.hand_center and wr.valid:
                pf.hand_center[side] = (wr.x, wr.y)

        return pf, proc_bgr

    def finalize(self, out: List[PoseFrame]) -> PoseSequence:
        """Wrap per-frame results, choosing the manipulating arm from the data."""
        arm = self._choose_active_arm(out)
        seq = PoseSequence(
            frames=out,
            active_arm=arm,
            backend=self.backend.name,
            notes=list(self.notes),
        )
        if seq.detection_rate < 0.5:
            seq.notes.append(
                f"Pose detected in only {seq.detection_rate:.0%} of processed "
                f"frames; segmentation confidence will be reduced."
            )
        if arm == "none":
            seq.notes.append(
                "No arm had sufficient visible motion to be treated as the "
                "manipulator."
            )
        return seq

    def extract(
        self,
        frames,
        preprocessor,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        total: Optional[int] = None,
    ) -> PoseSequence:
        """Convenience single-pass extraction (used by tests and the CLI)."""
        out: List[PoseFrame] = []
        for i, (frame_index, ts, bgr) in enumerate(frames):
            pf, _ = self.process_frame(frame_index, ts, bgr, preprocessor)
            out.append(pf)
            if progress_cb is not None and i % 5 == 0:
                progress_cb(i + 1, total or 0, "pose")
        if progress_cb is not None:
            progress_cb(len(out), total or len(out), "pose")
        return self.finalize(out)

    def _assign_hand_side(
        self, hand: Dict, pf: PoseFrame, w: int, h: int, scale: float
    ) -> Optional[str]:
        """Attach a detected hand to the nearer pose wrist.

        MediaPipe handedness assumes a mirrored webcam view, which does not hold
        for third-person demonstration footage, so geometric association to the
        pose wrists is used first and handedness only as a fallback.
        """
        pts = hand["points"]
        hw = (pts[HAND_WRIST][0] * w / scale, pts[HAND_WRIST][1] * h / scale)
        best, best_d = None, float("inf")
        for side in ("left", "right"):
            k = pf.kp(f"{side}_wrist")
            if not k.valid:
                continue
            d = math.dist(hw, (k.x, k.y))
            if d < best_d:
                best, best_d = side, d
        if best is not None:
            # Reject an association that is implausibly far from either wrist.
            limit = 0.35 * math.hypot(w / scale, h / scale)
            if best_d <= limit:
                return best
        label = hand.get("label", "")
        if label in ("left", "right"):
            return label
        return None

    @staticmethod
    def _choose_active_arm(frames: List[PoseFrame]) -> str:
        """Pick the manipulating arm by observed path length, not by assumption."""
        scores: Dict[str, float] = {}
        for side in ("left", "right"):
            pts = [
                (f.kp(f"{side}_wrist").x, f.kp(f"{side}_wrist").y)
                for f in frames
                if f.kp(f"{side}_wrist").valid
            ]
            vis = [
                f.kp(f"{side}_wrist").visibility
                for f in frames
                if f.pose_detected
            ]
            if len(pts) < 5:
                scores[side] = 0.0
                continue
            path = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))
            # Normalise by shoulder width so the two arms are compared fairly
            # regardless of the subject's distance from the camera.
            widths = [
                math.dist(
                    (f.kp("left_shoulder").x, f.kp("left_shoulder").y),
                    (f.kp("right_shoulder").x, f.kp("right_shoulder").y),
                )
                for f in frames
                if f.kp("left_shoulder").valid and f.kp("right_shoulder").valid
            ]
            bw = float(np.median(widths)) if widths else 150.0
            coverage = len(pts) / max(1, len(frames))
            mean_vis = float(np.mean(vis)) if vis else 0.0
            scores[side] = (path / max(bw, 1e-6)) * coverage * max(mean_vis, 0.1)

        if max(scores.values(), default=0.0) <= 1e-6:
            return "none"
        return max(scores, key=scores.get)

    def close(self) -> None:
        self.backend.close()
