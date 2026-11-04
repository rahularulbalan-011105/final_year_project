"""Object detection and tracking (Step 3 of the pipeline).

Strategy
--------
1. YOLOv8 (COCO) proposes boxes on every ``detect_every``-th processed frame.
2. A small IoU + centroid tracker stitches detections into tracks, which is
   enough for a short tabletop demonstration and adds no dependency beyond
   ultralytics itself.
3. The *manipuland* is then chosen by how the track behaves relative to the
   hand: proximity, correlated motion, and whether it moves at all. If no track
   scores above the threshold, the pipeline reports
   "Object tracking confidence insufficient." and continues without an object
   rather than nominating an arbitrary box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.config import ObjectConfig


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in source px
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class TrackObservation:
    frame_index: int
    timestamp: float
    bbox: Tuple[float, float, float, float]
    confidence: float
    #: False when the box was carried over from the previous detection frame.
    is_detection: bool = True

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class Track:
    track_id: int
    class_name: str
    observations: List[TrackObservation] = field(default_factory=list)
    age_since_update: int = 0

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def mean_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([o.confidence for o in self.observations]))

    @property
    def detected_fraction(self) -> float:
        """Share of observations that were real detections, not coasted boxes."""
        if not self.observations:
            return 0.0
        return float(np.mean([o.is_detection for o in self.observations]))

    def bbox_at(self, frame_index: int) -> Optional[Tuple[float, float, float, float]]:
        for o in self.observations:
            if o.frame_index == frame_index:
                return o.bbox
        return None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObjectDetector:
    """Thin YOLOv8 wrapper that degrades gracefully when weights are absent."""

    def __init__(self, cfg: ObjectConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.names: Dict[int, str] = {}
        self.unavailable_reason: Optional[str] = None
        if not cfg.enabled:
            self.unavailable_reason = "Object tracking disabled in configuration."
            return
        try:
            from ultralytics import YOLO

            weights = self._resolve_weights(cfg.weights)
            self.model = YOLO(str(weights))
            self.names = dict(self.model.names)
        except Exception as exc:  # noqa: BLE001
            self.unavailable_reason = (
                f"YOLO unavailable ({type(exc).__name__}: {exc}). "
                f"Continuing without object tracking."
            )

    @staticmethod
    def _resolve_weights(name: str) -> str:
        """Prefer an already-downloaded copy so the pipeline works offline."""
        candidates = [
            Path(name),
            Path.cwd() / name,
            Path(__file__).resolve().parent.parent / "models" / name,
            Path.home() / name,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return name  # let ultralytics fetch it

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def target_class(self) -> Optional[str]:
        """The hinted manipuland class, validated against the model vocabulary."""
        t = (self.cfg.target_class or "").strip().lower()
        if not t:
            return None
        return t if t in {str(v).lower() for v in self.names.values()} else None

    def unknown_target(self) -> Optional[str]:
        """The hint as given, when it is not a class this detector knows."""
        t = (self.cfg.target_class or "").strip().lower()
        return t if t and self.target_class is None else None

    def detect(self, bgr: np.ndarray, inv_scale: float) -> List[Detection]:
        """Detect objects; boxes are mapped back to source-frame pixels."""
        if self.model is None:
            return []
        # With a target hint, run at the lower threshold so the named object is
        # recalled even in motion blur; filtering to that class afterwards keeps
        # the extra false positives from reaching the tracker.
        conf = (
            min(self.cfg.target_det_confidence, self.cfg.det_confidence)
            if self.target_class else self.cfg.det_confidence
        )
        res = self.model.predict(
            bgr,
            conf=conf,
            iou=self.cfg.iou_threshold,
            verbose=False,
        )
        out: List[Detection] = []
        if not res:
            return out
        boxes = getattr(res[0], "boxes", None)
        if boxes is None:
            return out
        target = self.target_class
        for b in boxes:
            cls_id = int(b.cls.item())
            name = str(self.names.get(cls_id, cls_id))
            if name in self.cfg.excluded_classes:
                continue
            if target is not None and name.lower() != target:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(
                Detection(
                    bbox=(x1 * inv_scale, y1 * inv_scale,
                          x2 * inv_scale, y2 * inv_scale),
                    confidence=float(b.conf.item()),
                    class_id=cls_id,
                    class_name=name,
                )
            )
        return out


class IoUTracker:
    """Greedy IoU + centroid association.

    A detection matches an existing track when their IoU clears
    ``min_iou_for_match`` or, for fast small objects where boxes stop
    overlapping, when the centre jump stays under a fraction of the frame
    diagonal and the class agrees.
    """

    def __init__(self, cfg: ObjectConfig, frame_diagonal: float) -> None:
        self.cfg = cfg
        self.frame_diagonal = max(frame_diagonal, 1.0)
        self.tracks: List[Track] = []
        self._next_id = 1

    def _match_score(self, track: Track, det: Detection) -> float:
        last = track.observations[-1]
        if track.class_name != det.class_name:
            return -1.0
        i = iou(last.bbox, det.bbox)
        if i >= self.cfg.min_iou_for_match:
            return 1.0 + i
        jump = math.dist(last.center, det.center) / self.frame_diagonal
        if jump <= self.cfg.max_center_jump_frac:
            return 1.0 - jump
        return -1.0

    def update(
        self, detections: List[Detection], frame_index: int, timestamp: float
    ) -> None:
        unmatched = list(range(len(detections)))
        pairs: List[Tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            if not track.observations:
                continue
            for di in unmatched:
                s = self._match_score(track, detections[di])
                if s > 0:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            det = detections[di]
            self.tracks[ti].observations.append(
                TrackObservation(frame_index, timestamp, det.bbox, det.confidence, True)
            )
            self.tracks[ti].age_since_update = 0
            used_t.add(ti)
            used_d.add(di)

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            t = Track(track_id=self._next_id, class_name=det.class_name)
            self._next_id += 1
            t.observations.append(
                TrackObservation(frame_index, timestamp, det.bbox, det.confidence, True)
            )
            self.tracks.append(t)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.age_since_update += 1

    def coast(self, frame_index: int, timestamp: float) -> None:
        """Carry live tracks forward on frames where detection did not run.

        Coasted observations are flagged ``is_detection=False`` so track quality
        is judged only on real detections.
        """
        for track in self.tracks:
            if track.age_since_update > self.cfg.max_track_age_frames:
                continue
            if not track.observations:
                continue
            last = track.observations[-1]
            track.observations.append(
                TrackObservation(
                    frame_index, timestamp, last.bbox, last.confidence, False
                )
            )

    def finished_tracks(self) -> List[Track]:
        return [
            t for t in self.tracks
            if t.length >= self.cfg.min_track_length_frames
        ]


@dataclass
class ManipulandResult:
    """Outcome of selecting the manipulated object."""
    track: Optional[Track]
    score: float
    reliable: bool
    message: str
    candidates: List[Dict] = field(default_factory=list)


class ManipulandSelector:
    """Choose which track the hand is actually manipulating.

    Score components (all in [0, 1], averaged with fixed weights):

    * ``proximity``   - how close the hand gets to the box at its closest
    * ``co_motion``   - correlation between hand speed and object speed while
                        the hand is near the object
    * ``displacement``- how far the object moves overall (a static background
                        object is not a manipuland)
    * ``quality``     - detection density and mean confidence
    """

    WEIGHTS = {
        "proximity": 0.35,
        "co_motion": 0.25,
        "displacement": 0.20,
        "quality": 0.20,
    }

    def __init__(self, cfg: ObjectConfig) -> None:
        self.cfg = cfg

    def select(
        self,
        tracks: List[Track],
        hand_xy: np.ndarray,      # (n, 2) source px, NaN where unobserved
        timestamps: np.ndarray,   # (n,)
        frame_indices: np.ndarray,
        body_scale_px: float,
        target_class: Optional[str] = None,
    ) -> ManipulandResult:
        if not tracks:
            hint = (
                f" No '{target_class}' was detected anywhere in the clip -- check "
                f"the object is visible, unoccluded and well lit."
                if target_class else
                ""
            )
            return ManipulandResult(
                None, 0.0, False,
                "Object tracking confidence insufficient: no object tracks of "
                "sufficient length were detected." + hint,
            )

        scale = max(body_scale_px, 1e-6)
        hand_speed = self._speed(hand_xy, timestamps) / scale
        scored: List[Tuple[float, Track, Dict]] = []

        for track in tracks:
            obj_xy = self._align(track, frame_indices)
            if np.isnan(obj_xy).all():
                continue
            dist = np.linalg.norm(hand_xy - obj_xy, axis=1) / scale
            obj_speed = self._speed(obj_xy, timestamps) / scale

            with np.errstate(invalid="ignore"):
                min_dist = np.nanmin(dist) if np.isfinite(dist).any() else np.nan
            proximity = (
                0.0 if np.isnan(min_dist)
                else float(np.clip(1.0 - min_dist / 1.2, 0.0, 1.0))
            )

            near = dist < (3.0 * self.cfg.min_iou_for_match + 0.6)
            co_motion = self._correlation(hand_speed, obj_speed, near)

            finite = obj_xy[~np.isnan(obj_xy).any(axis=1)]
            displacement = (
                0.0 if len(finite) < 2
                else float(np.clip(
                    np.linalg.norm(finite.max(axis=0) - finite.min(axis=0))
                    / scale / 1.5, 0.0, 1.0))
            )

            density = track.detected_fraction
            quality = float(np.clip(0.5 * density + 0.5 * track.mean_confidence, 0, 1))

            total = (
                self.WEIGHTS["proximity"] * proximity
                + self.WEIGHTS["co_motion"] * co_motion
                + self.WEIGHTS["displacement"] * displacement
                + self.WEIGHTS["quality"] * quality
            )
            # Interaction gate: an object that never moves and never co-moves
            # with the hand is not being manipulated, however close it appears.
            interaction = max(co_motion, displacement)
            info = {
                "track_id": track.track_id,
                "class": track.class_name,
                "score": round(total, 4),
                "proximity": round(proximity, 4),
                "co_motion": round(co_motion, 4),
                "displacement": round(displacement, 4),
                "interaction": round(interaction, 4),
                "quality": round(quality, 4),
                "frames": track.length,
                "min_hand_distance_bodywidths": (
                    None if np.isnan(min_dist) else round(float(min_dist), 4)
                ),
            }
            # An explicit hint is user-supplied evidence of which object is the
            # manipuland, so the inference gates that exist to avoid nominating
            # an arbitrary box do not apply -- only the quality floor does.
            if target_class is not None:
                info["selected_by"] = "user-specified target class"
                scored.append((max(total, self.cfg.min_manipuland_score), track, info))
            elif density < self.cfg.min_track_density:
                info["rejected"] = "detection density below threshold"
                scored.append((-1.0, track, info))
            elif interaction < self.cfg.min_interaction_score:
                info["rejected"] = (
                    f"no interaction evidence (max(co-motion, displacement) = "
                    f"{interaction:.2f} < {self.cfg.min_interaction_score:.2f}); "
                    f"the object neither moved nor moved with the hand"
                )
                scored.append((-1.0, track, info))
            else:
                scored.append((total, track, info))

        scored.sort(key=lambda s: s[0], reverse=True)
        candidates = [info for _, _, info in scored[:6]]
        if not scored or scored[0][0] < self.cfg.min_manipuland_score:
            best = scored[0][0] if scored else 0.0
            reason = ""
            if scored and scored[0][2].get("rejected"):
                reason = " Best candidate was rejected: " + scored[0][2]["rejected"] + "."
            elif candidates:
                rejected = [c for c in candidates if c.get("rejected")]
                if rejected:
                    reason = " " + str(len(rejected)) + " candidate(s) were rejected "
                    reason += "for lack of interaction or detection density."
            return ManipulandResult(
                None, max(best, 0.0), False,
                "Object tracking confidence insufficient: best candidate scored "
                f"{max(best, 0.0):.2f} < {self.cfg.min_manipuland_score:.2f}."
                + reason,
                candidates,
            )

        total, track, _ = scored[0]
        return ManipulandResult(
            track, float(total), True,
            f"Manipulated object: '{track.class_name}' "
            f"(track {track.track_id}, score {total:.2f}).",
            candidates,
        )

    @staticmethod
    def _align(track: Track, frame_indices: np.ndarray) -> np.ndarray:
        """Resample a track's centres onto the pipeline's frame grid."""
        lookup = {o.frame_index: o.center for o in track.observations}
        out = np.full((len(frame_indices), 2), np.nan)
        for i, fi in enumerate(frame_indices):
            c = lookup.get(int(fi))
            if c is not None:
                out[i] = c
        return out

    @staticmethod
    def _speed(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        n = len(xy)
        out = np.full(n, np.nan)
        for i in range(n):
            lo, hi = max(i - 1, 0), min(i + 1, n - 1)
            dt = t[hi] - t[lo]
            if dt <= 0 or np.isnan(xy[hi]).any() or np.isnan(xy[lo]).any():
                continue
            out[i] = float(np.linalg.norm(xy[hi] - xy[lo]) / dt)
        return out

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = mask & ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 5:
            return 0.0
        av, bv = a[m], b[m]
        if np.std(av) < 1e-9 or np.std(bv) < 1e-9:
            return 0.0
        r = float(np.corrcoef(av, bv)[0, 1])
        return float(np.clip(r, 0.0, 1.0))
