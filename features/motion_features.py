"""Motion feature computation (Step 4a of the pipeline).

The whole point of this module is to turn raw per-frame detections into
**scale-invariant** kinematics. Every speed is divided by the subject's
shoulder width in pixels, so the units are *body-widths per second* and the
same thresholds work for a 480p clip filmed from 3 m away and a 4K clip filmed
from 1 m away.

Output: one tidy :class:`pandas.DataFrame`, one row per processed frame. This
frame is the only thing the segmentation and labelling stages see, which is
what keeps those stages swappable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pose.pose_extractor import PoseFrame, PoseSequence
from tracking.object_tracker import Track
from utils import signal_utils as su
from utils.config import FeatureConfig, PoseConfig

HAND_STATE_OPEN = "OPEN"
HAND_STATE_CLOSED = "CLOSED"
HAND_STATE_UNKNOWN = "UNKNOWN"


@dataclass
class FeatureTable:
    df: pd.DataFrame
    body_scale_px: float
    active_arm: str
    object_available: bool
    grip_available: bool
    coordinate_space: str = "2D_image"
    depth_metric_available: bool = False
    units: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def fps_effective(self) -> float:
        t = self.df["timestamp"].to_numpy()
        if len(t) < 2:
            return 0.0
        dt = float(np.median(np.diff(t)))
        return 1.0 / dt if dt > 0 else 0.0


class MotionFeatureExtractor:
    """Builds the feature table from a pose sequence and an optional object track."""

    def __init__(self, feat_cfg: FeatureConfig, pose_cfg: PoseConfig) -> None:
        self.cfg = feat_cfg
        self.pose_cfg = pose_cfg

    # -- public ----------------------------------------------------------- #
    def build(
        self,
        seq: PoseSequence,
        object_track: Optional[Track] = None,
    ) -> FeatureTable:
        frames = seq.frames
        if not frames:
            raise ValueError("No frames were processed; cannot build features.")

        notes: List[str] = []
        arm = seq.active_arm if seq.active_arm in ("left", "right") else "right"
        if seq.active_arm == "none":
            notes.append(
                "Active arm could not be determined from motion; defaulting to "
                "the right arm. Treat kinematics as low confidence."
            )

        t = np.array([f.timestamp for f in frames], dtype=float)
        fi = np.array([f.frame_index for f in frames], dtype=int)
        dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30.0
        if dt_med <= 0:
            dt_med = 1 / 30.0
        win_frames = max(3, int(round(self.cfg.smooth_window_s / dt_med)))
        max_gap = max(1, int(round(self.pose_cfg.max_interpolation_gap_s / dt_med)))

        body_scale_series = self._body_scale(frames)
        body_scale = su.robust_nanmedian(
            body_scale_series, default=self.cfg.fallback_scale_px
        )
        if not np.isfinite(body_scale) or body_scale < 1e-3:
            body_scale = self.cfg.fallback_scale_px
            notes.append(
                "Shoulder width was never measurable; using a fallback body "
                "scale, so normalised speeds are approximate."
            )

        df = pd.DataFrame({"frame_index": fi, "timestamp": t})
        df["body_scale_px"] = body_scale_series

        # ---- raw joint positions (source-frame pixels, 2D only) ---------- #
        for out_name, kp_name in (
            ("wrist", f"{arm}_wrist"),
            ("elbow", f"{arm}_elbow"),
            ("shoulder", f"{arm}_shoulder"),
        ):
            xs = np.array([f.kp(kp_name).x for f in frames], dtype=float)
            ys = np.array([f.kp(kp_name).y for f in frames], dtype=float)
            xs = su.interpolate_gaps(xs, max_gap)
            ys = su.interpolate_gaps(ys, max_gap)
            df[f"{out_name}_x"] = xs
            df[f"{out_name}_y"] = ys
            df[f"{out_name}_x_s"] = su.smooth(xs, win_frames, self.cfg.savgol_polyorder)
            df[f"{out_name}_y_s"] = su.smooth(ys, win_frames, self.cfg.savgol_polyorder)

        hx = np.array(
            [f.hand_center.get(arm, (np.nan, np.nan))[0] for f in frames], dtype=float
        )
        hy = np.array(
            [f.hand_center.get(arm, (np.nan, np.nan))[1] for f in frames], dtype=float
        )
        df["hand_x"] = su.interpolate_gaps(hx, max_gap)
        df["hand_y"] = su.interpolate_gaps(hy, max_gap)

        df["pose_confidence"] = [f.pose_confidence for f in frames]
        df["wrist_visibility"] = [f.kp(f"{arm}_wrist").visibility for f in frames]
        df["hand_confidence"] = [f.hand_confidence.get(arm, 0.0) for f in frames]
        # Relative depth is preserved but never treated as metric 3D.
        df["wrist_z_relative"] = [f.kp(f"{arm}_wrist").z_relative for f in frames]

        # ---- grip ------------------------------------------------------- #
        grip_ratio, grip_available, grip_note = self._grip_ratio(frames, arm, win_frames)
        df["grip_ratio"] = grip_ratio
        # Rate of aperture change. The *interval* over which the aperture moves
        # is what delimits a GRASP or RELEASE phase; a single threshold crossing
        # only marks its midpoint.
        df["grip_rate"] = su.derivative(grip_ratio, t)
        if grip_note:
            notes.append(grip_note)
        df["grip_source"] = [f.aperture_source.get(arm, "") for f in frames]

        if grip_available:
            closed = su.hysteresis_threshold(
                grip_ratio,
                low=self.pose_cfg.grip_closed_ratio,
                high=self.pose_cfg.grip_open_ratio,
            )
            # dtype=object, not the <U6 array np.where would infer from
            # "CLOSED"/"OPEN": assigning the longer "UNKNOWN" into a fixed-width
            # numpy string array truncates it silently to "UNKNOW".
            state = np.where(closed, HAND_STATE_CLOSED, HAND_STATE_OPEN).astype(object)
            state[np.isnan(grip_ratio)] = HAND_STATE_UNKNOWN
        else:
            closed = np.zeros(len(frames), dtype=bool)
            state = np.array([HAND_STATE_UNKNOWN] * len(frames), dtype=object)
        df["hand_state"] = state
        df["grip_closed"] = closed & grip_available

        # ---- kinematics, normalised to body-widths ---------------------- #
        wx = df["wrist_x_s"].to_numpy()
        wy = df["wrist_y_s"].to_numpy()
        vx = su.derivative(wx, t) / body_scale
        # Image y grows downward; negate so positive means upward in the world.
        vy = -su.derivative(wy, t) / body_scale
        speed = np.hypot(vx, vy)
        df["wrist_vx"] = vx
        df["wrist_vy"] = vy
        df["wrist_speed"] = su.smooth(speed, win_frames, self.cfg.savgol_polyorder)
        df["wrist_accel"] = np.abs(su.derivative(df["wrist_speed"].to_numpy(), t))
        df["wrist_heading_change"] = self._heading_change(vx, vy, t, win_frames)
        df["elbow_angle_deg"] = self._elbow_angle(df)

        # ---- object ----------------------------------------------------- #
        object_available = object_track is not None
        if object_available:
            ox, oy, ow, oh, oc = self._object_arrays(object_track, fi, max_gap)
        else:
            nan = np.full(len(frames), np.nan)
            ox, oy, ow, oh, oc = nan, nan.copy(), nan.copy(), nan.copy(), nan.copy()
        df["object_x"] = ox
        df["object_y"] = oy
        df["object_w"] = ow
        df["object_h"] = oh
        df["object_confidence"] = oc

        oxs = su.smooth(ox, win_frames, self.cfg.savgol_polyorder)
        oys = su.smooth(oy, win_frames, self.cfg.savgol_polyorder)
        ovx = su.derivative(oxs, t) / body_scale
        ovy = -su.derivative(oys, t) / body_scale
        df["obj_vx"] = ovx
        df["obj_vy"] = ovy
        df["obj_speed"] = su.smooth(
            np.hypot(ovx, ovy), win_frames, self.cfg.savgol_polyorder
        )

        # Hand-object geometry. The relative vector is the quantity an LfD
        # policy should generalise over, so it is stored explicitly.
        ref_x = np.where(np.isnan(df["hand_x"]), wx, df["hand_x"])
        ref_y = np.where(np.isnan(df["hand_y"]), wy, df["hand_y"])
        rel_x = (ref_x - oxs) / body_scale
        rel_y = -(ref_y - oys) / body_scale
        df["hand_obj_rel_x"] = rel_x
        df["hand_obj_rel_y"] = rel_y
        dist = np.hypot(rel_x, rel_y)
        df["hand_obj_dist"] = su.smooth(dist, win_frames, self.cfg.savgol_polyorder)
        df["hand_obj_dist_rate"] = su.derivative(
            df["hand_obj_dist"].to_numpy(), t
        )
        df["hand_obj_rel_speed"] = np.hypot(vx - ovx, vy - ovy)

        # ---- discrete states -------------------------------------------- #
        df["hand_moving"] = df["wrist_speed"] > self.cfg.wrist_moving_speed_bw
        df["object_moving"] = df["obj_speed"] > self.cfg.object_moving_speed_bw
        near = df["hand_obj_dist"].to_numpy() < self.cfg.contact_distance_bw
        near = np.where(np.isnan(df["hand_obj_dist"].to_numpy()), False, near)
        df["near_object"] = near

        if object_available and grip_available:
            contact = near & (df["grip_closed"].to_numpy()
                              | df["object_moving"].to_numpy())
            contact_basis = "grip_and_proximity"
        elif object_available:
            contact = near & df["object_moving"].to_numpy()
            contact_basis = "proximity_and_object_motion"
        elif grip_available:
            contact = df["grip_closed"].to_numpy()
            contact_basis = "grip_only"
        else:
            contact = np.zeros(len(frames), dtype=bool)
            contact_basis = "unavailable"
            notes.append(
                "Neither a reliable object track nor a reliable grip signal was "
                "available; contact-dependent phases (GRASP, RELEASE) cannot be "
                "asserted."
            )
        df["contact"] = self._fill_short_gaps(contact, max_gap)
        df["contact_basis"] = contact_basis

        units = {
            "positions": "pixels of the source frame (2D image plane)",
            "velocities": "body-widths per second (shoulder-width normalised)",
            "distances": "body-widths",
            "angles": "degrees",
            "grip_ratio": "0 = fully closed, 1 = fully open (per-video normalised)",
            "timestamps": "seconds from the start of the video",
        }
        return FeatureTable(
            df=df,
            body_scale_px=float(body_scale),
            active_arm=arm,
            object_available=object_available,
            grip_available=grip_available,
            coordinate_space=seq.coordinate_space,
            depth_metric_available=seq.depth_metric_available,
            units=units,
            notes=notes,
        )

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _body_scale(frames: List[PoseFrame]) -> np.ndarray:
        """Per-frame shoulder width in px, with a torso-height fallback."""
        out = np.full(len(frames), np.nan)
        for i, f in enumerate(frames):
            ls, rs = f.kp("left_shoulder"), f.kp("right_shoulder")
            if ls.valid and rs.valid:
                out[i] = math.dist((ls.x, ls.y), (rs.x, rs.y))
                continue
            # One shoulder visible: use shoulder-to-hip and rescale, since
            # torso height is roughly 1.5x shoulder width for adults.
            for side in ("left", "right"):
                sh, hip = f.kp(f"{side}_shoulder"), f.kp(f"{side}_hip")
                if sh.valid and hip.valid:
                    out[i] = math.dist((sh.x, sh.y), (hip.x, hip.y)) / 1.5
                    break
        return out

    def _grip_ratio(
        self, frames: List[PoseFrame], arm: str, win: int
    ) -> Tuple[np.ndarray, bool, Optional[str]]:
        """Normalise raw aperture into 0..1 using this video's own range.

        Per-video normalisation is used because absolute aperture depends on the
        person's hand proportions and the grasp type. If the observed range is
        too narrow, the hand never demonstrably opened or closed, and the signal
        is reported as unavailable instead of being thresholded into noise.
        """
        raw = np.array(
            [f.hand_aperture.get(arm, np.nan) for f in frames], dtype=float
        )
        valid = ~np.isnan(raw)
        if valid.sum() < max(5, 0.1 * len(frames)):
            return (
                np.full(len(frames), np.nan),
                False,
                "Hand aperture was measurable in too few frames; hand open/closed "
                "state is reported as UNKNOWN.",
            )

        lo = float(np.nanpercentile(raw, 5))
        hi = float(np.nanpercentile(raw, 95))
        if (hi - lo) < 0.12:
            return (
                np.full(len(frames), np.nan),
                False,
                f"Hand aperture varied by only {hi - lo:.3f} over the clip, which "
                f"is below the threshold for a reliable open/close decision; "
                f"hand state is reported as UNKNOWN.",
            )

        ratio = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        ratio = su.interpolate_gaps(ratio, max_gap=win)
        return su.smooth(ratio, win, 2), True, None

    @staticmethod
    def _heading_change(
        vx: np.ndarray, vy: np.ndarray, t: np.ndarray, win: int
    ) -> np.ndarray:
        """Rate of change of movement direction (deg/s).

        Direction reversals are among the strongest cues for a phase boundary --
        e.g. the descent-to-ascent turn at the moment of grasp.
        """
        n = len(vx)
        heading = np.arctan2(vy, vx)
        out = np.full(n, np.nan)
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            if dt <= 0 or np.isnan(heading[i]) or np.isnan(heading[i - 1]):
                continue
            d = heading[i] - heading[i - 1]
            d = (d + np.pi) % (2 * np.pi) - np.pi     # wrap to [-pi, pi]
            out[i] = abs(math.degrees(d) / dt)
        # Heading is meaningless when almost stationary; suppress that noise.
        speed = np.hypot(vx, vy)
        out = np.where(speed < 0.05, 0.0, out)
        return su.smooth(out, win, 2)

    @staticmethod
    def _elbow_angle(df: pd.DataFrame) -> np.ndarray:
        sx, sy = df["shoulder_x_s"].to_numpy(), df["shoulder_y_s"].to_numpy()
        ex, ey = df["elbow_x_s"].to_numpy(), df["elbow_y_s"].to_numpy()
        wx, wy = df["wrist_x_s"].to_numpy(), df["wrist_y_s"].to_numpy()
        u = np.stack([sx - ex, sy - ey], axis=1)
        v = np.stack([wx - ex, wy - ey], axis=1)
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.einsum("ij,ij->i", u, v) / (nu * nv)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    @staticmethod
    def _object_arrays(track: Track, frame_indices: np.ndarray, max_gap: int):
        lookup = {o.frame_index: o for o in track.observations}
        n = len(frame_indices)
        cx = np.full(n, np.nan)
        cy = np.full(n, np.nan)
        w = np.full(n, np.nan)
        h = np.full(n, np.nan)
        conf = np.full(n, np.nan)
        for i, fidx in enumerate(frame_indices):
            o = lookup.get(int(fidx))
            if o is None:
                continue
            x1, y1, x2, y2 = o.bbox
            cx[i], cy[i] = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            w[i], h[i] = x2 - x1, y2 - y1
            conf[i] = o.confidence
        return (
            su.interpolate_gaps(cx, max_gap),
            su.interpolate_gaps(cy, max_gap),
            su.interpolate_gaps(w, max_gap),
            su.interpolate_gaps(h, max_gap),
            conf,
        )

    @staticmethod
    def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        """Close single-frame dropouts in a boolean state.

        A one-frame detection miss should not split a contact interval into two,
        which would otherwise create spurious GRASP/RELEASE events.
        """
        m = np.asarray(mask, dtype=bool).copy()
        n = m.size
        i = 0
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                m[i:j] = True
            i = max(j, i + 1)
        return m
