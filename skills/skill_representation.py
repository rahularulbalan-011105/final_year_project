"""Structured skill representation and file output (Steps 6-7).

This module is the contract between the perception half of the project and the
learning half. Everything a downstream LfD model needs is here, and everything
here is traceable to an observation:

* ``segments[*].trajectory``  -- raw image-plane hand path, per phase
* ``segments[*].lfd``         -- phase-normalised time, the object-relative hand
                                 vector (the representation that generalises to
                                 a new object pose), and a discrete gripper
                                 command
* ``lfd_interface``           -- the symbolic plan, gripper event times, and an
                                 explicit statement of what is *not* available
                                 (metric depth), so the robot stage cannot
                                 silently treat pixels as metres.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from features.motion_features import FeatureTable
from segmentation.base import Segment
from skills.skill_classifier import SkillResult
from utils.signal_utils import format_timestamp

SCHEMA_VERSION = "1.0"


def _r(v, nd: int = 4):
    """JSON-safe rounding that turns NaN/inf into ``None``."""
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)
    return v


def _pairs(x: np.ndarray, y: np.ndarray, nd: int = 2) -> List[Optional[List[float]]]:
    out: List[Optional[List[float]]] = []
    for a, b in zip(x, y):
        if math.isnan(float(a)) or math.isnan(float(b)):
            out.append(None)   # an unobserved sample stays explicitly missing
        else:
            out.append([round(float(a), nd), round(float(b), nd)])
    return out


def _gripper_command(seg: Segment, grip_available: bool) -> str:
    """Discrete gripper intent for this phase, for the later robot stage."""
    if not grip_available:
        return "UNKNOWN"
    if seg.label == "GRASP":
        return "CLOSE"
    if seg.label == "RELEASE":
        return "OPEN"
    if seg.label in ("LIFT", "TRANSPORT", "PLACE"):
        return "HOLD_CLOSED"
    if seg.label in ("REACH", "APPROACH", "RETRACT", "IDLE"):
        return "HOLD_OPEN"
    return "UNKNOWN"


@dataclass
class SkillRepresentation:
    data: Dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent, ensure_ascii=False)


class RepresentationBuilder:
    """Assembles the JSON document and the three CSV tables."""

    def __init__(
        self,
        features: FeatureTable,
        segments: List[Segment],
        skill: SkillResult,
        video_meta: Dict[str, Any],
        context: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.ft = features
        self.df = features.df
        self.segments = segments
        self.skill = skill
        self.video_meta = video_meta
        self.context = context
        self.diagnostics = diagnostics

    # -- JSON ------------------------------------------------------------- #
    def build(self) -> SkillRepresentation:
        df = self.df
        doc: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": {
                "name": self.skill.task,
                "confidence": _r(self.skill.confidence),
                "matched_template": self.skill.matched_template,
                "rationale": self.skill.rationale,
                "alternatives": self.skill.candidates,
            },
            "video": self.video_meta,
            "extraction": {
                "pose_backend": self.context.get("pose_backend"),
                "active_arm": self.ft.active_arm,
                "coordinate_space": self.ft.coordinate_space,
                "depth_metric_available": self.ft.depth_metric_available,
                "depth_note": (
                    "Monocular RGB input: no metric depth is recoverable. All "
                    "coordinates are 2D image-plane pixels. MediaPipe's relative "
                    "z channel is retained per frame as 'wrist_z_relative' but is "
                    "unitless and uncalibrated, and is not used for any decision."
                ),
                "pose_detection_rate": _r(self.context.get("pose_detection_rate")),
                "grip_signal_available": self.ft.grip_available,
                "contact_basis": self.context.get("contact_basis"),
                "body_scale_px": _r(self.ft.body_scale_px, 2),
                "effective_sample_rate_hz": _r(self.ft.fps_effective, 3),
                "units": self.ft.units,
                "notes": list(self.ft.notes) + list(self.context.get("notes", [])),
            },
            "object": self.context.get("object_info", {"available": False}),
            "segments": [self._segment_doc(s) for s in self.segments],
            "phase_sequence": [s.label for s in self.segments],
            "lfd_interface": self._lfd_interface(),
            "diagnostics": self.diagnostics,
        }
        return SkillRepresentation(doc)

    def _segment_doc(self, seg: Segment) -> Dict[str, Any]:
        df = self.df
        a, b = seg.start_index, seg.end_index
        sl = df.iloc[a:b]
        scale = max(self.ft.body_scale_px, 1e-9)

        t = sl["timestamp"].to_numpy(dtype=float)
        tau = (
            ((t - t[0]) / (t[-1] - t[0])).tolist()
            if len(t) > 1 and t[-1] > t[0]
            else [0.0] * len(t)
        )

        traj: Dict[str, Any] = {
            "timestamps": [round(float(v), 4) for v in t],
            "frame_indices": [int(v) for v in sl["frame_index"].to_numpy()],
            "wrist": _pairs(sl["wrist_x"].to_numpy(), sl["wrist_y"].to_numpy()),
            "elbow": _pairs(sl["elbow_x"].to_numpy(), sl["elbow_y"].to_numpy()),
            "shoulder": _pairs(sl["shoulder_x"].to_numpy(), sl["shoulder_y"].to_numpy()),
            "hand": _pairs(sl["hand_x"].to_numpy(), sl["hand_y"].to_numpy()),
            "hand_state": sl["hand_state"].tolist(),
        }
        if self.ft.object_available:
            traj["object"] = _pairs(
                sl["object_x"].to_numpy(), sl["object_y"].to_numpy()
            )

        lfd: Dict[str, Any] = {
            "phase_normalized_time": [round(float(v), 4) for v in tau],
            "gripper_command": _gripper_command(seg, self.ft.grip_available),
            "wrist_trajectory_image_px": traj["wrist"],
            "mean_velocity_bodywidths_per_s": _r(seg.motion_features.get("mean_velocity")),
            "path_length_bodywidths": _r(seg.motion_features.get("path_length")),
        }
        if self.ft.object_available:
            lfd["hand_relative_to_object_bodywidths"] = _pairs(
                sl["hand_obj_rel_x"].to_numpy(), sl["hand_obj_rel_y"].to_numpy(), nd=4
            )
            lfd["object_frame_note"] = (
                "Object-relative vector (hand - object centre), normalised by "
                "shoulder width. This is the frame an LfD primitive should be "
                "fitted in so the learned motion transfers to a new object pose."
            )
        # Start/end waypoints make primitive fitting and sanity checks easy.
        lfd["start_waypoint"] = self._waypoint(a)
        lfd["end_waypoint"] = self._waypoint(b - 1)

        return {
            "id": seg.id,
            "skill": seg.label,
            "start_time": _r(seg.start_time, 3),
            "end_time": _r(seg.end_time, 3),
            "duration": _r(seg.duration, 3),
            "start_time_hms": format_timestamp(seg.start_time),
            "end_time_hms": format_timestamp(seg.end_time),
            "start_frame": int(sl["frame_index"].iloc[0]) if len(sl) else None,
            "end_frame": int(sl["frame_index"].iloc[-1]) if len(sl) else None,
            "n_frames": seg.n_frames,
            "confidence": _r(seg.confidence),
            "alternatives": [
                {"skill": k, "score": _r(v)} for k, v in seg.alternatives
            ],
            "motion_features": {k: _r(v, 5) for k, v in seg.motion_features.items()},
            "evidence": seg.evidence,
            "trajectory": traj,
            "lfd": lfd,
        }

    def _waypoint(self, i: int) -> Dict[str, Any]:
        row = self.df.iloc[i]
        wp = {
            "timestamp": _r(row["timestamp"], 4),
            "wrist_px": [_r(row["wrist_x"], 2), _r(row["wrist_y"], 2)],
            "hand_state": row["hand_state"],
        }
        if self.ft.object_available:
            wp["object_px"] = [_r(row["object_x"], 2), _r(row["object_y"], 2)]
            wp["hand_relative_to_object_bw"] = [
                _r(row["hand_obj_rel_x"]), _r(row["hand_obj_rel_y"])
            ]
        return wp

    def _lfd_interface(self) -> Dict[str, Any]:
        df = self.df
        events: List[Dict[str, Any]] = []
        if self.ft.grip_available:
            closed = df["grip_closed"].to_numpy(dtype=bool)
            t = df["timestamp"].to_numpy(dtype=float)
            for i in range(1, len(closed)):
                if closed[i] and not closed[i - 1]:
                    events.append({"time": _r(t[i], 3), "event": "GRIPPER_CLOSE"})
                elif closed[i - 1] and not closed[i]:
                    events.append({"time": _r(t[i], 3), "event": "GRIPPER_OPEN"})

        return {
            "phase_sequence": [
                {
                    "id": s.id,
                    "skill": s.label,
                    "start_time": _r(s.start_time, 3),
                    "end_time": _r(s.end_time, 3),
                    "gripper_command": _gripper_command(s, self.ft.grip_available),
                }
                for s in self.segments
            ],
            "gripper_events": events,
            "coordinate_space": "2D_image_pixels",
            "object_centric_frame_available": self.ft.object_available,
            "ready_for": [
                "movement-primitive fitting per phase (DMP / ProMP) on the "
                "phase-normalised trajectories",
                "discrete task-level policy over the phase sequence",
                "frame-level phase supervision for a later state/phase estimator",
            ],
            "blocking_requirements_for_robot_execution": [
                "Metric 3D: lift the 2D trajectories with a depth or stereo "
                "camera, or a calibrated monocular 3D pose model, before mapping "
                "to workspace coordinates.",
                "Camera-to-robot extrinsic calibration.",
                "Gripper aperture calibration: the grip signal here is a "
                "normalised, unitless ratio, not a finger separation in mm.",
            ],
        }

    # -- CSV -------------------------------------------------------------- #
    def segments_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.segments:
            mf = s.motion_features
            rows.append({
                "segment_id": s.id,
                "skill": s.label,
                "start_time": round(s.start_time, 3),
                "end_time": round(s.end_time, 3),
                "duration": round(s.duration, 3),
                "confidence": round(s.confidence, 4),
                "start_time_hms": format_timestamp(s.start_time),
                "end_time_hms": format_timestamp(s.end_time),
                "n_frames": s.n_frames,
                "mean_velocity_bw_s": _r(mf.get("mean_velocity")),
                "max_velocity_bw_s": _r(mf.get("max_velocity")),
                "path_length_bw": _r(mf.get("path_length")),
                "net_dx_bw": _r(mf.get("net_dx")),
                "net_dy_bw": _r(mf.get("net_dy")),
                "contact_fraction": _r(mf.get("contact_fraction")),
                "gripper_command": _gripper_command(s, self.ft.grip_available),
            })
        return pd.DataFrame(rows)

    def trajectory_dataframe(self) -> pd.DataFrame:
        df = self.df
        seg_id = np.zeros(len(df), dtype=int)
        seg_lab = np.array(["UNLABELLED"] * len(df), dtype=object)
        for s in self.segments:
            seg_id[s.start_index:s.end_index] = s.id
            seg_lab[s.start_index:s.end_index] = s.label

        out = pd.DataFrame({
            "timestamp": df["timestamp"].round(4),
            "frame_index": df["frame_index"],
            "wrist_x": df["wrist_x"].round(2),
            "wrist_y": df["wrist_y"].round(2),
            "elbow_x": df["elbow_x"].round(2),
            "elbow_y": df["elbow_y"].round(2),
            "shoulder_x": df["shoulder_x"].round(2),
            "shoulder_y": df["shoulder_y"].round(2),
            "object_x": df["object_x"].round(2),
            "object_y": df["object_y"].round(2),
            "hand_state": df["hand_state"],
            "hand_x": df["hand_x"].round(2),
            "hand_y": df["hand_y"].round(2),
            "grip_ratio": df["grip_ratio"].round(4),
            "wrist_speed_bw_s": df["wrist_speed"].round(5),
            "wrist_accel_bw_s2": df["wrist_accel"].round(5),
            "obj_speed_bw_s": df["obj_speed"].round(5),
            "hand_obj_dist_bw": df["hand_obj_dist"].round(5),
            "hand_obj_rel_x_bw": df["hand_obj_rel_x"].round(5),
            "hand_obj_rel_y_bw": df["hand_obj_rel_y"].round(5),
            "contact": df["contact"],
            "pose_confidence": df["pose_confidence"].round(4),
            "segment_id": seg_id,
            "segment_skill": seg_lab,
            "coordinate_space": "2D_image",
        })
        return out

    def write_all(self, out_dir: Path) -> Dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rep = self.build()

        paths = {
            "json": out_dir / "skill_segments.json",
            "segments_csv": out_dir / "skill_segments.csv",
            "trajectory_csv": out_dir / "trajectory.csv",
        }
        paths["json"].write_text(rep.to_json(), encoding="utf-8")
        self.segments_dataframe().to_csv(paths["segments_csv"], index=False)
        self.trajectory_dataframe().to_csv(paths["trajectory_csv"], index=False)
        return paths
