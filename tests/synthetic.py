"""Synthetic demonstration generator for validating the analysis stack.

This exists to answer the fair question "is the segmenter actually deriving
boundaries, or is it returning a fixed script?". It builds pose frames and an
object track from a *parameterised* ground-truth phase plan, so the same code
can be asked for a 5-second demo and an 11-second demo with different phase
orders, and the segmenter's output must follow.

Everything below the CV models is exercised end to end: feature extraction,
change-point detection, event detection, phase labelling and skill
classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pose.pose_extractor import Keypoint, PoseFrame, PoseSequence
from tracking.object_tracker import Track, TrackObservation

OPEN_APERTURE = 2.25
CLOSED_APERTURE = 1.05


@dataclass
class Phase:
    label: str
    duration: float
    #: Wrist target at the end of the phase, in pixels.
    target: Tuple[float, float]
    #: Gripper state at the end of the phase: "open" or "closed".
    grip: str
    #: Whether the object is rigidly attached to the hand during this phase.
    carry: bool = False


def min_jerk(s: float) -> float:
    """Minimum-jerk time scaling: smooth start and stop, like human reaching."""
    s = float(np.clip(s, 0.0, 1.0))
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def pick_and_place_plan(
    speed: float = 1.0,
    object_at: Tuple[float, float] = (560.0, 320.0),
    place_at: Tuple[float, float] = (760.0, 320.0),
    lift_height: float = 70.0,
    home: Tuple[float, float] = (380.0, 300.0),
) -> List[Phase]:
    """A canonical pick-and-place plan, stretched by ``speed``."""
    d = lambda x: x / max(speed, 1e-6)  # noqa: E731
    ox, oy = object_at
    px, py = place_at
    return [
        Phase("REACH",     d(1.20), (ox - 20, oy - 2), "open"),
        Phase("APPROACH",  d(0.60), (ox, oy), "open"),
        Phase("GRASP",     d(0.30), (ox, oy), "closed"),
        Phase("LIFT",      d(0.60), (ox, oy - lift_height), "closed", carry=True),
        Phase("TRANSPORT", d(1.30), (px, oy - lift_height), "closed", carry=True),
        Phase("PLACE",     d(0.60), (px, py), "closed", carry=True),
        Phase("RELEASE",   d(0.50), (px, py), "open"),
        Phase("RETRACT",   d(0.70), home, "open"),
    ]


def push_plan(
    speed: float = 1.0,
    object_at: Tuple[float, float] = (540.0, 330.0),
    push_to: Tuple[float, float] = (760.0, 330.0),
    home: Tuple[float, float] = (380.0, 300.0),
) -> List[Phase]:
    """A non-prehensile push: an open flat hand shoves the object.

    The hand starts relaxed (partly closed), flattens out to push, and relaxes
    again afterwards. That is what a real hand does, and it also means the
    aperture signal has genuine range -- a demonstration where the aperture never
    changes at all gives the pipeline no grip evidence to reason with, which is
    covered by its own test.
    """
    d = lambda x: x / max(speed, 1e-6)  # noqa: E731
    ox, oy = object_at
    return [
        Phase("REACH", d(1.10), (ox - 25, oy), "open"),
        Phase("PUSH",  d(1.40), (push_to[0] - 25, push_to[1]), "open", carry=True),
        Phase("RETRACT", d(0.80), home, "closed"),
    ]


@dataclass
class SyntheticDemo:
    sequence: PoseSequence
    track: Track
    ground_truth: List[Dict]
    fps: float

    @property
    def boundaries(self) -> List[float]:
        return [g["start_time"] for g in self.ground_truth[1:]]

    @property
    def labels(self) -> List[str]:
        return [g["label"] for g in self.ground_truth]

    @property
    def duration(self) -> float:
        return self.ground_truth[-1]["end_time"]


def generate(
    plan: Optional[Sequence[Phase]] = None,
    fps: float = 30.0,
    noise_px: float = 0.7,
    object_size: float = 44.0,
    shoulder: Tuple[float, float] = (400.0, 200.0),
    shoulder_width: float = 100.0,
    home: Tuple[float, float] = (380.0, 300.0),
    object_at: Tuple[float, float] = (560.0, 320.0),
    seed: int = 7,
    with_object: bool = True,
    with_grip: bool = True,
) -> SyntheticDemo:
    """Render a phase plan into pose frames plus an object track."""
    rng = np.random.default_rng(seed)
    plan = list(plan if plan is not None else pick_and_place_plan())

    # ---- sample the wrist path and the grip signal -------------------- #
    times: List[float] = []
    wrist: List[Tuple[float, float]] = []
    grip: List[float] = []
    carried: List[bool] = []
    gt: List[Dict] = []

    cursor = home
    grip_state = CLOSED_APERTURE if plan[0].grip == "open" else OPEN_APERTURE
    # Starting from the opposite state means the first phase's aperture ramp is
    # itself observable, as it is in real footage where a hand opens to reach.
    t = 0.0
    for ph in plan:
        n = max(2, int(round(ph.duration * fps)))
        start_t = t
        target_grip = CLOSED_APERTURE if ph.grip == "closed" else OPEN_APERTURE
        p0 = cursor
        for k in range(n):
            s = (k + 1) / n
            m = min_jerk(s)
            x = p0[0] + (ph.target[0] - p0[0]) * m
            y = p0[1] + (ph.target[1] - p0[1]) * m
            times.append(t)
            wrist.append((x, y))
            # The aperture ramps across the phase that changes it.
            grip.append(grip_state + (target_grip - grip_state) * m)
            carried.append(ph.carry)
            t += 1.0 / fps
        cursor = ph.target
        grip_state = target_grip
        gt.append({
            "label": ph.label,
            "start_time": round(start_t, 4),
            "end_time": round(t, 4),
        })

    n_frames = len(times)
    wrist_arr = np.array(wrist, dtype=float)
    wrist_arr += rng.normal(0.0, noise_px, wrist_arr.shape)

    # ---- object: static, then rigidly carried, then static ------------ #
    obj = np.zeros((n_frames, 2), dtype=float)
    obj[:] = object_at
    for i in range(1, n_frames):
        if carried[i]:
            obj[i] = obj[i - 1] + (wrist_arr[i] - wrist_arr[i - 1])
        else:
            obj[i] = obj[i - 1]
    obj += rng.normal(0.0, noise_px * 0.7, obj.shape)

    # ---- assemble pose frames ----------------------------------------- #
    sh_r = shoulder
    sh_l = (shoulder[0] - shoulder_width, shoulder[1])
    frames: List[PoseFrame] = []
    for i in range(n_frames):
        pf = PoseFrame(frame_index=i, timestamp=times[i], pose_detected=True,
                       pose_confidence=0.95)
        wx, wy = wrist_arr[i]
        # Elbow: bent between shoulder and wrist, offset outward a little.
        ex = 0.5 * (sh_r[0] + wx) + 8.0
        ey = 0.5 * (sh_r[1] + wy) + 18.0
        kps = {
            "right_shoulder": (sh_r[0], sh_r[1]),
            "left_shoulder": (sh_l[0], sh_l[1]),
            "right_elbow": (ex, ey),
            "left_elbow": (sh_l[0] - 10, sh_l[1] + 70),
            "right_wrist": (wx, wy),
            "left_wrist": (sh_l[0] - 14, sh_l[1] + 140),
            "right_hip": (sh_r[0] - 8, sh_r[1] + 150),
            "left_hip": (sh_l[0] + 8, sh_l[1] + 150),
            "right_index": (wx + 12, wy + 4),
            "right_thumb": (wx + 8, wy - 8),
            "right_pinky": (wx + 6, wy + 14),
        }
        for name, (x, y) in kps.items():
            pf.keypoints[name] = Keypoint(x=float(x), y=float(y), visibility=0.95)
        if with_grip:
            pf.hand_aperture["right"] = float(grip[i])
            pf.hand_confidence["right"] = 0.9
            pf.aperture_source["right"] = "hands"
        pf.hand_center["right"] = (float(wx), float(wy))
        frames.append(pf)

    seq = PoseSequence(frames=frames, active_arm="right", backend="synthetic")

    half = object_size / 2.0
    track = Track(track_id=1, class_name="box")
    if with_object:
        for i in range(n_frames):
            ox, oy = obj[i]
            track.observations.append(
                TrackObservation(
                    frame_index=i,
                    timestamp=times[i],
                    bbox=(ox - half, oy - half, ox + half, oy + half),
                    confidence=0.82,
                    is_detection=True,
                )
            )

    return SyntheticDemo(sequence=seq, track=track if with_object else Track(1, "box"),
                         ground_truth=gt, fps=fps)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def boundary_error(
    predicted: Sequence[float], truth: Sequence[float]
) -> Tuple[float, List[float]]:
    """For each true boundary, the distance to the nearest predicted one."""
    errs = []
    for tb in truth:
        if not predicted:
            errs.append(float("inf"))
            continue
        errs.append(min(abs(float(p) - float(tb)) for p in predicted))
    finite = [e for e in errs if math.isfinite(e)]
    return (float(np.mean(finite)) if finite else float("inf")), errs


def frame_accuracy(segments, truth: Sequence[Dict], fps: float) -> float:
    """Fraction of time whose predicted label matches the ground-truth label."""
    total = truth[-1]["end_time"]
    n = max(1, int(round(total * fps)))
    correct = 0
    for k in range(n):
        t = (k + 0.5) / fps
        true_lab = next(
            (g["label"] for g in truth if g["start_time"] <= t < g["end_time"]),
            truth[-1]["label"],
        )
        pred = next(
            (s.label for s in segments if s.start_time <= t < s.end_time), None
        )
        if pred == true_lab:
            correct += 1
    return correct / n


def label_order_preserved(segments, expected: Sequence[str]) -> bool:
    """Is ``expected`` an ordered subsequence of the predicted labels?"""
    preds = [s.label for s in segments]
    pos = 0
    for want in expected:
        found = False
        while pos < len(preds):
            if preds[pos] == want:
                found = True
                pos += 1
                break
            pos += 1
        if not found:
            return False
    return True
