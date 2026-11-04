"""Phase labelling (Step 5a of the pipeline).

Each segment is described by aggregate, scale-invariant motion statistics, and
each candidate phase label scores those statistics through soft membership
functions. Scores are then decoded with Viterbi over an allowed-transition
matrix, so the emitted sequence obeys a manipulation grammar (you cannot
RELEASE before you GRASP).

Two deliberate properties:

* **Graceful degradation.** Every evidence term may report "unavailable"; a
  label's score is the weighted mean of whatever *is* available, renormalised.
  So when no object was tracked, labels that depend only on kinematics still
  score normally, and labels that depend on contact lose their support instead
  of silently scoring zero.
* **No assertion without evidence.** If neither a grip signal nor an object
  track exists, contact cannot be established, and the contact-dependent labels
  (GRASP / LIFT / TRANSPORT / PLACE / RELEASE) are removed from the candidate
  set entirely rather than guessed.

The Viterbi step is the natural drop-in point for a learned model: an HMM only
changes the transition matrix, a TCN or Transformer only changes the emission
scores.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from segmentation.base import BaseLabeler, Segment, register_labeler
from utils.config import SkillConfig

EPS = 1e-9

#: Labels that cannot be claimed without contact evidence.
CONTACT_DEPENDENT = ("GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "PUSH", "PULL")

#: Labels available when contact cannot be established at all.
KINEMATIC_ONLY = ("IDLE", "REACH", "RETRACT", "UNKNOWN")

#: Allowed successors. Anything absent gets a large log-penalty rather than
#: being forbidden outright, so an unusual demonstration is still decodable.
TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "IDLE":      ("IDLE", "REACH", "APPROACH", "PUSH", "PULL", "UNKNOWN"),
    "REACH":     ("REACH", "APPROACH", "GRASP", "PUSH", "PULL", "IDLE", "UNKNOWN"),
    "APPROACH":  ("APPROACH", "GRASP", "PUSH", "PULL", "IDLE", "UNKNOWN"),
    "GRASP":     ("GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "UNKNOWN"),
    "LIFT":      ("LIFT", "TRANSPORT", "PLACE", "RELEASE", "UNKNOWN"),
    "TRANSPORT": ("TRANSPORT", "LIFT", "PLACE", "RELEASE", "UNKNOWN"),
    "PLACE":     ("PLACE", "RELEASE", "RETRACT", "TRANSPORT", "UNKNOWN"),
    "RELEASE":   ("RELEASE", "RETRACT", "IDLE", "REACH", "UNKNOWN"),
    "RETRACT":   ("RETRACT", "IDLE", "REACH", "UNKNOWN"),
    "PUSH":      ("PUSH", "IDLE", "RETRACT", "REACH", "PULL", "UNKNOWN"),
    "PULL":      ("PULL", "IDLE", "RETRACT", "REACH", "PUSH", "UNKNOWN"),
    "UNKNOWN":   tuple(),      # filled in below with everything
}
TRANSITIONS["UNKNOWN"] = tuple(TRANSITIONS.keys())

#: log-probability applied to a transition that is not in the allowed set.
OFF_GRAMMAR_LOGP = -3.5
ON_GRAMMAR_LOGP = 0.0
#: Extra cost for entering UNKNOWN. Without it, UNKNOWN -- which may precede any
#: label -- becomes a cheap bridge that lets the decoder hop between phases that
#: the grammar forbids adjoining directly.
ENTER_UNKNOWN_LOGP = -1.6


# --------------------------------------------------------------------------- #
# Soft membership helpers
# --------------------------------------------------------------------------- #
def ramp_up(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """0 below ``lo``, 1 above ``hi``, linear in between."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def ramp_down(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """1 below ``lo``, 0 above ``hi``."""
    r = ramp_up(x, lo, hi)
    return None if r is None else 1.0 - r


def band(x: Optional[float], lo: float, hi: float, soft: float) -> Optional[float]:
    """1 inside ``[lo, hi]``, decaying over ``soft`` outside it."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(np.clip(1.0 - d / max(soft, EPS), 0.0, 1.0))


def _nan_to_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def combine(terms: Dict[str, Tuple[float, Optional[float]]]) -> Tuple[float, Dict]:
    """Weighted mean over available terms, renormalising for missing ones.

    Returns ``(score, detail)`` where ``detail`` records which terms were used
    and which were unavailable -- this is what makes a label auditable.
    """
    num = 0.0
    den = 0.0
    detail: Dict[str, Any] = {}
    for name, (weight, value) in terms.items():
        if value is None:
            detail[name] = None
            continue
        detail[name] = round(float(value), 3)
        num += weight * float(value)
        den += weight
    if den <= 0:
        return 0.0, detail
    return float(np.clip(num / den, 0.0, 1.0)), detail


# --------------------------------------------------------------------------- #
# Segment-level feature aggregation
# --------------------------------------------------------------------------- #
def segment_features(
    df: pd.DataFrame, seg: Segment, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate per-frame features into segment-level descriptors.

    All spatial quantities are in body-widths so the same thresholds apply
    across videos.
    """
    a, b = seg.start_index, seg.end_index
    sl = df.iloc[a:b]
    scale = max(float(context.get("body_scale_px", 1.0)), EPS)

    def m(col: str) -> Optional[float]:
        if col not in sl.columns:
            return None
        v = sl[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        return float(np.mean(v)) if v.size else None

    def mx(col: str) -> Optional[float]:
        if col not in sl.columns:
            return None
        v = sl[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        return float(np.max(v)) if v.size else None

    def frac(col: str) -> Optional[float]:
        if col not in sl.columns:
            return None
        v = sl[col].to_numpy()
        if v.size == 0:
            return None
        return float(np.mean(v.astype(bool)))

    def first_last(col: str) -> Tuple[Optional[float], Optional[float]]:
        if col not in sl.columns:
            return None, None
        v = sl[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return None, None
        return float(v[0]), float(v[-1])

    wx = sl["wrist_x_s"].to_numpy(dtype=float)
    wy = sl["wrist_y_s"].to_numpy(dtype=float)
    ok = ~(np.isnan(wx) | np.isnan(wy))
    pts = np.stack([wx[ok], wy[ok]], axis=1) / scale
    if len(pts) >= 2:
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path_length = float(np.sum(steps))
        net_vec = pts[-1] - pts[0]
        net_disp = float(np.linalg.norm(net_vec))
        # Image y grows downward, so a positive net_dy means upward motion.
        net_dx = float(net_vec[0])
        net_dy = float(-net_vec[1])
        straightness = net_disp / path_length if path_length > EPS else None
    else:
        path_length = net_disp = 0.0
        net_dx = net_dy = 0.0
        straightness = None

    d0, d1 = first_last("hand_obj_dist")
    g0, g1 = first_last("grip_ratio")

    obj_path = None
    obj_net_dx = None
    if context.get("object_available"):
        ox = sl["object_x"].to_numpy(dtype=float)
        oy = sl["object_y"].to_numpy(dtype=float)
        ook = ~(np.isnan(ox) | np.isnan(oy))
        opts = np.stack([ox[ook], oy[ook]], axis=1) / scale
        if len(opts) >= 2:
            obj_path = float(np.sum(np.linalg.norm(np.diff(opts, axis=0), axis=1)))
            obj_net_dx = float(opts[-1][0] - opts[0][0])

    # Whether the hand moves away from the shoulder (extension) or toward it.
    sh_x = sl["shoulder_x_s"].to_numpy(dtype=float)
    sh_y = sl["shoulder_y_s"].to_numpy(dtype=float)
    reach_extent_delta = None
    ok2 = ~(np.isnan(sh_x) | np.isnan(sh_y) | np.isnan(wx) | np.isnan(wy))
    if ok2.sum() >= 2:
        ext = np.hypot(wx[ok2] - sh_x[ok2], wy[ok2] - sh_y[ok2]) / scale
        reach_extent_delta = float(ext[-1] - ext[0])

    speeds = sl["wrist_speed"].to_numpy(dtype=float)
    speeds = speeds[~np.isnan(speeds)]
    if speeds.size >= 4:
        half = speeds.size // 2
        decel = float(np.mean(speeds[:half]) - np.mean(speeds[half:]))
    else:
        decel = None

    # Terminal speed distinguishes a phase that *ends* the motion (PLACE, which
    # comes to rest on the surface) from one that merely passes through
    # (TRANSPORT). Segment-mean speed cannot tell them apart.
    def tail_mean(col: str, frac: float = 0.3) -> Optional[float]:
        if col not in sl.columns:
            return None
        v = sl[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return None
        k = max(2, int(round(v.size * frac)))
        return float(np.mean(v[-k:]))

    def head_mean(col: str, frac: float = 0.3) -> Optional[float]:
        if col not in sl.columns:
            return None
        v = sl[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return None
        k = max(2, int(round(v.size * frac)))
        return float(np.mean(v[:k]))

    return {
        "duration": seg.duration,
        "n_frames": seg.n_frames,
        "mean_velocity": m("wrist_speed"),
        "max_velocity": mx("wrist_speed"),
        "mean_acceleration": m("wrist_accel"),
        "path_length": path_length,
        "net_displacement": net_disp,
        "net_dx": net_dx,
        "net_dy": net_dy,
        "straightness": straightness,
        "mean_heading_change": m("wrist_heading_change"),
        "mean_vy": m("wrist_vy"),
        "deceleration": decel,
        "initial_velocity": head_mean("wrist_speed"),
        "terminal_velocity": tail_mean("wrist_speed"),
        "terminal_object_speed": tail_mean("obj_speed"),
        # Contact at each end of the segment. GRASP is the transition into
        # contact and RELEASE the transition out of it, so the *direction* of the
        # change matters -- a segment-mean contact fraction cannot express it,
        # and a low mean would otherwise let RELEASE be claimed for a segment
        # where contact never happened at all.
        "initial_contact_fraction": head_mean("contact"),
        "terminal_contact_fraction": tail_mean("contact"),
        "reach_extent_delta": reach_extent_delta,
        "contact_fraction": frac("contact"),
        "grip_closed_fraction": frac("grip_closed") if context.get("grip_available") else None,
        "grip_start": g0,
        "grip_end": g1,
        "grip_delta": (None if (g0 is None or g1 is None) else g1 - g0),
        "hand_obj_dist_start": d0,
        "hand_obj_dist_end": d1,
        "hand_obj_dist_delta": (None if (d0 is None or d1 is None) else d1 - d0),
        "mean_hand_obj_dist": m("hand_obj_dist"),
        "object_moving_fraction": frac("object_moving") if context.get("object_available") else None,
        "mean_object_speed": m("obj_speed"),
        "object_path_length": obj_path,
        "object_net_dx": obj_net_dx,
        "mean_pose_confidence": m("pose_confidence"),
        "mean_elbow_angle_deg": m("elbow_angle_deg"),
    }


# --------------------------------------------------------------------------- #
# Label scoring
# --------------------------------------------------------------------------- #
class PhaseScorer:
    """Soft rules mapping segment features to a score per phase label.

    Thresholds are expressed in body-widths and body-widths/second, the units
    produced by :mod:`features.motion_features`.
    """

    def __init__(self, ctx: Dict[str, Any]) -> None:
        self.ctx = ctx
        self.has_object = bool(ctx.get("object_available"))
        self.has_grip = bool(ctx.get("grip_available"))
        self.has_contact = ctx.get("contact_basis") != "unavailable"

    # Each scorer returns (score, evidence_detail).
    def score(self, label: str, f: Dict[str, Any]) -> Tuple[float, Dict]:
        fn: Callable[[Dict[str, Any]], Tuple[float, Dict]] = getattr(
            self, f"_{label.lower()}"
        )
        return fn(f)

    # -- no-contact phases ------------------------------------------------ #
    def _idle(self, f):
        return combine({
            "low_speed":       (3.0, ramp_down(f["mean_velocity"], 0.10, 0.30)),
            "short_path":      (2.0, ramp_down(f["path_length"], 0.10, 0.45)),
            "object_static":   (1.5, ramp_down(f["mean_object_speed"], 0.05, 0.20)),
            "no_grip_change":  (1.0, ramp_down(_abs(f["grip_delta"]), 0.08, 0.30)),
        })

    def _reach(self, f):
        return combine({
            "not_in_contact":  (3.0, ramp_down(f["contact_fraction"], 0.15, 0.55)),
            "moving":          (2.5, ramp_up(f["mean_velocity"], 0.15, 0.55)),
            "long_path":       (2.0, ramp_up(f["path_length"], 0.25, 1.10)),
            "closing_on_obj":  (2.5, ramp_down(f["hand_obj_dist_delta"], -0.10, 0.10)),
            "arm_extending":   (1.5, ramp_up(f["reach_extent_delta"], 0.02, 0.25)),
            "fairly_straight": (1.0, ramp_up(f["straightness"], 0.35, 0.80)),
            "hand_open":       (1.0, ramp_up(f["grip_start"], 0.45, 0.80)),
            "object_static":   (1.0, ramp_down(f["object_moving_fraction"], 0.20, 0.60)),
        })

    def _approach(self, f):
        return combine({
            "not_in_contact":  (2.5, ramp_down(f["contact_fraction"], 0.25, 0.70)),
            "close_to_object": (3.0, ramp_down(f["mean_hand_obj_dist"], 0.25, 0.85)),
            "still_closing":   (2.0, ramp_down(f["hand_obj_dist_delta"], -0.05, 0.08)),
            "decelerating":    (2.0, ramp_up(f["deceleration"], 0.0, 0.25)),
            "slow":            (1.5, band(f["mean_velocity"], 0.05, 0.40, 0.30)),
            "ends_slow":       (1.5, ramp_down(f["terminal_velocity"], 0.08, 0.45)),
            "short_duration":  (1.0, ramp_down(f["duration"], 0.5, 1.6)),
            "object_static":   (1.0, ramp_down(f["object_moving_fraction"], 0.20, 0.60)),
        })

    def _retract(self, f):
        return combine({
            "not_in_contact":  (3.0, ramp_down(f["contact_fraction"], 0.15, 0.55)),
            "moving":          (1.5, ramp_up(f["mean_velocity"], 0.12, 0.50)),
            "opening_gap":     (3.0, ramp_up(f["hand_obj_dist_delta"], 0.05, 0.35)),
            "arm_retracting":  (1.5, ramp_down(f["reach_extent_delta"], -0.02, -0.25)),
            "object_static":   (1.5, ramp_down(f["object_moving_fraction"], 0.20, 0.60)),
            "hand_open":       (1.0, ramp_up(f["grip_end"], 0.45, 0.80)),
        })

    # -- contact phases --------------------------------------------------- #
    def _grasp(self, f):
        return combine({
            "grip_closing":    (3.5, ramp_down(f["grip_delta"], -0.06, 0.06)),
            "ends_closed":     (2.0, ramp_down(f["grip_end"], 0.35, 0.70)),
            "at_object":       (3.0, ramp_down(f["mean_hand_obj_dist"], 0.20, 0.60)),
            "ends_in_contact": (2.5, ramp_up(f["terminal_contact_fraction"], 0.25, 0.65)),
            "little_motion":   (2.5, ramp_down(f["path_length"], 0.08, 0.40)),
            "short":           (1.5, ramp_down(f["duration"], 0.25, 1.10)),
            "object_not_yet_moving": (1.0, ramp_down(f["object_moving_fraction"], 0.30, 0.80)),
            "slow":            (1.5, ramp_down(f["mean_velocity"], 0.10, 0.40)),
        })

    def _release(self, f):
        return combine({
            "grip_opening":    (3.5, ramp_up(f["grip_delta"], -0.06, 0.06)),
            "ends_open":       (2.0, ramp_up(f["grip_end"], 0.35, 0.70)),
            # Releasing presupposes having held something: contact must be
            # present at the start of the segment and gone by its end.
            "started_in_contact": (3.5, ramp_up(f["initial_contact_fraction"], 0.20, 0.60)),
            "contact_ending":  (2.0, ramp_down(f["terminal_contact_fraction"], 0.25, 0.70)),
            "little_motion":   (2.0, ramp_down(f["path_length"], 0.10, 0.45)),
            "short":           (1.5, ramp_down(f["duration"], 0.25, 1.10)),
            "object_stopped":  (1.5, ramp_down(f["object_moving_fraction"], 0.25, 0.70)),
        })

    def _lift(self, f):
        # The vertical term must be *signed*: an unsigned dominance ratio scores
        # a descent just as highly as an ascent, which makes PLACE look like LIFT.
        return combine({
            "in_contact":      (3.0, ramp_up(f["contact_fraction"], 0.35, 0.80)),
            "grip_held":       (2.0, ramp_down(f["grip_end"], 0.35, 0.75)),
            "net_upward":      (3.5, ramp_up(f["net_dy"], 0.04, 0.35)),
            "upward_dominant": (2.5, ramp_up(
                _signed_ratio(f["net_dy"], f["net_dx"]), 0.3, 1.5)),
            "moving":          (1.5, ramp_up(f["mean_velocity"], 0.10, 0.45)),
            "object_follows":  (2.0, ramp_up(f["object_moving_fraction"], 0.20, 0.65)),
        })

    def _transport(self, f):
        return combine({
            "in_contact":      (3.0, ramp_up(f["contact_fraction"], 0.35, 0.80)),
            "grip_held":       (2.0, ramp_down(f["grip_end"], 0.35, 0.75)),
            "long_path":       (3.0, ramp_up(f["path_length"], 0.25, 1.20)),
            "sustained_speed": (2.5, ramp_up(f["mean_velocity"], 0.15, 0.55)),
            "object_moves":    (3.0, ramp_up(f["object_moving_fraction"], 0.25, 0.70)),
            "object_travels":  (2.0, ramp_up(f["object_path_length"], 0.20, 1.00)),
            # A purely vertical excursion is a lift or a place, not a transport,
            # so this counter-evidence needs real weight.
            "not_mostly_vertical": (2.5, ramp_down(_ratio(f["net_dy"], f["net_dx"]), 1.2, 2.5)),
            "grip_stable":     (1.0, ramp_down(_abs(f["grip_delta"]), 0.10, 0.35)),
            # Transport is a pass-through: it does not come to rest.
            "still_moving_at_end": (2.5, ramp_up(f["terminal_velocity"], 0.12, 0.50)),
        })

    def _place(self, f):
        return combine({
            "in_contact":      (2.5, ramp_up(f["contact_fraction"], 0.30, 0.75)),
            "grip_still_held": (1.5, ramp_down(f["grip_start"], 0.40, 0.75)),
            "net_downward":    (3.5, ramp_up(_neg(f["net_dy"]), 0.03, 0.30)),
            "downward_dominant": (2.5, ramp_up(
                _signed_ratio(_neg(f["net_dy"]), f["net_dx"]), 0.3, 1.5)),
            "decelerating":    (2.0, ramp_up(f["deceleration"], 0.0, 0.25)),
            # Terminal, not mean: a place ends at rest on the surface. The
            # thresholds are generous because a place descent can cover real
            # distance, and smoothing carries some speed past the boundary.
            "comes_to_rest":   (3.0, ramp_down(f["terminal_velocity"], 0.15, 0.70)),
            "object_settling": (2.0, ramp_down(f["terminal_object_speed"], 0.15, 0.70)),
        })

    # -- non-prehensile phases -------------------------------------------- #
    # PUSH/PULL are *non-prehensile*: the defining negative evidence is that the
    # hand is not gripping. Without a heavy weight on that, a carried object
    # looks exactly like a pushed one (hand near, object moving, gap stable), and
    # PUSH would outscore TRANSPORT on a genuine pick-and-place.
    def _push(self, f):
        return combine({
            "not_gripping":    (6.0, ramp_down(f["grip_closed_fraction"], 0.10, 0.45)),
            "hand_open":       (4.0, ramp_up(f["grip_start"], 0.45, 0.80)),
            "object_moves":    (3.5, ramp_up(f["object_moving_fraction"], 0.30, 0.75)),
            # A push must actually displace the object, not merely coincide with
            # a frame or two of detector jitter.
            "object_displaced": (3.0, ramp_up(f["object_path_length"], 0.15, 0.80)),
            "hand_near":       (3.0, ramp_down(f["mean_hand_obj_dist"], 0.25, 0.70)),
            "gap_stable":      (2.0, ramp_down(_abs(f["hand_obj_dist_delta"]), 0.08, 0.35)),
            "arm_extending":   (2.5, ramp_up(f["reach_extent_delta"], 0.02, 0.25)),
            "moving":          (1.5, ramp_up(f["mean_velocity"], 0.12, 0.50)),
        })

    def _pull(self, f):
        return combine({
            "not_gripping":    (6.0, ramp_down(f["grip_closed_fraction"], 0.10, 0.45)),
            "hand_open":       (3.0, ramp_up(f["grip_start"], 0.45, 0.80)),
            "object_moves":    (3.5, ramp_up(f["object_moving_fraction"], 0.30, 0.75)),
            "object_displaced": (3.0, ramp_up(f["object_path_length"], 0.15, 0.80)),
            "hand_near":       (3.0, ramp_down(f["mean_hand_obj_dist"], 0.25, 0.70)),
            "gap_stable":      (2.0, ramp_down(_abs(f["hand_obj_dist_delta"]), 0.08, 0.35)),
            "arm_retracting":  (2.5, ramp_down(f["reach_extent_delta"], -0.02, -0.25)),
            "moving":          (1.5, ramp_up(f["mean_velocity"], 0.12, 0.50)),
        })

    def _unknown(self, f):
        # UNKNOWN is a constant floor: it wins only when nothing else clears it.
        return 0.22, {"floor": 0.22}


def _abs(v) -> Optional[float]:
    v = _nan_to_none(v)
    return None if v is None else abs(v)


def _neg(v) -> Optional[float]:
    v = _nan_to_none(v)
    return None if v is None else -v


def _ratio(num, den) -> Optional[float]:
    """|num| / |den| with a floor on the denominator; None if unavailable."""
    n, d = _nan_to_none(num), _nan_to_none(den)
    if n is None or d is None:
        return None
    return abs(n) / max(abs(d), 0.02)


def _signed_ratio(num, den) -> Optional[float]:
    """num / |den|, keeping the sign of ``num`` so direction is preserved."""
    n, d = _nan_to_none(num), _nan_to_none(den)
    if n is None or d is None:
        return None
    return n / max(abs(d), 0.02)


# --------------------------------------------------------------------------- #
# Labeler
# --------------------------------------------------------------------------- #
@register_labeler("rule_viterbi")
class RuleViterbiLabeler(BaseLabeler):
    """Soft-rule emissions decoded with Viterbi over a manipulation grammar."""

    name = "rule_viterbi"

    def __init__(self, cfg: SkillConfig, temperature: float = 0.14) -> None:
        self.cfg = cfg
        self.temperature = temperature
        self.last_diagnostics: Dict[str, Any] = {}

    def label(
        self, segments: List[Segment], df: pd.DataFrame, context: Dict[str, Any]
    ) -> List[Segment]:
        """Label, then merge adjacent segments that received the same label.

        The change-point stage deliberately over-proposes boundaries so short
        phases are not missed. Two neighbouring intervals that decode to the same
        phase are one phase, so they are merged and the whole span is re-scored;
        this is what makes the reported granularity match the demonstration
        instead of the detector's sensitivity.
        """
        if not segments:
            return segments

        out = self._label_once(segments, df, context)
        merged = self._merge_same_label(out)
        if len(merged) != len(out):
            # Re-aggregate the merged spans' features and confidences, but keep
            # the labels. The decode already ran at the finest granularity, where
            # the evidence for each phase is sharpest; re-deciding on the merged
            # span systematically favours the coarser label and loses real
            # sub-phases (an APPROACH absorbed back into its REACH, for example).
            self._rescore_keeping_labels(merged, df, context)
        return merged

    @staticmethod
    def _merge_same_label(segments: List[Segment]) -> List[Segment]:
        """Collapse runs of neighbouring segments that decoded to one label.

        The change-point stage deliberately over-proposes boundaries so that
        short phases are not missed, which leaves a long phase split across
        several intervals. Consolidating them makes the reported granularity
        reflect the demonstration rather than the detector's sensitivity.
        """
        if len(segments) < 2:
            return segments
        merged: List[Segment] = []
        for seg in segments:
            if merged and merged[-1].label == seg.label:
                prev = merged[-1]
                prev.end_index = seg.end_index
                prev.end_time = seg.end_time
                # Record the strongest internal transition that was dissolved,
                # so sub-phase structure is still visible to the LfD stage.
                if seg.boundary_salience > prev.evidence.get(
                    "max_internal_boundary_salience", 0.0
                ):
                    prev.evidence["max_internal_boundary_salience"] = round(
                        seg.boundary_salience, 4
                    )
                    prev.evidence["internal_boundary_time"] = round(seg.start_time, 3)
                    prev.evidence["internal_boundary_sources"] = seg.boundary_sources
                continue
            merged.append(seg)
        for i, s in enumerate(merged, start=1):
            s.id = i
        return merged

    def _rescore_keeping_labels(
        self, segments: List[Segment], df: pd.DataFrame, context: Dict[str, Any]
    ) -> None:
        scorer = PhaseScorer(context)
        candidates = self._candidate_labels(scorer)
        trans = self._transition_matrix(candidates)
        quality = self._global_quality(context)
        path = [s.label for s in segments]

        for i, seg in enumerate(segments):
            feats = segment_features(df, seg, context)
            seg.motion_features = {
                k: (None if v is None else (round(v, 5) if isinstance(v, float) else v))
                for k, v in feats.items()
            }
            row: Dict[str, float] = {}
            for lab in candidates:
                row[lab], _ = scorer.score(lab, feats)
            if seg.label in candidates:
                _, det = scorer.score(seg.label, feats)
            else:
                det = {}

            probs = self._softmax(self._contextual_scores(
                row, trans, candidates,
                prev_label=path[i - 1] if i > 0 else None,
                next_label=path[i + 1] if i + 1 < len(path) else None,
            ))
            conf = float(probs.get(seg.label, 0.0)) * quality
            ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
            seg.alternatives = [
                (k, round(v, 4)) for k, v in ranked[:4] if k != seg.label
            ]
            seg.confidence = round(min(conf, 0.99), 4)
            seg.evidence.update({
                "decoded_label": seg.label,
                "raw_score": round(row.get(seg.label, 0.0), 4),
                "terms": det,
                "quality_factor": round(quality, 3),
                "candidate_scores": {k: round(v, 4) for k, v in ranked},
                "merged_from_multiple_intervals": True,
            })

    def _label_once(
        self, segments: List[Segment], df: pd.DataFrame, context: Dict[str, Any]
    ) -> List[Segment]:
        scorer = PhaseScorer(context)
        candidates = self._candidate_labels(scorer)

        emissions: List[Dict[str, float]] = []
        details: List[Dict[str, Dict]] = []
        for seg in segments:
            feats = segment_features(df, seg, context)
            seg.motion_features = {
                k: (None if v is None else (round(v, 5) if isinstance(v, float) else v))
                for k, v in feats.items()
            }
            row: Dict[str, float] = {}
            det: Dict[str, Dict] = {}
            for lab in candidates:
                s, d = scorer.score(lab, feats)
                row[lab] = s
                det[lab] = d
            emissions.append(row)
            details.append(det)

        path = self._viterbi(emissions, candidates)

        quality = self._global_quality(context)
        trans = self._transition_matrix(candidates)
        for i, (seg, lab, row, det) in enumerate(
            zip(segments, path, emissions, details)
        ):
            # Confidence is the local posterior *given the decoded neighbours*,
            # not the bare emission softmax. A label that is the only
            # grammatically coherent option at this point in the sequence should
            # not be penalised because some other phase has a similar raw score.
            probs = self._softmax(self._contextual_scores(
                row, trans, candidates,
                prev_label=path[i - 1] if i > 0 else None,
                next_label=path[i + 1] if i + 1 < len(path) else None,
            ))
            conf = float(probs.get(lab, 0.0)) * quality
            ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
            seg.alternatives = [(k, round(v, 4)) for k, v in ranked[:4] if k != lab]
            if conf < self.cfg.min_phase_confidence or lab == "UNKNOWN":
                seg.label = "UNKNOWN" if conf < self.cfg.min_phase_confidence else lab
            else:
                seg.label = lab
            seg.confidence = round(min(conf, 0.99), 4)
            seg.evidence = {
                "decoded_label": lab,
                "raw_score": round(row.get(lab, 0.0), 4),
                "terms": det.get(lab, {}),
                "quality_factor": round(quality, 3),
                "candidate_scores": {k: round(v, 4) for k, v in ranked},
            }

        self.last_diagnostics = {
            "labeler": self.name,
            "candidate_labels": list(candidates),
            "contact_basis": context.get("contact_basis"),
            "quality_factor": round(quality, 3),
        }
        return segments

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _candidate_labels(scorer: PhaseScorer) -> Tuple[str, ...]:
        """Only admit labels whose defining evidence is actually measurable."""
        if not scorer.has_contact:
            return KINEMATIC_ONLY

        labs = ["IDLE", "REACH", "APPROACH", "LIFT", "TRANSPORT", "PLACE",
                "RETRACT", "UNKNOWN"]
        # GRASP and RELEASE are claims about the *hand aperture*. Object motion
        # alone shows that an interaction happened, not that fingers closed, so
        # without a grip signal these must not be asserted. PUSH and PULL are
        # claims that the grasp is non-prehensile, which needs the same evidence.
        if scorer.has_grip:
            labs.extend(["GRASP", "RELEASE"])
            if scorer.has_object:
                labs.extend(["PUSH", "PULL"])
        return tuple(labs)

    @staticmethod
    def _transition_matrix(labels: Sequence[str]) -> np.ndarray:
        k = len(labels)
        idx = {l: i for i, l in enumerate(labels)}
        trans = np.full((k, k), OFF_GRAMMAR_LOGP, dtype=float)
        for a in labels:
            for b in TRANSITIONS.get(a, ()):
                if b in idx:
                    trans[idx[a], idx[b]] = ON_GRAMMAR_LOGP
        if "UNKNOWN" in idx:
            trans[:, idx["UNKNOWN"]] += ENTER_UNKNOWN_LOGP
        return trans

    @staticmethod
    def _contextual_scores(
        row: Dict[str, float],
        trans: np.ndarray,
        labels: Sequence[str],
        prev_label: Optional[str],
        next_label: Optional[str],
    ) -> Dict[str, float]:
        """Re-score each label with the grammar prior from its decoded neighbours.

        The transition terms are log-probabilities while emissions are scores in
        [0, 1], so the prior is converted to a multiplicative factor before being
        applied -- this keeps the result on the same scale the softmax expects.
        """
        idx = {l: i for i, l in enumerate(labels)}
        out: Dict[str, float] = {}
        for lab, score in row.items():
            logp = 0.0
            if prev_label in idx and lab in idx:
                logp += trans[idx[prev_label], idx[lab]]
            if next_label in idx and lab in idx:
                logp += trans[idx[lab], idx[next_label]]
            out[lab] = float(score * math.exp(logp))
        return out

    def _softmax(self, row: Dict[str, float]) -> Dict[str, float]:
        keys = list(row)
        v = np.array([row[k] for k in keys], dtype=float) / max(self.temperature, EPS)
        v -= v.max()
        e = np.exp(v)
        p = e / max(e.sum(), EPS)
        return dict(zip(keys, p.tolist()))

    @staticmethod
    def _global_quality(context: Dict[str, Any]) -> float:
        """Scale every confidence by how trustworthy the observations were."""
        pose_rate = float(context.get("pose_detection_rate", 1.0) or 0.0)
        q = 0.55 + 0.45 * float(np.clip(pose_rate, 0.0, 1.0))
        if not context.get("object_available"):
            q *= 0.82     # contact reasoning rests on the grip signal alone
        if not context.get("grip_available"):
            q *= 0.82     # contact reasoning rests on object motion alone
        return float(np.clip(q, 0.15, 1.0))

    @staticmethod
    def _viterbi(
        emissions: List[Dict[str, float]], labels: Sequence[str]
    ) -> List[str]:
        """Most likely label sequence under the transition grammar."""
        n = len(emissions)
        k = len(labels)
        idx = {l: i for i, l in enumerate(labels)}

        trans = np.full((k, k), OFF_GRAMMAR_LOGP, dtype=float)
        for a in labels:
            for b in TRANSITIONS.get(a, ()):
                if b in idx:
                    trans[idx[a], idx[b]] = ON_GRAMMAR_LOGP
        if "UNKNOWN" in idx:
            trans[:, idx["UNKNOWN"]] += ENTER_UNKNOWN_LOGP

        def logem(step: int, lab: str) -> float:
            return math.log(max(emissions[step].get(lab, 0.0), 1e-6))

        dp = np.full((n, k), -np.inf)
        back = np.zeros((n, k), dtype=int)
        for j, lab in enumerate(labels):
            dp[0, j] = logem(0, lab)
        for i in range(1, n):
            for j, lab in enumerate(labels):
                e = logem(i, lab)
                cand = dp[i - 1] + trans[:, j]
                best = int(np.argmax(cand))
                dp[i, j] = cand[best] + e
                back[i, j] = best

        path = [0] * n
        path[-1] = int(np.argmax(dp[-1]))
        for i in range(n - 1, 0, -1):
            path[i - 1] = int(back[i, path[i]])
        return [labels[j] for j in path]
