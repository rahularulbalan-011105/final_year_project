"""Temporal segmentation (Step 4 of the pipeline).

The default segmenter combines two independent sources of evidence:

1. **Statistical change points** -- PELT over a standardised multivariate
   feature matrix. This finds where the *regime* of the motion changes without
   knowing anything about manipulation.
2. **Interaction events** -- discrete transitions that are semantically
   meaningful for manipulation: the grip closing or opening, contact starting or
   ending, the object starting or stopping, wrist speed minima (motion pauses),
   hand-object distance minima, and vertical direction reversals.

Change points alone tend to miss short phases such as GRASP and RELEASE, whose
statistical footprint is small but whose semantic importance is large. Events
alone are brittle when the grip signal is weak. Taking the union and then
merging nearby proposals gets both.

No timestamp is ever hard-coded: every boundary traces back to a feature
crossing in this particular video.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from segmentation.base import BaseSegmenter, Segment, register_segmenter
from segmentation.changepoint import build_signal, default_penalty, pelt
from utils import signal_utils as su
from utils.config import SegmentationConfig


@dataclass
class BoundaryProposal:
    index: int
    source: str
    salience: float   # relative strength, used when pruning


@register_segmenter("changepoint_event")
class ChangePointEventSegmenter(BaseSegmenter):
    """PELT change points unioned with manipulation-event boundaries."""

    name = "changepoint_event"

    #: Events outrank statistical change points when two proposals collide,
    #: because an event index is the physically meaningful instant.
    SOURCE_PRIORITY = {
        "grip_transition_start": 7,
        "grip_transition_end": 7,
        "grip_close": 6,
        "grip_open": 6,
        "contact_start": 5,
        "contact_end": 5,
        "object_start": 4,
        "object_stop": 4,
        "distance_min": 3,
        "vertical_reversal": 3,
        "speed_min": 2,
        "changepoint": 1,
    }

    def __init__(self, cfg: SegmentationConfig) -> None:
        self.cfg = cfg
        self.last_diagnostics: Dict[str, Any] = {}

    # -- public ----------------------------------------------------------- #
    def segment(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[Segment]:
        n = len(df)
        t = df["timestamp"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t))) if n > 1 else 1 / 30.0
        if dt <= 0:
            dt = 1 / 30.0

        min_frames = max(2, int(round(self.cfg.min_segment_duration_s / dt)))
        merge_frames = max(1, int(round(self.cfg.boundary_merge_tolerance_s / dt)))

        if n < 2 * min_frames:
            self.last_diagnostics = {
                "reason": "video too short to segment",
                "n_frames": n,
            }
            return self.boundaries_to_segments([], df)

        cp_idx, cp_cols, penalty = self._changepoints(df, min_frames)
        proposals = [
            BoundaryProposal(i, "changepoint", self._salience(df, i, cp_cols))
            for i in cp_idx
        ]

        events: List[BoundaryProposal] = []
        if self.cfg.use_event_boundaries:
            events = self._event_boundaries(df, min_frames, cp_cols)
            proposals.extend(events)

        merged = self._merge(proposals, merge_frames, n)
        enforced = self._enforce_min_length(merged, min_frames, n)
        capped = self._cap_count(enforced, self.cfg.max_segments)

        self.last_diagnostics = {
            "method": self.name,
            "changepoint_features_used": cp_cols,
            "pelt_penalty": round(penalty, 4),
            "n_changepoints": len(cp_idx),
            "n_event_boundaries": len(events),
            "event_sources": sorted({p.source for p in events}),
            "n_boundaries_final": len(capped),
            "min_segment_frames": min_frames,
            "merge_tolerance_frames": merge_frames,
            "effective_sample_rate_hz": round(1.0 / dt, 3),
        }
        return self.boundaries_to_segments(
            [p.index for p in capped],
            df,
            salience={p.index: p.salience for p in capped},
            sources={p.index: p.source for p in capped},
        )

    # -- stage 1: statistical --------------------------------------------- #
    def _changepoints(
        self, df: pd.DataFrame, min_frames: int
    ) -> Tuple[List[int], List[str], float]:
        signal, used = build_signal(df, self.cfg.changepoint_features)
        penalty = default_penalty(signal.shape[0], signal.shape[1],
                                  self.cfg.penalty_scale)
        bkps = pelt(signal, penalty=penalty, min_size=min_frames, jump=1)
        return bkps, used, penalty

    # -- stage 2: interaction events -------------------------------------- #
    def _event_boundaries(
        self, df: pd.DataFrame, min_frames: int, cp_cols: List[str]
    ) -> List[BoundaryProposal]:
        out: List[BoundaryProposal] = []
        n = len(df)

        def add(indices, source: str) -> None:
            for i in np.atleast_1d(indices):
                i = int(i)
                if min_frames <= i <= n - min_frames:
                    out.append(
                        BoundaryProposal(i, source, self._salience(df, i, cp_cols))
                    )

        grip_closed = df["grip_closed"].to_numpy(dtype=bool)
        if grip_closed.any():
            add(su.rising_edges(grip_closed), "grip_close")
            add(su.falling_edges(grip_closed), "grip_open")

        # The aperture-change interval, not just its midpoint. A GRASP is
        # bounded by where the fingers start and stop closing, so both edges of
        # each aperture-transition interval are proposed. Without this, a short
        # GRASP gets absorbed into the neighbouring REACH or LIFT.
        if "grip_rate" in df.columns:
            starts, ends = self._transition_intervals(
                df["grip_rate"].to_numpy(dtype=float), min_frames
            )
            add(starts, "grip_transition_start")
            add(ends, "grip_transition_end")

        contact = df["contact"].to_numpy(dtype=bool)
        if contact.any():
            add(su.rising_edges(contact), "contact_start")
            add(su.falling_edges(contact), "contact_end")

        obj_moving = df["object_moving"].to_numpy(dtype=bool)
        obj_moving = np.where(np.isnan(df["obj_speed"].to_numpy()), False, obj_moving)
        if obj_moving.any():
            add(su.rising_edges(obj_moving), "object_start")
            add(su.falling_edges(obj_moving), "object_stop")

        # Motion pauses: local minima of wrist speed that are genuinely low
        # relative to this demonstration's own speed range.
        speed = df["wrist_speed"].to_numpy(dtype=float)
        if np.isfinite(speed).any():
            hi = float(np.nanpercentile(speed, 90))
            minima = su.local_minima(speed, min_separation=min_frames)
            low = [i for i in minima if np.isfinite(speed[i]) and speed[i] < 0.35 * hi]
            add(low, "speed_min")

        # Closest-approach instants between hand and object. Restricted to
        # genuinely near, out-of-contact minima: while an object is being
        # carried, the hand-object distance is near-constant, and its noise
        # produces a spurious minimum every few frames.
        dist = df["hand_obj_dist"].to_numpy(dtype=float)
        if np.isfinite(dist).any():
            near_thr = float(np.nanpercentile(dist, 40))
            minima = su.local_minima(dist, min_separation=min_frames)
            prominent = [
                i for i in minima
                if np.isfinite(dist[i]) and dist[i] <= near_thr and not contact[i]
            ]
            add(prominent, "distance_min")

        # Vertical direction reversals (the lift/place turning points).
        vy = df["wrist_vy"].to_numpy(dtype=float)
        add(self._sign_changes(vy, min_frames), "vertical_reversal")

        return out

    @staticmethod
    def _transition_intervals(
        rate: np.ndarray, min_frames: int
    ) -> Tuple[List[int], List[int]]:
        """Edges of the intervals over which a signal is actively changing.

        The threshold is a fraction of this video's own 98th-percentile rate, so
        a slow deliberate grasp and a snatch are both detected without tuning.
        """
        r = np.abs(np.asarray(rate, dtype=float))
        finite = r[np.isfinite(r)]
        if finite.size < 4:
            return [], []
        p98 = float(np.percentile(finite, 98))
        if p98 < 1e-6:
            return [], []
        thr = max(0.25 * p98, 0.12)
        active = np.where(np.isfinite(r), r > thr, False)

        starts: List[int] = []
        ends: List[int] = []
        i = 0
        n = active.size
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            # Ignore flickers that are too short to be a real phase.
            if (j - i) >= max(2, min_frames // 2):
                starts.append(i)
                ends.append(min(j, n - 1))
            i = j
        return starts, ends

    @staticmethod
    def _sign_changes(v: np.ndarray, min_sep: int, dead_zone: float = 0.08) -> List[int]:
        """Indices where a velocity component reverses sign meaningfully.

        Values inside ``dead_zone`` are treated as zero so that sensor jitter
        around standstill does not register as a reversal.
        """
        v = np.asarray(v, dtype=float)
        sign = np.zeros(v.size, dtype=int)
        sign[v > dead_zone] = 1
        sign[v < -dead_zone] = -1
        last = 0
        out: List[int] = []
        for i, s in enumerate(sign):
            if s == 0:
                continue
            if last != 0 and s != last:
                if not out or (i - out[-1]) >= min_sep:
                    out.append(i)
            last = s
        return out

    # -- salience & pruning ----------------------------------------------- #
    @staticmethod
    def _salience(df: pd.DataFrame, index: int, cols: List[str], half: int = 4) -> float:
        """How much the feature vector changes across a candidate boundary.

        Used to decide which boundaries to drop when there are too many, so the
        surviving ones are the strongest observed transitions.
        """
        n = len(df)
        a0, a1 = max(0, index - half), index
        b0, b1 = index, min(n, index + half)
        if a1 <= a0 or b1 <= b0:
            return 0.0
        total = 0.0
        for c in cols:
            if c not in df.columns:
                continue
            v = df[c].to_numpy(dtype=float)
            left, right = v[a0:a1], v[b0:b1]
            left = left[~np.isnan(left)]
            right = right[~np.isnan(right)]
            if left.size == 0 or right.size == 0:
                continue
            spread = float(np.nanstd(v[~np.isnan(v)])) if np.isfinite(v).any() else 0.0
            if spread < 1e-9:
                continue
            total += abs(float(np.mean(right) - np.mean(left))) / spread
        return total

    def _merge(
        self, proposals: List[BoundaryProposal], tol: int, n: int
    ) -> List[BoundaryProposal]:
        """Collapse clusters of nearby proposals into one boundary each."""
        valid = sorted(
            (p for p in proposals if 0 < p.index < n), key=lambda p: p.index
        )
        if not valid:
            return []
        # Cluster width is measured from the cluster's *first* member, not its
        # last. Chaining from the last member lets a dense run of proposals merge
        # into one arbitrarily wide cluster -- which was collapsing the whole
        # approach/grasp/lift transition into a single boundary.
        clusters: List[List[BoundaryProposal]] = [[valid[0]]]
        for p in valid[1:]:
            if p.index - clusters[-1][0].index <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        out: List[BoundaryProposal] = []
        for cl in clusters:
            # Keep the highest-priority source; break ties by salience.
            best = max(
                cl,
                key=lambda p: (
                    self.SOURCE_PRIORITY.get(p.source, 0), p.salience
                ),
            )
            sources = "+".join(sorted({p.source for p in cl}))
            out.append(
                BoundaryProposal(
                    best.index, sources, max(p.salience for p in cl)
                )
            )
        return out

    def _priority_of(self, source: str) -> int:
        """Highest detector priority among the sources merged into a boundary."""
        return max(
            (self.SOURCE_PRIORITY.get(s, 0) for s in str(source).split("+")),
            default=0,
        )

    def _enforce_min_length(
        self, proposals: List[BoundaryProposal], min_frames: int, n: int
    ) -> List[BoundaryProposal]:
        """Drop boundaries until every resulting segment is long enough.

        When two boundaries are too close and one has to go, the *semantic* one
        is kept over the statistical one: an interaction event marks a physical
        instant (the grip closing, the object starting to move), whereas a change
        point is only a statistical shadow of that same transition, usually a few
        frames late. Dropping the event in favour of the change point shifts the
        phase boundary off the real event and can flip the resulting label.
        """
        items = sorted(proposals, key=lambda p: p.index)
        while items:
            edges = [0] + [p.index for p in items] + [n]
            lengths = [b - a for a, b in zip(edges[:-1], edges[1:])]
            worst = int(np.argmin(lengths))
            if lengths[worst] >= min_frames:
                break
            # The offending segment is bounded by edges[worst]..edges[worst+1];
            # remove the weaker of the two boundaries that create it.
            left_i = worst - 1      # index into items for edges[worst]
            right_i = worst         # index into items for edges[worst + 1]
            cands = [i for i in (left_i, right_i) if 0 <= i < len(items)]
            if not cands:
                break
            drop = min(
                cands,
                key=lambda i: (self._priority_of(items[i].source), items[i].salience),
            )
            items.pop(drop)
        return items

    @staticmethod
    def _cap_count(
        proposals: List[BoundaryProposal], max_segments: int
    ) -> List[BoundaryProposal]:
        if len(proposals) + 1 <= max_segments:
            return sorted(proposals, key=lambda p: p.index)
        keep = sorted(proposals, key=lambda p: p.salience, reverse=True)[
            : max(0, max_segments - 1)
        ]
        return sorted(keep, key=lambda p: p.index)


@register_segmenter("uniform")
class UniformSegmenter(BaseSegmenter):
    """Fixed-width segmentation. Present only as an ablation baseline.

    Useful for demonstrating in the write-up that the change-point + event
    segmenter is doing real work: swap ``SegmentationConfig.method`` to
    ``"uniform"`` and the phase labels degrade.
    """

    name = "uniform"

    def __init__(self, cfg: SegmentationConfig, window_s: float = 0.6) -> None:
        self.cfg = cfg
        self.window_s = window_s
        self.last_diagnostics: Dict[str, Any] = {}

    def segment(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[Segment]:
        t = df["timestamp"].to_numpy(dtype=float)
        n = len(df)
        dt = float(np.median(np.diff(t))) if n > 1 else 1 / 30.0
        step = max(2, int(round(self.window_s / max(dt, 1e-6))))
        bounds = list(range(step, n, step))
        self.last_diagnostics = {"method": self.name, "window_frames": step}
        return self.boundaries_to_segments(bounds, df)
