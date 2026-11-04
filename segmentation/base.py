"""Interfaces for temporal segmentation and phase labelling.

These two abstractions are the extension points for the research roadmap. A
learned segmenter (HMM / TCN / Transformer / LSTM) only has to implement
:meth:`BaseSegmenter.segment` -- taking the feature DataFrame and returning
boundary indices -- and a learned classifier only has to implement
:meth:`BaseLabeler.label`. Nothing upstream or downstream changes, because the
contract is the feature table in and :class:`Segment` objects out.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class Segment:
    """One temporal phase of the demonstration."""
    id: int
    start_index: int          # inclusive, into the feature table
    end_index: int            # exclusive
    start_time: float
    end_time: float
    label: str = "UNKNOWN"
    confidence: float = 0.0
    #: Human-readable reasons the label was chosen, for auditing.
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: Segment-level aggregate motion features.
    motion_features: Dict[str, Any] = field(default_factory=dict)
    #: Runner-up labels with their scores.
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    #: Strength of the boundary that opens this segment, and which detector(s)
    #: proposed it. Recorded so a later merge step can tell a weak, incidental
    #: boundary from a strong transition worth preserving, and so the output is
    #: auditable back to the signal that produced it.
    boundary_salience: float = 0.0
    boundary_sources: str = "start_of_video"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def n_frames(self) -> int:
        return max(0, self.end_index - self.start_index)


class BaseSegmenter(ABC):
    """Proposes *unlabelled* temporal boundaries from motion features."""

    name = "base"

    @abstractmethod
    def segment(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[Segment]:
        """Return contiguous, gap-free segments covering the whole table."""

    # -- shared helpers --------------------------------------------------- #
    @staticmethod
    def boundaries_to_segments(
        boundaries: Sequence[int],
        df: pd.DataFrame,
        salience: Optional[Dict[int, float]] = None,
        sources: Optional[Dict[int, str]] = None,
    ) -> List[Segment]:
        """Turn interior boundary indices into contiguous segments."""
        n = len(df)
        t = df["timestamp"].to_numpy()
        edges = [0] + sorted({int(b) for b in boundaries if 0 < int(b) < n}) + [n]
        segs: List[Segment] = []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            if b <= a:
                continue
            # A segment's end time is the start of the next segment, so the
            # timeline has no gaps and no overlaps.
            end_t = float(t[b]) if b < n else float(t[-1])
            segs.append(
                Segment(
                    id=i,
                    start_index=a,
                    end_index=b,
                    start_time=float(t[a]),
                    end_time=end_t,
                    boundary_salience=(
                        float((salience or {}).get(a, 0.0)) if a > 0 else 0.0
                    ),
                    boundary_sources=(
                        (sources or {}).get(a, "unknown") if a > 0
                        else "start_of_video"
                    ),
                )
            )
        # Renumber after any dropped empties.
        for i, s in enumerate(segs, start=1):
            s.id = i
        return segs


class BaseLabeler(ABC):
    """Assigns a phase label and confidence to each segment."""

    name = "base"

    @abstractmethod
    def label(
        self, segments: List[Segment], df: pd.DataFrame, context: Dict[str, Any]
    ) -> List[Segment]:
        """Return the same segments with ``label``/``confidence`` populated."""


# --------------------------------------------------------------------------- #
# Registries -- swap implementations by name from utils.config
# --------------------------------------------------------------------------- #
SEGMENTERS: Dict[str, Callable[..., BaseSegmenter]] = {}
LABELERS: Dict[str, Callable[..., BaseLabeler]] = {}


def register_segmenter(key: str):
    def deco(cls):
        SEGMENTERS[key] = cls
        return cls
    return deco


def register_labeler(key: str):
    def deco(cls):
        LABELERS[key] = cls
        return cls
    return deco


def build_segmenter(key: str, *args, **kwargs) -> BaseSegmenter:
    if key not in SEGMENTERS:
        raise KeyError(
            f"Unknown segmenter '{key}'. Registered: {sorted(SEGMENTERS)}"
        )
    return SEGMENTERS[key](*args, **kwargs)


def build_labeler(key: str, *args, **kwargs) -> BaseLabeler:
    if key not in LABELERS:
        raise KeyError(f"Unknown labeler '{key}'. Registered: {sorted(LABELERS)}")
    return LABELERS[key](*args, **kwargs)
