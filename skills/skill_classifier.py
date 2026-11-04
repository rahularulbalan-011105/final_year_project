"""Skill identification (Step 5 of the pipeline).

The overall task is inferred from the *sequence* of phase labels, not from the
video directly. Each candidate skill is a template with required phases (which
must appear in order), optional phases that add support, and optional extra
geometric conditions.

If no template clears ``SkillConfig.min_task_confidence`` the result is
``UNKNOWN`` with its score reported -- the pipeline never invents a skill name.

Swapping this for a learned classifier means implementing
:meth:`BaseSkillClassifier.classify`; the input (labelled segments + feature
table) and the output (:class:`SkillResult`) stay the same.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from segmentation.base import Segment
from utils.config import SkillConfig


@dataclass
class SkillResult:
    task: str
    confidence: float
    matched_template: Optional[str] = None
    rationale: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)


class BaseSkillClassifier(ABC):
    name = "base"

    @abstractmethod
    def classify(
        self,
        segments: List[Segment],
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> SkillResult:
        ...


CLASSIFIERS: Dict[str, Callable[..., BaseSkillClassifier]] = {}


def register_classifier(key: str):
    def deco(cls):
        CLASSIFIERS[key] = cls
        return cls
    return deco


def build_classifier(key: str, *args, **kwargs) -> BaseSkillClassifier:
    if key not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier '{key}'. Registered: {sorted(CLASSIFIERS)}")
    return CLASSIFIERS[key](*args, **kwargs)


@dataclass
class SkillTemplate:
    name: str
    #: Must all appear, in this relative order.
    required: Tuple[str, ...]
    #: Add support when present, but are not necessary.
    optional: Tuple[str, ...] = ()
    #: Presence of any of these argues against the template.
    forbidden: Tuple[str, ...] = ()
    #: True when the skill is defined by acting on an object. Such a template is
    #: ineligible without a reliable object track: a grip signal alone shows the
    #: hand closing, not that anything was picked up.
    requires_object: bool = True
    #: Extra geometric test: ``fn(segments, df, context) -> float in [0, 1]``
    #: or ``None`` when it cannot be evaluated.
    condition: Optional[Callable] = None
    condition_weight: float = 0.0
    condition_description: str = ""


def _ordered_subsequence_coverage(
    required: Sequence[str], observed: Sequence[str]
) -> Tuple[float, List[int]]:
    """Greedily match ``required`` as an ordered subsequence of ``observed``.

    Returns the fraction matched and the indices in ``observed`` that matched,
    so the caller can weight by those segments' own confidences.
    """
    hits: List[int] = []
    pos = 0
    for want in required:
        found = -1
        for i in range(pos, len(observed)):
            if observed[i] == want:
                found = i
                break
        if found >= 0:
            hits.append(found)
            pos = found + 1
    return (len(hits) / max(len(required), 1)), hits


def _placed_higher(segments: List[Segment], df: pd.DataFrame, ctx: Dict) -> Optional[float]:
    """Did the object come to rest higher than where it was picked up?

    Evidence for stacking. Uses image-plane vertical position only; no depth is
    assumed. Returns None when no object was tracked, so the STACK template
    simply loses this term rather than scoring zero.

    The comparison is against the object's *final resting* position, not against
    a particular PLACE segment: with several PLACE segments, picking one of them
    can sample the object mid-lift and make any pick-and-place look like a stack.
    """
    if not ctx.get("object_available") or "object_y" not in df.columns:
        return None
    grasp = [s for s in segments if s.label == "GRASP"]
    if not grasp or not any(s.label == "PLACE" for s in segments):
        return None
    scale = max(float(ctx.get("body_scale_px", 1.0)), 1e-9)

    oy = df["object_y"].to_numpy(dtype=float)

    def mean_y(lo: int, hi: int) -> Optional[float]:
        v = oy[lo:hi]
        v = v[~np.isnan(v)]
        return float(np.mean(v)) if v.size else None

    g = grasp[0]
    y_pick = mean_y(g.start_index, g.end_index)
    tail = max(1, int(round(0.12 * len(oy))))
    y_rest = mean_y(len(oy) - tail, len(oy))
    if y_pick is None or y_rest is None:
        return None
    # Smaller y is higher in the image.
    rise_bw = (y_pick - y_rest) / scale
    return float(np.clip(rise_bw / 0.35, 0.0, 1.0))


def _object_actually_moved(segments, df: pd.DataFrame, ctx: Dict) -> Optional[float]:
    """Did the tracked object travel a meaningful distance overall?"""
    if not ctx.get("object_available"):
        return None
    ox = df["object_x"].to_numpy(dtype=float)
    oy = df["object_y"].to_numpy(dtype=float)
    ok = ~(np.isnan(ox) | np.isnan(oy))
    if ok.sum() < 2:
        return None
    scale = max(float(ctx.get("body_scale_px", 1.0)), 1e-9)
    pts = np.stack([ox[ok], oy[ok]], axis=1) / scale
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return float(np.clip(span / 0.6, 0.0, 1.0))


TEMPLATES: Tuple[SkillTemplate, ...] = (
    SkillTemplate(
        name="PICK_AND_PLACE",
        required=("GRASP", "PLACE", "RELEASE"),
        optional=("REACH", "APPROACH", "LIFT", "TRANSPORT", "RETRACT"),
        condition=_object_actually_moved,
        condition_weight=0.15,
        condition_description="tracked object was displaced during the demonstration",
    ),
    SkillTemplate(
        name="STACK",
        required=("GRASP", "PLACE", "RELEASE"),
        optional=("REACH", "APPROACH", "LIFT", "TRANSPORT", "RETRACT"),
        condition=_placed_higher,
        condition_weight=0.30,
        condition_description="object finished higher in the image than where it was grasped",
    ),
    SkillTemplate(
        name="PICK_AND_MOVE",
        required=("GRASP", "TRANSPORT"),
        optional=("REACH", "APPROACH", "LIFT", "RETRACT"),
        forbidden=("PLACE",),
        condition=_object_actually_moved,
        condition_weight=0.15,
        condition_description="tracked object was displaced during the demonstration",
    ),
    SkillTemplate(
        name="TRANSFER",
        required=("GRASP", "TRANSPORT", "RELEASE"),
        optional=("REACH", "APPROACH", "LIFT", "RETRACT"),
        forbidden=("PLACE",),
    ),
    SkillTemplate(
        name="PUSH",
        required=("PUSH",),
        optional=("REACH", "APPROACH", "RETRACT", "IDLE"),
        forbidden=("GRASP",),
    ),
    SkillTemplate(
        name="PULL",
        required=("PULL",),
        optional=("REACH", "APPROACH", "RETRACT", "IDLE"),
        forbidden=("GRASP",),
    ),
    SkillTemplate(
        name="GRASP",
        required=("GRASP",),
        optional=("REACH", "APPROACH", "LIFT"),
        forbidden=("PLACE", "TRANSPORT", "RELEASE"),
        requires_object=False,
    ),
    SkillTemplate(
        name="RELEASE",
        required=("RELEASE",),
        optional=("RETRACT", "IDLE", "PLACE"),
        forbidden=("GRASP",),
        requires_object=False,
    ),
)


@register_classifier("template")
class TemplateSkillClassifier(BaseSkillClassifier):
    """Sequence-template matching over the decoded phase labels."""

    name = "template"

    # Relative contribution of each evidence source to a template's score.
    W_REQUIRED = 0.55
    W_OPTIONAL = 0.15
    W_DURATION = 0.15
    W_SEGCONF = 0.15

    def __init__(self, cfg: SkillConfig) -> None:
        self.cfg = cfg

    def classify(
        self, segments: List[Segment], df: pd.DataFrame, context: Dict[str, Any]
    ) -> SkillResult:
        labelled = [s for s in segments if s.label != "UNKNOWN"]
        observed = [s.label for s in segments]
        total_duration = sum(s.duration for s in segments) or 1e-9

        if not labelled:
            return SkillResult(
                task="UNKNOWN",
                confidence=0.0,
                rationale=(
                    "No segment could be labelled with sufficient confidence, so "
                    "no task is asserted."
                ),
            )

        has_object = bool(context.get("object_available"))
        rejected: List[Dict[str, Any]] = []

        scored: List[Tuple[float, SkillTemplate, Dict[str, Any]]] = []
        for tpl in TEMPLATES:
            cov, hit_idx = _ordered_subsequence_coverage(tpl.required, observed)

            # "Required" means required. A partial match is not a match: allowing
            # one would let a TRANSFER be declared from a GRASP and a RELEASE
            # with no TRANSPORT anywhere in the demonstration.
            if cov < 1.0 - 1e-9:
                if cov > 0.0:
                    rejected.append({
                        "task": tpl.name,
                        "rejected": "missing required phase(s)",
                        "required_phases": list(tpl.required),
                        "required_coverage": round(cov, 4),
                    })
                continue

            if tpl.requires_object and not has_object:
                rejected.append({
                    "task": tpl.name,
                    "rejected": (
                        "this skill is defined by acting on an object, and no "
                        "object track was reliable enough to confirm one"
                    ),
                })
                continue

            present = set(observed)
            opt_hits = len(present & set(tpl.optional))
            opt = opt_hits / max(len(tpl.optional), 1) if tpl.optional else 0.0

            forbidden_hits = sorted(present & set(tpl.forbidden))
            # A forbidden phase does not veto outright -- a single mislabelled
            # short segment should not eliminate an otherwise strong match -- but
            # it must cost enough to lose to a template that fits cleanly.
            forbidden_penalty = 0.5 ** len(forbidden_hits)

            covered = sum(
                s.duration for s in segments
                if s.label in set(tpl.required) | set(tpl.optional)
            )
            dur = covered / total_duration

            if hit_idx:
                segconf = float(np.mean([segments[i].confidence for i in hit_idx]))
            else:
                segconf = 0.0

            base = (
                self.W_REQUIRED * cov
                + self.W_OPTIONAL * opt
                + self.W_DURATION * dur
                + self.W_SEGCONF * segconf
            )

            cond_value = None
            if tpl.condition is not None:
                cond_value = tpl.condition(segments, df, context)
            if tpl.condition_weight > 0 and cond_value is not None:
                base = (
                    base * (1.0 - tpl.condition_weight)
                    + tpl.condition_weight * float(cond_value)
                )

            # Evidence discount. A hand-only template (GRASP, RELEASE) stays
            # eligible without an object track, but it is then resting entirely
            # on the aperture signal: the hand closed, or opened. That the hand
            # opened is not proof that anything was being held, so the claim is
            # scaled down to reflect the missing evidence -- the same reasoning
            # the phase labeler applies to its own confidences.
            evidence_factor = 1.0 if has_object else 0.82

            total = float(np.clip(
                base * forbidden_penalty * evidence_factor, 0.0, 1.0
            ))
            scored.append((total, tpl, {
                "task": tpl.name,
                "score": round(total, 4),
                "required_coverage": round(cov, 4),
                "required_phases": list(tpl.required),
                "optional_coverage": round(opt, 4),
                "duration_coverage": round(dur, 4),
                "mean_matched_segment_confidence": round(segconf, 4),
                "forbidden_present": forbidden_hits,
                "evidence_factor": round(evidence_factor, 3),
                "condition": tpl.condition_description or None,
                "condition_value": (
                    None if cond_value is None else round(float(cond_value), 4)
                ),
            }))

        scored.sort(key=lambda s: s[0], reverse=True)
        candidates = [info for _, _, info in scored[:6]] + rejected[:6]

        if not scored or scored[0][0] < self.cfg.min_task_confidence:
            best = scored[0][0] if scored else 0.0
            if scored:
                why = (
                    f"Best template match was '{scored[0][1].name}' at "
                    f"{best:.2f}, below the "
                    f"{self.cfg.min_task_confidence:.2f} confidence floor."
                )
            else:
                why = (
                    "No skill template had all of its required phases present "
                    "in order"
                    + ("" if has_object else
                       " (and no reliable object track was available, which rules "
                       "out every object-directed skill)")
                    + "."
                )
            return SkillResult(
                task="UNKNOWN",
                confidence=round(float(best), 4),
                matched_template=None,
                rationale=(
                    f"{why} Observed phase sequence: {' -> '.join(observed)}."
                ),
                candidates=candidates,
            )

        total, tpl, info = scored[0]
        margin = total - (scored[1][0] if len(scored) > 1 else 0.0)
        return SkillResult(
            task=tpl.name,
            confidence=round(float(total), 4),
            matched_template=tpl.name,
            rationale=(
                f"Phase sequence {' -> '.join(observed)} matched template "
                f"'{tpl.name}' (required {'/'.join(tpl.required)} found in order, "
                f"coverage {info['required_coverage']:.2f}, "
                f"{info['duration_coverage']:.0%} of the timeline accounted for, "
                f"margin over next best {margin:.2f})."
            ),
            candidates=candidates,
        )
