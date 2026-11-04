"""Quantitative evaluation of the segmentation and skill-identification stack.

Reports the standard temporal-action-segmentation metric suite (Lea et al.,
"Temporal Convolutional Networks for Action Segmentation and Detection", CVPR
2017), computed against parameterised synthetic ground truth:

* **MoF**        - mean-over-frames accuracy, the fraction of time labelled correctly
* **Edit**       - normalised Levenshtein distance over the segment label
                   sequence; penalises over-segmentation, ignores exact timing
* **F1@{10,25,50}** - segmental F1 at IoU thresholds; a predicted segment counts
                   as a hit if it shares a label with an unmatched ground-truth
                   segment at IoU above the threshold
* **Boundary MAE** - for each true boundary, distance to the nearest predicted one
* **Task acc.**  - top-1 accuracy of the identified skill

Run:
    py -3.11 tests/benchmark.py              # full suite, writes outputs/metrics.json
    py -3.11 tests/benchmark.py --quick      # fewer conditions
    py -3.11 tests/benchmark.py --video X    # also measure runtime on a real clip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from features.motion_features import MotionFeatureExtractor
from segmentation.base import build_labeler, build_segmenter
from skills.skill_classifier import build_classifier
from utils.config import Config
from tests import synthetic as syn

import segmentation.temporal_segmenter  # noqa: F401  (registry side effects)
import segmentation.phase_labeler       # noqa: F401
import skills.skill_classifier          # noqa: F401


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def frame_labels(segments, truth, fps: float) -> Tuple[List[str], List[str]]:
    """Sample predicted and true labels on a common frame grid."""
    total = truth[-1]["end_time"]
    n = max(1, int(round(total * fps)))
    pred, gt = [], []
    for k in range(n):
        t = (k + 0.5) / fps
        gt.append(next(
            (g["label"] for g in truth if g["start_time"] <= t < g["end_time"]),
            truth[-1]["label"],
        ))
        pred.append(next(
            (s.label for s in segments if s.start_time <= t < s.end_time),
            segments[-1].label if segments else "UNKNOWN",
        ))
    return pred, gt


def mof(pred: Sequence[str], gt: Sequence[str]) -> float:
    if not gt:
        return 0.0
    return float(np.mean([p == g for p, g in zip(pred, gt)]))


def _collapse(labels: Sequence[str]) -> List[str]:
    out: List[str] = []
    for l in labels:
        if not out or out[-1] != l:
            out.append(l)
    return out


def edit_score(pred: Sequence[str], gt: Sequence[str]) -> float:
    """Normalised Levenshtein over collapsed label sequences, in [0, 100]."""
    a, b = _collapse(pred), _collapse(gt)
    m, n = len(a), len(b)
    if max(m, n) == 0:
        return 100.0
    d = np.zeros((m + 1, n + 1), dtype=int)
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float((1.0 - d[m, n] / max(m, n)) * 100.0)


def _intervals(labels: Sequence[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out


def segmental_f1(pred: Sequence[str], gt: Sequence[str], threshold: float) -> float:
    """Segmental F1 at an IoU threshold, over frame-sampled label sequences."""
    p_int = _intervals(pred)
    g_int = _intervals(gt)
    if not p_int and not g_int:
        return 100.0
    used = [False] * len(g_int)
    tp = 0
    for lab, ps, pe in p_int:
        best_iou, best_j = 0.0, -1
        for j, (glab, gs, ge) in enumerate(g_int):
            if glab != lab or used[j]:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= threshold:
            used[best_j] = True
            tp += 1
    fp = len(p_int) - tp
    fn = len(g_int) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall) * 100.0)


def boundary_mae(segments, truth) -> float:
    pred = [s.start_time for s in segments[1:]]
    true = [g["start_time"] for g in truth[1:]]
    if not true:
        return float("nan")
    if not pred:
        return float("inf")
    return float(np.mean([min(abs(p - t) for p in pred) for t in true]))


def boundary_detection_f1(segments, truth, tolerance: float) -> float:
    """Boundary-level F1 within a time tolerance (seconds)."""
    pred = sorted(s.start_time for s in segments[1:])
    true = sorted(g["start_time"] for g in truth[1:])
    if not true and not pred:
        return 100.0
    matched = [False] * len(pred)
    tp = 0
    for t in true:
        for i, p in enumerate(pred):
            if not matched[i] and abs(p - t) <= tolerance:
                matched[i] = True
                tp += 1
                break
    fp, fn = len(pred) - tp, len(true) - tp
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall) * 100.0)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    name: str
    demo: syn.SyntheticDemo
    expected_task: str
    cfg: Config


def analyse(demo: syn.SyntheticDemo, cfg: Config):
    cfg = cfg.resolve()
    ft = MotionFeatureExtractor(cfg.features, cfg.pose).build(
        demo.sequence, demo.track if demo.track.observations else None
    )
    ctx = {
        "body_scale_px": ft.body_scale_px,
        "object_available": ft.object_available,
        "grip_available": ft.grip_available,
        "contact_basis": str(ft.df["contact_basis"].iloc[0]),
        "pose_detection_rate": demo.sequence.detection_rate,
        "pose_backend": "synthetic",
    }
    segs = build_segmenter(cfg.segmentation.method, cfg.segmentation).segment(ft.df, ctx)
    segs = build_labeler(cfg.skills.labeler, cfg.skills).label(segs, ft.df, ctx)
    skill = build_classifier(cfg.skills.classifier, cfg.skills).classify(segs, ft.df, ctx)
    return ft, segs, skill


def evaluate(case: Case) -> Dict:
    t0 = time.perf_counter()
    _, segs, skill = analyse(case.demo, case.cfg)
    elapsed = time.perf_counter() - t0
    pred, gt = frame_labels(segs, case.demo.ground_truth, case.demo.fps)
    return {
        "case": case.name,
        "expected_task": case.expected_task,
        "predicted_task": skill.task,
        "task_correct": skill.task == case.expected_task,
        "task_confidence": round(skill.confidence, 4),
        "mof": round(mof(pred, gt) * 100, 2),
        "edit": round(edit_score(pred, gt), 2),
        "f1_10": round(segmental_f1(pred, gt, 0.10), 2),
        "f1_25": round(segmental_f1(pred, gt, 0.25), 2),
        "f1_50": round(segmental_f1(pred, gt, 0.50), 2),
        "boundary_mae_ms": round(boundary_mae(segs, case.demo.ground_truth) * 1000, 1),
        "boundary_f1_100ms": round(
            boundary_detection_f1(segs, case.demo.ground_truth, 0.10), 2),
        "boundary_f1_200ms": round(
            boundary_detection_f1(segs, case.demo.ground_truth, 0.20), 2),
        "n_pred_segments": len(segs),
        "n_true_segments": len(case.demo.ground_truth),
        "analysis_s": round(elapsed, 4),
        "duration_s": round(case.demo.duration, 2),
    }


def aggregate(rows: List[Dict], keys: Sequence[str]) -> Dict:
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows
                if isinstance(r.get(k), (int, float, bool)) and np.isfinite(r[k])]
        if vals:
            out[k] = round(float(np.mean(vals)), 2)
    out["n"] = len(rows)
    return out


METRIC_KEYS = ("mof", "edit", "f1_10", "f1_25", "f1_50",
               "boundary_mae_ms", "boundary_f1_100ms", "boundary_f1_200ms",
               "task_correct", "n_pred_segments", "analysis_s")


def build_cases(quick: bool) -> List[Case]:
    cfg = Config()
    speeds = (0.7, 1.0, 1.4) if quick else (0.6, 0.8, 1.0, 1.2, 1.5)
    noises = (0.7,) if quick else (0.4, 0.7, 1.2)
    seeds = (1, 2) if quick else (1, 2, 3)
    objects = ((560.0, 320.0),) if quick else ((560.0, 320.0), (620.0, 300.0))

    cases: List[Case] = []
    for sp in speeds:
        for nz in noises:
            for sd in seeds:
                for obj in objects:
                    cases.append(Case(
                        name=f"pick_and_place/speed={sp}/noise={nz}/obj={obj[0]:.0f}/seed={sd}",
                        demo=syn.generate(
                            syn.pick_and_place_plan(speed=sp, object_at=obj),
                            noise_px=nz, seed=sd, object_at=obj,
                        ),
                        expected_task="PICK_AND_PLACE",
                        cfg=cfg,
                    ))
    for sp in speeds:
        for sd in seeds:
            cases.append(Case(
                name=f"push/speed={sp}/seed={sd}",
                demo=syn.generate(syn.push_plan(speed=sp), noise_px=0.7, seed=sd,
                                  object_at=(540.0, 330.0)),
                expected_task="PUSH",
                cfg=cfg,
            ))
    return cases


def ablations(quick: bool) -> Dict[str, Dict]:
    """Compare the full method against reduced variants on the same data."""
    variants: Dict[str, Config] = {}

    full = Config()
    variants["full (change-point + events)"] = full

    cp_only = Config()
    cp_only.segmentation.use_event_boundaries = False
    variants["change-point only"] = cp_only

    uniform = Config()
    uniform.segmentation.method = "uniform"
    variants["uniform windows (baseline)"] = uniform

    coarse = Config()
    coarse.segmentation.penalty_scale = 4.0
    variants["change-point + events, penalty x2.5"] = coarse

    speeds = (0.7, 1.0, 1.4) if quick else (0.6, 0.8, 1.0, 1.2, 1.5)
    seeds = (1, 2) if quick else (1, 2, 3)

    results: Dict[str, Dict] = {}
    for name, cfg in variants.items():
        rows = []
        for sp in speeds:
            for sd in seeds:
                demo = syn.generate(syn.pick_and_place_plan(speed=sp),
                                    noise_px=0.7, seed=sd)
                rows.append(evaluate(Case(f"{name}/{sp}/{sd}", demo,
                                          "PICK_AND_PLACE", cfg)))
        results[name] = aggregate(rows, METRIC_KEYS)
    return results


def robustness(quick: bool) -> Dict[str, Dict]:
    """Degradation as observation noise rises and as evidence channels are removed."""
    out: Dict[str, Dict] = {}
    cfg = Config()
    seeds = (1, 2) if quick else (1, 2, 3, 4)

    for nz in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        rows = [
            evaluate(Case(f"noise={nz}/{sd}",
                          syn.generate(syn.pick_and_place_plan(), noise_px=nz, seed=sd),
                          "PICK_AND_PLACE", cfg))
            for sd in seeds
        ]
        out[f"landmark noise {nz:.1f} px"] = aggregate(rows, METRIC_KEYS)

    channels = {
        "all channels": dict(with_object=True, with_grip=True),
        "no object track": dict(with_object=False, with_grip=True),
        "no grip signal": dict(with_object=True, with_grip=False),
        "neither": dict(with_object=False, with_grip=False),
    }
    for name, kw in channels.items():
        rows = [
            evaluate(Case(f"{name}/{sd}",
                          syn.generate(syn.pick_and_place_plan(), noise_px=0.7,
                                       seed=sd, **kw),
                          "PICK_AND_PLACE", cfg))
            for sd in seeds
        ]
        out[f"evidence: {name}"] = aggregate(rows, METRIC_KEYS)
    return out


def smoothing_tradeoff(quick: bool) -> Dict[str, Dict]:
    """Temporal resolution vs noise rejection, measured.

    The smoothing window is clamped to the minimum segment duration, so this
    sweeps both together. A long window rejects landmark jitter but blurs short
    phases into their neighbours; a short window resolves a fast PLACE but lets
    noise through. The two columns that matter are task accuracy on fast
    demonstrations and MoF under heavy noise.
    """
    speeds = (0.8, 1.2, 1.5) if quick else (0.6, 0.8, 1.0, 1.2, 1.5)
    seeds = (1, 2) if quick else (1, 2, 3)
    out: Dict[str, Dict] = {}

    for win in (0.10, 0.16, 0.22, 0.30):
        rows_fast, rows_noisy = [], []
        for sp in speeds:
            for sd in seeds:
                cfg = Config()
                cfg.segmentation.min_segment_duration_s = win
                cfg.features.smooth_window_s = win
                rows_fast.append(evaluate(Case(
                    f"w{win}/sp{sp}/{sd}",
                    syn.generate(syn.pick_and_place_plan(speed=sp),
                                 noise_px=0.7, seed=sd),
                    "PICK_AND_PLACE", cfg)))
        for sd in seeds:
            for nz in (2.0, 3.5, 5.0):
                cfg = Config()
                cfg.segmentation.min_segment_duration_s = win
                cfg.features.smooth_window_s = win
                rows_noisy.append(evaluate(Case(
                    f"w{win}/nz{nz}/{sd}",
                    syn.generate(syn.pick_and_place_plan(), noise_px=nz, seed=sd),
                    "PICK_AND_PLACE", cfg)))
        agg = aggregate(rows_fast, METRIC_KEYS)
        agg["mof_noisy"] = aggregate(rows_noisy, METRIC_KEYS)["mof"]
        agg["task_correct_noisy"] = aggregate(rows_noisy, METRIC_KEYS)["task_correct"]
        out[f"window {win:.2f} s"] = agg
    return out


def negative_controls() -> List[Dict]:
    """Clips containing no manipulation must not yield a skill."""
    cfg = Config()
    cases = [
        ("static person (no motion)",
         syn.generate([syn.Phase("IDLE", 4.0, (380.0, 300.0), "open")],
                      seed=2, with_object=False)),
        ("arm motion, no object, no grip signal",
         syn.generate(syn.pick_and_place_plan(), with_object=False,
                      with_grip=False, seed=13)),
        ("reach and retract only, no contact",
         syn.generate([syn.Phase("REACH", 1.5, (540.0, 310.0), "open"),
                       syn.Phase("RETRACT", 1.5, (380.0, 300.0), "open")],
                      seed=4, with_object=False)),
    ]
    out = []
    for name, demo in cases:
        _, segs, skill = analyse(demo, cfg)
        out.append({
            "case": name,
            "reported_task": skill.task,
            "confidence": round(skill.confidence, 4),
            "correctly_refused": skill.task == "UNKNOWN",
            "phase_sequence": " -> ".join(s.label for s in segs),
        })
    return out


def runtime_on_video(video: str) -> Optional[Dict]:
    """Wall-clock cost of the real perception stack, per stage."""
    from pipeline import SkillExtractionPipeline
    from video.video_loader import probe

    meta = probe(video)
    rows = {}

    # Warm-up run: the first invocation pays for TFLite/XNNPACK delegate setup
    # and YOLO weight loading, which would otherwise be charged entirely to
    # whichever configuration happens to run first.
    warm = Config()
    warm.video.target_long_edge = 480
    warm.video.max_frames = 20
    warm.viz.write_video = False
    SkillExtractionPipeline(warm).run(video)

    for long_edge in (640, 960):
        cfg = Config()
        cfg.video.target_long_edge = long_edge
        cfg.viz.write_video = True
        res = SkillExtractionPipeline(cfg).run(video)
        if not res.ok:
            continue
        frames = res.diagnostics["frame_diagnostics"]["frames_processed"]
        t = res.timings
        rows[f"{long_edge}px long edge"] = {
            "frames_processed": frames,
            "perception_s": t.get("perception_s"),
            "features_s": t.get("features_s"),
            "segmentation_s": t.get("segmentation_s"),
            "render_s": t.get("render_s"),
            "total_s": t.get("total_s"),
            "perception_fps": round(frames / max(t.get("perception_s", 1e-9), 1e-9), 2),
            "realtime_factor": round(
                t.get("total_s", 0) / max(meta.duration_s, 1e-9), 2),
        }
    return {
        "video": Path(video).name,
        "duration_s": round(meta.duration_s, 2),
        "resolution": meta.resolution,
        "fps": meta.fps,
        "runs": rows,
    } or None


# --------------------------------------------------------------------------- #
def table(title: str, rows: Dict[str, Dict], cols: Sequence[str]) -> str:
    head = f"\n{title}\n" + "-" * len(title) + "\n"
    head += f"{'condition':<34}" + "".join(f"{c:>13}" for c in cols) + "\n"
    for name, vals in rows.items():
        head += f"{name:<34}" + "".join(
            f"{vals.get(c, float('nan')):>13.2f}" if isinstance(vals.get(c), (int, float))
            else f"{'-':>13}" for c in cols
        ) + "\n"
    return head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--video", default=None, help="real clip for runtime measurement")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "metrics.json"))
    args = ap.parse_args(argv)

    print("Running benchmark (synthetic ground truth)...")
    cases = build_cases(args.quick)
    rows = [evaluate(c) for c in cases]

    pnp = [r for r in rows if r["expected_task"] == "PICK_AND_PLACE"]
    push = [r for r in rows if r["expected_task"] == "PUSH"]

    overall = {
        "all demonstrations": aggregate(rows, METRIC_KEYS),
        "pick-and-place": aggregate(pnp, METRIC_KEYS),
        "push": aggregate(push, METRIC_KEYS),
    }
    cols = ("mof", "edit", "f1_10", "f1_25", "f1_50", "boundary_mae_ms",
            "task_correct", "n")
    print(table("Overall", overall, cols))

    abl = ablations(args.quick)
    print(table("Ablation (pick-and-place)", abl, cols))

    rob = robustness(args.quick)
    print(table("Robustness", rob, cols))

    smooth = smoothing_tradeoff(args.quick)
    print(table("Smoothing window / temporal resolution trade-off", smooth,
                ("mof", "task_correct", "boundary_mae_ms",
                 "mof_noisy", "task_correct_noisy", "n")))

    neg = negative_controls()
    print("\nNegative controls (must report UNKNOWN)")
    print("-" * 39)
    for n in neg:
        mark = "OK " if n["correctly_refused"] else "FAIL"
        print(f"  [{mark}] {n['case']:<38} -> {n['reported_task']} "
              f"({n['confidence']:.2f})")

    run = runtime_on_video(args.video) if args.video else None
    if run:
        print("\nRuntime")
        print("-" * 7)
        print(f"  clip: {run['video']} ({run['resolution']}, {run['duration_s']}s)")
        for k, v in run["runs"].items():
            print(f"  {k:<20} perception {v['perception_fps']:>6.2f} fps | "
                  f"total {v['total_s']:>6.2f}s | {v['realtime_factor']:>5.2f}x realtime")

    doc = {
        "overall": overall,
        "per_case": rows,
        "ablation": abl,
        "robustness": rob,
        "smoothing_tradeoff": smooth,
        "negative_controls": neg,
        "runtime": run,
        "n_demonstrations": len(rows),
        "metric_definitions": {
            "mof": "mean-over-frames accuracy (%)",
            "edit": "normalised Levenshtein edit score over segment sequence (%)",
            "f1_10/25/50": "segmental F1 at IoU 0.10/0.25/0.50 (%)",
            "boundary_mae_ms": "mean distance from each true boundary to the "
                               "nearest predicted boundary (ms)",
            "boundary_f1_100ms": "boundary detection F1 within a 100 ms tolerance (%)",
            "task_correct": "fraction of clips whose identified skill was correct",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
