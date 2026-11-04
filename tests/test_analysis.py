"""Validation of the analysis stack against synthetic ground truth.

Run:  python tests/test_analysis.py        (no pytest needed)
      pytest tests/test_analysis.py -v     (works too)

The important test is ``test_boundaries_follow_the_data``: the same plan is
rendered at two different speeds and with a different object position, and the
recovered boundaries must move with the data. A pipeline returning canned
timestamps fails it.
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def analyse(demo: syn.SyntheticDemo, cfg: Config | None = None):
    """Run features -> segmentation -> labelling -> classification."""
    cfg = cfg or Config()
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
    labeler = build_labeler(cfg.skills.labeler, cfg.skills)
    segs = labeler.label(segs, ft.df, ctx)
    skill = build_classifier(cfg.skills.classifier, cfg.skills).classify(segs, ft.df, ctx)
    return ft, segs, skill


def _report(name: str, demo, segs, skill) -> None:
    mean_err, errs = syn.boundary_error([s.start_time for s in segs[1:]],
                                        demo.boundaries)
    acc = syn.frame_accuracy(segs, demo.ground_truth, demo.fps)
    print(f"\n--- {name} ---")
    print(f"  ground truth : {' -> '.join(demo.labels)}")
    print(f"  predicted    : {' -> '.join(s.label for s in segs)}")
    print(f"  task         : {skill.task} ({skill.confidence:.2f})")
    print(f"  boundary MAE : {mean_err * 1000:.0f} ms   per-boundary: "
          f"{[round(e, 3) for e in errs]}")
    print(f"  frame acc    : {acc:.1%}")


# --------------------------------------------------------------------------- #
def test_pick_and_place_is_recovered():
    demo = syn.generate(syn.pick_and_place_plan())
    ft, segs, skill = analyse(demo)
    _report("pick_and_place", demo, segs, skill)

    assert ft.object_available, "synthetic object track should be usable"
    assert ft.grip_available, "synthetic grip signal should be usable"
    assert skill.task == "PICK_AND_PLACE", f"got {skill.task}"

    # The contact-defining phases must appear in the right order.
    assert syn.label_order_preserved(segs, ["GRASP", "PLACE", "RELEASE"]), \
        [s.label for s in segs]

    mean_err, _ = syn.boundary_error([s.start_time for s in segs[1:]],
                                     demo.boundaries)
    assert mean_err < 0.30, f"boundary MAE {mean_err:.3f}s is too large"

    acc = syn.frame_accuracy(segs, demo.ground_truth, demo.fps)
    assert acc > 0.55, f"frame accuracy {acc:.2%} too low"


def test_boundaries_follow_the_data():
    """Change the demonstration; the boundaries must change with it."""
    slow = syn.generate(syn.pick_and_place_plan(speed=0.6), seed=3)
    fast = syn.generate(
        syn.pick_and_place_plan(
            speed=1.7, object_at=(620.0, 300.0), place_at=(820.0, 340.0)
        ),
        seed=11,
    )
    _, segs_slow, skill_slow = analyse(slow)
    _, segs_fast, skill_fast = analyse(fast)
    _report("slow (speed 0.6)", slow, segs_slow, skill_slow)
    _report("fast (speed 1.7, moved object)", fast, segs_fast, skill_fast)

    # Grasp onset is the sharpest event; it must track the ground truth.
    def grasp_onset(segs):
        g = [s for s in segs if s.label == "GRASP"]
        return g[0].start_time if g else None

    gt_slow = next(g["start_time"] for g in slow.ground_truth if g["label"] == "GRASP")
    gt_fast = next(g["start_time"] for g in fast.ground_truth if g["label"] == "GRASP")
    p_slow, p_fast = grasp_onset(segs_slow), grasp_onset(segs_fast)

    assert p_slow is not None and p_fast is not None, "GRASP not found in both runs"
    assert abs(p_slow - gt_slow) < 0.45, f"{p_slow:.2f} vs truth {gt_slow:.2f}"
    assert abs(p_fast - gt_fast) < 0.45, f"{p_fast:.2f} vs truth {gt_fast:.2f}"
    # And the two must be genuinely different, not a constant.
    assert abs(p_slow - p_fast) > 1.0, (
        f"grasp onset barely moved ({p_slow:.2f} vs {p_fast:.2f}) despite very "
        f"different demonstrations"
    )


def test_push_is_not_labelled_pick_and_place():
    demo = syn.generate(syn.push_plan(), seed=5)
    _, segs, skill = analyse(demo)
    _report("push", demo, segs, skill)
    labels = [s.label for s in segs]
    assert "GRASP" not in labels, f"a non-prehensile push was labelled GRASP: {labels}"
    assert skill.task in ("PUSH", "UNKNOWN"), f"got {skill.task}"


def test_no_object_degrades_honestly():
    """Without an object track, contact rests on the grip signal alone."""
    demo = syn.generate(syn.pick_and_place_plan(), with_object=False, seed=9)
    ft, segs, skill = analyse(demo)
    _report("no object track", demo, segs, skill)
    assert not ft.object_available
    assert str(ft.df["contact_basis"].iloc[0]) == "grip_only"
    # Grip alone still supports grasp/release; confidences must be discounted.
    assert all(s.confidence <= 0.90 for s in segs)


def test_no_evidence_refuses_contact_phases():
    """With neither object nor grip, contact phases must not be asserted."""
    demo = syn.generate(
        syn.pick_and_place_plan(), with_object=False, with_grip=False, seed=13
    )
    ft, segs, skill = analyse(demo)
    _report("no object, no grip", demo, segs, skill)
    assert not ft.object_available and not ft.grip_available
    labels = {s.label for s in segs}
    forbidden = {"GRASP", "RELEASE", "LIFT", "TRANSPORT", "PLACE", "PUSH", "PULL"}
    assert not (labels & forbidden), f"asserted contact phases without evidence: {labels}"
    assert skill.task == "UNKNOWN", f"invented task {skill.task} without contact evidence"


def test_static_video_is_idle_not_a_skill():
    """A person standing still must not yield a manipulation skill."""
    plan = [syn.Phase("IDLE", 3.0, (380.0, 300.0), "open")]
    demo = syn.generate(plan, seed=2, with_object=False)
    _, segs, skill = analyse(demo)
    _report("static", demo, segs, skill)
    assert skill.task == "UNKNOWN", f"invented {skill.task} from a static clip"


def test_uniform_segmenter_is_worse():
    """The ablation baseline should score lower -- evidence the method matters."""
    demo = syn.generate(syn.pick_and_place_plan(), seed=21)
    cfg_u = Config()
    cfg_u.segmentation.method = "uniform"
    _, segs_cp, _ = analyse(demo)
    _, segs_un, _ = analyse(demo, cfg_u)
    acc_cp = syn.frame_accuracy(segs_cp, demo.ground_truth, demo.fps)
    acc_un = syn.frame_accuracy(segs_un, demo.ground_truth, demo.fps)
    print(f"\n--- ablation ---\n  changepoint_event acc {acc_cp:.1%} "
          f"vs uniform acc {acc_un:.1%}")
    assert acc_cp >= acc_un, "the change-point segmenter should not lose to uniform"


def test_hand_state_labels_are_not_truncated():
    """Guard against numpy fixed-width string truncation.

    np.where("CLOSED", "OPEN") infers dtype <U6, and assigning the 7-character
    "UNKNOWN" into it silently yields "UNKNOW" -- which then propagates into
    trajectory.csv and the JSON as a corrupted label.
    """
    from features.motion_features import (
        HAND_STATE_CLOSED, HAND_STATE_OPEN, HAND_STATE_UNKNOWN,
    )

    valid = {HAND_STATE_OPEN, HAND_STATE_CLOSED, HAND_STATE_UNKNOWN}
    for name, demo in (
        ("grip available", syn.generate(syn.pick_and_place_plan(), seed=1)),
        ("grip unavailable",
         syn.generate(syn.pick_and_place_plan(), with_grip=False, seed=1)),
    ):
        ft, _, _ = analyse(demo)
        seen = set(ft.df["hand_state"].unique())
        assert seen <= valid, f"{name}: unexpected hand_state values {seen - valid}"
    print("\n--- hand_state ---\n  labels intact in both grip conditions")


def test_pelt_recovers_a_known_step():
    """Unit check on the change-point detector itself."""
    from segmentation.changepoint import default_penalty, pelt

    rng = np.random.default_rng(0)
    sig = np.concatenate([
        rng.normal(0.0, 0.25, 120),
        rng.normal(3.0, 0.25, 120),
        rng.normal(-1.5, 0.25, 120),
    ])[:, None]
    bkps = pelt(sig, penalty=default_penalty(len(sig), 1, 1.6), min_size=10)
    print(f"\n--- pelt unit ---\n  detected {bkps} (truth [120, 240])")
    assert len(bkps) == 2, bkps
    assert abs(bkps[0] - 120) <= 4 and abs(bkps[1] - 240) <= 4, bkps


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"\nPASS  {fn.__name__}")
        except AssertionError as exc:
            failed.append((fn.__name__, exc))
            print(f"\nFAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"\nERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 62)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    for name, exc in failed:
        print(f"  FAILED {name}: {exc}")
    raise SystemExit(1 if failed else 0)
