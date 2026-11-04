"""Signal-health report for a demonstration video.

Wrong phase labels are almost always a symptom: one of the four input signals
(pose, grip, object, contact) was weak, and the labeller did the best it could
with what it was given. This tool checks each signal independently and says
which one to fix, in filming terms.

    py -3.11 diagnose.py demo.mp4
    py -3.11 diagnose.py demo.mp4 --object bottle
    py -3.11 diagnose.py --list-objects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from pipeline import SkillExtractionPipeline
from utils.config import Config

OK, WARN, BAD = "OK  ", "WARN", "BAD "


def bar(value: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * n + "." * (width - n)


def line(status: str, name: str, value: str, advice: str = "") -> None:
    print(f"  [{status}] {name:<26} {value}")
    if advice:
        for chunk in _wrap(advice, 86):
            print(f"         -> {chunk}")


def _wrap(text: str, width: int):
    words, cur = text.split(), ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            yield cur
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        yield cur


def list_objects() -> int:
    try:
        from ultralytics import YOLO
        from tracking.object_tracker import ObjectDetector
        det = ObjectDetector(Config().objects)
        if not det.available:
            print(det.unavailable_reason)
            return 1
        names = sorted(str(v) for v in det.names.values())
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load the detector: {exc}")
        return 1

    good = [n for n in (
        "bottle", "cup", "bowl", "wine glass", "book", "cell phone", "mouse",
        "remote", "keyboard", "scissors", "banana", "apple", "orange",
        "sports ball", "teddy bear", "vase", "fork", "knife", "spoon", "clock",
        "toothbrush", "laptop", "backpack", "handbag", "umbrella", "suitcase",
    ) if n in names]
    print("Classes well suited to a tabletop manipulation demo:\n")
    for i in range(0, len(good), 4):
        print("   " + "".join(f"{n:<18}" for n in good[i:i + 4]))
    print(f"\nAll {len(names)} detectable classes:\n")
    for i in range(0, len(names), 4):
        print("   " + "".join(f"{n:<18}" for n in names[i:i + 4]))
    print("\nPass one with --object, e.g.  py -3.11 cli.py demo.mp4 --object bottle")
    print("If your object is not listed, the detector cannot see it: either swap")
    print("the object, or run with --no-objects and rely on the grip signal alone.")
    return 0


def diagnose(video: str, target: str | None, stride: int) -> int:
    cfg = Config()
    cfg.viz.write_video = False
    cfg.video.frame_stride = stride
    cfg.objects.target_class = target

    print(f"\nAnalysing {Path(video).name} ...")
    res = SkillExtractionPipeline(cfg).run(video)
    if not res.ok:
        print(f"\nFAILED: {res.message}")
        return 1

    df = res.features.df
    ft = res.features
    rep = res.representation
    diag = res.diagnostics["frame_diagnostics"]
    n = len(df)

    print("\n" + "=" * 92)
    print("SIGNAL HEALTH REPORT")
    print("=" * 92)

    problems: list[str] = []

    # ---- 1. pose ------------------------------------------------------- #
    print("\n1. BODY POSE  (drives every velocity, and which arm is analysed)")
    rate = float(diag["pose_detection_rate"])
    vis = float(np.nanmean(df["wrist_visibility"].to_numpy(dtype=float)))
    missing = float(np.mean(np.isnan(df["wrist_x"].to_numpy(dtype=float))))
    status = OK if rate > 0.9 else (WARN if rate > 0.6 else BAD)
    line(status, "person detected", f"{bar(rate)} {rate:6.1%} of frames",
         "" if rate > 0.9 else
         "Get the whole upper body in frame. The pose model needs torso and both "
         "shoulders, not just the arm.")
    if rate <= 0.9:
        problems.append("pose detection")

    status = OK if missing < 0.1 else (WARN if missing < 0.35 else BAD)
    line(status, "wrist position missing", f"{bar(1 - missing)} {missing:6.1%} of frames",
         "" if missing < 0.1 else
         "The wrist leaves frame or is occluded. Widen the shot, or move the "
         "camera so your arm never crosses behind your body.")
    if missing >= 0.1:
        problems.append("wrist visibility")

    status = OK if vis > 0.7 else (WARN if vis > 0.5 else BAD)
    line(status, "wrist confidence", f"{bar(vis)} {vis:6.2f} mean visibility",
         "" if vis > 0.7 else "Improve lighting and avoid a background the same "
         "colour as your sleeve.")
    line(OK, "arm analysed", f"{ft.active_arm}",
         "If that is the wrong arm, the other one moved more. Keep the "
         "non-demonstrating arm still.")

    # ---- 2. motion geometry -------------------------------------------- #
    print("\n2. MOTION GEOMETRY  (is the action visible in the image plane?)")
    wx = df["wrist_x"].to_numpy(dtype=float)
    wy = df["wrist_y"].to_numpy(dtype=float)
    ok = ~(np.isnan(wx) | np.isnan(wy))
    scale = max(ft.body_scale_px, 1e-6)
    if ok.sum() > 2:
        span_x = (np.nanmax(wx[ok]) - np.nanmin(wx[ok])) / scale
        span_y = (np.nanmax(wy[ok]) - np.nanmin(wy[ok])) / scale
        span = max(span_x, span_y)
        aspect = min(span_x, span_y) / max(span_x, span_y, 1e-6)
    else:
        span = span_x = span_y = aspect = 0.0

    status = OK if span > 0.8 else (WARN if span > 0.4 else BAD)
    line(status, "wrist travel", f"{bar(span / 2.5)} {span:5.2f} body-widths",
         "" if span > 0.8 else
         "The hand barely moves across the image. You are probably filming down "
         "the line of motion -- move the camera 90 degrees to the side so the "
         "reach and transport sweep ACROSS frame, not toward the lens.")
    if span <= 0.8:
        problems.append("camera angle (motion toward the lens)")

    status = OK if aspect > 0.25 else WARN
    line(status, "motion spread (h vs v)",
         f"{bar(aspect)} {span_x:4.2f} horiz / {span_y:4.2f} vert",
         "" if aspect > 0.25 else
         "Motion is almost entirely along one axis. LIFT and PLACE are detected "
         "from vertical travel, TRANSPORT from horizontal -- you need both to be "
         "visible. A 45-degree side view gives the best separation.")

    speeds = df["wrist_speed"].to_numpy(dtype=float)
    p90 = float(np.nanpercentile(speeds, 90)) if np.isfinite(speeds).any() else 0.0
    status = OK if 0.5 < p90 < 6.0 else WARN
    line(status, "peak wrist speed", f"{bar(p90 / 8)} {p90:5.2f} body-widths/s",
         "" if 0.5 < p90 < 6.0 else
         ("Motion is very slow; phases may merge into IDLE." if p90 <= 0.5 else
          "Motion is very fast -- short phases like GRASP and PLACE get blurred. "
          "Slow the demonstration down and pause at the grasp and the place."))
    if not 0.5 < p90 < 6.0:
        problems.append("demonstration speed")

    # ---- 3. grip ------------------------------------------------------- #
    print("\n3. GRIP SIGNAL  (the ONLY source of GRASP and RELEASE)")
    if ft.grip_available:
        g = df["grip_ratio"].to_numpy(dtype=float)
        rng = float(np.nanpercentile(g, 95) - np.nanpercentile(g, 5))
        hand_frames = float(np.mean(
            (df["grip_source"] == "hands").to_numpy()))
        line(OK, "grip available", f"{bar(rng)} range {rng:4.2f} (normalised)")
        status = OK if hand_frames > 0.6 else (WARN if hand_frames > 0.25 else BAD)
        line(status, "from 21-landmark model", f"{bar(hand_frames)} {hand_frames:6.1%}",
             "" if hand_frames > 0.6 else
             "Most frames fell back to the coarse pose-based aperture estimate. "
             "Keep the grasping hand facing the camera and unobstructed -- if "
             "your fingers wrap behind the object, the hand model loses them.")
        if hand_frames <= 0.6:
            problems.append("hand visibility during the grasp")
        states = df["hand_state"].value_counts().to_dict()
        line(OK, "hand states observed",
             ", ".join(f"{k} {v}" for k, v in states.items()))
        if "CLOSED" not in states:
            problems.append("hand never registered as closed")
            line(BAD, "no CLOSED frames", "-",
                 "The hand never closed enough to register. Make the grasp "
                 "deliberate: open hand clearly during the reach, close it "
                 "firmly on the object, hold for ~0.5 s.")
    else:
        line(BAD, "grip signal", "UNAVAILABLE",
             "Without this, GRASP, RELEASE, PUSH and PULL are removed from the "
             "label set entirely -- which is very likely why your labels look "
             "wrong. The aperture never varied enough across the clip. Film the "
             "hand opening before the grasp and opening again at the release, "
             "with the palm toward the camera.")
        problems.append("GRIP SIGNAL MISSING (most likely cause of wrong labels)")

    # ---- 4. object ----------------------------------------------------- #
    print("\n4. OBJECT TRACK  (gives contact, and all object-directed skills)")
    obj = rep.get("object", {})
    if ft.object_available:
        det_frac = 1.0 - float(np.mean(np.isnan(df["object_x"].to_numpy(dtype=float))))
        line(OK, "object", f"{obj.get('class')} (track {obj.get('track_id')})")
        status = OK if det_frac > 0.7 else (WARN if det_frac > 0.4 else BAD)
        line(status, "detected in", f"{bar(det_frac)} {det_frac:6.1%} of frames",
             "" if det_frac > 0.7 else
             "Intermittent detection. Use a larger object, better lighting, or a "
             "plainer background; or pass --object <class> to lower the "
             "detection threshold for it.")
        if det_frac <= 0.7:
            problems.append("object detection dropouts")
        d = df["hand_obj_dist"].to_numpy(dtype=float)
        if np.isfinite(d).any():
            line(OK, "closest hand approach", f"{np.nanmin(d):4.2f} body-widths")
    else:
        line(BAD, "object", "NOT TRACKED",
             obj.get("message", ""))
        line(WARN, "consequence", "all object-directed skills refused",
             "PICK_AND_PLACE, PUSH, TRANSFER etc. cannot be reported without an "
             "object. Use a COCO class (run --list-objects) and pass it with "
             "--object, e.g. --object bottle.")
        problems.append("OBJECT NOT TRACKED")

    # ---- 5. contact ---------------------------------------------------- #
    print("\n5. CONTACT  (separates REACH/APPROACH from LIFT/TRANSPORT/PLACE)")
    basis = str(df["contact_basis"].iloc[0])
    cfrac = float(np.mean(df["contact"].to_numpy(dtype=bool)))
    status = {"grip_and_proximity": OK, "proximity_and_object_motion": WARN,
              "grip_only": WARN, "unavailable": BAD}.get(basis, WARN)
    line(status, "inferred from", basis.replace("_", " "))
    status2 = OK if 0.1 < cfrac < 0.85 else WARN
    line(status2, "in contact", f"{bar(cfrac)} {cfrac:6.1%} of the clip",
         "" if 0.1 < cfrac < 0.85 else
         ("Contact is never established -- the grasp is not being seen."
          if cfrac <= 0.1 else
          "Contact for almost the whole clip. Start and end with the hand away "
          "from the object, and hold still for a second at each end."))

    # ---- result -------------------------------------------------------- #
    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    task = rep.get("task", {})
    print(f"  task     : {task.get('name')}  (confidence {task.get('confidence')})")
    print(f"  phases   : {' -> '.join(s.label for s in res.segments)}")
    print(f"  segments : {len(res.segments)}")

    print("\n" + "=" * 92)
    if problems:
        print("FIX THESE, IN ORDER")
        print("=" * 92)
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    else:
        print("All signals healthy. If labels are still wrong, the issue is the")
        print("demonstration structure rather than the filming -- make each phase")
        print("distinct, with a brief pause between them.")
    print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="path to the demonstration video")
    ap.add_argument("--object", default=None,
                    help="the COCO class being manipulated, e.g. bottle")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--list-objects", action="store_true",
                    help="print the detectable object classes and exit")
    args = ap.parse_args(argv)

    if args.list_objects:
        return list_objects()
    if not args.video:
        ap.error("a video path is required (or use --list-objects)")
    return diagnose(args.video, args.object, args.stride)


if __name__ == "__main__":
    raise SystemExit(main())
