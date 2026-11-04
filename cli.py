"""Headless runner: same pipeline, no browser.

    python cli.py path/to/demo.mp4
    python cli.py demo.mp4 --stride 2 --no-video --out outputs/run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import SkillExtractionPipeline
from utils.config import Config
from utils.signal_utils import format_timestamp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract manipulation skills from a video.")
    p.add_argument("video", help="path to .mp4/.avi/.mov")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=4000)
    p.add_argument("--long-edge", type=int, default=960)
    p.add_argument("--no-objects", action="store_true", help="skip YOLO detection")
    p.add_argument("--object", default=None,
                   help="the COCO class being manipulated, e.g. bottle. Restricts "
                        "detection to it and trusts it as the manipuland.")
    p.add_argument("--no-video", action="store_true", help="skip annotated render")
    p.add_argument("--segmenter", default="changepoint_event",
                   choices=["changepoint_event", "uniform"])
    p.add_argument("--penalty", type=float, default=None, help="PELT penalty scale")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config()
    cfg.video.frame_stride = args.stride
    cfg.video.max_frames = args.max_frames
    cfg.video.target_long_edge = args.long_edge
    cfg.objects.enabled = not args.no_objects
    cfg.objects.target_class = args.object
    cfg.viz.write_video = not args.no_video
    cfg.segmentation.method = args.segmenter
    if args.penalty is not None:
        cfg.segmentation.penalty_scale = args.penalty
    if args.out:
        cfg.output_dir = Path(args.out)

    def progress(frac: float, msg: str) -> None:
        if not args.quiet:
            print(f"[{frac * 100:5.1f}%] {msg}", flush=True)

    res = SkillExtractionPipeline(cfg).run(args.video, progress=progress)

    if not res.ok:
        print(f"\nFAILED: {res.message}", file=sys.stderr)
        for n in res.notes:
            print(n, file=sys.stderr)
        return 1

    task = res.representation.get("task", {})
    print("\n" + "=" * 62)
    print(f"TASK: {task.get('name')}   confidence {task.get('confidence')}")
    print("=" * 62)
    for s in res.segments:
        print(
            f"  {s.id:>2}. {s.label:<10} "
            f"{format_timestamp(s.start_time)} -> {format_timestamp(s.end_time)}"
            f"  ({s.duration:5.2f}s, conf {s.confidence:.2f})"
        )
    print("-" * 62)
    print(f"  rationale: {task.get('rationale')}")
    if res.notes:
        print("\n  notes:")
        for n in res.notes:
            print(f"   - {n}")
    print("\n  outputs:")
    for k, v in res.output_files.items():
        print(f"   - {k}: {v}")
    for k, v in res.plot_files.items():
        print(f"   - plot/{k}: {v}")
    print(f"\n  timings: {json.dumps(res.timings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
