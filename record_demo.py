"""Record a demonstration clip from a webcam, for testing the pipeline.

    python record_demo.py                       # 12 s from the default camera
    python record_demo.py --seconds 8 --out demos/pick1.mp4
    python record_demo.py --camera 1 --list     # find the right camera index

Press 'q' to stop early, Esc to abort without saving.

Framing that gives the pipeline a fair chance
---------------------------------------------
* Film from the side or at ~45 degrees, not straight down the arm -- the wrist
  has to travel across the image, not towards the lens.
* Keep the shoulder, elbow, wrist AND the object in frame for the whole clip.
* Use an object YOLOv8's COCO vocabulary knows: bottle, cup, bowl, book,
  cell phone, mouse, remote, scissors, banana, apple, orange, sports ball,
  teddy bear, vase.
* Hold still for ~1 s at the start and end, and pause briefly at the grasp and
  at the place. Those pauses are real signal -- the speed minima they create are
  part of what the segmenter keys on.
* Keep the grasping hand visible; if your fingers are hidden behind the object
  the aperture signal degrades and the system will say so.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def list_cameras(limit: int = 5) -> None:
    print("Probing camera indices...")
    found = False
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {i}: available ({w}x{h})")
            found = True
        cap.release()
    if not found:
        print("  no cameras responded")


def record(out: Path, seconds: float, camera: int, fps: float, countdown: int) -> int:
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {camera}. Try --list to see what is available.",
              file=sys.stderr)
        return 1

    ok, frame = cap.read()
    if not ok or frame is None:
        print("Camera opened but returned no frames.", file=sys.stderr)
        cap.release()
        return 1
    h, w = frame.shape[:2]

    # Countdown so you are not recording yourself reaching for the keyboard.
    t_end = time.time() + countdown
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            break
        left = int(t_end - time.time()) + 1
        cv2.putText(frame, str(left), (w // 2 - 30, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    3.0, (40, 220, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, "get in frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("record_demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Aborted.")
            return 130

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for tag in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        print("No usable mp4 encoder in this OpenCV build.", file=sys.stderr)
        cap.release()
        return 1

    print(f"Recording {seconds:.0f}s to {out} -- press 'q' to stop early.")
    start = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        n += 1
        elapsed = time.time() - start
        disp = frame.copy()
        cv2.putText(disp, f"REC {elapsed:4.1f}s / {seconds:.0f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2, cv2.LINE_AA)
        cv2.imshow("record_demo", disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q") or elapsed >= seconds:
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if n < 5:
        print("Too few frames captured; nothing useful was saved.", file=sys.stderr)
        return 1
    print(f"Saved {n} frames to {out}")
    print(f"\nNow run:\n    py -3.11 cli.py \"{out}\"")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demos/demo.mp4")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--list", action="store_true", help="probe camera indices and exit")
    args = p.parse_args(argv)

    if args.list:
        list_cameras()
        return 0
    return record(Path(args.out), args.seconds, args.camera, args.fps, args.countdown)


if __name__ == "__main__":
    raise SystemExit(main())
