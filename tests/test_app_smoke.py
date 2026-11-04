"""Smoke tests for the Streamlit dashboard, using Streamlit's own test harness.

These execute the real app script in-process and assert it raises nothing --
both on the initial (no upload) screen and, when a video path is supplied, on the
full results screen with plots, tables, segment jump buttons and downloads.

    python tests/test_app_smoke.py [path/to/demo.mp4]

Without a video argument only the initial screen is checked, which needs no
model inference and runs in a couple of seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP = ROOT / "app.py"


def _app_test(timeout: int = 300):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=timeout)


def test_initial_screen_renders():
    at = _app_test(timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Skill Extraction" in t.value for t in at.title)
    # The upload prompt must be the only thing shown before a file is provided.
    assert any("Upload" in i.value for i in at.info)
    print("PASS  initial screen renders, no exceptions")


def test_sidebar_controls_exist():
    at = _app_test(timeout=120)
    at.run()
    n = (
        len(at.sidebar.number_input)
        + len(at.sidebar.checkbox)
        + len(at.sidebar.slider)
        + len(at.sidebar.selectbox)
    )
    assert n >= 10, f"expected the full settings panel, found {n} widgets"
    print(f"PASS  sidebar exposes {n} settings widgets")


def test_results_screen_renders(video: str):
    """Run the pipeline, inject the result, and render the results screen."""
    from pipeline import SkillExtractionPipeline
    from utils.config import Config

    cfg = Config()
    cfg.viz.write_video = False          # plots and tables are what we assert on
    cfg.video.frame_stride = 2           # keep the smoke test quick
    res = SkillExtractionPipeline(cfg).run(video)
    assert res.ok, res.message

    at = _app_test(timeout=300)
    at.session_state["result"] = res
    at.session_state["video_path"] = str(video)
    at.session_state["upload_key"] = "injected"
    at.session_state["seek_to"] = 0
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]

    # One "Jump" button per segment, plus the Process Video button.
    jump = [b for b in at.button if b.label == "Jump"]
    assert len(jump) == len(res.segments), (
        f"{len(jump)} jump buttons for {len(res.segments)} segments"
    )

    # AppTest gained a `download_button` accessor only in later Streamlit
    # versions; `get()` works on both, so the assertion is version independent.
    downloads = list(at.get("download_button"))
    labels = [getattr(d, "label", "") for d in downloads]
    for expected in ("skill_segments.json", "skill_segments.csv", "trajectory.csv"):
        assert any(expected in l for l in labels), (
            f"missing download for {expected}; found {labels}"
        )
    print(
        f"PASS  results screen renders: {len(res.segments)} segments, "
        f"{len(jump)} jump buttons, {len(downloads)} downloads, "
        f"{len(at.tabs)} tab groups"
    )

    # Clicking a segment must move the player without raising.
    jump[min(1, len(jump) - 1)].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["seek_to"] >= 0
    print("PASS  segment jump updates the player position")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    failures = []
    for fn in (test_initial_screen_renders, test_sidebar_controls_exist):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
    if video:
        try:
            test_results_screen_renders(video)
        except Exception as exc:  # noqa: BLE001
            failures.append(("test_results_screen_renders", exc))
            print(f"FAIL  test_results_screen_renders: {exc}")
    else:
        print("SKIP  test_results_screen_renders (pass a video path to enable)")
    raise SystemExit(1 if failures else 0)
