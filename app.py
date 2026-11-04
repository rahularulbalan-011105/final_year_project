"""Streamlit dashboard for LfD skill extraction from human demonstration video.

Run either way:

    streamlit run app.py
    python app.py            # re-execs itself under the Streamlit runtime
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _relaunch_under_streamlit() -> None:
    """Allow `python app.py` to work, as the acceptance test expects."""
    import subprocess

    print("Starting the Streamlit server (equivalent to: streamlit run app.py)\n")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.call(cmd))


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_under_streamlit():
    _relaunch_under_streamlit()


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
import json
import math
import shutil
import tempfile

import pandas as pd
import streamlit as st

from pipeline import PipelineResult, SkillExtractionPipeline
from utils.config import PHASE_COLORS_HEX, Config
from utils.signal_utils import format_timestamp
from video.video_loader import VideoLoadError, plan_processing, probe

st.set_page_config(
    page_title="LfD Skill Extraction",
    page_icon=":robot_face:",
    layout="wide",
)

SESSION_DIR = Path(tempfile.gettempdir()) / "lfd_skill_extraction"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# Streamlit renamed/removed several width arguments between 1.33 and 1.53, and
# both versions are present on this machine. These shims try the modern spelling
# first and fall back, so the dashboard runs on either.
def _image(path, caption: str | None = None) -> None:
    for kwargs in ({"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            st.image(str(path), caption=caption, **kwargs)
            return
        except TypeError:
            continue
        except Exception as exc:
            st.warning(f"Could not display {Path(path).name}: {exc}")
            return


def _rerun() -> None:
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn is not None:
        fn()


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("video_path", None)
    st.session_state.setdefault("seek_to", 0)
    st.session_state.setdefault("upload_key", None)


def _save_upload(uploaded) -> Path:
    dest = SESSION_DIR / f"upload_{uploaded.name}"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(uploaded, fh)
    return dest


def _phase_chip(label: str) -> str:
    colour = PHASE_COLORS_HEX.get(label, "#666666")
    return (
        f"<span style='background:{colour};color:#111;padding:2px 8px;"
        f"border-radius:10px;font-size:0.76rem;font-weight:700;"
        f"letter-spacing:.02em'>{label}</span>"
    )


def _timeline_html(segments, total: float) -> str:
    if total <= 0:
        return ""
    bars = []
    for s in segments:
        pct = 100.0 * s.duration / total
        colour = PHASE_COLORS_HEX.get(s.label, "#666")
        tip = (
            f"{s.id}. {s.label}  {format_timestamp(s.start_time)}"
            f"-{format_timestamp(s.end_time)}  conf {s.confidence:.2f}"
        )
        bars.append(
            f"<div title='{tip}' style='flex:0 0 {pct:.4f}%;background:{colour};"
            f"height:30px;border-right:1px solid #fff;overflow:hidden;"
            f"font-size:0.6rem;color:#111;text-align:center;line-height:30px;"
            f"font-weight:700'>{s.label if pct > 7 else ''}</div>"
        )
    return (
        "<div style='display:flex;width:100%;border-radius:5px;overflow:hidden;"
        "border:1px solid #d8d8d8'>" + "".join(bars) + "</div>"
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_config() -> Config:
    cfg = Config()
    st.sidebar.title("Pipeline settings")
    st.sidebar.caption(
        "These control detector sensitivity only. No phase timing is configured "
        "anywhere -- boundaries come from the video."
    )

    with st.sidebar.expander("Video processing", expanded=True):
        cfg.video.frame_stride = st.number_input(
            "Frame stride", 1, 10, cfg.video.frame_stride, 1,
            help="Process every Nth frame. 1 gives the best temporal resolution.",
        )
        cfg.video.target_long_edge = st.select_slider(
            "Inference resolution (long edge, px)",
            options=[480, 640, 768, 960, 1280],
            value=960,
        )
        cfg.video.max_frames = st.number_input(
            "Max frames to analyse", 60, 20000, cfg.video.max_frames, 100
        )

    with st.sidebar.expander("Object tracking", expanded=False):
        cfg.objects.enabled = st.checkbox(
            "Enable YOLO object detection", cfg.objects.enabled
        )
        cfg.objects.target_class = st.text_input(
            "Object being manipulated (optional)",
            value="",
            placeholder="e.g. bottle, cup, book",
            help="Naming the object restricts detection to that class and trusts "
                 "it as the manipuland, which removes the largest source of "
                 "wrong labels. Must be one of the 80 COCO classes.",
        ).strip() or None
        cfg.objects.det_confidence = st.slider(
            "Detection confidence", 0.05, 0.9, cfg.objects.det_confidence, 0.05
        )
        cfg.objects.detect_every = st.number_input(
            "Detect every Nth analysed frame", 1, 10, cfg.objects.detect_every, 1
        )
        cfg.objects.min_manipuland_score = st.slider(
            "Min manipuland score", 0.0, 0.9, cfg.objects.min_manipuland_score, 0.05,
            help="Below this the object is reported as unreliable instead of used.",
        )

    with st.sidebar.expander("Segmentation", expanded=False):
        cfg.segmentation.method = st.selectbox(
            "Segmenter", ["changepoint_event", "uniform"], 0,
            help="'uniform' is a fixed-window ablation baseline for comparison.",
        )
        cfg.segmentation.penalty_scale = st.slider(
            "PELT penalty scale", 0.4, 4.0, cfg.segmentation.penalty_scale, 0.1,
            help="Higher = fewer change points.",
        )
        cfg.segmentation.min_segment_duration_s = st.slider(
            "Min segment duration (s)", 0.05, 1.0,
            cfg.segmentation.min_segment_duration_s, 0.01,
        )
        cfg.segmentation.use_event_boundaries = st.checkbox(
            "Use interaction-event boundaries", cfg.segmentation.use_event_boundaries
        )

    with st.sidebar.expander("Confidence floors", expanded=False):
        cfg.skills.min_phase_confidence = st.slider(
            "Min phase confidence", 0.0, 0.9, cfg.skills.min_phase_confidence, 0.05
        )
        cfg.skills.min_task_confidence = st.slider(
            "Min task confidence", 0.0, 0.9, cfg.skills.min_task_confidence, 0.05
        )

    cfg.viz.write_video = st.sidebar.checkbox(
        "Render annotated video", True,
        help="The slowest stage. Disable for a faster numeric-only run.",
    )
    return cfg


# --------------------------------------------------------------------------- #
# Result panels
# --------------------------------------------------------------------------- #
def render_results(res: PipelineResult, source_video: Path) -> None:
    rep = res.representation
    segments = res.segments
    total = float(res.video_meta.get("duration_s") or 0.0) or 1e-6

    st.markdown("### Result")
    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        video_path = res.processed_video or source_video
        is_processed = res.processed_video is not None
        st.caption(
            "Processed video with pose, object, phase and timeline overlays"
            if is_processed else "Original upload (annotated render unavailable)"
        )
        try:
            st.video(
                video_path.read_bytes(),
                start_time=int(st.session_state.get("seek_to", 0)),
            )
        except Exception as exc:
            st.warning(f"The browser could not play the rendered video ({exc}).")
        if is_processed:
            st.download_button(
                "Download processed_video.mp4",
                data=video_path.read_bytes(),
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

    with right:
        task = rep.get("task", {})
        name = task.get("name", "UNKNOWN")
        conf = float(task.get("confidence") or 0.0)
        st.markdown("**Detected task**")
        if name == "UNKNOWN":
            st.warning(f"UNKNOWN  (best template score {conf:.2f})")
        else:
            st.success(f"{name}   —   confidence {conf:.2f}")
        st.caption(task.get("rationale", ""))

        st.markdown("**Segments**  \n"
                    "<span style='font-size:.78rem;color:#777'>Click a segment to "
                    "jump the player to it (whole-second precision).</span>",
                    unsafe_allow_html=True)

        for s in segments:
            c1, c2 = st.columns([3.1, 1.0])
            with c1:
                st.markdown(
                    f"{s.id}. {_phase_chip(s.label)} "
                    f"<code>{format_timestamp(s.start_time)} - "
                    f"{format_timestamp(s.end_time)}</code> "
                    f"<span style='color:#888;font-size:.78rem'>"
                    f"{s.duration:.2f}s · conf {s.confidence:.2f}</span>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("Jump", key=f"jump_{s.id}", use_container_width=True):
                    st.session_state["seek_to"] = int(math.floor(s.start_time))
                    _rerun()

    st.markdown("---")
    st.markdown("**Segmentation timeline**")
    st.markdown(_timeline_html(segments, total), unsafe_allow_html=True)

    if res.notes:
        with st.expander(f"Reliability notes ({len(res.notes)})", expanded=False):
            for n in res.notes:
                st.markdown(f"- {n}")

    tabs = st.tabs([
        "Plots", "Segments table", "Trajectory data", "Evidence",
        "JSON", "Downloads", "Diagnostics",
    ])

    with tabs[0]:
        pf = res.plot_files
        c1, c2 = st.columns(2)
        with c1:
            if "wrist_trajectory" in pf:
                _image(pf["wrist_trajectory"])
        with c2:
            if "object_trajectory" in pf:
                _image(pf["object_trajectory"])
            else:
                st.info(
                    "No object trajectory plot: no object track passed the "
                    "reliability threshold for this video."
                )
        if "velocity" in pf:
            _image(pf["velocity"])
        if "interaction" in pf:
            _image(pf["interaction"])
        if "timeline" in pf:
            _image(pf["timeline"])

    with tabs[1]:
        seg_csv = res.output_files.get("segments_csv")
        if seg_csv and Path(seg_csv).exists():
            st.dataframe(pd.read_csv(seg_csv), use_container_width=True, height=330)

    with tabs[2]:
        traj_csv = res.output_files.get("trajectory_csv")
        if traj_csv and Path(traj_csv).exists():
            tdf = pd.read_csv(traj_csv)
            st.caption(
                f"{len(tdf)} samples · positions in source-frame pixels (2D image "
                f"plane) · velocities in body-widths/s"
            )
            st.dataframe(tdf.head(400), use_container_width=True, height=330)

    with tabs[3]:
        st.caption(
            "Why each phase was chosen. Terms are soft scores in [0,1]; "
            "`null` means the evidence was unavailable for this video."
        )
        for s in segments:
            with st.expander(
                f"Segment {s.id}: {s.label} "
                f"({format_timestamp(s.start_time)}-{format_timestamp(s.end_time)})"
            ):
                st.json(s.evidence, expanded=False)
                st.markdown("**Segment motion features**")
                st.json(s.motion_features, expanded=False)

    with tabs[4]:
        st.json(rep, expanded=False)

    with tabs[5]:
        for key, label, mime in (
            ("json", "skill_segments.json", "application/json"),
            ("segments_csv", "skill_segments.csv", "text/csv"),
            ("trajectory_csv", "trajectory.csv", "text/csv"),
        ):
            p = res.output_files.get(key)
            if p and Path(p).exists():
                st.download_button(
                    f"Download {label}",
                    data=Path(p).read_bytes(),
                    file_name=label,
                    mime=mime,
                    key=f"dl_{key}",
                )
        st.caption(f"All outputs were also written to `{Config().output_dir}`.")

    with tabs[6]:
        st.json(res.diagnostics, expanded=False)
        st.json(res.timings, expanded=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    st.title("Learning from Demonstration — Skill Extraction from RGB Video")
    st.caption(
        "Video → pose/hand tracking → object tracking → motion features → "
        "temporal segmentation → skill identification → structured "
        "representation. Phase boundaries are derived from the observed motion; "
        "nothing is hard-coded."
    )

    cfg = sidebar_config()

    uploaded = st.file_uploader(
        "Upload a demonstration video",
        type=["mp4", "avi", "mov"],
        help="A single-person manipulation demonstration, e.g. pick-and-place.",
    )

    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state["upload_key"] != key:
            st.session_state["upload_key"] = key
            st.session_state["video_path"] = str(_save_upload(uploaded))
            st.session_state["result"] = None
            st.session_state["seek_to"] = 0

    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.info("Upload an .mp4, .avi or .mov file to begin.")
        return

    video_path = Path(video_path)

    # ---- metadata ------------------------------------------------------- #
    try:
        meta = probe(video_path)
    except VideoLoadError as exc:
        st.error(str(exc))
        return

    st.markdown("### Video")
    c1, c2 = st.columns([1.0, 1.4], gap="large")
    with c1:
        st.video(video_path.read_bytes())
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("FPS", f"{meta.fps:.2f}")
        m2.metric("Frames", f"{meta.n_frames}")
        m3.metric("Duration", f"{meta.duration_s:.2f} s")
        m4, m5 = st.columns(2)
        m4.metric("Resolution", meta.resolution)
        m5.metric("File", video_path.name if len(video_path.name) < 22
                  else video_path.name[:19] + "...")
        if meta.frame_count_is_estimate:
            st.warning(
                "This container did not report a reliable frame rate or frame "
                "count; timestamps are estimated."
            )
        plan = plan_processing(meta, cfg.video.frame_stride, cfg.video.max_frames)
        st.caption(
            f"Will analyse {plan['frames_to_process']} frames at "
            f"{plan['effective_sample_rate_hz']} Hz, covering "
            f"{plan['covered_duration_s']:.2f}s"
            + (" (clip is truncated by the frame cap)" if plan["truncated"] else "")
        )
        run = st.button("Process Video", type="primary", use_container_width=True)

    if run:
        bar = st.progress(0.0)
        status = st.empty()

        def on_progress(frac: float, msg: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0))
            status.info(msg)

        pipeline = SkillExtractionPipeline(cfg)
        with st.spinner("Analysing the demonstration..."):
            res = pipeline.run(video_path, progress=on_progress)
        bar.empty()
        status.empty()
        st.session_state["result"] = res
        st.session_state["seek_to"] = 0

    res = st.session_state.get("result")
    if res is None:
        return
    if not res.ok:
        st.error(res.message)
        for n in res.notes:
            st.code(n)
        return

    render_results(res, video_path)


main()
