# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

Skill extraction from human demonstration video — Phase 1 of a
Learning-from-Demonstration (LfD) system. An RGB video of a person doing a
tabletop manipulation task goes in; a structured skill representation
(`skill_segments.json`) comes out, ready for a downstream LfD model.

**Scope is perception only.** No robot control, no IK, no simulation, no
hardware, no SNN state estimation. Do not add them. What is still missing before
robot execution is stated in code, in
`lfd_interface.blocking_requirements_for_robot_execution` of the output JSON
(metric 3D lifting, camera-to-robot calibration, gripper aperture calibration).

`README.md` is the user-facing reference and is unusually complete — read the
relevant section before changing a stage. `PROJECT_REPORT.md` holds the measured
results and ablations; `outputs/metrics.json` is its raw data.

## Running things

**Always use Python 3.11 explicitly: `py -3.11`.** The bare `py`/`python` on this
machine resolves to 3.14, where the pose stage cannot run (that mediapipe build
has dropped `mp.solutions` and its `tasks` backend raises
`AttributeError: function 'free' not found`). Every other module imports fine on
3.14, so a failure localised to `pose/` is usually the wrong interpreter.

```bash
py -3.11 -m pip install -r requirements.txt

py -3.11 app.py                        # Streamlit dashboard (re-execs itself)
py -3.11 cli.py demo.mp4               # headless, same pipeline
py -3.11 cli.py demo.mp4 --no-video --stride 2 --out outputs/run1
py -3.11 cli.py demo.mp4 --object bottle   # name the manipuland (COCO class)

py -3.11 diagnose.py demo.mp4          # which signal is breaking the labels
py -3.11 diagnose.py --list-objects    # COCO classes the detector knows
py -3.11 record_demo.py                # record a webcam clip

py -3.11 tests/test_analysis.py        # validation suite (no pytest needed)
py -3.11 tests/benchmark.py            # full metric suite -> outputs/metrics.json
py -3.11 tests/test_app_smoke.py       # dashboard renders
```

Tests are plain scripts with a `main()`; they also work under pytest. They run
entirely on synthetic ground truth from `tests/synthetic.py`, so they need no
video and no CV models — run them after any change to the analysis stack.

## Architecture

One linear pipeline, orchestrated by `pipeline.py`:

```
video/        decode, resize, coordinate mapping
pose/         MediaPipe Pose (33 lm) + Hands (21 lm), two backends
tracking/     YOLOv8n detection, IoU+centroid tracker, manipuland selection
features/     scale-invariant kinematics -> one tidy DataFrame
segmentation/ PELT change points + interaction events -> Segment[]
segmentation/ phase labelling: soft rules + Viterbi over a transition matrix
skills/       sequence-template task classification, then JSON/CSV emission
visualization/annotated video + 5 PNG figures
```

Perception is a **single decode pass** (pose, hands and detection share each
frame). Rendering needs a second pass because labels are only known after the
whole sequence is analysed.

The only contract between stages is **feature table in, `Segment` objects out**,
so any stage is replaceable without touching the others.

### Extension points

`segmentation/base.py` defines `BaseSegmenter` / `BaseLabeler` plus name
registries; `skills/skill_classifier.py` has the classifier equivalent. Register
a new implementation and select it by key from `utils.config`:

```python
@register_segmenter("my_tcn")
class MyTCNSegmenter(BaseSegmenter):
    def segment(self, df, context):
        return self.boundaries_to_segments(my_model.predict(...), df)
```

Then `SegmentationConfig.method = "my_tcn"`. This is where a learned model
(HMM / TCN / Transformer / LSTM) goes — the Viterbi emission step in
`phase_labeler.py` is the natural drop-in point. Registry modules are imported
for side effects in `pipeline.py` and in each test; keep the
`# noqa: F401  (registry side effects)` imports when adding a module.

A `uniform` fixed-window segmenter exists purely as an ablation baseline
(`test_uniform_segmenter_is_worse` asserts it degrades the labels).

## Conventions that must hold

These are the research result, not style preferences. Each is enforced in code
and covered by a test — do not relax one to make output look better.

- **No hard-coded timestamps.** Every boundary traces back to a feature crossing
  in that particular video. `test_boundaries_follow_the_data` renders the same
  plan at two speeds with the object moved and fails any canned timing.
- **No fabricated 3D.** All coordinates are 2D source-frame pixels tagged
  `coordinate_space: "2D_image"` with `depth_metric_available: false`.
  MediaPipe's `z` is kept as `wrist_z_relative` but used in no decision.
- **Missing means NaN.** A landmark below `min_landmark_visibility` is written
  NaN, never carried forward. Gaps longer than `max_interpolation_gap_s` (0.3 s)
  stay missing. Off-frame landmarks are passed through unclamped — clamping
  would fabricate a position.
- **Refuse rather than invent.** No grip range → grip unavailable and GRASP /
  RELEASE / PUSH / PULL leave the candidate set. No interaction evidence → the
  track is rejected with a recorded reason (image-plane proximity alone is not
  evidence). Nothing over `min_task_confidence` → `task: "UNKNOWN"` with a
  rationale. A template missing a required phase is not a partial match.
  See README §4 for the full table.
- **Graceful degradation, not silent zeros.** Any evidence term may report
  "unavailable"; a label's score is the renormalised weighted mean of what *is*
  available. Never feed a missing channel in as zeros.
- **Scale invariance.** Every velocity is divided by shoulder width in pixels —
  units are body-widths and body-widths/s. New thresholds must be expressed in
  those units so they work across resolutions and camera distances.
- **Every threshold lives in `utils/config.py`.** Do not bury a numeric constant
  in an algorithm. `Config.resolve()` enforces cross-setting relationships (e.g.
  the smoothing window never exceeds `min_segment_duration_s`) and is re-run by
  the pipeline after the dashboard sidebar mutates settings.
- **Evidence is auditable.** Segments carry `evidence`, `alternatives`,
  `boundary_salience` and `boundary_sources`; the pipeline carries `notes` and
  `diagnostics` through to the JSON. Keep new logic explainable the same way.
- **Dependency-light.** PELT is implemented in `segmentation/changepoint.py`
  with numpy only; `ruptures` is deliberately not a dependency, and scipy is an
  optional fast path in `utils/signal_utils.py`. Do not add a library for
  something ~120 lines of numpy already does.

Failures surface as messages, not stack traces into the void: `pipeline.run()`
returns `PipelineResult(ok=False, message=...)`, and the pose extractor reports
both backend attempts when neither works.

## Code style

Python 3.11, `from __future__ import annotations` at the top of every module.
Dataclasses for every structured value (`Config`, `Segment`, `PipelineResult`,
`PoseFrame`, `Track`, `SkillResult`). Type hints throughout. Module docstrings
explain *why* a design choice was made, with a "Design notes" or "Strategy"
section — match that density; comments here justify decisions rather than
restate code. Section banners (`# ---- n. stage ---- #`) mark pipeline steps.
Phase vocabulary and both colour maps are in `utils/config.py`
(`PHASE_LABELS`, `PHASE_COLORS_BGR`, `PHASE_COLORS_HEX`) — add a phase there.

`app.py` carries version shims for Streamlit 1.33 vs 1.53 (both are installed);
keep new widget calls behind the same try-fallback pattern.

## Things not to commit

`.gitignore` covers them: `outputs/*` (regenerate by rerunning), `models/*.task`
(fetched at runtime by the tasks backend), `*.pt` (YOLO weights), `__pycache__`.
