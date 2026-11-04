# Skill Extraction from Human Demonstration Video (LfD — Phase 1)

Upload an RGB video of a person performing a manipulation task. The system
extracts the human motion, tracks the manipulated object, segments the
demonstration into temporal phases, identifies the overall skill, and emits a
structured representation for a downstream Learning-from-Demonstration model.

**Scope of this phase.** Perception and skill extraction only. No robot control,
no inverse kinematics, no simulation, no hardware, no SNN state estimation. The
pipeline stops at the extracted skill representation, and
`lfd_interface.blocking_requirements_for_robot_execution` in the output JSON
states exactly what is still missing before any of that becomes possible.

```
VIDEO
  → preprocessing (video/)
  → pose + hand extraction (pose/)
  → object detection + tracking (tracking/)
  → motion features (features/)
  → temporal segmentation (segmentation/)
  → phase labelling (segmentation/)
  → skill identification (skills/)
  → structured representation (skills/)
  → JSON + CSV + annotated video + plots (visualization/)
```

---

## Quick start

```bash
py -3.11 -m pip install -r requirements.txt

py -3.11 app.py                    # launches the Streamlit dashboard
# or
py -3.11 -m streamlit run app.py

py -3.11 cli.py demo.mp4           # headless, same pipeline
py -3.11 cli.py demo.mp4 --no-video --stride 2 --out outputs/run1

py -3.11 record_demo.py            # record a test clip from your webcam
py -3.11 diagnose.py demo.mp4      # which signal is breaking your labels?
py -3.11 diagnose.py --list-objects   # object classes the detector knows
py -3.11 cli.py demo.mp4 --object bottle   # name the object you are manipulating

py -3.11 tests/benchmark.py                # full metric suite -> outputs/metrics.json
py -3.11 tests/test_analysis.py            # analysis validation (synthetic truth)
py -3.11 tests/test_app_smoke.py           # dashboard renders without error
py -3.11 tests/test_app_smoke.py demo.mp4  # ... including the results screen
```

### Which Python

**Use Python 3.11.** Verified on this machine with `mediapipe 0.10.21`,
`opencv 4.11`, `ultralytics 8.3.228`, `streamlit 1.33` — everything runs offline
with no model downloads.

```
py -3.11 app.py          # the py launcher selects 3.11 for you
```

**Python 3.14 does not work for the perception stage.** The `mediapipe 0.10.30`
build installed there has dropped the legacy `mp.solutions` API *and* its
`mediapipe.tasks` API raises `AttributeError: function 'free' not found` on 3.14.
The pose extractor tries both backends and reports the failure explicitly rather
than half-starting:

```
RuntimeError: No usable MediaPipe backend. Attempts ->
  solutions: this mediapipe build has no 'solutions' module |
  tasks: AttributeError: function 'free' not found
```

Every other module imports cleanly on 3.14, so only the pose stage is blocked.
The two-backend design means that once a 3.14-compatible mediapipe ships, the
`tasks` backend will take over automatically (it downloads `.task` bundles into
`models/` on first use).

`ffmpeg` on PATH is optional but recommended — the annotated video is remuxed to
baseline H.264 / yuv420p so browsers can play it inline.

---

## 1. Models, and why

| Stage | Model | Why this one |
|---|---|---|
| Body pose | **MediaPipe Pose (BlazePose)**, 33 landmarks | Real-time on CPU, per-landmark visibility scores that serve as honest confidence, and the de-facto baseline in video-LfD work. Gives shoulder / elbow / wrist directly. |
| Hand | **MediaPipe Hands**, 21 landmarks per hand | The only cheap way to get an aperture signal, which is what makes GRASP and RELEASE detectable at all. |
| Object | **YOLOv8n** (COCO-80) via `ultralytics` | Small, fast, no training needed, and the weights are commonly already cached so the pipeline runs offline. Open-vocabulary detection is the obvious upgrade path. |
| Object tracking | **IoU + centroid tracker** (in-repo) | ~120 lines, no extra dependency, sufficient for a short tabletop demonstration. Swappable for ByteTrack/BoT-SORT. |
| Segmentation | **PELT change-point detection** (L2 cost, in-repo) + interaction-event detection + **Viterbi** decode | `ruptures` is not required: the closed form for a piecewise-constant-mean cost is short enough to implement directly, which keeps the dependency list to what is already installed. |

### On 3D

Monocular RGB gives **no metric depth**. Every coordinate this system produces is
2D image-plane pixels, tagged `coordinate_space: "2D_image"` with
`depth_metric_available: false`. MediaPipe's `z` channel is retained per frame as
`wrist_z_relative` but it is unitless, uncalibrated, and **not used in any
decision**. No 3D coordinate is fabricated anywhere.

### Scale invariance

Every velocity is divided by the subject's shoulder width in pixels, so the units
are **body-widths per second** and distances are **body-widths**. The same
thresholds therefore work for a 480p clip filmed from 3 m away and a 4K clip
filmed from 1 m away, without per-video tuning.

---

## 2. Module contracts

| Module | Input | Output |
|---|---|---|
| `video/video_loader.py` | file path | `VideoMeta(fps, n_frames, duration, w, h)` + frame iterator `(index, timestamp_s, bgr)` |
| `video/preprocessing.py` | BGR frame | resized BGR + RGB, plus `FrameGeometry` so all coordinates map back to source pixels |
| `pose/pose_extractor.py` | frames | `PoseSequence` — per frame: shoulder/elbow/wrist/hand keypoints (2D px), visibility, raw hand aperture; plus the automatically chosen manipulating arm |
| `tracking/object_tracker.py` | frames + hand path | `Track[]`, then a `ManipulandResult` naming the manipulated object **or declining to** |
| `features/motion_features.py` | `PoseSequence` + `Track` | one tidy `DataFrame`, one row per processed frame, all kinematics scale-normalised |
| `segmentation/temporal_segmenter.py` | feature `DataFrame` | `Segment[]` — boundaries only, no labels |
| `segmentation/phase_labeler.py` | `Segment[]` + features | same segments with `label`, `confidence`, and per-term `evidence` |
| `skills/skill_classifier.py` | labelled segments | `SkillResult(task, confidence, rationale, candidates)` |
| `skills/skill_representation.py` | everything above | `skill_segments.json`, `skill_segments.csv`, `trajectory.csv` |
| `visualization/` | everything above | `processed_video.mp4` + 5 PNG figures |

Because the only contract between stages is *feature table in, `Segment` objects
out*, any stage can be replaced without touching the others.

---

## 3. How temporal segmentation works

No timestamp is hard-coded anywhere. Three stages:

**(a) Feature matrix.** Columns are z-scored: wrist speed, acceleration
magnitude, heading-change rate, vertical velocity, hand–object distance, object
speed, grip aperture. Channels that are entirely missing (`obj_speed` with no
object) or constant are dropped rather than fed in as zeros.

**(b) Boundary proposal** — the union of two independent sources:

- *Statistical:* PELT over the feature matrix with penalty `β = k·d·log n`. Finds
  where the regime of the motion changes, knowing nothing about manipulation.
- *Semantic:* discrete interaction events — the **interval** over which the grip
  aperture changes (both edges, not just the threshold crossing), contact
  start/end, object motion onset/offset, wrist-speed minima, prominent
  hand–object distance minima, and vertical direction reversals.

Change points alone miss GRASP and RELEASE, whose statistical footprint is small
but whose semantic importance is large. Events alone are brittle when the grip
signal is weak. Nearby proposals are then clustered (cluster width is capped at
the tolerance, measured from the cluster's first member — chaining from the last
member collapses an entire transition into one boundary), and segments shorter
than the minimum are absorbed.

**(c) Labelling.** Per-segment aggregate features feed soft membership functions,
one scoring function per phase. Scores are decoded with **Viterbi** over an
allowed-transition matrix, so the sequence obeys a manipulation grammar — you
cannot RELEASE before you GRASP. Adjacent segments that decode to the same label
are then merged and re-scored *keeping* their labels: re-deciding on the merged
span systematically favours the coarser label and loses real sub-phases.

Confidence for each segment is the local posterior **given its decoded
neighbours**, scaled by a global quality factor derived from the pose detection
rate and from which evidence channels were available.

### Swapping in a learned model

```python
from segmentation.base import BaseSegmenter, register_segmenter

@register_segmenter("my_tcn")
class MyTCNSegmenter(BaseSegmenter):
    def segment(self, df, context):
        boundaries = my_model.predict(df[FEATURE_COLS].to_numpy())
        return self.boundaries_to_segments(boundaries, df)
```

Then set `SegmentationConfig.method = "my_tcn"`. The same applies to
`BaseLabeler` (HMM / TCN / Transformer / LSTM emissions) and
`BaseSkillClassifier`. A `uniform` fixed-window segmenter is included as an
ablation baseline: switch to it and the phase labels measurably degrade, which is
the evidence that the change-point stage is doing real work.

---

## 4. What the system refuses to claim

This is the part that matters for a research result. Every one of these is
enforced in code and covered by a test.

| Situation | Behaviour |
|---|---|
| Landmark below the visibility threshold | Written as `NaN`, never carried forward. Gaps longer than 0.3 s are left missing rather than interpolated. |
| Hand aperture varies by less than a threshold over the clip | Grip is reported **unavailable**; `hand_state` is `UNKNOWN` throughout. |
| Object never moves and never co-moves with the hand | Track rejected for lack of interaction evidence, with the reason recorded. Image-plane proximity alone is not evidence — a background object can project straight onto the hand. |
| No reliable object track | `"Object tracking confidence insufficient: ..."`, and every object-directed skill (PICK_AND_PLACE, PUSH, TRANSFER, …) becomes ineligible. |
| No grip signal | GRASP, RELEASE, PUSH and PULL are removed from the candidate set — they are claims about the hand aperture. Object motion shows an interaction happened, not that fingers closed. |
| Neither grip nor object | No contact can be established; only IDLE / REACH / RETRACT / UNKNOWN are admissible. |
| A template's required phase is missing | Not a match. A partial match is rejected outright, so a GRASP plus a RELEASE with no TRANSPORT cannot be reported as a TRANSFER. |
| Nothing clears the confidence floor | `task: "UNKNOWN"` with the best score and a rationale naming the observed phase sequence. |
| No human detected | The run fails with an explicit message instead of returning empty segments. |
| Video exceeds the frame cap | Reported in `notes`, with the covered duration stated. |

---

## 5. Outputs

Written to `outputs/` (or `--out`):

| File | Contents |
|---|---|
| `skill_segments.json` | Complete representation: task, per-segment phases with trajectories and motion features, the evidence behind every label, the `lfd_interface` block, and full diagnostics. |
| `skill_segments.csv` | `segment_id, skill, start_time, end_time, duration, confidence` plus per-segment kinematics and the gripper command. |
| `trajectory.csv` | Per frame: `timestamp, wrist_x/y, elbow_x/y, shoulder_x/y, object_x/y, hand_state`, plus grip ratio, speeds, hand–object distance, contact flag and segment id. |
| `processed_video.mp4` | Skeleton, hand, object box, wrist trail, HUD (task / phase / timestamp / hand state) and a colour-coded timeline with a playhead. |
| `*.png` | Wrist trajectory, object trajectory, velocity with boundaries overlaid, grip + contact, and the segment timeline. |

### The LfD hand-off

Each segment carries an `lfd` block:

- `phase_normalized_time` — τ ∈ [0, 1], the phase variable a movement primitive
  is fitted against
- `hand_relative_to_object_bodywidths` — the **object-centric** hand vector. This
  is the frame to fit a DMP/ProMP in, because it is what transfers when the
  object is somewhere else next time.
- `gripper_command` — `CLOSE` / `OPEN` / `HOLD_CLOSED` / `HOLD_OPEN` / `UNKNOWN`
- `start_waypoint` / `end_waypoint`

And at the top level, `lfd_interface` gives the symbolic plan (the phase
sequence), the gripper event times, and the three blocking requirements before
robot execution: metric 3D lifting, camera-to-robot calibration, and gripper
aperture calibration. The per-frame `segment_id` in `trajectory.csv` is the
frame-level phase supervision for a later state/phase estimator.

---

## 6. Validation and measured results

**Quantitative results, ablations, robustness curves and runtime are in
[PROJECT_REPORT.md](PROJECT_REPORT.md)** (raw data: `outputs/metrics.json`,
regenerate with `py -3.11 tests/benchmark.py`). Headline: over 105 synthetic
demonstrations with known ground truth — MoF 78.8%, F1@50 79.0, boundary MAE
103 ms, **100% family-level task accuracy with 0% false skills**. Removing the
interaction-event boundaries costs 93 percentage points of task accuracy.

### Correctness tests

`tests/synthetic.py` renders a **parameterised** ground-truth phase plan into pose
frames and an object track, so the whole analysis stack below the CV models can be
measured against known boundaries. `tests/test_analysis.py` asserts:

- `test_pick_and_place_is_recovered` — phase order and boundary error
- `test_boundaries_follow_the_data` — the same plan at two speeds with the object
  moved; the recovered grasp onset must track the truth in both and must differ
  substantially between them. **A pipeline returning canned timestamps fails
  this test.**
- `test_push_is_not_labelled_pick_and_place` — no GRASP on a non-prehensile push
- `test_no_object_degrades_honestly`, `test_no_evidence_refuses_contact_phases` —
  the refusals above
- `test_static_video_is_idle_not_a_skill` — a static clip yields no skill
- `test_uniform_segmenter_is_worse` — the ablation comparison
- `test_pelt_recovers_a_known_step` — unit test of the change-point detector

Current result on the canonical pick-and-place: the exact phase sequence
`REACH → APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETRACT`
recovered, task `PICK_AND_PLACE` at 0.99, **89.7 % frame accuracy**, boundary MAE
81 ms (all boundaries within 100 ms except the inherently soft REACH/APPROACH
deceleration boundary).

---

## 7. Known limitations

- **REACH vs APPROACH** is a soft distinction (onset of deceleration near the
  object); it is the least reliable boundary.
- **Grip from a monocular hand model** is a normalised, unitless aperture ratio,
  not a finger separation. Self-occlusion when the hand wraps an object degrades
  it, which is why the per-video range check exists.
- **YOLOv8n is limited to the COCO-80 vocabulary.** An unusual manipuland will
  simply not be detected, and the system will say so rather than invent one.
- **Single-arm, single-object.** The manipulating arm is chosen by observed path
  length; bimanual tasks are not modelled.
- **Off-frame landmarks.** BlazePose extrapolates joints that leave the frame, so
  a coordinate can legitimately be negative or beyond the image width. These are
  passed through rather than clamped — clamping would fabricate a position — so
  check `wrist_visibility` / `pose_confidence` when a value looks out of bounds.
- **Frame cap** defaults to 4000 processed frames; longer videos need a larger cap
  or a larger stride, and truncation is reported.

## 8. Layout

```
app.py                          Streamlit dashboard
cli.py                          headless runner
pipeline.py                     orchestration (single decode pass for perception)
requirements.txt
utils/config.py                 every threshold, in one place
utils/signal_utils.py           NaN-tolerant smoothing, derivatives, hysteresis
video/video_loader.py           probing + frame iteration
video/preprocessing.py          resize + colour, with coordinate mapping
pose/pose_extractor.py          MediaPipe pose + hands, two backends
tracking/object_tracker.py      YOLO detection, IoU tracker, manipuland selection
features/motion_features.py     scale-invariant kinematics -> DataFrame
segmentation/base.py            BaseSegmenter / BaseLabeler + registries
segmentation/changepoint.py     PELT with an L2 cost (numpy only)
segmentation/temporal_segmenter.py  change points + interaction events
segmentation/phase_labeler.py   soft rules + Viterbi decode
skills/skill_classifier.py      sequence-template matching
skills/skill_representation.py  JSON + CSV emission
visualization/visualizer.py     annotated video
visualization/plots.py          analysis figures
tests/synthetic.py              parameterised ground-truth generator
tests/test_analysis.py          validation suite
models/                         downloaded .task bundles (tasks backend only)
outputs/                        results
```
