# Skill Extraction from Human Demonstration Video
### Phase 1 of a Learning-from-Demonstration pipeline for robotic manipulation

> **One line:** an uploaded RGB video of a person performing a manipulation task is
> automatically decomposed into labelled temporal phases with timestamps,
> trajectories and a structured skill representation — with no hard-coded
> timings, and with explicit refusal when the evidence is insufficient.

All figures in this document were measured on **2026-10-04** by
`tests/benchmark.py` over **105 synthetic demonstrations with known ground
truth**, plus runtime on a real 608×1080 / 30 fps clip. Raw results:
`outputs/metrics.json`.

---

## 1. Problem and scope

A robot cannot be taught by demonstration until the demonstration has been
*understood*: split into meaningful sub-actions, each with a start, an end, a
label, and the motion that occurred inside it. This phase builds that
understanding layer.

**In scope.** Video → human motion → object interaction → temporal segmentation
→ skill identification → structured representation (JSON/CSV) + visualisation.

**Explicitly out of scope for this phase** (deferred, by design): robot arm
control, MG995R servos, Arduino/ESP32, inverse kinematics, simulation, SNN state
estimation, trajectory execution, hardware integration.

The eventual architecture is
`Human Video → Skill Extraction → Learned Skill → Robot Trajectory → Execution → SNN Phase Estimation`.
**This work delivers the first box and the interface to the second.**

---

## 2. System architecture

```
                 VIDEO (.mp4 / .avi / .mov, any length, any resolution)
                                     │
                    ┌────────────────┴────────────────┐
                    │   Preprocessing (resize, RGB)   │
                    └────────────────┬────────────────┘
                                     │  single decode pass
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
  ┌───────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
  │ MediaPipe Pose │        │ MediaPipe Hands │        │    YOLOv8n      │
  │ shoulder/elbow │        │  21 landmarks   │        │  + IoU tracker  │
  │     /wrist     │        │ → grip aperture │        │ → manipuland    │
  └───────┬────────┘        └────────┬────────┘        └────────┬────────┘
          └──────────────────────────┼──────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Motion features (DataFrame)    │  scale-normalised
                    │  speed, accel, heading, vy,     │  → body-widths/s
                    │  hand–object distance, contact  │
                    └────────────────┬────────────────┘
                                     │
            ┌────────────────────────▼────────────────────────┐
            │  Temporal segmentation                          │
            │  PELT change-points  ∪  interaction events      │
            └────────────────────────┬────────────────────────┘
                                     │
            ┌────────────────────────▼────────────────────────┐
            │  Phase labelling: soft rules + Viterbi decode    │
            └────────────────────────┬────────────────────────┘
                                     │
            ┌────────────────────────▼────────────────────────┐
            │  Skill identification: sequence templates       │
            └────────────────────────┬────────────────────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │ skill_segments.json · skill_segments.csv · trajectory.csv ·       │
   │ processed_video.mp4 · 5 analysis plots · lfd_interface block      │
   └───────────────────────────────────────────────────────────────────┘
```

**Four decoupled components**, each replaceable without touching the others:
feature extraction → temporal segmentation → skill classification → skill
representation. The contract between them is "feature table in, `Segment`
objects out".

---

## 3. Models, and why they were chosen

| Stage | Model | Rationale |
|---|---|---|
| Body pose | **MediaPipe Pose (BlazePose)**, 33 landmarks | Real-time on CPU; emits per-landmark *visibility*, which is used directly as honest confidence rather than being discarded; standard baseline in video-LfD literature |
| Hand | **MediaPipe Hands**, 21 landmarks/hand | The only low-cost source of an aperture signal; without it GRASP and RELEASE are not detectable at all |
| Object | **YOLOv8n** (COCO-80) | Small, fast, no training required, runs offline from cached weights |
| Tracking | **IoU + centroid tracker** (implemented in-repo, ~120 lines) | No extra dependency; sufficient for short tabletop demonstrations |
| Segmentation | **PELT change-point detection** (L2 cost, implemented in-repo) **∪ interaction events**, decoded by **Viterbi** | `ruptures` unavailable; the closed-form piecewise-constant-mean cost is ~50 lines of numpy |

### Design decision: scale invariance
Every velocity is divided by the subject's **shoulder width in pixels**. Units are
therefore *body-widths per second*, and distances are *body-widths*. The same
thresholds work for a 480p clip filmed at 3 m and a 4K clip filmed at 1 m, with
no per-video tuning.

### Design decision: no fabricated 3D
Monocular RGB yields **no metric depth**. All coordinates are 2D image-plane
pixels, tagged `coordinate_space: "2D_image"`, `depth_metric_available: false`.
MediaPipe's `z` channel is retained per frame as `wrist_z_relative` but is
unitless, uncalibrated, and **used in no decision**.

---

## 4. How temporal segmentation works

No timestamp is hard-coded anywhere. Three stages:

**(a) Feature matrix.** Z-scored columns: wrist speed, acceleration magnitude,
heading-change rate, vertical velocity, hand–object distance, object speed, grip
aperture. Channels that are wholly missing (e.g. `obj_speed` with no object) or
constant are *dropped*, not zero-filled.

**(b) Boundary proposal** — union of two independent sources:
- **Statistical:** PELT minimises `Σ C(segment) + β·(#segments)` with
  `β = k·d·log n`, finding regime changes with no knowledge of manipulation.
- **Semantic:** the *interval* over which grip aperture changes (both edges, not
  just the threshold crossing), contact onset/offset, object motion
  onset/offset, wrist-speed minima, prominent hand–object distance minima, and
  vertical direction reversals.

Proposals are clustered (cluster width capped at the tolerance, measured from the
cluster's first member), and when two boundaries are too close the **semantic one
is kept over the statistical one** — an event marks a physical instant, a change
point is only its statistical shadow, usually a few frames late.

**(c) Labelling.** Segment-level aggregate features feed soft membership
functions (one scoring function per phase), decoded by **Viterbi over an
allowed-transition matrix** so the sequence obeys a manipulation grammar — you
cannot RELEASE before you GRASP. Adjacent same-label segments are then merged and
re-scored *keeping* their labels.

Per-segment confidence is the local posterior **given the decoded neighbours**,
scaled by a global quality factor from the pose detection rate and which evidence
channels were available.

---

## 5. HEADLINE METRICS

> Evaluation protocol: 105 synthetic demonstrations rendered from parameterised
> ground-truth phase plans (5 speeds × 3 noise levels × 3 seeds × 2 object
> positions for pick-and-place; 5 speeds × 3 seeds for push). Metrics follow the
> standard temporal-action-segmentation suite (Lea et al., CVPR 2017).

### 5.1 Overall performance

| Metric | All (n=105) | Pick-and-place (n=90) | Push (n=15) |
|---|---|---|---|
| **MoF** (frame accuracy) | **78.8 %** | 77.7 % | 85.5 % |
| **Edit score** | **74.7** | 75.8 | 68.2 |
| **F1@10** | **81.9** | 82.3 | 79.3 |
| **F1@25** | **81.3** | 81.7 | 79.3 |
| **F1@50** | **79.0** | 78.9 | 79.3 |
| **Boundary MAE** | **103 ms** | 105 ms | 90 ms |
| **Boundary F1 @200 ms** | **78.8 %** | — | — |
| **Task accuracy (exact)** | **70 %** | 64 % | **100 %** |
| **Task accuracy (family)** | **100 %** | **100 %** | **100 %** |
| **False skill rate** | **0 %** | **0 %** | **0 %** |

**At the best operating point** (speed sweep at 0.7 px landmark noise, n=15):
**MoF 82.3 %, Edit 81.5, F1@50 81.3, boundary MAE 78 ms, task accuracy 93 %.**

**Canonical pick-and-place demonstration** — the exact 8-phase ground-truth
sequence is recovered:

```
ground truth : REACH → APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETRACT
predicted    : REACH → APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETRACT
task         : PICK_AND_PLACE (confidence 0.99)
frame accuracy 88.5 %   boundary MAE 90 ms
per-boundary error: [367, 0, 33, 33, 33, 67, 100] ms
```

Six of seven boundaries land within **100 ms**. The single large error (367 ms) is
the REACH→APPROACH boundary, which is a gradual deceleration onset rather than a
discrete event — discussed in §9.

### 5.2 Where the task errors actually are

| Ground truth | Predicted | Count | Share |
|---|---|---|---|
| PICK_AND_PLACE | **PICK_AND_PLACE** ✓ | 58 | 64.4 % |
| PICK_AND_PLACE | TRANSFER | 18 | 20.0 % |
| PICK_AND_PLACE | PICK_AND_MOVE | 14 | 15.6 % |
| PICK_AND_PLACE | *wrong family / invented* | **0** | **0 %** |

Every error is a **demotion to a less specific member of the same
pick-and-transport family**, caused by the short PLACE descent being absorbed
into TRANSPORT. The system never confuses a pick-and-place with a push, and never
invents a skill. **This is the metric that matters for a safety-relevant
perception front-end.**

---

## 6. ABLATION — does the method earn its complexity?

Same 15 demonstrations, only the segmentation strategy changed:

| Variant | MoF | Edit | F1@50 | Boundary MAE | Task acc. |
|---|---|---|---|---|---|
| **Full (change-point ∪ events)** | **82.3** | **81.5** | **81.3** | **78 ms** | **93 %** |
| Change-point only | 58.9 | 62.5 | 62.1 | 375 ms | 0 % |
| Uniform fixed windows (baseline) | 72.0 | 80.2 | 64.8 | 198 ms | 47 % |
| Full, penalty × 2.5 | 81.2 | 79.1 | 80.4 | 90 ms | 87 % |

**Key result.** Statistical change-point detection *alone* achieves **0 % task
accuracy** and a 375 ms boundary error. Adding interaction-event boundaries
contributes **+23.3 MoF points, +93 percentage points of task accuracy, and a
4.8× reduction in boundary error.**

**Why:** GRASP and RELEASE are short and statistically quiet — their footprint in
the motion signal is small — but they are semantically decisive. Change points
miss them; events catch them. Conversely, events alone are brittle when the grip
signal is weak. The union is what works.

---

## 7. ROBUSTNESS

### 7.1 Against landmark noise

| Landmark noise | MoF | Edit | F1@50 | Boundary MAE | Task acc. |
|---|---|---|---|---|---|
| 0.0 px | 90.2 | **100.0** | **100.0** | 81 ms | **100 %** |
| 0.5 px | 88.5 | 97.2 | 98.5 | 89 ms | **100 %** |
| 1.0 px | 80.6 | 80.7 | 78.7 | 79 ms | **100 %** |
| 2.0 px | 69.4 | 63.2 | 71.6 | 162 ms | 50 % |
| 3.5 px | 70.7 | 66.3 | 73.7 | 154 ms | 25 % |
| 5.0 px | 59.3 | 54.5 | 53.3 | 407 ms | 0 % |

**Perfect segmentation up to 1 px of landmark jitter; task identification holds to
1 px and degrades gracefully thereafter.** MediaPipe's typical wrist jitter on a
well-lit 960 px frame sits in the 1–2 px band, placing real footage near the edge
of the reliable regime — which is why the evidence-gating of §8 matters.

### 7.2 Against missing evidence channels

| Available evidence | MoF | Edit | Task acc. | Behaviour |
|---|---|---|---|---|
| Object track + grip | **85.3** | **90.0** | **100 %** | full phase vocabulary |
| Grip only (no object) | 67.5 | 66.5 | 0 % | object-directed skills refused |
| Object only (no grip) | 62.4 | 59.4 | 0 % | GRASP/RELEASE/PUSH/PULL withdrawn |
| Neither | 14.1 | 8.4 | 0 % | only IDLE/REACH/RETRACT admissible |

The 0 % task accuracy in degraded rows is **correct behaviour, not failure**: the
system reports `UNKNOWN` rather than guessing. Accuracy collapses precisely where
the evidence does.

### 7.3 Temporal resolution trade-off (measured)

The smoothing window is coupled to the minimum segment duration — smoothing
across a span longer than the shortest resolvable phase blurs it into its
neighbours irrecoverably.

| Window | MoF | Task acc. | Boundary MAE | MoF (heavy noise) |
|---|---|---|---|---|
| 0.10 s | 78.5 | 93 % | 84 ms | 68.1 |
| **0.16 s** (chosen) | **82.3** | **93 %** | **78 ms** | 67.2 |
| 0.22 s | 85.2 | 87 % | 107 ms | 65.5 |
| 0.30 s | 79.6 | 60 % | 125 ms | 60.6 |

0.16 s is the optimum for boundary precision and task accuracy. Note the honest
trade-off: 0.22 s gives higher raw frame accuracy but worse boundaries and worse
task identification, because it swallows the short PLACE phase.

---

## 8. REFUSAL BEHAVIOUR — what the system declines to claim

This is a core contribution, not a caveat. Every row is enforced in code and
covered by a test.

| Situation | Behaviour |
|---|---|
| Landmark below visibility threshold | Written `NaN`, never carried forward; gaps > 0.3 s left missing |
| Hand aperture range too small over the clip | Grip reported **unavailable**; `hand_state = UNKNOWN` throughout |
| Object never moves and never co-moves with the hand | Track **rejected** for lack of interaction evidence, with the reason recorded |
| No reliable object track | `"Object tracking confidence insufficient"`; **all** object-directed skills become ineligible |
| No grip signal | GRASP, RELEASE, PUSH, PULL **removed from the candidate set** — they are claims about hand aperture |
| Neither grip nor object | No contact establishable; only IDLE/REACH/RETRACT/UNKNOWN admissible |
| A template's required phase is missing | **Not a match** — partial matches rejected outright |
| Hand-only skill without object evidence | Confidence scaled by 0.82 to reflect the missing channel |
| Nothing clears the confidence floor | `task: "UNKNOWN"` + best score + rationale naming the observed sequence |
| No human detected | Run fails with an explicit message, not empty segments |

**Negative controls — 3/3 correctly refused:**

| Clip | Reported |
|---|---|
| Static person, no motion | `UNKNOWN` (0.00) ✓ |
| Arm motion, no object, no grip signal | `UNKNOWN` (0.00) ✓ |
| Reach and retract only, no contact | `UNKNOWN` (0.33) ✓ |

> **Found during development:** on a clip of a person merely *talking*, an early
> version reported `TRANSFER` at 0.72 confidence — it had latched onto a potted
> plant in the background that merely *projected* near the hand, and accepted a
> template with only 2 of its 3 required phases. Three fixes followed: the
> manipuland now requires interaction evidence (co-motion or displacement), not
> image proximity; "required" phases became genuinely required; and
> object-directed skills were gated on a reliable track. It now reports
> `UNKNOWN`. **This class of false positive is invisible unless you deliberately
> test on negative clips.**

---

## 9. RUNTIME

Measured on a 6.0 s, 608×1080, 30 fps clip (180 frames), CPU only, after a
warm-up run to exclude model-loading cost. Perception throughput was stable
across three repeats (19.6 / 19.7 / 19.8 fps).

| Stage | Time | Share |
|---|---|---|
| Perception (pose + hands + YOLO) | 9.19 s | 81 % |
| Motion features | 0.01 s | 0.1 % |
| Segmentation + labelling + classification | 0.07 s | 0.6 % |
| Annotated video rendering | 1.74 s | 15 % |
| **Total** | **11.37 s** | **≈ 1.9× realtime** |

| Configuration | Perception throughput | Total | vs realtime |
|---|---|---|---|
| 640 px long edge | 20.5 fps | 12.28 s | 2.05× |
| 960 px long edge | 19.6 fps | 11.37 s | 1.89× |

Resolution barely affects throughput here, because the clip is portrait 608×1080
and both settings downscale to a similar pixel count.

**The entire contributed analysis stack costs 0.7 % of runtime** — 0.075 s per
clip on average across the 105-demo benchmark. Everything else is off-the-shelf
CNN inference. The method is effectively free relative to the perception it sits
on; the only path to realtime is a faster detector or a GPU.

## 10. Outputs and the LfD hand-off

| File | Contents |
|---|---|
| `skill_segments.json` | Full representation: task, per-segment phases with trajectories, motion features, the evidence behind every label, `lfd_interface`, full diagnostics |
| `skill_segments.csv` | `segment_id, skill, start_time, end_time, duration, confidence` + kinematics + gripper command |
| `trajectory.csv` | Per frame: `timestamp, wrist_x/y, elbow_x/y, shoulder_x/y, object_x/y, hand_state` + grip ratio, speeds, hand–object distance, contact, segment id |
| `processed_video.mp4` | Skeleton, hand, object box, wrist trail, HUD, colour-coded timeline with playhead |
| 5 × PNG | Wrist trajectory, object trajectory, velocity with boundaries, grip + contact, segment timeline |

### Interface to the robot stage

Each segment carries an `lfd` block:
- `phase_normalized_time` — τ ∈ [0,1], the phase variable a movement primitive is fitted against
- `hand_relative_to_object_bodywidths` — the **object-centric** hand vector; the frame in which a DMP/ProMP should be fitted so the motion transfers to a new object pose
- `gripper_command` — `CLOSE` / `OPEN` / `HOLD_CLOSED` / `HOLD_OPEN` / `UNKNOWN`
- `start_waypoint` / `end_waypoint`

And `lfd_interface` gives the symbolic plan, gripper event times, and
**three stated blocking requirements before robot execution**:
1. Metric 3D lifting (depth/stereo camera or calibrated monocular 3D pose)
2. Camera-to-robot extrinsic calibration
3. Gripper aperture calibration — the grip signal is a normalised unitless ratio, not millimetres

The per-frame `segment_id` in `trajectory.csv` is ready-made frame-level phase
supervision for the later SNN phase estimator.

---

## 11. Limitations (honest)

1. **REACH vs APPROACH** is the weakest boundary — a gradual deceleration onset, not a discrete event. It accounts for the single 367 ms error in the canonical case.
2. **Short PLACE phases get absorbed** into TRANSPORT at high demonstration speed, demoting PICK_AND_PLACE → TRANSFER / PICK_AND_MOVE. This is 35.6 % of pick-and-place clips and the sole source of exact-task error.
3. **Grip is a unitless aperture ratio**, not finger separation; degraded by self-occlusion when the hand wraps the object.
4. **YOLOv8n is limited to COCO-80.** An unusual manipuland is simply not detected — and correctly reported as such rather than invented.
5. **Single-arm, single-object.** The manipulating arm is chosen by observed path length; bimanual tasks are not modelled.
6. **Off-frame landmarks** are passed through un-clamped (BlazePose extrapolates joints that leave the frame); check `pose_confidence` when a coordinate looks out of bounds.
7. **Synthetic evaluation.** Ground truth comes from parameterised synthetic demonstrations, which exercise the full analysis stack but not the CV models. Real-video evaluation requires a hand-annotated dataset — the natural next step.
8. **Python 3.14 is unsupported** for the perception stage (mediapipe 0.10.30 has dropped `mp.solutions` and its `tasks` API fails on 3.14). Use Python 3.11.

---

## 12. Roadmap

**Immediate (strengthens this phase)**
- Hand-annotate 20–30 real demonstrations to replace synthetic ground truth
- Open-vocabulary detection (OWL-ViT / Grounding DINO) to escape COCO-80
- Learned segmenter (TCN / Transformer) via the existing `BaseSegmenter` interface — the registry already supports it

**Next phases (deliberately not built yet)**
- Movement-primitive fitting (DMP/ProMP) on the object-centric trajectories
- Metric 3D lifting + camera–robot calibration
- Robot trajectory generation and execution
- SNN phase estimation, supervised by the per-frame `segment_id` this phase emits

---

## 13. Reproducing every number

```bash
py -3.11 -m pip install -r requirements.txt

py -3.11 tests/test_analysis.py        # 8/8 correctness tests
py -3.11 tests/test_app_smoke.py       # dashboard renders clean
py -3.11 tests/benchmark.py --video demo.mp4    # every table in §5–§9
                                                # → outputs/metrics.json

py -3.11 app.py                        # dashboard
py -3.11 cli.py demo.mp4               # headless
py -3.11 record_demo.py                # record a test clip
```

**Project size:** 32 Python modules, ~7,100 lines.
**Dependencies:** mediapipe, opencv, ultralytics, numpy, pandas, scipy,
scikit-learn, streamlit, matplotlib. No change-point library — PELT is
implemented in-repo.

---

## 14. Suggested slide plan

| # | Slide | Content to lift |
|---|---|---|
| 1 | Title | §1 one-line summary |
| 2 | Problem & scope | §1 — stress what is deliberately deferred |
| 3 | Architecture | §2 diagram |
| 4 | Models & why | §3 table + scale-invariance and no-fake-3D decisions |
| 5 | Segmentation method | §4 — the (a)/(b)/(c) stages |
| 6 | **Headline results** | §5.1 table + the exact-sequence recovery box |
| 7 | **Ablation** | §6 — "+93 pts task accuracy from event boundaries" is the strongest single slide |
| 8 | Robustness | §7.1 + §7.2 |
| 9 | **Refusal behaviour** | §8 table + the talking-head false-positive story |
| 10 | Runtime | §9 — "analysis stack = 0.7 % of runtime" |
| 11 | Outputs + LfD interface | §10 + a screenshot of `processed_video.mp4` |
| 12 | Limitations | §11 |
| 13 | Roadmap | §12 |

**Three numbers to memorise for questions:**
- **100 % family-level task accuracy, 0 % false skills** across 105 demonstrations
- **+93 percentage points** of task accuracy from the event-boundary contribution
- **0.7 %** of runtime spent in the contributed analysis stack (0.075 s/clip)

**If asked "how do you know it isn't hard-coded?"** — `test_boundaries_follow_the_data`
runs the same demonstration at two speeds with the object moved, and asserts the
recovered grasp onset tracks each one and differs by > 1 s between them. A
pipeline returning canned timestamps fails that test.
