"""Central configuration for the LfD skill-extraction pipeline.

Every tunable lives here so that no numeric threshold is buried in the
algorithms. Nothing in this file encodes *when* a phase happens -- only how
sensitive the detectors are.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------------------------------- #
# Video / preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    # Long edge is downscaled to this before inference (None = keep native).
    target_long_edge: Optional[int] = 960
    # Process every Nth frame (1 = all). Timestamps stay in real seconds.
    frame_stride: int = 1
    # Hard cap so a very long upload cannot hang the UI. None = no cap.
    max_frames: Optional[int] = 4000
    allowed_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov")


# --------------------------------------------------------------------------- #
# Pose / hand extraction
# --------------------------------------------------------------------------- #
@dataclass
class PoseConfig:
    # "auto" | "solutions" | "tasks"
    backend: str = "auto"
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # A landmark below this visibility is treated as missing, not guessed.
    min_landmark_visibility: float = 0.45
    max_num_hands: int = 2
    # Which arm to treat as the manipulator: "auto" | "left" | "right".
    active_arm: str = "auto"
    # Gaps up to this many seconds are linearly interpolated; longer stay NaN.
    max_interpolation_gap_s: float = 0.30
    # Grip aperture ratio below/above which the hand is CLOSED/OPEN.
    # Hysteresis: the far threshold must be crossed to flip state.
    grip_closed_ratio: float = 0.55
    grip_open_ratio: float = 0.75


# --------------------------------------------------------------------------- #
# Object detection / tracking
# --------------------------------------------------------------------------- #
@dataclass
class ObjectConfig:
    enabled: bool = True
    weights: str = "yolov8n.pt"
    det_confidence: float = 0.25
    iou_threshold: float = 0.45
    # Detection runs every Nth processed frame; tracks coast in between.
    detect_every: int = 2
    # COCO classes that are never candidate manipulands.
    excluded_classes: Tuple[str, ...] = ("person",)
    # Optional hint: the class the user says is being manipulated, e.g. "bottle".
    # When set, only this class is considered as a candidate manipuland, a lower
    # detection threshold is used for it, and the interaction gate is relaxed --
    # the user has already supplied the evidence that it is the target, so the
    # system does not have to infer it. This removes the single largest source of
    # downstream label error: tracking the wrong object.
    target_class: Optional[str] = None
    # Detection threshold applied to the hinted class (lower = more recall).
    target_det_confidence: float = 0.12
    # Track association.
    min_iou_for_match: float = 0.20
    max_center_jump_frac: float = 0.25   # fraction of frame diagonal
    max_track_age_frames: int = 12
    min_track_length_frames: int = 8
    # Below this manipulation score the object is reported as unreliable
    # instead of being invented.
    min_manipuland_score: float = 0.35
    # A candidate must also show *interaction*, not just image-plane proximity:
    # max(co-motion with the hand, own displacement) has to clear this. In a
    # monocular view a distant background object can project right onto the hand,
    # so proximity alone is not evidence that it is being manipulated.
    min_interaction_score: float = 0.25
    # A track must be detected in at least this fraction of the frames it spans.
    min_track_density: float = 0.35


# --------------------------------------------------------------------------- #
# Motion features
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    # Savitzky-Golay window in seconds (an odd frame count is derived from it).
    #
    # This must not exceed SegmentationConfig.min_segment_duration_s: smoothing
    # across a span longer than the shortest phase you intend to resolve blurs
    # that phase into its neighbours, and no amount of change-point sensitivity
    # recovers it afterwards. Config.resolve() enforces the relationship, so the
    # minimum segment duration is the single knob that sets temporal resolution.
    smooth_window_s: float = 0.16
    savgol_polyorder: int = 2
    # Fallback body scale in px when shoulders are not visible.
    fallback_scale_px: float = 150.0
    # Hand-object distance (in body-widths) under which contact is plausible.
    contact_distance_bw: float = 0.45
    # Object speed (body-widths/s) above which the object counts as moving.
    object_moving_speed_bw: float = 0.12
    # Wrist speed (body-widths/s) above which the hand counts as moving.
    wrist_moving_speed_bw: float = 0.18


# --------------------------------------------------------------------------- #
# Temporal segmentation
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationConfig:
    # Registered segmenter key; see segmentation.registry.
    method: str = "changepoint_event"
    # PELT penalty multiplier: beta = penalty_scale * d * log(n).
    penalty_scale: float = 1.6
    # Shortest admissible segment.
    min_segment_duration_s: float = 0.16
    # Boundaries closer together than this are merged into one.
    boundary_merge_tolerance_s: float = 0.12
    # Upper bound on segment count (guards against over-segmentation).
    max_segments: int = 24
    # Feature columns fed to the change-point detector.
    changepoint_features: Tuple[str, ...] = (
        "wrist_speed",
        "wrist_accel",
        "wrist_heading_change",
        "wrist_vy",
        "hand_obj_dist",
        "obj_speed",
        "grip_ratio",
    )
    # Whether discrete interaction events also propose boundaries.
    use_event_boundaries: bool = True


# --------------------------------------------------------------------------- #
# Phase labelling / skill classification
# --------------------------------------------------------------------------- #
@dataclass
class SkillConfig:
    labeler: str = "rule_viterbi"
    classifier: str = "template"
    # Below this the task is reported UNKNOWN rather than guessed.
    min_task_confidence: float = 0.40
    # Segments whose best-label confidence is below this become UNKNOWN.
    min_phase_confidence: float = 0.25


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
@dataclass
class VizConfig:
    write_video: bool = True
    codec_preference: Tuple[str, ...] = ("avc1", "mp4v")
    remux_with_ffmpeg: bool = True
    timeline_height_px: int = 78
    font_scale: float = 0.6
    draw_trail_seconds: float = 1.5


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    output_dir: Path = OUTPUT_DIR

    def __post_init__(self) -> None:
        self.resolve()

    def resolve(self) -> "Config":
        """Enforce relationships between settings that must stay consistent.

        Called on construction and again by the pipeline, so that settings
        changed afterwards (by the dashboard's sidebar, for instance) are still
        reconciled rather than silently contradicting each other.
        """
        # Never smooth across more than the shortest resolvable phase.
        self.features.smooth_window_s = min(
            self.features.smooth_window_s,
            self.segmentation.min_segment_duration_s,
        )
        return self

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d


# --------------------------------------------------------------------------- #
# Phase vocabulary
# --------------------------------------------------------------------------- #
PHASE_LABELS: List[str] = [
    "IDLE",
    "REACH",
    "APPROACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETRACT",
    "PUSH",
    "PULL",
    "UNKNOWN",
]

# Display colours (BGR, for OpenCV overlays).
PHASE_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "IDLE":      (128, 128, 128),
    "REACH":     (219, 152, 52),
    "APPROACH":  (241, 196, 15),
    "GRASP":     (60, 76, 231),
    "LIFT":      (96, 174, 39),
    "TRANSPORT": (182, 89, 155),
    "PLACE":     (15, 196, 241),
    "RELEASE":   (34, 126, 230),
    "RETRACT":   (173, 168, 148),
    "PUSH":      (80, 62, 44),
    "PULL":      (133, 160, 22),
    "UNKNOWN":   (70, 70, 70),
}

# Matplotlib / HTML hex equivalents.
PHASE_COLORS_HEX: Dict[str, str] = {
    "IDLE":      "#808080",
    "REACH":     "#3498db",
    "APPROACH":  "#f1c40f",
    "GRASP":     "#e74c3c",
    "LIFT":      "#27ae60",
    "TRANSPORT": "#9b59b6",
    "PLACE":     "#e8a33d",
    "RELEASE":   "#e67e22",
    "RETRACT":   "#94a8ad",
    "PUSH":      "#2c3e50",
    "PULL":      "#16a085",
    "UNKNOWN":   "#464646",
}
