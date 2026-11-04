"""End-to-end orchestration.

    VIDEO -> preprocessing -> pose/hand -> object tracking -> motion features
          -> temporal segmentation -> phase labelling -> skill identification
          -> structured representation -> JSON/CSV + annotated video + plots

Perception runs in a single decode pass (pose, hands and detection share each
frame). Rendering needs a second pass because segment labels are only known
after the whole sequence has been analysed.
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from features.motion_features import MotionFeatureExtractor
from pose.pose_extractor import PoseExtractor, PoseFrame
from segmentation.base import Segment, build_labeler, build_segmenter
from skills.skill_classifier import SkillResult, build_classifier
from skills.skill_representation import RepresentationBuilder
from tracking.object_tracker import (
    IoUTracker,
    ManipulandSelector,
    ObjectDetector,
    Track,
)
from utils.config import Config
from video.preprocessing import Preprocessor
from video.video_loader import VideoReader, plan_processing, probe

# Import for registry side effects: these modules register themselves.
import segmentation.temporal_segmenter  # noqa: F401
import segmentation.phase_labeler       # noqa: F401
import skills.skill_classifier          # noqa: F401

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class PipelineResult:
    ok: bool
    message: str = ""
    video_meta: Dict[str, Any] = field(default_factory=dict)
    representation: Dict[str, Any] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)
    features: Any = None
    skill: Optional[SkillResult] = None
    output_files: Dict[str, Path] = field(default_factory=dict)
    plot_files: Dict[str, Path] = field(default_factory=dict)
    processed_video: Optional[Path] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class SkillExtractionPipeline:
    """Runs the full extraction. Every stage is replaceable via ``Config``."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()

    # -- public ----------------------------------------------------------- #
    def run(
        self,
        video_path: str | Path,
        output_dir: Optional[Path] = None,
        progress: ProgressCb = None,
        render_video: Optional[bool] = None,
    ) -> PipelineResult:
        cfg = self.cfg.resolve()
        out_dir = Path(output_dir or cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        notes: List[str] = []
        timings: Dict[str, float] = {}

        def tick(frac: float, msg: str) -> None:
            if progress is not None:
                progress(float(np.clip(frac, 0.0, 1.0)), msg)

        try:
            # ---- 1. video ------------------------------------------------ #
            t0 = time.time()
            meta = probe(video_path)
            plan = plan_processing(
                meta, cfg.video.frame_stride, cfg.video.max_frames
            )
            if plan["truncated"]:
                notes.append(
                    f"Video is longer than the {cfg.video.max_frames}-frame "
                    f"processing cap; only the first "
                    f"{plan['covered_duration_s']:.1f}s were analysed. Raise "
                    f"VideoConfig.max_frames or increase frame_stride to cover "
                    f"the whole clip."
                )
            if meta.frame_count_is_estimate:
                notes.append(
                    "The container did not report a reliable frame count or "
                    "frame rate; timestamps are derived from an assumed rate."
                )
            tick(0.03, f"Loaded {Path(meta.path).name} ({meta.resolution}, "
                       f"{meta.fps:.2f} fps)")

            # ---- 2 + 3. perception (single decode pass) ------------------ #
            (pose_seq, tracks, det_note, frame_diag,
             target_class, unknown_target) = self._perception_pass(meta, plan, tick)
            if unknown_target:
                notes.append(
                    f"'{unknown_target}' is not one of the 80 COCO classes this "
                    f"detector knows, so the hint was ignored and object "
                    f"selection fell back to automatic inference. Run "
                    f"`py -3.11 diagnose.py --list-objects` to see the usable "
                    f"class names."
                )
            if det_note:
                notes.append(det_note)
            timings["perception_s"] = round(time.time() - t0, 2)

            if not pose_seq.frames:
                return PipelineResult(
                    ok=False,
                    message="No frames could be processed from this video.",
                    video_meta=meta.to_dict(),
                )
            if pose_seq.detection_rate <= 0.02:
                return PipelineResult(
                    ok=False,
                    message=(
                        "No human pose was detected in this video. The pipeline "
                        "needs a visible person performing the demonstration; it "
                        "cannot extract a manipulation skill from footage with no "
                        "detectable human."
                    ),
                    video_meta=meta.to_dict(),
                    notes=notes + list(pose_seq.notes),
                )

            # ---- object selection ---------------------------------------- #
            t1 = time.time()
            manip, object_info = self._select_object(pose_seq, tracks, target_class)
            if not manip.reliable:
                notes.append(manip.message)
            tick(0.68, manip.message)

            # ---- 4. motion features -------------------------------------- #
            ft = MotionFeatureExtractor(cfg.features, cfg.pose).build(
                pose_seq, manip.track
            )
            notes.extend(ft.notes)
            timings["features_s"] = round(time.time() - t1, 2)
            tick(0.72, f"Computed motion features for {len(ft.df)} samples")

            context: Dict[str, Any] = {
                "body_scale_px": ft.body_scale_px,
                "object_available": ft.object_available,
                "grip_available": ft.grip_available,
                "contact_basis": str(ft.df["contact_basis"].iloc[0]),
                "pose_detection_rate": pose_seq.detection_rate,
                "pose_backend": pose_seq.backend,
                "active_arm": ft.active_arm,
                "object_info": object_info,
                "notes": list(pose_seq.notes),
            }

            # ---- 5. temporal segmentation -------------------------------- #
            t2 = time.time()
            segmenter = build_segmenter(cfg.segmentation.method, cfg.segmentation)
            segments = segmenter.segment(ft.df, context)
            seg_diag = dict(getattr(segmenter, "last_diagnostics", {}))
            tick(0.80, f"Found {len(segments)} temporal segments")

            # ---- 6. phase labelling -------------------------------------- #
            labeler = build_labeler(cfg.skills.labeler, cfg.skills)
            segments = labeler.label(segments, ft.df, context)
            lab_diag = dict(getattr(labeler, "last_diagnostics", {}))
            timings["segmentation_s"] = round(time.time() - t2, 2)
            tick(0.85, "Labelled phases: " + " -> ".join(s.label for s in segments))

            # ---- 7. skill identification --------------------------------- #
            classifier = build_classifier(cfg.skills.classifier, cfg.skills)
            skill = classifier.classify(segments, ft.df, context)
            tick(0.88, f"Task: {skill.task} ({skill.confidence:.2f})")

            # ---- 8. representation + files ------------------------------- #
            video_meta_doc = dict(meta.to_dict())
            video_meta_doc["filename"] = Path(meta.path).name
            video_meta_doc["processing"] = plan

            diagnostics = {
                "segmenter": seg_diag,
                "labeler": lab_diag,
                "classifier": classifier.name,
                "frame_diagnostics": frame_diag,
                "config": cfg.to_dict(),
                "timings_s": timings,
            }
            builder = RepresentationBuilder(
                ft, segments, skill, video_meta_doc, context, diagnostics
            )
            files = builder.write_all(out_dir)
            representation = builder.build().data
            tick(0.90, "Wrote skill_segments.json / .csv and trajectory.csv")

            # ---- 9. visualisation ---------------------------------------- #
            from visualization import plots
            from visualization.visualizer import SegmentVideoRenderer

            plot_files = plots.save_all(
                ft.df, segments, out_dir, meta.width, meta.height,
                total_duration=max(meta.duration_s, 1e-6),
            )

            processed: Optional[Path] = None
            do_render = cfg.viz.write_video if render_video is None else render_video
            if do_render:
                t3 = time.time()
                tick(0.92, "Rendering annotated video")
                renderer = SegmentVideoRenderer(cfg.viz)
                processed = renderer.render(
                    video_path=meta.path,
                    out_path=out_dir / "processed_video.mp4",
                    df=ft.df,
                    segments=segments,
                    total_duration=max(meta.duration_s, 1e-6),
                    render_long_edge=cfg.video.target_long_edge or 960,
                    object_available=ft.object_available,
                    task_label=skill.task,
                    task_confidence=skill.confidence,
                    progress_cb=lambda i, n, st: tick(
                        0.92 + 0.07 * (i / max(n, 1)), "Rendering annotated video"
                    ),
                    max_frames=(
                        None if cfg.video.max_frames is None
                        else cfg.video.max_frames * cfg.video.frame_stride
                    ),
                )
                timings["render_s"] = round(time.time() - t3, 2)
                if processed is None:
                    notes.append(
                        "The annotated video could not be encoded by this OpenCV "
                        "build; all numeric outputs are still available."
                    )

            if processed is not None:
                files["processed_video"] = processed
            timings["total_s"] = round(time.time() - t0, 2)
            tick(1.0, "Done")

            return PipelineResult(
                ok=True,
                message=f"Extracted {len(segments)} segments; task = {skill.task}",
                video_meta=video_meta_doc,
                representation=representation,
                segments=segments,
                features=ft,
                skill=skill,
                output_files=files,
                plot_files=plot_files,
                processed_video=processed,
                diagnostics=diagnostics,
                notes=notes,
                timings=timings,
            )

        except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
            return PipelineResult(
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
                notes=notes + [traceback.format_exc(limit=8)],
            )

    # -- stages ----------------------------------------------------------- #
    def _perception_pass(self, meta, plan, tick):
        """One decode pass producing pose frames and object tracks."""
        cfg = self.cfg
        reader = VideoReader(
            meta.path, stride=cfg.video.frame_stride, max_frames=cfg.video.max_frames
        )
        pre = Preprocessor(cfg.video.target_long_edge)
        extractor = PoseExtractor(cfg.pose)
        detector = ObjectDetector(cfg.objects)
        diagonal = math.hypot(meta.width, meta.height)
        tracker = IoUTracker(cfg.objects, diagonal)

        det_note = detector.unavailable_reason
        total = plan["frames_to_process"]
        frames: List[PoseFrame] = []
        n_detect_calls = 0

        try:
            for i, (fidx, ts, bgr) in enumerate(reader):
                pf, proc_bgr = extractor.process_frame(fidx, ts, bgr, pre)
                frames.append(pf)

                if detector.available:
                    inv = 1.0 / (pre.geometry.scale if pre.geometry else 1.0)
                    if i % max(1, cfg.objects.detect_every) == 0:
                        dets = detector.detect(proc_bgr, inv)
                        tracker.update(dets, fidx, ts)
                        n_detect_calls += 1
                    else:
                        tracker.coast(fidx, ts)

                if i % 4 == 0:
                    tick(
                        0.03 + 0.62 * ((i + 1) / max(total, 1)),
                        f"Analysing frame {i + 1}/{total} "
                        f"(pose, hands{', objects' if detector.available else ''})",
                    )
        finally:
            extractor.close()

        seq = extractor.finalize(frames)
        diag = {
            "frames_processed": len(frames),
            "pose_detection_rate": round(seq.detection_rate, 4),
            "pose_backend": seq.backend,
            "active_arm": seq.active_arm,
            "object_detector_available": detector.available,
            "detection_calls": n_detect_calls,
            "raw_tracks": len(tracker.tracks),
            "tracks_kept": len(tracker.finished_tracks()),
            "target_class": detector.target_class,
        }
        return (seq, tracker.finished_tracks(), det_note, diag,
                detector.target_class, detector.unknown_target())

    def _select_object(self, seq, tracks: List[Track], target_class=None):
        """Pick the manipulated object, or decline to."""
        cfg = self.cfg
        arm = seq.active_arm if seq.active_arm in ("left", "right") else "right"
        frames = seq.frames
        t = np.array([f.timestamp for f in frames], dtype=float)
        fi = np.array([f.frame_index for f in frames], dtype=int)

        hand = np.full((len(frames), 2), np.nan)
        for i, f in enumerate(frames):
            c = f.hand_center.get(arm)
            if c is None:
                k = f.kp(f"{arm}_wrist")
                c = (k.x, k.y) if k.valid else None
            if c is not None:
                hand[i] = c

        widths = []
        for f in frames:
            ls, rs = f.kp("left_shoulder"), f.kp("right_shoulder")
            if ls.valid and rs.valid:
                widths.append(math.dist((ls.x, ls.y), (rs.x, rs.y)))
        body_scale = (
            float(np.median(widths)) if widths else cfg.features.fallback_scale_px
        )

        result = ManipulandSelector(cfg.objects).select(
            tracks, hand, t, fi, body_scale, target_class=target_class
        )
        info = {
            "available": result.reliable,
            "message": result.message,
            "score": round(result.score, 4),
            "candidates_considered": result.candidates,
        }
        if result.track is not None:
            info.update({
                "class": result.track.class_name,
                "track_id": result.track.track_id,
                "observed_frames": result.track.length,
                "detected_fraction": round(result.track.detected_fraction, 4),
                "mean_detection_confidence": round(result.track.mean_confidence, 4),
            })
        return result, info
