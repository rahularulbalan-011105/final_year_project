"""Static analysis plots for the dashboard and the report.

Every figure is annotated with the segment boundaries the pipeline actually
found, so a reader can check by eye that a boundary coincides with a real event
in the signal rather than an arbitrary time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # headless: Streamlit renders the figure objects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from segmentation.base import Segment
from utils.config import PHASE_COLORS_HEX

GRID = dict(alpha=0.25, linewidth=0.6)


def _legend_if_labelled(ax) -> None:
    """Draw a legend only when something on the axes is actually labelled.

    Panels degrade to a "signal unavailable" message when the grip or object
    channel is missing, and calling legend() on an empty axes emits a warning.
    """
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, frameon=False, loc="upper right")


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _shade_segments(ax, segments: List[Segment], alpha: float = 0.18) -> None:
    for s in segments:
        ax.axvspan(
            s.start_time, s.end_time,
            color=PHASE_COLORS_HEX.get(s.label, "#888888"), alpha=alpha, lw=0,
        )
        ax.axvline(s.start_time, color="#444444", lw=0.5, alpha=0.5)


def _legend(fig, segments: List[Segment]) -> None:
    seen: List[str] = []
    for s in segments:
        if s.label not in seen:
            seen.append(s.label)
    handles = [
        Patch(facecolor=PHASE_COLORS_HEX.get(l, "#888888"), label=l) for l in seen
    ]
    if handles:
        fig.legend(
            handles=handles, loc="lower center", ncol=min(len(handles), 6),
            fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02),
        )


def wrist_trajectory_figure(
    df: pd.DataFrame, segments: List[Segment], width: int, height: int
) -> Figure:
    """Hand path in the image plane, coloured by phase."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=130)
    for s in segments:
        sl = df.iloc[s.start_index:s.end_index]
        ax.plot(
            sl["wrist_x"], sl["wrist_y"],
            color=PHASE_COLORS_HEX.get(s.label, "#888888"),
            lw=2.2, solid_capstyle="round",
            label=s.label,
        )
    wx = df["wrist_x"].to_numpy(dtype=float)
    wy = df["wrist_y"].to_numpy(dtype=float)
    ok = ~(np.isnan(wx) | np.isnan(wy))
    if ok.any():
        ax.scatter(wx[ok][0], wy[ok][0], s=42, marker="o",
                   facecolor="white", edgecolor="#222", zorder=5, label="_start")
        ax.annotate("start", (wx[ok][0], wy[ok][0]), fontsize=7,
                    xytext=(6, 6), textcoords="offset points")
        ax.scatter(wx[ok][-1], wy[ok][-1], s=42, marker="s",
                   facecolor="#222", edgecolor="white", zorder=5)
        ax.annotate("end", (wx[ok][-1], wy[ok][-1]), fontsize=7,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)      # image convention: y down
    ax.set_aspect("equal", adjustable="box")
    _style(ax, "Wrist trajectory (image plane, 2D)", "x (px)", "y (px)")
    _legend(fig, segments)
    fig.tight_layout()
    return fig


def object_trajectory_figure(
    df: pd.DataFrame, segments: List[Segment], width: int, height: int
) -> Optional[Figure]:
    if "object_x" not in df.columns or df["object_x"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=130)
    for s in segments:
        sl = df.iloc[s.start_index:s.end_index]
        ax.plot(sl["object_x"], sl["object_y"],
                color=PHASE_COLORS_HEX.get(s.label, "#888888"), lw=2.4)
    ax.plot(df["wrist_x"], df["wrist_y"], color="#999999", lw=0.8, ls="--",
            label="wrist (reference)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    _style(ax, "Tracked object trajectory", "x (px)", "y (px)")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def velocity_figure(df: pd.DataFrame, segments: List[Segment]) -> Figure:
    """Wrist speed / acceleration with the detected boundaries overlaid."""
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.2), dpi=130, sharex=True)
    t = df["timestamp"]

    ax = axes[0]
    _shade_segments(ax, segments)
    ax.plot(t, df["wrist_speed"], color="#c0392b", lw=1.5, label="wrist speed")
    if "obj_speed" in df.columns and not df["obj_speed"].isna().all():
        ax.plot(t, df["obj_speed"], color="#2980b9", lw=1.3, label="object speed")
    _style(ax, "Speed with detected phase boundaries", "", "body-widths / s")
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[1]
    _shade_segments(ax, segments)
    ax.plot(t, df["wrist_accel"], color="#8e44ad", lw=1.2,
            label="|wrist acceleration|")
    ax.plot(t, df["wrist_vy"], color="#16a085", lw=1.2,
            label="vertical velocity (+up)")
    ax.axhline(0, color="#555", lw=0.6)
    _style(ax, "Acceleration and vertical velocity", "time (s)",
           "body-widths / s (per s)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def interaction_figure(df: pd.DataFrame, segments: List[Segment]) -> Figure:
    """Grip aperture, hand-object distance and contact -- the contact evidence."""
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.0), dpi=130, sharex=True)
    t = df["timestamp"]

    ax = axes[0]
    _shade_segments(ax, segments)
    if not df["grip_ratio"].isna().all():
        ax.plot(t, df["grip_ratio"], color="#d35400", lw=1.5,
                label="grip ratio (1 = open)")
    else:
        ax.text(0.5, 0.5, "grip signal unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888")
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Hand aperture", "", "ratio")
    _legend_if_labelled(ax)

    ax = axes[1]
    _shade_segments(ax, segments)
    if not df["hand_obj_dist"].isna().all():
        ax.plot(t, df["hand_obj_dist"], color="#2c3e50", lw=1.5,
                label="hand-object distance")
    else:
        ax.text(0.5, 0.62, "no object tracked", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888")
    contact = df["contact"].to_numpy(dtype=bool)
    if contact.any():
        ax.fill_between(t, 0, 1, where=contact, transform=ax.get_xaxis_transform(),
                        color="#27ae60", alpha=0.18, label="contact")
    _style(ax, "Hand-object distance and inferred contact", "time (s)",
           "body-widths")
    _legend_if_labelled(ax)
    fig.tight_layout()
    return fig


def timeline_figure(segments: List[Segment], total_duration: float) -> Figure:
    """The segmentation as a single labelled bar."""
    fig, ax = plt.subplots(figsize=(7.4, 1.9), dpi=130)
    for s in segments:
        ax.barh(
            0, s.duration, left=s.start_time, height=0.6,
            color=PHASE_COLORS_HEX.get(s.label, "#888888"),
            edgecolor="white", linewidth=0.8,
        )
        if s.duration > total_duration * 0.045:
            ax.text(
                s.start_time + s.duration / 2, 0, s.label,
                ha="center", va="center", fontsize=6.5, rotation=0,
                color="#111", fontweight="bold",
            )
        ax.text(
            s.start_time + s.duration / 2, -0.45, f"{s.confidence:.2f}",
            ha="center", va="center", fontsize=6, color="#555",
        )
    ax.set_xlim(0, max(total_duration, 1e-6))
    ax.set_ylim(-0.7, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Detected temporal segments (label / confidence)",
                 fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def save_all(
    df: pd.DataFrame,
    segments: List[Segment],
    out_dir: Path,
    width: int,
    height: int,
    total_duration: float,
) -> Dict[str, Path]:
    """Render every figure to PNG and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    figures = {
        "wrist_trajectory": wrist_trajectory_figure(df, segments, width, height),
        "velocity": velocity_figure(df, segments),
        "interaction": interaction_figure(df, segments),
        "timeline": timeline_figure(segments, total_duration),
    }
    obj = object_trajectory_figure(df, segments, width, height)
    if obj is not None:
        figures["object_trajectory"] = obj

    for name, fig in figures.items():
        p = out_dir / f"{name}.png"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[name] = p
    return paths
