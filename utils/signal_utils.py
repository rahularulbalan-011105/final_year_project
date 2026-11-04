"""Numeric helpers shared by the feature and segmentation stages.

Deliberately dependency-light: numpy + an optional scipy fast path, so the
pipeline still runs if scipy is unavailable.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional fast path
    from scipy.signal import savgol_filter as _savgol
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def odd_window(n_frames: float, minimum: int = 5) -> int:
    """Round a desired window length up to an odd integer >= ``minimum``."""
    w = int(round(n_frames))
    if w < minimum:
        w = minimum
    if w % 2 == 0:
        w += 1
    return w


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than ``max_gap`` samples.

    Longer runs stay NaN: a missing measurement is reported as missing rather
    than invented. Leading/trailing NaNs are never extrapolated.
    """
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return x

    idx = np.arange(n)
    good = idx[valid]
    # Walk each interior gap between consecutive valid samples.
    for a, b in zip(good[:-1], good[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            x[a + 1:b] = np.interp(idx[a + 1:b], [a, b], [x[a], x[b]])
    return x


def smooth(x: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """NaN-tolerant smoothing (Savitzky-Golay when scipy is present).

    NaNs are bridged for the filter pass and then restored, so smoothing never
    fabricates values where there was no observation.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return x

    filled = x.copy()
    if nan_mask.any():
        idx = np.arange(n)
        filled[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])

    w = min(odd_window(window), n if n % 2 == 1 else n - 1)
    if w < 3:
        out = filled
    elif _HAVE_SCIPY and w > polyorder:
        out = _savgol(filled, w, polyorder, mode="interp")
    else:
        kernel = np.ones(w) / w
        padded = np.pad(filled, (w // 2, w // 2), mode="edge")
        out = np.convolve(padded, kernel, mode="valid")

    out = np.asarray(out, dtype=float)
    out[nan_mask] = np.nan
    return out


def derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference time derivative that tolerates NaNs and uneven dt."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    if n < 2:
        return out
    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        dt = t[hi] - t[lo]
        if dt <= 0 or np.isnan(x[hi]) or np.isnan(x[lo]):
            continue
        out[i] = (x[hi] - x[lo]) / dt
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise, treating NaN as the column mean and guarding zero variance."""
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return np.zeros_like(x)
    mu = float(np.mean(x[valid]))
    sd = float(np.std(x[valid]))
    out = np.where(valid, x, mu)
    if sd < 1e-9:
        return out - mu
    return (out - mu) / sd


def robust_nanmedian(x: Sequence[float], default: float = np.nan) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return default
    return float(np.median(arr))


def hysteresis_threshold(
    x: np.ndarray,
    low: float,
    high: float,
    initial: Optional[bool] = None,
) -> np.ndarray:
    """Two-threshold (Schmitt) binariser.

    Returns a boolean array that is True while the signal is in the "low"
    state (below ``low``) and only flips back once it exceeds ``high``. Used
    for the hand open/closed decision so that noise near a single threshold
    cannot produce a burst of spurious grasp events.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.zeros(n, dtype=bool)
    state = bool(initial) if initial is not None else False
    for i in range(n):
        v = x[i]
        if np.isnan(v):
            out[i] = state
            continue
        if state:
            if v > high:
                state = False
        else:
            if v < low:
                state = True
        out[i] = state
    return out


def local_minima(x: np.ndarray, min_separation: int = 1) -> np.ndarray:
    """Indices of local minima, greedily thinned to ``min_separation`` apart."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.array([], dtype=int)
    filled = x.copy()
    if np.isnan(filled).any():
        idx = np.arange(n)
        m = np.isnan(filled)
        if (~m).sum() < 2:
            return np.array([], dtype=int)
        filled[m] = np.interp(idx[m], idx[~m], filled[~m])
    cand = [i for i in range(1, n - 1)
            if filled[i] <= filled[i - 1] and filled[i] < filled[i + 1]]
    cand.sort(key=lambda i: filled[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_separation for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), dtype=int)


def rising_edges(mask: np.ndarray) -> np.ndarray:
    """Indices where a boolean array transitions False -> True."""
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.array([], dtype=int)
    return np.where(~m[:-1] & m[1:])[0] + 1


def falling_edges(mask: np.ndarray) -> np.ndarray:
    """Indices where a boolean array transitions True -> False."""
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.array([], dtype=int)
    return np.where(m[:-1] & ~m[1:])[0] + 1


def format_timestamp(t: float) -> str:
    """Seconds -> MM:SS.mmm, the format used in the UI and the CSV."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "--:--.--"
    t = max(0.0, float(t))
    minutes = int(t // 60)
    seconds = t - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"
