"""Multivariate offline change-point detection (PELT with an L2 cost).

Implemented here rather than taken from ``ruptures`` so the project depends only
on numpy. For the piecewise-constant-mean L2 cost the segment cost has a closed
form from cumulative sums, which makes each candidate evaluation O(d):

    C(a, b) = sum_d [ S2[d] - S1[d]^2 / (b - a) ]

PELT then finds the exact minimiser of

    sum_over_segments C(segment) + beta * (number_of_segments)

in roughly O(n) amortised time thanks to the pruning rule of Killick et al.
(2012), "Optimal detection of changepoints with a linear computational cost".
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class L2Cost:
    """Piecewise-constant-mean cost with O(1) segment queries."""

    def __init__(self, signal: np.ndarray) -> None:
        # signal: (n, d), finite.
        s = np.asarray(signal, dtype=float)
        if s.ndim == 1:
            s = s[:, None]
        self.n, self.d = s.shape
        # Leading zero row so cost(a, b) is a plain difference.
        self._c1 = np.vstack([np.zeros((1, self.d)), np.cumsum(s, axis=0)])
        self._c2 = np.vstack([np.zeros((1, self.d)), np.cumsum(s ** 2, axis=0)])

    def cost(self, a: int, b: int) -> float:
        m = b - a
        if m <= 0:
            return 0.0
        s1 = self._c1[b] - self._c1[a]
        s2 = self._c2[b] - self._c2[a]
        val = float(np.sum(s2 - (s1 ** 2) / m))
        # Guard tiny negatives from floating-point cancellation.
        return max(val, 0.0)


def pelt(
    signal: np.ndarray,
    penalty: float,
    min_size: int = 2,
    jump: int = 1,
) -> List[int]:
    """Return interior change-point indices for ``signal``.

    Parameters
    ----------
    signal:
        ``(n, d)`` array. Columns should already be standardised, otherwise a
        high-variance channel dominates the cost.
    penalty:
        ``beta``. Larger values yield fewer change points.
    min_size:
        Minimum number of samples in a segment.
    jump:
        Only consider change points at multiples of ``jump`` (1 = exact).
    """
    s = np.asarray(signal, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    n = s.shape[0]
    if n < 2 * min_size:
        return []

    cost = L2Cost(s)
    admissible: List[int] = [0]
    # f[tau] = best penalised cost of segmenting the first tau samples.
    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)

    grid = [tau for tau in range(min_size, n + 1) if tau % jump == 0 or tau == n]
    for tau in grid:
        best_val, best_t = np.inf, 0
        candidates = [t for t in admissible if tau - t >= min_size]
        if not candidates:
            continue
        costs = []
        for t in candidates:
            v = f[t] + cost.cost(t, tau) + penalty
            costs.append(v)
            if v < best_val:
                best_val, best_t = v, t
        f[tau] = best_val
        prev[tau] = best_t

        # Pruning: a candidate that already exceeds the current optimum can
        # never become optimal later, because segment cost is non-decreasing.
        # Candidates still inside the min_size window were not evaluated at all,
        # so they must be retained -- they become admissible once tau advances.
        kept = [t for t, v in zip(candidates, costs) if v - penalty <= best_val]
        too_recent = [t for t in admissible if tau - t < min_size]
        admissible = sorted(set(kept) | set(too_recent) | {tau})

    if not np.isfinite(f[n]):
        return []

    # Backtrack.
    bkps: List[int] = []
    tau = n
    guard = 0
    while tau > 0 and guard <= n + 1:
        t = int(prev[tau])
        if t <= 0:
            break
        bkps.append(t)
        tau = t
        guard += 1
    return sorted(bkps)


def default_penalty(n: int, d: int, scale: float = 1.6) -> float:
    """BIC-flavoured penalty for standardised signals: ``scale * d * log(n)``."""
    if n < 2:
        return float("inf")
    return float(scale) * float(max(d, 1)) * float(np.log(n))


def build_signal(
    df, columns, fallback_column: str = "wrist_speed"
) -> Tuple[np.ndarray, List[str]]:
    """Assemble a standardised matrix from the usable feature columns.

    Columns that are entirely missing (for example ``obj_speed`` when no object
    was tracked) or constant are dropped rather than fed in as zeros, so their
    absence cannot dilute the cost function.
    """
    from utils.signal_utils import zscore

    used: List[str] = []
    cols: List[np.ndarray] = []
    for c in columns:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        if np.isnan(v).all():
            continue
        finite = v[~np.isnan(v)]
        if finite.size < 3 or float(np.std(finite)) < 1e-9:
            continue
        cols.append(zscore(v))
        used.append(c)

    if not cols:
        v = df[fallback_column].to_numpy(dtype=float)
        cols = [zscore(v)]
        used = [fallback_column]
    return np.stack(cols, axis=1), used
