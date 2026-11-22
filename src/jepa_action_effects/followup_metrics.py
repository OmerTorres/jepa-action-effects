"""Pure measurements for the REVIEW-BRANCH-002 follow-up (exploratory, D026).

- **Absolute endpoint error** ‖decoded_end − true_end‖. The published
  `endpoint_err_*` values are displacement errors, which cancel shared position bias.
- **Length/direction decomposition** of predicted vs true displacement. The
  projection slope Σ(p·t)/Σ(t·t) pools shorter vectors and direction errors; this
  separates them.
- **Cluster bootstrap**: resample clusters (families or archive episodes) with all of
  their rows.
- **Record comparison**: the maximum absolute difference over the numeric leaves of
  two nested records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

ZERO_LENGTH = 1e-9  # predicted displacement treated as zero (undefined direction)


def absolute_endpoint_error(decoded_end: np.ndarray, true_end: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(decoded_end, float) - np.asarray(true_end, float), axis=-1)


def displacement_error(
    decoded_end: np.ndarray,
    decoded_start: np.ndarray,
    true_end: np.ndarray,
    true_start: np.ndarray,
) -> np.ndarray:
    decoded = np.asarray(decoded_end, float) - np.asarray(decoded_start, float)
    true = np.asarray(true_end, float) - np.asarray(true_start, float)
    return np.linalg.norm(decoded - true, axis=-1)


def decompose(predicted: np.ndarray, true: np.ndarray) -> dict[str, float | None]:
    """Pooled summaries of (N,2) predicted vs true displacements (true must be nonzero)."""
    p = np.asarray(predicted, float).reshape(-1, 2)
    t = np.asarray(true, float).reshape(-1, 2)
    if len(t) == 0:
        return {key: None for key in _DECOMPOSE_KEYS}
    t_len = np.linalg.norm(t, axis=1)
    p_len = np.linalg.norm(p, axis=1)
    if np.any(t_len <= 0):
        raise ValueError("True displacements must be nonzero")
    # A zero predicted vector has no direction: it stays in the length and projection
    # statistics but is excluded from the angle statistics and counted as missed motion.
    defined = p_len > ZERO_LENGTH
    cross = t[defined, 0] * p[defined, 1] - t[defined, 1] * p[defined, 0]
    angle = np.abs(np.degrees(np.arctan2(cross, np.sum(p[defined] * t[defined], axis=1))))
    parallel = np.sum(p * t, axis=1) / t_len
    return {
        "projection_slope": float(np.sum(p * t) / np.sum(t * t)),
        "length_ratio_pooled": float(p_len.sum() / t_len.sum()),
        "length_ratio_median": float(np.median(p_len / t_len)),
        "abs_angle_mean_deg": float(np.mean(angle)) if len(angle) else None,
        "abs_angle_median_deg": float(np.median(angle)) if len(angle) else None,
        "parallel_ratio_pooled": float(parallel.sum() / t_len.sum()),
        "count": int(len(t)),
        "undefined_angle_count": int(np.sum(~defined)),
    }


_DECOMPOSE_KEYS = (
    "projection_slope",
    "length_ratio_pooled",
    "length_ratio_median",
    "abs_angle_mean_deg",
    "abs_angle_median_deg",
    "parallel_ratio_pooled",
    "count",
    "undefined_angle_count",
)


def cluster_bootstrap(
    clusters: Sequence[Sequence[Any]],
    statistic: Callable[[list[Any]], float | None],
    seed: int,
    samples: int = 2000,
) -> dict[str, Any]:
    """Resample whole clusters; `statistic` receives the flattened rows."""
    clusters = [list(cluster) for cluster in clusters if len(cluster)]
    rows = [row for cluster in clusters for row in cluster]
    estimate = statistic(rows) if rows else None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples if clusters else 0):
        chosen = rng.integers(0, len(clusters), len(clusters))
        value = statistic([row for index in chosen for row in clusters[index]])
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci95": np.percentile(draws, (2.5, 97.5)).tolist() if len(draws) >= samples // 2 else None,
        "clusters": len(clusters),
        "rows": len(rows),
    }


def max_abs_difference(left: Any, right: Any, skip: frozenset[str] = frozenset()) -> float:
    """Largest absolute numeric difference; structure mismatches raise ValueError."""
    if isinstance(left, dict) and isinstance(right, dict):
        keys = (set(left) | set(right)) - skip
        if set(left) - skip != set(right) - skip:
            raise ValueError(f"Key mismatch: {sorted((set(left) ^ set(right)) - skip)}")
        return max((max_abs_difference(left[k], right[k], skip) for k in keys), default=0.0)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        if len(left) != len(right):
            raise ValueError("Length mismatch")
        return max(
            (max_abs_difference(a, b, skip) for a, b in zip(left, right, strict=True)), default=0.0
        )
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        if left != right:
            raise ValueError(f"Value mismatch: {left!r} vs {right!r}")
        return 0.0
    if isinstance(left, int | float) and isinstance(right, int | float):
        return float(abs(left - right))
    if left != right:
        raise ValueError(f"Value mismatch: {left!r} vs {right!r}")
    return 0.0
