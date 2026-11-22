"""ROBUST-001 instruments: a kNN block readout and an off-manifold score (exploratory, D034).

The kNN readout averages the block position of the k nearest real encoded frames. It
cannot leave the bank's label range, so it clamps out-of-range predictions and can itself
manufacture reversion. That is the point: its failure mode differs from the ridge
readout's (linear extrapolation) and the MLPs' (saturation). Agreement across them is
corroboration, not validation on predicted latents (D031).

The off-manifold score is d₁, the distance from a latent to its nearest bank latent,
compared with d₁ of encoded real frames from the same group.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

K_GRID = (1, 2, 4, 8, 16, 32, 64)
QUANTILE = 95.0


@dataclass
class KnnReadout:
    """Mean label of the k nearest bank latents (uniform weights, Euclidean)."""

    bank: np.ndarray
    labels: np.ndarray
    k: int

    def __post_init__(self) -> None:
        self.bank = np.asarray(self.bank, np.float64)
        self.labels = np.asarray(self.labels, np.float64)
        if len(self.bank) != len(self.labels):
            raise ValueError("bank and labels need the same number of rows")
        if not 1 <= self.k <= len(self.bank):
            raise ValueError("k must be between 1 and the bank size")
        self._index = NearestNeighbors(algorithm="brute").fit(self.bank)

    def neighbours(self, queries: np.ndarray, k: int | None = None):
        k = self.k if k is None else k
        return self._index.kneighbors(np.asarray(queries, np.float64), n_neighbors=k)

    def predict(self, queries: np.ndarray) -> np.ndarray:
        _, index = self.neighbours(queries)
        return self.labels[index].mean(axis=1)

    def nearest_distance(self, queries: np.ndarray) -> np.ndarray:
        distance, _ = self.neighbours(queries, 1)
        return distance[:, 0]


def select_k(
    bank: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    grid: Sequence[int] = K_GRID,
) -> dict:
    """Validation mean Euclidean error per k; the smallest error wins (ties: smaller k)."""
    largest = max(grid)
    readout = KnnReadout(bank, labels, largest)
    _, index = readout.neighbours(queries, largest)
    truth = np.asarray(truth, np.float64)
    errors = {}
    for k in sorted(grid):
        prediction = readout.labels[index[:, :k]].mean(axis=1)
        errors[k] = float(np.mean(np.linalg.norm(prediction - truth, axis=1)))
    best = min(sorted(errors), key=lambda k: errors[k])
    return {"k": best, "validation_mean_error": errors}


def offmanifold_threshold(reference_distances: np.ndarray, quantile: float = QUANTILE) -> float:
    """The `quantile` percentile of d₁ over real latents of the same group."""
    reference = np.asarray(reference_distances, float)
    if not len(reference):
        raise ValueError("a reference group needs at least one real latent")
    return float(np.percentile(reference, quantile))


def max_pairwise(points: np.ndarray) -> np.ndarray:
    """Largest pairwise Euclidean distance among instruments, (N, I, 2) -> (N,)."""
    points = np.asarray(points, float)
    difference = points[:, :, None, :] - points[:, None, :, :]
    return np.linalg.norm(difference, axis=-1).max(axis=(1, 2))


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left, right = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.ptp(left[keep]) == 0 or np.ptp(right[keep]) == 0:
        return None
    return float(spearmanr(left[keep], right[keep]).statistic)
