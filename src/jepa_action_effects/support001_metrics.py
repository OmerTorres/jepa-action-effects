"""SUPPORT-001 estimands (D036): archive support map and competing distortion accounts.

All geometry is expressed in the block frame (rotation by −block angle), so a query is
compared with archive windows that put the agent in the same place relative to the T.

Accounts map the true displacement t (and support covariates) to the decoded predicted
displacement p:
- N: p = κ·t
- A: p = clip(α + β‖a‖, 0, 2)·t              (attenuation by command magnitude)
- B: p = w·t + (1 − w)·m(s), w = σ(θ₀ + θ₁ℓ)  (reversion toward the typical transition)
- C: p = min(1, c/‖a‖)·t                     (clamp at a support-edge command)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors

K = 512
RADIUS = 0.1
ACCOUNTS = ("N", "A", "B", "C")


def rotate(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate (N,2) vectors by `angle` (N,) radians."""
    v = np.asarray(vectors, float)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


def block_frame(agent, agent_previous, block, angle, command, displacement=None):
    """State features (agent offset, agent recent motion) and vectors in the block frame."""
    angle = np.asarray(angle, float)
    offset = rotate(np.asarray(agent, float) - np.asarray(block, float), -angle)
    motion = rotate(np.asarray(agent, float) - np.asarray(agent_previous, float), -angle)
    features = np.column_stack([offset, motion])
    out = {"s": features, "a": rotate(command, -angle)}
    if displacement is not None:
        out["d"] = rotate(displacement, -angle)
    return out


@dataclass
class SupportMap:
    """kNN over archive windows: typical displacement m(s) and log command support ℓ."""

    s: np.ndarray
    a: np.ndarray
    d: np.ndarray
    k: int = K
    radius: float = RADIUS

    def __post_init__(self) -> None:
        self.s = np.asarray(self.s, float)
        self.a = np.asarray(self.a, float)
        self.d = np.asarray(self.d, float)
        self.mean = self.s.mean(axis=0)
        self.scale = self.s.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self._index = NearestNeighbors(n_neighbors=self.k).fit((self.s - self.mean) / self.scale)

    def query(self, s: np.ndarray, a: np.ndarray, angle: np.ndarray) -> dict[str, np.ndarray]:
        """m(s) in the world frame, ℓ, and the neighbourhood's command-magnitude quantile."""
        distance, index = self._index.kneighbors((np.asarray(s, float) - self.mean) / self.scale)
        a = np.asarray(a, float)
        near = np.linalg.norm(self.a[index] - a[:, None, :], axis=2) <= self.radius
        count = near.sum(axis=1)
        magnitudes = np.linalg.norm(self.a[index], axis=2)
        quantile = (magnitudes < np.linalg.norm(a, axis=1)[:, None]).mean(axis=1)
        return {
            "m": rotate(self.d[index].mean(axis=1), np.asarray(angle, float)),
            "log_support": np.log((count + 0.5) / (self.k + 1)),
            "count": count,
            "magnitude_quantile": quantile,
            "neighbour_distance": distance[:, -1],
        }


# --------------------------------------------------------------------------- accounts
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(account: str, params: Sequence[float], t, m, log_support, magnitude) -> np.ndarray:
    t = np.asarray(t, float)
    magnitude = np.asarray(magnitude, float)
    if account == "N":
        return params[0] * t
    if account == "A":
        gain = np.clip(params[0] + params[1] * magnitude, 0.0, 2.0)
        return gain[:, None] * t
    if account == "B":
        w = _sigmoid(params[0] + params[1] * np.asarray(log_support, float))[:, None]
        return w * t + (1 - w) * np.asarray(m, float)
    if account == "C":
        gain = np.minimum(1.0, params[0] / np.maximum(magnitude, 1e-9))
        return gain[:, None] * t
    raise ValueError(account)


INITIAL = {"N": [1.0], "A": [1.0, -0.5], "B": [2.0, 0.5], "C": [0.5]}
BOUNDS = {
    "N": ([0.0], [2.0]),
    "A": ([-2.0, -5.0], [3.0, 5.0]),
    "B": ([-20.0, -10.0], [20.0, 10.0]),
    "C": ([0.01], [2.0]),
}


def fit(account: str, p, t, m, log_support, magnitude) -> dict[str, Any]:
    """Least squares on the stacked 2-D residual p − f(t, ...)."""
    p = np.asarray(p, float)

    def residual(params):
        return (p - predict(account, params, t, m, log_support, magnitude)).ravel()

    result = least_squares(residual, INITIAL[account], bounds=BOUNDS[account])
    return {
        "params": [float(x) for x in result.x],
        "mean_squared_error": float(np.mean(np.sum(residual(result.x).reshape(-1, 2) ** 2, 1))),
        "converged": bool(result.success),
    }


def squared_errors(fits: Mapping[str, Mapping[str, Any]], p, t, m, log_support, magnitude):
    """Per-unit ‖p − f_X‖² for every account with frozen parameters."""
    p = np.asarray(p, float)
    return {
        name: np.sum((p - predict(name, fit_["params"], t, m, log_support, magnitude)) ** 2, axis=1)
        for name, fit_ in fits.items()
    }


def decide(boot: Mapping[str, Mapping[str, Any]], accounts: Sequence[str] = ACCOUNTS) -> dict:
    """An account is established if X − Y has a CI below 0 for every other Y."""
    winners = []
    for x in accounts:
        if all(
            (ci := boot[f"{x}_minus_{y}"]["ci"]) is not None and ci[1] < 0
            for y in accounts
            if y != x
        ):
            winners.append(x)
    return {
        "verdict": f"account_{winners[0]}_established" if winners else "no_account_established",
        "winners": winners,
    }


def pairwise(errors: Mapping[str, np.ndarray], accounts: Sequence[str] = ACCOUNTS):
    return {
        f"{x}_minus_{y}": np.asarray(errors[x]) - np.asarray(errors[y])
        for x in accounts
        for y in accounts
        if x != y
    }


def slope(p: np.ndarray, t: np.ndarray) -> float | None:
    p, t = np.asarray(p, float), np.asarray(t, float)
    denominator = float(np.sum(t * t))
    return float(np.sum(p * t) / denominator) if denominator > 0 else None
