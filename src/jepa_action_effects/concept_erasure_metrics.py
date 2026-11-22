"""Pure weighted metrics and linear concept erasure for ERASURE-001."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def family_weights(family_ids: np.ndarray) -> np.ndarray:
    """Give every family equal total weight, divided across its rows."""
    ids = np.asarray(family_ids)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("Family IDs must be a nonempty vector")
    unique, counts = np.unique(ids, return_counts=True)
    count_by_id = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_by_id[item] for item in ids], dtype=np.float64)
    return weights / weights.sum()


def weighted_covariance(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum()
    return (values * normalized[:, None]).T @ targets


def _symmetric_powers(matrix: np.ndarray, rtol: float) -> tuple[np.ndarray, np.ndarray, int]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    cutoff = float(max(values[-1], 0.0) * rtol) if len(values) else 0.0
    keep = values > cutoff
    sqrt = (vectors[:, keep] * np.sqrt(values[keep])) @ vectors[:, keep].T
    inverse_sqrt = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    return sqrt, inverse_sqrt, int(np.sum(keep))


@dataclass(frozen=True)
class LinearEraser:
    mean: np.ndarray
    matrix: np.ndarray
    removed_rank: int
    covariance_rank: int
    rtol: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return (values - self.mean) @ self.matrix.T + self.mean


def fit_leace(
    x: np.ndarray,
    concept: np.ndarray,
    family_ids: np.ndarray,
    rtol: float = 1e-10,
) -> tuple[LinearEraser, dict[str, float | int]]:
    """Fit the continuous-label LEACE affine map in the Euclidean metric.

    With W = Cov(X)^(-1/2), LEACE removes the projection onto
    col(W Cov(X, Z)): E = I - W^+ P W. The finite-sample fit guarantees zero
    weighted linear cross-covariance up to numerical precision.
    """
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(concept, dtype=np.float64)
    if values.ndim != 2 or labels.ndim != 2 or len(values) != len(labels):
        raise ValueError("Features and concepts must be aligned matrices")
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("Features and concepts must be finite")
    weights = family_weights(family_ids)
    mean = np.sum(values * weights[:, None], axis=0)
    label_mean = np.sum(labels * weights[:, None], axis=0)
    centered = values - mean
    centered_labels = labels - label_mean
    covariance = weighted_covariance(centered, centered, weights)
    cross = weighted_covariance(centered, centered_labels, weights)
    sqrt, inverse_sqrt, covariance_rank = _symmetric_powers(covariance, rtol)
    directions = inverse_sqrt @ cross
    u, singular, _ = np.linalg.svd(directions, full_matrices=False)
    direction_cutoff = float((singular[0] if len(singular) else 0.0) * rtol)
    removed_rank = int(np.sum(singular > direction_cutoff))
    projector = u[:, :removed_rank] @ u[:, :removed_rank].T
    matrix = np.eye(values.shape[1]) - sqrt @ projector @ inverse_sqrt
    eraser = LinearEraser(mean, matrix, removed_rank, covariance_rank, rtol)
    transformed = eraser.transform(values)
    residual = weighted_covariance(transformed - mean, centered_labels, weights)
    diagnostics = {
        "removed_rank": removed_rank,
        "covariance_rank": covariance_rank,
        "covariance_min_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
        "covariance_max_eigenvalue": float(np.linalg.eigvalsh(covariance)[-1]),
        "idempotence_max_abs": float(np.max(np.abs(matrix @ matrix - matrix))),
        "cross_covariance_before_fro": float(np.linalg.norm(cross)),
        "cross_covariance_after_fro": float(np.linalg.norm(residual)),
    }
    return eraser, diagnostics


def random_eraser(
    mean: np.ndarray, dimension: int, rank: int, rng: np.random.Generator
) -> LinearEraser:
    if not 0 <= rank <= dimension:
        raise ValueError("Invalid removal rank")
    directions = rng.normal(size=(dimension, rank))
    q = np.linalg.qr(directions, mode="reduced")[0]
    matrix = np.eye(dimension) - q @ q.T
    return LinearEraser(np.asarray(mean), matrix, rank, dimension, 0.0)


@dataclass(frozen=True)
class PositionProbe:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return values @ self.coefficient.T + self.intercept


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    family_ids: np.ndarray,
    alpha: float,
) -> PositionProbe:
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    weights = family_weights(family_ids)
    mean = np.sum(values * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((values - mean) ** 2 * weights[:, None], axis=0))
    scale[scale == 0] = 1.0
    model = Ridge(alpha=alpha)
    model.fit((values - mean) / scale, labels, sample_weight=weights)
    return PositionProbe(mean, scale, model.coef_, model.intercept_, float(alpha))


def family_mse(
    prediction: np.ndarray, target: np.ndarray, family_ids: np.ndarray
) -> dict[str, float]:
    squared = (np.asarray(prediction) - np.asarray(target)) ** 2
    weights = family_weights(family_ids)
    return {
        "joint": float(np.sum(np.mean(squared, axis=1) * weights)),
        "x": float(np.sum(squared[:, 0] * weights)),
        "y": float(np.sum(squared[:, 1] * weights)),
    }


def select_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_ids: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    calibration_ids: np.ndarray,
) -> tuple[PositionProbe, dict[str, float]]:
    models = {}
    scores = {}
    for alpha in RIDGE_ALPHAS:
        model = fit_probe(train_x, train_y, train_ids, alpha)
        score = family_mse(model.predict(calibration_x), calibration_y, calibration_ids)["joint"]
        models[alpha] = model
        scores[str(alpha)] = score
    chosen = min(RIDGE_ALPHAS, key=lambda alpha: (scores[str(alpha)], -alpha))
    return models[chosen], scores


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(ordered[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def score_probe(
    probe: PositionProbe,
    x: np.ndarray,
    y: np.ndarray,
    family_ids: np.ndarray,
    constant: np.ndarray,
) -> dict[str, float | dict[str, float] | None]:
    prediction = probe.predict(x)
    errors = np.linalg.norm(prediction - y, axis=1)
    weights = family_weights(family_ids)
    probe_mse = family_mse(prediction, y, family_ids)
    baseline_mse = family_mse(np.broadcast_to(constant, y.shape), y, family_ids)
    gain = None if baseline_mse["joint"] == 0 else 1 - probe_mse["joint"] / baseline_mse["joint"]
    return {
        "gain": None if gain is None else float(gain),
        "probe_mse": probe_mse,
        "constant_mse": baseline_mse,
        "mean_euclidean_error": float(np.sum(errors * weights)),
        "p95_euclidean_error": weighted_quantile(errors, weights, 0.95),
    }


def bootstrap_gain(
    prediction: np.ndarray,
    target: np.ndarray,
    family_ids: np.ndarray,
    constant: np.ndarray,
    seed: int,
    samples: int = 2000,
    strata: np.ndarray | None = None,
) -> list[float]:
    ids = np.asarray(family_ids)
    unique = np.unique(ids)
    if strata is None:
        stratum_by_family = {family: "all" for family in unique}
    else:
        strata = np.asarray(strata)
        if len(strata) != len(ids):
            raise ValueError("Bootstrap strata and family IDs must align")
        stratum_by_family = {}
        for family in unique:
            values = np.unique(strata[ids == family])
            if len(values) != 1:
                raise ValueError("Each family must belong to exactly one bootstrap stratum")
            stratum_by_family[family] = values[0]
    pools = {
        stratum: np.asarray([family for family in unique if stratum_by_family[family] == stratum])
        for stratum in set(stratum_by_family.values())
    }
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(samples):
        chosen = np.concatenate(
            [rng.choice(pool, size=len(pool), replace=True) for pool in pools.values()]
        )
        probe_errors = []
        constant_errors = []
        for family in chosen:
            mask = ids == family
            probe_errors.append(float(np.mean((prediction[mask] - target[mask]) ** 2)))
            constant_errors.append(float(np.mean((target[mask] - constant) ** 2)))
        denominator = float(np.mean(constant_errors))
        result.append(
            np.nan if denominator == 0 else 1 - float(np.mean(probe_errors)) / denominator
        )
    return result


def interval(values: list[float]) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return np.quantile(finite, [0.025, 0.975]).tolist()
