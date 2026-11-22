"""Family-weighted linear and bounded nonlinear state readouts for PILOT-001."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
NONLINEAR_SEEDS = (20260931, 20260932, 20260933)


@dataclass(frozen=True)
class LinearReadouts:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float
    block_coef: np.ndarray
    block_intercept: np.ndarray
    agent_coef: np.ndarray
    agent_intercept: np.ndarray
    orientation_coef: np.ndarray
    orientation_intercept: np.ndarray

    def _features(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        return (values - self.feature_mean) / self.feature_scale

    def predict(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        features = self._features(embeddings)
        return {
            "block": features @ self.block_coef.T + self.block_intercept,
            "agent": features @ self.agent_coef.T + self.agent_intercept,
            "orientation_sincos": (features @ self.orientation_coef.T + self.orientation_intercept),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "alpha": np.asarray(self.alpha),
            "block_coef": self.block_coef,
            "block_intercept": self.block_intercept,
            "agent_coef": self.agent_coef,
            "agent_intercept": self.agent_intercept,
            "orientation_coef": self.orientation_coef,
            "orientation_intercept": self.orientation_intercept,
        }


def _targets(states: np.ndarray) -> dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError("Readout states must have shape (N,5)")
    return {
        "agent": states[:, :2],
        "block": states[:, 2:4],
        "orientation": np.column_stack((np.sin(states[:, 4]), np.cos(states[:, 4]))),
    }


def fit_linear(
    embeddings: np.ndarray,
    states: np.ndarray,
    family_ids: np.ndarray,
    alpha: float,
) -> LinearReadouts:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 192 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite with shape (N,192)")
    families = np.asarray(family_ids)
    if len(values) != len(states) or len(values) != len(families):
        raise ValueError("Embeddings, states, and family IDs must align")
    unique, counts = np.unique(families, return_counts=True)
    count_by_family = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_by_family[value] for value in families], dtype=np.float64)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    scale[scale == 0] = 1.0
    features = (values - mean) / scale
    labels = _targets(states)

    def fit(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = Ridge(alpha=alpha)
        model.fit(features, target, sample_weight=weights)
        return np.asarray(model.coef_), np.asarray(model.intercept_)

    block_coef, block_intercept = fit(labels["block"])
    agent_coef, agent_intercept = fit(labels["agent"])
    orientation_coef, orientation_intercept = fit(labels["orientation"])
    return LinearReadouts(
        mean,
        scale,
        float(alpha),
        block_coef,
        block_intercept,
        agent_coef,
        agent_intercept,
        orientation_coef,
        orientation_intercept,
    )


def select_ridge(
    train_embeddings: np.ndarray,
    train_states: np.ndarray,
    train_family_ids: np.ndarray,
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[LinearReadouts, dict[str, float]]:
    scores: dict[str, float] = {}
    models = {}
    for alpha in RIDGE_ALPHAS:
        model = fit_linear(train_embeddings, train_states, train_family_ids, alpha)
        family_errors = []
        for embeddings, states in validation.values():
            predicted = model.predict(embeddings)["block"]
            family_errors.append(float(np.mean(np.linalg.norm(predicted - states[:, 2:4], axis=1))))
        score = float(np.mean(family_errors))
        scores[str(alpha)] = score
        models[alpha] = model
    selected = min(RIDGE_ALPHAS, key=lambda value: (scores[str(value)], -value))
    return models[selected], scores


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    return float(
        sorted_values[min(np.searchsorted(cumulative, threshold, side="left"), len(values) - 1)]
    )


def readout_gate(
    family_errors: dict[str, np.ndarray],
    family_contacts: dict[str, np.ndarray],
    delta: float,
) -> dict[str, Any]:
    """Evaluate overall and pair-regime gates with equal total weight per family."""
    result: dict[str, Any] = {}
    regimes: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in ("neither", "exactly_one", "both")
    }
    for family_id, errors in family_errors.items():
        contacts = family_contacts[family_id]
        participants = {name: set() for name in regimes}
        for left in range(len(contacts)):
            for right in range(left + 1, len(contacts)):
                regime = ("neither", "exactly_one", "both")[
                    int(contacts[left]) + int(contacts[right])
                ]
                participants[regime].update((left, right))
        for regime, indices in participants.items():
            if indices:
                regimes[regime][family_id] = errors[sorted(indices)]

    def summarize(groups: dict[str, np.ndarray]) -> dict[str, Any]:
        if not groups:
            return {
                "family_count": 0,
                "mean_error": None,
                "p95_error": None,
                "sufficient_families": False,
                "passed": False,
            }
        values = np.concatenate(list(groups.values()))
        weights = np.concatenate(
            [np.full(len(errors), 1.0 / len(errors)) for errors in groups.values()]
        )
        weights /= len(groups)
        return {
            "family_count": len(groups),
            "mean_error": float(np.sum(values * weights)),
            "p95_error": weighted_quantile(values, weights, 0.95),
            "sufficient_families": len(groups) >= 10,
            "passed": bool(
                len(groups) >= 10
                and np.sum(values * weights) <= delta / 4
                and weighted_quantile(values, weights, 0.95) <= delta / 2
            ),
        }

    result["overall"] = summarize(family_errors)
    for regime, groups in regimes.items():
        result[regime] = summarize(groups)
    result["passed"] = bool(all(item["passed"] for item in result.values()))
    result["thresholds"] = {"mean": delta / 4, "p95": delta / 2, "minimum_families": 10}
    return result


class NonlinearReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(self, inputs):
        return self.network(inputs)


def fit_nonlinear(
    train_embeddings: np.ndarray,
    train_states: np.ndarray,
    train_family_ids: np.ndarray,
    validation: dict[str, tuple[np.ndarray, np.ndarray]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    seed: int,
) -> tuple[NonlinearReadout, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    model = NonlinearReadout()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    features = torch.tensor((train_embeddings - feature_mean) / feature_scale, dtype=torch.float32)
    labels_np = _targets(train_states)
    raw_labels = np.column_stack((labels_np["block"], labels_np["agent"], labels_np["orientation"]))
    target_mean = raw_labels.mean(axis=0)
    target_scale = raw_labels.std(axis=0)
    target_scale[target_scale == 0] = 1.0
    labels = torch.tensor((raw_labels - target_mean) / target_scale, dtype=torch.float32)
    families = np.asarray(train_family_ids)
    unique, counts = np.unique(families, return_counts=True)
    count_by_family = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = torch.tensor(
        [1.0 / count_by_family[value] for value in families], dtype=torch.float32
    )
    weights /= weights.sum()
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        squared = torch.mean((model(features) - labels) ** 2, dim=1)
        loss = torch.sum(squared * weights)
        loss.backward()
        optimizer.step()
        model.eval()
        family_errors = []
        with torch.inference_mode():
            for embeddings, states in validation.values():
                inputs = torch.tensor(
                    (embeddings - feature_mean) / feature_scale, dtype=torch.float32
                )
                block = model(inputs)[:, :2].numpy() * target_scale[:2] + target_mean[:2]
                family_errors.append(float(np.mean(np.linalg.norm(block - states[:, 2:4], axis=1))))
        score = float(np.mean(family_errors))
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return (
        model,
        target_mean,
        target_scale,
        {"seed": seed, "best_epoch": best_epoch, "validation_mean_error": best_score},
    )
