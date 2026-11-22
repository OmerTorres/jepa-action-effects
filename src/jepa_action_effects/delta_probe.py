"""Zero-anchored displacement readouts and family-level measurement gates."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from jepa_action_effects.latent_cache import array_digest
from jepa_action_effects.pilot_readouts import weighted_quantile

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEEDS = (20260931, 20260932, 20260933)
REGIMES = ("neither", "exactly_one", "both")
REQUIRED = (
    "endpoint_overall",
    "endpoint_challenge",
    "endpoint_ordinary",
    "endpoint_neither",
    "endpoint_exactly_one",
    "endpoint_both",
    "effect_overall",
    "effect_neither",
    "effect_exactly_one",
    "effect_both",
    "effect_meaningful",
)


def transition_targets(family: dict, embeddings: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.asarray(embeddings["history"][-1], dtype=np.float64)
    future = np.asarray(embeddings["future"][:, -1], dtype=np.float64)
    if history.shape != (192,) or future.shape != (6, 192):
        raise ValueError("Expected projected CLS at one history and six two-step endpoints")
    start = np.asarray(family["history_states"][-1, 2:4], dtype=np.float64)
    truth = np.asarray(family["future_states"][:, -1, 2:4], dtype=np.float64) - start
    if truth.shape != (6, 2):
        raise ValueError("Endpoint displacement label alignment changed")
    return history, future, truth


def deduplicated_training(
    families: dict, embeddings: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    contexts, differences, targets, ids = [], [], [], []
    duplicates = conflicts = 0
    for family_id in sorted(families):
        family = families[family_id]
        start, future, truth = transition_targets(family, embeddings[family_id])
        seen: dict[tuple[str, str], np.ndarray] = {}
        history_digest = array_digest(family["history_frames"][-1])
        for index in range(6):
            key = (history_digest, array_digest(family["future_frames"][index, -1]))
            if key in seen:
                duplicates += 1
                if not np.allclose(seen[key], truth[index], rtol=0, atol=1e-6):
                    conflicts += 1
                continue
            seen[key] = truth[index]
            contexts.append(start)
            differences.append(future[index] - start)
            targets.append(truth[index])
            ids.append(family_id)
    if conflicts:
        raise ValueError(
            f"Conflicting block displacement labels for {conflicts} identical frame pairs"
        )
    return (
        np.asarray(contexts),
        np.asarray(differences),
        np.asarray(targets),
        np.asarray(ids),
        {"retained_pairs": len(ids), "duplicate_pairs": duplicates, "conflicts": conflicts},
    )


def family_weights(ids: np.ndarray) -> np.ndarray:
    names, counts = np.unique(ids, return_counts=True)
    if len(names) == 0:
        raise ValueError("No fit families")
    by_id = dict(zip(names.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1 / by_id[name] for name in ids], dtype=np.float64)
    return weights / weights.sum()


@dataclass(frozen=True)
class DirectRidge:
    scale: np.ndarray
    coef: np.ndarray
    alpha: float

    def predict(self, start: np.ndarray, future: np.ndarray) -> np.ndarray:
        difference = np.asarray(future) - np.asarray(start)
        return (difference / self.scale) @ self.coef.T

    def arrays(self) -> dict[str, np.ndarray]:
        return {"scale": self.scale, "coef": self.coef, "alpha": np.asarray(self.alpha)}


def fit_ridge(
    differences: np.ndarray, targets: np.ndarray, ids: np.ndarray, alpha: float
) -> DirectRidge:
    weights = family_weights(ids)
    scale = np.sqrt(np.average(differences**2, axis=0, weights=weights))
    scale[scale == 0] = 1
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(
        differences / scale,
        targets,
        sample_weight=weights * len(np.unique(ids)),
    )
    return DirectRidge(scale, np.asarray(model.coef_), float(alpha))


def select_ridge(
    differences: np.ndarray, targets: np.ndarray, ids: np.ndarray, tuning: dict
) -> tuple[DirectRidge, dict]:
    models, scores = {}, {}
    for alpha in ALPHAS:
        fitted = fit_ridge(differences, targets, ids, alpha)
        per_family = []
        for family, embeddings in tuning.values():
            start, future, truth = transition_targets(family, embeddings)
            per_family.append(
                float(np.mean(np.linalg.norm(fitted.predict(start, future) - truth, axis=1)))
            )
        models[alpha] = fitted
        scores[str(alpha)] = float(np.mean(per_family))
    chosen = min(ALPHAS, key=lambda alpha: (scores[str(alpha)], -alpha))
    return models[chosen], scores


class AnchoredNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, context: torch.Tensor, difference: torch.Tensor) -> torch.Tensor:
        active = self.network(torch.cat((context, difference), dim=-1))
        zero = self.network(torch.cat((context, torch.zeros_like(difference)), dim=-1))
        return active - zero


def _summary(groups: dict[str, list[float]]) -> dict[str, Any]:
    if not groups:
        return {
            "family_count": 0,
            "observation_count": 0,
            "mean": None,
            "p95": None,
            "status": "unassessable",
        }
    values = np.concatenate([np.asarray(value) for value in groups.values()])
    weights = np.concatenate(
        [np.full(len(value), 1 / (len(groups) * len(value))) for value in groups.values()]
    )
    mean = float(np.sum(values * weights))
    p95 = weighted_quantile(values, weights, 0.95)
    status = (
        "unassessable" if len(groups) < 10 else "pass" if mean <= 1.28 and p95 <= 2.56 else "fail"
    )
    return {
        "family_count": len(groups),
        "observation_count": len(values),
        "mean": mean,
        "p95": p95,
        "status": status,
    }


def measurement_gate(data: dict[str, tuple[dict, dict]], predict, delta: float = 5.12) -> dict:
    groups: dict[str, dict[str, list[float]]] = {key: {} for key in REQUIRED}
    for family_id, (family, embeddings) in data.items():
        start, future, truth = transition_targets(family, embeddings)
        predicted = np.asarray(predict(start, future), dtype=np.float64)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError("Readout displacement shape or values are invalid")
        contact = np.asarray(family["contacts"], dtype=bool)
        if contact.shape != (6,):
            raise ValueError("Contact labels must align with six candidates")
        errors = np.linalg.norm(predicted - truth, axis=1)
        cohort = "challenge" if family_id.split(":", 1)[-1].startswith("challenge") else "ordinary"
        unique = {}
        for index in range(6):
            digest = array_digest(family["future_frames"][index, -1])
            if digest in unique and not np.allclose(
                truth[unique[digest]], truth[index], rtol=0, atol=1e-6
            ):
                raise ValueError("Conflicting displacement labels on identical endpoint frames")
            unique.setdefault(digest, index)
        groups["endpoint_overall"][family_id] = errors[list(unique.values())].tolist()
        groups[f"endpoint_{cohort}"][family_id] = errors[list(unique.values())].tolist()
        participants = {key: set() for key in REGIMES}
        pair_errors = {key: [] for key in REGIMES}
        meaningful_errors = []
        all_pair_errors = []
        for left, right in itertools.combinations(range(6), 2):
            regime = REGIMES[int(contact[left]) + int(contact[right])]
            participants[regime].update((left, right))
            effect = truth[left] - truth[right]
            error = float(np.linalg.norm((predicted[left] - predicted[right]) - effect))
            pair_errors[regime].append(error)
            all_pair_errors.append(error)
            if np.linalg.norm(effect) > delta:
                meaningful_errors.append(error)
        groups["effect_overall"][family_id] = all_pair_errors
        if meaningful_errors:
            groups["effect_meaningful"][family_id] = meaningful_errors
        for regime in REGIMES:
            if participants[regime]:
                by_frame = {}
                for index in sorted(participants[regime]):
                    by_frame.setdefault(array_digest(family["future_frames"][index, -1]), index)
                groups[f"endpoint_{regime}"][family_id] = errors[list(by_frame.values())].tolist()
                groups[f"effect_{regime}"][family_id] = pair_errors[regime]
    summaries = {key: _summary(group) for key, group in groups.items()}
    statuses = {key: summaries[key]["status"] for key in REQUIRED}
    overall = (
        "unassessable"
        if "unassessable" in statuses.values()
        else "pass"
        if all(value == "pass" for value in statuses.values())
        else "fail"
    )
    return {
        "status": overall,
        "required": summaries,
        "thresholds": {
            "mean": 1.28,
            "p95": 2.56,
            "minimum_families": 10,
            "meaningful_delta": delta,
        },
    }
