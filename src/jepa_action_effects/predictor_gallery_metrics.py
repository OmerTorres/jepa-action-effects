"""Pure, family-weighted predictor-to-rendered-gallery measurements."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from jepa_action_effects.gallery_metrics import query_score

REGIMES = ("neither", "exactly_one", "both")
FIELDS = (
    "matched_margin",
    "persistence_margin",
    "wrong_margin",
    "persistence_advantage",
    "action_advantage",
    "matched_accuracy",
    "persistence_accuracy",
    "wrong_accuracy",
    "matched_d_true",
)


def candidate_scores(
    predictions: np.ndarray, persistence: np.ndarray, bank: np.ndarray, eta: float
) -> dict:
    """Score one outcome bank against all six whole-sequence predictions."""
    predictions = np.asarray(predictions)
    if predictions.shape != (6, 192):
        raise ValueError("Expected six projected terminal predictions")
    predicted = [query_score(vector, bank, eta) for vector in predictions]
    old = query_score(persistence, bank, eta)
    return {"predictions": predicted, "persistence": old}


def endpoint_metrics(row: dict, index: int) -> dict:
    predicted = row["predictions"]
    matched = predicted[index]
    wrong = [item for j, item in enumerate(predicted) if j != index]
    persistence = row["persistence"]
    result = {
        "matched_margin": matched["margin"],
        "persistence_margin": persistence["margin"],
        "wrong_margin": float(np.mean([item["margin"] for item in wrong])),
        "matched_accuracy": float(matched["correct"]),
        "persistence_accuracy": float(persistence["correct"]),
        "wrong_accuracy": float(np.mean([item["correct"] for item in wrong])),
        "matched_d_true": matched["d_true"],
    }
    result["persistence_advantage"] = result["matched_margin"] - result["persistence_margin"]
    result["action_advantage"] = result["matched_margin"] - result["wrong_margin"]
    return result


def family_mean(endpoints: list[dict], indices: set[int] | None = None) -> dict | None:
    selected = [
        endpoint_metrics(row, index)
        for index, row in enumerate(endpoints)
        if row.get("eligible") and (indices is None or index in indices)
    ]
    if not selected:
        return None
    return {
        "count": len(selected),
        **{key: float(np.mean([row[key] for row in selected])) for key in FIELDS},
    }


def participants(labels: list[bool], regime: str) -> set[int]:
    if len(labels) != 6 or regime not in REGIMES:
        raise ValueError("Expected six labels and a contact regime")
    return {
        k
        for i in range(6)
        for j in range(i + 1, 6)
        if REGIMES[int(labels[i]) + int(labels[j])] == regime
        for k in (i, j)
    }


def derangement_expectation(scores: np.ndarray) -> float:
    """Reference implementation used by tests; operational path uses off diagonal."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (6, 6):
        raise ValueError("Expected six targets and six predictions")
    arrangements = [p for p in permutations(range(6)) if all(p[i] != i for i in range(6))]
    if len(arrangements) != 265:
        raise ValueError("Derangement count changed")
    return float(np.mean([np.mean([values[i, p[i]] for i in range(6)]) for p in arrangements]))


def aggregate(rows: list[dict], seed: int, check=lambda: None) -> dict:
    """Bootstrap complete families; preserve cohort counts in pooled draws."""
    groups = {}
    for name in ("challenge", "ordinary", "pooled"):
        selected = [
            row
            for row in rows
            if row["summary"] is not None and (name == "pooled" or row["cohort"] == name)
        ]
        if not selected:
            groups[name] = {"status": "unassessable", "families": 0}
            continue
        rng = np.random.default_rng(seed + {"challenge": 0, "ordinary": 1, "pooled": 2}[name])
        metrics = {}
        for field in FIELDS:
            values = np.array([row["summary"][field] for row in selected], dtype=np.float64)
            cohorts = [
                np.array([i for i, row in enumerate(selected) if row["cohort"] == cohort])
                for cohort in ("challenge", "ordinary")
            ]
            draws = []
            for j in range(2000):
                if j % 100 == 0:
                    check()
                indices = np.concatenate([rng.choice(ids, len(ids)) for ids in cohorts if len(ids)])
                draws.append(float(np.mean(values[indices])))
            interval = np.percentile(draws, (2.5, 97.5))
            metrics[field] = {"mean": float(np.mean(values)), "ci95": interval.tolist()}
        groups[name] = {
            "status": "assessable" if len(selected) >= 10 else "unassessable",
            "families": len(selected),
            "endpoints": sum(row["summary"]["count"] for row in selected),
            "metrics": metrics,
        }
    return groups


def latent_summary(records: list[dict], seed: int, check=lambda: None) -> dict:
    """Reuse LATENT-001 family scores with cohort-preserving intervals."""
    output = {}
    for group in ("challenge", "ordinary", "pooled"):
        selected = [row for row in records if group == "pooled" or row["cohort"] == group]
        cohorts = [
            np.array([i for i, row in enumerate(selected) if row["cohort"] == cohort])
            for cohort in ("challenge", "ordinary")
        ]
        rng = np.random.default_rng(seed + {"challenge": 3, "ordinary": 4, "pooled": 5}[group])
        output[group] = {}
        for key in ("L", "W", "A", "P", "B"):
            values = np.array([row["latent"][key] for row in selected], dtype=np.float64)
            draws = []
            for j in range(2000):
                if j % 100 == 0:
                    check()
                indices = np.concatenate([rng.choice(ids, len(ids)) for ids in cohorts if len(ids)])
                draws.append(float(np.mean(values[indices])))
            output[group][key] = {
                "mean": float(np.mean(values)),
                "ci95": np.percentile(draws, (2.5, 97.5)).tolist(),
                "families": len(selected),
            }
    return output
