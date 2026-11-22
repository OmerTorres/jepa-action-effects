"""Pure gallery scoring and equal-family reductions for GALLERY-001."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def query_score(query: np.ndarray, templates: np.ndarray, eta: float) -> dict:
    """Template zero is the true position; remaining templates are negatives."""
    q = np.asarray(query, dtype=np.float64)
    bank = np.asarray(templates, dtype=np.float64)
    if q.shape != (192,) or bank.shape != (5, 192) or not np.isfinite(bank).all():
        raise ValueError("Expected one finite 192-vector and five finite templates")
    if not np.isfinite(q).all() or not np.isfinite(eta) or eta <= 0:
        raise ValueError("Query and numerical floor must be finite")
    distances = np.mean((bank - q) ** 2, axis=1)
    true = float(distances[0])
    wrong = float(np.min(distances[1:]))
    margin = wrong - true
    return {
        "distances": distances.tolist(),
        "d_true": true,
        "d_wrong": wrong,
        "margin": margin,
        "ratio": margin / max(wrong, eta),
        "correct": bool(margin > eta),
        "tie": bool(abs(margin) <= eta),
        "collapsed": bool(wrong <= eta),
    }


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    numbers = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if numbers.ndim != 1 or mass.shape != numbers.shape or not len(numbers):
        raise ValueError("Quantile requires nonempty aligned vectors")
    if not (0 <= q <= 1) or not np.isfinite(numbers).all():
        raise ValueError("Invalid quantile or values")
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Quantile weights must be finite and positive")
    order = np.argsort(numbers, kind="stable")
    ranked = numbers[order]
    cumulative = np.cumsum(mass[order]) / np.sum(mass)
    return float(ranked[min(int(np.searchsorted(cumulative, q, side="left")), len(ranked) - 1)])


def family_metrics(candidates: Sequence[dict]) -> dict | None:
    """Average queries per candidate, then candidates; retain equal-family ratio weights."""
    eligible = [item for item in candidates if len(item.get("queries", [])) == 2]
    if not eligible:
        return None
    accuracies = []
    ratios = []
    weighted_ratios = []
    for item in eligible:
        scores = item["queries"]
        accuracies.append(float(np.mean([row["correct"] for row in scores])))
        ratios.append(float(np.mean([row["ratio"] for row in scores])))
        weighted_ratios.extend((float(row["ratio"]), 1 / (len(eligible) * 2)) for row in scores)
    return {
        "candidate_count": len(eligible),
        "accuracy": float(np.mean(accuracies)),
        "ratio_mean": float(np.mean(ratios)),
        "weighted_ratios": weighted_ratios,
    }


def group_summary(
    families: Sequence[dict],
    seed: int,
    bootstrap: int = 2000,
    check_budget: Callable[[], None] | None = None,
) -> dict:
    """Family bootstrap; pooled samples preserve challenge/ordinary counts."""
    valid = [item for item in families if item.get("metrics") is not None]
    if not valid:
        return {"status": "unassessable", "families": 0}
    by_cohort = {
        cohort: [item for item in valid if item["cohort"] == cohort]
        for cohort in ("challenge", "ordinary")
    }

    def reduce(rows: Sequence[dict]) -> tuple[float, float]:
        accuracy = float(np.mean([item["metrics"]["accuracy"] for item in rows]))
        ratios = []
        weights = []
        for item in rows:
            for ratio, weight in item["metrics"]["weighted_ratios"]:
                ratios.append(ratio)
                weights.append(weight / len(rows))
        return accuracy, weighted_quantile(ratios, weights, 0.05)

    estimate = reduce(valid)
    rng = np.random.default_rng(seed)
    draws = []
    for draw in range(bootstrap):
        if check_budget is not None and draw % 100 == 0:
            check_budget()
        sampled = []
        for rows in by_cohort.values():
            if rows:
                sampled.extend(rows[index] for index in rng.integers(0, len(rows), len(rows)))
        draws.append(reduce(sampled))
    samples = np.asarray(draws, dtype=np.float64)
    return {
        "status": "assessable",
        "families": len(valid),
        "candidates": sum(item["metrics"]["candidate_count"] for item in valid),
        "accuracy": estimate[0],
        "ratio_p05": estimate[1],
        "accuracy_interval": np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
        "ratio_p05_interval": np.quantile(samples[:, 1], [0.025, 0.975]).tolist(),
    }
