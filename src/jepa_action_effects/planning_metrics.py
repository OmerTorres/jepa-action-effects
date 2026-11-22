"""Pure regret measurements and sealed rules for H3-PLAN-001.

Each family has six candidates and up to four grid goals (block target positions). A
chooser picks the candidate whose final latent is closest to the goal image's latent,
which is LeWM's planning cost. Simulator regret is ‖q_end(â) − g‖ − min_a ‖q_end(a) − g‖.

The choosers compared are:

- `jepa`: predicted latents;
- `real_latent`: encoded real future frames, the same cost with a perfect predictor;
- `random`: the expected regret of a uniform choice.

A goal is informative when the candidates' simulator costs differ by more than
delta. That is decided from simulator outcomes only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from jepa_action_effects.next_metrics import bootstrap, mean_of, paired_contrast

DELTA = 5.12
MIN_FAMILIES = 50


def goal_regrets(
    block_end: np.ndarray,
    goal: Sequence[float],
    predicted: np.ndarray,
    real: np.ndarray,
    goal_latent: np.ndarray,
    delta: float = DELTA,
) -> dict[str, Any]:
    """Regret of each chooser for one goal; latents are (C,D), block_end is (C,2)."""
    block_end = np.asarray(block_end, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    cost = np.linalg.norm(block_end - target, axis=1)
    best = float(cost.min())
    latent = np.asarray(goal_latent, dtype=np.float64)
    choice = {
        "jepa": int(np.argmin(np.linalg.norm(np.asarray(predicted) - latent, axis=1))),
        "real_latent": int(np.argmin(np.linalg.norm(np.asarray(real) - latent, axis=1))),
    }
    return {
        "jepa": float(cost[choice["jepa"]] - best),
        "real_latent": float(cost[choice["real_latent"]] - best),
        "random": float(cost.mean() - best),
        "choice": choice,
        "informative": bool(cost.max() - best > delta),
        "cost_spread": float(cost.max() - best),
    }


def family_record(
    identifier: str,
    cohort: str,
    block_end: np.ndarray,
    goals: Sequence[Sequence[float]],
    valid: Sequence[bool],
    predicted: np.ndarray,
    real: np.ndarray,
    goal_latents: np.ndarray,
) -> dict[str, Any]:
    """Family means over valid goals, and over valid informative goals."""
    rows = [
        goal_regrets(block_end, goal, predicted, real, goal_latents[index])
        for index, goal in enumerate(goals)
        if valid[index]
    ]

    def mean(selected):
        if not selected:
            return None
        return {key: float(np.mean([row[key] for row in selected])) for key in CHOOSERS}

    return {
        "id": identifier,
        "cohort": cohort,
        "valid_goals": len(rows),
        "informative_goals": sum(row["informative"] for row in rows),
        "all": mean(rows),
        "informative": mean([row for row in rows if row["informative"]]),
        "goals": rows,
    }


CHOOSERS = ("jepa", "real_latent", "random")


def _specification() -> dict[str, Any]:
    spec: dict[str, Any] = {}
    for subset in ("all", "informative"):
        for chooser in CHOOSERS:
            spec[f"regret_{chooser}_{subset}"] = mean_of((subset, chooser))
        spec[f"jepa_minus_real_{subset}"] = paired_contrast(
            (subset, "jepa"), (subset, "real_latent")
        )
        spec[f"real_minus_random_{subset}"] = paired_contrast(
            (subset, "real_latent"), (subset, "random")
        )
        spec[f"jepa_minus_random_{subset}"] = paired_contrast((subset, "jepa"), (subset, "random"))
    return spec


SPECIFICATION = _specification()


def summarize_planning(records: Sequence[dict], seed: int, samples: int = 2000) -> dict[str, Any]:
    records = [r for r in records if r["all"] is not None]
    summary: dict[str, Any] = {}
    for offset, (name, statistic) in enumerate(SPECIFICATION.items()):
        summary[name] = bootstrap(records, statistic, seed + offset, samples)
    summary["families"] = len(records)
    summary["criteria"] = apply_planning_criteria(summary)
    return summary


def apply_planning_criteria(summary: dict[str, Any]) -> dict[str, Any]:
    """H3 holds if the predictor adds regret over a perfect-predictor latent chooser.

    The latent cost must itself work (real-latent choice beats random), otherwise the
    comparison says nothing about prediction error.
    """
    works = summary["real_minus_random_all"]["ci95"]
    checks = {
        "latent_cost_beats_random": works is not None and works[1] < 0,
        "enough_families": summary["families"] >= MIN_FAMILIES,
    }
    if not all(checks.values()):
        return {"checks": checks, "passed": False, "h3": "inconclusive_check_failed"}
    added = summary["jepa_minus_real_all"]["ci95"]
    return {
        "checks": checks,
        "passed": True,
        "h3": "supported" if added is not None and added[0] > 0 else "not_established",
    }
