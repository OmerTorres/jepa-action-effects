"""Regret under a latent planning cost with agent-matched goal images (H3-PLAN-002).

H3-PLAN-001 rendered one goal image per goal with the agent at its start pose. Latent
distance is dominated by the agent, so that cost chose `hold` for most goals and did
worse than random even with real future latents.

Here each candidate c gets its own goal image: the agent and block angle are taken
from c's realized final pose, and only the block position is moved to the goal. The
cost ‖z_c − e(goal_{g,c})‖ therefore compares block placement. This is a controlled
measurement that uses the realized agent pose, not a deployable planner.

Records have the same shape as `planning_metrics.family_record`, so
`planning_metrics.summarize_planning` and its sealed rule apply unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from jepa_action_effects.planning_metrics import CHOOSERS, DELTA


def matched_goal_regrets(
    block_end: np.ndarray,
    goal: Sequence[float],
    predicted: np.ndarray,
    real: np.ndarray,
    goal_latents: np.ndarray,
    delta: float = DELTA,
) -> dict[str, Any]:
    """Regret per chooser for one goal; goal_latents is (C,D), one image per candidate."""
    block_end = np.asarray(block_end, dtype=np.float64)
    cost = np.linalg.norm(block_end - np.asarray(goal, dtype=np.float64), axis=1)
    best = float(cost.min())
    targets = np.asarray(goal_latents, dtype=np.float64)
    if targets.shape != np.shape(predicted) or targets.shape != np.shape(real):
        raise ValueError("Goal latents must match the candidate latents (C,D)")
    choice = {
        "jepa": int(np.argmin(np.linalg.norm(np.asarray(predicted) - targets, axis=1))),
        "real_latent": int(np.argmin(np.linalg.norm(np.asarray(real) - targets, axis=1))),
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
    """goal_latents is (G,C,D); `valid[g]` requires all C goal images of goal g valid."""
    goal_latents = np.asarray(goal_latents)
    if goal_latents.shape[:2] != (len(goals), len(block_end)):
        raise ValueError("Goal latents must be (goals, candidates, D)")
    rows = [
        matched_goal_regrets(block_end, goal, predicted, real, goal_latents[index])
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


def choice_histogram(records: Sequence[dict], chooser: str, candidates: int = 6) -> list[int]:
    """How often each candidate index was chosen (checks for a `hold`-type confound)."""
    counts = [0] * candidates
    for record in records:
        for row in record["goals"]:
            counts[row["choice"][chooser]] += 1
    return counts
