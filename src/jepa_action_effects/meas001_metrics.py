"""MEAS-001 gates and summaries (D032 §6; pure functions, labels supplied by the validator).

A *record* is one inverted query, already scored against simulator labels:

    {"family", "scene", "query", "status", "block_center" (accepted only),
     "best_block" (best searched block centre, any status), "true_block",
     "unsupported_true_geometry" (bool)}

Families are the resampling unit; every scene × query cell stays separate and failed
cells are never pooled away. Rejected queries count against coverage, and their best-pose
errors are reported separately so selective coverage cannot hide poor accuracy. Passing
these gates shows measurement feasibility on real latents only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

GATES = {
    "min_accepted": 10,  # of 12 families per scene × query cell
    "endpoint_mean": 1.28,
    "endpoint_p95": 2.56,
    "displacement_mean": 2.56,
    "floor_min_families": 10,
    "floor_ratio": 0.25,
    "moved_delta": 5.12,
}
LOW_MOTION = {"challenge": "m0.15"}  # reported separately, not a gated endpoint cell


def _error(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _bootstrap(values: Sequence[float], statistic, seed: int, draws: int) -> list[float] | None:
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _p95(values) -> float:
    return float(np.percentile(np.asarray(values, float), 95))


def endpoint_cell(
    records: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    gates: Mapping[str, float] = GATES,
    seed: int = 0,
    draws: int = 2000,
) -> dict[str, Any]:
    """Coverage and accuracy for one scene × query cell; `families` is the full cell."""
    by_family = {r["family"]: r for r in records}
    accepted = [by_family[f] for f in families if by_family.get(f, {}).get("status") == "accepted"]
    errors = [_error(r["block_center"], r["true_block"]) for r in accepted]
    rejected = [
        _error(r["best_block"], r["true_block"])
        for f in families
        if (r := by_family.get(f)) and r["status"] != "accepted" and r.get("best_block")
    ]
    statuses = {}
    for family in families:
        status = by_family.get(family, {}).get("status", "missing")
        statuses[status] = statuses.get(status, 0) + 1
    result: dict[str, Any] = {
        "families": len(families),
        "accepted": len(accepted),
        "statuses": statuses,
        "unsupported_true_geometry": sum(
            bool(by_family.get(f, {}).get("unsupported_true_geometry")) for f in families
        ),
        "mean_error": float(np.mean(errors)) if errors else None,
        "p95_error": _p95(errors) if errors else None,
        "mean_error_ci95": _bootstrap(errors, np.mean, seed, draws),
        "p95_error_ci95": _bootstrap(errors, _p95, seed + 1, draws),
        "rejected_best_pose_mean_error": float(np.mean(rejected)) if rejected else None,
        "rejected_count": len(rejected),
    }
    result["checks"] = {
        "coverage": len(accepted) >= gates["min_accepted"],
        "mean": result["mean_error"] is not None and result["mean_error"] <= gates["endpoint_mean"],
        "p95": result["p95_error"] is not None and result["p95_error"] <= gates["endpoint_p95"],
    }
    result["passed"] = all(result["checks"].values())
    return result


def displacement_cell(
    starts: Sequence[Mapping[str, Any]],
    ends: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    gates: Mapping[str, float] = GATES,
    seed: int = 0,
    draws: int = 2000,
) -> dict[str, Any]:
    """Displacement error on jointly accepted start/end pairs, plus the relative floor."""
    start = {r["family"]: r for r in starts}
    end = {r["family"]: r for r in ends}
    errors, moving_errors, moving_true = [], [], []
    for family in families:
        a, b = start.get(family), end.get(family)
        if not a or not b or a["status"] != "accepted" or b["status"] != "accepted":
            continue
        measured = np.subtract(b["block_center"], a["block_center"])
        true = np.subtract(b["true_block"], a["true_block"])
        error = float(np.linalg.norm(measured - true))
        errors.append(error)
        if np.linalg.norm(true) > gates["moved_delta"]:
            moving_errors.append(error)
            moving_true.append(float(np.linalg.norm(true)))
    floor: dict[str, Any] = {"moving_families": len(moving_errors)}
    if len(moving_errors) >= gates["floor_min_families"]:
        floor["ratio"] = float(np.mean(moving_errors) / np.median(moving_true))
        floor["status"] = "passed" if floor["ratio"] < gates["floor_ratio"] else "failed"
    else:
        floor["ratio"] = None
        floor["status"] = "unassessable"
    mean = float(np.mean(errors)) if errors else None
    return {
        "joint_accepted": len(errors),
        "mean_error": mean,
        "mean_error_ci95": _bootstrap(errors, np.mean, seed, draws),
        "relative_floor": floor,
        "passed": mean is not None and mean <= gates["displacement_mean"],
    }


def decision(
    endpoint_cells: Mapping[str, Mapping[str, Any]],
    displacement_cells: Mapping[str, Mapping[str, Any]],
    integrity_passed: bool,
    controls_passed: bool,
    complete: bool,
) -> dict[str, Any]:
    """D032 outcome; failed cells stay ineligible and are never pooled away."""
    if not integrity_passed:
        return {"outcome": "unresolved", "reason": "integrity failure", "eligible_cells": []}
    if not complete:
        return {"outcome": "unresolved", "reason": "incomplete assessment", "eligible_cells": []}
    eligible = []
    for cell, result in endpoint_cells.items():
        passed = result["passed"]
        if cell in displacement_cells:
            moving = displacement_cells[cell]
            passed = passed and moving["passed"]
            # An unassessable floor (too little motion) cannot show feasibility either.
            passed = passed and moving["relative_floor"]["status"] == "passed"
        if passed:
            eligible.append(cell)
    if not controls_passed or not eligible:
        reason = "known-failure controls failed" if not controls_passed else "no cell passed"
        return {"outcome": "not_feasible", "reason": reason, "eligible_cells": eligible}
    outcome = "feasible_all" if len(eligible) == len(endpoint_cells) else "feasible_subset"
    return {"outcome": outcome, "reason": "", "eligible_cells": sorted(eligible)}


PREDICTION_CELLS = {
    "endpoints": ("challenge:m0.30", "challenge:m0.60", "archive:x1.0", "archive:x2.0"),
    "starts": ("challenge:start", "archive:start"),
}


def predictions_allowed(
    outcome: Mapping[str, Any],
    endpoint_cells: Mapping[str, Mapping[str, Any]],
    displacement_cells: Mapping[str, Mapping[str, Any]],
    integrity_passed: bool,
    complete: bool,
    controls_passed: bool,
) -> bool:
    """D032 §7, derived from the complete assessment decision (MEAS-REVIEW-001 F2).

    Requires integrity, a complete assessment, passing known-failure controls, a feasible
    decision, both start cells, and the four named endpoint cells *including* their
    displacement and relative-floor gates (all or none).
    """
    if not (integrity_passed and complete and controls_passed):
        return False
    if outcome.get("outcome") not in ("feasible_all", "feasible_subset"):
        return False
    eligible = set(outcome.get("eligible_cells", []))
    for cell in PREDICTION_CELLS["starts"]:
        if endpoint_cells.get(cell, {}).get("passed") is not True or cell not in eligible:
            return False
    for cell in PREDICTION_CELLS["endpoints"]:
        moving = displacement_cells.get(cell)
        if cell not in eligible or moving is None or not moving.get("passed"):
            return False
        if moving.get("relative_floor", {}).get("status") != "passed":
            return False
    return True


ENGINEERING = {
    "min_coverage": 0.8,
    "min_restart_agreement": 0.8,
    "endpoint_mean": 1.28,
    "endpoint_p95": 2.56,
    "synthetic_within": 1.28,
    "min_synthetic_recovered": 0.8,
}


def engineering_criteria(
    records: Sequence[Mapping[str, Any]], criteria: Mapping[str, float] = ENGINEERING
) -> dict[str, Any]:
    """Per-cohort checks on engineering queries; mirrors the assessment gates (D033)."""
    result: dict[str, Any] = {"cohorts": {}}
    for cohort in sorted({r["scene"] for r in records}):
        rows = [r for r in records if r["scene"] == cohort and r["kind"] == "real"]
        if not rows:
            continue
        accepted = [r for r in rows if r["status"] == "accepted"]
        agreement = [
            all(r["converged"]) and r.get("restart_distance", np.inf) <= 1.28 for r in rows
        ]
        errors = [_error(r["block_center"], r["true_block"]) for r in accepted]
        cell = {
            "queries": len(rows),
            "coverage": len(accepted) / len(rows),
            "restart_agreement": float(np.mean(agreement)),
            "mean_error": float(np.mean(errors)) if errors else None,
            "p95_error": _p95(errors) if errors else None,
        }
        cell["checks"] = {
            "coverage": cell["coverage"] >= criteria["min_coverage"],
            "restart": cell["restart_agreement"] >= criteria["min_restart_agreement"],
            "mean": cell["mean_error"] is not None
            and cell["mean_error"] <= criteria["endpoint_mean"],
            "p95": cell["p95_error"] is not None and cell["p95_error"] <= criteria["endpoint_p95"],
        }
        result["cohorts"][cohort] = cell
    synthetic = [r for r in records if r["kind"] == "synthetic"]
    recovered = [
        r["status"] == "accepted"
        and _error(r["block_center"], r["true_block"]) <= criteria["synthetic_within"]
        for r in synthetic
    ]
    result["synthetic"] = {
        "queries": len(synthetic),
        "recovered": float(np.mean(recovered)) if recovered else None,
        "passed": bool(recovered)
        and float(np.mean(recovered)) >= criteria["min_synthetic_recovered"],
    }
    result["passed"] = (
        bool(result["cohorts"])
        and all(all(c["checks"].values()) for c in result["cohorts"].values())
        and result["synthetic"]["passed"]
    )
    return result
