"""MEAS-001 gate logic on synthetic scored records."""

from __future__ import annotations

import numpy as np

from jepa_action_effects.meas001_metrics import (
    decision,
    displacement_cell,
    endpoint_cell,
    engineering_criteria,
    predictions_allowed,
)

FAMILIES = [f"f{i:02d}" for i in range(12)]


def _record(family, status="accepted", error=0.5, start=(100.0, 100.0), shift=(0.0, 0.0)):
    true = np.add(start, shift)
    record = {
        "family": family,
        "status": status,
        "true_block": true.tolist(),
        "best_block": (true + [error, 0.0]).tolist(),
    }
    if status == "accepted":
        record["block_center"] = (true + [error, 0.0]).tolist()
    return record


def test_coverage_counts_abstentions_and_missing_families() -> None:
    records = [_record(f) for f in FAMILIES[:9]] + [_record(FAMILIES[9], "ambiguous", 30.0)]
    cell = endpoint_cell(records, FAMILIES, draws=50)
    assert cell["accepted"] == 9 and not cell["checks"]["coverage"]
    assert cell["statuses"] == {"accepted": 9, "ambiguous": 1, "missing": 2}
    assert cell["rejected_best_pose_mean_error"] == 30.0  # reported, not hidden


def test_accuracy_gates_use_mean_and_p95() -> None:
    good = endpoint_cell([_record(f, error=0.5) for f in FAMILIES], FAMILIES, draws=50)
    assert good["passed"]
    tail = [_record(f, error=0.5) for f in FAMILIES[:11]] + [_record(FAMILIES[11], error=8.0)]
    bad = endpoint_cell(tail, FAMILIES, draws=50)
    assert bad["checks"]["mean"] and not bad["checks"]["p95"] and not bad["passed"]


def test_displacement_and_floor() -> None:
    starts = [_record(f, error=0.3) for f in FAMILIES]
    ends = [_record(f, error=0.8, shift=(20.0, 0.0)) for f in FAMILIES]
    cell = displacement_cell(starts, ends, FAMILIES, draws=50)
    assert cell["joint_accepted"] == 12 and cell["passed"]
    assert cell["relative_floor"]["status"] == "passed"
    assert abs(cell["relative_floor"]["ratio"] - 0.5 / 20.0) < 1e-9
    still = [_record(f, error=0.8, shift=(1.0, 0.0)) for f in FAMILIES]
    assert displacement_cell(starts, still, FAMILIES, draws=50)["relative_floor"]["status"] == (
        "unassessable"
    )


def test_decision_never_pools_failed_cells() -> None:
    passing = {"passed": True}
    moving_ok = {"passed": True, "relative_floor": {"status": "passed"}}
    moving_still = {"passed": True, "relative_floor": {"status": "unassessable"}}
    cells = {"a:start": passing, "a:x1.0": passing, "a:x2.0": {"passed": False}}
    result = decision(cells, {"a:x1.0": moving_ok}, True, True, True)
    assert result["outcome"] == "feasible_subset"
    assert result["eligible_cells"] == ["a:start", "a:x1.0"]
    assert decision(cells, {"a:x1.0": moving_still}, True, True, True)["eligible_cells"] == [
        "a:start"
    ]
    assert decision(cells, {}, False, True, True)["outcome"] == "unresolved"
    assert decision(cells, {}, True, True, False)["outcome"] == "unresolved"
    assert decision(cells, {}, True, False, True)["outcome"] == "not_feasible"


def _prediction_case():
    names = ["challenge:m0.30", "challenge:m0.60", "archive:x1.0", "archive:x2.0"]
    cells = {name: {"passed": True} for name in [*names, "challenge:start", "archive:start"]}
    moving = {n: {"passed": True, "relative_floor": {"status": "passed"}} for n in names}
    return cells, moving


def test_prediction_check_needs_the_complete_decision() -> None:
    cells, moving = _prediction_case()
    outcome = decision(cells, moving, True, True, True)
    assert predictions_allowed(outcome, cells, moving, True, True, True)
    # Each failing prerequisite closes the check (MEAS-REVIEW-001 F2).
    assert not predictions_allowed(outcome, cells, moving, False, True, True)  # integrity
    assert not predictions_allowed(outcome, cells, moving, True, False, True)  # incomplete
    assert not predictions_allowed(outcome, cells, moving, True, True, False)  # controls
    start_failed = {**cells, "archive:start": {"passed": False}}
    outcome_failed = decision(start_failed, moving, True, True, True)
    assert not predictions_allowed(outcome_failed, start_failed, moving, True, True, True)


def test_failed_floors_or_controls_block_predictions_even_if_positions_pass() -> None:
    # Review reproduction 1: every position cell passes but every displacement floor fails.
    cells, moving = _prediction_case()
    floors_failed = {n: {"passed": True, "relative_floor": {"status": "failed"}} for n in moving}
    outcome = decision(cells, floors_failed, True, True, True)
    assert not predictions_allowed(outcome, cells, floors_failed, True, True, True)
    # Review reproduction 2: known-failure controls fail (decision not_feasible).
    outcome = decision(cells, moving, True, False, True)
    assert outcome["outcome"] == "not_feasible"
    assert not predictions_allowed(outcome, cells, moving, True, True, False)
    # A displacement mean failure on one named cell also blocks the all-or-none rule.
    one_bad = {**moving, "archive:x2.0": {"passed": False, "relative_floor": {"status": "passed"}}}
    outcome = decision(cells, one_bad, True, True, True)
    assert not predictions_allowed(outcome, cells, one_bad, True, True, True)


def test_engineering_criteria_per_cohort() -> None:
    def row(scene, status="accepted", error=0.4, kind="real", converged=(True, True)):
        record = {"scene": scene, "kind": kind, "status": status, "converged": list(converged)}
        record["restart_distance"] = 0.2
        record["true_block"] = [100.0, 100.0]
        if status == "accepted":
            record["block_center"] = [100.0 + error, 100.0]
        return record

    rows = [row("challenge") for _ in range(10)] + [row("ordinary") for _ in range(10)]
    rows += [row("challenge", kind="synthetic") for _ in range(5)]
    assert engineering_criteria(rows)["passed"]
    rows += [row("ordinary", status="ambiguous") for _ in range(4)]  # coverage 10/14 < 0.8
    result = engineering_criteria(rows)
    assert not result["passed"] and not result["cohorts"]["ordinary"]["checks"]["coverage"]
