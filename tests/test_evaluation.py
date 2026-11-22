"""Analytic regression tests for EVAL-001 metrics and aggregation."""

from dataclasses import replace

import numpy as np
import pytest

from jepa_action_effects.evaluation import (
    BranchPrediction,
    BranchTruth,
    EvaluationConfig,
    FamilyTruth,
    Goal,
    aggregate_families,
    evaluate_family,
)

CONFIG = EvaluationConfig(1.0, 100.0, 0.0, 2)


def _truth(
    endpoints=((2.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    contacts=(True, False, True),
    *,
    family_id="family",
):
    ids = ("a", "b", "c")
    return FamilyTruth(
        family_id,
        np.zeros(2),
        np.zeros(2),
        tuple(
            BranchTruth(
                candidate_id,
                endpoint,
                np.asarray(endpoint) + 1,
                contacted,
                np.asarray([[0, 0], endpoint], dtype=float),
                "fixture_step",
            )
            for candidate_id, endpoint, contacted in zip(ids, endpoints, contacts, strict=True)
        ),
        2,
        split="development",
        group_id="g1",
    )


def _predictions(truth, transform=lambda endpoint: endpoint, *, agent=True):
    return tuple(
        BranchPrediction(
            branch.candidate_id,
            transform(branch.block_endpoint),
            transform(branch.agent_endpoint) if agent else None,
        )
        for branch in truth.branches
    )


def _evaluate(truth, predictions, goals=None):
    if goals is None:
        goals = (Goal("origin", np.zeros(2)),)
    return evaluate_family(
        truth,
        tuple(predictions),
        tuple(goals),
        CONFIG,
        method="fixture",
        goal_provenance="analytic",
    )


def test_oracle_persistence_doubled_and_reversed_fixtures():
    truth = _truth(endpoints=((2, 0), (0, 0), (0, 2)))
    oracle = _evaluate(truth, _predictions(truth))
    assert all(metric.block_endpoint_error == 0 for metric in oracle.branch_metrics)
    assert all(metric.effect_error == 0 for metric in oracle.pair_metrics)
    assert oracle.goal_metrics[0].regret == 0

    persistence = _evaluate(truth, _predictions(truth, lambda _: np.zeros(2)))
    assert all(metric.predicted_effect_magnitude == 0 for metric in persistence.pair_metrics)
    assert all(
        metric.effect_error == metric.true_effect_magnitude for metric in persistence.pair_metrics
    )
    assert all(
        metric.missed_meaningful_effect is True
        for metric in persistence.pair_metrics
        if metric.effect_label == "meaningful"
    )

    doubled = _evaluate(truth, _predictions(truth, lambda endpoint: 2 * endpoint))
    assert all(
        metric.effect_error == pytest.approx(metric.true_effect_magnitude)
        for metric in doubled.pair_metrics
    )

    reversed_effect = _evaluate(truth, _predictions(truth, lambda endpoint: -endpoint))
    meaningful = [
        metric for metric in reversed_effect.pair_metrics if metric.effect_label == "meaningful"
    ]
    assert all(
        metric.effect_error == pytest.approx(2 * metric.true_effect_magnitude)
        for metric in meaningful
    )
    assert all(metric.direction_error_radians == pytest.approx(np.pi) for metric in meaningful)


def test_spurious_motion_shared_bias_and_zero_vector_direction():
    zero_truth = _truth(endpoints=((0, 0), (0, 0), (0, 0)))
    predictions = list(_predictions(zero_truth))
    predictions[0] = BranchPrediction("a", np.array([3.0, 0.0]), np.ones(2))
    spurious = _evaluate(zero_truth, predictions)
    assert spurious.pair_metrics[0].effect_error == 3
    assert spurious.pair_metrics[0].exact_zero_true_effect
    assert spurious.pair_metrics[0].direction_error_radians is None

    truth = _truth(endpoints=((3, 0), (0, 0), (0, 2)))
    biased = _evaluate(truth, _predictions(truth, lambda endpoint: endpoint + 5))
    assert all(metric.effect_error == 0 for metric in biased.pair_metrics)
    assert all(metric.block_endpoint_error > 0 for metric in biased.branch_metrics)


def test_contact_strata_boundary_motion_and_transient_return():
    truth = _truth(endpoints=((1, 0), (0, 0), (0, 2)), contacts=(False, False, True))
    evaluation = _evaluate(truth, _predictions(truth))
    assert {metric.contact_stratum for metric in evaluation.pair_metrics} == {
        "neither",
        "exactly_one",
    }
    at_boundary = next(
        metric for metric in evaluation.pair_metrics if metric.true_effect_magnitude == 1
    )
    assert at_boundary.effect_label == "negligible"
    stationary_boundary = next(
        metric for metric in evaluation.branch_metrics if metric.candidate_id == "a"
    )
    assert stationary_boundary.true_max_displacement == 1
    assert stationary_boundary.true_stationary_no_contact is True

    two_branch_truth = replace(truth, branches=truth.branches[:2])
    boundary_goal = _evaluate(two_branch_truth, _predictions(two_branch_truth)).goal_metrics[0]
    assert boundary_goal.true_cost_spread == 1
    assert not boundary_goal.informative

    moving = BranchTruth(
        "moving", np.zeros(2), np.zeros(2), False, np.array([[2, 0], [0, 0]]), "substep"
    )
    moved_truth = FamilyTruth("transient", np.zeros(2), np.zeros(2), (moving,), 2)
    moved_eval = _evaluate(
        moved_truth,
        (BranchPrediction("moving", np.zeros(2)),),
    )
    metric = moved_eval.branch_metrics[0]
    assert metric.true_max_displacement == 2
    assert metric.true_stationary_no_contact is False
    assert metric.block_endpoint_error == 0


def test_regret_ties_permutations_and_random_expectation():
    truth = _truth(endpoints=((0, 0), (2, 0), (4, 0)))
    predictions = (
        BranchPrediction("c", np.array([0.0, 0.0])),
        BranchPrediction("b", np.array([0.0, 0.0])),
        BranchPrediction("a", np.array([5.0, 0.0])),
    )
    goals = (Goal("g", np.zeros(2)),)
    evaluation = _evaluate(truth, predictions, goals)
    goal = evaluation.goal_metrics[0]
    assert goal.selected_candidate_id == "b"
    assert goal.oracle_candidate_id == "a"
    assert goal.regret == 2
    assert goal.uniform_random_expected_regret == 2
    assert goal.informative

    permuted_truth = replace(truth, branches=tuple(reversed(truth.branches)))
    permuted = _evaluate(permuted_truth, tuple(reversed(predictions)), goals)
    assert permuted.goal_metrics == evaluation.goal_metrics
    assert permuted.pair_metrics == evaluation.pair_metrics


def test_failures_malformed_inputs_and_serialization_are_explicit():
    truth = _truth()
    failed = list(_predictions(truth))
    failed[0] = BranchPrediction("a", None, failure_reason="decoder failed")
    evaluation = _evaluate(truth, failed)
    assert not evaluation.evaluation_eligible
    assert evaluation.truth_eligible
    assert evaluation.prediction_failures == {"a": "decoder failed"}
    assert evaluation.to_dict()["branch_metrics"] == ()

    incomplete = replace(truth, eligible=False, exclusion_reasons=("incomplete",))
    evaluation = _evaluate(incomplete, _predictions(incomplete))
    assert not evaluation.truth_eligible
    assert evaluation.exclusion_reasons == ("incomplete",)

    with pytest.raises(ValueError, match="sets must match"):
        _evaluate(truth, _predictions(truth)[:-1])
    with pytest.raises(ValueError, match="finite 2-D"):
        BranchTruth("x", [np.nan, 0], [0, 0], False)
    with pytest.raises(ValueError, match="finite 2-D"):
        BranchPrediction("x", [0, 1, 2])


def test_family_aggregation_weights_families_and_reports_empty_subsets():
    truth_one_full = _truth(family_id="one")
    truth_one = replace(truth_one_full, branches=truth_one_full.branches[:2])
    family_one = _evaluate(truth_one, _predictions(truth_one))
    truth_two = _truth(
        endpoints=((4, 0), (0, 0), (0, 3)),
        contacts=(True, True, True),
        family_id="two",
    )
    predictions_two = _predictions(truth_two, lambda endpoint: np.zeros(2))
    family_two = _evaluate(truth_two, predictions_two, goals=())
    summary = aggregate_families((family_one, family_two))
    assert summary["eligible_family_count"] == 2
    assert summary["contact_strata"]["both"]["family_count"] == 1
    assert summary["contact_strata"]["neither"]["mean_family_effect_error"] is None
    assert summary["mean_family_block_endpoint_error"] == pytest.approx(7 / 6)
    assert summary["meaningful_effect_pair_count"] > 0
    assert summary["mean_family_regret_all_goals"] == 0
