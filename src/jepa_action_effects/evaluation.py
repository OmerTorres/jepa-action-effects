"""Physical endpoint, paired-effect, contact-stratum, and regret evaluation."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _point(value: Any, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must be a finite 2-D point")
    result = point.copy()
    result.setflags(write=False)
    return result


def _trajectory(value: Any | None, label: str) -> np.ndarray | None:
    if value is None:
        return None
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0:
        raise ValueError(f"{label} must have shape (time, 2) with at least one row")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{label} must be finite")
    result = trajectory.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EvaluationConfig:
    delta: float
    workspace_width: float
    sim_epsilon: float
    horizon_native_steps: int
    informative_cost_spread_strict: bool = True

    def __post_init__(self) -> None:
        for name in ("delta", "workspace_width", "sim_epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.delta <= 0 or self.workspace_width <= 0:
            raise ValueError("delta and workspace_width must be positive")
        if self.horizon_native_steps <= 0:
            raise ValueError("horizon_native_steps must be positive")


@dataclass(frozen=True)
class BranchTruth:
    candidate_id: str
    block_endpoint: np.ndarray
    agent_endpoint: np.ndarray
    contacted: bool
    block_trajectory: np.ndarray | None = None
    trajectory_resolution: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        object.__setattr__(self, "block_endpoint", _point(self.block_endpoint, "block_endpoint"))
        object.__setattr__(self, "agent_endpoint", _point(self.agent_endpoint, "agent_endpoint"))
        object.__setattr__(
            self,
            "block_trajectory",
            _trajectory(self.block_trajectory, "block_trajectory"),
        )
        if self.block_trajectory is not None and not self.trajectory_resolution:
            raise ValueError("trajectory_resolution is required with block_trajectory")


@dataclass(frozen=True)
class BranchPrediction:
    candidate_id: str
    block_endpoint: np.ndarray | None
    agent_endpoint: np.ndarray | None = None
    block_trajectory: np.ndarray | None = None
    trajectory_resolution: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if self.failure_reason:
            if any(
                value is not None
                for value in (self.block_endpoint, self.agent_endpoint, self.block_trajectory)
            ):
                raise ValueError("failed predictions cannot contain predicted values")
            return
        if self.block_endpoint is None:
            raise ValueError("block_endpoint is required unless failure_reason is present")
        object.__setattr__(
            self, "block_endpoint", _point(self.block_endpoint, "predicted block_endpoint")
        )
        if self.agent_endpoint is not None:
            object.__setattr__(
                self, "agent_endpoint", _point(self.agent_endpoint, "predicted agent_endpoint")
            )
        object.__setattr__(
            self,
            "block_trajectory",
            _trajectory(self.block_trajectory, "predicted block_trajectory"),
        )
        if self.block_trajectory is not None and not self.trajectory_resolution:
            raise ValueError("trajectory_resolution is required with predicted trajectory")


@dataclass(frozen=True)
class Goal:
    goal_id: str
    position: np.ndarray

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id must be nonempty")
        object.__setattr__(self, "position", _point(self.position, "goal position"))


@dataclass(frozen=True)
class FamilyTruth:
    family_id: str
    current_block_position: np.ndarray
    current_agent_position: np.ndarray
    branches: tuple[BranchTruth, ...]
    horizon_native_steps: int
    split: str | None = None
    group_id: str | None = None
    units: str = "workspace"
    eligible: bool = True
    exclusion_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must be nonempty")
        object.__setattr__(
            self, "current_block_position", _point(self.current_block_position, "current block")
        )
        object.__setattr__(
            self, "current_agent_position", _point(self.current_agent_position, "current agent")
        )
        object.__setattr__(self, "branches", tuple(self.branches))
        ids = [branch.candidate_id for branch in self.branches]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("truth candidate IDs must be nonempty and unique")
        if self.horizon_native_steps <= 0:
            raise ValueError("horizon_native_steps must be positive")
        if self.eligible and self.exclusion_reasons:
            raise ValueError("eligible truth cannot have exclusion reasons")
        if not self.eligible and not self.exclusion_reasons:
            raise ValueError("ineligible truth requires an exclusion reason")


@dataclass(frozen=True)
class BranchMetric:
    candidate_id: str
    contacted: bool
    block_endpoint_error: float
    agent_endpoint_error: float | None
    true_max_displacement: float | None
    true_stationary_no_contact: bool | None
    predicted_max_displacement: float | None
    predicted_trajectory_resolution: str | None


@dataclass(frozen=True)
class PairMetric:
    candidate_a: str
    candidate_b: str
    contact_stratum: str
    true_effect: tuple[float, float]
    predicted_effect: tuple[float, float]
    true_effect_magnitude: float
    predicted_effect_magnitude: float
    effect_error: float
    magnitude_error: float
    direction_error_radians: float | None
    effect_label: str
    exact_zero_true_effect: bool
    missed_meaningful_effect: bool | None


@dataclass(frozen=True)
class GoalMetric:
    goal_id: str
    selected_candidate_id: str
    oracle_candidate_id: str
    selected_true_cost: float
    oracle_true_cost: float
    regret: float
    oracle_regret: float
    uniform_random_expected_regret: float
    true_cost_spread: float
    informative: bool


@dataclass(frozen=True)
class FamilyEvaluation:
    family_id: str
    method: str
    truth_eligible: bool
    evaluation_eligible: bool
    exclusion_reasons: tuple[str, ...]
    prediction_failures: dict[str, str]
    branch_metrics: tuple[BranchMetric, ...] = ()
    pair_metrics: tuple[PairMetric, ...] = ()
    goal_metrics: tuple[GoalMetric, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contact_stratum(a: bool, b: bool) -> str:
    count = int(a) + int(b)
    return ("neither", "exactly_one", "both")[count]


def _direction_error(true_effect: np.ndarray, predicted_effect: np.ndarray, delta: float):
    true_norm = float(np.linalg.norm(true_effect))
    predicted_norm = float(np.linalg.norm(predicted_effect))
    if true_norm <= delta or predicted_norm == 0:
        return None
    cosine = float(np.dot(true_effect, predicted_effect) / (true_norm * predicted_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _max_displacement(trajectory: np.ndarray | None, start: np.ndarray) -> float | None:
    if trajectory is None:
        return None
    return float(np.max(np.linalg.norm(trajectory - start, axis=1)))


def _unique_by_id(values: tuple[Any, ...], label: str) -> dict[str, Any]:
    result = {value.candidate_id: value for value in values}
    if len(result) != len(values):
        raise ValueError(f"{label} candidate IDs must be unique")
    return result


def evaluate_family(
    truth: FamilyTruth,
    predictions: tuple[BranchPrediction, ...],
    goals: tuple[Goal, ...],
    config: EvaluationConfig,
    *,
    method: str,
    goal_provenance: str,
) -> FamilyEvaluation:
    """Evaluate one family without allowing predictions to define eligibility."""
    if not method or not goal_provenance:
        raise ValueError("method and goal_provenance must be nonempty")
    if truth.horizon_native_steps != config.horizon_native_steps:
        raise ValueError("truth and evaluation horizons differ")
    truth_by_id = _unique_by_id(truth.branches, "truth")
    prediction_by_id = _unique_by_id(tuple(predictions), "prediction")
    if set(truth_by_id) != set(prediction_by_id):
        raise ValueError("truth and prediction candidate ID sets must match exactly")
    goal_by_id = {goal.goal_id: goal for goal in goals}
    if len(goal_by_id) != len(goals):
        raise ValueError("goal IDs must be unique")

    failures = {
        candidate_id: prediction.failure_reason
        for candidate_id, prediction in prediction_by_id.items()
        if prediction.failure_reason is not None
    }
    reasons = list(truth.exclusion_reasons)
    if failures:
        reasons.append("prediction_failure")
    eligible = truth.eligible and not failures
    metadata = {
        "split": truth.split,
        "group_id": truth.group_id,
        "units": truth.units,
        "horizon_native_steps": truth.horizon_native_steps,
        "delta": config.delta,
        "goal_provenance": goal_provenance,
        "pair_policy": "all_unordered_canonical_candidate_id",
        "tie_policy": "canonical_candidate_id",
    }
    if not eligible:
        return FamilyEvaluation(
            truth.family_id,
            method,
            truth.eligible,
            False,
            tuple(reasons),
            failures,
            metadata=metadata,
        )

    branch_metrics: list[BranchMetric] = []
    for candidate_id in sorted(truth_by_id):
        branch = truth_by_id[candidate_id]
        prediction = prediction_by_id[candidate_id]
        assert prediction.block_endpoint is not None
        true_max = _max_displacement(branch.block_trajectory, truth.current_block_position)
        predicted_max = _max_displacement(prediction.block_trajectory, truth.current_block_position)
        branch_metrics.append(
            BranchMetric(
                candidate_id,
                branch.contacted,
                float(np.linalg.norm(prediction.block_endpoint - branch.block_endpoint)),
                None
                if prediction.agent_endpoint is None
                else float(np.linalg.norm(prediction.agent_endpoint - branch.agent_endpoint)),
                true_max,
                None if branch.contacted or true_max is None else true_max <= config.delta,
                predicted_max,
                prediction.trajectory_resolution,
            )
        )

    pair_metrics: list[PairMetric] = []
    for candidate_a, candidate_b in itertools.combinations(sorted(truth_by_id), 2):
        branch_a, branch_b = truth_by_id[candidate_a], truth_by_id[candidate_b]
        prediction_a, prediction_b = prediction_by_id[candidate_a], prediction_by_id[candidate_b]
        assert prediction_a.block_endpoint is not None and prediction_b.block_endpoint is not None
        true_effect = branch_a.block_endpoint - branch_b.block_endpoint
        predicted_effect = prediction_a.block_endpoint - prediction_b.block_endpoint
        true_magnitude = float(np.linalg.norm(true_effect))
        predicted_magnitude = float(np.linalg.norm(predicted_effect))
        meaningful = true_magnitude > config.delta
        pair_metrics.append(
            PairMetric(
                candidate_a,
                candidate_b,
                _contact_stratum(branch_a.contacted, branch_b.contacted),
                tuple(float(value) for value in true_effect),
                tuple(float(value) for value in predicted_effect),
                true_magnitude,
                predicted_magnitude,
                float(np.linalg.norm(predicted_effect - true_effect)),
                abs(predicted_magnitude - true_magnitude),
                _direction_error(true_effect, predicted_effect, config.delta),
                "meaningful" if meaningful else "negligible",
                true_magnitude == 0,
                predicted_magnitude <= config.delta if meaningful else None,
            )
        )

    goal_metrics: list[GoalMetric] = []
    candidate_ids = sorted(truth_by_id)
    for goal_id in sorted(goal_by_id):
        goal = goal_by_id[goal_id]
        true_costs = {
            candidate_id: float(
                np.linalg.norm(truth_by_id[candidate_id].block_endpoint - goal.position)
            )
            for candidate_id in candidate_ids
        }
        predicted_costs = {
            candidate_id: float(
                np.linalg.norm(prediction_by_id[candidate_id].block_endpoint - goal.position)
            )
            for candidate_id in candidate_ids
        }
        selected = min(candidate_ids, key=lambda key: (predicted_costs[key], key))
        oracle = min(candidate_ids, key=lambda key: (true_costs[key], key))
        oracle_cost = true_costs[oracle]
        regret = true_costs[selected] - oracle_cost
        if regret < -1e-12:
            raise AssertionError("regret cannot be negative")
        spread = max(true_costs.values()) - min(true_costs.values())
        informative = (
            spread > config.delta
            if config.informative_cost_spread_strict
            else spread >= config.delta
        )
        goal_metrics.append(
            GoalMetric(
                goal_id,
                selected,
                oracle,
                true_costs[selected],
                oracle_cost,
                max(0.0, regret),
                0.0,
                float(np.mean(list(true_costs.values()))) - oracle_cost,
                spread,
                informative,
            )
        )

    return FamilyEvaluation(
        truth.family_id,
        method,
        True,
        True,
        (),
        {},
        tuple(branch_metrics),
        tuple(pair_metrics),
        tuple(goal_metrics),
        metadata,
    )


def aggregate_families(evaluations: tuple[FamilyEvaluation, ...]) -> dict[str, Any]:
    """Average within family first, then give each contributing family equal weight."""
    eligible = [evaluation for evaluation in evaluations if evaluation.evaluation_eligible]
    strata = ("neither", "exactly_one", "both")
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in strata:
        family_values = []
        pair_count = 0
        for evaluation in eligible:
            values = [
                metric.effect_error
                for metric in evaluation.pair_metrics
                if metric.contact_stratum == stratum
            ]
            if values:
                family_values.append(float(np.mean(values)))
                pair_count += len(values)
        by_stratum[stratum] = {
            "family_count": len(family_values),
            "pair_count": pair_count,
            "mean_family_effect_error": (
                None if not family_values else float(np.mean(family_values))
            ),
        }

    endpoint_family_means = [
        float(np.mean([metric.block_endpoint_error for metric in evaluation.branch_metrics]))
        for evaluation in eligible
    ]
    agent_endpoint_family_means = []
    negligible_family_means = []
    meaningful_missed_family_rates = []
    negligible_pair_count = 0
    meaningful_pair_count = 0
    missed_meaningful_pair_count = 0
    for evaluation in eligible:
        agent_values = [
            metric.agent_endpoint_error
            for metric in evaluation.branch_metrics
            if metric.agent_endpoint_error is not None
        ]
        if agent_values:
            agent_endpoint_family_means.append(float(np.mean(agent_values)))
        negligible = [
            metric.predicted_effect_magnitude
            for metric in evaluation.pair_metrics
            if metric.effect_label == "negligible"
        ]
        if negligible:
            negligible_family_means.append(float(np.mean(negligible)))
            negligible_pair_count += len(negligible)
        meaningful = [
            metric.missed_meaningful_effect
            for metric in evaluation.pair_metrics
            if metric.effect_label == "meaningful"
        ]
        if meaningful:
            meaningful_missed_family_rates.append(float(np.mean(meaningful)))
            meaningful_pair_count += len(meaningful)
            missed_meaningful_pair_count += sum(bool(value) for value in meaningful)
    goal_family_means = [
        float(np.mean([metric.regret for metric in evaluation.goal_metrics]))
        for evaluation in eligible
        if evaluation.goal_metrics
    ]
    random_regret_family_means = [
        float(
            np.mean([metric.uniform_random_expected_regret for metric in evaluation.goal_metrics])
        )
        for evaluation in eligible
        if evaluation.goal_metrics
    ]
    informative_family_means = []
    for evaluation in eligible:
        values = [metric.regret for metric in evaluation.goal_metrics if metric.informative]
        if values:
            informative_family_means.append(float(np.mean(values)))
    return {
        "input_family_count": len(evaluations),
        "eligible_family_count": len(eligible),
        "mean_family_block_endpoint_error": (
            None if not endpoint_family_means else float(np.mean(endpoint_family_means))
        ),
        "mean_family_agent_endpoint_error": (
            None if not agent_endpoint_family_means else float(np.mean(agent_endpoint_family_means))
        ),
        "agent_endpoint_family_count": len(agent_endpoint_family_means),
        "mean_family_predicted_effect_magnitude_negligible": (
            None if not negligible_family_means else float(np.mean(negligible_family_means))
        ),
        "negligible_effect_family_count": len(negligible_family_means),
        "negligible_effect_pair_count": negligible_pair_count,
        "mean_family_missed_meaningful_effect_rate": (
            None
            if not meaningful_missed_family_rates
            else float(np.mean(meaningful_missed_family_rates))
        ),
        "meaningful_effect_family_count": len(meaningful_missed_family_rates),
        "meaningful_effect_pair_count": meaningful_pair_count,
        "missed_meaningful_effect_pair_count": missed_meaningful_pair_count,
        "mean_family_regret_all_goals": (
            None if not goal_family_means else float(np.mean(goal_family_means))
        ),
        "mean_family_uniform_random_expected_regret": (
            None if not random_regret_family_means else float(np.mean(random_regret_family_means))
        ),
        "mean_family_regret_informative_goals": (
            None if not informative_family_means else float(np.mean(informative_family_means))
        ),
        "informative_goal_family_count": len(informative_family_means),
        "contact_strata": by_stratum,
    }
