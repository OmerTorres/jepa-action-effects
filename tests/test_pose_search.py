"""Analytic fixtures for the MEAS-001 pose search (no simulator or model)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jepa_action_effects.pose_search import (
    BLOCK,
    CalibrationFailure,
    Inversion,
    SearchConfig,
    SearchResult,
    Trace,
    calibrate_tau,
    classify,
    invert,
    seed_sequence,
)

TARGET = np.array([100.0, 300.0, 250.0, 260.0, 1.0])
# Same 2,048-evaluation cap as the default, split into two longer local starts.
ENGINE = SearchConfig(local_starts=2, local_evaluations=320)


def _wrapped(poses: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(poses, float) - target
    delta[:, 4] = (delta[:, 4] + math.pi) % (2 * math.pi) - math.pi
    return delta


class Bowl:
    """Quadratic residual around one pose; every pose is geometry-valid."""

    def __init__(self, target=TARGET, scale=(1, 1, 1, 1, 0.02), gain=1e-3):
        self.target, self.scale, self.gain = np.asarray(target, float), np.asarray(scale), gain
        self.calls = 0

    def valid(self, poses):
        return np.ones(len(poses), dtype=bool)

    def evaluate(self, poses):
        self.calls += len(poses)
        return self.gain * np.sum((_wrapped(poses, self.target) / self.scale) ** 2, axis=1)


class Flat(Bowl):
    def evaluate(self, poses):
        self.calls += len(poses)
        return np.full(len(poses), 0.5)


class TwoBowls(Bowl):
    """Equal minima whose block centres are 60 units apart."""

    def evaluate(self, poses):
        self.calls += len(poses)
        other = self.target.copy()
        other[2] += 60.0
        first = np.sum((_wrapped(poses, self.target) / self.scale) ** 2, axis=1)
        second = np.sum((_wrapped(poses, other) / self.scale) ** 2, axis=1)
        return self.gain * np.minimum(first, second)


class CurvedMean(Bowl):
    """Encoder f(s) = (s, s²) on block x; target is the mean of f(−1) and f(+1).

    Its inverse is ±sqrt(1/2), neither realized endpoint nor their physical mean (D031).
    """

    def evaluate(self, poses):
        self.calls += len(poses)
        s = (np.asarray(poses, float)[:, 2] - 250.0) / 100.0
        latent = s**2 + (s**2 - 1.0) ** 2  # |f(s) − (0, 1)|²
        rest = np.sum((_wrapped(poses, self.target) / self.scale)[:, [0, 1, 3, 4]] ** 2, axis=1)
        return latent + 1e-3 * rest


class Invalid(Bowl):
    def valid(self, poses):
        return np.zeros(len(poses), dtype=bool)


def test_known_minimum_is_recovered_and_accepted() -> None:
    assert ENGINE.evaluation_cap == SearchConfig().evaluation_cap == 2048
    result = classify(invert(Bowl(), ENGINE, 7, "bowl"), tau=1e-3)
    assert result["status"] == "accepted"
    assert np.linalg.norm(np.array(result["block_center"]) - TARGET[BLOCK]) < 0.5
    assert all(result["converged"])


def test_residual_above_tau_is_rejected() -> None:
    result = classify(invert(Bowl(gain=1.0), ENGINE, 7, "bowl"), tau=1e-9)
    assert result["status"] == "residual_reject"
    assert result["best_pose"] is not None  # kept for diagnostics


def test_flat_objective_is_ambiguous_not_confident() -> None:
    result = classify(invert(Flat(), SearchConfig(), 7, "flat"), tau=1e-3)
    assert result["status"] in {"ambiguous", "optimizer_unresolved"}
    assert result["status"] != "accepted"
    assert result["near_optimal_diameter"] > SearchConfig().near_optimal_diameter


def test_competing_minima_are_ambiguous() -> None:
    result = classify(invert(TwoBowls(), SearchConfig(), 7, "two"), tau=0.05)
    assert result["status"] in {"ambiguous", "optimizer_unresolved"}
    assert result["near_optimal_diameter"] > 50.0


def test_curved_latent_mean_inverts_to_two_intermediate_poses() -> None:
    inversion = invert(CurvedMean(), SearchConfig(), 7, "curved")
    best, residual = inversion.best
    assert residual == pytest.approx(0.75, abs=0.01)
    assert abs(abs(best[2] - 250.0) - 100.0 / math.sqrt(2)) < 2.0  # not 250 ± 100, not 250
    assert classify(inversion, tau=1.0)["status"] in {"ambiguous", "optimizer_unresolved"}
    assert classify(inversion, tau=0.5)["status"] in {"residual_reject", "optimizer_unresolved"}


def test_all_invalid_geometry_is_unsupported() -> None:
    objective = Invalid()
    result = classify(invert(objective, SearchConfig(), 7, "invalid"), tau=1.0)
    assert result["status"] == "unsupported_geometry"
    assert objective.calls == 0
    assert result["invalid_attempts"] > 0


def test_tiny_budget_is_unresolved() -> None:
    config = SearchConfig(coarse=64, local_starts=1, local_evaluations=12, nuisance_evaluations=12)
    result = classify(invert(Bowl(), config, 7, "tiny"), tau=1.0)
    assert result["status"] == "optimizer_unresolved"
    assert result["converged"] == [False, False]


def _result(pose, residual=0.0, converged=True):
    return SearchResult(np.asarray(pose, float), residual, converged, 1)


def _inversion(searches, extra=()):
    trace = Trace()
    for result in searches:
        trace.poses.append(result.best_pose)
        trace.residuals.append(result.best_residual)
        trace.phases.append(1)
        trace.searches.append(0)
        trace.cached.append(False)
    for pose, residual in extra:
        trace.poses.append(np.asarray(pose, float))
        trace.residuals.append(residual)
        trace.phases.append(2)
        trace.searches.append(-1)
        trace.cached.append(False)
    return Inversion(list(searches), [], trace, SearchConfig())


def test_restart_disagreement_is_unresolved() -> None:
    a, b = TARGET.copy(), TARGET.copy()
    b[2] += 2.0  # 2 units apart, above the 1.28 restart tolerance
    assert classify(_inversion([_result(a), _result(b)]), 1.0)["status"] == "optimizer_unresolved"
    b[2] = TARGET[2] + 1.0
    assert classify(_inversion([_result(a), _result(b)]), 1.0)["status"] == "accepted"


def test_nuisance_escape_makes_a_query_ambiguous() -> None:
    far = TARGET.copy()
    far[3] += 5.0  # a near-optimal block centre 5 units away, found by a nuisance challenge
    inversion = _inversion([_result(TARGET), _result(TARGET)], extra=[(far, 0.01)])
    assert classify(inversion, tau=0.05)["status"] == "ambiguous"
    assert classify(inversion, tau=0.001)["status"] == "accepted"


def test_search_is_deterministic_and_respects_the_cap() -> None:
    config = SearchConfig()
    first = invert(Bowl(), config, 11, "same")
    second = invert(Bowl(), config, 11, "same")
    a, b = first.trace.arrays(), second.trace.arrays()
    assert all(np.array_equal(a[key], b[key]) for key in a)
    assert first.trace.evaluations <= config.evaluation_cap == 2048
    assert first.trace.evaluations == int((~a["cached"]).sum())
    other = invert(Bowl(), config, 11, "different-query")
    assert not np.array_equal(other.trace.arrays()["poses"][:5], a["poses"][:5])


def test_objective_calls_equal_charged_evaluations() -> None:
    objective = Bowl()
    inversion = invert(objective, SearchConfig(), 3, "count")
    assert objective.calls == inversion.trace.evaluations


def test_seed_streams_are_independent() -> None:
    a = seed_sequence(1, "q", 0).generate_state(4)
    assert not np.array_equal(a, seed_sequence(1, "q", 1).generate_state(4))
    assert not np.array_equal(a, seed_sequence(1, "r", 0).generate_state(4))
    assert np.array_equal(a, seed_sequence(1, "q", 0).generate_state(4))


def test_tau_rule_and_calibration_failure() -> None:
    values = {"archive": [0.1] * 11 + [0.9], "challenge": [0.2] * 12}
    expected = max(np.percentile(values["archive"], 95), np.percentile(values["challenge"], 95))
    assert calibrate_tau(values, epsilon=1e-12) == pytest.approx(expected)
    assert calibrate_tau({"a": [0.0, 0.0]}, epsilon=1e-6) == pytest.approx(1e-6)
    with pytest.raises(CalibrationFailure):
        calibrate_tau({"archive": [0.1, None]}, epsilon=1e-12)
    with pytest.raises(CalibrationFailure):
        calibrate_tau({"archive": [0.1, math.inf]}, epsilon=1e-12)


def test_config_round_trip_and_validation() -> None:
    config = SearchConfig()
    assert SearchConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        SearchConfig(batch=48)
    with pytest.raises(ValueError):
        SearchConfig(minimum_step=(0.0, 1, 1, 1, 1))


# --------------------------------------------------------------------------- version 2 (D037)
V2 = SearchConfig(version=2, local_starts=2, local_evaluations=320, refine_evaluations=300)


def _manual(searches, extra_pose=None, extra_residual=None, nuisance=(), tau_amb=0.01):
    """Hand-built evidence: two independent searches plus optional nuisance candidates."""
    trace = Trace()
    for index, (pose, residual, _) in enumerate(searches):
        trace.poses.append(np.asarray(pose, float))
        trace.residuals.append(residual)
        trace.phases.append(1)
        trace.searches.append(index)
        trace.cached.append(False)
    if extra_pose is not None:
        trace.poses.append(np.asarray(extra_pose, float))
        trace.residuals.append(extra_residual)
        trace.phases.append(2)
        trace.searches.append(-1)
        trace.cached.append(False)
    results = [
        SearchResult(np.asarray(pose, float), residual, converged, 64)
        for pose, residual, converged in searches
    ]
    return Inversion(results, list(nuisance), trace, V2, {"tau_amb": tau_amb})


def test_v2_never_reports_a_pose_only_a_nuisance_search_found() -> None:
    # MEAS-REVIEW-001 F3 reproduction: both searches converge at block (250, 260) with
    # residual 0.1; one nuisance candidate reaches (260, 260) with residual 0 and did
    # not converge. Version 1 accepted (260, 260); version 2 must not.
    pose = [100.0, 300.0, 250.0, 260.0, 1.0]
    far = [100.0, 300.0, 260.0, 260.0, 1.0]
    nuisance = [{"status": "fitted", "converged": False, "residual": 0.0, "block": far[2:4]}]
    inversion = _manual([(pose, 0.1, True), (pose, 0.1, True)], far, 0.0, nuisance)
    result = classify(inversion, tau=0.2)
    assert result["status"] == "optimizer_unresolved"
    assert result["reason"] == "nuisance_incomplete"
    converged = [{"status": "fitted", "converged": True, "residual": 0.0, "block": far[2:4]}]
    inversion = _manual([(pose, 0.1, True), (pose, 0.1, True)], far, 0.0, converged)
    result = classify(inversion, tau=0.2)
    assert result["status"] == "optimizer_unresolved"
    assert result["reason"] == "nuisance_better_basin"
    assert result["best_pose"][2:4] == [250.0, 260.0]  # reported pose is search-supported


def test_v2_treats_exhausted_nuisance_challenges_as_unresolved() -> None:
    pose = [100.0, 300.0, 250.0, 260.0, 1.0]
    budget = [{"status": "budget", "axis": 0, "offset": 5.12}]
    result = classify(_manual([(pose, 0.01, True), (pose, 0.01, True)], nuisance=budget), 0.2)
    assert result["status"] == "optimizer_unresolved" and result["reason"] == "nuisance_incomplete"
    invalid = [{"status": "invalid_start", "axis": 0, "offset": 5.12}]  # not applicable
    result = classify(_manual([(pose, 0.01, True), (pose, 0.01, True)], nuisance=invalid), 0.2)
    assert result["status"] == "accepted"


def test_v2_separates_acceptance_from_the_ambiguity_band() -> None:
    pose = [100.0, 300.0, 250.0, 260.0, 1.0]
    distant = [100.0, 300.0, 280.0, 260.0, 1.0]
    searches = [(pose, 0.01, True), (pose, 0.01, True)]
    # A distant candidate with residual 0.05 is inside a v1-style band of τ = 0.2 but far
    # outside a resolution band of 0.005, so v2 accepts while v1 semantics would not.
    inversion = _manual(searches, distant, 0.05, tau_amb=0.005)
    assert classify(inversion, tau=0.2)["status"] == "accepted"
    assert classify(inversion, tau=0.2, ambiguity_tau=0.2)["status"] == "ambiguous"
    assert classify(_manual(searches, tau_amb=None), tau=0.2)["reason"] == "resolution_unavailable"


class Valley(Bowl):
    """A narrow diagonal valley coupling agent x and block x (stalls axis-wise search)."""

    def evaluate(self, poses):
        self.calls += len(poses)
        d = _wrapped(poses, self.target)
        along = (d[:, 0] + d[:, 2]) / math.sqrt(2)
        across = (d[:, 0] - d[:, 2]) / math.sqrt(2)
        rest = d[:, 1] ** 2 + d[:, 3] ** 2 + (d[:, 4] / 0.02) ** 2
        return 1e-3 * (along**2 + 400 * across**2 + rest)


def test_v2_refinement_recovers_a_diagonal_valley_and_is_charged() -> None:
    objective = Valley()
    inversion = invert(objective, V2, 7, "valley")
    assert objective.calls >= inversion.trace.evaluations  # resolution probes are extra
    assert inversion.trace.evaluations <= V2.evaluation_cap
    reported, residual = inversion.reported
    assert np.linalg.norm(reported[BLOCK] - TARGET[BLOCK]) < 0.5
    assert inversion.resolution["tau_amb"] > 0
    result = classify(inversion, tau=0.05)
    assert result["status"] == "accepted", result
    phases = inversion.trace.arrays()["phases"]
    assert (phases == 3).any()  # refine phase recorded


def test_v1_configs_serialize_without_v2_keys() -> None:
    legacy = SearchConfig().legacy_dict()
    assert "version" not in legacy and "refine_evaluations" not in legacy
    assert SearchConfig.from_dict(legacy) == SearchConfig()
    assert SearchConfig.from_dict(V2.to_dict()) == V2
    assert (
        V2.evaluation_cap
        == SearchConfig(local_starts=2, local_evaluations=320).evaluation_cap + 600
    )
