"""TRAIN-001 gates, paired bootstrap, verdict truth tables and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.train001_metrics import (
    _contrasts,
    cell_gate,
    derangement,
    fit_calibration,
    magnitude_diagnostics,
    ordinary_outcome,
    paired_bootstrap,
    pooled_verdict,
    stratified_indices,
)
from jepa_action_effects.train001_protocol import DECISION_RULES, Protocol

PROTOCOL = Protocol(bootstrap_draws=300)
V = DECISION_RULES["verdicts"]


def _cell(n=60, moved=60, noise=0.5, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    true = np.zeros((n, 2))
    true[:moved] = rng.normal(size=(moved, 2)) * 5 + np.array([20.0, 0.0])
    real = gain * true + rng.normal(scale=noise, size=(n, 2))
    error = np.linalg.norm(real - true, axis=1)
    return true, real, error


def test_accurate_cell_passes_all_checks():
    gate = cell_gate(*_cell(), PROTOCOL, 1)
    assert gate["passed"], gate
    assert 0.85 < gate["slope_real_ci"][0] and gate["slope_real_ci"][1] < 1.15


def test_sparse_cell_fails_the_moved_count():
    gate = cell_gate(*_cell(n=60, moved=40), PROTOCOL, 1)
    assert gate["moved"] == 40
    assert not gate["checks"]["moved_count"] and not gate["passed"]


def test_degenerate_cell_with_no_motion_fails():
    gate = cell_gate(*_cell(n=20, moved=0), PROTOCOL, 1)
    assert gate["slope_real"] is None and gate["slope_real_ci"] is None and gate["floor"] is None
    assert not gate["passed"]


def test_attenuated_readout_fails_the_band():
    gate = cell_gate(*_cell(gain=0.7), PROTOCOL, 1)
    assert not gate["checks"]["slope_interval_in_band"]


def test_floor_uses_all_families_over_median_moved_displacement():
    true = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [1.0, 0.0]])
    error = np.array([1.0, 2.0, 3.0, 10.0])
    gate = cell_gate(true, true, error, Protocol(min_moved_families=1, bootstrap_draws=50), 0)
    # numerator mean over all four (4.0); denominator median over moved (20.0)
    assert gate["floor"] == pytest.approx(0.2)
    assert gate["mean_real_displacement_error"] == pytest.approx(4.0)
    assert gate["median_moved_true_displacement"] == pytest.approx(20.0)


def test_stratified_indices_preserve_scene_counts():
    scenes = np.array([0] * 7 + [1] * 3)
    index = stratified_indices(scenes, np.random.default_rng(0))
    assert np.sum(scenes[index] == 0) == 7 and np.sum(scenes[index] == 1) == 3


def test_bootstrap_resamples_families_jointly_across_conditions():
    rng = np.random.default_rng(0)
    scenes = np.array([0] * 30 + [1] * 30)
    family_effect = rng.normal(scale=10, size=60)
    balanced, archive = family_effect + 1.0, family_effect + 3.0
    boot = paired_bootstrap({"diff": balanced - archive, "abs": balanced}, scenes, 200, 3)
    # Paired: the constant difference has zero-width CI although families vary widely.
    assert boot["diff"]["ci"] == pytest.approx([-2.0, -2.0])
    assert boot["abs"]["ci"][1] - boot["abs"]["ci"][0] > 1.0
    assert set(boot["diff"]["by_scene"]) == {"challenge", "archive"}


def test_scene_categories_receive_equal_weight():
    scenes = np.array([0] * 90 + [1] * 10)
    values = np.concatenate([np.zeros(90), np.ones(10)])
    assert paired_bootstrap({"x": values}, scenes, 20, 0)["x"]["estimate"] == pytest.approx(0.5)


def test_training_seeds_stay_distinct():
    seeds = (1, 2, 3)
    values = {"frozen": np.zeros(4)}
    for seed in seeds:
        values[f"balanced-{seed}"] = np.full(4, float(seed))
        values[f"archive_weighted-{seed}"] = np.zeros(4)
    contrasts = _contrasts(values, seeds)
    assert [contrasts[f"balanced_minus_archive@{s}"][0] for s in seeds] == [1.0, 2.0, 3.0]
    assert contrasts["balanced_minus_archive"][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("development", "confirmation", "ba", "bf", "expected"),
    [
        (False, True, [-2, -1], [-2, -1], V["development_fail"]),
        (False, False, [-2, -1], [-2, -1], V["development_fail"]),
        (True, False, [-2, -1], [-2, -1], V["confirmation_fail"]),
        (True, True, [-2, -1], [-2, -1], V["both_ci_below_zero"]),
        (True, True, [-2, -1], [-1, 1], V["otherwise"]),
        (True, True, [1, 2], [-1, 1], V["required_ci_above_zero"]),
        (True, True, [-2, -1], [0.5, 1], V["required_ci_above_zero"]),
        (True, True, None, [-2, -1], V["otherwise"]),
    ],
)
def test_pooled_verdict_truth_table(development, confirmation, ba, bf, expected):
    assert pooled_verdict(development, confirmation, ba, bf)["verdict"] == expected


def test_subgroup_success_cannot_set_the_pooled_verdict():
    # Development failure stays not assessable even with favorable confirmation intervals.
    result = pooled_verdict(False, True, [-5, -4], [-5, -4])
    assert result["verdict"] == V["development_fail"]


def test_error_increase_names_the_contrast():
    result = pooled_verdict(True, True, [-1, 1], [0.2, 0.9])
    assert result["error_increase_contrasts"] == ["balanced_minus_frozen"]


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ([-1.0, 1.0], "non_inferiority_within_tolerance_established"),
        ([0.5, 2.0], "non_inferiority_not_established"),
        ([1.5, 2.0], "regression_beyond_tolerance_supported"),
        (None, "non_inferiority_not_established"),
    ],
)
def test_ordinary_outcome_table(interval, expected):
    assert ordinary_outcome(interval, 1.28) == expected


def test_calibration_closed_form_and_clamp():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(10, 4, 2)) * 10
    scenes = np.array([0] * 5 + [1] * 5)
    assert fit_calibration(0.5 * true, true, scenes) == pytest.approx(2.0)
    assert fit_calibration(-true, true, scenes) == 0.0
    with pytest.raises(ValueError):
        fit_calibration(np.zeros_like(true), true, scenes)


def test_calibration_weights_scenes_equally():
    true = np.ones((4, 1, 2))
    predicted = np.stack([np.ones((1, 2))] * 3 + [2 * np.ones((1, 2))])
    scenes = np.array([0, 0, 0, 1])
    # weights 1/6 each in scene 0, 1/2 in scene 1: num = 0.5*2 + 0.5*4 ... closed form
    numerator = 3 * (1 / 6) * 2 + 0.5 * 4
    denominator = 3 * (1 / 6) * 2 + 0.5 * 8
    assert fit_calibration(predicted, true, scenes) == pytest.approx(numerator / denominator)


def test_derangement_has_no_fixed_points_and_is_order_independent():
    for unit in ("a", "b", "challenge-conf-001"):
        permutation = derangement(unit, 7)
        assert sorted(permutation) == [0, 1, 2, 3]
        assert all(p != i for i, p in enumerate(permutation))
        assert permutation == derangement(unit, 7)


def test_zero_predicted_displacement_has_undefined_direction():
    n = 4
    table = {
        "scenes": np.zeros(n, dtype=int),
        "true_start": np.zeros((n, 2)),
        "true_end": np.tile(np.array([[10.0, 0.0]] * 4), (n, 1, 1)),
        "decoded_start": np.zeros((n, 2)),
        "real_end": np.tile(np.array([[10.0, 0.0]] * 4), (n, 1, 1)),
    }
    result = magnitude_diagnostics(table, np.zeros((n, 4, 2)), 5.12)
    cell = result["challenge:m0.60"]
    assert cell["decomposition"]["undefined_angle_count"] == n
    assert cell["decomposition"]["abs_angle_mean_deg"] is None
    assert cell["slope_pred"] == 0.0
