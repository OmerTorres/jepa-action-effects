"""Fixtures for the CALIB-001 calibration-side measurements (D030)."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.calib001_metrics import (
    classify,
    estimates,
    interval,
    row_statistics,
    summarize_group,
)


def _stats(predicted, real, true, delta=0.0, latent=None) -> dict[str, float]:
    summed = row_statistics(predicted, real, true, delta, latent).sum(axis=0)
    return {key: float(value) for key, value in estimates(summed).items()}


def _hedging(n: int = 20000, seed: int = 0):
    """t = 20·x·u + η with η independent of x; p = E[t | x] = 20·x·u."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    mean = 20.0 * x * np.array([[0.6, 0.8]])
    true = mean + rng.normal(scale=10.0, size=(n, 2))
    return mean, true


def test_conditional_mean_is_calibrated_while_projection_slope_is_below_one() -> None:
    predicted, true = _hedging()
    result = _stats(predicted, true, true)
    # E[p·t] = 400 and E|t|² = 400 + 200, so the projection slope is about 2/3.
    assert result["b_rev"] == pytest.approx(2 / 3, abs=0.02)
    assert result["b_cal_k0"] == pytest.approx(1.0, abs=0.02)
    assert result["b_rev"] == pytest.approx(result["r2_k0"] / result["b_cal_k0"], rel=1e-12)


def test_uniform_shrinkage_puts_the_calibration_slope_above_one() -> None:
    true = np.random.default_rng(1).normal(scale=15.0, size=(500, 2))
    result = _stats(0.5 * true, true, true)
    assert result["b_rev"] == pytest.approx(0.5)
    assert result["b_cal_k0"] == pytest.approx(2.0)
    assert result["r2_k0"] == pytest.approx(1.0)


def test_reversion_toward_a_typical_effect_crosses_one() -> None:
    typical = np.array([15.0, 0.0])
    small, large = np.array([[5.0, 0.0]] * 10), np.array([[40.0, 0.0]] * 10)
    below = _stats(0.6 * small + 0.4 * typical, small, small)["b_cal_k0"]
    above = _stats(0.6 * large + 0.4 * typical, large, large)["b_cal_k0"]
    assert below == pytest.approx(45 / 81)  # small effects over-predicted
    assert above == pytest.approx(1200 / 900)  # large effects under-predicted


def test_noise_correction_undoes_readout_attenuation() -> None:
    rng = np.random.default_rng(2)
    true = rng.normal(scale=10.0, size=(20000, 2))
    predicted = true + rng.normal(scale=5.0, size=true.shape)
    real = true + rng.normal(scale=5.0, size=true.shape)
    result = _stats(predicted, real, true)
    assert result["b_rev"] == pytest.approx(1.0, abs=0.02)  # noise in p is harmless here
    assert result["b_cal_k0"] == pytest.approx(200 / 250, abs=0.02)
    assert result["b_cal_k1"] == pytest.approx(1.0, abs=0.03)
    assert result["b_cal_k0"] < result["b_cal_k0.5"] < result["b_cal_k1"] < result["b_cal_k2"]


def test_selecting_moved_rows_biases_the_calibration_slope_upward() -> None:
    predicted, true = _hedging(seed=3)
    everything = _stats(predicted, true, true)["b_cal_k0"]
    moved = np.linalg.norm(true, axis=1) > 25.0
    selected = _stats(predicted[moved], true[moved], true[moved])["b_cal_k0"]
    assert everything == pytest.approx(1.0, abs=0.02)
    assert selected > everything + 0.1


def test_static_excess_measures_spurious_predicted_motion() -> None:
    real = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    true = np.zeros_like(real)
    result = _stats(real + np.array([3.0, 4.0]), real, true, delta=5.12)
    assert result["n_moved"] == 0
    assert result["static_excess"] == pytest.approx(25.0)
    assert np.isnan(result["b_rev"])  # no motion to project onto
    assert np.isnan(result["b_cal_k0"])  # nor a calibration slope, which would be 0 trivially


def test_latent_slopes_need_no_readout() -> None:
    rng = np.random.default_rng(4)
    change = rng.normal(size=(6, 192))
    vectors = rng.normal(size=(6, 2))
    result = _stats(vectors, vectors, vectors, latent={1: (0.5 * change, change)})
    assert result["z1_b_rev"] == pytest.approx(0.5)
    assert result["z1_b_cal"] == pytest.approx(2.0)
    assert np.isnan(result["z2_b_cal"])  # step 2 not supplied


def test_nonpositive_corrected_denominator_is_undefined() -> None:
    rng = np.random.default_rng(5)
    true = rng.normal(scale=3.0, size=(40, 2))
    real = true + rng.normal(scale=10.0, size=true.shape)  # readout noise swamps Σ|p|²
    result = _stats(0.1 * true, real, true)
    assert result["b_cal_k0"] == pytest.approx(10.0)
    assert np.isnan(result["b_cal_k1"])
    rows = row_statistics(0.1 * true, real, true, 5.12)
    group = summarize_group(["noisy"], rows[:, None, :], draws=200, seed=6)
    assert group["cells"]["noisy"]["class_k1"] == "undefined"
    assert group["cells"]["noisy"]["b_cal_k1"]["estimate"] is None


def test_classification_rule() -> None:
    assert classify([1.05, 1.4]) == "above_1"
    assert classify([0.5, 0.95]) == "below_1"
    assert classify([0.9, 1.1]) == "contains_1"
    assert classify(None) == "undefined"


def test_interval_needs_half_of_the_draws_defined() -> None:
    assert interval(np.array([np.nan] * 6 + [1.0] * 4)) is None
    assert interval(np.array([np.nan] * 4 + [1.0] * 6)) == [1.0, 1.0]


def test_group_bootstrap_is_deterministic_and_paired() -> None:
    rng = np.random.default_rng(7)
    true = rng.normal(scale=10.0, size=(30, 6, 2))
    predicted = 0.7 * true + rng.normal(scale=3.0, size=true.shape)
    units = np.stack(
        [row_statistics(predicted[u], true[u], true[u], 5.12).sum(axis=0) for u in range(30)]
    )
    sums = np.stack([units, units], axis=1)  # two identical cells
    contrast = [("same", "a", "b", "b_cal_k1")]
    first = summarize_group(["a", "b"], sums, draws=300, seed=8, contrasts=contrast)
    second = summarize_group(["a", "b"], sums, draws=300, seed=8, contrasts=contrast)
    assert first == second
    assert first["contrasts"]["same"]["estimate"] == 0.0
    assert first["contrasts"]["same"]["ci95"] == [0.0, 0.0]
    assert first["cells"]["a"]["class_k1"] == "above_1"
    with pytest.raises(ValueError):
        summarize_group(["a"], sums, draws=10, seed=8)


# ---------------------------------------------------------------- limits of the scalar gain (D031)
def test_exact_latent_mean_through_nonlinear_decoder_departs_from_one() -> None:
    # Latent z ∈ {1, 3} equally likely and t = (z², 0). The optimal latent predictor gives
    # E[z] = 2, which an exact decoder maps to p = (4, 0), whereas E[t] = (5, 0).
    result = _stats([[4.0, 0.0], [4.0, 0.0]], [[1.0, 0.0], [9.0, 0.0]], [[1.0, 0.0], [9.0, 0.0]])
    assert result["b_cal_k0"] == pytest.approx(1.25)  # not evidence of predictor bias


def test_pure_rotation_lowers_both_slopes_without_changing_magnitude() -> None:
    true = np.random.default_rng(9).normal(scale=20.0, size=(500, 2))
    angle = np.radians(60.0)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    result = _stats(true @ rotation, true, true)
    assert result["b_rev"] == pytest.approx(0.5)
    assert result["b_cal_k0"] == pytest.approx(0.5)  # below 1, yet no excess magnitude
    assert result["energy_ratio"] == pytest.approx(1.0)
    assert result["alignment"] == pytest.approx(0.5)
    assert result["r2_k0"] == pytest.approx(0.25)  # alignment², not lost information


def test_unit_scalar_gain_does_not_imply_conditional_calibration() -> None:
    result = _stats([[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [1.0, 0.0]], [[3.0, 0.0], [1.0, 0.0]])
    assert result["b_cal_k0"] == pytest.approx(1.0)  # yet each prediction is wrong


def test_slopes_factor_into_magnitude_and_alignment() -> None:
    rng = np.random.default_rng(10)
    true = rng.normal(scale=10.0, size=(300, 2))
    result = _stats(0.7 * true + rng.normal(scale=4.0, size=true.shape), true, true)
    g = np.sqrt(result["energy_ratio"])
    assert result["b_rev"] == pytest.approx(result["alignment"] * g)
    assert result["b_cal_k0"] == pytest.approx(result["alignment"] / g)
