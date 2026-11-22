"""Fixtures for the REVIEW-BRANCH-002 follow-up measurements."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.followup_metrics import (
    absolute_endpoint_error,
    cluster_bootstrap,
    decompose,
    displacement_error,
    max_abs_difference,
)


def test_shared_position_bias_cancels_in_displacement_but_not_endpoint() -> None:
    true_start, true_end = np.array([200.0, 200.0]), np.array([[230.0, 210.0]])
    bias = np.array([100.0, 100.0])
    decoded_start, decoded_end = true_start + bias, true_end + bias
    assert displacement_error(decoded_end, decoded_start, true_end, true_start)[0] == pytest.approx(
        0.0
    )
    assert absolute_endpoint_error(decoded_end, true_end)[0] == pytest.approx(141.42, abs=0.01)


def _rotate(vectors, degrees):
    a = np.radians(degrees)
    return vectors @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def test_decomposition_separates_length_from_direction() -> None:
    rng = np.random.default_rng(0)
    true = rng.normal(size=(200, 2)) * 20 + 1
    shorter = decompose(0.6 * true, true)
    assert shorter["length_ratio_pooled"] == pytest.approx(0.6)
    assert shorter["abs_angle_mean_deg"] == pytest.approx(0.0, abs=1e-9)
    assert shorter["projection_slope"] == pytest.approx(0.6)
    turned = decompose(_rotate(true, 40.0), true)
    assert turned["length_ratio_pooled"] == pytest.approx(1.0)
    assert turned["abs_angle_mean_deg"] == pytest.approx(40.0)
    assert turned["projection_slope"] == pytest.approx(np.cos(np.radians(40.0)))
    with pytest.raises(ValueError):
        decompose(true[:1], np.zeros((1, 2)))


def test_cluster_bootstrap_widens_for_duplicated_clusters() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=60)
    duplicated = [[v] * 5 for v in values]  # five identical windows per episode

    def mean(rows):
        return float(np.mean(rows))

    naive = cluster_bootstrap([[v] for c in duplicated for v in c], mean, seed=2, samples=500)
    clustered = cluster_bootstrap(duplicated, mean, seed=2, samples=500)
    assert naive["estimate"] == pytest.approx(clustered["estimate"])
    width = lambda r: r["ci95"][1] - r["ci95"][0]  # noqa: E731
    assert width(clustered) > 1.8 * width(naive)


def test_max_abs_difference_and_mismatches() -> None:
    left = {"a": 1.0, "b": [1, 2.5], "c": {"d": True}, "skip": 3}
    right = {"a": 1.25, "b": [1, 2.5], "c": {"d": True}, "skip": 99}
    assert max_abs_difference(left, right, frozenset({"skip"})) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        max_abs_difference({"a": 1}, {"b": 1})
    with pytest.raises(ValueError):
        max_abs_difference({"c": {"d": True}}, {"c": {"d": False}})


def test_zero_predicted_vector_has_undefined_direction() -> None:
    only_zero = decompose(np.zeros((1, 2)), np.array([[10.0, 0.0]]))
    assert only_zero["abs_angle_mean_deg"] is None
    assert only_zero["abs_angle_median_deg"] is None
    assert only_zero["undefined_angle_count"] == 1
    assert only_zero["length_ratio_pooled"] == pytest.approx(0.0)
    assert only_zero["projection_slope"] == pytest.approx(0.0)
    mixed = decompose(np.array([[0.0, 0.0], [0.0, 5.0]]), np.array([[10.0, 0.0], [5.0, 0.0]]))
    assert mixed["abs_angle_mean_deg"] == pytest.approx(90.0)  # only the defined vector
    assert mixed["undefined_angle_count"] == 1
    assert mixed["count"] == 2
