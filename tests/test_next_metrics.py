"""Behavioral fixtures for the NEXT-001 pure measurements and D021 decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.latent_metrics import cost_matrix, pair_rows
from jepa_action_effects.next_metrics import (
    apply_criteria,
    archive_blocks,
    block_features,
    bootstrap,
    boundary_pairs,
    family_displacements,
    floor_ratio,
    latent_pair_totals,
    mean_of,
    summarize_phys,
    support_summary,
)

CHALLENGE_IDS = ["hold", "lateral_0", "lateral_1", "lateral_2", "lateral_3", "push"]


def _challenge_controls() -> np.ndarray:
    toward = np.array([0.0, -1.0])
    lateral = np.array([1.0, 0.0])
    vectors = [np.zeros(2)] + [toward * 0.6 + lateral * offset for offset in (-0.2, -0.1, 0.1, 0.2)]
    vectors.append(toward * 0.6)
    return np.stack([np.tile(vector, (10, 1)) for vector in vectors])


def test_block_features_magnitude_and_spread() -> None:
    constant = np.tile([0.6, 0.0], (5, 1))
    varied = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=float)
    features = block_features(np.stack([constant, varied]))
    assert features["magnitude"] == pytest.approx([0.6, 0.2])
    assert features["spread"] == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError):
        block_features(np.zeros((4, 2)))


def test_archive_blocks_respect_episodes_and_drop_nan() -> None:
    actions = np.arange(24, dtype=float).reshape(12, 2)
    actions[7] = np.nan
    blocks = archive_blocks(actions, np.array([0, 6]), np.array([6, 6]))
    # Episode 0 contributes rows 0-4; episode 1 rows 6-10 contain the NaN row 7.
    assert blocks.shape == (1, 5, 2)
    assert blocks[0, 0].tolist() == [0.0, 1.0]


def test_support_summary_flags_out_of_support_constant_pushes() -> None:
    rng = np.random.default_rng(0)
    archive = rng.normal(0, 0.2, size=(5000, 5, 2))
    familiar = archive[:50] + 1e-3
    extreme = np.tile([0.9, 0.0], (50, 5, 1))
    ordinary = support_summary(archive, familiar, seed=1, sample=3000)
    unusual = support_summary(archive, extreme, seed=1, sample=3000)
    assert ordinary["study_fraction_above_archive_p99_magnitude"] < 0.2
    assert unusual["study_fraction_above_archive_p99_magnitude"] == 1.0
    assert unusual["study_fraction_joint_tail_below_1e-3"] == 1.0
    assert unusual["study_fraction_nn_beyond_archive_p99"] == 1.0


def test_boundary_pairs_exclude_hold_and_distant_laterals() -> None:
    pairs = boundary_pairs(_challenge_controls(), CHALLENGE_IDS)
    names = {(CHALLENGE_IDS[i], CHALLENGE_IDS[j]) for i, j in pairs}
    assert ("lateral_0", "lateral_1") in names
    assert ("lateral_1", "push") in names
    assert ("lateral_0", "push") in names  # offset 0.2 sits on the tolerance
    assert ("lateral_0", "lateral_3") not in names  # 0.4 apart
    assert not any("hold" in pair for pair in names)


def test_latent_pair_slope_recovers_shrinkage_from_cost_terms() -> None:
    rng = np.random.default_rng(3)
    real = rng.normal(size=(6, 192))
    predicted = 0.6 * real + 0.1
    rows = pair_rows(
        cost_matrix(predicted, real),
        np.zeros(6, dtype=bool),
        np.zeros((6, 2)),
        np.zeros((6, 2)),
        np.zeros(2),
        np.zeros(6, dtype=bool),
        predicted,
        real,
    )
    totals = latent_pair_totals(rows, lambda row: True)
    assert totals["count"] == 15
    assert totals["inner"] / totals["energy"] == pytest.approx(0.6)


def _family(scale: float, readout_noise: float, rng, cohort: str = "challenge") -> dict:
    start = np.array([250.0, 250.0])
    contacts = [False, True, True, False, True, True]
    true_disp = np.array([[0, 0], [30, 0], [25, 5], [0, 0], [40, -10], [35, 0]], dtype=float)
    true_disp = true_disp * rng.uniform(0.8, 1.2)
    end = start + true_disp
    bias = np.array([7.0, -3.0])  # constant readout bias must cancel
    decoded_start = start + bias
    real = end + bias + rng.normal(0, readout_noise, size=(6, 2))
    pred = decoded_start + scale * true_disp + rng.normal(0, readout_noise, size=(6, 2))
    record = family_displacements(
        pred,
        real,
        decoded_start,
        end,
        start,
        contacts,
        boundary_pairs(_challenge_controls(), CHALLENGE_IDS),
        goals=[(256.0, 256.0), (384.0, 128.0)],
    )
    return {"cohort": cohort, **record}


def test_family_displacements_oracle_bias_cancels_and_slopes() -> None:
    record = _family(1.0, 0.0, np.random.default_rng(0))
    assert record["endpoint"]["D"] == pytest.approx(0.0)
    assert record["boundary_all"]["D"] == pytest.approx(0.0)
    terms = record["slope"]["moved"]
    assert terms["pred_inner"] / terms["energy"] == pytest.approx(1.0)
    assert terms["real_inner"] / terms["energy"] == pytest.approx(1.0)
    assert record["regret"]["jepa"] == pytest.approx(0.0)
    shrunk = _family(0.5, 0.0, np.random.default_rng(0))
    moved = shrunk["slope"]["moved"]
    assert moved["pred_inner"] / moved["energy"] == pytest.approx(0.5)
    assert shrunk["boundary_all"]["D"] > 0


def test_bootstrap_preserves_cohorts_and_skips_missing_values() -> None:
    rows = [
        {"cohort": "challenge", "value": {"x": 1.0}},
        {"cohort": "challenge", "value": None},
        {"cohort": "ordinary", "value": {"x": 3.0}},
    ]
    result = bootstrap(rows, mean_of(("value", "x")), seed=4, samples=200)
    assert result["estimate"] == 2.0
    assert result["families"] == 3
    assert result["valid_draws"] >= 100


def test_decision_rules_distinguish_oracle_shrinkage_and_noisy_floor() -> None:
    rng = np.random.default_rng(11)
    oracle = [_family(1.0, 0.5, rng) for _ in range(40)]
    shrunk = [_family(0.5, 0.5, rng) for _ in range(40)]
    noisy = [_family(0.5, 25.0, rng) for _ in range(40)]
    oracle_summary = summarize_phys(oracle, seed=5, samples=400)
    shrunk_summary = summarize_phys(shrunk, seed=5, samples=400)
    noisy_summary = summarize_phys(noisy, seed=5, samples=400)
    assert oracle_summary["criteria"]["floor"] == "passed"
    assert oracle_summary["criteria"]["physical_failure"] == "not_established"
    assert oracle_summary["criteria"]["shrinkage"] == "not_established"
    assert shrunk_summary["criteria"] == {
        "floor": "passed",
        "physical_failure": "supported",
        "shrinkage": "supported",
        "h1": shrunk_summary["criteria"]["h1"],
    }
    assert noisy_summary["criteria"]["floor"] == "failed"
    assert noisy_summary["criteria"]["shrinkage"] == "inconclusive_readout_floor"
    assert floor_ratio(noisy) > floor_ratio(oracle)


def test_criteria_require_intervals() -> None:
    missing = {"estimate": None, "ci95": None}
    summary = {
        "floor_ratio": {"estimate": 0.1, "ci95": None},
        "D_boundary": missing,
        "slope_pred": missing,
        "slope_pred_minus_real": missing,
        "h1_contrast": missing,
    }
    assert apply_criteria(summary)["physical_failure"] == "not_established"
