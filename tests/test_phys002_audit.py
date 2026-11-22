"""Model-free fixtures for the PHYS-002 audit helpers."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.phys002_audit import (
    alignment_matrix,
    binned_slopes,
    latent_statistics,
    slope,
)


def _family(pred_scale: tuple[float, float], offset: bool = False) -> dict:
    rng = np.random.default_rng(0)
    step1 = rng.normal(scale=20.0, size=(6, 2))
    true = np.stack([step1, 2 * step1], axis=1)  # constant velocity
    start = np.array([3.0, -2.0])  # decoded start bias must cancel
    if offset:
        pred = np.stack([np.zeros((6, 2)), step1], axis=1)
    else:
        pred = true * np.asarray(pred_scale)[None, :, None]
    return {"true": true, "start": start, "pred": pred + start, "real": true + start}


def test_slope_recovers_scale_and_rejects_zero_energy() -> None:
    true = np.array([[1.0, 2.0], [-3.0, 0.5]])
    assert slope(0.6 * true, true) == pytest.approx(0.6)
    assert slope(true, np.zeros((2, 2))) is None


def test_alignment_matrix_separates_offset_from_shrinkage() -> None:
    shrunk = alignment_matrix([_family((0.9, 0.6))])
    assert shrunk["real_step2_vs_true2"] == pytest.approx(1.0)
    assert shrunk["pred_step1_vs_true1"] == pytest.approx(0.9)
    assert shrunk["pred_step2_vs_true2"] == pytest.approx(0.6)
    shifted = alignment_matrix([_family((1.0, 1.0), offset=True)])
    assert shifted["pred_step1_vs_true1"] == pytest.approx(0.0)
    assert shifted["pred_step2_vs_true1"] == pytest.approx(1.0)


def test_latent_statistics_oracle_and_persistence() -> None:
    rng = np.random.default_rng(1)
    z_start = rng.normal(size=192)
    z_real = z_start + rng.normal(size=(6, 2, 192))
    oracle = latent_statistics([{"z_start": z_start, "z_real": z_real, "z_pred": z_real}])
    assert oracle == pytest.approx({"latent_slope": 1.0, "pred_over_persistence_mse": 0.0})
    persistence = np.broadcast_to(z_start, z_real.shape)
    still = latent_statistics([{"z_start": z_start, "z_real": z_real, "z_pred": persistence}])
    assert still == pytest.approx({"latent_slope": 0.0, "pred_over_persistence_mse": 1.0})


def test_binned_slopes_counts_by_true_magnitude() -> None:
    true = np.array([[10.0, 0.0], [30.0, 0.0], [60.0, 0.0], [2.0, 0.0]])
    rows = binned_slopes(0.5 * true, true)
    assert [cell["count"] for cell in rows.values()] == [1, 1, 1, 0]
    assert rows["5.12-20"]["slope_pred"] == pytest.approx(0.5)
    assert rows["80-400"]["slope_pred"] is None
