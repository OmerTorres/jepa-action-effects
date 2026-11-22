"""Derived SUPPORT-001 account slopes: projection slopes, not account parameters."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.support001_derived import account_slopes, command_tail


def _unit(t, m, w_logit=0.0):
    t, m = np.atleast_2d(np.asarray(t, float)), np.atleast_2d(np.asarray(m, float))
    n = len(t)
    return {
        "t": t,
        "m": m,
        "log_support": np.full(n, w_logit),
        "magnitude": np.full(n, 0.6),
    }


FITS = {"A": {"params": [1.0, -0.5]}, "B": {"params": [0.0, 1.0]}, "C": {"params": [0.3]}}


def test_reversion_slope_includes_typical_transition_and_differs_from_its_weight():
    # w = sigmoid(0) = 0.5 for every unit; the slope still depends on m and t.
    short = account_slopes(FITS, [_unit([10.0, 0.0], [5.0, 0.0])])
    long = account_slopes(FITS, [_unit([20.0, 0.0], [5.0, 0.0])])
    assert short["B"] == pytest.approx(0.75)
    assert long["B"] == pytest.approx(0.625)
    assert short["B"] != pytest.approx(0.5)


def test_gain_accounts_reduce_to_their_gain_and_zero_energy_is_undefined():
    slopes = account_slopes(FITS, [_unit([3.0, 4.0], [1.0, 1.0]), _unit([-6.0, 1.0], [0, 0])])
    assert slopes["A"] == pytest.approx(0.7)
    assert slopes["C"] == pytest.approx(0.5)
    empty = account_slopes(FITS, [_unit([0.0, 0.0], [5.0, 0.0])])
    assert empty == {"A": None, "B": None, "C": None}


def test_command_tail_counts_windows_above_thresholds():
    tail = command_tail(np.array([[0.3, 0.4], [0.0, 0.7], [0.81, 0.0]]))
    assert tail == {"windows": 3, "max": pytest.approx(0.81), "above_0.60": 2, "above_0.80": 1}
