"""CALIB-001 measurements: scalar gains of decoded displacements (exploratory, D030/D031).

Write C = Σp·t, P = Σ|p|² and T = Σ|t|² for decoded predicted displacements p and simulator
displacements t. The two through-origin scalar slopes are:
- the projection slope b_rev = C/T, which regresses the prediction on the outcome;
- the scalar gain b_cal = C/P, the optimal through-origin factor mapping predictions to
  outcomes.

With magnitude ratio g = sqrt(P/T) and uncentered alignment ρ = C/sqrt(PT), they factor as
b_rev = ρ·g and b_cal = ρ/g. This is algebra, not a decomposition into causes: a pure
rotation of exact displacements leaves g = 1 and lowers both slopes. Magnitude (g) and
alignment (ρ) are therefore reported separately.

What b_cal can and cannot show (D031):
- b_cal = 1 is necessary, not sufficient, for p = E[t | inputs] in physical units.
- The model predicts latents. A nonlinear decoder does not commute with expectation, so
  b_cal ≠ 1 does not separate predictor bias from latent uncertainty acting through
  decoder geometry.

The optional adjustment subtracts k·Σ|r − t|², the real-frame displacement error, from P.
It is a sensitivity analysis under an additive error model p = p* + e with E[p*·e] = 0 and
E[e·t] = 0 and error energy k times the real-frame error. It does not recover physical
truth. k = 0, the directly measured statistic, is primary.

Every estimand is a ratio of sums. Each resampling unit (a family, or an ordinary window)
is therefore reduced once to per-cell sufficient statistics, and the bootstrap sums
resampled units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

K_VALUES = (0.0, 0.5, 1.0, 2.0)
STEPS = (1, 2)
STATISTICS = (
    "n",
    "pt",
    "pp",
    "tt",
    "rt",
    "rr",
    "noise",
    "n_moved",
    "pt_moved",
    "tt_moved",
    "rt_moved",
    "n_static",
    "pp_static",
    "rr_static",
    *(f"z{step}_{name}" for step in STEPS for name in ("pt", "pp", "tt")),
)
INDEX = {name: position for position, name in enumerate(STATISTICS)}
REPORTED = (
    "b_rev",
    *(f"b_cal_k{k:g}" for k in K_VALUES),
    "r2_k0",
    "r2_k1",
    "energy_ratio",
    "alignment",
    "b_cal_real",
    "noise_rms",
    "static_excess",
    *(f"z{step}_{name}" for step in STEPS for name in ("b_rev", "b_cal", "energy_ratio")),
    "b_rev_moved",
    "ratio_moved",
    "slope_real_moved",
)


def row_statistics(
    predicted: np.ndarray,
    real: np.ndarray,
    true: np.ndarray,
    delta: float,
    latent: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Per-row sufficient statistics, shape (N, len(STATISTICS)).

    `predicted`, `real` and `true` are (N, 2) displacements: decoded prediction, decoded
    real future and simulator truth. `latent` optionally maps a step to (Δẑ, Δz), the
    predicted and real changes of the projected latent from the current frame.
    """
    p = np.asarray(predicted, float).reshape(-1, 2)
    r = np.asarray(real, float).reshape(-1, 2)
    t = np.asarray(true, float).reshape(-1, 2)
    if not len(p) == len(r) == len(t):
        raise ValueError("predicted, real and true need the same number of rows")
    moved = (np.linalg.norm(t, axis=1) > delta).astype(float)
    static = 1.0 - moved
    pt, pp, tt = np.sum(p * t, 1), np.sum(p * p, 1), np.sum(t * t, 1)
    rt, rr = np.sum(r * t, 1), np.sum(r * r, 1)
    columns = {
        "n": np.ones(len(t)),
        "pt": pt,
        "pp": pp,
        "tt": tt,
        "rt": rt,
        "rr": rr,
        "noise": np.sum((r - t) ** 2, 1),
        "n_moved": moved,
        "pt_moved": pt * moved,
        "tt_moved": tt * moved,
        "rt_moved": rt * moved,
        "n_static": static,
        "pp_static": pp * static,
        "rr_static": rr * static,
    }
    for step, (change_predicted, change_real) in (latent or {}).items():
        a = np.asarray(change_predicted, float).reshape(len(t), -1)
        b = np.asarray(change_real, float).reshape(len(t), -1)
        columns[f"z{step}_pt"] = np.sum(a * b, 1)
        columns[f"z{step}_pp"] = np.sum(a * a, 1)
        columns[f"z{step}_tt"] = np.sum(b * b, 1)
    rows = np.zeros((len(t), len(STATISTICS)))
    for name, values in columns.items():
        rows[:, INDEX[name]] = values
    return rows


def _ratio(numerator: Any, denominator: Any) -> np.ndarray:
    """numerator / denominator where the denominator is positive, NaN elsewhere."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, float), np.asarray(denominator, float)
    )
    result = np.full(numerator.shape, np.nan)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def estimates(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Estimands from summed statistics of shape (..., len(STATISTICS)); NaN if undefined."""
    s = {name: np.asarray(sums, float)[..., position] for name, position in INDEX.items()}
    result = {
        "n": s["n"],
        "n_moved": s["n_moved"],
        "b_rev": _ratio(s["pt"], s["tt"]),
        "b_rev_moved": _ratio(s["pt_moved"], s["tt_moved"]),
        "ratio_moved": _ratio(s["pt_moved"], s["rt_moved"]),
        "slope_real_moved": _ratio(s["rt_moved"], s["tt_moved"]),
        "b_cal_real": _ratio(s["rt"], s["rr"]),
        "noise_rms": np.sqrt(_ratio(s["noise"], s["n"])),
        "static_excess": _ratio(s["pp_static"] - s["rr_static"], s["n_static"]),
    }
    for k in K_VALUES:
        corrected = s["pp"] - k * s["noise"]
        # A calibration slope needs outcome motion; with t ≡ 0 it is 0 by construction.
        result[f"b_cal_k{k:g}"] = _ratio(s["pt"], np.where(s["tt"] > 0, corrected, np.nan))
        if k in (0.0, 1.0):
            # R² needs both factors of the denominator positive, not just their product.
            product = np.where((corrected > 0) & (s["tt"] > 0), corrected * s["tt"], np.nan)
            result[f"r2_k{k:g}"] = _ratio(s["pt"] ** 2, product)
    # Uncorrected: b_rev = alignment·sqrt(energy_ratio), b_cal = alignment/sqrt(energy_ratio).
    result["energy_ratio"] = _ratio(s["pp"], s["tt"])
    result["alignment"] = _ratio(s["pt"], np.sqrt(np.where(s["pp"] > 0, s["pp"], np.nan) * s["tt"]))
    for step in STEPS:
        result[f"z{step}_b_rev"] = _ratio(s[f"z{step}_pt"], s[f"z{step}_tt"])
        result[f"z{step}_b_cal"] = _ratio(s[f"z{step}_pt"], s[f"z{step}_pp"])
        result[f"z{step}_energy_ratio"] = _ratio(s[f"z{step}_pp"], s[f"z{step}_tt"])
    return result


def resample(unit_sums: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Bootstrap totals (draws, ...) from per-unit sums (units, ...), resampling units."""
    unit_sums = np.asarray(unit_sums, float)
    units = unit_sums.shape[0]
    rng = np.random.default_rng(seed)
    totals = np.empty((draws, *unit_sums.shape[1:]))
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, units, units), minlength=units)
        totals[draw] = np.tensordot(counts, unit_sums, axes=1)
    return totals


def interval(values: np.ndarray) -> list[float] | None:
    """Percentile 95% interval; None unless at least half of the draws are defined."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite) or len(finite) < len(values) / 2:
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def classify(ci: Sequence[float] | None) -> str:
    """Position of a calibration-slope interval relative to 1 (D030)."""
    if ci is None:
        return "undefined"
    low, high = ci
    if low > 1:
        return "above_1"
    if high < 1:
        return "below_1"
    return "contains_1"


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def summarize_group(
    cell_names: Sequence[str],
    unit_sums: np.ndarray,
    draws: int,
    seed: int,
    contrasts: Sequence[tuple[str, str, str, str]] = (),
) -> dict[str, Any]:
    """Estimates, intervals and paired contrasts for one resampling group.

    `unit_sums` has shape (units, cells, len(STATISTICS)). Every cell and contrast uses
    the same resampled units, so contrasts are paired. A contrast is
    (label, left cell, right cell, estimand) and reports left − right.
    """
    unit_sums = np.asarray(unit_sums, float)
    if unit_sums.shape[1] != len(cell_names):
        raise ValueError("One column of sums per cell name is required")
    point = estimates(unit_sums.sum(axis=0))
    boot = estimates(resample(unit_sums, draws, seed))
    cells: dict[str, Any] = {}
    for column, name in enumerate(cell_names):
        entry: dict[str, Any] = {
            "n": int(point["n"][column]),
            "n_moved": int(point["n_moved"][column]),
        }
        for key in REPORTED:
            entry[key] = {
                "estimate": _finite(point[key][column]),
                "ci95": interval(boot[key][:, column]),
            }
        entry["class_k1"] = classify(entry["b_cal_k1"]["ci95"])
        entry["class_k0"] = classify(entry["b_cal_k0"]["ci95"])
        entry["robust_above_1"] = entry["class_k1"] == "above_1" and entry["class_k0"] == "above_1"
        cells[name] = entry
    column = {name: position for position, name in enumerate(cell_names)}
    paired = {}
    for label, left, right, key in contrasts:
        a, b = column[left], column[right]
        paired[label] = {
            "estimand": key,
            "left": left,
            "right": right,
            "estimate": _finite(point[key][a] - point[key][b]),
            "ci95": interval(boot[key][:, a] - boot[key][:, b]),
        }
    return {
        "units": int(unit_sums.shape[0]),
        "draws": draws,
        "seed": seed,
        "cells": cells,
        "contrasts": paired,
    }
