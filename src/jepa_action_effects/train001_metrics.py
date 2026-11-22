"""TRAIN-001 measurement: gates, paired family bootstrap, verdicts and diagnostics.

All physical quantities decode projected latents with a frozen READOUT-002 MLP into
block (x, y) workspace units. Labels are kept distinct:

- *endpoint error*: ||decoded end - true end|| (absolute terminal block position);
- *displacement error*: ||(decoded end - decoded start) - (true end - true start)||;
- *projection slope*: sum(p.t)/sum(t.t) over moved branches (true displacement > 5.12);
- *latent MSE*: mean squared error of projected embeddings (representation, not physics).

Families are the resampling unit. Conditions and training seeds are compared on the
same resampled families; seeds are never resampled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from jepa_action_effects.followup_metrics import decompose
from jepa_action_effects.protocol_gates import readout_gate_per_cell
from jepa_action_effects.train001_protocol import (
    DECISION_RULES,
    HIGH_MAGNITUDES,
    MAGNITUDES,
    SCENES,
    Protocol,
    magnitude_label,
)

HIGH_INDEX = [MAGNITUDES.index(m) for m in HIGH_MAGNITUDES]


# --------------------------------------------------------------------------- basics
def norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=-1)


def projection_slope(predicted: np.ndarray, true: np.ndarray) -> float | None:
    p, t = np.asarray(predicted, float).reshape(-1, 2), np.asarray(true, float).reshape(-1, 2)
    energy = float(np.sum(t * t))
    if len(t) == 0 or energy <= 0:
        return None
    return float(np.sum(p * t) / energy)


def scene_mean(values: np.ndarray, scenes: np.ndarray) -> float:
    """Mean within each scene present, then equal-weight average of scene means."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean([values[scenes == s].mean() for s in np.unique(scenes)]))


def percentile_interval(draws: list[float | None], total: int) -> list[float] | None:
    valid = [d for d in draws if d is not None and np.isfinite(d)]
    if len(valid) < total / 2:
        return None
    return [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]


def stratified_indices(scenes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One family-bootstrap draw, resampling within each scene (counts preserved)."""
    parts = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        parts.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(parts)


def derangement(unit_id: str, seed: int, size: int = 4) -> list[int]:
    """Seeded permutation with no fixed point, independent of processing order."""
    key = int(hashlib.sha256(unit_id.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            return permutation.tolist()


# --------------------------------------------------------------------------- tables
def family_table(real_records: list[dict], readout: str) -> dict[str, np.ndarray]:
    """Arrays from real-frame records: truth and real-frame decodes, one readout."""
    return {
        "ids": np.asarray([r["unit"] for r in real_records]),
        "scenes": np.asarray([SCENES.index(r["scene"]) for r in real_records]),
        "true_start": np.asarray([r["true_start"] for r in real_records], float),
        "true_end": np.asarray([r["true_end"] for r in real_records], float),
        "decoded_start": np.asarray([r["decoded"][readout]["start"] for r in real_records], float),
        "real_end": np.asarray([r["decoded"][readout]["real_end"] for r in real_records], float),
    }


def prediction_table(model_records: list[dict], readout: str) -> dict[str, np.ndarray]:
    return {
        "pred_end": np.asarray([r["decoded"][readout]["pred_end"] for r in model_records], float),
        "perm_end": np.asarray([r["decoded"][readout]["perm_end"] for r in model_records], float),
        "latent_mse": np.asarray([r["latent_mse"] for r in model_records], float),
        "latent_mse_permuted": np.asarray([r["latent_mse_permuted"] for r in model_records], float),
    }


# --------------------------------------------------------------------------- gates
def cell_gate(
    true_displacement: np.ndarray,
    real_displacement: np.ndarray,
    real_displacement_error: np.ndarray,
    protocol: Protocol,
    seed: int,
) -> dict[str, Any]:
    """Measurement gate for one scene x magnitude cell (real-frame controls only).

    floor = mean real-frame displacement error over all accepted families in the cell
            / median true displacement length over moved families.
    """
    true_displacement = np.asarray(true_displacement, float)
    real_displacement = np.asarray(real_displacement, float)
    moved = norms(true_displacement) > protocol.delta
    n_moved = int(moved.sum())
    slope = projection_slope(real_displacement[moved], true_displacement[moved])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(protocol.bootstrap_draws):
        index = rng.choice(len(moved), size=len(moved), replace=True) if len(moved) else []
        chosen = np.asarray(index, dtype=int)
        mask = moved[chosen]
        draws.append(
            projection_slope(real_displacement[chosen][mask], true_displacement[chosen][mask])
        )
    interval = percentile_interval(draws, protocol.bootstrap_draws) if len(moved) else None
    median_moved = float(np.median(norms(true_displacement[moved]))) if n_moved else None
    floor = (
        float(np.mean(real_displacement_error)) / median_moved
        if median_moved and len(real_displacement_error)
        else None
    )
    band = readout_gate_per_cell({"cell": interval}, protocol.readout_band)["passed"]
    checks = {
        "moved_count": n_moved >= protocol.min_moved_families,
        "slope_interval_in_band": bool(band),
        "floor_below_limit": floor is not None
        and np.isfinite(floor)
        and floor < protocol.floor_limit,
    }
    return {
        "families": int(len(moved)),
        "moved": n_moved,
        "slope_real": slope,
        "slope_real_ci": interval,
        "mean_real_displacement_error": float(np.mean(real_displacement_error))
        if len(real_displacement_error)
        else None,
        "median_moved_true_displacement": median_moved,
        "floor": floor,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def measurement_gates(table: Mapping[str, np.ndarray], protocol: Protocol, seed: int) -> dict:
    """All primary cells (scene x {0.45, 0.60}) from real-frame decodes."""
    true_displacement = table["true_end"] - table["true_start"][:, None]
    real_displacement = table["real_end"] - table["decoded_start"][:, None]
    real_error = norms(real_displacement - true_displacement)
    cells = {}
    for scene_index, scene in enumerate(SCENES):
        rows = table["scenes"] == scene_index
        for offset, magnitude in enumerate(HIGH_MAGNITUDES):
            branch = MAGNITUDES.index(magnitude)
            cells[f"{scene}:{magnitude_label(magnitude)}"] = cell_gate(
                true_displacement[rows, branch],
                real_displacement[rows, branch],
                real_error[rows, branch],
                protocol,
                seed + 10 * scene_index + offset,
            )
    return {
        "cells": cells,
        "failing_cells": sorted(k for k, v in cells.items() if not v["passed"]),
        "passed": all(v["passed"] for v in cells.values()),
    }


# --------------------------------------------------------------------------- errors
def endpoint_errors(pred_end: np.ndarray, true_end: np.ndarray) -> np.ndarray:
    return norms(np.asarray(pred_end) - np.asarray(true_end))


def high_mean(errors: np.ndarray) -> np.ndarray:
    """(F,4) -> (F,) mean over the 0.45 and 0.60 branches."""
    return np.asarray(errors, float)[:, HIGH_INDEX].mean(axis=1)


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap(
    differences: Mapping[str, np.ndarray],
    scenes: np.ndarray,
    draws: int,
    seed: int,
    aggregate: Callable[[np.ndarray, np.ndarray], float] = scene_mean,
    labels: tuple[str, ...] = SCENES,
) -> dict[str, dict[str, Any]]:
    """Joint scene-stratified family bootstrap of several per-family contrasts.

    Every contrast uses the same resampled families in each draw, so conditions and
    seeds stay paired; training seeds are separate contrasts and are never resampled.
    """
    rng = np.random.default_rng(seed)
    scenes = np.asarray(scenes)
    samples: dict[str, list[float]] = {name: [] for name in differences}
    scene_samples: dict[str, dict[int, list[float]]] = {
        name: {int(s): [] for s in np.unique(scenes)} for name in differences
    }
    for _ in range(draws):
        index = stratified_indices(scenes, rng)
        for name, values in differences.items():
            chosen = np.asarray(values, float)[index]
            samples[name].append(aggregate(chosen, scenes[index]))
            for scene in scene_samples[name]:
                scene_samples[name][scene].append(float(chosen[scenes[index] == scene].mean()))
    result = {}
    for name, values in differences.items():
        values = np.asarray(values, float)
        result[name] = {
            "estimate": aggregate(values, scenes),
            "ci": percentile_interval(samples[name], draws),
            "by_scene": {
                labels[scene]: {
                    "estimate": float(values[scenes == scene].mean()),
                    "ci": percentile_interval(scene_samples[name][scene], draws),
                }
                for scene in scene_samples[name]
            },
        }
    return result


def interval_sign(interval: list[float] | None) -> str:
    if interval is None:
        return "undefined"
    if interval[1] < 0:
        return "below_zero"
    if interval[0] > 0:
        return "above_zero"
    return "spans_zero"


# --------------------------------------------------------------------------- verdicts
def pooled_verdict(
    development_passed: bool,
    confirmation_passed: bool,
    balanced_minus_archive: list[float] | None,
    balanced_minus_frozen: list[float] | None,
) -> dict[str, Any]:
    rules = DECISION_RULES["verdicts"]
    signs = {
        "balanced_minus_archive": interval_sign(balanced_minus_archive),
        "balanced_minus_frozen": interval_sign(balanced_minus_frozen),
    }
    if not development_passed:
        verdict = rules["development_fail"]
    elif not confirmation_passed:
        verdict = rules["confirmation_fail"]
    elif all(sign == "below_zero" for sign in signs.values()):
        verdict = rules["both_ci_below_zero"]
    elif any(sign == "above_zero" for sign in signs.values()):
        verdict = rules["required_ci_above_zero"]
    else:
        verdict = rules["otherwise"]
    increases = [name for name, sign in signs.items() if sign == "above_zero"]
    return {"verdict": verdict, "interval_signs": signs, "error_increase_contrasts": increases}


def ordinary_outcome(interval: list[float] | None, tolerance: float) -> str:
    rules = DECISION_RULES["ordinary"]
    if interval is not None and interval[1] < tolerance:
        return rules["upper_below_tolerance"]
    if interval is not None and interval[0] > tolerance:
        return rules["lower_above_tolerance"]
    return rules["otherwise"]


# --------------------------------------------------------------------------- calibration
def fit_calibration(
    predicted_displacement: np.ndarray, true_displacement: np.ndarray, scenes: np.ndarray
) -> float:
    """Nonnegative scalar s minimizing family- and scene-weighted squared error.

    Each scene receives equal total weight, each family equal weight within its scene,
    and each of its branches equal weight within the family.
    """
    p = np.asarray(predicted_displacement, float)
    t = np.asarray(true_displacement, float)
    scenes = np.asarray(scenes)
    present = np.unique(scenes)
    weights = np.zeros(len(scenes))
    for scene in present:
        members = scenes == scene
        weights[members] = 1.0 / (len(present) * members.sum())
    numerator = float(np.sum(weights[:, None] * np.sum(p * t, axis=-1)))
    denominator = float(np.sum(weights[:, None] * np.sum(p * p, axis=-1)))
    if denominator <= 0:
        raise ValueError("Calibration needs nonzero predicted displacements")
    return max(0.0, numerator / denominator)


# --------------------------------------------------------------------------- diagnostics
def magnitude_diagnostics(
    table: Mapping[str, np.ndarray], pred_end: np.ndarray, delta: float
) -> dict[str, Any]:
    """Per scene x magnitude point estimates for one model's predictions."""
    true_displacement = table["true_end"] - table["true_start"][:, None]
    predicted = pred_end - table["decoded_start"][:, None]
    real = table["real_end"] - table["decoded_start"][:, None]
    endpoint = endpoint_errors(pred_end, table["true_end"])
    displacement = norms(predicted - true_displacement)
    result = {}
    for scene_index, scene in enumerate(SCENES):
        rows = table["scenes"] == scene_index
        if not rows.any():
            continue
        for branch, magnitude in enumerate(MAGNITUDES):
            t = true_displacement[rows, branch]
            moved = norms(t) > delta
            p, r = predicted[rows, branch][moved], real[rows, branch][moved]
            slope_pred = projection_slope(p, t[moved])
            slope_real = projection_slope(r, t[moved])
            result[f"{scene}:{magnitude_label(magnitude)}"] = {
                "families": int(rows.sum()),
                "moved": int(moved.sum()),
                "endpoint_error_mean": float(endpoint[rows, branch].mean()),
                "displacement_error_mean": float(displacement[rows, branch].mean()),
                "slope_pred": slope_pred,
                "slope_real": slope_real,
                "slope_ratio_pred_over_real": (
                    slope_pred / slope_real if slope_pred is not None and slope_real else None
                ),
                "decomposition": decompose(p, t[moved]) if moved.any() else None,
                "median_true_displacement": float(np.median(norms(t))) if len(t) else None,
            }
    return result


# --------------------------------------------------------------------------- summary
def _seed_average(values: Mapping[str, np.ndarray], condition: str, seeds) -> np.ndarray:
    return np.mean([values[f"{condition}-{seed}"] for seed in seeds], axis=0)


def _contrasts(values: Mapping[str, np.ndarray], seeds) -> dict[str, np.ndarray]:
    """Primary contrasts (seed-averaged) plus the same contrasts for each seed."""
    balanced = _seed_average(values, "balanced", seeds)
    archive = _seed_average(values, "archive_weighted", seeds)
    frozen = values["frozen"]
    result = {
        "balanced_minus_archive": balanced - archive,
        "balanced_minus_frozen": balanced - frozen,
        "archive_minus_frozen": archive - frozen,
    }
    for seed in seeds:
        b, a = values[f"balanced-{seed}"], values[f"archive_weighted-{seed}"]
        result[f"balanced_minus_archive@{seed}"] = b - a
        result[f"balanced_minus_frozen@{seed}"] = b - frozen
        result[f"archive_minus_frozen@{seed}"] = a - frozen
    return result


def _seed_spread(boot: Mapping[str, dict], seeds) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("balanced_minus_archive", "balanced_minus_frozen", "archive_minus_frozen"):
        estimates = [boot[f"{name}@{seed}"]["estimate"] for seed in seeds]
        result[name] = {
            "per_seed": dict(zip([str(s) for s in seeds], estimates, strict=True)),
            "sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None,
            "range": [float(min(estimates)), float(max(estimates))],
        }
    return result


def summarize(
    family_real: list[dict],
    family_models: Mapping[str, list[dict]],
    window_real: list[dict],
    window_models: Mapping[str, list[dict]],
    protocol: Protocol,
    calibration: Mapping[str, float],
    development_passed: bool,
    readouts,
    phase: str,
) -> dict[str, Any]:
    """All TRAIN-001 quantities for one cohort. Records must be aligned by unit order."""
    seeds = protocol.training_seeds
    draws = protocol.bootstrap_draws
    models = sorted(family_models)
    result: dict[str, Any] = {"phase": phase, "families": len(family_real), "readouts": {}}
    latent = {m: np.asarray([r["latent_mse"] for r in family_models[m]], float) for m in models}
    terminal = {m: latent[m][:, HIGH_INDEX, 1].mean(axis=1) for m in models}
    scenes = np.asarray([SCENES.index(r["scene"]) for r in family_real])
    latent_boot = paired_bootstrap(
        _contrasts(terminal, seeds), scenes, draws, protocol.primary_bootstrap_seed + 1
    )
    result["secondary_terminal_latent_mse"] = {
        "label": "representation prediction (not physical prediction)",
        "means": {m: scene_mean(terminal[m], scenes) for m in models},
        "contrasts": latent_boot,
        "interval_signs": {
            name: interval_sign(latent_boot[name]["ci"])
            for name in ("balanced_minus_archive", "balanced_minus_frozen", "archive_minus_frozen")
        },
        "seed_spread": _seed_spread(latent_boot, seeds),
        "latent_mse_by_step": {
            m: [scene_mean(latent[m][:, :, step].mean(axis=1), scenes) for step in (0, 1)]
            for m in models
        },
    }
    for index, readout in enumerate(readouts):
        table = family_table(family_real, readout)
        gates = measurement_gates(table, protocol, protocol.gate_bootstrap_seed)
        start = table["decoded_start"][:, None]
        predicted = {m: prediction_table(family_models[m], readout) for m in models}
        endpoint = {m: endpoint_errors(predicted[m]["pred_end"], table["true_end"]) for m in models}
        endpoint["persistence"] = endpoint_errors(
            np.repeat(start, len(MAGNITUDES), axis=1), table["true_end"]
        )
        scale = float(calibration[readout])
        endpoint["calibrated_frozen"] = endpoint_errors(
            start + scale * (predicted["frozen"]["pred_end"] - start), table["true_end"]
        )
        high = {m: high_mean(values) for m, values in endpoint.items()}
        contrasts = _contrasts(high, seeds)
        balanced = _seed_average(high, "balanced", seeds)
        archive = _seed_average(high, "archive_weighted", seeds)
        contrasts.update(
            {
                "calibrated_frozen_minus_frozen": high["calibrated_frozen"] - high["frozen"],
                "balanced_minus_calibrated_frozen": balanced - high["calibrated_frozen"],
                "balanced_minus_persistence": balanced - high["persistence"],
            }
        )
        absolute = {f"abs:{m}": values for m, values in high.items()}
        absolute["abs:balanced(seed-mean)"] = balanced
        absolute["abs:archive_weighted(seed-mean)"] = archive
        boot = paired_bootstrap(
            {**contrasts, **absolute}, table["scenes"], draws, protocol.primary_bootstrap_seed
        )
        permuted = {
            f"{m}:permuted_minus_correct": high_mean(
                endpoint_errors(predicted[m]["perm_end"], table["true_end"])
            )
            - high[m]
            for m in models
        }
        permuted_boot = paired_bootstrap(
            permuted, table["scenes"], draws, protocol.primary_bootstrap_seed + 2
        )
        verdict = pooled_verdict(
            development_passed,
            gates["passed"],
            boot["balanced_minus_archive"]["ci"],
            boot["balanced_minus_frozen"]["ci"],
        )
        diagnostics = {
            m: magnitude_diagnostics(table, predicted[m]["pred_end"], protocol.delta)
            for m in models
        }
        result["readouts"][readout] = {
            "role": "primary" if index == 0 else "sensitivity",
            "measurement_gates": gates,
            "primary": {
                "label": "absolute terminal block-position error, 0.45/0.60 branches",
                "bootstrap": boot,
                "seed_spread": _seed_spread(boot, seeds),
                "verdict": verdict,
                "exploratory_subgroups": {
                    name: boot[name]["by_scene"]
                    for name in ("balanced_minus_archive", "balanced_minus_frozen")
                },
            },
            "action_permuted": permuted_boot,
            "calibration_scale": scale,
            "diagnostics": diagnostics,
            "ordinary": _ordinary(window_real, window_models, readout, protocol),
        }
    verdicts = {r: v["primary"]["verdict"]["verdict"] for r, v in result["readouts"].items()}
    result["readout_sensitivity"] = {
        "verdicts": verdicts,
        "measurement_dependent": len(set(verdicts.values())) > 1,
    }
    primary = result["readouts"][readouts[0]]
    result["pooled_verdict"] = primary["primary"]["verdict"]
    result["ordinary_outcome"] = primary["ordinary"]["outcome"]
    return result


def _ordinary(window_real, window_models, readout: str, protocol: Protocol) -> dict[str, Any]:
    seeds = protocol.training_seeds
    true_end = np.asarray([r["true_end"] for r in window_real], float)
    errors = {
        m: endpoint_errors(
            np.asarray([r["decoded"][readout]["pred_end"] for r in records], float), true_end
        )[:, 0]
        for m, records in window_models.items()
    }
    real_end = np.asarray([r["decoded"][readout]["real_end"] for r in window_real], float)
    errors["real-frames"] = endpoint_errors(real_end, true_end)[:, 0]
    strata = np.zeros(len(window_real), dtype=int)
    contrasts = _contrasts(errors, seeds)
    absolute = {f"abs:{m}": values for m, values in errors.items()}
    boot = paired_bootstrap(
        {**contrasts, **absolute},
        strata,
        protocol.bootstrap_draws,
        protocol.ordinary_bootstrap_seed,
        labels=("ordinary",),
    )
    interval = boot["balanced_minus_frozen"]["ci"]
    return {
        "label": "ordinary archive windows, absolute terminal block-position error",
        "windows": len(window_real),
        "bootstrap": boot,
        "tolerance": protocol.regression_tolerance,
        "outcome": ordinary_outcome(interval, protocol.regression_tolerance),
    }
