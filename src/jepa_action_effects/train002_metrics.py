"""TRAIN-002 estimands (D035): held-out-dose transfer and generic fine-tuning contrasts.

Arms are `frozen`, the reused TRAIN-001 `balanced` and `archive_weighted` runs, and the
new `dose_omitted` and `archive_only` runs. Per-family errors are averaged over training
seeds within an arm; the bootstrap resamples families within scene and never seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jepa_action_effects.train001_metrics import (
    endpoint_errors,
    family_table,
    magnitude_diagnostics,
    paired_bootstrap,
    percentile_interval,
    projection_slope,
    scene_mean,
    stratified_indices,
)
from jepa_action_effects.train001_protocol import MAGNITUDES, SCENES, magnitude_label

ARMS = ("frozen", "archive_weighted", "balanced", "dose_omitted", "archive_only")
TRAINED_ARMS = ARMS[1:]
HELD_OUT = MAGNITUDES.index(0.60)
IN_SUPPORT = MAGNITUDES.index(0.45)
DOSE_OMITTED_WEIGHTS = (1 / 3, 1 / 3, 1 / 3, 0.0)


def arm_of(model_id: str) -> str:
    """`balanced-20261111` -> `balanced`; `frozen` -> `frozen`."""
    for arm in ARMS:
        if model_id == arm or model_id.startswith(f"{arm}-"):
            return arm
    raise ValueError(model_id)


def arm_average(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average per-model arrays over the seeds of each arm."""
    grouped: dict[str, list[np.ndarray]] = {}
    for model_id, array in values.items():
        grouped.setdefault(arm_of(model_id), []).append(np.asarray(array, float))
    return {arm: np.mean(arrays, axis=0) for arm, arrays in grouped.items()}


def contrasts_at(errors: Mapping[str, np.ndarray], branch: int) -> dict[str, np.ndarray]:
    """D035 contrasts on one magnitude branch of seed-averaged (F, 4) errors."""
    e = {arm: np.asarray(v, float)[:, branch] for arm, v in errors.items()}
    out = {}
    pairs = {
        "dose_omitted_minus_balanced": ("dose_omitted", "balanced"),
        "frozen_minus_dose_omitted": ("frozen", "dose_omitted"),
        "frozen_minus_balanced": ("frozen", "balanced"),
        "frozen_minus_archive_only": ("frozen", "archive_only"),
        "archive_only_minus_balanced": ("archive_only", "balanced"),
        "balanced_minus_archive_weighted": ("balanced", "archive_weighted"),
    }
    for name, (left, right) in pairs.items():
        if left in e and right in e:
            out[name] = e[left] - e[right]
    return out


def transfer_fraction(
    frozen: np.ndarray,
    omitted: np.ndarray,
    balanced: np.ndarray,
    scenes: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """T = scene-mean(frozen − omitted) / scene-mean(frozen − balanced), paired draws.

    Uses the same seed and stratified resampling as `paired_bootstrap`, so its draws
    coincide with the primary contrast's.
    """
    gain_omitted = np.asarray(frozen, float) - np.asarray(omitted, float)
    gain_balanced = np.asarray(frozen, float) - np.asarray(balanced, float)
    scenes = np.asarray(scenes)

    def ratio(index: np.ndarray) -> float | None:
        denominator = scene_mean(gain_balanced[index], scenes[index])
        if not np.isfinite(denominator) or denominator <= 0:
            return None
        return float(scene_mean(gain_omitted[index], scenes[index]) / denominator)

    rng = np.random.default_rng(seed)
    samples = [ratio(stratified_indices(scenes, rng)) for _ in range(draws)]
    return {
        "estimate": ratio(np.arange(len(scenes))),
        "ci": percentile_interval(samples, draws),
        "note": "1 = held-out gain equals in-support gain; 0 = no transfer",
    }


def primary_verdict(interval: Sequence[float] | None, gate_passed: bool) -> str:
    if not gate_passed:
        return "not_assessable"
    if interval is None:
        return "held_out_dose_deficit_not_established"
    low, high = interval
    if low > 0:
        return "held_out_dose_deficit_supported"
    if high < 0:
        return "omission_better_supported"
    return "held_out_dose_deficit_not_established"


def generic_verdict(interval: Sequence[float] | None, gate_passed: bool) -> str:
    if not gate_passed:
        return "not_assessable"
    if interval is not None and interval[0] > 0:
        return "generic_finetuning_gain_supported"
    return "generic_finetuning_gain_not_established"


def slope_ratios(
    table: Mapping[str, np.ndarray], predicted: Mapping[str, np.ndarray], delta: float
):
    """Pred/real projection-slope ratio per arm × scene × magnitude (moved families).

    The arm's slope uses seed-averaged predicted endpoints; the slope is linear in the
    prediction, so this equals the slope of the pooled seed predictions.
    """
    true_displacement = table["true_end"] - table["true_start"][:, None]
    real = table["real_end"] - table["decoded_start"][:, None]
    out: dict[str, dict[str, Any]] = {}
    for arm, pred_end in predicted.items():
        displacement = np.asarray(pred_end, float) - table["decoded_start"][:, None]
        for scene_index, scene in enumerate(SCENES):
            rows = table["scenes"] == scene_index
            for branch, magnitude in enumerate(MAGNITUDES):
                t = true_displacement[rows, branch]
                moved = np.linalg.norm(t, axis=1) > delta
                if not moved.any():
                    continue
                slope_pred = projection_slope(displacement[rows, branch][moved], t[moved])
                slope_real = projection_slope(real[rows, branch][moved], t[moved])
                out.setdefault(arm, {})[f"{scene}:{magnitude_label(magnitude)}"] = {
                    "moved": int(moved.sum()),
                    "slope_pred": slope_pred,
                    "slope_real": slope_real,
                    "ratio": slope_pred / slope_real
                    if slope_pred is not None and slope_real
                    else None,
                }
    return out


def _family_errors(table, model_records: Mapping[str, list[dict]], readout: str):
    pred = {
        m: np.asarray([r["decoded"][readout]["pred_end"] for r in records], float)
        for m, records in model_records.items()
    }
    errors = {m: endpoint_errors(p, table["true_end"]) for m, p in pred.items()}
    return pred, errors


def summarize(
    family_real: list[dict],
    family_models: Mapping[str, list[dict]],
    window_real: list[dict],
    window_models: Mapping[str, list[dict]],
    readouts: Sequence[str],
    gates: Mapping[str, Any],
    draws: int,
    seed: int,
    ordinary_seed: int,
    delta: float,
    tolerance: float,
) -> dict[str, Any]:
    """All D035 quantities. Records must be aligned by unit order across models."""
    scenes = np.asarray([SCENES.index(r["scene"]) for r in family_real])
    gate_passed = all(gates["cells"][f"{s}:m0.60"]["passed"] for s in SCENES)
    result: dict[str, Any] = {
        "families": len(family_real),
        "windows": len(window_real),
        "gate_passed_0.60": gate_passed,
        "readouts": {},
    }
    latent = {
        m: np.asarray([r["latent_mse"] for r in records], float)[:, HELD_OUT, 1]
        for m, records in family_models.items()
    }
    latent_arm = arm_average(latent)
    latent_diffs = {
        name: latent_arm[a] - latent_arm[b]
        for name, (a, b) in {
            "dose_omitted_minus_balanced": ("dose_omitted", "balanced"),
            "frozen_minus_archive_only": ("frozen", "archive_only"),
        }.items()
    }
    result["latent_0.60"] = {
        "means": {arm: scene_mean(v, scenes) for arm, v in latent_arm.items()},
        "bootstrap": paired_bootstrap(latent_diffs, scenes, draws, seed + 1),
    }
    for readout in readouts:
        table = family_table(family_real, readout)
        pred, errors = _family_errors(table, family_models, readout)
        arm_errors = arm_average(errors)
        arm_pred = arm_average(pred)
        held = contrasts_at(arm_errors, HELD_OUT)
        support = {f"in_support:{k}": v for k, v in contrasts_at(arm_errors, IN_SUPPORT).items()}
        boot = paired_bootstrap({**held, **support}, scenes, draws, seed)
        per_seed = {}
        for model_id in sorted(errors):
            if arm_of(model_id) == "dose_omitted":
                partner = model_id.replace("dose_omitted", "balanced")
                per_seed[model_id] = float(
                    scene_mean(errors[model_id][:, HELD_OUT] - errors[partner][:, HELD_OUT], scenes)
                )
        p1 = boot["dose_omitted_minus_balanced"]["ci"]
        generic = boot["frozen_minus_archive_only"]["ci"]
        result["readouts"][readout] = {
            "means_0.60": {
                arm: scene_mean(v[:, HELD_OUT], scenes) for arm, v in arm_errors.items()
            },
            "means_0.45": {
                arm: scene_mean(v[:, IN_SUPPORT], scenes) for arm, v in arm_errors.items()
            },
            "bootstrap": boot,
            "per_seed_dose_omitted_minus_balanced": per_seed,
            "transfer_fraction": transfer_fraction(
                arm_errors["frozen"][:, HELD_OUT],
                arm_errors["dose_omitted"][:, HELD_OUT],
                arm_errors["balanced"][:, HELD_OUT],
                scenes,
                draws,
                seed,
            ),
            "slope_ratios": slope_ratios(table, arm_pred, delta),
            "frozen_diagnostics": magnitude_diagnostics(table, pred["frozen"], delta),
            "primary_verdict": primary_verdict(p1, gate_passed),
            "generic_verdict": generic_verdict(generic, gate_passed),
            "ordinary": _ordinary(
                window_real, window_models, readout, draws, ordinary_seed, tolerance
            ),
        }
    primary = result["readouts"][readouts[0]]
    result["primary_verdict"] = primary["primary_verdict"]
    result["generic_verdict"] = primary["generic_verdict"]
    result["readout_agreement"] = {
        name: len({_sign(result["readouts"][r]["bootstrap"][name]["ci"]) for r in readouts}) == 1
        for name in ("dose_omitted_minus_balanced", "frozen_minus_archive_only")
    }
    return result


def _sign(interval: Sequence[float] | None) -> str:
    if interval is None:
        return "undefined"
    if interval[0] > 0:
        return "above_zero"
    if interval[1] < 0:
        return "below_zero"
    return "spans_zero"


def _ordinary(window_real, window_models, readout, draws, seed, tolerance) -> dict[str, Any]:
    true_end = np.asarray([r["true_end"] for r in window_real], float)
    errors = {
        m: endpoint_errors(
            np.asarray([r["decoded"][readout]["pred_end"] for r in records], float), true_end
        )[:, 0]
        for m, records in window_models.items()
    }
    arm = arm_average(errors)
    real = endpoint_errors(
        np.asarray([r["decoded"][readout]["real_end"] for r in window_real], float), true_end
    )[:, 0]
    diffs = {f"{a}_minus_frozen": arm[a] - arm["frozen"] for a in arm if a != "frozen"}
    strata = np.zeros(len(window_real), dtype=int)
    boot = paired_bootstrap(diffs, strata, draws, seed, labels=("ordinary",))
    return {
        "means": {a: float(v.mean()) for a, v in arm.items()} | {"real-frames": float(real.mean())},
        "bootstrap": {k: {"estimate": v["estimate"], "ci": v["ci"]} for k, v in boot.items()},
        "tolerance": tolerance,
    }
