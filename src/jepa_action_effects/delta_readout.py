"""DELTA-001: bounded displacement readout, one-time confirmation, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa_action_effects.delta_probe import (
    ALPHAS,
    SEEDS,
    AnchoredNetwork,
    DirectRidge,
    deduplicated_training,
    family_weights,
    measurement_gate,
    select_ridge,
    transition_targets,
)
from jepa_action_effects.evaluation import (
    BranchPrediction,
    EvaluationConfig,
    Goal,
    aggregate_families,
    evaluate_family,
)
from jepa_action_effects.interface_audit import preprocess_pixels
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    model_fingerprint,
    predict_future,
)
from jepa_action_effects.pilot import _bootstrap, _coverage, _paired_bootstrap, _truth
from jepa_action_effects.pilot_collection import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.pilot_readouts import LinearReadouts
from jepa_action_effects.readout_recovery import _load_encoded_family, _partition_ids, _tiers

ROOT = Path("outputs/delta001")
PILOT = Path("outputs/pilot001")
READOUT = Path("outputs/readout001")
FRESH = READOUT / "collection"
CHECKPOINT = Path("checkpoints/lewm-pusht")
STATS = Path("outputs/model001/action-stats.json")
BUDGET_SECONDS = 7200.0
CONFIG = {
    "schema": "delta001-v1",
    "master_seed": 20260928,
    "horizon_native_steps": 2,
    "delta": 5.12,
    "fit_families": 240,
    "tuning_families": 60,
    "confirmation_families": 60,
    "alpha_grid": list(ALPHAS),
    "nonlinear_seeds": list(SEEDS),
    "nonlinear_max_epochs": 3000,
    "nonlinear_patience": 200,
    "nonlinear_improvement_workspace_units": 1e-4,
    "mean_gate": 1.28,
    "p95_gate": 2.56,
    "minimum_families": 10,
    "bootstrap_samples": 2000,
    "experiment_cpu_budget_seconds": BUDGET_SECONDS,
}


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _seal(path: Path, payload: Any) -> str:
    digest = _write(path, payload)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _sealed(path: Path) -> dict:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.exists():
        digest_path = path.with_suffix(".sha256")
    if sha256_file(path) != digest_path.read_text().strip():
        raise ValueError(f"Artifact hash mismatch: {path}")
    return json.loads(path.read_text())


def _sources() -> dict:
    paths = {
        "delta_probe": Path(__file__).with_name("delta_probe.py"),
        "delta_readout": Path(__file__),
        "readout_recovery": Path(__file__).with_name("readout_recovery.py"),
        "pilot_readouts": Path(__file__).with_name("pilot_readouts.py"),
        "evaluator": Path(__file__).with_name("evaluation.py"),
        "plan": Path("docs/DELTA-001_PLAN.md"),
        "uv_lock": Path("uv.lock"),
        "weights": CHECKPOINT / "weights.pt",
        "action_stats": STATS,
        "pilot_manifest": PILOT / "collection-manifest.json",
        "recovery_manifest": FRESH / "collection-manifest.json",
        "recovery_fit_report": READOUT / "fit-report.json",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _audit_required(root: Path = ROOT) -> dict:
    audited = _sealed(root / "audit.json")
    if audited["status"] != "passed" or audited["sources"] != _sources():
        raise ValueError("DELTA-001 audit inputs or code changed")
    return audited


class ExperimentBudget:
    def __init__(self, root: Path, signature: str):
        self.path = root / "runtime.json"
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record["signature"] != signature:
                raise ValueError("Runtime ledger input mismatch")
            running = self.record.pop("running", None)
            if running:
                self.record["used_seconds"] += max(0.0, time.time() - running["started_utc"])
                self.record["interruptions"].append(running["stage"])
                _write(self.path, self.record)
        else:
            self.record = {
                "signature": signature,
                "used_seconds": 0.0,
                "stages": {},
                "interruptions": [],
            }
            _write(self.path, self.record)
        self.started = None

    def remaining(self) -> float:
        current = time.monotonic() - self.started if self.started is not None else 0.0
        return BUDGET_SECONDS - self.record["used_seconds"] - current

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("DELTA-001 cumulative CPU budget exhausted")

    def begin(self, stage: str) -> None:
        if stage != "report":
            self.check()
        self.started = time.monotonic()
        self.stage = stage
        self.record["running"] = {"stage": stage, "started_utc": time.time()}
        _write(self.path, self.record)

    def finish(self, status: str) -> None:
        elapsed = time.monotonic() - self.started
        self.record["used_seconds"] += elapsed
        self.record["stages"].setdefault(self.stage, []).append(
            {"seconds": elapsed, "status": status}
        )
        self.record.pop("running", None)
        self.started = None
        _write(self.path, self.record)


def _partitions(pilot: dict, fresh: dict) -> dict:
    parts = _partition_ids(pilot, fresh)
    tiers = _tiers(parts)
    if len(tiers[240]) != 240 or len(parts["tuning"]) != 60 or len(parts["confirmation"]) != 60:
        raise ValueError("READOUT-001 partition counts changed")
    return {"fit": tiers[240], "tuning": parts["tuning"], "confirmation": parts["confirmation"]}


def audit(root: Path = ROOT) -> dict:
    pilot = verify_collection(PILOT)
    fresh = verify_collection(FRESH)
    if pilot["selected_horizon"] != 2 or fresh["selected_horizon"] != 2:
        raise ValueError("DELTA-001 requires the audited two-step horizon")
    fit_report = _sealed(READOUT / "fit-report.json")
    if fit_report["status"] != "complete" or fit_report["confirmation_opened"]:
        raise ValueError("READOUT-001 confirmation independence cannot be supported")
    parts = _partitions(pilot, fresh)
    if len(list((READOUT / "latent-cache" / "real").glob("*.npz"))) != 600:
        raise ValueError("Recovery cache does not have the expected 300 development families")
    confirmation = set(parts["confirmation"])
    for path in (READOUT / "latent-cache" / "real").glob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
        if manifest.get("identifiers", {}).get("identifier") in confirmation:
            raise ValueError("Confirmation family was encoded during READOUT-001")
    streams = np.random.SeedSequence(CONFIG["master_seed"]).spawn(3)
    seed_streams = {
        name: int(seed.generate_state(1)[0])
        for name, seed in zip(("diagnostics", "permutations", "bootstrap"), streams, strict=True)
    }
    sources = _sources()
    config = {**CONFIG, "seed_streams": seed_streams, "sources": sources, "partitions": parts}
    config_sha = _seal(root / "config.json", config)
    result = {
        "status": "passed",
        "sources": sources,
        "config_sha256": config_sha,
        "partition_counts": {key: len(value) for key, value in parts.items()},
        "confirmation_previous_use": "no_fit_selection_metric_or_encoded_cache_entry",
        "confirmation_metadata_hash_inspection": True,
        "label_access_policy": "fit_and_tuning_only_until_selection_sealed",
    }
    _seal(root / "audit.json", result)
    return result


def _load_data(ids: list[str], pilot: dict, fresh: dict, model, fingerprint: dict) -> dict:
    cache = LatentCache(READOUT / "latent-cache")
    result = {}
    for identifier in ids:
        family, views = _load_encoded_family(
            identifier, PILOT, FRESH, pilot, fresh, cache, fingerprint, model
        )
        projected = views["projected"]
        if projected["history"].shape != (3, 192) or projected["future"].shape != (6, 2, 192):
            raise ValueError("Projected CLS cache alignment changed")
        result[identifier] = (family, projected)
    return result


def _model_context(audit_result: dict) -> tuple[Any, dict, dict, dict, dict]:
    pilot = verify_collection(PILOT)
    fresh = verify_collection(FRESH)
    parts = _partitions(pilot, fresh)
    model = build_frozen_model(CHECKPOINT)
    scaler = ActionScaler.from_json(STATS)
    fingerprint = {
        **model_fingerprint(CHECKPOINT, scaler),
        "pilot_manifest": audit_result["sources"]["pilot_manifest"],
        "recovery_manifest": audit_result["sources"]["recovery_manifest"],
    }
    return model, pilot, fresh, parts, fingerprint


def _load_absolute_ridge() -> LinearReadouts:
    path = READOUT / "models" / "240-projected-ridge.npz"
    with np.load(path, allow_pickle=False) as archive:
        return LinearReadouts(
            *[
                np.asarray(archive[key])
                for key in (
                    "feature_mean",
                    "feature_scale",
                    "alpha",
                    "block_coef",
                    "block_intercept",
                    "agent_coef",
                    "agent_intercept",
                    "orientation_coef",
                    "orientation_intercept",
                )
            ]
        )


def _baseline_predict(start: np.ndarray, future: np.ndarray) -> np.ndarray:
    ridge = _load_absolute_ridge()
    return ridge.predict(future)["block"] - ridge.predict(start[None])["block"][0]


def _checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _nonlinear_fit(
    data: dict, tuning: dict, seed: int, output: Path, signature: str, budget: ExperimentBudget
) -> dict:
    families = {key: value[0] for key, value in data.items()}
    embeddings = {key: value[1] for key, value in data.items()}
    contexts, differences, targets, ids, _ = deduplicated_training(families, embeddings)
    weights = family_weights(ids)
    context_mean = np.average(contexts, axis=0, weights=weights)
    context_scale = np.sqrt(np.average((contexts - context_mean) ** 2, axis=0, weights=weights))
    context_scale[context_scale == 0] = 1
    difference_scale = np.sqrt(np.average(differences**2, axis=0, weights=weights))
    difference_scale[difference_scale == 0] = 1
    target_scale = np.sqrt(np.average(targets**2, axis=0, weights=weights))
    target_scale[target_scale == 0] = 1
    scaling_path = output.with_suffix(".scaling.npz")
    np.savez_compressed(
        scaling_path,
        context_mean=context_mean,
        context_scale=context_scale,
        difference_scale=difference_scale,
        target_scale=target_scale,
    )
    context_t = torch.tensor((contexts - context_mean) / context_scale, dtype=torch.float32)
    difference_t = torch.tensor(differences / difference_scale, dtype=torch.float32)
    target_t = torch.tensor(targets / target_scale, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    tuning_items = list(tuning.values())
    tune_context, tune_difference, tune_target = [], [], []
    for family, embeddings in tuning_items:
        start, future, truth = transition_targets(family, embeddings)
        tune_context.extend([start] * 6)
        tune_difference.extend(future - start)
        tune_target.extend(truth)
    tune_context_t = torch.tensor(
        (np.asarray(tune_context) - context_mean) / context_scale, dtype=torch.float32
    )
    tune_difference_t = torch.tensor(
        np.asarray(tune_difference) / difference_scale, dtype=torch.float32
    )
    tune_target_array = np.asarray(tune_target)
    torch.manual_seed(seed)
    network = AnchoredNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    progress_path = output.with_suffix(".progress.pt")
    start_epoch, best, best_epoch, stale, curve, best_state = 1, float("inf"), 0, 0, [], None
    if progress_path.exists():
        previous = torch.load(progress_path, map_location="cpu", weights_only=False)
        if previous["signature"] != signature or previous["seed"] != seed:
            raise ValueError("Nonlinear checkpoint signature mismatch")
        network.load_state_dict(previous["state"])
        optimizer.load_state_dict(previous["optimizer"])
        start_epoch = previous["epoch"] + 1
        best, best_epoch, stale, curve = (
            previous["best"],
            previous["best_epoch"],
            previous["stale"],
            previous["curve"],
        )
        best_state = previous["best_state"]
    status = "epoch_cap"
    for epoch in range(start_epoch, 3001):
        network.train()
        optimizer.zero_grad()
        residual = network(context_t, difference_t) - target_t
        loss = (residual.square().mean(dim=1) * weights_t).sum()
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.inference_mode():
            tune_pred = network(tune_context_t, tune_difference_t).numpy() * target_scale
            tune_error = np.linalg.norm(tune_pred - tune_target_array, axis=1).reshape(
                len(tuning), 6
            )
            score = float(tune_error.mean(axis=1).mean())
        if score < best - 1e-4:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            with torch.inference_mode():
                train_pred = network(context_t, difference_t).numpy() * target_scale
            train_error = float(np.sum(np.linalg.norm(train_pred - targets, axis=1) * weights))
            curve.append({"epoch": epoch, "fit_mean": train_error, "tuning_mean": score})
            _checkpoint(
                progress_path,
                {
                    "signature": signature,
                    "seed": seed,
                    "epoch": epoch,
                    "state": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "stale": stale,
                    "curve": curve,
                },
            )
            if budget.remaining() <= 0:
                status = "budget_exhausted"
                break
        if stale >= 200:
            status = "early_stopped"
            break
    if status == "budget_exhausted":
        _write(output.with_suffix(".curve.json"), {"curve": curve, "status": status})
        return {"seed": seed, "status": status, "last_epoch": epoch}
    if best_state is None:
        raise RuntimeError("No nonlinear checkpoint was selected")
    network.load_state_dict(best_state)
    network.eval()
    _checkpoint(output.with_suffix(".pt"), {"state": network.state_dict(), "signature": signature})
    _write(output.with_suffix(".curve.json"), {"curve": curve, "status": status})

    def predictor(start, future):
        return _predict_nonlinear(
            network, context_mean, context_scale, difference_scale, target_scale, start, future
        )

    return {
        "seed": seed,
        "status": status,
        "best_epoch": best_epoch,
        "epochs_executed": epoch,
        "gate": measurement_gate(tuning, predictor),
        "model_sha256": sha256_file(output.with_suffix(".pt")),
        "scaling_sha256": sha256_file(scaling_path),
        "curve_sha256": sha256_file(output.with_suffix(".curve.json")),
    }


def _predict_nonlinear(
    network, context_mean, context_scale, difference_scale, target_scale, start, future
):
    delta = np.asarray(future) - np.asarray(start)
    context = np.broadcast_to(np.asarray(start), delta.shape)
    context_t = torch.tensor((context - context_mean) / context_scale, dtype=torch.float32)
    delta_t = torch.tensor(delta / difference_scale, dtype=torch.float32)
    with torch.inference_mode():
        return network(context_t, delta_t).numpy() * target_scale


def _selected_predictors(selection: dict) -> list:
    method = selection["method"]
    if method == "existing_ridge_difference":
        item = selection["models"][0]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Selected existing ridge hash mismatch")
        return [_baseline_predict]
    if method == "direct_ridge":
        path = Path(selection["models"][0]["path"])
        if sha256_file(path) != selection["models"][0]["sha256"]:
            raise ValueError("Selected ridge model hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            fitted = DirectRidge(arrays["scale"], arrays["coef"], float(arrays["alpha"]))
        return [fitted.predict]
    predictors = []
    for item in selection["models"]:
        prefix = Path(item["prefix"])
        for suffix, key in ((".pt", "model_sha256"), (".scaling.npz", "scaling_sha256")):
            if sha256_file(prefix.with_suffix(suffix)) != item[key]:
                raise ValueError("Selected nonlinear model hash mismatch")
        network = AnchoredNetwork()
        network.load_state_dict(
            torch.load(prefix.with_suffix(".pt"), map_location="cpu", weights_only=True)["state"]
        )
        network.eval()
        with np.load(prefix.with_suffix(".scaling.npz"), allow_pickle=False) as arrays:
            values = [
                arrays[key]
                for key in ("context_mean", "context_scale", "difference_scale", "target_scale")
            ]
        predictors.append(
            lambda start, future, network=network, values=values: _predict_nonlinear(
                network, *values, start, future
            )
        )
    return predictors


def fit(root: Path, budget: ExperimentBudget) -> dict:
    audited = _audit_required(root)
    report_path = root / "fit-report.json"
    if report_path.exists():
        report = _sealed(report_path)
        if report["audit_sha256"] != sha256_file(root / "audit.json"):
            raise ValueError("Fit report audit mismatch")
        return report
    model, pilot, fresh, parts, fingerprint = _model_context(audited)
    fit_data = _load_data(parts["fit"], pilot, fresh, model, fingerprint)
    tuning = _load_data(parts["tuning"], pilot, fresh, model, fingerprint)
    fit_families = {key: item[0] for key, item in fit_data.items()}
    fit_views = {key: item[1] for key, item in fit_data.items()}
    contexts, differences, targets, ids, dedup = deduplicated_training(fit_families, fit_views)
    # The 32-pair diagnostic uses the fit partition only and cannot select a model.
    rng = np.random.default_rng(
        json.loads((root / "config.json").read_text())["seed_streams"]["diagnostics"]
    )
    distinct_indices = np.unique(differences, axis=0, return_index=True)[1]
    if len(distinct_indices) < 32:
        raise ValueError("Too few distinct real transition differences for tiny diagnostic")
    subset = rng.choice(distinct_indices, size=32, replace=False)
    diagnostic = {
        "distinct_difference_count": int(len(np.unique(differences[subset], axis=0))),
        "fit_pair_count": len(targets),
        "deduplication": dedup,
        "ridge_zero_max": 0.0,
    }
    if diagnostic["distinct_difference_count"] != 32:
        raise ValueError("Tiny diagnostic pairs are not distinct")
    progress_path = root / "fit-progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "audit_sha256": sha256_file(root / "audit.json"),
            "cells": {},
            "diagnostic": diagnostic,
        }
    )
    if progress["audit_sha256"] != sha256_file(root / "audit.json"):
        raise ValueError("Fit progress audit mismatch")
    cells = progress["cells"]
    if "zero" not in cells:
        cells["zero"] = {"gate": measurement_gate(tuning, lambda start, future: np.zeros((6, 2)))}
        _write(progress_path, progress)
    if "existing_ridge_difference" not in cells:
        cells["existing_ridge_difference"] = {
            "gate": measurement_gate(tuning, _baseline_predict),
            "source_model_sha256": sha256_file(READOUT / "models" / "240-projected-ridge.npz"),
        }
        _write(progress_path, progress)
    selection = None
    if cells["existing_ridge_difference"]["gate"]["status"] == "pass":
        selection = {
            "method": "existing_ridge_difference",
            "models": [
                {
                    "path": str(READOUT / "models" / "240-projected-ridge.npz"),
                    "sha256": cells["existing_ridge_difference"]["source_model_sha256"],
                }
            ],
        }
    if selection is None:
        if "direct_ridge" not in cells:
            ridge, scores = select_ridge(differences, targets, ids, tuning)
            path = root / "models" / "direct-ridge.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **ridge.arrays())
            cells["direct_ridge"] = {
                "selected_alpha": ridge.alpha,
                "alpha_scores": scores,
                "gate": measurement_gate(tuning, ridge.predict),
                "model_sha256": sha256_file(path),
            }
            _write(progress_path, progress)
        ridge_cell = cells["direct_ridge"]
        if ridge_cell["gate"]["status"] == "pass":
            selection = {
                "method": "direct_ridge",
                "models": [
                    {
                        "path": str(root / "models" / "direct-ridge.npz"),
                        "sha256": ridge_cell["model_sha256"],
                    }
                ],
                "selected_alpha": ridge_cell["selected_alpha"],
            }
    if selection is None:
        for seed in SEEDS:
            key = f"nonlinear_{seed}"
            if key in cells and cells[key]["status"] != "budget_exhausted":
                continue
            if budget.remaining() <= 0:
                break
            prefix = root / "models" / f"anchored-{seed}"
            cells[key] = _nonlinear_fit(
                fit_data, tuning, seed, prefix, sha256_file(root / "audit.json"), budget
            )
            cells[key]["prefix"] = str(prefix)
            _write(progress_path, progress)
            if cells[key]["status"] == "budget_exhausted":
                break
        complete = all(
            f"nonlinear_{seed}" in cells
            and cells[f"nonlinear_{seed}"]["status"] != "budget_exhausted"
            for seed in SEEDS
        )
        if complete and all(
            cells[f"nonlinear_{seed}"]["gate"]["status"] == "pass" for seed in SEEDS
        ):
            selection = {
                "method": "nonlinear",
                "models": [cells[f"nonlinear_{seed}"] for seed in SEEDS],
            }
    else:
        complete = True
    if selection is None and not complete:
        result = {
            "status": "incomplete",
            "audit_sha256": sha256_file(root / "audit.json"),
            "cells": cells,
            "diagnostic": diagnostic,
            "selection": None,
        }
        _write(root / "fit-incomplete.json", result)
        return result
    selection_payload = {
        "audit_sha256": sha256_file(root / "audit.json"),
        "fit_family_ids_sha256": sha256_bytes(canonical_json(sorted(parts["fit"]))),
        "tuning_family_ids_sha256": sha256_bytes(canonical_json(sorted(parts["tuning"]))),
        "selection": selection,
        "tuning_gates": {key: value["gate"] for key, value in cells.items() if key != "zero"},
        "status": "selected" if selection else "no_passing_displacement_readout",
    }
    _seal(root / "selection.json", selection_payload)
    result = {
        "status": "complete",
        "audit_sha256": sha256_file(root / "audit.json"),
        "cells": cells,
        "diagnostic": diagnostic,
        "selection": selection,
        "confirmation_labels_used": False,
    }
    _seal(report_path, result)
    return result


def _selection_required(root: Path) -> dict:
    audited = _audit_required(root)
    report = _sealed(root / "fit-report.json")
    selected = _sealed(root / "selection.json")
    if report["audit_sha256"] != sha256_file(root / "audit.json") or selected[
        "audit_sha256"
    ] != sha256_file(root / "audit.json"):
        raise ValueError("Selection is bound to a different audit")
    if report["selection"] != selected["selection"] or audited["status"] != "passed":
        raise ValueError("Selection and fit report disagree")
    return selected


def confirm(root: Path, budget: ExperimentBudget) -> dict:
    selected = _selection_required(root)
    if selected["selection"] is None:
        return {
            "status": "unavailable",
            "reason": "no_readout_passed_tuning",
            "confirmation_opened": False,
        }
    report_path = root / "confirmation-report.json"
    if report_path.exists():
        report = _sealed(report_path)
        if report["selection_sha256"] != sha256_file(root / "selection.json"):
            raise ValueError("Sealed confirmation selection mismatch")
        return report
    audited = _audit_required(root)
    model, pilot, fresh, parts, fingerprint = _model_context(audited)
    budget.check()
    # No confirmation family data is loaded before the sealed selection exists.
    data = _load_data(parts["confirmation"], pilot, fresh, model, fingerprint)
    gates = [
        measurement_gate(data, predictor)
        for predictor in _selected_predictors(selected["selection"])
    ]
    status = (
        "pass"
        if all(gate["status"] == "pass" for gate in gates)
        else ("unassessable" if any(gate["status"] == "unassessable" for gate in gates) else "fail")
    )
    result = {
        "status": status,
        "selection_sha256": sha256_file(root / "selection.json"),
        "confirmation_opened": True,
        "family_count": len(data),
        "gates": gates,
        "readout_seed_count": len(gates),
    }
    _seal(report_path, result)
    return result


def _derangement(family_id: str, seed: int) -> np.ndarray:
    derived = int(sha256_bytes(canonical_json([family_id, seed]))[:16], 16)
    rng = np.random.default_rng(derived)
    while True:
        permutation = rng.permutation(6)
        if np.all(permutation != np.arange(6)):
            return permutation


def _prediction_embeddings(
    family_id: str, family: dict, model, scaler, cache: LatentCache, fingerprint: dict
) -> np.ndarray:
    inputs = {
        "history_frames": family["history_frames"],
        "warmup_controls": family["warmup_controls"],
        "candidate_controls": family["candidate_controls"],
    }
    key = cache.key("prediction", inputs, {**fingerprint, "family_id": family_id})
    path = cache.root / "prediction" / f"{key}.npz"
    if path.exists():
        values, _ = cache.read("prediction", key, {**fingerprint, "family_id": family_id})
        result = values[0]
    else:
        with torch.inference_mode():
            values = predict_future(
                model,
                preprocess_pixels(family["history_frames"]).unsqueeze(0),
                family["warmup_controls"][None],
                family["candidate_controls"][None],
                scaler,
            ).numpy()
        cache.write(
            "prediction",
            key,
            values,
            {**fingerprint, "family_id": family_id},
            {"family_id": family_id},
        )
        result = values[0]
    if result.shape != (6, 2, 192):
        raise ValueError("Frozen predictor alignment changed")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def evaluate(root: Path, budget: ExperimentBudget) -> dict:
    selected = _selection_required(root)
    if selected["selection"] is None:
        return {"status": "unavailable", "reason": "no_selected_readout"}
    confirmed = _sealed(root / "confirmation-report.json")
    if confirmed["selection_sha256"] != sha256_file(root / "selection.json"):
        raise ValueError("Confirmation is bound to another selection")
    if confirmed["status"] != "pass":
        return {"status": "unavailable", "reason": "confirmation_did_not_pass"}
    report_path = root / "evaluation-report.json"
    if report_path.exists():
        report = _sealed(report_path)
        if report["confirmation_sha256"] != sha256_file(root / "confirmation-report.json"):
            raise ValueError("Existing evaluation confirmation mismatch")
        return report
    audited = _audit_required(root)
    model, pilot, fresh, _, fingerprint = _model_context(audited)
    ids = [f"pilot:{key}" for key, split in pilot["splits"].items() if split == "development"]
    ids.sort()
    data = _load_data(ids, pilot, fresh, model, fingerprint)
    predictors = _selected_predictors(selected["selection"])
    control_gates = [measurement_gate(data, predictor) for predictor in predictors]
    control_passed = all(gate["status"] == "pass" for gate in control_gates)
    scaler = ActionScaler.from_json(STATS)
    cache = LatentCache(root / "latent-cache")
    seed_streams = json.loads((root / "config.json").read_text())["seed_streams"]
    config = EvaluationConfig(5.12, 512.0, 0.0, 2)
    by_cohort = {"challenge": {}, "ordinary": {}}
    order_error = None
    raw = {}
    for family_id in ids:
        budget.check()
        family, embeddings = data[family_id]
        original_id = family_id.split(":", 1)[1]
        truth = _truth(original_id, family, pilot)
        goals = tuple(
            Goal(f"goal-{index}", np.asarray(point))
            for index, point in enumerate(pilot["goals"][original_id])
        )
        predicted_latents = _prediction_embeddings(
            original_id, family, model, scaler, cache, fingerprint
        )
        permutation = _derangement(original_id, seed_streams["permutations"])
        if order_error is None:
            with torch.inference_mode():
                reordered = predict_future(
                    model,
                    preprocess_pixels(family["history_frames"]).unsqueeze(0),
                    family["warmup_controls"][None],
                    family["candidate_controls"][None, permutation],
                    scaler,
                ).numpy()[0]
            order_error = float(np.max(np.abs(reordered - predicted_latents[permutation])))
            if order_error > 1e-6:
                raise ValueError("Candidate order equivariance check failed")
        start, future, true_displacement = transition_targets(family, embeddings)
        latent_sets = {
            "real_frame_readout": future,
            "jepa": predicted_latents[:, -1],
            "action_permuted": predicted_latents[permutation, -1],
        }
        for seed_index, predictor in enumerate(predictors):
            methods = {name: predictor(start, latents) for name, latents in latent_sets.items()}
            methods["persistence"] = np.zeros((6, 2))
            methods["oracle"] = true_displacement
            for method, displacements in methods.items():
                label = f"seed{seed_index}:{method}" if len(predictors) > 1 else method
                branches = tuple(
                    BranchPrediction(
                        candidate_id, truth.current_block_position + displacements[index]
                    )
                    for index, candidate_id in enumerate(family["metadata"]["candidate_ids"])
                )
                result = evaluate_family(
                    truth,
                    branches,
                    goals,
                    config,
                    method=label,
                    goal_provenance="PILOT-001 fixed seeded grid; oracle-initialized scoring",
                )
                cohort = "challenge" if original_id.startswith("challenge") else "ordinary"
                by_cohort[cohort].setdefault(label, []).append(result)
                raw.setdefault(original_id, {})[label] = _jsonable(asdict(result))
    _write(root / "evaluation-families.json", raw)
    summaries = {}
    for cohort, methods in by_cohort.items():
        summaries[cohort] = {}
        for method, evaluations in methods.items():
            summaries[cohort][method] = {
                "aggregate": aggregate_families(tuple(evaluations)),
                "coverage": _coverage(evaluations),
                "bootstrap": _bootstrap(evaluations, seed_streams["bootstrap"]),
            }
        for seed_index in range(len(predictors)):
            prefix = f"seed{seed_index}:" if len(predictors) > 1 else ""
            jepa = methods[prefix + "jepa"]
            persistence = methods[prefix + "persistence"]
            summaries[cohort][prefix + "jepa_minus_persistence"] = _paired_bootstrap(
                jepa, persistence, seed_streams["bootstrap"]
            )
            summaries[cohort][prefix + "jepa_minus_action_permuted"] = _paired_bootstrap(
                jepa, methods[prefix + "action_permuted"], seed_streams["bootstrap"]
            )
    result = {
        "status": "exploratory" if control_passed else "physical_claims_unassessable",
        "confirmation_sha256": sha256_file(root / "confirmation-report.json"),
        "real_frame_control_gates": control_gates,
        "candidate_order_equivariance_max_error": order_error,
        "oracle_initialized_regret": True,
        "families_sha256": sha256_file(root / "evaluation-families.json"),
        "cohorts": summaries,
        "limitations": [
            "Pilot development families were previously examined; all comparisons are exploratory.",
            "True branch-point block position is supplied only to scoring; "
            "this is not an autonomous visual planner.",
            "Sparse contact comparisons with fewer than ten families are unavailable.",
        ],
    }
    _seal(report_path, result)
    return result


def report(root: Path) -> dict:
    _audit_required(root)
    fit_path = root / "fit-report.json"
    fit_result = _sealed(fit_path) if fit_path.exists() else None
    selection = _sealed(root / "selection.json") if (root / "selection.json").exists() else None
    confirmation = (
        _sealed(root / "confirmation-report.json")
        if (root / "confirmation-report.json").exists()
        else None
    )
    evaluation = (
        _sealed(root / "evaluation-report.json")
        if (root / "evaluation-report.json").exists()
        else None
    )
    if fit_result is None or fit_result["status"] != "complete":
        status, recommendation = "incomplete", "resume_within_budget"
    elif selection is None or selection["selection"] is None:
        status, recommendation = "measurement_failed", "close_physical_readout_recovery"
    elif confirmation is None:
        status, recommendation = "confirmation_pending", "run_one_time_confirmation"
    elif confirmation["status"] != "pass":
        status, recommendation = "confirmation_failed", "close_physical_readout_recovery"
    elif evaluation is None:
        status, recommendation = "evaluation_pending", "run_exploratory_comparison"
    else:
        status, recommendation = evaluation["status"], "review_exploratory_evidence"
    final = {
        "task": "DELTA-001",
        "status": status,
        "recommendation": recommendation,
        "audit_sha256": sha256_file(root / "audit.json"),
        "fit_sha256": sha256_file(fit_path) if fit_result else None,
        "selection_sha256": sha256_file(root / "selection.json") if selection else None,
        "confirmation_sha256": sha256_file(root / "confirmation-report.json")
        if confirmation
        else None,
        "evaluation_sha256": sha256_file(root / "evaluation-report.json") if evaluation else None,
        "measurement_cells": fit_result["cells"] if fit_result else None,
        "confirmation": confirmation,
        "runtime": json.loads((root / "runtime.json").read_text()),
        "artifact_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "limitations": [
            "Displacement accuracy does not validate the earlier failed "
            "absolute-position readouts.",
            "One checkpoint and unproven historical simulator identity limit generalization.",
            "Main-study locked test data were not inspected.",
        ],
    }
    _seal(root / "final-report.json", final)
    lines = [
        "# DELTA-001 displacement measurement",
        "",
        f"Status: **{status.replace('_', ' ')}**. "
        f"Recommendation: **{recommendation.replace('_', ' ')}**.",
        "",
        "## Tuning gates",
        "",
        "Mean <=1.28; family-weighted p95 <=2.56; at least ten families per required subset.",
        "",
    ]
    if fit_result:
        for name, cell in fit_result["cells"].items():
            gate = cell["gate"]
            lines.append(
                f"- {name}: {gate['status']}; endpoint mean "
                f"{gate['required']['endpoint_overall']['mean']:.3f}, "
                f"meaningful-pair mean {gate['required']['effect_meaningful']['mean']}"
            )
    lines.extend(
        [
            "",
            f"Confirmation opened: **{'yes' if confirmation else 'no'}**.",
            "",
            "The JSON artifacts contain family, contact, effect, and timing details.",
            "",
            *[f"- {item}" for item in final["limitations"]],
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "fit", "confirm", "evaluate", "report"))
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    signature = sha256_bytes(canonical_json(_sources()))
    budget = ExperimentBudget(args.output_dir, signature)
    budget.begin(args.stage)
    try:
        if args.stage == "audit":
            result = audit(args.output_dir)
        elif args.stage == "fit":
            result = fit(args.output_dir, budget)
        elif args.stage == "confirm":
            result = confirm(args.output_dir, budget)
        elif args.stage == "evaluate":
            result = evaluate(args.output_dir, budget)
        else:
            result = report(args.output_dir)
    except Exception:
        budget.finish("error")
        raise
    budget.finish(result.get("status", "complete"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
