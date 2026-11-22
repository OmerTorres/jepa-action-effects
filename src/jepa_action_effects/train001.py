"""TRAIN-001: does magnitude-balanced fine-tuning correct prediction errors?

    python -m jepa_action_effects.train001 audit
    python -m jepa_action_effects.train001 collect-development   # simulator/archive only
    python -m jepa_action_effects.train001 cache                 # frozen encoder only
    python -m jepa_action_effects.train001 pilot                 # eligibility, then gates
    python -m jepa_action_effects.train001 train                 # six runs, calibration, dev eval
    python -m jepa_action_effects.train001 collect-confirmation  # fresh families + encoding
    python -m jepa_action_effects.train001 seal
    python -m jepa_action_effects.train001 confirm               # strict validation first
    python -m jepa_action_effects.train001 report

Add `--smoke` to run the tiny implementation check in `outputs/train001-smoke`.
Protocol: docs/TRAIN-001_PLAN.md with the D028 amendments. Every stage is charged to
the cumulative 24-hour ledger (`train001_budget`). Stages are idempotent: sealed
outputs are returned, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import seal, sealed, write_json
from jepa_action_effects.next_common import ARCHIVE_H5, CHECKPOINT, runtime_info, source_hashes
from jepa_action_effects.pilot_collection import load_family, sha256_file
from jepa_action_effects.train001_budget import Budget, BudgetLocked, BudgetStop
from jepa_action_effects.train001_collect import (
    DataIntegrityError,
    _atomic_npz,
    canonical_sha256,
    collect_phase,
    exclusion_registry,
    load_window,
)
from jepa_action_effects.train001_protocol import (
    CONDITIONS,
    DECISION_RULES,
    MAGNITUDES,
    READOUTS,
    SCENES,
    Protocol,
)
from jepa_action_effects.train001_provenance import (
    COMPLETE,
    ENTRY,
    model_ids,
    record_key,
    require_valid,
)

READOUT2 = Path("outputs/readout002")
ARCHIVE_H5_BYTES = 46_300_921_856
ARCHIVE_H5_SHA256 = "b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb"
STAGES = (
    "audit",
    "collect-development",
    "cache",
    "pilot",
    "train",
    "collect-confirmation",
    "seal",
    "confirm",
    "report",
)


class GateFailure(RuntimeError):
    """An implementation, optimization or budget gate failed; dependent stages stop."""


# --------------------------------------------------------------------------- provenance
def sources() -> dict[str, str]:
    """Every module that affects collection, inference or statistics, plus the lock."""
    here = Path(__file__).parent
    names = (
        "train001.py",
        "train001_protocol.py",
        "train001_budget.py",
        "train001_collect.py",
        "train001_model.py",
        "train001_metrics.py",
        "train001_provenance.py",
        "model_inference.py",
        "interface_audit.py",
        "next_common.py",
        "latent_audit.py",
        "latent_cache.py",
        "pilot_collection.py",
        "simulator_replay.py",
        "shift001.py",
        "shift001_metrics.py",
        "followup_metrics.py",
        "protocol_gates.py",
        "provenance.py",
        "readout_scale.py",
    )
    return source_hashes({name: here / name for name in names} | {"uv.lock": Path("uv.lock")})


def _git() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _require(path: Path, message: str) -> dict:
    if not path.exists():
        raise GateFailure(message)
    return sealed(path)


def frozen_fingerprint() -> dict[str, Any]:
    from jepa_action_effects.model_inference import ActionScaler, model_fingerprint
    from jepa_action_effects.next_common import STATS

    return model_fingerprint(CHECKPOINT, ActionScaler.from_json(STATS))


def prediction_fingerprint(
    base: dict[str, Any], model_id: str, entry: dict[str, Any], threads: int
):
    """Identity of every prediction record: checkpoint bytes, condition and seed included."""
    return {
        **base,
        "model_id": model_id,
        "kind": entry["kind"],
        "checkpoint_sha256": entry.get("checkpoint_sha256"),
        "condition": entry.get("condition"),
        "training_seed": entry.get("seed"),
        "train001_inference_sha256": sha256_file(Path(__file__).with_name("train001_model.py")),
        "threads": threads,
    }


def base_weights_sha256() -> str:
    from jepa_action_effects.interface_audit import WEIGHTS_SHA256

    return WEIGHTS_SHA256


# --------------------------------------------------------------------------- readouts
def readout_decoders(root: Path = READOUT2, expected: dict[str, str] | None = None):
    """Three READOUT-002 MLPs: projected (N,192) -> block (N,2); primary first."""
    from jepa_action_effects.readout_scale import mlp_predictor

    selection = sealed(root / "selection.json")
    if selection["selected"] != READOUTS[0]:
        raise ValueError("The primary readout must be the sealed READOUT-002 selection")
    decoders, digests = {}, {}
    for name in READOUTS:
        path = root / "models" / f"{name}.npz"
        digest = sha256_file(path)
        if expected is not None and expected[name] != digest:
            raise ValueError(f"Readout artifact changed: {name}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        state = {
            "network": {k[4:]: v for k, v in arrays.items() if k.startswith("net.")},
            **{k: arrays[k] for k in ("x_mean", "x_scale", "y_mean", "y_scale")},
        }
        full = mlp_predictor(state)
        decoders[name] = (lambda f: lambda x: f(np.asarray(x, np.float64))[:, :2])(full)
        digests[name] = digest
    if digests[READOUTS[0]] != selection["path_sha256"]:
        raise ValueError("Primary readout bytes differ from the sealed selection")
    return decoders, digests


# --------------------------------------------------------------------------- stage: audit
def audit(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    root = protocol.run_root
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != sources() or sealed(root / "config.json") != protocol.payload():
            raise GateFailure("Sources or protocol changed after audit: use a new run identity")
        return old
    from jepa_action_effects.doctor import diagnose
    from jepa_action_effects.interface_audit import CONFIG_SHA256, WEIGHTS_SHA256
    from jepa_action_effects.next_common import STATS

    checkpoint = {
        "weights_sha256": sha256_file(CHECKPOINT / "weights.pt"),
        "config_sha256": sha256_file(CHECKPOINT / "config.json"),
        "scaler_sha256": sha256_file(STATS),
    }
    if (
        checkpoint["weights_sha256"] != WEIGHTS_SHA256
        or checkpoint["config_sha256"] != CONFIG_SHA256
    ):
        raise GateFailure("Checkpoint bytes differ from the audited revision")
    size = ARCHIVE_H5.stat().st_size
    budget.unit("archive_hash", 600.0)
    digest = sha256_file(ARCHIVE_H5) if protocol.name == "train001" else "not-hashed-in-smoke"
    if size != ARCHIVE_H5_BYTES or (protocol.name == "train001" and digest != ARCHIVE_H5_SHA256):
        raise GateFailure("Archive HDF5 differs from the pinned extraction")
    _, readouts = readout_decoders()
    registry = exclusion_registry(Path("outputs"), root)
    exclusions_sha = seal(root / "exclusions.json", registry)
    config_sha = seal(root / "config.json", protocol.payload())
    result = {
        "status": "passed",
        "sources": sources(),
        "config_sha256": config_sha,
        "exclusions_sha256": exclusions_sha,
        "exclusion_counts": {
            "seeds": len(registry["seeds"]),
            "episodes": len(registry["episodes"]),
            "history_sha256": len(registry["history_sha256"]),
            "by_source": registry["sources"],
        },
        "checkpoint": checkpoint,
        "archive": {"path": str(ARCHIVE_H5), "bytes": size, "sha256": digest},
        "readouts": readouts,
        "git": _git(),
        "doctor": diagnose(),
        "runtime": runtime_info(),
    }
    seal(root / "audit.json", result)
    return result


def _h5():
    import h5py

    return h5py.File(ARCHIVE_H5, "r")


def _check_audit(protocol: Protocol) -> dict:
    audit_result = _require(protocol.run_root / "audit.json", "Run audit first")
    if audit_result["sources"] != sources():
        raise GateFailure("Sources changed after audit: use a new run identity")
    if sealed(protocol.run_root / "config.json") != protocol.payload():
        raise GateFailure("Protocol changed after audit: use a new run identity")
    return audit_result


# --------------------------------------------------------------------------- stage: collect
def collect(protocol: Protocol, budget: Budget, phase: str) -> dict[str, Any]:
    root = protocol.run_root
    _check_audit(protocol)
    path = root / phase / "collection.json"
    if path.exists():
        return sealed(path)
    registry = sealed(root / "exclusions.json")
    prior = [] if phase == "development" else [sealed(root / "development" / "collection.json")]
    started = time.monotonic()
    with _h5() as h5:
        collection = collect_phase(phase, protocol, root, registry, h5, prior, budget)
    collection["seconds"] = time.monotonic() - started
    seal(path, collection)
    return collection


# --------------------------------------------------------------------------- stage: cache
def build_cache(protocol: Protocol, budget: Budget, phase: str) -> dict[str, Any]:
    from jepa_action_effects.next_common import encode_frames, load_model

    root = protocol.run_root
    manifest = root / "cache" / f"{phase}.json"
    if manifest.exists():
        return sealed(manifest)
    collection = _require(root / phase / "collection.json", f"Collect {phase} first")
    model, _, fingerprint = load_model()
    out = root / "cache" / phase
    records: dict[str, dict[str, Any]] = {}
    seconds = {"families": 0.0, "windows": 0.0}
    for kind, entries in (("families", collection["families"]), ("windows", collection["windows"])):
        for unit, entry in sorted(entries.items()):
            budget.unit("encode")
            began = time.monotonic()
            source = root / entry["relative_path"]
            if sha256_file(source) != entry["file_sha256"]:
                raise DataIntegrityError(f"{unit} changed after admission")
            if kind == "families":
                family = load_family(source)
                frames = np.concatenate(
                    [family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3)]
                )
                z = encode_frames(model, frames)["projected"].astype(np.float32)
                arrays = {
                    "history_z": z[:3],
                    "future_z": z[3:].reshape(len(MAGNITUDES), 2, -1),
                    "history_states": family["history_states"],
                    "future_states": family["future_states"],
                    "past_controls": family["warmup_controls"][-10:].astype(np.float32),
                    "candidate_controls": family["candidate_controls"].astype(np.float32),
                }
            else:
                window = load_window(source)
                arrays = {
                    "z": encode_frames(model, window["frames"])["projected"].astype(np.float32),
                    "states": window["states"],
                    "controls": window["controls"].astype(np.float32),
                }
            path = out / f"{unit}.npz"
            _atomic_npz(path, arrays, {"unit": unit, "input_sha256": entry["file_sha256"]})
            records[unit] = {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "input_sha256": entry["file_sha256"],
                "kind": "family" if kind == "families" else "window",
                "scene": entry["scene"],
                "partition": entry["partition"],
            }
            elapsed = time.monotonic() - began
            seconds[kind] += elapsed
            budget.record_unit("encode", elapsed)
    payload = {"phase": phase, "fingerprint": fingerprint, "records": records, "seconds": seconds}
    seal(manifest, payload)
    return payload


def load_cache(root: Path, manifest: dict, units: list[str]) -> list[dict[str, Any]]:
    result = []
    for unit in units:
        entry = manifest["records"][unit]
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise DataIntegrityError(f"Cache record changed: {unit}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        result.append({**arrays, "id": unit, "scene": entry["scene"], "kind": entry["kind"]})
    return result


def _units(collection: dict, partition: str, flag: str | None = None) -> list[str]:
    return sorted(
        unit
        for unit, record in collection["families"].items()
        if record["partition"] == partition and (flag is None or record.get(flag))
    )


# --------------------------------------------------------------------------- scoring
def real_record(cache: dict[str, Any], decoders: dict) -> dict[str, Any]:
    """Truth and real-future readout decodes; independent of any predictor."""
    if cache["kind"] == "family":
        start_z, end_z = cache["history_z"][2:3], cache["future_z"][:, 1]
        true_start = cache["history_states"][2, 2:4]
        true_end = cache["future_states"][:, -1, 2:4]
    else:
        start_z, end_z = cache["z"][2:3], cache["z"][4:5]
        true_start, true_end = cache["states"][2, 2:4], cache["states"][4:5, 2:4]
    return {
        "unit": cache["id"],
        "kind": cache["kind"],
        "scene": cache["scene"],
        "true_start": np.asarray(true_start, float).tolist(),
        "true_end": np.asarray(true_end, float).tolist(),
        "decoded": {
            name: {"start": decode(start_z)[0].tolist(), "real_end": decode(end_z).tolist()}
            for name, decode in decoders.items()
        },
    }


def score_model(
    model, caches: list[dict], decoders: dict, scaler, protocol: Protocol
) -> list[dict]:
    """Correct and action-permuted predictions (families) or single-branch windows."""
    import torch

    from jepa_action_effects.train001_metrics import derangement
    from jepa_action_effects.train001_model import normalized_actions, predict

    families = [c for c in caches if c["kind"] == "family"]
    windows = [c for c in caches if c["kind"] == "window"]
    records: list[dict] = []
    if families:
        history = torch.from_numpy(np.stack([c["history_z"] for c in families]))
        actions = normalized_actions(
            scaler,
            np.stack([c["past_controls"] for c in families]),
            np.stack([c["candidate_controls"] for c in families]),
        ).float()
        permutations = [derangement(c["id"], protocol.permutation_seed) for c in families]
        permuted = torch.stack([actions[i, p] for i, p in enumerate(permutations)])
        correct = predict(model, history, actions)
        wrong = predict(model, history, permuted)
        for i, cache in enumerate(families):
            target = cache["future_z"]
            records.append(
                {
                    "unit": cache["id"],
                    "kind": "family",
                    "scene": cache["scene"],
                    "permutation": permutations[i],
                    "latent_mse": ((correct[i] - target) ** 2).mean(axis=2).tolist(),
                    "latent_mse_permuted": ((wrong[i] - target) ** 2).mean(axis=2).tolist(),
                    "decoded": {
                        name: {
                            "pred_end": decode(correct[i][:, 1]).tolist(),
                            "perm_end": decode(wrong[i][:, 1]).tolist(),
                        }
                        for name, decode in decoders.items()
                    },
                }
            )
    if windows:
        history = torch.from_numpy(np.stack([c["z"][:3] for c in windows]))
        actions = normalized_actions(
            scaler,
            np.stack([c["controls"][:10] for c in windows]),
            np.stack([c["controls"][None, 10:] for c in windows]),
        ).float()
        predicted = predict(model, history, actions)
        for i, cache in enumerate(windows):
            records.append(
                {
                    "unit": cache["id"],
                    "kind": "window",
                    "scene": "ordinary",
                    "latent_mse": ((predicted[i] - cache["z"][None, 3:5]) ** 2)
                    .mean(axis=2)
                    .tolist(),
                    "decoded": {
                        name: {"pred_end": decode(predicted[i][:, 1]).tolist()}
                        for name, decode in decoders.items()
                    },
                }
            )
    return records


def _model_factory():
    from jepa_action_effects.model_inference import build_frozen_model

    return lambda: build_frozen_model(CHECKPOINT)


def _scaler():
    from jepa_action_effects.model_inference import ActionScaler
    from jepa_action_effects.next_common import STATS

    return ActionScaler.from_json(STATS)


def _weights(collection: dict) -> dict[str, tuple[float, ...]]:
    return {
        "archive_weighted": tuple(float(w) for w in collection["weights"]["weights"]),
        "balanced": tuple(1.0 / len(MAGNITUDES) for _ in MAGNITUDES),
    }


def _spec(protocol: Protocol, condition: str, seed: int, lr: float, updates: int, weights):
    from jepa_action_effects.train001_model import RunSpec

    return RunSpec(
        condition=condition,
        seed=seed,
        learning_rate=lr,
        updates=updates,
        weights=tuple(weights),
        weight_decay=protocol.weight_decay,
        gradient_clip=protocol.gradient_clip,
        per_scene=protocol.families_per_scene_batch,
        threads=protocol.threads,
    )


# --------------------------------------------------------------------------- stage: pilot
def eligibility(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    """Development measurement gates from real-frame controls only, before any training."""
    from jepa_action_effects.train001_metrics import family_table, measurement_gates

    root = protocol.run_root
    path = root / "pilot" / "eligibility.json"
    if path.exists():
        return sealed(path)
    collection = sealed(root / "development" / "collection.json")
    cache = sealed(root / "cache" / "development.json")
    decoders, digests = readout_decoders()
    units = _units(collection, "validation")
    budget.unit("eligibility")
    real = [
        real_record(c, {READOUTS[0]: decoders[READOUTS[0]]}) for c in load_cache(root, cache, units)
    ]
    gates = measurement_gates(
        family_table(real, READOUTS[0]), protocol, protocol.gate_bootstrap_seed
    )
    payload = {
        "cohort": "development validation families",
        "families": len(units),
        "readout": READOUTS[0],
        "readout_sha256": digests[READOUTS[0]],
        "gates": gates,
        "status": "eligible" if gates["passed"] else "not_assessable_under_measurement_protocol",
        "policy": DECISION_RULES["gate_policy"],
        "computed_before_training": not (root / "runs").exists(),
    }
    seal(path, payload)
    return payload


def pilot(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    import torch

    from jepa_action_effects.train001_model import (
        family_tensors,
        implementation_checks,
        latent_mse,
        rollout_equivalence,
        state_digest,
        train_run,
        trainable_state,
    )

    root = protocol.run_root
    _check_audit(protocol)
    eligible = eligibility(protocol, budget)
    path = root / "pilot" / "pilot.json"
    if path.exists():
        return sealed(path)
    torch.set_num_threads(protocol.threads)
    collection = sealed(root / "development" / "collection.json")
    cache = sealed(root / "cache" / "development.json")
    scaler = _scaler()
    factory = _model_factory()
    digest = state_digest(trainable_state(factory()))
    pilot_train = family_tensors(
        load_cache(root, cache, _units(collection, "train", "pilot")), scaler
    )
    pilot_val = family_tensors(
        load_cache(root, cache, _units(collection, "validation", "pilot")), scaler
    )
    overfit = family_tensors(
        load_cache(root, cache, _units(collection, "train", "overfit")), scaler
    )
    report: dict[str, Any] = {"eligibility_status": eligible["status"]}

    budget.unit("implementation_checks")
    report["implementation"] = implementation_checks(factory, digest, pilot_train)
    frozen = factory()
    pixel_families = [
        load_family(root / collection["families"][unit]["relative_path"]) for unit in pilot_val.ids
    ]
    report["equivalence"] = rollout_equivalence(frozen, scaler, pixel_families, pilot_val)
    report["equivalence"]["tolerance"] = protocol.equivalence_tolerance

    seed = protocol.training_seeds[0]
    overfit_spec = _spec(
        protocol,
        "balanced",
        seed,
        protocol.overfit_learning_rate,
        protocol.overfit_updates,
        _weights(collection)["balanced"],
    )
    result = train_run(
        factory,
        digest,
        overfit,
        overfit_spec,
        budget=budget,
        eval_every=protocol.overfit_check_every,
        eval_data=overfit,
    )
    losses = [value for _, value in result["eval_curve"]]
    reduction = 1.0 - min(losses[1:]) / losses[0]
    report["overfit"] = {
        "families": len(overfit.ids),
        "curve": result["eval_curve"],
        "reduction": reduction,
        "required": protocol.overfit_required_reduction,
        "passed": reduction >= protocol.overfit_required_reduction,
    }

    trials, train_seconds, train_updates = [], 0.0, 0
    weights = _weights(collection)
    for lr in protocol.learning_rates:
        for condition in CONDITIONS:
            began = time.monotonic()
            run = train_run(
                factory,
                digest,
                pilot_train,
                _spec(protocol, condition, seed, lr, protocol.sweep_updates, weights[condition]),
                budget=budget,
            )
            elapsed = time.monotonic() - began
            train_seconds += elapsed
            train_updates += protocol.sweep_updates
            trials.append(
                {
                    "learning_rate": lr,
                    "condition": condition,
                    "validation_latent_mse": latent_mse(run["model"], pilot_val),
                    "final_train_loss": run["curve"][-1][1],
                    "seconds": elapsed,
                }
            )
    table = {
        str(lr): float(
            np.mean([t["validation_latent_mse"] for t in trials if t["learning_rate"] == lr])
        )
        for lr in protocol.learning_rates
    }
    best = min(table.values())
    selected = min(
        lr
        for lr in protocol.learning_rates
        if table[str(lr)] <= best * (1 + protocol.tie_relative_tolerance)
    )
    report["learning_rate_sweep"] = {"trials": trials, "mean_by_rate": table}
    report["selected_learning_rate"] = selected

    started = time.monotonic()
    decoders, _ = readout_decoders()
    caches = load_cache(root, cache, pilot_val.ids)
    score_model(frozen, caches, decoders, scaler, protocol)
    eval_per_family = (time.monotonic() - started) / max(1, len(caches))
    report["throughput"] = throughput = _projection(
        protocol, collection, cache, train_seconds / train_updates, eval_per_family, budget
    )
    checks = {
        "implementation": report["implementation"]["passed"],
        "equivalence": report["equivalence"]["max_abs"] <= protocol.equivalence_tolerance,
        "overfit": report["overfit"]["passed"],
        "budget": throughput["fits"],
    }
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    report["runtime"] = runtime_info() | {"threads": torch.get_num_threads()}
    seal(path, report)
    if report["status"] != "passed":
        raise GateFailure(f"Pilot gate failed: {checks}")
    return report


def _projection(protocol, collection, cache, seconds_per_update, eval_per_family, budget) -> dict:
    families = len(collection["families"])
    windows = max(1, len(collection["windows"]))
    collect_per_family = collection["seconds"] / (families + windows)
    encode_family = cache["seconds"]["families"] / families
    encode_window = cache["seconds"]["windows"] / windows
    runs = len(CONDITIONS) * len(protocol.training_seeds)
    validation = 2 * protocol.validation_per_scene
    conf_families = 2 * protocol.confirmation_per_scene
    ordinary = protocol.ordinary_confirmation
    work = {
        "training": runs * protocol.updates * seconds_per_update,
        "validation_curves": runs
        * (protocol.updates // protocol.validation_every)
        * validation
        * eval_per_family,
        "calibration_and_development_eval": (2 * protocol.train_per_scene + 7 * validation)
        * eval_per_family,
        "confirmation_collection": (conf_families + ordinary) * collect_per_family,
        "confirmation_encoding": conf_families * encode_family + ordinary * encode_window,
        "confirmation_scoring": 8 * (conf_families + ordinary) * eval_per_family,
    }
    projected = sum(work.values()) * protocol.contingency
    remaining = protocol.budget_seconds - budget.used()
    return {
        "seconds_per_update": seconds_per_update,
        "eval_seconds_per_family": eval_per_family,
        "collect_seconds_per_unit": collect_per_family,
        "encode_seconds_per_family": encode_family,
        "work_seconds": work,
        "contingency": protocol.contingency,
        "projected_seconds": projected,
        "remaining_seconds": remaining,
        "used_seconds": budget.used(),
        "fits": projected <= remaining,
    }


# --------------------------------------------------------------------------- stage: train
def train(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    import torch

    from jepa_action_effects.train001_model import (
        family_tensors,
        state_digest,
        train_run,
        trainable_state,
    )

    root = protocol.run_root
    _check_audit(protocol)
    pilot_report = _require(root / "pilot" / "pilot.json", "Run the pilot first")
    if pilot_report["status"] != "passed":
        raise GateFailure("Pilot did not pass; training is blocked")
    eligibility_path = root / "pilot" / "eligibility.json"
    _require(eligibility_path, "Eligibility must be sealed before training")
    torch.set_num_threads(protocol.threads)
    collection = sealed(root / "development" / "collection.json")
    cache = sealed(root / "cache" / "development.json")
    scaler = _scaler()
    factory = _model_factory()
    digest = state_digest(trainable_state(factory()))
    data = family_tensors(load_cache(root, cache, _units(collection, "train")), scaler)
    validation = family_tensors(load_cache(root, cache, _units(collection, "validation")), scaler)
    weights = _weights(collection)
    lr = pilot_report["selected_learning_rate"]
    runs = {}
    for seed in protocol.training_seeds:
        for condition in CONDITIONS:
            run_dir = root / "runs" / f"{condition}-{seed}"
            record_path = run_dir / "run.json"
            if record_path.exists():
                runs[f"{condition}-{seed}"] = sealed(record_path)
                continue
            spec = _spec(protocol, condition, seed, lr, protocol.updates, weights[condition])
            spec.checkpoint_updates = protocol.checkpoint_updates
            spec.resume_every = protocol.resume_every
            spec.curve_every = protocol.curve_every
            spec.validation_every = protocol.validation_every
            began = time.monotonic()
            result = train_run(factory, digest, data, spec, run_dir, validation, budget)
            payload = {
                "condition": condition,
                "seed": seed,
                "learning_rate": lr,
                "updates": protocol.updates,
                "weights": list(weights[condition]),
                "spec": spec.__dict__,
                "base_weights_sha256": base_weights_sha256(),
                "base_trainable_digest": digest,
                "eligibility_sha256": sha256_file(eligibility_path),
                "pilot_sha256": sha256_file(root / "pilot" / "pilot.json"),
                "collection_sha256": sha256_file(root / "development" / "collection.json"),
                "cache_sha256": sha256_file(root / "cache" / "development.json"),
                "checkpoints": result["checkpoints"],
                "final_checkpoint": str(
                    (run_dir / f"checkpoint-{protocol.updates:05d}.pt").relative_to(root)
                ),
                "curve": result["curve"],
                "validation_curve": result["validation_curve"],
                "final_state_digest": result["final_state_digest"],
                "seconds_this_session": time.monotonic() - began,
                "runtime": runtime_info() | {"threads": torch.get_num_threads()},
            }
            seal(record_path, payload)
            runs[f"{condition}-{seed}"] = payload
    calibration = fit_calibration_stage(protocol, budget, collection, cache, scaler)
    development = development_evaluation(protocol, budget, collection, cache, scaler, calibration)
    return {"runs": sorted(runs), "calibration": calibration, "development": bool(development)}


def fit_calibration_stage(protocol, budget, collection, cache, scaler) -> dict[str, Any]:
    from jepa_action_effects.train001_metrics import fit_calibration

    root = protocol.run_root
    path = root / "calibration.json"
    if path.exists():
        return sealed(path)
    decoders, digests = readout_decoders()
    budget.unit("calibration")
    caches = load_cache(root, cache, _units(collection, "train"))
    frozen = _model_factory()()
    predictions = score_model(frozen, caches, decoders, scaler, protocol)
    real = [real_record(c, decoders) for c in caches]
    scenes = np.asarray([SCENES.index(r["scene"]) for r in real])
    true = (
        np.asarray([r["true_end"] for r in real])
        - np.asarray([r["true_start"] for r in real])[:, None]
    )
    scales = {}
    for name in READOUTS:
        start = np.asarray([r["decoded"][name]["start"] for r in real])[:, None]
        predicted = np.asarray([p["decoded"][name]["pred_end"] for p in predictions]) - start
        scales[name] = fit_calibration(predicted, true, scenes)
    payload = {
        "label": "state-supervised output correction of the frozen model (not a latent predictor)",
        "families": len(caches),
        "scales": scales,
        "readout_sha256": digests,
    }
    seal(path, payload)
    return payload


def _evaluate_models(protocol, budget, caches, decoders, scaler, registry_paths) -> dict[str, list]:
    from jepa_action_effects.train001_model import load_trained

    factory = _model_factory()
    records = {}
    for model_id, checkpoint in registry_paths.items():
        budget.unit("model_eval")
        model = factory()
        if checkpoint is not None:
            model = load_trained(model, checkpoint[0], checkpoint[1])
        records[model_id] = score_model(model, caches, decoders, scaler, protocol)
    return records


def development_evaluation(protocol, budget, collection, cache, scaler, calibration) -> dict:
    """Exploratory development evaluation; eligibility is already sealed and unchanged."""
    from jepa_action_effects.train001_metrics import summarize

    root = protocol.run_root
    path = root / "development-eval.json"
    if path.exists():
        return sealed(path)
    eligible = _require(root / "pilot" / "eligibility.json", "Eligibility missing")
    decoders, _ = readout_decoders()
    families = load_cache(root, cache, _units(collection, "validation"))
    windows = load_cache(root, cache, sorted(collection["windows"]))
    paths = {"frozen": None}
    for seed in protocol.training_seeds:
        for condition in CONDITIONS:
            run = sealed(root / "runs" / f"{condition}-{seed}" / "run.json")
            paths[f"{condition}-{seed}"] = (
                root / run["final_checkpoint"],
                run["checkpoints"][str(protocol.updates)],
            )
    records = _evaluate_models(protocol, budget, families + windows, decoders, scaler, paths)
    split = {
        m: ([r for r in v if r["kind"] == "family"], [r for r in v if r["kind"] == "window"])
        for m, v in records.items()
    }
    summary = summarize(
        [real_record(c, decoders) for c in families],
        {m: v[0] for m, v in split.items()},
        [real_record(c, decoders) for c in windows],
        {m: v[1] for m, v in split.items()},
        protocol,
        calibration["scales"],
        eligible["status"] == "eligible",
        READOUTS,
        "development (exploratory)",
    )
    summary["label"] = "EXPLORATORY development evaluation; not confirmation evidence"
    seal(path, summary)
    return summary


# --------------------------------------------------------------------------- stage: confirmation
def collect_confirmation(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    root = protocol.run_root
    for seed in protocol.training_seeds:
        for condition in CONDITIONS:
            _require(root / "runs" / f"{condition}-{seed}" / "run.json", "Training incomplete")
    _require(root / "development-eval.json", "Development evaluation incomplete")
    collection = collect(protocol, budget, "confirmation")
    build_cache(protocol, budget, "confirmation")
    return {"counts": collection["counts"]}


def build_seal(
    root: Path,
    protocol: Protocol,
    current_sources: dict[str, str],
    base_fingerprint: dict[str, Any],
    readout_digests: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = {
        "config": "config.json",
        "audit": "audit.json",
        "exclusions": "exclusions.json",
        "collection_development": "development/collection.json",
        "collection_confirmation": "confirmation/collection.json",
        "cache_development": "cache/development.json",
        "cache_confirmation": "cache/confirmation.json",
        "eligibility": "pilot/eligibility.json",
        "pilot": "pilot/pilot.json",
        "calibration": "calibration.json",
    }
    for relative in files.values():
        sealed(root / relative)
    pilot_report = sealed(root / "pilot" / "pilot.json")
    confirmation = sealed(root / "confirmation" / "collection.json")
    development = sealed(root / "development" / "collection.json")
    registry: dict[str, dict[str, Any]] = {
        "real-frames": {"kind": "real"},
        "frozen": {"kind": "frozen", "checkpoint_sha256": base_weights_sha256()},
    }
    for condition in CONDITIONS:
        for seed in protocol.training_seeds:
            run_path = root / "runs" / f"{condition}-{seed}" / "run.json"
            run = sealed(run_path)
            registry[f"{condition}-{seed}"] = {
                "kind": "trained",
                "condition": condition,
                "seed": seed,
                "learning_rate": run["learning_rate"],
                "updates": run["updates"],
                "run": str(run_path.relative_to(root)),
                "run_sha256": sha256_file(run_path),
                "checkpoint": run["final_checkpoint"],
                "checkpoint_sha256": sha256_file(root / run["final_checkpoint"]),
                "base_weights_sha256": run["base_weights_sha256"],
            }
    fingerprints = {
        model_id: canonical_sha256(
            base_fingerprint
            if entry["kind"] == "real"
            else prediction_fingerprint(base_fingerprint, model_id, entry, protocol.threads)
        )
        for model_id, entry in registry.items()
    }
    units = sorted({*confirmation["families"], *confirmation["windows"]})
    payload = {
        "task": "TRAIN-001",
        "sources": dict(current_sources),
        "files": files,
        "file_sha256": {key: sha256_file(root / relative) for key, relative in files.items()},
        "learning_rate": pilot_report["selected_learning_rate"],
        "archive_weights": development["weights"],
        "registry": registry,
        "fingerprints": fingerprints,
        "base_fingerprint": base_fingerprint,
        "readouts": readout_digests,
        "readout_root": str(READOUT2),
        "expected_keys": sorted(record_key(u, m) for u in units for m in model_ids(protocol)),
        "decision_rules": DECISION_RULES,
        **(extra or {}),
    }
    seal(root / "seal.json", payload)
    return payload


def seal_stage(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    root = protocol.run_root
    if (root / "seal.json").exists():
        return sealed(root / "seal.json")
    _check_audit(protocol)
    _, digests = readout_decoders()
    payload = build_seal(
        root,
        protocol,
        sources(),
        frozen_fingerprint(),
        digests,
        {"git": _git(), "ledger_used_seconds": budget.used(), "sealed_at": time.time()},
    )
    require_valid(root, protocol, ENTRY, sources(), frozen_fingerprint(), base_weights_sha256())
    return {"sealed": True, "expected_records": len(payload["expected_keys"])}


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Sidecar first, then an atomic move, so a kill never leaves an unverifiable record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    path.with_name(path.name + ".sha256").write_text(sha256_file(temporary) + "\n")
    os.replace(temporary, path)


def confirm(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    from jepa_action_effects.train001_metrics import summarize
    from jepa_action_effects.train001_model import load_trained

    root = protocol.run_root
    seal_payload = _require(root / "seal.json", "Seal first")
    base = frozen_fingerprint()
    arguments = (root, protocol)
    require_valid(*arguments, ENTRY, sources(), base, base_weights_sha256())
    opened = root / "confirm" / "opened.json"
    if not opened.exists():
        write_json(
            opened, {"seal_sha256": sha256_file(root / "seal.json"), "opened_at": time.time()}
        )
    summary_path = root / "confirm" / "summary.json"
    if summary_path.exists():
        require_valid(*arguments, COMPLETE, sources(), base, base_weights_sha256())
        return sealed(summary_path)
    collection = sealed(root / "confirmation" / "collection.json")
    cache = sealed(root / "cache" / "confirmation.json")
    decoders, _ = readout_decoders(expected=seal_payload["readouts"])
    units = sorted(collection["families"]) + sorted(collection["windows"])
    caches = {c["id"]: c for c in load_cache(root, cache, units)}
    scaler = _scaler()
    factory = _model_factory()
    records_dir = root / "confirm" / "records"
    for model_id in model_ids(protocol):
        pending = [
            u for u in units if not (records_dir / f"{record_key(u, model_id)}.json").exists()
        ]
        if not pending:
            continue
        budget.unit("model_eval")
        entry = seal_payload["registry"][model_id]
        if entry["kind"] == "real":
            computed = [real_record(caches[u], decoders) for u in pending]
        else:
            model = factory()
            if entry["kind"] == "trained":
                model = load_trained(model, root / entry["checkpoint"], entry["checkpoint_sha256"])
            computed = score_model(model, [caches[u] for u in pending], decoders, scaler, protocol)
        for record in computed:
            record.update(
                {
                    "model": model_id,
                    "input_sha256": cache["records"][record["unit"]]["sha256"],
                    "fingerprint_sha256": seal_payload["fingerprints"][model_id],
                    "readout_sha256": seal_payload["readouts"],
                }
            )
            write_record(records_dir / f"{record_key(record['unit'], model_id)}.json", record)
        budget.heartbeat(force=True)
    require_valid(*arguments, COMPLETE, sources(), base, base_weights_sha256())

    def load_records(model_id: str, kind: str) -> list[dict]:
        chosen = (
            sorted(collection["families"]) if kind == "family" else sorted(collection["windows"])
        )
        return [
            json.loads((records_dir / f"{record_key(u, model_id)}.json").read_text())
            for u in chosen
        ]

    predictors = [m for m in model_ids(protocol) if m != "real-frames"]
    eligible = sealed(root / "pilot" / "eligibility.json")
    summary = summarize(
        load_records("real-frames", "family"),
        {m: load_records(m, "family") for m in predictors},
        load_records("real-frames", "window"),
        {m: load_records(m, "window") for m in predictors},
        protocol,
        sealed(root / "calibration.json")["scales"],
        eligible["status"] == "eligible",
        READOUTS,
        "confirmation",
    )
    summary["development_eligibility"] = eligible["status"]
    seal(summary_path, summary)
    return summary


def report(protocol: Protocol, budget: Budget) -> dict[str, Any]:
    from jepa_action_effects.train001_report import write_report

    root = protocol.run_root
    require_valid(root, protocol, COMPLETE, sources(), frozen_fingerprint(), base_weights_sha256())
    return write_report(protocol, root, budget)


# --------------------------------------------------------------------------- CLI
def run_stage(stage: str, protocol: Protocol) -> Any:
    functions = {
        "audit": audit,
        "collect-development": lambda p, b: collect(p, b, "development"),
        "cache": lambda p, b: build_cache(p, b, "development"),
        "pilot": pilot,
        "train": train,
        "collect-confirmation": collect_confirmation,
        "seal": seal_stage,
        "confirm": confirm,
        "report": report,
    }
    budget = Budget(
        Path(protocol.budget_root),
        limit_seconds=protocol.budget_seconds,
        shutdown_allowance=protocol.shutdown_allowance_seconds,
    )
    budget.start(stage, protocol.root)
    status = "failed"
    started = time.monotonic()
    try:
        result = functions[stage](protocol, budget)
        status = "completed"
        return result
    except BudgetStop:
        status = "budget_stop"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        budget.finish(status)
        timing = protocol.run_root / "runtime.json"
        data = json.loads(timing.read_text()) if timing.exists() else {"stages": {}}
        data["stages"].setdefault(stage, []).append(
            {"seconds": time.monotonic() - started, "status": status}
        )
        write_json(timing, data)


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--smoke", action="store_true", help="tiny implementation check")
    args = parser.parse_args(argv)
    protocol = Protocol.smoke() if args.smoke else Protocol.production()
    torch.set_num_threads(protocol.threads)
    try:
        result = run_stage(args.stage, protocol)
    except BudgetStop as error:
        print(json.dumps({"stage": args.stage, "status": "budget_stop", "detail": str(error)}))
        sys.exit(3)
    except BudgetLocked as error:
        print(json.dumps({"stage": args.stage, "status": "locked", "detail": str(error)}))
        sys.exit(4)
    except (GateFailure, DataIntegrityError) as error:
        print(json.dumps({"stage": args.stage, "status": "stopped", "detail": str(error)}))
        sys.exit(2)
    summary = result if isinstance(result, dict) else {"result": str(result)}
    print(json.dumps({"stage": args.stage, "status": "completed", "keys": sorted(summary)[:20]}))


if __name__ == "__main__":
    main()
