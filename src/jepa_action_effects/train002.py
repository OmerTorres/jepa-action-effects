"""TRAIN-002: support-manipulated fine-tuning of the LeWM predictor (D035).

    python -m jepa_action_effects.train002 audit
    python -m jepa_action_effects.train002 collect-windows    # archive-only training data
    python -m jepa_action_effects.train002 collect-confirmation
    python -m jepa_action_effects.train002 cache
    python -m jepa_action_effects.train002 train              # dose_omitted, archive_only
    python -m jepa_action_effects.train002 gate               # confirmation real frames only
    python -m jepa_action_effects.train002 seal
    python -m jepa_action_effects.train002 confirm
    python -m jepa_action_effects.train002 report

Reuses the sealed TRAIN-001 modules without editing them (their hashes are part of the
TRAIN-001 seal). The six TRAIN-001 runs are reused after their checkpoint hashes are
checked against that seal. Every stage is charged to `outputs/train002-budget`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import seal, sealed, write_json
from jepa_action_effects.next_common import MANIFESTS, runtime_info
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001 import (
    GateFailure,
    _git,
    _model_factory,
    _scaler,
    _spec,
    _units,
    base_weights_sha256,
    build_cache,
    frozen_fingerprint,
    load_cache,
    readout_decoders,
    real_record,
    score_model,
    write_record,
)
from jepa_action_effects.train001_budget import Budget, BudgetLocked, BudgetStop
from jepa_action_effects.train001_collect import (
    DataIntegrityError,
    canonical_sha256,
    collect_phase,
    collect_windows,
    exclusion_registry,
    verify_integrity,
)
from jepa_action_effects.train001_protocol import READOUTS, SCENES, Protocol

TRAIN001 = Path("outputs/train001")
OUTPUTS = Path("outputs")
MANIFEST = MANIFESTS / "train002-runtime.json"
DRAFT = Path("docs/TRAIN-002_RESULT_DRAFT.md")
LEARNING_RATE = 1e-5
TRAIN_WINDOWS = 2400
WINDOWS_PER_UPDATE = 64
NEW_ARMS = ("dose_omitted", "archive_only")
REUSED_ARMS = ("archive_weighted", "balanced")
BUDGET_HOURS = 6.0


def protocol() -> Protocol:
    """TRAIN-001's family process and gates with TRAIN-002 roots, seeds and sizes."""
    return Protocol(
        name="train002",
        schema="train002-v1",
        root="outputs/train002",
        budget_root="outputs/train002-budget",
        collection_seed=20261300,
        ordinary_validation=TRAIN_WINDOWS,  # the archive-only training windows
        budget_hours=BUDGET_HOURS,
        gate_bootstrap_seed=20261321,
        primary_bootstrap_seed=20261322,
        ordinary_bootstrap_seed=20261323,
        permutation_seed=20261324,
    )


def config_payload(p: Protocol) -> dict[str, Any]:
    from jepa_action_effects.train002_metrics import ARMS, DOSE_OMITTED_WEIGHTS

    return {
        "task": "TRAIN-002 (D035)",
        "base_protocol": p.payload(),
        "arms": list(ARMS),
        "new_arms": list(NEW_ARMS),
        "reused_arms": list(REUSED_ARMS),
        "dose_omitted_weights": list(DOSE_OMITTED_WEIGHTS),
        "learning_rate": LEARNING_RATE,
        "train_windows": TRAIN_WINDOWS,
        "windows_per_update": WINDOWS_PER_UPDATE,
        "held_out_magnitude": 0.60,
        "gate_cells": [f"{s}:m{m:.2f}" for s in SCENES for m in (0.45, 0.60)],
    }


# --------------------------------------------------------------------------- helpers
def _sources() -> dict[str, str]:
    here = Path(__file__).parent
    names = [
        "train002",
        "train002_metrics",
        "train001",
        "train001_collect",
        "train001_model",
        "train001_metrics",
        "train001_protocol",
        "train001_budget",
        "model_inference",
        "next_common",
    ]
    return {name: sha256_file(here / f"{name}.py") for name in names}


def _require(path: Path, message: str) -> dict:
    if not path.exists():
        raise GateFailure(message)
    return sealed(path)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _h5():
    import h5py

    from jepa_action_effects.next_common import ARCHIVE_H5

    return h5py.File(ARCHIVE_H5, "r")


def _train001_seal() -> dict[str, Any]:
    return sealed(TRAIN001 / "seal.json")


def reused_runs(t1_seal: dict[str, Any], seeds) -> dict[str, dict[str, Any]]:
    """TRAIN-001 trained models, each checkpoint verified against the TRAIN-001 seal."""
    reused = {}
    for arm in REUSED_ARMS:
        for seed in seeds:
            model_id = f"{arm}-{seed}"
            entry = t1_seal["registry"][model_id]
            path = TRAIN001 / entry["checkpoint"]
            _check(
                sha256_file(path) == entry["checkpoint_sha256"], f"{model_id} checkpoint changed"
            )
            reused[model_id] = {
                "kind": "trained",
                "arm": arm,
                "seed": seed,
                "source": "TRAIN-001",
                "checkpoint": str(path),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "run_sha256": entry["run_sha256"],
            }
    return reused


# --------------------------------------------------------------------------- stages
def audit(p: Protocol, budget: Budget) -> dict[str, Any]:
    root = p.run_root
    if (root / "audit.json").exists():
        return sealed(root / "audit.json")
    t1 = _train001_seal()
    files = t1["file_sha256"]
    for key in ("collection_development", "cache_development"):
        path = TRAIN001 / t1["files"][key]
        _check(sha256_file(path) == files[key], f"TRAIN-001 {key} changed")
    _check(t1["registry"]["frozen"]["checkpoint_sha256"] == base_weights_sha256(), "Base weights")
    reused = reused_runs(t1, p.training_seeds)
    root.mkdir(parents=True, exist_ok=True)
    seal(root / "config.json", config_payload(p))
    registry = exclusion_registry(OUTPUTS, root)
    seal(root / "exclusions.json", registry)
    payload = {
        "train001_seal_sha256": sha256_file(TRAIN001 / "seal.json"),
        "train001_development": {
            k: files[k] for k in ("collection_development", "cache_development")
        },
        "reused": reused,
        "base_weights_sha256": base_weights_sha256(),
        "excluded": {"seeds": len(registry["seeds"]), "episodes": len(registry["episodes"])},
        "sources": _sources(),
        "git": _git(),
    }
    seal(root / "audit.json", payload)
    return payload


def _check_audit(p: Protocol) -> dict[str, Any]:
    audit_payload = _require(p.run_root / "audit.json", "Run audit first")
    _check(sealed(p.run_root / "config.json") == config_payload(p), "Protocol changed since audit")
    _check(
        sha256_file(TRAIN001 / "seal.json") == audit_payload["train001_seal_sha256"],
        "TRAIN-001 seal changed since audit",
    )
    return audit_payload


def collect_training_windows(p: Protocol, budget: Budget) -> dict[str, Any]:
    root = p.run_root
    path = root / "train_windows" / "collection.json"
    if path.exists():
        return sealed(path)
    _check_audit(p)
    registry = sealed(root / "exclusions.json")
    directory = root / "train_windows" / "windows"
    if directory.exists():
        for stale in directory.glob("*.npz"):
            stale.unlink()
    log: dict[str, int] = {}
    with _h5() as h5:
        available = h5["ep_len"][:] >= 21
        available[[e for e in registry["episodes"] if e < len(available)]] = False
        windows = collect_windows(directory, "development", p, h5, available, log, budget)
    for record in windows:
        record["relative_path"] = str(Path(record.pop("path")).relative_to(root))
        record["partition"] = "train"
        record["scene"] = "ordinary"
    collection = {
        "phase": "train_windows",
        "families": {},
        "windows": {record["id"]: record for record in windows},
        "rejections": dict(sorted(log.items())),
    }
    _check(len(collection["windows"]) == TRAIN_WINDOWS, "Window count differs from protocol")
    collection["integrity"] = verify_integrity([collection], registry, root)
    seal(path, collection)
    return collection


def collect_confirmation(p: Protocol, budget: Budget) -> dict[str, Any]:
    root = p.run_root
    path = root / "confirmation" / "collection.json"
    if path.exists():
        return sealed(path)
    _check_audit(p)
    registry = sealed(root / "exclusions.json")
    prior = [_require(root / "train_windows" / "collection.json", "Collect windows first")]
    started = time.monotonic()
    with _h5() as h5:
        collection = collect_phase("confirmation", p, root, registry, h5, prior, budget)
    collection["seconds"] = time.monotonic() - started
    seal(path, collection)
    return collection


def cache(p: Protocol, budget: Budget) -> dict[str, Any]:
    _check_audit(p)
    return {
        phase: build_cache(p, budget, phase)["phase"] for phase in ("train_windows", "confirmation")
    }


def window_tensors(records: list[dict[str, Any]], scaler):
    """Single-branch 'families' from cached archive windows (all in the archive scene)."""
    import torch

    from jepa_action_effects.train001_model import FamilyTensors, normalized_actions

    z = np.stack([record["z"] for record in records]).astype(np.float32)
    controls = np.stack([record["controls"] for record in records])
    return FamilyTensors(
        [record["id"] for record in records],
        np.full(len(records), SCENES.index("archive")),
        torch.from_numpy(z[:, :3]),
        torch.from_numpy(z[:, None, 3:5]),
        normalized_actions(scaler, controls[:, :10], controls[:, None, 10:]).float(),
    )


def optimization_gate(curve: list[list[float]], window: int = 100) -> dict[str, Any]:
    """Mean loss over the last `window` updates must be below that over the first."""
    points = np.asarray(curve, float)
    first = points[points[:, 0] <= window, 1]
    last = points[points[:, 0] > points[-1, 0] - window, 1]
    ok = len(first) > 0 and len(last) > 0 and float(last.mean()) < float(first.mean())
    return {"first_mean": float(first.mean()), "last_mean": float(last.mean()), "passed": ok}


def _data(p: Protocol, arm: str, scaler):
    from jepa_action_effects.train001_model import family_tensors

    t1_collection = sealed(TRAIN001 / "development" / "collection.json")
    t1_cache = sealed(TRAIN001 / "cache" / "development.json")
    validation = family_tensors(
        load_cache(TRAIN001, t1_cache, _units(t1_collection, "validation")), scaler
    )
    if arm == "dose_omitted":
        data = family_tensors(
            load_cache(TRAIN001, t1_cache, _units(t1_collection, "train")), scaler
        )
        lineage = {
            "collection_sha256": sha256_file(TRAIN001 / "development" / "collection.json"),
            "cache_sha256": sha256_file(TRAIN001 / "cache" / "development.json"),
        }
        return data, validation, lineage
    root = p.run_root
    collection = sealed(root / "train_windows" / "collection.json")
    manifest = sealed(root / "cache" / "train_windows.json")
    data = window_tensors(load_cache(root, manifest, sorted(collection["windows"])), scaler)
    lineage = {
        "collection_sha256": sha256_file(root / "train_windows" / "collection.json"),
        "cache_sha256": sha256_file(root / "cache" / "train_windows.json"),
    }
    return data, validation, lineage


def train(p: Protocol, budget: Budget) -> dict[str, Any]:
    import torch

    from jepa_action_effects.train001_model import state_digest, train_run, trainable_state
    from jepa_action_effects.train002_metrics import DOSE_OMITTED_WEIGHTS

    root = p.run_root
    _check_audit(p)
    torch.set_num_threads(p.threads)
    scaler = _scaler()
    factory = _model_factory()
    digest = state_digest(trainable_state(factory()))
    runs = {}
    for arm in NEW_ARMS:
        data, validation, lineage = _data(p, arm, scaler)
        weights = DOSE_OMITTED_WEIGHTS if arm == "dose_omitted" else (1.0,)
        for seed in p.training_seeds:
            run_dir = root / "runs" / f"{arm}-{seed}"
            record_path = run_dir / "run.json"
            if record_path.exists():
                runs[f"{arm}-{seed}"] = sealed(record_path)
                continue
            spec = _spec(p, arm, seed, LEARNING_RATE, p.updates, weights)
            if arm == "archive_only":
                spec.per_scene = WINDOWS_PER_UPDATE
            spec.checkpoint_updates = p.checkpoint_updates
            spec.resume_every = p.resume_every
            spec.curve_every = p.curve_every
            spec.validation_every = p.validation_every
            began = time.monotonic()
            result = train_run(factory, digest, data, spec, run_dir, validation, budget)
            gate = optimization_gate(result["curve"])
            final = run_dir / f"checkpoint-{p.updates:05d}.pt"
            payload = {
                "arm": arm,
                "seed": seed,
                "spec": spec.__dict__,
                "weights": list(weights),
                "training_units": len(data.ids),
                **lineage,
                "base_weights_sha256": base_weights_sha256(),
                "base_trainable_digest": digest,
                "checkpoints": result["checkpoints"],
                "final_checkpoint": str(final.relative_to(root)),
                "final_checkpoint_sha256": sha256_file(final),
                "curve": result["curve"],
                "validation_curve": result["validation_curve"],
                "optimization_gate": gate,
                "seconds_this_session": time.monotonic() - began,
                "runtime": runtime_info() | {"threads": torch.get_num_threads()},
            }
            seal(record_path, payload)
            _check(gate["passed"], f"{arm}-{seed} failed the optimization gate")
            runs[f"{arm}-{seed}"] = payload
    return {"runs": sorted(runs)}


def _confirmation_caches(p: Protocol) -> tuple[dict, dict, dict]:
    root = p.run_root
    collection = sealed(root / "confirmation" / "collection.json")
    manifest = sealed(root / "cache" / "confirmation.json")
    units = sorted(collection["families"]) + sorted(collection["windows"])
    caches = {c["id"]: c for c in load_cache(root, manifest, units)}
    return collection, manifest, caches


def gate(p: Protocol, budget: Budget) -> dict[str, Any]:
    """Measurement gates from confirmation real frames; no predictor is run."""
    from jepa_action_effects.train001_metrics import family_table, measurement_gates

    root = p.run_root
    path = root / "gate.json"
    if path.exists():
        return sealed(path)
    _check_audit(p)
    _check(not (root / "seal.json").exists(), "Gate must precede the seal")
    collection, manifest, caches = _confirmation_caches(p)
    decoders, digests = readout_decoders()
    families = [real_record(caches[u], decoders) for u in sorted(collection["families"])]
    gates = {
        readout: measurement_gates(family_table(families, readout), p, p.gate_bootstrap_seed)
        for readout in READOUTS
    }
    payload = {
        "label": "confirmation real frames only; computed before any predictor scoring",
        "primary": gates[READOUTS[0]],
        "per_readout": gates,
        "readouts": digests,
        "cache_sha256": sha256_file(root / "cache" / "confirmation.json"),
    }
    seal(path, payload)
    return {"passed": payload["primary"]["passed"], "failing": payload["primary"]["failing_cells"]}


SEALED_FILES = {
    "config": "config.json",
    "audit": "audit.json",
    "exclusions": "exclusions.json",
    "train_windows": "train_windows/collection.json",
    "confirmation": "confirmation/collection.json",
    "cache_train_windows": "cache/train_windows.json",
    "cache_confirmation": "cache/confirmation.json",
    "gate": "gate.json",
}


def model_registry(p: Protocol, audit_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = p.run_root
    registry: dict[str, dict[str, Any]] = {
        "real-frames": {"kind": "real"},
        "frozen": {"kind": "frozen", "checkpoint_sha256": base_weights_sha256()},
        **reused_runs(_train001_seal(), p.training_seeds),
    }
    for arm in NEW_ARMS:
        for seed in p.training_seeds:
            run_path = root / "runs" / f"{arm}-{seed}" / "run.json"
            run = _require(run_path, f"Train {arm}-{seed} first")
            _check(run["optimization_gate"]["passed"], f"{arm}-{seed} optimization gate")
            checkpoint = root / run["final_checkpoint"]
            _check(sha256_file(checkpoint) == run["final_checkpoint_sha256"], "Checkpoint changed")
            registry[f"{arm}-{seed}"] = {
                "kind": "trained",
                "arm": arm,
                "seed": seed,
                "source": "TRAIN-002",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run["final_checkpoint_sha256"],
                "run_sha256": sha256_file(run_path),
            }
    for model_id, entry in audit_payload["reused"].items():
        _check(registry[model_id]["checkpoint_sha256"] == entry["checkpoint_sha256"], model_id)
    return registry


def seal_stage(p: Protocol, budget: Budget) -> dict[str, Any]:
    root = p.run_root
    if (root / "seal.json").exists():
        return sealed(root / "seal.json")
    audit_payload = _check_audit(p)
    for relative in SEALED_FILES.values():
        _require(root / relative, f"Missing {relative}")
    registry = model_registry(p, audit_payload)
    _, digests = readout_decoders()
    gate_payload = sealed(root / "gate.json")
    _check(gate_payload["readouts"] == digests, "Readouts changed since the gate")
    base = frozen_fingerprint()
    payload = {
        "task": "TRAIN-002 (D035)",
        "sources": _sources(),
        "file_sha256": {k: sha256_file(root / v) for k, v in SEALED_FILES.items()},
        "registry": registry,
        "fingerprints": {
            m: canonical_sha256(
                {**base, "model_id": m, **{k: e.get(k) for k in ("checkpoint_sha256",)}}
            )
            for m, e in registry.items()
        },
        "base_fingerprint": base,
        "readouts": digests,
        "gate_primary_passed": gate_payload["primary"]["passed"],
        "git": _git(),
        "ledger_used_seconds": budget.used(),
        "sealed_at": time.time(),
    }
    seal(root / "seal.json", payload)
    return {"sealed": True, "models": len(registry)}


def verify_seal(p: Protocol) -> dict[str, Any]:
    """Refuse to score or report if any sealed input, checkpoint or source changed."""
    root = p.run_root
    payload = _require(root / "seal.json", "Seal first")
    _check(payload["sources"] == _sources(), "Source changed after the seal")
    for key, relative in SEALED_FILES.items():
        _check(sha256_file(root / relative) == payload["file_sha256"][key], f"{relative} changed")
    for model_id, entry in payload["registry"].items():
        if entry["kind"] == "trained":
            _check(sha256_file(Path(entry["checkpoint"])) == entry["checkpoint_sha256"], model_id)
    return payload


def _model_ids(registry: dict[str, Any]) -> list[str]:
    return [
        "real-frames",
        "frozen",
        *sorted(k for k, v in registry.items() if v["kind"] == "trained"),
    ]


def confirm(p: Protocol, budget: Budget) -> dict[str, Any]:
    from jepa_action_effects.train001_model import load_trained
    from jepa_action_effects.train002_metrics import summarize

    root = p.run_root
    payload = verify_seal(p)
    summary_path = root / "confirm" / "summary.json"
    if summary_path.exists():
        return sealed(summary_path)
    opened = root / "confirm" / "opened.json"
    if not opened.exists():
        write_json(
            opened, {"seal_sha256": sha256_file(root / "seal.json"), "opened_at": time.time()}
        )
    collection, manifest, caches = _confirmation_caches(p)
    decoders, _ = readout_decoders(expected=payload["readouts"])
    units = sorted(collection["families"]) + sorted(collection["windows"])
    scaler, factory = _scaler(), _model_factory()
    records_dir = root / "confirm" / "records"
    for model_id in _model_ids(payload["registry"]):
        pending = [u for u in units if not (records_dir / f"{u}__{model_id}.json").exists()]
        if not pending:
            continue
        budget.unit("model_eval")
        entry = payload["registry"][model_id]
        if entry["kind"] == "real":
            computed = [real_record(caches[u], decoders) for u in pending]
        else:
            model = factory()
            if entry["kind"] == "trained":
                model = load_trained(model, Path(entry["checkpoint"]), entry["checkpoint_sha256"])
            computed = score_model(model, [caches[u] for u in pending], decoders, scaler, p)
        for record in computed:
            record.update(
                {
                    "model": model_id,
                    "input_sha256": manifest["records"][record["unit"]]["sha256"],
                    "fingerprint_sha256": payload["fingerprints"][model_id],
                    "readout_sha256": payload["readouts"],
                }
            )
            write_record(records_dir / f"{record['unit']}__{model_id}.json", record)
        budget.heartbeat(force=True)

    def load(model_id: str, kind: str) -> list[dict]:
        chosen = (
            sorted(collection["families"]) if kind == "family" else sorted(collection["windows"])
        )
        out = []
        for unit in chosen:
            path = records_dir / f"{unit}__{model_id}.json"
            digest = path.with_name(path.name + ".sha256").read_text().strip()
            _check(sha256_file(path) == digest, f"Record changed: {path.name}")
            record = json.loads(path.read_text())
            _check(record["fingerprint_sha256"] == payload["fingerprints"][model_id], path.name)
            _check(record["input_sha256"] == manifest["records"][unit]["sha256"], path.name)
            out.append(record)
        return out

    predictors = [m for m in _model_ids(payload["registry"]) if m != "real-frames"]
    summary = summarize(
        load("real-frames", "family"),
        {m: load(m, "family") for m in predictors},
        load("real-frames", "window"),
        {m: load(m, "window") for m in predictors},
        READOUTS,
        sealed(root / "gate.json")["primary"],
        p.bootstrap_draws,
        p.primary_bootstrap_seed,
        p.ordinary_bootstrap_seed,
        p.delta,
        p.regression_tolerance,
    )
    seal(summary_path, summary)
    return {"primary": summary["primary_verdict"], "generic": summary["generic_verdict"]}


def report(p: Protocol, budget: Budget) -> dict[str, Any]:
    from jepa_action_effects.train002_report import draft

    root = p.run_root
    payload = verify_seal(p)
    summary = _require(root / "confirm" / "summary.json", "Confirm first")
    gate_payload = sealed(root / "gate.json")
    runs = {
        f"{arm}-{seed}": sealed(root / "runs" / f"{arm}-{seed}" / "run.json")
        for arm in NEW_ARMS
        for seed in p.training_seeds
    }
    manifest = {
        "task": "TRAIN-002 (D035)",
        "status": "complete",
        "seal_sha256": sha256_file(root / "seal.json"),
        "summary_sha256": sha256_file(root / "confirm" / "summary.json"),
        "gate": gate_payload["primary"],
        "summary": summary,
        "runs": {
            k: {
                "optimization_gate": v["optimization_gate"],
                "seconds": v["seconds_this_session"],
                "training_units": v["training_units"],
                "final_checkpoint_sha256": v["final_checkpoint_sha256"],
            }
            for k, v in runs.items()
        },
        "registry": {
            k: {kk: vv for kk, vv in v.items() if kk != "checkpoint"}
            for k, v in payload["registry"].items()
        },
        "sources": payload["sources"],
        "ledger_used_seconds": budget.used(),
        "runtime": runtime_info(),
    }
    DRAFT.write_text(draft(manifest))
    write_json(MANIFEST, manifest)
    return {"primary": summary["primary_verdict"], "generic": summary["generic_verdict"]}


STAGES = {
    "audit": audit,
    "collect-windows": collect_training_windows,
    "collect-confirmation": collect_confirmation,
    "cache": cache,
    "train": train,
    "gate": gate,
    "seal": seal_stage,
    "confirm": confirm,
    "report": report,
}


def run_stage(stage: str, p: Protocol) -> Any:
    budget = Budget(
        Path(p.budget_root),
        limit_seconds=p.budget_seconds,
        shutdown_allowance=p.shutdown_allowance_seconds,
    )
    budget.start(stage, p.root)
    status, started = "failed", time.monotonic()
    try:
        result = STAGES[stage](p, budget)
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
        timing = p.run_root / "runtime.json"
        p.run_root.mkdir(parents=True, exist_ok=True)
        data = json.loads(timing.read_text()) if timing.exists() else {"stages": {}}
        data["stages"].setdefault(stage, []).append(
            {"seconds": time.monotonic() - started, "status": status}
        )
        write_json(timing, data)


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args(argv)
    p = protocol()
    torch.set_num_threads(p.threads)
    for stage in STAGES if args.stage == "all" else [args.stage]:
        try:
            result = run_stage(stage, p)
        except BudgetStop as error:
            print(json.dumps({"stage": stage, "status": "budget_stop", "detail": str(error)}))
            sys.exit(3)
        except BudgetLocked as error:
            print(json.dumps({"stage": stage, "status": "locked", "detail": str(error)}))
            sys.exit(4)
        except (GateFailure, DataIntegrityError) as error:
            print(json.dumps({"stage": stage, "status": "stopped", "detail": str(error)}))
            sys.exit(2)
        print(
            json.dumps({"stage": stage, "status": "completed", "result": result}, default=str)[
                :2000
            ]
        )


if __name__ == "__main__":
    main()
