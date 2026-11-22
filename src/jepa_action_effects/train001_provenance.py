"""Strict TRAIN-001 provenance validation before and during confirmation (D028).

Two modes:

- `entry` (first confirm call and every resume): every sealed input, the complete
  model registry, readouts, caches, eligibility and learning-rate lineage must be
  valid. Existing confirmation records must be a valid, duplicate-free subset of the
  expected keys; missing records are pending work.
- `complete` (summary and report): additionally, records must cover the expected keys
  exactly.

Foreign, corrupt or stale records fail in both modes. A record is bound to its model
by the prediction fingerprint, which includes the actual checkpoint hash, condition
and training seed; a record computed by one checkpoint cannot pass as another's.
The sealed config is compared with the code-generated protocol, so an edited JSON
with a regenerated sidecar cannot redefine the protocol.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.provenance import ProvenanceError, _sidecar_ok
from jepa_action_effects.train001_collect import (
    DataIntegrityError,
    expected_counts,
    observed_counts,
    verify_integrity,
)
from jepa_action_effects.train001_protocol import Protocol

ENTRY, COMPLETE = "entry", "complete"
SEALED_FILES = (
    "config.json",
    "audit.json",
    "exclusions.json",
    "development/collection.json",
    "confirmation/collection.json",
    "cache/development.json",
    "cache/confirmation.json",
    "pilot/eligibility.json",
    "pilot/pilot.json",
    "calibration.json",
    "seal.json",
)


def model_ids(protocol: Protocol) -> list[str]:
    from jepa_action_effects.train001_protocol import CONDITIONS

    return [
        "real-frames",
        "frozen",
        *[f"{c}-{s}" for c in CONDITIONS for s in protocol.training_seeds],
    ]


def record_key(unit: str, model: str) -> str:
    return f"{unit}__{model}"


def validate(
    root: Path,
    protocol: Protocol,
    mode: str,
    current_sources: Mapping[str, str],
    current_fingerprint: Mapping[str, Any],
    base_weights_sha256: str,
) -> dict[str, Any]:
    """Return {"checks": [...], "valid": bool}; never modifies the run."""
    if mode not in (ENTRY, COMPLETE):
        raise ValueError(mode)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> bool:
        checks.append({"check": name, "status": "passed" if ok else "failed", "detail": detail})
        return bool(ok)

    def load(relative: str) -> dict | None:
        path = root / relative
        return json.loads(path.read_text()) if path.exists() else None

    # 1. sidecars
    for relative in SEALED_FILES:
        path = root / relative
        check(f"sealed:{relative}", path.exists() and _sidecar_ok(path))
    seal = load("seal.json")
    if seal is None:
        check("seal:present", False)
        return {"checks": checks, "valid": False}
    config, audit = load("config.json"), load("audit.json")

    # 2. protocol and sources
    check("config:matches_code", config == protocol.payload())
    check("sources:audit", audit is not None and audit.get("sources") == dict(current_sources))
    check("sources:seal", seal.get("sources") == dict(current_sources))

    # 3. sealed file hashes
    for key, relative in seal["files"].items():
        path = root / relative
        check(
            f"seal_hash:{key}",
            path.exists() and sha256_file(path) == seal["file_sha256"].get(key),
        )

    # 4. collections: bytes, identities, disjointness, counts
    exclusions = load("exclusions.json") or {"seeds": [], "episodes": [], "history_sha256": []}
    collections = {
        phase: load(f"{phase}/collection.json") for phase in ("development", "confirmation")
    }
    if all(collections.values()):
        try:
            verify_integrity(list(collections.values()), exclusions, root)
            check("collections:integrity", True)
        except DataIntegrityError as error:
            check("collections:integrity", False, str(error))
        for phase, collection in collections.items():
            check(
                f"collections:counts:{phase}",
                observed_counts(collection) == expected_counts(phase, protocol),
            )
    else:
        check("collections:present", False)

    # 5. caches: bytes, inputs, frozen-encoder fingerprint
    caches = {phase: load(f"cache/{phase}.json") for phase in ("development", "confirmation")}
    for phase, cache in caches.items():
        collection = collections.get(phase)
        if cache is None or collection is None:
            check(f"cache:{phase}:present", False)
            continue
        units = {**collection["families"], **collection["windows"]}
        check(f"cache:{phase}:coverage", set(cache["records"]) == set(units))
        bad = [
            unit
            for unit, entry in cache["records"].items()
            if unit not in units
            or entry["input_sha256"] != units[unit]["file_sha256"]
            or not (root / entry["path"]).exists()
            or sha256_file(root / entry["path"]) != entry["sha256"]
        ]
        check(f"cache:{phase}:bytes_and_inputs", not bad, bad[:5])
        check(f"cache:{phase}:fingerprint", cache["fingerprint"] == dict(current_fingerprint))

    # 6. lineage: pilot, eligibility, learning rate
    pilot, eligibility = load("pilot/pilot.json"), load("pilot/eligibility.json")
    check("pilot:passed", pilot is not None and pilot.get("status") == "passed")
    check(
        "pilot:learning_rate",
        pilot is not None and pilot.get("selected_learning_rate") == seal["learning_rate"],
    )
    check("eligibility:present", eligibility is not None and "status" in eligibility)

    # 7. model registry
    registry = seal["registry"]
    expected_models = model_ids(protocol)
    check("registry:models", sorted(registry) == sorted(expected_models), sorted(registry))
    for model_id in expected_models:
        entry = registry.get(model_id)
        if entry is None or entry["kind"] != "trained":
            continue
        run_path, checkpoint = root / entry["run"], root / entry["checkpoint"]
        run = json.loads(run_path.read_text()) if run_path.exists() else None
        ok = (
            run is not None
            and _sidecar_ok(run_path)
            and sha256_file(run_path) == entry["run_sha256"]
            and checkpoint.exists()
            and sha256_file(checkpoint) == entry["checkpoint_sha256"]
            and run["condition"] == entry["condition"]
            and run["seed"] == entry["seed"]
            and run["learning_rate"] == seal["learning_rate"]
            and run["updates"] == protocol.updates
            and run["base_weights_sha256"] == base_weights_sha256
            and run["eligibility_sha256"] == seal["file_sha256"]["eligibility"]
            and run["pilot_sha256"] == seal["file_sha256"]["pilot"]
            and run["checkpoints"][str(protocol.updates)] == entry["checkpoint_sha256"]
        )
        check(f"registry:{model_id}", ok)

    # 8. readouts
    for name, digest in seal["readouts"].items():
        path = Path(seal["readout_root"]) / "models" / f"{name}.npz"
        check(f"readout:{name}", path.exists() and sha256_file(path) == digest)

    # 9. confirmation records
    confirmation = collections.get("confirmation")
    cache = caches.get("confirmation")
    if confirmation is not None and cache is not None:
        units = sorted({*confirmation["families"], *confirmation["windows"]})
        expected = {record_key(unit, model) for unit in units for model in expected_models}
        check("records:expected_keys_sealed", set(seal["expected_keys"]) == expected)
        present, bad = _records(root, seal, cache)
        foreign = sorted(set(present) - expected)
        check("records:no_foreign", not foreign, foreign[:5])
        check("records:valid", not bad, bad[:5])
        if mode == COMPLETE:
            missing = sorted(expected - set(present))
            check("records:complete", not missing, {"missing": len(missing)})
        opened = root / "confirm" / "opened.json"
        if opened.exists():
            marker = json.loads(opened.read_text())
            check(
                "confirm:opened_marker",
                marker.get("seal_sha256") == sha256_file(root / "seal.json"),
            )
    return {"checks": checks, "valid": all(c["status"] == "passed" for c in checks)}


def _records(root: Path, seal: dict, cache: dict) -> tuple[list[str], list[str]]:
    directory = root / "confirm" / "records"
    present, bad = [], []
    if not directory.exists():
        return present, bad
    for path in sorted(directory.iterdir()):
        if path.name.endswith(".sha256") or path.name.startswith("."):
            continue
        if path.suffix != ".json":
            bad.append(f"stray:{path.name}")
            continue
        key = path.stem
        if not _sidecar_ok(path):
            bad.append(f"sidecar:{key}")
            present.append(key)
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            bad.append(f"corrupt:{key}")
            continue
        present.append(key)
        unit, model = record.get("unit"), record.get("model")
        if key != record_key(str(unit), str(model)):
            bad.append(f"key_mismatch:{key}")
            continue
        entry = cache["records"].get(unit)
        if entry is None or record.get("input_sha256") != entry["sha256"]:
            bad.append(f"input:{key}")
        if record.get("fingerprint_sha256") != seal["fingerprints"].get(model):
            bad.append(f"fingerprint:{key}")
        if record.get("readout_sha256") != seal["readouts"]:
            bad.append(f"readout:{key}")
    return present, bad


def require_valid(*args, **kwargs) -> dict[str, Any]:
    report = validate(*args, **kwargs)
    if not report["valid"]:
        failed = [c for c in report["checks"] if c["status"] != "passed"]
        raise ProvenanceError(f"TRAIN-001 provenance failed: {failed[:8]}")
    return report
