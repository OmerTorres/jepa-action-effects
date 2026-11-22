"""TRAIN-001 strict entry/resume and completion validation, with mutation tests."""

from __future__ import annotations

import json

import pytest

import jepa_action_effects.train001 as train001
from jepa_action_effects.interface_audit import WEIGHTS_SHA256
from jepa_action_effects.latent_audit import seal
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001 import build_seal, write_record
from jepa_action_effects.train001_protocol import CONDITIONS, READOUTS, Protocol
from jepa_action_effects.train001_provenance import COMPLETE, ENTRY, model_ids, record_key, validate

SOURCES = {"train001.py": "abc"}
FINGERPRINT = {"weights_sha256": "w", "device": "cpu"}


def _protocol(root):
    return Protocol(
        root=str(root),
        training_seeds=(1,),
        train_per_scene=1,
        validation_per_scene=1,
        confirmation_per_scene=1,
        ordinary_validation=1,
        ordinary_confirmation=1,
        updates=4,
    )


def _unit(root, phase, identifier, scene, partition, index):
    path = root / phase / "families" / f"{identifier}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"family {identifier}".encode())
    record = {
        "id": identifier,
        "scene": scene,
        "partition": partition,
        "relative_path": str(path.relative_to(root)),
        "file_sha256": sha256_file(path),
        "recipe_sha256": f"recipe-{identifier}",
        "history_sha256": f"history-{identifier}",
        "seed": 1000 + index if scene == "challenge" else (0 if scene == "archive" else None),
        "episode": None if scene == "challenge" else 500 + index,
    }
    return record


def _collection(root, phase, specs):
    families, windows = {}, {}
    for index, (identifier, scene, partition) in enumerate(specs):
        record = _unit(
            root, phase, identifier, scene, partition, index + (0 if phase == "development" else 50)
        )
        (windows if scene == "ordinary" else families)[identifier] = record
    payload = {"phase": phase, "families": families, "windows": windows}
    if phase == "development":
        payload["weights"] = {"episodes": [900], "weights": [0.4, 0.3, 0.2, 0.1]}
    seal(root / phase / "collection.json", payload)
    cache = {}
    for identifier, record in {**families, **windows}.items():
        path = root / "cache" / phase / f"{identifier}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"cache {identifier}".encode())
        cache[identifier] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "input_sha256": record["file_sha256"],
            "kind": "window" if record["scene"] == "ordinary" else "family",
            "scene": record["scene"],
            "partition": record["partition"],
        }
    seal(root / "cache" / f"{phase}.json", {"fingerprint": FINGERPRINT, "records": cache})


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    readouts = tmp_path / "readout002"
    (readouts / "models").mkdir(parents=True)
    digests = {}
    for name in READOUTS:
        path = readouts / "models" / f"{name}.npz"
        path.write_bytes(name.encode())
        digests[name] = sha256_file(path)
    monkeypatch.setattr(train001, "READOUT2", readouts)
    protocol = _protocol(root)
    seal(root / "config.json", protocol.payload())
    seal(root / "audit.json", {"sources": SOURCES})
    seal(root / "exclusions.json", {"seeds": [], "episodes": [], "history_sha256": []})
    _collection(
        root,
        "development",
        [
            ("challenge-dev-000", "challenge", "train"),
            ("challenge-dev-001", "challenge", "validation"),
            ("archive-dev-000", "archive", "train"),
            ("archive-dev-001", "archive", "validation"),
            ("ordinary-dev-000", "ordinary", "validation"),
        ],
    )
    _collection(
        root,
        "confirmation",
        [
            ("challenge-conf-000", "challenge", "confirmation"),
            ("archive-conf-000", "archive", "confirmation"),
            ("ordinary-conf-000", "ordinary", "confirmation"),
        ],
    )
    seal(root / "pilot" / "eligibility.json", {"status": "eligible"})
    seal(root / "pilot" / "pilot.json", {"status": "passed", "selected_learning_rate": 1e-4})
    seal(root / "calibration.json", {"scales": dict.fromkeys(READOUTS, 1.1)})
    for condition in CONDITIONS:
        run_dir = root / "runs" / f"{condition}-1"
        checkpoint = run_dir / "checkpoint-00004.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"weights {condition}".encode())
        seal(
            run_dir / "run.json",
            {
                "condition": condition,
                "seed": 1,
                "learning_rate": 1e-4,
                "updates": 4,
                "base_weights_sha256": WEIGHTS_SHA256,
                "eligibility_sha256": sha256_file(root / "pilot" / "eligibility.json"),
                "pilot_sha256": sha256_file(root / "pilot" / "pilot.json"),
                "checkpoints": {"4": sha256_file(checkpoint)},
                "final_checkpoint": str(checkpoint.relative_to(root)),
            },
        )
    payload = build_seal(root, protocol, SOURCES, FINGERPRINT, digests)
    return root, protocol, payload


def _check(root, protocol, mode=ENTRY, sources=SOURCES):
    return validate(root, protocol, mode, sources, FINGERPRINT, WEIGHTS_SHA256)


def _failed(report):
    return {c["check"] for c in report["checks"] if c["status"] != "passed"}


def _write(root, seal_payload, unit, model, **changes):
    cache = json.loads((root / "cache" / "confirmation.json").read_text())
    record = {
        "unit": unit,
        "model": model,
        "input_sha256": cache["records"][unit]["sha256"],
        "fingerprint_sha256": seal_payload["fingerprints"][model],
        "readout_sha256": seal_payload["readouts"],
        **changes,
    }
    path = root / "confirm" / "records" / f"{record_key(unit, model)}.json"
    write_record(path, record)
    return path


def _write_all(root, protocol, seal_payload):
    for unit in ("challenge-conf-000", "archive-conf-000", "ordinary-conf-000"):
        for model in model_ids(protocol):
            _write(root, seal_payload, unit, model)


def test_entry_accepts_an_empty_confirmation(run):
    root, protocol, _ = run
    report = _check(root, protocol)
    assert report["valid"], _failed(report)


def test_partial_records_are_valid_for_entry_but_not_completion(run):
    root, protocol, payload = run
    _write(root, payload, "challenge-conf-000", "frozen")
    assert _check(root, protocol)["valid"]
    assert _failed(_check(root, protocol, COMPLETE)) == {"records:complete"}
    _write_all(root, protocol, payload)
    assert _check(root, protocol, COMPLETE)["valid"]


def _tamper(path):
    path.write_bytes(path.read_bytes() + b"x")


MUTATIONS = {
    "checkpoint": lambda root, p: _tamper(root / "runs" / "balanced-1" / "checkpoint-00004.pt"),
    "readout": lambda root, p: _tamper(
        root.parent / "readout002" / "models" / f"{READOUTS[1]}.npz"
    ),
    "cache": lambda root, p: _tamper(root / "cache" / "confirmation" / "archive-conf-000.npz"),
    "family_bytes": lambda root, p: _tamper(
        root / "confirmation" / "families" / "challenge-conf-000.npz"
    ),
    "record_content": lambda root, p: _tamper(_write(root, p, "archive-conf-000", "frozen")),
    "foreign_record": lambda root, p: _write(root, p, "challenge-conf-000", "frozen").rename(
        root / "confirm" / "records" / f"{record_key('challenge-conf-999', 'frozen')}.json"
    ),
    "swapped_model": lambda root, p: _write(
        root,
        p,
        "challenge-conf-000",
        "archive_weighted-1",
        fingerprint_sha256=p["fingerprints"]["balanced-1"],
    ),
    "stale_input": lambda root, p: _write(
        root,
        p,
        "challenge-conf-000",
        "frozen",
        input_sha256=json.loads((root / "cache" / "confirmation.json").read_text())["records"][
            "archive-conf-000"
        ]["sha256"],
    ),
    "corrupt_record": lambda root, p: write_record(
        root / "confirm" / "records" / f"{record_key('challenge-conf-000', 'frozen')}.json", {}
    ),
    "config_resealed": lambda root, p: seal(
        root / "config.json", {**json.loads((root / "config.json").read_text()), "floor_limit": 0.5}
    ),
    "run_resealed": lambda root, p: seal(
        root / "runs" / "balanced-1" / "run.json",
        {
            **json.loads((root / "runs" / "balanced-1" / "run.json").read_text()),
            "learning_rate": 1e-3,
        },
    ),
    "eligibility_resealed": lambda root, p: seal(
        root / "pilot" / "eligibility.json", {"status": "not_assessable_under_measurement_protocol"}
    ),
    "stray_file": lambda root, p: (
        (root / "confirm" / "records").mkdir(parents=True, exist_ok=True)
        or (root / "confirm" / "records" / "notes.txt").write_text("x")
    ),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutations_are_rejected(run, mutation):
    root, protocol, payload = run
    assert _check(root, protocol)["valid"]
    MUTATIONS[mutation](root, payload)
    assert not _check(root, protocol)["valid"], mutation


def test_changed_source_is_rejected(run):
    root, protocol, _ = run
    report = _check(root, protocol, sources={"train001.py": "changed"})
    assert {"sources:audit", "sources:seal"} <= _failed(report)


def test_opened_marker_must_match_the_seal(run):
    root, protocol, _ = run
    (root / "confirm").mkdir(exist_ok=True)
    (root / "confirm" / "opened.json").write_text(json.dumps({"seal_sha256": "other"}))
    assert "confirm:opened_marker" in _failed(_check(root, protocol))


def test_prediction_fingerprints_differ_by_checkpoint(run):
    _, _, payload = run
    values = [payload["fingerprints"][m] for m in payload["registry"]]
    assert len(set(values)) == len(values)
