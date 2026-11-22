"""Mutation tests for the sealed-run provenance validator and corrected protocol gates.

Each case reproduces a counterexample from BRANCH_REVIEW_2026-09-29.md on a mock run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jepa_action_effects.latent_audit import seal
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.protocol_gates import faithful, readout_gate_per_cell
from jepa_action_effects.provenance import (
    ProvenanceError,
    require_valid,
    reserved_manifest_index,
    validate_run,
)

SOURCES = {"runner": "aaa", "metrics": "bbb"}
FINGERPRINT = {"weights": "w1", "scaler": "s1"}


def _mock_run(root: Path) -> None:
    seal(root / "config.json", {"schema": "mock-v1", "goal": [100, 100]})
    seal(
        root / "audit.json",
        {"sources": SOURCES, "config_sha256": sha256_file(root / "config.json")},
    )
    seal(root / "develop.json", {"fingerprint": FINGERPRINT})
    probe = root / "probe" / "models" / "p.npz"
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"probe-weights")
    seal(
        root / "probe" / "selection.json",
        {"path": "models/p.npz", "path_sha256": sha256_file(probe)},
    )
    family = root / "reserved" / "families" / "a.npz"
    family.parent.mkdir(parents=True)
    family.write_bytes(b"family-a")
    manifest = root / "reserved" / "collection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "family_files": {"a": {"path": "families/a.npz", "sha256": sha256_file(family)}},
                "goals": {"a": [[100, 100]]},
            }
        )
    )
    seal(
        root / "seal.json",
        {
            "sources": SOURCES,
            "config_sha256": sha256_file(root / "config.json"),
            "audit_sha256": sha256_file(root / "audit.json"),
            "develop_sha256": sha256_file(root / "develop.json"),
            "reserved_manifest_sha256": sha256_file(manifest),
            "readout_selection_sha256": sha256_file(root / "probe" / "selection.json"),
        },
    )
    seal(root / "confirm.json", {"fingerprint": FINGERPRINT})
    seal(
        root / "confirm" / "reserved__a.json",
        {"id": "reserved:a", "input_sha256": sha256_file(family), "fingerprint": FINGERPRINT},
    )


def _validate(root: Path, sources=SOURCES, fingerprint=FINGERPRINT, mode="strict"):
    return validate_run(
        root,
        sources,
        reserved_manifest_index(root),
        [root / "confirm"],
        selection_path=root / "probe" / "selection.json",
        current_fingerprint=fingerprint,
        mode=mode,
    )


def _failed(report) -> set[str]:
    return {c["check"] for c in report["checks"] if c["status"] == "failed"}


def test_consistent_run_is_valid(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    report = _validate(tmp_path)
    assert report["valid"], _failed(report)
    require_valid(tmp_path, SOURCES, reserved_manifest_index(tmp_path), [tmp_path / "confirm"])


def test_config_changed_and_resealed_after_freeze_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    seal(tmp_path / "config.json", {"schema": "mock-v1", "goal": [400, 400]})  # new sidecar too
    failed = _failed(_validate(tmp_path))
    assert {"seal:config_sha256", "audit:config_sha256"} <= failed


def test_reserved_goal_changed_after_freeze_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    manifest = tmp_path / "reserved" / "collection-manifest.json"
    payload = json.loads(manifest.read_text())
    payload["goals"]["a"] = [[400, 400]]
    manifest.write_text(json.dumps(payload))
    assert "seal:reserved_manifest_sha256" in _failed(_validate(tmp_path))


def test_stale_cache_for_missing_family_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    seal(tmp_path / "confirm" / "reserved__b.json", {"id": "reserved:b", "input_sha256": "x"})
    assert "records:known_family" in _failed(_validate(tmp_path))
    with pytest.raises(ProvenanceError, match="records:known_family"):
        require_valid(tmp_path, SOURCES, reserved_manifest_index(tmp_path), [tmp_path / "confirm"])


def test_changed_family_bytes_invalidate_cached_record(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    (tmp_path / "reserved" / "families" / "a.npz").write_bytes(b"tampered")
    failed = _failed(_validate(tmp_path))
    assert "families:present" in failed or "records:known_family" in failed


def test_source_change_between_develop_and_seal_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    changed = {**SOURCES, "metrics": "ccc"}
    seal(
        tmp_path / "audit.json",
        {"sources": changed, "config_sha256": sha256_file(tmp_path / "config.json")},
    )
    assert "sources:audit_equals_seal" in _failed(_validate(tmp_path))
    assert "sources:current_equals_seal" in _failed(_validate(tmp_path, sources=changed))


def test_scaler_change_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    report = _validate(tmp_path, fingerprint={**FINGERPRINT, "scaler": "s2"})
    assert "fingerprint:current_equals_develop" in _failed(report)


def test_missing_record_metadata_fails_strict_and_is_reported_in_legacy(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    seal(tmp_path / "confirm" / "reserved__a.json", {"id": "reserved:a"})
    strict = _failed(_validate(tmp_path))
    assert {"records:input_sha256_recorded", "records:fingerprint_recorded"} <= strict
    legacy = _validate(tmp_path, mode="legacy")
    assert legacy["valid"]
    assert sum(c["status"] == "not_recorded" for c in legacy["checks"]) == 2
    with pytest.raises(ValueError):
        _validate(tmp_path, mode="lenient")


def test_modified_probe_bytes_are_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    (tmp_path / "probe" / "models" / "p.npz").write_bytes(b"other-weights")
    assert "probe:artifact_sha256" in _failed(_validate(tmp_path))


def test_cached_record_with_foreign_fingerprint_is_rejected(tmp_path: Path) -> None:
    _mock_run(tmp_path)
    family = tmp_path / "reserved" / "families" / "a.npz"
    seal(
        tmp_path / "confirm" / "reserved__a.json",
        {
            "id": "reserved:a",
            "input_sha256": sha256_file(family),
            "fingerprint": {**FINGERPRINT, "scaler": "s2"},
        },
    )
    assert "records:fingerprint" in _failed(_validate(tmp_path))


def test_per_cell_readout_gate_catches_opposite_dose_biases() -> None:
    pooled = {"challenge_constant": [0.93, 0.99]}  # 0.8 and 1.1 gains averaged
    per_dose = {"challenge_m0.30_constant": [0.78, 0.82], "challenge_m0.60_constant": [1.08, 1.12]}
    assert readout_gate_per_cell(pooled)["passed"]
    result = readout_gate_per_cell(per_dose)
    assert not result["passed"] and set(result["failing_cells"]) == {"challenge_m0.30_constant"}
    assert not readout_gate_per_cell({})["passed"]


def test_two_sided_faithfulness() -> None:
    assert not faithful([2.0, 3.0])
    assert not faithful([0.5, 1.5])
    assert not faithful(None)
    assert faithful([0.95, 1.05])
    assert not faithful([1.003, 1.083], tolerance=0.05)
