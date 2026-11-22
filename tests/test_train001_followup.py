"""TRAIN-001 review follow-up: training-lineage validation and contrast-level sensitivity."""

from __future__ import annotations

import copy
import json

import pytest
from test_train001_provenance import FINGERPRINT, SOURCES, run  # noqa: F401  (fixture)

from jepa_action_effects.interface_audit import WEIGHTS_SHA256
from jepa_action_effects.latent_audit import seal
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001 import build_seal
from jepa_action_effects.train001_followup import (
    contrast_sensitivity,
    validate_lineage,
    validate_with_lineage,
)
from jepa_action_effects.train001_protocol import CONDITIONS, READOUTS
from jepa_action_effects.train001_provenance import ENTRY, validate

ARCHIVE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # the fixture's development weights


def _complete_runs(root, protocol, **overrides):
    """Add lineage fields to the fixture's runs, then build the seal (as `seal` would)."""
    for condition in CONDITIONS:
        path = root / "runs" / f"{condition}-1" / "run.json"
        payload = json.loads(path.read_text())
        weights = [0.25] * 4 if condition == "balanced" else ARCHIVE_WEIGHTS
        payload.update(
            {
                "collection_sha256": sha256_file(root / "development" / "collection.json"),
                "cache_sha256": sha256_file(root / "cache" / "development.json"),
                "weights": weights,
                "spec": {
                    "condition": condition,
                    "seed": 1,
                    "learning_rate": 1e-4,
                    "updates": protocol.updates,
                    "weights": weights,
                    "weight_decay": protocol.weight_decay,
                    "gradient_clip": protocol.gradient_clip,
                    "per_scene": protocol.families_per_scene_batch,
                    "threads": protocol.threads,
                    "checkpoint_updates": list(protocol.checkpoint_updates),
                    "resume_every": protocol.resume_every,
                    "curve_every": protocol.curve_every,
                    "validation_every": protocol.validation_every,
                },
            }
        )
        for key, value in overrides.get(condition, {}).items():
            if key.startswith("spec."):
                payload["spec"][key[5:]] = value
            else:
                payload[key] = value
        seal(path, payload)
    digests = {
        name: sha256_file(root.parent / "readout002" / "models" / f"{name}.npz")
        for name in READOUTS
    }
    build_seal(root, protocol, SOURCES, FINGERPRINT, digests)


def test_matching_lineage_passes(run):  # noqa: F811
    root, protocol, _ = run
    _complete_runs(root, protocol)
    assert validate_lineage(root, protocol, WEIGHTS_SHA256)["valid"]
    assert validate_with_lineage(root, protocol, ENTRY, SOURCES, FINGERPRINT, WEIGHTS_SHA256)[
        "valid"
    ]


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"balanced": {"collection_sha256": "f" * 64}}, "collection_sha256"),
        ({"archive_weighted": {"cache_sha256": "e" * 64}}, "cache_sha256"),
        ({"archive_weighted": {"weights": [0.25] * 4}}, "weights"),
        ({"balanced": {"spec.learning_rate": 1e-3}}, "spec.learning_rate"),
        ({"balanced": {"spec.updates": 3}}, "spec.updates"),
        ({"archive_weighted": {"spec.weights": [0.25] * 4}}, "spec.weights"),
    ],
)
def test_stale_training_lineage_is_rejected(run, overrides, problem):  # noqa: F811
    root, protocol, _ = run
    _complete_runs(root, protocol, **overrides)
    # The sealed validator alone admits the stale run (the reviewed gap) ...
    assert validate(root, protocol, ENTRY, SOURCES, FINGERPRINT, WEIGHTS_SHA256)["valid"]
    # ... the lineage check rejects it.
    report = validate_with_lineage(root, protocol, ENTRY, SOURCES, FINGERPRINT, WEIGHTS_SHA256)
    assert not report["valid"]
    failed = [c for c in report["checks"] if c["status"] != "passed"]
    assert any(problem in c["detail"] for c in failed)


def _summary(alternative_interval=None, alternative_outcome="non_inferiority_not_established"):
    def entry(estimate, ci):
        return {"estimate": estimate, "ci": ci}

    def boot(ba_ci):
        base = {
            "balanced_minus_archive": entry(-0.1, ba_ci),
            "balanced_minus_frozen": entry(-11.0, [-14.0, -9.0]),
            "archive_minus_frozen": entry(-11.0, [-14.0, -9.0]),
            "calibrated_frozen_minus_frozen": entry(-0.3, [-0.4, -0.2]),
        }
        for name in ("balanced_minus_archive", "balanced_minus_frozen"):
            base[name]["by_scene"] = {
                "challenge": entry(-0.3, [-0.6, -0.1]),
                "archive": entry(0.1, [-0.7, 1.0]),
            }
        return base

    block = {
        "primary": {"bootstrap": boot([-0.5, 0.4]), "verdict": {"verdict": "not_assessable"}},
        "ordinary": {
            "bootstrap": {
                "balanced_minus_frozen": entry(1.0, [0.1, 2.0]),
                "archive_minus_frozen": entry(0.7, [-0.2, 1.6]),
                "balanced_minus_archive": entry(0.4, [0.05, 0.7]),
            },
            "outcome": "non_inferiority_not_established",
        },
        "measurement_gates": {"cells": {"archive:m0.45": {"passed": True}}},
    }
    readouts = {name: copy.deepcopy(block) for name in READOUTS}
    if alternative_interval is not None:
        readouts[READOUTS[1]]["primary"]["bootstrap"]["balanced_minus_archive"]["ci"] = (
            alternative_interval
        )
    readouts[READOUTS[2]]["ordinary"]["outcome"] = alternative_outcome
    return {
        "readouts": readouts,
        "readout_sensitivity": {
            "measurement_dependent": False,
            "verdicts": dict.fromkeys(READOUTS, "not_assessable"),
        },
    }


def test_sensitivity_detects_conflicting_contrasts_after_eligibility_fails():
    # The sealed verdict-string flag says False although one readout's interval
    # lies far above zero (the review's counterexample).
    summary = _summary(alternative_interval=[1397.6, 1400.9])
    assert summary["readout_sensitivity"]["measurement_dependent"] is False
    result = contrast_sensitivity(summary)
    assert result["sign_disagreements"] == ["primary:balanced_minus_archive"]
    assert result["contrast_level_measurement_dependent"]


def test_sensitivity_agreement_and_outcome_disagreement():
    assert not contrast_sensitivity(_summary())["contrast_level_measurement_dependent"]
    changed = contrast_sensitivity(
        _summary(alternative_outcome="regression_beyond_tolerance_supported")
    )
    assert not changed["ordinary_outcome"]["agree"]
    assert changed["contrast_level_measurement_dependent"]


def test_failed_validation_writes_nothing_and_exits_nonzero(tmp_path, monkeypatch):
    # TRAIN-REVIEW-003: a failed combined validation must not replace the tracked
    # presentation or manifest, and the command must fail.
    import jepa_action_effects.train001 as train001
    import jepa_action_effects.train001_followup as followup
    import jepa_action_effects.train001_report as report

    manifest = tmp_path / "train001-followup.json"
    manifest.write_text('{"existing": true}\n')
    before = manifest.read_bytes()
    failed = {"check": "lineage:balanced-1", "status": "failed", "detail": ["cache_sha256"]}

    def forbidden(*args, **kwargs):
        raise AssertionError("report output written after failed validation")

    monkeypatch.setattr(followup, "MANIFEST", manifest)
    monkeypatch.setattr(followup, "sealed", lambda path: _summary())
    monkeypatch.setattr(
        followup, "validate_with_lineage", lambda *a, **k: {"checks": [failed], "valid": False}
    )
    monkeypatch.setattr(train001, "sources", lambda: SOURCES)
    monkeypatch.setattr(train001, "frozen_fingerprint", lambda: FINGERPRINT)
    monkeypatch.setattr(report, "regenerate_presentation", forbidden)
    monkeypatch.setattr(followup, "write_json", forbidden)

    with pytest.raises(followup.ValidationFailure) as error:
        followup.run()
    assert error.value.failed == [failed]
    with pytest.raises(SystemExit) as exit_info:
        followup.main([])
    assert exit_info.value.code == 1
    assert manifest.read_bytes() == before
