"""REPORT-002: training-curve export checks run hashes before copying curves."""

from __future__ import annotations

import json

import pytest

from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.training_curves import RunMismatch, collect


def _run(root, run_id, arm):
    path = root / run_id / "run.json"
    path.parent.mkdir(parents=True)
    record = {
        "spec": {"updates": 20, "learning_rate": 1e-5},
        "curve": [[1, 0.2, 1.5], [10, 0.1, 0.9]],
        "validation_curve": [[10, 0.3]],
        "condition": arm,
    }
    path.write_text(json.dumps(record))
    return path


def test_collect_copies_curves_after_hash_check(tmp_path):
    path = _run(tmp_path, "balanced-7", "balanced")
    registry = {"balanced-7": {"run_sha256": sha256_file(path)}}
    payload = collect(registry, {"balanced": tmp_path}, seeds=(7,))
    run = payload["runs"]["balanced-7"]
    assert run["train_curve"] == [[1, 0.2], [10, 0.1]]
    assert run["grad_norm"] == [[1, 1.5], [10, 0.9]]
    assert run["validation_curve"] == [[10, 0.3]] and run["updates"] == 20


def test_collect_refuses_a_changed_run(tmp_path):
    _run(tmp_path, "balanced-7", "balanced")
    with pytest.raises(RunMismatch):
        collect({"balanced-7": {"run_sha256": "0" * 64}}, {"balanced": tmp_path}, seeds=(7,))
