"""Behavioral tests for ERASURE-001 erasure, isolation, and resumability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_action_effects import concept_erasure as runner
from jepa_action_effects.concept_erasure_metrics import (
    bootstrap_gain,
    family_weights,
    fit_leace,
    interval,
    random_eraser,
    score_probe,
    select_probe,
)


def test_leace_erases_target_and_is_idempotent_with_correlated_nuisance():
    rng = np.random.default_rng(10)
    concept = rng.normal(size=(240, 2))
    nuisance = rng.normal(size=(240, 2))
    x = np.column_stack(
        (concept, nuisance, concept[:, :1] + nuisance[:, :1], rng.normal(size=(240, 3)))
    )
    ids = np.repeat([f"family-{index:03}" for index in range(80)], 3)
    eraser, diagnostics = fit_leace(x, concept, ids)
    erased = eraser.transform(x)
    assert diagnostics["removed_rank"] == 2
    assert diagnostics["cross_covariance_after_fro"] < 1e-9
    assert diagnostics["idempotence_max_abs"] < 1e-9
    assert np.max(np.abs(eraser.transform(erased) - erased)) < 1e-9
    # A nuisance dimension independent of the concept remains useful.
    assert np.corrcoef(erased[:, 3], nuisance[:, 1])[0, 1] > 0.99


def test_rank_deficient_erasure_and_random_control():
    rng = np.random.default_rng(11)
    concept = rng.normal(size=(40, 1))
    x = np.column_stack((concept, concept, np.zeros((40, 3))))
    ids = np.repeat(np.arange(20), 2)
    eraser, diagnostics = fit_leace(x, concept, ids)
    assert diagnostics["covariance_rank"] == 1
    assert eraser.removed_rank == 1
    random = random_eraser(eraser.mean, 5, 1, rng)
    assert np.max(np.abs(random.matrix @ random.matrix - random.matrix)) < 1e-10


def test_equal_family_weighting_and_fit_only_probe_selection():
    assert family_weights(np.array(["many"] * 3 + ["one"])).tolist() == pytest.approx(
        [1 / 6, 1 / 6, 1 / 6, 1 / 2]
    )
    rng = np.random.default_rng(12)
    train_x = np.zeros((80, 192))
    train_x[:, :2] = rng.normal(size=(80, 2))
    train_y = train_x[:, :2] * 3 + 5
    train_ids = np.repeat([f"f{i}" for i in range(40)], 2)
    calibration_x = np.zeros((20, 192))
    calibration_x[:, :2] = rng.normal(size=(20, 2))
    calibration_y = calibration_x[:, :2] * 3 + 5
    calibration_ids = np.repeat([f"v{i}" for i in range(10)], 2)
    model, scores = select_probe(
        train_x, train_y, train_ids, calibration_x, calibration_y, calibration_ids
    )
    assert str(model.alpha) in scores
    result = score_probe(
        model,
        calibration_x,
        calibration_y,
        calibration_ids,
        np.average(train_y, axis=0),
    )
    assert result["gain"] > 0.9


def test_bootstrap_family_weighting_and_zero_variance():
    target = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    prediction = target.copy()
    ids = np.array(["a", "a", "b"])
    boot = bootstrap_gain(prediction, target, ids, np.zeros(2), 13, 40)
    assert all(value == 1 for value in boot if np.isfinite(value))
    assert interval(boot) == [1.0, 1.0]
    assert interval([np.nan]) is None
    stratified = bootstrap_gain(
        prediction,
        target,
        ids,
        np.zeros(2),
        13,
        40,
        strata=np.array(["challenge", "challenge", "ordinary"]),
    )
    assert stratified == boot


def test_partition_is_nested_disjoint_and_excludes_confirmation():
    pilot = {
        "splits": {f"challenge-{i}": "fit" for i in range(40)}
        | {f"ordinary-{i}": "fit" for i in range(20)}
    }
    fresh = {
        "splits": {f"challenge-{i}": "fit" for i in range(120)}
        | {f"ordinary-{i}": "fit" for i in range(60)}
        | {f"challenge-v{i}": "validation" for i in range(40)}
        | {f"ordinary-v{i}": "validation" for i in range(20)}
        | {f"challenge-d{i}": "development" for i in range(40)}
        | {f"ordinary-d{i}": "development" for i in range(20)}
    }
    groups = runner._partition(pilot, fresh)
    assert list(map(len, groups.values())) == [240, 30, 30, 60]
    assert not set(groups["assessment"]) & set(groups["excluded_confirmation"])
    assert sum("challenge" in item for item in groups["assessment"]) == 20


def test_identical_frame_conflicts_are_rejected():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    family = {
        "history_frames": np.stack((frame, frame)),
        "history_states": np.asarray([[1, 2, 3, 4, 0], [2, 2, 3, 4, 0]]),
    }
    with pytest.raises(ValueError, match="Conflicting labels"):
        runner._deduplicated("new:fixture", family, np.zeros((2, 192)))


def test_sealed_artifact_and_budget_reject_mismatch(tmp_path: Path, monkeypatch):
    runner.seal(tmp_path / "audit.json", {"audit_seconds": 0.0})
    path = tmp_path / "value.json"
    runner.seal(path, {"value": 1})
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="mismatch"):
        runner.sealed(path)
    times = iter((10.0, 11.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(runner.time, "time", lambda: 10.0)
    budget = runner.Budget(tmp_path, "same")
    budget.begin("fit")
    budget.finish("complete")
    assert budget.data["used_seconds"] == 1
    budget.data["used_seconds"] = runner.EXPERIMENT_LIMIT
    runner.write_json(budget.path, budget.data)
    with pytest.raises(TimeoutError):
        runner.Budget(tmp_path, "same").check()


def test_assessment_opening_requires_fit_and_is_persistent(tmp_path: Path):
    runner.seal(tmp_path / "audit.json", {"status": "passed"})
    assert not (tmp_path / "assessment-opening.json").exists()
    opening = {"signature": "fixed", "assessment_ids": ["a"]}
    runner.seal(tmp_path / "assessment-opening.json", opening)
    assert runner.sealed(tmp_path / "assessment-opening.json") == opening
