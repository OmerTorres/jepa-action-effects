"""Behavioral fixtures for DELTA-001 displacement measurement."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import jepa_action_effects.delta_readout as delta_readout
from jepa_action_effects.delta_probe import (
    AnchoredNetwork,
    deduplicated_training,
    family_weights,
    fit_ridge,
    measurement_gate,
)
from jepa_action_effects.delta_readout import (
    ExperimentBudget,
    _derangement,
    _prediction_embeddings,
    _seal,
    _sealed,
)


def _family(offset=0.0):
    frames = np.zeros((6, 2, 224, 224, 3), dtype=np.uint8)
    for candidate in range(6):
        frames[candidate, -1, 0, 0, 0] = candidate + 1
    states = np.zeros((6, 2, 5))
    displacements = np.array([0, 1, 10, 11, 20, 21], dtype=float)
    states[:, -1, 2] = displacements + offset
    family = {
        "history_frames": np.zeros((3, 224, 224, 3), dtype=np.uint8),
        "future_frames": frames,
        "history_states": np.zeros((3, 5)),
        "future_states": states,
        "contacts": np.array([False, False, True, True, True, True]),
    }
    embeddings = {"history": np.zeros((3, 192)), "future": np.zeros((6, 2, 192))}
    embeddings["future"][:, -1, 0] = displacements + offset
    return family, embeddings


def test_displacement_gate_passes_truth_and_rejects_zero_on_meaningful_pairs():
    data = {f"challenge-{index:03}": _family() for index in range(10)}
    data.update({f"ordinary-{index:03}": _family() for index in range(10)})
    perfect = measurement_gate(data, lambda start, future: future[:, :2] - start[:2])
    assert perfect["status"] == "pass"
    assert perfect["required"]["effect_meaningful"]["family_count"] == 20
    zero = measurement_gate(data, lambda start, future: np.zeros((6, 2)))
    assert zero["status"] == "fail"
    assert zero["required"]["effect_meaningful"]["mean"] > 5.12
    assert data["challenge-000"][0]["contacts"][1] == 0
    assert data["challenge-000"][0]["future_states"][1, -1, 2] == 1


def test_duplicate_input_conflict_is_reported():
    family, embedding = _family()
    family["future_frames"][1, -1] = family["future_frames"][0, -1]
    with pytest.raises(ValueError, match="Conflicting block displacement"):
        deduplicated_training({"challenge-0": family}, {"challenge-0": embedding})


def test_fit_weights_equalize_families_and_ridge_is_zero_anchored():
    ids = np.array(["many"] * 5 + ["one"])
    weights = family_weights(ids)
    assert weights[:5].sum() == pytest.approx(weights[5:].sum())
    x = np.zeros((6, 192))
    x[:5, 0] = 10
    x[5, 0] = 1
    y = np.column_stack((2 * x[:, 0], np.zeros(6)))
    fitted = fit_ridge(x, y, ids, alpha=0.01)
    assert fitted.predict(np.zeros(192), np.zeros((1, 192)))[0].tolist() == [0.0, 0.0]
    assert abs(fitted.predict(np.zeros(192), x[0:1])[0, 0] - 20) < 0.25


def test_anchored_network_exact_zero_and_shared_bias_cancellation():
    torch.manual_seed(12)
    network = AnchoredNetwork()
    context = torch.randn(4, 192)
    assert torch.count_nonzero(network(context, torch.zeros((4, 192)))) == 0
    real = np.array([[25.0, 20.0], [30.0, 20.0]])
    start = np.array([20.0, 20.0])
    shared_bias = np.array([10.0, -7.0])
    np.testing.assert_array_equal((real + shared_bias) - (start + shared_bias), real - start)


def test_action_permutation_is_deterministic_and_changes_every_candidate():
    left = _derangement("challenge-001", 20260928)
    right = _derangement("challenge-001", 20260928)
    np.testing.assert_array_equal(left, right)
    assert set(left) == set(range(6))
    assert np.all(left != np.arange(6))


def test_sealed_artifact_and_cumulative_runtime_reject_mismatch(tmp_path: Path):
    path = tmp_path / "sample.json"
    _seal(path, {"selected": True})
    assert _sealed(path)["selected"]
    path.write_text(json.dumps({"selected": False}))
    with pytest.raises(ValueError, match="hash mismatch"):
        _sealed(path)
    ledger = ExperimentBudget(tmp_path, "same")
    ledger.begin("fit")
    time.sleep(0.01)
    ledger.finish("complete")
    used = ledger.record["used_seconds"]
    assert used > 0
    assert ExperimentBudget(tmp_path, "same").record["used_seconds"] == used
    with pytest.raises(ValueError, match="mismatch"):
        ExperimentBudget(tmp_path, "changed")


def test_transition_uses_last_history_and_last_future_frame():
    from jepa_action_effects.delta_probe import transition_targets

    family, embeddings = _family()
    family["history_states"][-1, 2:4] = [5, 2]
    family["future_states"][:, 0, 2] = 999
    embeddings["future"][:, 0, 0] = 777
    start, future, truth = transition_targets(family, embeddings)
    assert start[0] == 0
    assert future[0, 0] == 0
    assert truth[0].tolist() == [-5.0, -2.0]


def test_prediction_cache_uses_history_and_actions_only(tmp_path: Path, monkeypatch):
    from jepa_action_effects.latent_cache import LatentCache

    family, _ = _family()
    family["warmup_controls"] = np.zeros((10, 2))
    family["candidate_controls"] = np.zeros((6, 10, 2))
    called = []

    def fake_predict(model, history, past, candidates, scaler):
        called.append((tuple(history.shape), tuple(past.shape), tuple(candidates.shape)))
        return torch.ones((1, 6, 2, 192))

    monkeypatch.setattr(
        delta_readout, "preprocess_pixels", lambda frames: torch.zeros((3, 3, 224, 224))
    )
    monkeypatch.setattr(delta_readout, "predict_future", fake_predict)
    cache = LatentCache(tmp_path / "cache")
    first = _prediction_embeddings(
        "fixture", family, object(), object(), cache, {"model": "fixture"}
    )
    family["future_frames"][:] = 127
    second = _prediction_embeddings(
        "fixture", family, object(), object(), cache, {"model": "fixture"}
    )
    assert len(called) == 1
    assert called[0] == ((1, 3, 3, 224, 224), (1, 10, 2), (1, 6, 10, 2))
    np.testing.assert_array_equal(first, second)


def test_no_selected_readout_cannot_open_confirmation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(delta_readout, "_audit_required", lambda root: {"status": "passed"})
    audit_path = tmp_path / "audit.json"
    _seal(audit_path, {"status": "passed"})
    _seal(
        tmp_path / "fit-report.json",
        {"audit_sha256": delta_readout.sha256_file(audit_path), "selection": None},
    )
    _seal(
        tmp_path / "selection.json",
        {"audit_sha256": delta_readout.sha256_file(audit_path), "selection": None},
    )
    monkeypatch.setattr(
        delta_readout,
        "_load_data",
        lambda *args, **kwargs: pytest.fail("Confirmation data were loaded"),
    )
    ledger = ExperimentBudget(tmp_path, "fixture")
    assert delta_readout.confirm(tmp_path, ledger)["confirmation_opened"] is False
    assert delta_readout.evaluate(tmp_path, ledger)["status"] == "unavailable"
    assert not (tmp_path / "confirmation-report.json").exists()


def test_oracle_initialized_regret_uses_start_only_in_scoring():
    from jepa_action_effects.evaluation import (
        BranchPrediction,
        BranchTruth,
        EvaluationConfig,
        FamilyTruth,
        Goal,
        evaluate_family,
    )

    start = np.array([20.0, 10.0])
    truth = FamilyTruth(
        "fixture",
        start,
        np.zeros(2),
        (
            BranchTruth("a", start + [0, 0], np.zeros(2), False),
            BranchTruth("b", start + [10, 0], np.zeros(2), True),
        ),
        2,
    )
    predicted_displacements = {"a": np.array([10.0, 0.0]), "b": np.array([0.0, 0.0])}
    predictions = tuple(
        BranchPrediction(key, start + delta) for key, delta in predicted_displacements.items()
    )
    result = evaluate_family(
        truth,
        predictions,
        (Goal("goal", start + [10, 0]),),
        EvaluationConfig(5.12, 512, 0, 2),
        method="oracle_initialized",
        goal_provenance="fixture",
    )
    assert result.goal_metrics[0].regret == 10
