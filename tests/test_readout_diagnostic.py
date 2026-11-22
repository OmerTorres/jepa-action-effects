"""Behavioral checks for READOUT-001 measurement and partition invariants."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.pilot import physical_competence_status
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.pilot_readouts import fit_linear
from jepa_action_effects.readout_diagnostic import (
    _alignment_error,
    _duplicate_conflicts,
    _require_audit,
    _tiny_fit,
)
from jepa_action_effects.readout_recovery import (
    _development_ids,
    _load_encoded_family,
    _resume_progress,
    _tiers,
)


def _parts():
    def ids(prefix, challenge, ordinary):
        return [f"{prefix}:challenge-{index:03}" for index in range(challenge)] + [
            f"{prefix}:ordinary-{index:03}" for index in range(ordinary)
        ]

    return {
        "pilot_fit": ids("pilot", 40, 20),
        "new_fit": ids("new", 120, 60),
        "tuning": ids("tuning", 40, 20),
        "confirmation": ids("confirmation", 40, 20),
    }


def test_nested_fit_partitions_and_confirmation_isolation():
    parts = _parts()
    tiers = _tiers(parts)
    assert list(map(len, tiers.values())) == [60, 120, 240]
    assert set(tiers[60]) < set(tiers[120]) < set(tiers[240])
    assert not set(parts["confirmation"]) & set(_development_ids(parts))
    assert len(_development_ids(parts)) == 300
    parts["confirmation"][0] = parts["new_fit"][0]
    with pytest.raises(ValueError, match="overlap"):
        _tiers(parts)


def test_duplicate_frame_conflicts_report_labels():
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    history = np.stack([frame] * 3)
    future = np.stack([frame] * 12).reshape(6, 2, 224, 224, 3)
    history_states = np.tile([10, 20, 100, 200, 0], (3, 1)).astype(float)
    future_states = np.tile([10, 20, 100, 200, 0], (12, 1)).reshape(6, 2, 5).astype(float)
    future_states[0, 0, 0] = 11
    family = {
        "history_frames": history,
        "future_frames": future,
        "history_states": history_states,
        "future_states": future_states,
    }
    result = _duplicate_conflicts({"fit": family}, {"fit": "fit"})
    assert result["agent_conflict_count"] == 1
    assert result["block_conflict_count"] == 0
    future_states[0, 1, 2] = 101
    assert _duplicate_conflicts({"fit": family}, {"fit": "fit"})["block_conflict_count"] == 1


def test_history_future_alignment_and_circular_angle():
    history = np.zeros((3, 5))
    future = np.zeros((2, 5))
    stored = {"history_states": history.copy(), "future_states": future[None].copy()}
    stored["history_states"][0, 4] = 2 * np.pi
    assert _alignment_error(history, future, stored) < 1e-12
    stored["future_states"][0, 1, 2] = 1
    assert _alignment_error(history, future, stored) == 1


def test_projected_and_raw_cls_cache_separation(tmp_path: Path, monkeypatch):
    import jepa_action_effects.readout_recovery as recovery

    family_path = tmp_path / "family.npz"
    np.savez_compressed(
        family_path,
        metadata=np.array(json.dumps({"family_id": "challenge-000"})),
        history_frames=np.zeros((3, 224, 224, 3), dtype=np.uint8),
        future_frames=np.zeros((6, 2, 224, 224, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        recovery, "preprocess_pixels", lambda frames: torch.zeros((15, 3, 224, 224))
    )
    monkeypatch.setattr(recovery, "encode_real", lambda model, pixels: torch.ones((1, 15, 192)))

    class FakeModel:
        def encoder(self, pixels, interpolate_pos_encoding):
            assert interpolate_pos_encoding
            return SimpleNamespace(last_hidden_state=torch.zeros((15, 1, 192)))

    manifest = {"family_files": {"challenge-000": {"path": "family.npz"}}}
    _, views = _load_encoded_family(
        "pilot:challenge-000",
        tmp_path,
        tmp_path,
        manifest,
        manifest,
        LatentCache(tmp_path / "cache"),
        {"checkpoint": "fixture"},
        FakeModel(),
    )
    assert views["projected"]["future"].shape == (6, 2, 192)
    assert views["raw_cls"]["future"].shape == (6, 2, 192)
    assert np.all(views["projected"]["future"] == 1)
    assert np.all(views["raw_cls"]["future"] == 0)
    assert len(list((tmp_path / "cache" / "real").glob("*.npz"))) == 2


def test_fit_only_equal_family_feature_statistics():
    x = np.zeros((6, 192))
    x[:5, 0] = 10
    x[5, 0] = 0
    states = np.zeros((6, 5))
    states[:, 2:4] = x[:, :2]
    model = fit_linear(x, states, np.array(["many"] * 5 + ["one"]), 1.0)
    assert model.feature_mean[0] == pytest.approx(5.0)
    assert model.feature_scale[0] == pytest.approx(5.0)
    assert model.feature_mean[1] == 0
    assert model.feature_scale[1] == 1


def test_audit_hash_rejects_mismatch(tmp_path: Path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"status": "passed"}))
    (tmp_path / "audit.sha256").write_text(sha256_file(path))
    assert _require_audit(tmp_path)["status"] == "passed"
    path.write_text(json.dumps({"status": "blocked"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        _require_audit(tmp_path)


def test_fit_progress_resumes_only_matching_inputs(tmp_path: Path):
    path = tmp_path / "fit-progress.json"
    assert _resume_progress(path, "matching") == {"results": {}, "selection": None}
    path.write_text(json.dumps({"signature": "matching", "results": {"60": {}}, "selection": None}))
    assert _resume_progress(path, "matching")["results"] == {"60": {}}
    with pytest.raises(ValueError, match="signature mismatch"):
        _resume_progress(path, "changed")


def test_failed_readout_makes_physical_competence_unassessable():
    assert physical_competence_status(False, 10, 1) == "unassessable"
    assert physical_competence_status(True, None, 1) == "unassessable"
    assert physical_competence_status(True, 10, 1) == "fail"
    assert physical_competence_status(True, 1, 10) == "pass"


def test_tiny_subset_synthetic_recovery(tmp_path: Path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 192))
    states = np.zeros((64, 5))
    states[:, 2:4] = 2 * x[:, :2] + np.array([10, -7])
    result = _tiny_fit(x, states, tmp_path)
    assert result["least_squares_max_error"] < 1e-8
    assert result["nonlinear_final_mean_error"] < 1
    assert (tmp_path / "tiny-curve.json").exists()
