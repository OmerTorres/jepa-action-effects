"""PILOT-001 generation, readout, manifest, and horizon regressions."""

import json
from pathlib import Path

import numpy as np
import pytest

import jepa_action_effects.pilot_collection as pilot_collection
from jepa_action_effects.pilot_collection import (
    PilotConfig,
    config_payload,
    propose_challenge,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.pilot_readouts import readout_gate, select_ridge


def test_challenge_proposal_is_seeded_bounded_and_uses_actual_geometry():
    config = PilotConfig()
    left, reason_left = propose_challenge(config, 0, np.random.default_rng(12))
    right, reason_right = propose_challenge(config, 0, np.random.default_rng(12))
    assert reason_left == reason_right
    assert left is not None and right is not None
    left_recipe, left_candidates, metadata = left
    right_recipe, right_candidates, _ = right
    np.testing.assert_array_equal(left_recipe.warmup_controls, right_recipe.warmup_controls)
    np.testing.assert_array_equal(
        np.stack([value.controls for value in left_candidates]),
        np.stack([value.controls for value in right_candidates]),
    )
    assert np.max(np.abs(np.stack([value.controls for value in left_candidates]))) <= 1
    assert np.asarray(metadata["actual_block_vertices"]).shape == (8, 2)
    assert config_payload(config) == config_payload(config)


def test_family_weighted_ridge_selection_and_readout_gate():
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(40, 192))
    block = embeddings[:, :2] * 2 + np.array([200, 250])
    agent = embeddings[:, 2:4] + np.array([100, 120])
    angle = embeddings[:, 4]
    states = np.column_stack((agent, block, angle))
    family_ids = np.repeat([f"fit-{index}" for index in range(10)], 4)
    validation = {
        f"validation-{index}": (
            embeddings[index * 4 : index * 4 + 4],
            states[index * 4 : index * 4 + 4],
        )
        for index in range(10)
    }
    model, scores = select_ridge(embeddings, states, family_ids, validation)
    assert model.alpha in {0.01, 0.1, 1, 10, 100, 1000}
    assert len(scores) == 6
    predicted = model.predict(embeddings)["block"]
    errors = {
        f"validation-{index}": np.linalg.norm(
            predicted[index * 4 : index * 4 + 4] - block[index * 4 : index * 4 + 4], axis=1
        )
        for index in range(10)
    }
    contacts = {family_id: np.array([False, False, True, True]) for family_id in errors}
    gate = readout_gate(errors, contacts, delta=20.0)
    assert gate["overall"]["family_count"] == 10
    assert gate["neither"]["sufficient_families"]
    assert gate["exactly_one"]["sufficient_families"]
    assert gate["both"]["sufficient_families"]
    assert gate["passed"]


def test_collection_manifest_rejects_hash_mismatch(tmp_path: Path):
    manifest = {"status": "complete", "family_files": {}}
    path = tmp_path / "collection-manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "collection-manifest.sha256").write_text(sha256_file(path))
    assert verify_collection(tmp_path) == manifest
    path.write_text(json.dumps({**manifest, "changed": True}))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_collection(tmp_path)


def test_collection_attempt_cap_and_rejections_are_recorded(tmp_path: Path, monkeypatch):
    config = PilotConfig(
        challenge_target=0,
        ordinary_target=1,
        challenge_attempt_cap=0,
        ordinary_attempt_cap=2,
        challenge_split_counts=(0, 0, 0),
        ordinary_split_counts=(1, 0, 0),
    )

    def fail_replay(*args, **kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(pilot_collection, "run_family", fail_replay)
    manifest = pilot_collection.collect(tmp_path, config)
    assert manifest["status"] == "quota_shortfall"
    assert manifest["attempts"] == {"challenge": 0, "ordinary": 2}
    assert [value["reason"] for value in manifest["rejections"]] == [
        "replay_error:fixture failure",
        "replay_error:fixture failure",
    ]


def test_interrupted_collection_resumes_from_saved_rng_state(tmp_path: Path, monkeypatch):
    config = PilotConfig(
        challenge_target=0,
        ordinary_target=1,
        challenge_attempt_cap=0,
        ordinary_attempt_cap=1,
        challenge_split_counts=(0, 0, 0),
        ordinary_split_counts=(1, 0, 0),
    )
    original = pilot_collection.run_family

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot_collection, "run_family", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pilot_collection.collect(tmp_path, config)
    progress = json.loads((tmp_path / "collection-progress.json").read_text())
    assert progress["attempts"] == {"challenge": 0, "ordinary": 0}

    monkeypatch.setattr(pilot_collection, "run_family", original)
    manifest = pilot_collection.collect(tmp_path, config)
    assert len(manifest["accepted"]["ordinary"]) == 1
    assert manifest["attempts"]["ordinary"] == 1
    assert not (tmp_path / "collection-progress.json").exists()
