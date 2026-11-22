"""Model-free fixtures for the DIST-001, REANALYSIS-001, READOUT-002 and PHYS-002 stages."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jepa_action_effects import distribution_audit as dist
from jepa_action_effects import phys002, readout_scale
from jepa_action_effects.latent_audit import seal
from jepa_action_effects.latent_metrics import cost_matrix, pair_rows
from jepa_action_effects.predictor_gallery_metrics import candidate_scores
from jepa_action_effects.reanalysis import gallery_motion_split, latent_family, latent_summary

CHALLENGE_IDS = ["hold", "lateral_0", "lateral_1", "lateral_2", "lateral_3", "push"]


def _controls() -> np.ndarray:
    vectors = [np.zeros(2)] + [np.array([o, -0.6]) for o in (-0.2, -0.1, 0.1, 0.2)]
    vectors.append(np.array([0.0, -0.6]))
    return np.stack([np.tile(v, (10, 1)) for v in vectors]).astype(np.float32)


# --------------------------------------------------------------------------- READOUT-002
def test_targets_and_evaluate_perfect_prediction() -> None:
    labels = np.array([[10, 20, 100, 200, 0.5], [30, 40, 300, 100, -2.0], [5, 5, 50, 60, 3.0]])
    y = readout_scale.targets(labels)
    assert y.shape == (3, 6)
    scores = readout_scale.evaluate(y, y)
    assert scores["block"]["mean"] == 0 and scores["block"]["r2"] == pytest.approx(1.0)
    assert scores["orientation"]["mean_abs_degrees"] == pytest.approx(0.0, abs=1e-9)


def test_ridge_recovers_linear_embedding_and_roundtrips(tmp_path) -> None:
    rng = np.random.default_rng(0)
    labels = np.column_stack((rng.uniform(30, 480, (3000, 4)), rng.uniform(-np.pi, np.pi, 3000)))
    y = readout_scale.targets(labels)
    mixing = rng.normal(size=(6, 192))
    x = y @ mixing + rng.normal(0, 1e-3, size=(3000, 192))
    probe, _ = readout_scale.fit_ridge(x[:2500], y[:2500], x[2500:], y[2500:])
    scores = readout_scale.evaluate(probe.predict(x[2500:]), y[2500:])
    assert scores["block"]["mean"] < 0.5 and scores["agent"]["r2"] > 0.999
    np.savez_compressed(tmp_path / "ridge.npz", **probe.arrays())
    from jepa_action_effects.pilot_collection import sha256_file

    seal(
        tmp_path / "selection.json",
        {"kind": "ridge", "path": "ridge.npz", "path_sha256": sha256_file(tmp_path / "ridge.npz")},
    )
    block, _ = readout_scale.load_block_probe(tmp_path)
    assert np.allclose(block(x[2500:]), probe.predict(x[2500:])[:, :2])


def test_mlp_numpy_forward_matches_training_validation(tmp_path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1200, 192))
    labels = np.column_stack((rng.uniform(0, 500, (1200, 4)), rng.uniform(-3, 3, 1200)))
    labels[:, 2] = 250 + 40 * np.tanh(x[:, 0])
    y = readout_scale.targets(labels)
    state, info = readout_scale.fit_mlp(x[:1000], y[:1000], x[1000:], y[1000:], 7, epochs=15)
    predicted = readout_scale.mlp_predictor(state)(x[1000:])
    error = float(np.mean(np.linalg.norm(predicted[:, :2] - y[1000:, :2], axis=1)))
    assert error == pytest.approx(info["validation_block_mean"], rel=1e-4)
    assert info["epochs_run"] <= 15


def test_render_only_placement_labels_exact_poses() -> None:
    env = readout_scale._new_env()
    try:
        pose = np.array([100.0, 100.0, 300.0, 320.0, 0.7])
        assert readout_scale._place(env, pose)
        assert np.allclose(readout_scale._labels(env), pose)
        first = env.render().copy()
        readout_scale._place(env, pose + np.array([0, 0, 2.0, 0, 0]))
        assert not np.array_equal(first, env.render())
        overlap = np.array([300.0, 320.0, 300.0, 320.0, 0.7])
        assert not readout_scale._place(env, overlap)
    finally:
        env.close()


def test_sensitivity_summary_gain_is_one_for_exact_decoder() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(20, 192))
    directions = rng.normal(size=(20, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sizes = np.array([0.5, 2.0])
    shifted = np.repeat(base[:, None], 2, axis=1).copy()
    shifted[:, :, :2] += sizes[None, :, None] * directions[:, None]
    data = {
        "base": base,
        "shifted": shifted,
        "directions": directions,
        "valid": np.ones((20, 2), dtype=bool),
        "sizes": sizes,
        "identical_images": np.zeros((20, 2), dtype=bool),
    }
    rows = readout_scale.sensitivity_summary(lambda v: v[:, :2], data)
    assert rows["0.5"]["gain"] == pytest.approx(1.0)
    assert rows["2.0"]["across_mean"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- DIST-001
def test_identity_stage_on_simulator_generated_archive(tmp_path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    state0 = np.array([100.0, 100.0, 300.0, 250.0, 0.4, 0.0, 0.0])
    controls = np.array([[0.1, -0.2], [0.0, -0.3], [0.2, 0.1]], dtype=np.float32)
    replay = dist._replay(state0, controls)
    path = tmp_path / "archive.h5"
    with h5py.File(path, "w") as h5:
        h5["pixels"] = replay["frames"]
        h5["state"] = replay["states"]
        h5["action"] = np.vstack((controls, np.zeros((1, 2), np.float32)))
        h5["ep_len"] = np.array([4])
        h5["ep_offset"] = np.array([0])
    monkeypatch.setattr(dist, "IDENTITY_STEPS", 3)
    monkeypatch.setattr(dist, "IDENTITY_EPISODES", 1)
    keys = dist.keys(tmp_path, path)
    assert keys["state_key"] == "state" and keys["pixels_key"] == "pixels"
    result = dist.identity(tmp_path, path)
    assert result["reset_frame_identical_fraction"] == 1.0
    assert result["rows"][0]["identical_fraction"] == 1.0
    assert result["rendering_mismatch_flag"] is False


# --------------------------------------------------------------------------- REANALYSIS-001
def _bank(position: float) -> np.ndarray:
    values = np.zeros((5, 192), dtype=np.float32)
    values[:, 0] = position + np.array([0, -3, -2, 2, 3])
    return values


def test_gallery_motion_split_separates_moved_endpoints() -> None:
    predictions = np.zeros((6, 192), dtype=np.float32)
    predictions[:, 0] = 20.0
    persistence = np.zeros(192, dtype=np.float32)
    endpoints = [
        {"eligible": True, **candidate_scores(predictions, persistence, _bank(20.0), 1e-12)}
        for _ in range(6)
    ]
    record = {
        "cohort": "challenge",
        "motion": [True, True, False, False, False, False],
        "scales": {
            scale: {"endpoints": endpoints} for scale in ("negative_10.24", "negative_20.48")
        },
    }
    result = gallery_motion_split([record] * 12, seed=3)
    moved = result["negative_10.24"]["moved"]["challenge"]
    assert moved["endpoints"] == 24
    assert moved["metrics"]["matched_accuracy"]["mean"] == 1.0


def test_latent_family_boundary_and_slope() -> None:
    rng = np.random.default_rng(4)
    real = rng.normal(size=(6, 192))
    predicted = 0.5 * real
    contacts = np.array([False, True, True, False, True, True])
    rows = pair_rows(
        cost_matrix(predicted, real),
        contacts,
        np.zeros((6, 2)),
        np.zeros((6, 2)),
        np.zeros(2),
        np.zeros(6, dtype=bool),
        predicted,
        real,
    )
    record = {"id": "new:challenge-001", "cohort": "challenge", "pairs": rows}
    family = latent_family(record, _controls(), CHALLENGE_IDS)
    assert family["designed_pairs"] == 7
    assert family["boundary"]["exactly_one"] is not None and family["boundary"]["both"] is not None
    summary = latent_summary([family] * 12, seed=5)
    assert summary["slope"]["designed"]["estimate"] == pytest.approx(0.5)
    assert summary["spread_ratio"]["all"]["estimate"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- PHYS-002
def test_score_family_uses_history_only_and_frozen_probe(monkeypatch) -> None:
    family = {
        "history_frames": np.zeros((3, 224, 224, 3), np.uint8),
        "future_frames": np.zeros((6, 2, 224, 224, 3), np.uint8),
        "candidate_controls": _controls(),
        "warmup_controls": np.zeros((10, 2), np.float32),
        "history_states": np.tile([100.0, 100.0, 250.0, 250.0, 0.0], (3, 1)),
        "future_states": np.tile([100.0, 100.0, 250.0, 250.0, 0.0], (6, 2, 1)),
        "contacts": np.array([False, True, True, False, True, True]),
        "metadata": {"candidate_ids": CHALLENGE_IDS},
    }
    family["future_states"][:, -1, 2] += np.arange(6) * 10.0
    real = np.zeros((15, 192), np.float32)
    real[3:, 0] = 250 + np.repeat(np.arange(6) * 10.0, 2)
    real[:3, 0] = 250
    real[:, 1] = 250
    predicted = real[3:].reshape(6, 2, 192).copy()
    seen = {}

    def fake_predict(model, scaler, fam):
        seen["future_frames_used"] = False
        return predicted

    monkeypatch.setattr(phys002, "encode_frames", lambda model, frames: {"projected": real})
    monkeypatch.setattr(phys002, "predict_family", fake_predict)
    record, _ = phys002.score_family(
        "reserved:challenge-000", family, [[256.0, 256.0]], None, None, lambda v: v[:, :2]
    )
    assert record["cohort"] == "challenge"
    assert record["endpoint"]["D"] == pytest.approx(0.0, abs=1e-6)
    assert record["slope"]["moved"]["count"] == 5
    assert record["boundary_all"]["count"] == 7


def test_confirm_requires_seal_and_does_not_open(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        phys002.confirm(tmp_path)
    assert not (tmp_path / "confirmation-opened.json").exists()


def test_reserved_config_is_disjoint_and_reserved_only() -> None:
    config = phys002._reserved_config([1, 2, 3])
    assert config.challenge_split_counts == (0, 0, 100)
    assert config.forbidden_reset_seeds == (1, 2, 3)
    assert config.horizon_policy == "fixed_two"
    variant = phys002._reserved_config([1], push=0.35)
    assert variant.challenge_push_magnitude == 0.35 and variant.ordinary_target == 0
    assert json.dumps(variant.forbidden_reset_seeds)
