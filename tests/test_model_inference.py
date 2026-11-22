"""Behavioral tests for MODEL-001 normalization, rollout isolation, and caching."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from jepa_action_effects.action_stats import compute_action_statistics
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    manual_predict_future,
    predict_future,
)


class FakeJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.encoded_shapes = []

    def encode(self, info):
        pixels = info["pixels"]
        self.encoded_shapes.append(tuple(pixels.shape))
        base = pixels.mean(dim=(2, 3, 4), keepdim=False)
        info["emb"] = base[..., None].expand(*base.shape, 192).clone()
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def action_encoder(self, actions):
        return actions.mean(-1, keepdim=True).expand(*actions.shape[:-1], 192)

    def predict(self, embeddings, action_embeddings):
        return embeddings + action_embeddings

    def rollout(self, info, action_sequence, history_size=3):
        context = self.encode({"pixels": info["pixels"][:, 0]})["emb"]
        batch, samples, total = action_sequence.shape[:3]
        embeddings = (
            context[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, 3, 192).clone()
        )
        actions = action_sequence.reshape(batch * samples, total, 10)
        for step in range(total - 2):
            encoded_actions = self.action_encoder(actions[:, : 3 + step])
            next_embedding = self.predict(embeddings[:, -3:], encoded_actions[:, -3:])[:, -1:]
            embeddings = torch.cat((embeddings, next_embedding), dim=1)
        info["predicted_emb"] = embeddings.reshape(batch, samples, total + 1, 192)
        return info


def test_real_checkpoint_load_is_strict_frozen_and_eval():
    checkpoint = Path("checkpoints/lewm-pusht")
    if not (checkpoint / "weights.pt").exists():
        pytest.skip("ignored checkpoint artifact is not installed")
    model = build_frozen_model(checkpoint)
    assert len(model.state_dict()) == 303
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_action_statistics_match_upstream_conventions_and_nan_filtering():
    actions = np.array([[0.0, 1.0], [1.0, 3.0], [np.nan, 9.0], [2.0, 5.0]])
    report = compute_action_statistics(actions)
    valid = actions[[0, 1, 3]]
    np.testing.assert_allclose(report["training_scaler"]["mean"], valid.mean(0))
    np.testing.assert_allclose(report["training_scaler"]["scale"], valid.std(0, ddof=1))
    np.testing.assert_allclose(report["evaluation_scaler"]["scale"], valid.std(0, ddof=0))
    assert report["rows_removed_nan"] == 1


@pytest.mark.parametrize(
    "actions",
    [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [np.inf, 1.0]])],
)
def test_action_statistics_reject_invalid_scale_or_values(actions):
    with pytest.raises(ValueError):
        compute_action_statistics(actions)


@pytest.mark.parametrize("horizon", [1, 2, 4])
@pytest.mark.parametrize("samples", [1, 6])
def test_rollout_matches_manual_and_encodes_only_context(horizon, samples):
    model = FakeJEPA().eval()
    scaler = ActionScaler((0.1, -0.2), (0.5, 0.25))
    context = torch.zeros(1, 3, 3, 224, 224)
    past = np.linspace(-0.5, 0.5, 20).reshape(1, 10, 2)
    future = np.linspace(-0.4, 0.4, samples * horizon * 10).reshape(1, samples, 5 * horizon, 2)
    expected = manual_predict_future(model, context, past, future, scaler)
    model.encoded_shapes.clear()
    actual = predict_future(model, context, past, future, scaler)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (1, samples, horizon, 192)
    assert model.encoded_shapes == [(1, 3, 3, 224, 224)]


def test_candidate_order_equivariance_and_input_immutability():
    model = FakeJEPA().eval()
    scaler = ActionScaler((0.0, 0.0), (1.0, 1.0))
    context = torch.zeros(1, 3, 3, 224, 224)
    past = np.zeros((1, 10, 2), dtype=np.float32)
    future = np.arange(120, dtype=np.float32).reshape(1, 6, 10, 2) / 120
    original = future.copy()
    forward = predict_future(model, context, past, future, scaler)
    reverse = predict_future(model, context, past, future[:, ::-1].copy(), scaler)
    torch.testing.assert_close(forward.flip(1), reverse)
    np.testing.assert_array_equal(future, original)


def test_validation_rejects_bad_inputs():
    model = FakeJEPA().eval()
    scaler = ActionScaler((0.0, 0.0), (1.0, 1.0))
    pixels = torch.zeros(1, 3, 3, 224, 224)
    past = np.zeros((1, 10, 2))
    with pytest.raises(ValueError, match="whole number"):
        predict_future(model, pixels, past, np.zeros((1, 1, 6, 2)), scaler)
    with pytest.raises(ValueError, match="bounds"):
        predict_future(model, pixels, past, np.full((1, 1, 5, 2), 1.01), scaler)
    with pytest.raises(ValueError, match="positive"):
        ActionScaler((0.0, 0.0), (0.0, 1.0))


def test_cache_round_trip_invalidation_and_corruption(tmp_path: Path):
    cache = LatentCache(tmp_path)
    inputs = {"pixels": np.zeros((1, 3), dtype=np.float32)}
    fingerprint = {"checkpoint": "a", "scaler": "b", "horizon": 2}
    key = cache.key("real", inputs, fingerprint)
    path = cache.write("real", key, np.zeros((1, 2, 192)), fingerprint, {"episode": 4})
    values, manifest = cache.read("real", key, fingerprint)
    assert values.dtype == np.float32 and manifest["identifiers"] == {"episode": 4}
    assert cache.key("real", {"pixels": np.ones((1, 3))}, fingerprint) != key
    assert cache.key("real", inputs, {**fingerprint, "scaler": "changed"}) != key
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Invalid cache"):
        cache.read("real", key, fingerprint)


def test_cache_manifest_rejects_fingerprint_mismatch(tmp_path: Path):
    cache = LatentCache(tmp_path)
    fingerprint = {"model": "one"}
    key = cache.key("prediction", {"actions": np.zeros(2)}, fingerprint)
    cache.write("prediction", key, np.zeros((1, 1, 192)), fingerprint)
    with pytest.raises(ValueError, match="fingerprint"):
        cache.read("prediction", key, {"model": "two"})


def test_cache_manifest_is_json_serializable(tmp_path: Path):
    cache = LatentCache(tmp_path)
    fingerprint = {"model": "one"}
    key = cache.key("real", {"frame": np.zeros(1)}, fingerprint)
    cache.write("real", key, np.zeros((1, 192)), fingerprint)
    _, manifest = cache.read("real", key, fingerprint)
    json.dumps(manifest)
