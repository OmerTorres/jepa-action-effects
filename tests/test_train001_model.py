"""TRAIN-001 training boundary, rollout, loss, sampler and resume on a tiny real-class JEPA."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_worldmodel.wm.lewm import JEPA, MLP, ARPredictor, Embedder
from torch import nn

from jepa_action_effects.model_inference import ActionScaler, predict_future
from jepa_action_effects.train001_budget import BudgetStop, NullBudget
from jepa_action_effects.train001_model import (
    FamilyTensors,
    PairedSampler,
    RunSpec,
    branch_losses,
    configure_trainable,
    count_trainable,
    frozen_state,
    implementation_checks,
    normalized_actions,
    predict,
    rollout,
    rollout_families,
    state_digest,
    states_equal,
    train_run,
    trainable_state,
    weighted_loss,
)

SCALER = ActionScaler((0.01, -0.02), (0.2, 0.25))


class _Output:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class PixelEncoder(nn.Module):
    """Stand-in ViT: pooled pixels -> (N, 2, 192) hidden states."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 192)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(2, 3))
        hidden = torch.tanh(self.linear(pooled))
        return _Output(torch.stack((hidden, -hidden), dim=1))


def tiny_model(seed: int = 0) -> JEPA:
    torch.manual_seed(seed)
    model = JEPA(
        encoder=PixelEncoder(),
        predictor=ARPredictor(
            num_frames=3,
            depth=1,
            heads=2,
            mlp_dim=32,
            input_dim=192,
            hidden_dim=192,
            dim_head=8,
            dropout=0.1,
        ),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        projector=MLP(192, 16, 192, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(192, 16, 192, norm_fn=nn.BatchNorm1d),
    )
    # AdaLN-zero blocks start at zero (actions would have no effect): randomize weights.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    # Non-trivial BN statistics so that train-mode BN would visibly change them.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval().requires_grad_(False)


def data(families: int = 8, seed: int = 1) -> FamilyTensors:
    rng = np.random.default_rng(seed)
    past = rng.uniform(-0.3, 0.3, (families, 10, 2)).astype(np.float32)
    candidates = rng.uniform(-0.6, 0.6, (families, 4, 10, 2)).astype(np.float32)
    return FamilyTensors(
        [f"f{i}" for i in range(families)],
        np.asarray([i % 2 for i in range(families)]),
        torch.from_numpy(rng.normal(size=(families, 3, 192)).astype(np.float32)),
        torch.from_numpy(rng.normal(size=(families, 4, 2, 192)).astype(np.float32)),
        normalized_actions(SCALER, past, candidates).float(),
    )


def spec(**changes) -> RunSpec:
    values = {
        "condition": "balanced",
        "seed": 5,
        "learning_rate": 1e-3,
        "updates": 6,
        "weights": (0.25, 0.25, 0.25, 0.25),
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "per_scene": 2,
        "threads": 1,
    }
    values.update(changes)
    return RunSpec(**values)


BASE = state_digest(trainable_state(tiny_model()))


def test_only_intended_parameters_change_and_batchnorm_is_frozen():
    frozen = frozen_state(tiny_model())
    before = trainable_state(tiny_model())
    result = train_run(tiny_model, BASE, data(), spec(), expected_parameters=None)
    model = result["model"]
    assert states_equal(frozen, frozen_state(model))
    assert not states_equal(before, trainable_state(model))
    changed = [k for k in before if not torch.equal(before[k], trainable_state(model)[k])]
    assert all(k.startswith(("predictor.", "action_encoder.", "pred_proj.")) for k in changed)
    assert not any("running_" in k for k in changed)


def test_modes_enable_predictor_dropout_only():
    model = tiny_model()
    parameters = configure_trainable(model)
    assert model.predictor.training
    assert not model.pred_proj.training and not model.projector.training
    assert not model.encoder.training
    assert sum(p.numel() for p in parameters) == count_trainable(model)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.projector.parameters())


def test_future_targets_cannot_change_predictions():
    model = tiny_model()
    first = data(seed=3)
    changed = FamilyTensors(
        first.ids, first.scenes, first.history, first.targets * 100 + 7, first.actions
    )
    assert np.array_equal(
        predict(model, first.history, first.actions),
        predict(model, changed.history, changed.actions),
    )


def test_rollout_matches_upstream_pixel_path_and_action_alignment():
    model = tiny_model()
    rng = np.random.default_rng(4)
    pixels = torch.from_numpy(rng.normal(size=(2, 3, 3, 224, 224)).astype(np.float32))
    past = rng.uniform(-0.3, 0.3, (2, 10, 2)).astype(np.float32)
    candidates = rng.uniform(-0.6, 0.6, (2, 4, 10, 2)).astype(np.float32)
    upstream = predict_future(model, pixels, past, candidates, SCALER).numpy()
    with torch.no_grad():
        history = model.encode({"pixels": pixels})["emb"]
    ours = predict(model, history, normalized_actions(SCALER, past, candidates).float())
    assert np.max(np.abs(upstream - ours)) < 1e-5
    swapped = candidates.copy()
    swapped[:, :, :5], swapped[:, :, 5:] = candidates[:, :, 5:], candidates[:, :, :5]
    other = predict(model, history, normalized_actions(SCALER, past, swapped).float())
    assert not np.allclose(ours[:, :, 0], other[:, :, 0])


def test_gradient_flows_through_first_prediction():
    model = tiny_model()
    configure_trainable(model)
    model.eval()
    history = torch.randn(3, 3, 192, requires_grad=True)
    actions = torch.randn(3, 4, 10)
    second = rollout(model, history, actions)[:, 1]
    second.sum().backward()
    # h0 enters only the first step; a gradient on it from the second output proves the
    # second prediction consumed the first predicted embedding.
    assert history.grad[:, 0].abs().sum() > 0


def test_weighted_loss_matches_analytic_fixture():
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 8.0], [4.0, 4.0, 4.0, 4.0]])
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    scenes = np.array([0, 0, 1])
    # family: 3.0, 3.2, 4.0 -> scene means 3.1 and 4.0 -> 3.55
    assert float(weighted_loss(losses, weights, scenes)) == pytest.approx(3.55)
    predicted = torch.zeros(1, 4, 2, 3)
    target = torch.ones(1, 4, 2, 3) * torch.tensor([1.0, 2.0, 3.0, 4.0])[None, :, None, None]
    assert branch_losses(predicted, target).tolist() == [[1.0, 4.0, 9.0, 16.0]]


def test_sampler_is_paired_and_wraps_without_duplicates():
    members = {0: list(range(10)), 1: list(range(10, 20))}
    left, right = PairedSampler(members, 8, 7), PairedSampler(members, 8, 7)
    left_batches = [left.next_batch() for _ in range(12)]
    assert left_batches == [right.next_batch() for _ in range(12)]
    for batch in left_batches:
        assert len(batch) == len(set(batch)) == 16
    stream = [i for batch in left_batches for i in batch if i < 10]
    assert sorted(stream[:10]) == list(range(10))


def test_both_arms_receive_identical_batches_and_updates(monkeypatch):
    seen: dict[str, list] = {"balanced": [], "archive_weighted": []}
    original = PairedSampler.next_batch

    def recording(self):
        batch = original(self)
        seen[current].append(list(batch))
        return batch

    monkeypatch.setattr(PairedSampler, "next_batch", recording)
    for current in ("balanced", "archive_weighted"):
        weights = (0.25,) * 4 if current == "balanced" else (0.4, 0.3, 0.2, 0.1)
        train_run(
            tiny_model,
            BASE,
            data(),
            spec(condition=current, weights=weights),
            expected_parameters=None,
        )
    assert seen["balanced"] == seen["archive_weighted"]
    assert len(seen["balanced"]) == 6


def test_training_refuses_a_non_original_start():
    def leaked():
        model = tiny_model()
        with torch.no_grad():
            model.predictor.pos_embedding.add_(1.0)
        return model

    with pytest.raises(RuntimeError, match="original checkpoint"):
        train_run(leaked, BASE, data(), spec(), expected_parameters=None)


class StopAt(NullBudget):
    def __init__(self, stop: int):
        super().__init__()
        self.calls, self.stop = 0, stop

    def unit(self, kind, estimate=None):
        if kind == "update":
            if self.calls == self.stop:
                raise BudgetStop("test stop")
            self.calls += 1


def test_save_resume_reproduces_uninterrupted_run(tmp_path):
    run = spec(updates=10, resume_every=5, checkpoint_updates=(10,))
    full = train_run(tiny_model, BASE, data(), run, tmp_path / "full", expected_parameters=None)
    with pytest.raises(BudgetStop):
        train_run(
            tiny_model,
            BASE,
            data(),
            run,
            tmp_path / "split",
            budget=StopAt(7),
            expected_parameters=None,
        )
    saved = torch.load(tmp_path / "split" / "resume.pt", weights_only=False)
    assert saved["update"] == 7
    counter = StopAt(10**6)
    resumed = train_run(
        tiny_model, BASE, data(), run, tmp_path / "split", budget=counter, expected_parameters=None
    )
    assert counter.calls == 3  # only the remaining updates ran
    assert resumed["final_state_digest"] == full["final_state_digest"]
    assert resumed["checkpoints"] == full["checkpoints"]


def test_resume_rejects_a_different_specification(tmp_path):
    run = spec(updates=10, resume_every=5)
    with pytest.raises(BudgetStop):
        train_run(
            tiny_model, BASE, data(), run, tmp_path, budget=StopAt(6), expected_parameters=None
        )
    with pytest.raises(ValueError, match="different run"):
        train_run(
            tiny_model,
            BASE,
            data(),
            spec(updates=10, resume_every=5, learning_rate=0.5),
            tmp_path,
            expected_parameters=None,
        )


def test_implementation_checks_pass_on_tiny_model():
    model = tiny_model()
    count = count_trainable(model)
    report = implementation_checks(tiny_model, BASE, data(), expected_parameters=count)
    assert report["passed"], report


def test_rollout_families_shapes():
    model = tiny_model()
    batch = data(4)
    predicted = rollout_families(model, batch.history, batch.actions)
    assert predicted.shape == (4, 4, 2, 192)
