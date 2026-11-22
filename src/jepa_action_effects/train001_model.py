"""TRAIN-001 fine-tuning: trainable boundary, differentiable rollout and training loop.

- Trainable: `predictor`, `action_encoder`, `pred_proj` (11,740,334 parameters).
  Frozen: visual encoder, visual projector, action normalization, readouts.
- Modes: the whole model is in eval mode and only the predictor is switched to train
  mode, so predictor dropout is active while both BatchNorm layers (`projector.net.1`,
  `pred_proj.net.1`) keep their frozen running statistics. `pred_proj`'s weights and BN
  affine parameters remain trainable.
- Rollout from cached real history embeddings. Step 1 uses real history
  `[h0,h1,h2]` with action blocks `[p0,p1,c0]`; step 2 uses `[h1,h2,z3_hat]` with
  `[p1,c0,c1]`. The first prediction is fed back (gradient flows through it); future
  targets enter only the loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from jepa_action_effects.train001_budget import Budget, BudgetStop, NullBudget
from jepa_action_effects.train001_protocol import (
    EXPECTED_TRAINABLE_PARAMETERS,
    SCENES,
    TRAINABLE_MODULES,
)

HISTORY = 3
LATENT = 192


# --------------------------------------------------------------------------- boundary
def trainable_prefixes() -> tuple[str, ...]:
    return tuple(f"{name}." for name in TRAINABLE_MODULES)


def configure_trainable(model: nn.Module) -> list[nn.Parameter]:
    """Enable gradients on the selected modules only and set training modes."""
    model.requires_grad_(False)
    for name in TRAINABLE_MODULES:
        getattr(model, name).requires_grad_(True)
    set_training_modes(model)
    return trainable_parameters(model)


def set_training_modes(model: nn.Module) -> None:
    model.eval()
    model.predictor.train()


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(trainable_prefixes())
    ]


def count_trainable(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in trainable_parameters(model))


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith(trainable_prefixes())
    }


def frozen_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor that must never change: frozen weights and all BN buffers."""
    state = model.state_dict()
    return {
        key: value.detach().clone()
        for key, value in state.items()
        if not key.startswith(trainable_prefixes())
        or "running_" in key
        or "num_batches_tracked" in key
    }


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


# --------------------------------------------------------------------------- rollout
def normalized_actions(scaler, past: np.ndarray, candidates: np.ndarray) -> torch.Tensor:
    """(N,10,2) past and (N,B,10,2) candidate controls -> (N,B,4,10) model actions."""
    from jepa_action_effects.model_inference import _model_actions

    actions, horizon = _model_actions(past, candidates, scaler)
    if horizon != 2:
        raise ValueError("TRAIN-001 uses exactly two future action blocks")
    return actions


def rollout(model: nn.Module, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Differentiable two-step autoregressive prediction.

    history: (N,3,192) real observed embeddings; actions: (N,4,10) normalized blocks
    [p0,p1,c0,c1]. Returns (N,2,192) predictions for t+5 and t+10.
    """
    if history.shape[1:] != (HISTORY, LATENT) or actions.shape[1:] != (4, 10):
        raise ValueError(f"Unexpected shapes {tuple(history.shape)} {tuple(actions.shape)}")
    action_embeddings = model.action_encoder(actions)
    first = model.predict(history, action_embeddings[:, 0:3])[:, -1:]
    context = torch.cat((history[:, 1:], first), dim=1)
    second = model.predict(context, action_embeddings[:, 1:4])[:, -1:]
    return torch.cat((first, second), dim=1)


def rollout_families(model: nn.Module, history: torch.Tensor, actions: torch.Tensor):
    """history (F,3,192), actions (F,B,4,10) -> predictions (F,B,2,192)."""
    families, branches = actions.shape[:2]
    expanded = history[:, None].expand(-1, branches, -1, -1).reshape(-1, HISTORY, LATENT)
    predicted = rollout(model, expanded, actions.reshape(-1, 4, 10))
    return predicted.reshape(families, branches, 2, LATENT)


# --------------------------------------------------------------------------- data
@dataclass
class FamilyTensors:
    ids: list[str]
    scenes: np.ndarray  # (F,) index into SCENES
    history: torch.Tensor  # (F,3,192)
    targets: torch.Tensor  # (F,4,2,192)
    actions: torch.Tensor  # (F,4,4,10)

    def subset(self, indices: Sequence[int]) -> FamilyTensors:
        index = torch.as_tensor(list(indices), dtype=torch.long)
        return FamilyTensors(
            [self.ids[i] for i in indices],
            self.scenes[list(indices)],
            self.history[index],
            self.targets[index],
            self.actions[index],
        )


def family_tensors(records: list[dict[str, Any]], scaler) -> FamilyTensors:
    """Build tensors from cache arrays (history_z, future_z, past_controls, candidate_controls)."""
    history = np.stack([record["history_z"] for record in records]).astype(np.float32)
    targets = np.stack([record["future_z"] for record in records]).astype(np.float32)
    past = np.stack([record["past_controls"] for record in records])
    candidates = np.stack([record["candidate_controls"] for record in records])
    return FamilyTensors(
        [record["id"] for record in records],
        np.asarray([SCENES.index(record["scene"]) for record in records]),
        torch.from_numpy(history),
        torch.from_numpy(targets),
        normalized_actions(scaler, past, candidates).float(),
    )


# --------------------------------------------------------------------------- losses
def branch_losses(predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(F,B,2,192) -> (F,B) latent MSE averaged over steps and coordinates."""
    return ((predicted - targets) ** 2).mean(dim=(2, 3))


def weighted_loss(losses: torch.Tensor, weights: torch.Tensor, scenes: np.ndarray) -> torch.Tensor:
    """Magnitude-weight within family, average families within scene, then scenes."""
    per_family = losses @ weights
    scene_means = [per_family[torch.as_tensor(scenes == s)].mean() for s in np.unique(scenes)]
    return torch.stack(scene_means).mean()


def equal_weights(count: int = 4) -> torch.Tensor:
    return torch.full((count,), 1.0 / count)


def latent_mse(model: nn.Module, data: FamilyTensors, batch: int = 64) -> float:
    """Equally weighted validation latent MSE (families, scenes, magnitudes, steps)."""
    was_training = model.predictor.training
    model.eval()
    try:
        with torch.no_grad():
            losses = torch.cat(
                [
                    branch_losses(
                        rollout_families(
                            model, data.history[i : i + batch], data.actions[i : i + batch]
                        ),
                        data.targets[i : i + batch],
                    )
                    for i in range(0, len(data.ids), batch)
                ]
            )
        return float(weighted_loss(losses, equal_weights(losses.shape[1]), data.scenes))
    finally:
        if was_training:
            set_training_modes(model)


def predict(model: nn.Module, history: torch.Tensor, actions: torch.Tensor, batch: int = 64):
    """Evaluation-mode predictions (F,B,2,192) as float32 numpy."""
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                rollout_families(model, history[i : i + batch], actions[i : i + batch])
                for i in range(0, len(history), batch)
            ]
        ).numpy()


# --------------------------------------------------------------------------- sampler
class PairedSampler:
    """Per-scene epoch permutations, shared across both conditions of a training seed.

    Wrap rule: when a batch spans an epoch boundary, families already in the partial
    batch are moved to the end of the new permutation, so no family repeats within a
    batch and each is seen once per epoch.
    """

    def __init__(self, members: dict[int, list[int]], per_scene: int, seed: int):
        self.per_scene = per_scene
        self.members = {scene: list(indices) for scene, indices in members.items()}
        for scene, indices in self.members.items():
            if len(indices) < per_scene:
                raise ValueError(f"Scene {scene} has fewer than {per_scene} families")
        self.rngs = {
            scene: np.random.default_rng(np.random.SeedSequence([seed, scene]))
            for scene in self.members
        }
        self.permutations = {scene: self._permute(scene) for scene in self.members}
        self.positions = dict.fromkeys(self.members, 0)

    def _permute(self, scene: int) -> list[int]:
        return [
            self.members[scene][i] for i in self.rngs[scene].permutation(len(self.members[scene]))
        ]

    def _scene_batch(self, scene: int) -> list[int]:
        permutation, position = self.permutations[scene], self.positions[scene]
        taken = permutation[position : position + self.per_scene]
        if len(taken) < self.per_scene:
            fresh = self._permute(scene)
            fresh = [i for i in fresh if i not in taken] + [i for i in fresh if i in taken]
            needed = self.per_scene - len(taken)
            taken = taken + fresh[:needed]
            self.permutations[scene], self.positions[scene] = fresh, needed
        else:
            self.positions[scene] = position + self.per_scene
        return taken

    def next_batch(self) -> list[int]:
        return [index for scene in sorted(self.members) for index in self._scene_batch(scene)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "permutations": {str(k): v for k, v in self.permutations.items()},
            "positions": {str(k): v for k, v in self.positions.items()},
            "rngs": {str(k): rng.bit_generator.state for k, rng in self.rngs.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in self.members:
            self.permutations[key] = list(state["permutations"][str(key)])
            self.positions[key] = int(state["positions"][str(key)])
            self.rngs[key].bit_generator.state = state["rngs"][str(key)]


def scene_members(data: FamilyTensors) -> dict[int, list[int]]:
    return {
        int(scene): [int(i) for i in np.flatnonzero(data.scenes == scene)]
        for scene in np.unique(data.scenes)
    }


# --------------------------------------------------------------------------- checkpoints
def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    from jepa_action_effects.pilot_collection import sha256_file

    return sha256_file(path)


def load_trained(model: nn.Module, path: Path, expected_sha256: str | None = None) -> nn.Module:
    """Load a trained trainable-module state into a freshly built frozen model."""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"Checkpoint bytes changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError("Checkpoint keys differ from the trainable boundary")
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False)
    return model


# --------------------------------------------------------------------------- training
@dataclass
class RunSpec:
    condition: str
    seed: int
    learning_rate: float
    updates: int
    weights: tuple[float, ...]
    weight_decay: float
    gradient_clip: float
    per_scene: int
    threads: int
    checkpoint_updates: tuple[int, ...] = ()
    resume_every: int = 0
    curve_every: int = 10
    validation_every: int = 0


def train_run(
    model_factory: Callable[[], nn.Module],
    base_digest: str,
    data: FamilyTensors,
    spec: RunSpec,
    run_dir: Path | None = None,
    validation: FamilyTensors | None = None,
    budget: Budget | None = None,
    eval_every: int = 0,
    eval_data: FamilyTensors | None = None,
    expected_parameters: int | None = EXPECTED_TRAINABLE_PARAMETERS,
) -> dict[str, Any]:
    """Train one condition/seed from a fresh checkpoint, resuming if a resume state exists.

    `eval_every`/`eval_data` record the eval-mode weighted training objective on a fixed
    set (used by the overfit diagnostic).
    """
    budget = budget or NullBudget()
    torch.set_num_threads(spec.threads)
    model = model_factory()
    if state_digest(trainable_state(model)) != base_digest:
        raise RuntimeError("Training did not start from the original checkpoint")
    parameters = configure_trainable(model)
    if expected_parameters is not None and count_trainable(model) != expected_parameters:
        raise RuntimeError("Unexpected trainable-parameter count")
    frozen_before = frozen_state(model)
    optimizer = torch.optim.AdamW(parameters, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    sampler = PairedSampler(scene_members(data), spec.per_scene, spec.seed)
    weights = torch.tensor(spec.weights, dtype=torch.float32)
    torch.manual_seed(spec.seed)
    start, curve, validation_curve, eval_curve = 0, [], [], []
    resume_path = run_dir / "resume.pt" if run_dir is not None else None
    if resume_path is not None and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["spec"] != spec.__dict__:
            raise ValueError("Resume state belongs to a different run specification")
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        start = int(state["update"])
        curve, validation_curve = state["curve"], state["validation_curve"]
        eval_curve = state["eval_curve"]
    set_training_modes(model)

    def save_resume(update: int) -> None:
        if resume_path is None:
            return
        _atomic_torch_save(
            resume_path,
            {
                "spec": spec.__dict__,
                "model": trainable_state(model),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "update": update,
                "curve": curve,
                "validation_curve": validation_curve,
                "eval_curve": eval_curve,
            },
        )

    if eval_every and start == 0:
        eval_curve.append([0, _eval_objective(model, eval_data, weights)])
    checkpoints: dict[int, str] = {}
    for update in range(start, spec.updates):
        try:
            budget.unit("update")
        except BudgetStop:
            save_resume(update)
            raise
        began = time.monotonic()
        batch = sampler.next_batch()
        predicted = rollout_families(model, data.history[batch], data.actions[batch])
        loss = weighted_loss(
            branch_losses(predicted, data.targets[batch]), weights, data.scenes[batch]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip)
        optimizer.step()
        budget.record_unit("update", time.monotonic() - began)
        done = update + 1
        if done % spec.curve_every == 0 or done == 1:
            curve.append([done, float(loss.detach()), float(norm)])
        if validation is not None and spec.validation_every and done % spec.validation_every == 0:
            validation_curve.append([done, latent_mse(model, validation)])
        if eval_every and done % eval_every == 0:
            eval_curve.append([done, _eval_objective(model, eval_data, weights)])
        if run_dir is not None and done in spec.checkpoint_updates:
            path = run_dir / f"checkpoint-{done:05d}.pt"
            _atomic_torch_save(path, {"state": trainable_state(model), "update": done})
            checkpoints[done] = file_sha256(path)
        if spec.resume_every and done % spec.resume_every == 0:
            save_resume(done)
    if not states_equal(frozen_before, frozen_state(model)):
        raise RuntimeError("A frozen parameter or BatchNorm statistic changed")
    if run_dir is not None:
        for done in spec.checkpoint_updates:
            path = run_dir / f"checkpoint-{done:05d}.pt"
            checkpoints[done] = file_sha256(path)
    return {
        "model": model,
        "curve": curve,
        "validation_curve": validation_curve,
        "eval_curve": eval_curve,
        "checkpoints": {str(k): v for k, v in sorted(checkpoints.items())},
        "final_state_digest": state_digest(trainable_state(model)),
    }


def _eval_objective(model: nn.Module, data: FamilyTensors, weights: torch.Tensor) -> float:
    model.eval()
    try:
        with torch.no_grad():
            losses = branch_losses(
                rollout_families(model, data.history, data.actions), data.targets
            )
            return float(weighted_loss(losses, weights, data.scenes))
    finally:
        set_training_modes(model)


# --------------------------------------------------------------------------- checks
def implementation_checks(
    model_factory: Callable[[], nn.Module],
    base_digest: str,
    data: FamilyTensors,
    steps: int = 3,
    expected_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> dict[str, Any]:
    """Boundary, frozen tensors, finiteness and gradient checks on real batches."""
    model = model_factory()
    fresh = state_digest(trainable_state(model)) == base_digest
    parameters = configure_trainable(model)
    count = count_trainable(model)
    frozen_before = frozen_state(model)
    trainable_before = trainable_state(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    members = scene_members(data)
    batch = [i for scene in sorted(members) for i in members[scene][:4]]
    finite_outputs, gradients_ok = True, True
    for _ in range(steps):
        predicted = rollout_families(model, data.history[batch], data.actions[batch])
        finite_outputs &= bool(torch.isfinite(predicted).all())
        loss = weighted_loss(
            branch_losses(predicted, data.targets[batch]), equal_weights(), data.scenes[batch]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in parameters]
        gradients_ok &= all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        gradients_ok &= any(float(g.abs().sum()) > 0 for g in gradients if g is not None)
        optimizer.step()
    frozen_unchanged = states_equal(frozen_before, frozen_state(model))
    trainable_changed = not states_equal(trainable_before, trainable_state(model))
    no_grad_on_frozen = all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith(trainable_prefixes())
    )
    passed = (
        fresh
        and count == expected_parameters
        and finite_outputs
        and gradients_ok
        and frozen_unchanged
        and trainable_changed
        and no_grad_on_frozen
    )
    return {
        "passed": bool(passed),
        "started_from_base": fresh,
        "trainable_parameters": count,
        "finite_outputs": finite_outputs,
        "gradients_finite_nonzero": gradients_ok,
        "frozen_and_batchnorm_unchanged": frozen_unchanged,
        "trainable_changed": trainable_changed,
        "no_gradient_on_frozen": no_grad_on_frozen,
    }


def rollout_equivalence(model: nn.Module, scaler, families: list[dict], data: FamilyTensors):
    """Max |cached rollout - pixel-path predict_future| for the frozen model (eval)."""
    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import predict_future

    cached = predict(model, data.history, data.actions)
    errors = []
    for index, family in enumerate(families):
        pixel = predict_future(
            model,
            preprocess_pixels(family["history_frames"]).unsqueeze(0),
            family["warmup_controls"][-10:][None],
            family["candidate_controls"][None],
            scaler,
        ).numpy()[0]
        errors.append(float(np.max(np.abs(pixel - cached[index]))))
    return {"max_abs": max(errors), "families": len(errors)}


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=float))
