"""Frozen, leakage-free LeWM inference for the audited PushT checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from jepa_action_effects.interface_audit import (
    CHECKPOINT_REVISION,
    CONFIG_SHA256,
    WEIGHTS_SHA256,
    sha256,
)

HISTORY_SIZE = 3
ACTION_BLOCK = 5
ACTION_DIM = 2
LATENT_DIM = 192
ARCHITECTURE_ID = "lewm-vit-tiny-p14-h3-ca231f9"


@dataclass(frozen=True)
class ActionScaler:
    """Per-coordinate action normalization recovered from the training archive."""

    mean: tuple[float, float]
    scale: tuple[float, float]
    convention: str = "evaluation_population_std"

    def __post_init__(self) -> None:
        values = np.asarray((*self.mean, *self.scale), dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Action scaler values must be finite two-coordinate statistics")
        if np.any(np.asarray(self.scale) <= 0):
            raise ValueError("Action scales must be positive")

    def transform(self, controls: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(controls, dtype=torch.float32).clone()
        if tensor.shape[-1:] != (ACTION_DIM,):
            raise ValueError("Raw controls must have final shape dimension 2")
        if not torch.isfinite(tensor).all():
            raise ValueError("Raw controls must be finite")
        if torch.any((tensor < -1) | (tensor > 1)):
            raise ValueError("Raw controls must remain within the simulator bounds [-1, 1]")
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        scale = torch.tensor(self.scale, dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean) / scale

    @classmethod
    def from_json(cls, path: Path) -> ActionScaler:
        payload = json.loads(path.read_text())
        stats = payload["evaluation_scaler"]
        return cls(tuple(stats["mean"]), tuple(stats["scale"]))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _validate_pixels(pixels: np.ndarray | torch.Tensor, *, history: bool) -> torch.Tensor:
    tensor = torch.as_tensor(pixels).clone()
    expected_time = HISTORY_SIZE if history else None
    if tensor.ndim != 5 or tensor.shape[2:] != (3, 224, 224):
        raise ValueError("Pixels must have shape (B,T,3,224,224)")
    if expected_time is not None and tensor.shape[1] != expected_time:
        raise ValueError(f"Context must contain exactly {HISTORY_SIZE} frames")
    if not tensor.is_floating_point():
        raise ValueError("Pixels must already be preprocessed floating-point tensors")
    tensor = tensor.to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(tensor).all():
        raise ValueError("Pixels must be finite")
    return tensor


def build_frozen_model(checkpoint_dir: Path) -> nn.Module:
    """Construct the source-audited architecture and strictly load the checkpoint."""
    from stable_worldmodel.wm.lewm import JEPA, MLP, ARPredictor, Embedder
    from transformers import ViTConfig, ViTModel

    if sha256(checkpoint_dir / "config.json") != CONFIG_SHA256:
        raise ValueError("Unexpected checkpoint config SHA-256")
    if sha256(checkpoint_dir / "weights.pt") != WEIGHTS_SHA256:
        raise ValueError("Unexpected checkpoint weights SHA-256")
    hub_config = json.loads((checkpoint_dir / "config.json").read_text())
    expected = {
        "size": "tiny",
        "patch_size": 14,
        "image_size": 224,
        "pretrained": False,
        "use_mask_token": False,
    }
    actual = {key: hub_config["encoder"][key] for key in expected}
    if actual != expected:
        raise ValueError(f"Unsupported encoder configuration: {actual}")

    encoder = ViTModel(
        ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=224,
            patch_size=14,
        ),
        add_pooling_layer=False,
        use_mask_token=False,
    )
    predictor_config = {
        key: value for key, value in hub_config["predictor"].items() if key != "_target_"
    }
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_config),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        projector=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d),
    )
    weights = torch.load(checkpoint_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen model invariant failed")
    return model


def encode_real(model: nn.Module, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Encode preprocessed real frames without accepting actions or future state."""
    pixels = _validate_pixels(frames, history=False)
    with torch.inference_mode():
        result = model.encode({"pixels": pixels})["emb"].detach().cpu()
    if result.shape != (pixels.shape[0], pixels.shape[1], LATENT_DIM):
        raise RuntimeError(f"Unexpected real embedding shape {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise RuntimeError("Real embeddings contain nonfinite values")
    return result


def _model_actions(
    history: np.ndarray | torch.Tensor,
    candidates: np.ndarray | torch.Tensor,
    scaler: ActionScaler,
) -> tuple[torch.Tensor, int]:
    past = torch.as_tensor(history, dtype=torch.float32).clone()
    future = torch.as_tensor(candidates, dtype=torch.float32).clone()
    if past.ndim != 3 or past.shape[1:] != (10, ACTION_DIM):
        raise ValueError("Past controls must have shape (B,10,2)")
    if future.ndim != 4 or future.shape[0] != past.shape[0] or future.shape[-1] != ACTION_DIM:
        raise ValueError("Candidate controls must have shape (B,S,5K,2)")
    if future.shape[2] == 0 or future.shape[2] % ACTION_BLOCK:
        raise ValueError(
            "Each candidate must contain a positive whole number of five-control blocks"
        )
    horizon = future.shape[2] // ACTION_BLOCK
    normalized_past = scaler.transform(past).reshape(past.shape[0], HISTORY_SIZE - 1, 10)
    normalized_future = scaler.transform(future).reshape(
        future.shape[0], future.shape[1], horizon, 10
    )
    history_by_candidate = normalized_past[:, None].expand(-1, future.shape[1], -1, -1)
    return torch.cat((history_by_candidate, normalized_future), dim=2), horizon


def predict_future(
    model: nn.Module,
    context_pixels: np.ndarray | torch.Tensor,
    past_controls: np.ndarray | torch.Tensor,
    candidate_controls: np.ndarray | torch.Tensor,
    scaler: ActionScaler,
) -> torch.Tensor:
    """Predict K future latents using only observed context and supplied actions."""
    context = _validate_pixels(context_pixels, history=True)
    actions, horizon = _model_actions(past_controls, candidate_controls, scaler)
    if actions.shape[0] != context.shape[0]:
        raise ValueError("Pixel and action batch sizes differ")
    samples = actions.shape[1]
    info = {"pixels": context[:, None].expand(-1, samples, -1, -1, -1, -1).clone()}
    with torch.inference_mode():
        output = model.rollout(info, actions.clone(), history_size=HISTORY_SIZE)
        predictions = output["predicted_emb"][:, :, HISTORY_SIZE:].detach().cpu()
    expected = (context.shape[0], samples, horizon, LATENT_DIM)
    if predictions.shape != expected:
        raise RuntimeError(
            f"Unexpected prediction shape {tuple(predictions.shape)}; expected {expected}"
        )
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Predicted embeddings contain nonfinite values")
    return predictions


def manual_predict_future(
    model: nn.Module,
    context_pixels: np.ndarray | torch.Tensor,
    past_controls: np.ndarray | torch.Tensor,
    candidate_controls: np.ndarray | torch.Tensor,
    scaler: ActionScaler,
) -> torch.Tensor:
    """Independent autoregressive implementation used to verify rollout alignment."""
    context = _validate_pixels(context_pixels, history=True)
    actions, horizon = _model_actions(past_controls, candidate_controls, scaler)
    batch, samples = actions.shape[:2]
    with torch.inference_mode():
        encoded = encode_real(model, context)
        embeddings = (
            encoded[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, 3, -1).clone()
        )
        flat_actions = actions.reshape(batch * samples, HISTORY_SIZE - 1 + horizon, 10)
        predicted = []
        for step in range(horizon):
            action_prefix = flat_actions[:, : HISTORY_SIZE + step]
            action_embeddings = model.action_encoder(action_prefix)
            next_embedding = model.predict(
                embeddings[:, -HISTORY_SIZE:], action_embeddings[:, -HISTORY_SIZE:]
            )[:, -1:]
            embeddings = torch.cat((embeddings, next_embedding), dim=1)
            predicted.append(next_embedding)
        result = torch.cat(predicted, dim=1).reshape(batch, samples, horizon, LATENT_DIM)
    return result.detach().cpu()


def model_fingerprint(checkpoint_dir: Path, scaler: ActionScaler) -> dict[str, Any]:
    import stable_worldmodel.wm.lewm as pinned_lewm

    return {
        "architecture": ARCHITECTURE_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "weights_sha256": sha256(checkpoint_dir / "weights.pt"),
        "config_sha256": sha256(checkpoint_dir / "config.json"),
        "scaler_sha256": scaler.fingerprint(),
        "precision": "float32",
        "device": "cpu",
        "history_frames": HISTORY_SIZE,
        "controls_per_block": ACTION_BLOCK,
        "action_order": "control-major flattened x,y pairs",
        "preprocessing": "rgb-float-imagenet-normalized-224",
        "inference_source_sha256": sha256(Path(__file__)),
        "pinned_lewm_source_sha256": sha256(Path(pinned_lewm.__file__)),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
    }
