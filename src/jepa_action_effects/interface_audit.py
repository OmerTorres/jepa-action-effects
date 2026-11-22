"""ENV-001 source-pinned simulator/shape audit; never runs model inference."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np

SWM_REVISION = "44c45bd1f774995aac0cc2b41444ebea24c1d5ad"
CHECKPOINT_REVISION = "22b330c28c27ead4bfd1888615af1340e3fe9052"
WEIGHTS_SHA256 = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG_SHA256 = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
ACTION_BLOCK = 5
HISTORY_SIZE = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env():
    """Construct the audited simulator without changing upstream physics."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_worldmodel.envs.pusht.env import PushT

    provenance = json.loads(distribution("stable-worldmodel").read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != SWM_REVISION:
        raise RuntimeError("Simulator revision differs from the audited contract")
    return PushT(resolution=224, relative=True, render_action=False, with_target=True)


def preprocess_pixels(frames: np.ndarray):
    """Apply upstream eval's RGB/float/ImageNet/resize order to THWC uint8."""
    import torch
    from torchvision.transforms import v2

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB frames shaped (time, height, width, 3)")
    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Resize(224, antialias=True),
        ]
    )
    return transform(torch.from_numpy(frames.copy()).permute(0, 3, 1, 2))


def ordinary_transition(seed: int = 42, future_blocks: int = 1) -> dict:
    """Return past context, raw actions, and a separately held future target.

    Raw actions are deliberately NOT predictor inputs until training-data scaling
    is recovered. This checks shapes/alignment, not checkpoint compatibility.
    """
    env = make_env()
    try:
        observation, _ = env.reset(seed=seed)
        frames = [env.render()]
        states = [observation["state"].copy()]
        # Three distinct blocks make ordering visible; all controls are in bounds.
        if future_blocks not in (1, 2):
            raise ValueError("Ordinary smoke supports one or two future blocks")
        block_actions = [[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0], [0.0, -0.2]]
        controls = np.repeat(
            np.array(block_actions[: HISTORY_SIZE - 1 + future_blocks], dtype=np.float32),
            ACTION_BLOCK,
            axis=0,
        )
        flags = []
        for index, action in enumerate(controls, start=1):
            observation, _, terminated, truncated, _ = env.step(action)
            flags.append([bool(terminated), bool(truncated)])
            if terminated or truncated:
                raise RuntimeError(f"Smoke trajectory ended at control {index}")
            if index % ACTION_BLOCK == 0:
                frames.append(env.render())
                states.append(observation["state"].copy())
        frames = np.stack(frames)
        # Future frames never enter context preprocessing.
        context = preprocess_pixels(frames[:HISTORY_SIZE]).unsqueeze(0)
        target = preprocess_pixels(frames[HISTORY_SIZE:]).unsqueeze(0)
        blocks = controls.reshape(HISTORY_SIZE - 1 + future_blocks, 2 * ACTION_BLOCK)
        return {
            "context_pixels": context,
            "target_pixels": target,
            "action_history_raw": blocks[: HISTORY_SIZE - 1].copy(),
            "candidate_actions_raw": blocks[HISTORY_SIZE - 1 :].copy(),
            "states": np.stack(states),
            "frames": frames,
            "termination_flags": flags,
            "physics_dt": env.dt,
            "control_hz": env.control_hz,
        }
    finally:
        env.close()


def audit_checkpoint(directory: Path) -> dict:
    """Verify bytes and inspect tensors safely; do not instantiate a model."""
    import torch

    for name, expected in [("config.json", CONFIG_SHA256), ("weights.pt", WEIGHTS_SHA256)]:
        if sha256(directory / name) != expected:
            raise ValueError(f"Unexpected {name} SHA-256")
    config = json.loads((directory / "config.json").read_text())
    weights = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    expected_shapes = {
        "encoder.embeddings.cls_token": [1, 1, 192],
        "encoder.embeddings.position_embeddings": [1, 257, 192],
        "encoder.embeddings.patch_embeddings.projection.weight": [192, 3, 14, 14],
        "predictor.pos_embedding": [1, 3, 192],
        "action_encoder.patch_embed.weight": [10, 10, 1],
    }
    for key, expected in expected_shapes.items():
        if list(weights[key].shape) != expected:
            raise ValueError(f"Unexpected tensor shape for {key}")
    if not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("Nonfinite checkpoint tensor")
    return {
        "repository": "quentinll/lewm-pusht",
        "revision": CHECKPOINT_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "config_sha256": CONFIG_SHA256,
        "weights_bytes": (directory / "weights.pt").stat().st_size,
        "state_dict_entries": len(weights),
        "tensor_elements_including_buffers": sum(value.numel() for value in weights.values()),
        "checked_tensor_shapes": expected_shapes,
        "config": config,
        "all_tensors_finite": True,
        "strict_model_load_performed": False,
        "inference_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lewm-pusht"))
    parser.add_argument("--output", type=Path, default=Path("outputs/env001/report.json"))
    args = parser.parse_args()
    transition = ordinary_transition()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(np.concatenate(transition["frames"], axis=1)).save(
        args.output.parent / "ordinary-transition.png"
    )
    packages = [
        "stable-worldmodel",
        "torch",
        "torchvision",
        "numpy",
        "gymnasium",
        "pymunk",
        "pygame",
        "opencv-python-headless",
        "shapely",
        "h5py",
        "hdf5plugin",
    ]
    report = {
        "scope": "Simulator transition and checkpoint tensor audit; no model inference",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "uv_lock_sha256": sha256(Path("uv.lock")),
        "simulator_revision": SWM_REVISION,
        "checkpoint": audit_checkpoint(args.checkpoint_dir),
        "transition": {
            "seed": 42,
            "context_control_indices": [0, 5, 10],
            "target_control_indices": [15],
            "context_shape": list(transition["context_pixels"].shape),
            "target_shape": list(transition["target_pixels"].shape),
            "action_history_raw": transition["action_history_raw"].tolist(),
            "candidate_actions_raw": transition["candidate_actions_raw"].tolist(),
            "states": transition["states"].tolist(),
            "termination_flags": transition["termination_flags"],
            "physics_dt": transition["physics_dt"],
            "control_hz": transition["control_hz"],
            "native_prediction_seconds": ACTION_BLOCK / transition["control_hz"],
            "frame_sha256": [hashlib.sha256(f.tobytes()).hexdigest() for f in transition["frames"]],
            "action_normalization": "UNRESOLVED: raw controls only; not inference-ready",
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}; no checkpoint inference was performed.")


if __name__ == "__main__":
    main()
