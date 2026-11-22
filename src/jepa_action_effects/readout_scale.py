"""READOUT-002: a readout trained on enough block poses to measure its own floor.

    python -m jepa_action_effects.readout_scale collect   # render + encode, resumable shards
    python -m jepa_action_effects.readout_scale fit       # ridge/MLP probes, learning curve
    python -m jepa_action_effects.readout_scale report

Frames are rendered with the validated render-only pose setter (angle before position,
shape reindexing) from uniformly sampled, geometry-valid agent/block poses. Labels are
the exact body poses. No study family, predictor output or reserved label is used for
fitting or selection; the 60 READOUT tuning families are an evaluation-only check of
transfer to real study frames. This stage is descriptive: there is no pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    FRESH,
    MANIFESTS,
    Stopwatch,
    collection_manifest,
    encode_frames,
    load_family,
    load_model,
    runtime_info,
    seal,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file

ROOT = Path("outputs/readout002")
SEED = 20261003
SHARD = 1000
SPLITS = {"train": 50, "validation": 10}  # shards of SHARD frames
SENSITIVITY_BASES = 300
SENSITIVITY_SHIFTS = (0.5, 1.0, 2.0, 5.0)
LEARNING_CURVE = (1000, 2000, 5000, 10000, 20000, 50000)
RIDGE_ALPHAS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
MLP_SEEDS = (20261004, 20261005, 20261006)
BLOCK_RANGE = (60.0, 452.0)
AGENT_RANGE = (30.0, 482.0)
GROUPS = {"block": slice(0, 2), "orientation": slice(2, 4), "agent": slice(4, 6)}


# --------------------------------------------------------------------------- targets
def targets(labels: np.ndarray) -> np.ndarray:
    """(N,5) [agent_x, agent_y, block_x, block_y, angle] -> (N,6) block, sin/cos, agent."""
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Labels must be (N,5) agent_xy, block_xy, angle")
    return np.column_stack(
        (values[:, 2:4], np.sin(values[:, 4]), np.cos(values[:, 4]), values[:, 0:2])
    )


def evaluate(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """R², Euclidean mean/p95 for positions and absolute angle error in degrees."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    result: dict[str, Any] = {}
    for name, columns in GROUPS.items():
        error = prediction[:, columns] - truth[:, columns]
        centered = truth[:, columns] - truth[:, columns].mean(axis=0)
        r2 = 1.0 - float(np.sum(error**2) / max(np.sum(centered**2), 1e-12))
        cell = {"r2": r2}
        if name != "orientation":
            distance = np.linalg.norm(error, axis=1)
            cell.update({"mean": float(distance.mean()), "p95": float(np.percentile(distance, 95))})
        else:
            predicted_angle = np.arctan2(prediction[:, 2], prediction[:, 3])
            true_angle = np.arctan2(truth[:, 2], truth[:, 3])
            wrapped = np.angle(np.exp(1j * (predicted_angle - true_angle)))
            cell["mean_abs_degrees"] = float(np.degrees(np.abs(wrapped)).mean())
        result[name] = cell
    return result


# --------------------------------------------------------------------------- probes
class RidgeProbe:
    """Standardized ridge with one penalty per target group, chosen on validation."""

    def __init__(self, mean, scale, coef, intercept, alphas):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), np.asarray(intercept)
        self.alphas = dict(alphas)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef.T + (
            self.intercept
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
            "alphas": np.asarray([self.alphas[name] for name in GROUPS]),
        }

    @classmethod
    def from_arrays(cls, arrays) -> RidgeProbe:
        return cls(
            arrays["mean"],
            arrays["scale"],
            arrays["coef"],
            arrays["intercept"],
            dict(zip(GROUPS, arrays["alphas"].tolist(), strict=True)),
        )


def fit_ridge(
    x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, alphas=RIDGE_ALPHAS
) -> tuple[RidgeProbe, dict]:
    from sklearn.linear_model import Ridge

    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    features = (x - mean) / scale
    features_val = (np.asarray(x_val, dtype=np.float64) - mean) / scale
    coef = np.zeros((y.shape[1], x.shape[1]))
    intercept = np.zeros(y.shape[1])
    chosen, scores = {}, {}
    for name, columns in GROUPS.items():
        best = None
        scores[name] = {}
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(features, y[:, columns])
            error = float(np.mean((model.predict(features_val) - y_val[:, columns]) ** 2))
            scores[name][str(alpha)] = error
            if best is None or error < best[0] - 1e-15:
                best = (error, alpha, model)
        _, alpha, model = best
        chosen[name] = alpha
        coef[columns] = model.coef_
        intercept[columns] = model.intercept_
    return RidgeProbe(mean, scale, coef, intercept, chosen), scores


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    hidden: int = 256,
    epochs: int = 300,
    batch: int = 512,
    patience: int = 20,
    learning_rate: float = 1e-3,
) -> tuple[dict, dict]:
    """Mini-batch MLP trained until validation block error stops improving."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    x_scale[x_scale == 0] = 1.0
    y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
    y_scale[y_scale == 0] = 1.0
    inputs = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    outputs = torch.tensor((y - y_mean) / y_scale, dtype=torch.float32)
    val_inputs = torch.tensor((x_val - x_mean) / x_scale, dtype=torch.float32)
    network = nn.Sequential(
        nn.Linear(x.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, y.shape[1]),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    best_error, best_state, best_epoch, stale, curve = math.inf, None, 0, 0, []
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch):
            index = torch.as_tensor(order[start : start + batch])
            optimizer.zero_grad()
            loss = torch.mean((network(inputs[index]) - outputs[index]) ** 2)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            predicted = network(val_inputs).numpy() * y_scale + y_mean
        error = float(np.mean(np.linalg.norm(predicted[:, :2] - y_val[:, :2], axis=1)))
        curve.append(error)
        if error < best_error - 1e-4:
            best_error, best_epoch, stale = error, epoch + 1, 0
            best_state = {key: value.clone() for key, value in network.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    network.load_state_dict(best_state)
    state = {
        "network": {key: value.numpy() for key, value in network.state_dict().items()},
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "hidden": hidden,
    }
    info = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "converged_before_cap": len(curve) < epochs,
        "validation_block_mean": best_error,
        "curve_every_epoch": curve,
    }
    return state, info


def mlp_predictor(state: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Pure-numpy forward pass of the saved MLP (Linear-ReLU-Linear-ReLU-Linear)."""
    weights = state["network"]

    def predict(x: np.ndarray) -> np.ndarray:
        h = (np.asarray(x, dtype=np.float64) - state["x_mean"]) / state["x_scale"]
        for layer in ("0", "2"):
            h = np.maximum(h @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"], 0)
        out = h @ weights["4.weight"].T + weights["4.bias"]
        return out * state["y_scale"] + state["y_mean"]

    return predict


def load_block_probe(root: Path = ROOT) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """The frozen READOUT-002 selection: projected (N,192) -> block (N,2)."""
    selection = sealed(root / "selection.json")
    if sha256_file(root / selection["path"]) != selection["path_sha256"]:
        raise ValueError("Selected READOUT-002 probe artifact changed")
    with np.load(root / selection["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if selection["kind"] == "ridge":
        probe = RidgeProbe.from_arrays(arrays)
        full = probe.predict
    else:
        state = {
            "network": {key[4:]: value for key, value in arrays.items() if key.startswith("net.")},
            **{key: arrays[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")},
        }
        full = mlp_predictor(state)
    return (lambda x: full(x)[:, :2]), selection


# --------------------------------------------------------------------------- collection
def _sample_pose(rng: np.random.Generator) -> np.ndarray:
    agent = rng.uniform(*AGENT_RANGE, size=2)
    block = rng.uniform(*BLOCK_RANGE, size=2)
    return np.array([*agent, *block, rng.uniform(-math.pi, math.pi)])


def _place(env, pose: np.ndarray) -> bool:
    from jepa_action_effects.gallery_render import geometry, set_block_pose

    env.agent.position = (float(pose[0]), float(pose[1]))
    env.space.reindex_shapes_for_body(env.agent)
    set_block_pose(env, float(pose[4]), (float(pose[2]), float(pose[3])))
    return geometry(env)[0]


def _labels(env) -> np.ndarray:
    return np.array(
        [*env.agent.position, *env.block.position, float(env.block.angle)], dtype=np.float64
    )


def _new_env():
    from jepa_action_effects.pilot_collection import FAR_GOAL
    from jepa_action_effects.simulator_replay import InstrumentedPushT

    env = InstrumentedPushT()
    env.reset(seed=0, options={"goal_state": FAR_GOAL})
    return env


def _shard(model, split: str, index: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, hash_split(split), index]))
    env = _new_env()
    frames, labels, rejected = [], [], 0
    try:
        while len(frames) < SHARD:
            pose = _sample_pose(rng)
            if not _place(env, pose):
                rejected += 1
                continue
            frames.append(env.render().copy())
            labels.append(_labels(env))
    finally:
        env.close()
    frames = np.stack(frames)
    encoded = {"projected": [], "raw_cls": []}
    for start in range(0, SHARD, 100):
        batch = encode_frames(model, frames[start : start + 100], raw_cls=True)
        for key in encoded:
            encoded[key].append(batch[key])
    return {
        "projected": np.concatenate(encoded["projected"]).astype(np.float32),
        "raw_cls": np.concatenate(encoded["raw_cls"]).astype(np.float32),
        "labels": np.stack(labels),
        "rejected": np.asarray(rejected),
        "examples": frames[:8] if index == 0 else np.zeros((0, 224, 224, 3), np.uint8),
    }


def hash_split(split: str) -> int:
    return {"train": 1, "validation": 2, "sensitivity": 3}[split]


def _sensitivity(model) -> dict[str, np.ndarray]:
    """Base poses plus small block translations along seeded random directions."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, hash_split("sensitivity")]))
    env = _new_env()
    base, shifted, shifts, valid = [], [], [], []
    try:
        while len(base) < SENSITIVITY_BASES:
            pose = _sample_pose(rng)
            if not _place(env, pose):
                continue
            base_frame = env.render().copy()
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            row_frames, row_valid = [], []
            for size in SENSITIVITY_SHIFTS:
                moved = pose.copy()
                moved[2:4] += size * direction
                ok = _place(env, moved)
                row_frames.append(env.render().copy())
                row_valid.append(ok)
            _place(env, pose)
            base.append(base_frame)
            shifted.append(np.stack(row_frames))
            shifts.append(direction)
            valid.append(row_valid)
    finally:
        env.close()
    base = np.stack(base)
    shifted = np.stack(shifted)
    frames = np.concatenate((base, shifted.reshape(-1, 224, 224, 3)))
    projected = np.concatenate(
        [encode_frames(model, frames[s : s + 100])["projected"] for s in range(0, len(frames), 100)]
    )
    count = len(base)
    return {
        "base": projected[:count].astype(np.float32),
        "shifted": projected[count:]
        .reshape(count, len(SENSITIVITY_SHIFTS), 192)
        .astype(np.float32),
        "directions": np.stack(shifts),
        "valid": np.asarray(valid, dtype=bool),
        "sizes": np.asarray(SENSITIVITY_SHIFTS),
        "identical_images": np.all(shifted == base[:, None], axis=(2, 3, 4)),
    }


def _study_check(model) -> dict[str, np.ndarray]:
    """Real frames and exact labels of the 60 READOUT tuning families (evaluation only)."""
    manifest = collection_manifest(FRESH)
    ids = sorted(key for key, split in manifest["splits"].items() if split == "validation")
    projected, labels, families = [], [], []
    for family_id in ids:
        family = load_family(FRESH / manifest["family_files"][family_id]["path"])
        frames = np.concatenate(
            (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
        )
        states = np.concatenate((family["history_states"], family["future_states"].reshape(-1, 5)))
        projected.append(encode_frames(model, frames)["projected"])
        labels.append(states)
        families.extend([family_id] * len(frames))
    return {
        "projected": np.concatenate(projected).astype(np.float32),
        "labels": np.concatenate(labels),
        "family_ids": np.asarray(families),
    }


def collect(root: Path = ROOT) -> dict[str, Any]:
    model, _, fingerprint = load_model()
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, count in SPLITS.items():
        for index in range(count):
            path = shards / f"{split}-{index:03d}.npz"
            if not path.exists():
                data = _shard(model, split, index)
                temporary = path.with_name(path.stem + ".partial.npz")
                np.savez_compressed(temporary, **data)
                temporary.replace(path)
            written[path.name] = path.stat().st_size
    for name, builder in (("sensitivity", _sensitivity), ("study-check", _study_check)):
        path = root / f"{name}.npz"
        if not path.exists():
            np.savez_compressed(path, **builder(model))
    result = {
        "status": "complete",
        "fingerprint": fingerprint,
        "shards": written,
        "frames": {split: count * SHARD for split, count in SPLITS.items()},
        "seed": SEED,
        "block_range": BLOCK_RANGE,
        "agent_range": AGENT_RANGE,
    }
    write_json(root / "collect.json", result)
    return result


def _load_split(root: Path, split: str, representation: str = "projected"):
    xs, labels, rejected = [], [], 0
    for index in range(SPLITS[split]):
        with np.load(root / "shards" / f"{split}-{index:03d}.npz", allow_pickle=False) as data:
            xs.append(data[representation].astype(np.float64))
            labels.append(data["labels"])
            rejected += int(data["rejected"])
    return np.concatenate(xs), np.concatenate(labels), rejected


# --------------------------------------------------------------------------- fitting
def sensitivity_summary(predict: Callable[[np.ndarray], np.ndarray], data) -> dict[str, Any]:
    """Decoded block shift along the true direction, per translation size."""
    base = predict(data["base"].astype(np.float64))[:, :2]
    rows = {}
    for k, size in enumerate(data["sizes"]):
        mask = data["valid"][:, k]
        decoded = predict(data["shifted"][:, k].astype(np.float64))[:, :2] - base
        along = np.sum(decoded * data["directions"], axis=1)[mask]
        across = np.abs(
            decoded[:, 0] * data["directions"][:, 1] - decoded[:, 1] * data["directions"][:, 0]
        )[mask]
        rows[str(float(size))] = {
            "valid": int(mask.sum()),
            "identical_image_fraction": float(np.mean(data["identical_images"][mask, k])),
            "along_mean": float(np.mean(along)),
            "along_sd": float(np.std(along)),
            "gain": float(np.mean(along) / size),
            "snr": float(np.mean(along) / max(np.std(along), 1e-12)),
            "across_mean": float(np.mean(across)),
        }
    return rows


def fit(root: Path = ROOT) -> dict[str, Any]:
    collected = json.loads((root / "collect.json").read_text())
    if collected["status"] != "complete":
        raise ValueError("Collection incomplete")
    x, labels, rejected_train = _load_split(root, "train")
    x_val, labels_val, rejected_val = _load_split(root, "validation")
    y, y_val = targets(labels), targets(labels_val)
    with np.load(root / "study-check.npz", allow_pickle=False) as data:
        study = {key: data[key] for key in data.files}
    with np.load(root / "sensitivity.npz", allow_pickle=False) as data:
        sensitivity = {key: data[key] for key in data.files}
    y_study = targets(study["labels"])
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    curve = []
    for size in LEARNING_CURVE:
        probe, _ = fit_ridge(x[:size], y[:size], x_val, y_val)
        curve.append({"frames": size, "validation": evaluate(probe.predict(x_val), y_val)})

    ridge, ridge_scores = fit_ridge(x, y, x_val, y_val)
    np.savez_compressed(models / "ridge.npz", **ridge.arrays())
    candidates = {
        "ridge": {
            "path": "models/ridge.npz",
            "kind": "ridge",
            "alphas": ridge.alphas,
            "alpha_scores": ridge_scores,
            "validation": evaluate(ridge.predict(x_val), y_val),
            "study_check": evaluate(ridge.predict(study["projected"]), y_study),
            "sensitivity": sensitivity_summary(ridge.predict, sensitivity),
        }
    }
    for seed in MLP_SEEDS:
        state, info = fit_mlp(x, y, x_val, y_val, seed)
        path = models / f"mlp-{seed}.npz"
        np.savez_compressed(
            path,
            **{f"net.{key}": value for key, value in state["network"].items()},
            **{key: state[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")},
        )
        predict = mlp_predictor(state)
        candidates[f"mlp-{seed}"] = {
            "path": f"models/{path.name}",
            "kind": "mlp",
            "training": {key: value for key, value in info.items() if key != "curve_every_epoch"},
            "validation": evaluate(predict(x_val), y_val),
            "study_check": evaluate(predict(study["projected"]), y_study),
            "sensitivity": sensitivity_summary(predict, sensitivity),
        }
        write_json(
            root / f"mlp-{seed}-curve.json", {"validation_block_mean": info["curve_every_epoch"]}
        )
    raw_x, _, _ = _load_split(root, "train", "raw_cls")
    raw_val, _, _ = _load_split(root, "validation", "raw_cls")
    raw_probe, _ = fit_ridge(raw_x, y, raw_val, y_val)
    # Selection uses validation block error only; study-check and sensitivity are reported.
    selected = min(
        candidates, key=lambda name: (candidates[name]["validation"]["block"]["mean"], name)
    )
    selection = {
        "selected": selected,
        "kind": candidates[selected]["kind"],
        "path": candidates[selected]["path"],
        "path_sha256": sha256_file(root / candidates[selected]["path"]),
        "rule": "lowest validation block mean error among ridge and three MLP seeds",
    }
    seal(root / "selection.json", selection)
    result = {
        "status": "complete",
        "frames": {"train": len(x), "validation": len(x_val)},
        "rejected_poses": {"train": rejected_train, "validation": rejected_val},
        "learning_curve_ridge": curve,
        "candidates": candidates,
        "raw_cls_ridge_reference": {
            "validation": evaluate(raw_probe.predict(raw_val), y_val),
            "note": "unprojected CLS cannot decode predictor outputs; reference only",
        },
        "selection": selection,
    }
    write_json(root / "fit.json", result)
    return result


def report(root: Path = ROOT) -> dict[str, Any]:
    fitted = json.loads((root / "fit.json").read_text())
    summary = {
        "task": "READOUT-002",
        "runtime": runtime_info(),
        "sources": source_hashes(
            {
                "runner": Path(__file__),
                "common": Path(__file__).with_name("next_common.py"),
                "uv_lock": Path("uv.lock"),
            }
        ),
        "collect": json.loads((root / "collect.json").read_text()),
        **{key: fitted[key] for key in fitted if key != "candidates"},
        "candidates": {
            name: {key: value for key, value in cell.items() if key != "alpha_scores"}
            for name, cell in fitted["candidates"].items()
        },
    }
    summary["collect"].pop("shards", None)
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "readout002-runtime.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "fit", "report"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    stage = {"collect": collect, "fit": fit, "report": report}[args.stage]
    result = Stopwatch(ROOT).run(args.stage, stage, ROOT)
    print(json.dumps(result, indent=2, default=str)[:8000])


if __name__ == "__main__":
    main()
