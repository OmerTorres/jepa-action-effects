"""Run MODEL-001 strict-load, leakage, archive-window, cache, and throughput checks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import torch

from jepa_action_effects.action_stats import DATASET_REVISION, file_sha256
from jepa_action_effects.interface_audit import ordinary_transition, preprocess_pixels
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    encode_real,
    manual_predict_future,
    model_fingerprint,
    predict_future,
)


def _selected_windows(h5: h5py.File, count: int = 32) -> list[tuple[int, int]]:
    lengths = h5["ep_len"][:]
    selected = [(int(episode), 0) for episode, length in enumerate(lengths) if length >= 21][:count]
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} episodes have a 21-control window; need {count}")
    return selected


def _archive_window(
    h5: h5py.File, episode: int, start: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor]:
    offset = int(h5["ep_offset"][episode]) + start
    frames = h5["pixels"][offset : offset + 21 : 5]
    if frames.shape != (5, 224, 224, 3):
        raise ValueError(f"Unexpected archive pixel window shape {frames.shape}")
    controls = h5["action"][offset : offset + 20]
    if controls.shape != (20, 2):
        raise ValueError(f"Unexpected archive action window shape {controls.shape}")
    pixels = preprocess_pixels(frames)
    return (
        pixels[:3].unsqueeze(0),
        controls[:10][None],
        controls[10:][None, None],
        pixels[3:].unsqueeze(0),
    )


def _mse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.mean((actual - expected) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lewm-pusht"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/model001/action-stats.json"))
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/quentinll-lewm-pusht") / DATASET_REVISION / "pusht_expert_train.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model001"))
    args = parser.parse_args()

    scaler = ActionScaler.from_json(args.stats)
    model = build_frozen_model(args.checkpoint_dir)
    fingerprint = model_fingerprint(args.checkpoint_dir, scaler)
    cache = LatentCache(args.output_dir / "cache")

    transition = ordinary_transition(future_blocks=2)
    context = transition["context_pixels"]
    past = transition["action_history_raw"].reshape(1, 10, 2)
    future = transition["candidate_actions_raw"].reshape(1, 1, 10, 2)
    targets = transition["target_pixels"]

    batchnorm_before = {
        name: module.running_mean.clone()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    }
    prediction = predict_future(model, context, past, future, scaler)
    manual = manual_predict_future(model, context, past, future, scaler)
    torch.testing.assert_close(prediction, manual, rtol=1e-5, atol=1e-6)
    real = encode_real(model, targets)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            torch.testing.assert_close(module.running_mean, batchnorm_before[name], rtol=0, atol=0)

    prediction_key = cache.key(
        "prediction", {"context": context.numpy(), "past": past, "future": future}, fingerprint
    )
    real_key = cache.key("real", {"targets": targets.numpy()}, fingerprint)
    cache.write(
        "prediction", prediction_key, prediction.numpy(), fingerprint, {"family": "sim-seed-42"}
    )
    cache.write("real", real_key, real.numpy(), fingerprint, {"family": "sim-seed-42"})
    cache.read("prediction", prediction_key, fingerprint)
    cache.read("real", real_key, fingerprint)

    archive_predictions = []
    archive_reals = []
    archive_contexts = []
    archive_input_contexts = []
    archive_input_past = []
    archive_input_future = []
    archive_input_targets = []
    selected: list[tuple[int, int]]
    with h5py.File(args.h5, "r", swmr=True) as h5:
        selected = _selected_windows(h5)
        # Selection is complete before the first model call and independent of model errors.
        for episode, start in selected:
            a_context, a_past, a_future, a_targets = _archive_window(h5, episode, start)
            archive_input_contexts.append(a_context.numpy())
            archive_input_past.append(a_past)
            archive_input_future.append(a_future)
            archive_input_targets.append(a_targets.numpy())
            archive_contexts.append(encode_real(model, a_context)[:, -1:])
            archive_predictions.append(predict_future(model, a_context, a_past, a_future, scaler))
            archive_reals.append(encode_real(model, a_targets))
    archive_prediction = torch.cat(archive_predictions)
    archive_real = torch.cat(archive_reals)
    archive_current = torch.cat(archive_contexts)
    identifiers = {"split": "checkpoint-training-archive", "episode_start_windows": selected}
    h5_digest = file_sha256(args.h5)
    archive_prediction_inputs = {
        "context": np.concatenate(archive_input_contexts),
        "past": np.concatenate(archive_input_past),
        "future": np.concatenate(archive_input_future),
    }
    archive_real_inputs = {"targets": np.concatenate(archive_input_targets)}
    archive_prediction_key = cache.key(
        "prediction",
        archive_prediction_inputs,
        fingerprint,
    )
    archive_real_key = cache.key(
        "real",
        archive_real_inputs,
        fingerprint,
    )
    cache.write(
        "prediction", archive_prediction_key, archive_prediction.numpy(), fingerprint, identifiers
    )
    cache.write("real", archive_real_key, archive_real.numpy(), fingerprint, identifiers)

    six_candidates = np.repeat(future, 6, axis=1)
    six_candidates[:, 1::2] = six_candidates[:, 1::2, ::-1].copy()
    start_time = time.perf_counter()
    benchmark_prediction = predict_future(model, context, past, six_candidates, scaler)
    rollout_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    benchmark_real = encode_real(model, targets)
    encoding_seconds = time.perf_counter() - start_time
    reordered = predict_future(model, context, past, six_candidates[:, ::-1].copy(), scaler)
    torch.testing.assert_close(benchmark_prediction.flip(1), reordered, rtol=1e-5, atol=1e-6)

    changed_targets = targets.flip(1).clone()
    unchanged_prediction = predict_future(model, context, past, future, scaler)
    torch.testing.assert_close(prediction, unchanged_prediction, rtol=0, atol=0)
    if torch.equal(encode_real(model, targets), encode_real(model, changed_targets)):
        raise RuntimeError("Leakage fixture failed to alter separately encoded targets")

    permuted = predict_future(
        model, context, past, future[:, :, [*range(5, 10), *range(5)]], scaler
    )
    report = {
        "scope": (
            "trained-checkpoint interface and training-archive diagnostics; not physical competence"
        ),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ["torch", "transformers", "scikit-learn", "h5py"]
        },
        "fingerprint": fingerprint,
        "dataset": {
            "revision": DATASET_REVISION,
            "h5_sha256": h5_digest,
            "selected_windows": selected,
            "selection_rule": (
                "start zero in first 32 episodes with at least 21 controls, fixed before inference"
            ),
        },
        "strict_load": {
            "state_entries": len(model.state_dict()),
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "simulator_smoke": {
            "context_controls": [0, 5, 10],
            "target_controls": [15, 20],
            "latent_mse": _mse(prediction[:, 0], real),
            "persistence_mse": _mse(encode_real(model, context[:, -1:]).expand_as(real), real),
            "permuted_action_prediction_mse": _mse(prediction, permuted),
            "manual_rollout_max_abs_difference": float(torch.max(torch.abs(prediction - manual))),
            "future_target_leakage_check": "passed",
            "batchnorm_unchanged": True,
        },
        "archive_training_diagnostics": {
            "windows": len(selected),
            "latent_mse": _mse(archive_prediction[:, 0], archive_real),
            "persistence_mse": _mse(archive_current.expand_as(archive_real), archive_real),
        },
        "benchmark": {
            "six_candidate_k2_rollout_seconds": rollout_seconds,
            "two_target_encoding_seconds": encoding_seconds,
            "estimated_120_family_seconds_with_50_percent_contingency": rollout_seconds * 120 * 1.5,
            "prediction_cache_bytes": int(benchmark_prediction.numel() * 4),
            "real_cache_bytes": int(benchmark_real.numel() * 4),
        },
        "cache": {
            "sim_prediction_key": prediction_key,
            "sim_real_key": real_key,
            "archive_prediction_key": archive_prediction_key,
            "archive_real_key": archive_real_key,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
