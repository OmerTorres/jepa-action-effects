"""Prepare the pinned PushT training archive and recover action statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

DATASET_REVISION = "655cd446b9929369d7d406001da85c15d1457850"
ARCHIVE_SHA256 = "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318"
ARCHIVE_BYTES = 13_136_247_974
ARCHIVE_URL = (
    "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/"
    f"{DATASET_REVISION}/pusht_expert_train.h5.zst"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    """Download with curl's resume support; immutable hash validation follows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "--fail", "--location", "--continue-at", "-", "--output", str(path), ARCHIVE_URL],
        check=True,
    )


def extract_archive(archive: Path, output: Path) -> None:
    if archive.stat().st_size != ARCHIVE_BYTES or file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Training archive size or SHA-256 does not match the pinned artifact")
    # The compressed frame may omit its content size. Five archive sizes is a
    # conservative extraction allowance for this image-heavy HDF5 artifact.
    required_free_bytes = ARCHIVE_BYTES * 5 + 2 * 1024**3
    if shutil.disk_usage(output.parent).free < required_free_bytes:
        raise OSError("Insufficient free space for conservative extraction allowance")
    temporary = output.with_suffix(output.suffix + ".partial")
    subprocess.run(
        ["zstd", "--decompress", "--keep", "--force", str(archive), "-o", str(temporary)],
        check=True,
    )
    temporary.replace(output)


def compute_action_statistics(actions: np.ndarray) -> dict:
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected action dataset with shape (rows, 2)")
    valid = values[~np.isnan(values).any(axis=1)]
    if valid.shape[0] < 2 or not np.isfinite(valid).all():
        raise ValueError("Need at least two finite action rows after upstream NaN filtering")
    torch_values = torch.from_numpy(valid)
    train_mean = torch_values.mean(0)
    train_scale = torch_values.std(0, correction=1)
    evaluator = StandardScaler().fit(valid)
    for scale in (train_scale.numpy(), evaluator.scale_):
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Action statistics contain a nonfinite or zero scale")
    return {
        "rows_total": int(values.shape[0]),
        "rows_valid": int(valid.shape[0]),
        "rows_removed_nan": int(values.shape[0] - valid.shape[0]),
        "training_scaler": {
            "mean": train_mean.double().tolist(),
            "scale": train_scale.double().tolist(),
            "standard_deviation": "sample (torch correction=1)",
        },
        "evaluation_scaler": {
            "mean": evaluator.mean_.tolist(),
            "scale": evaluator.scale_.tolist(),
            "standard_deviation": "population (sklearn StandardScaler)",
        },
        "scale_absolute_difference": np.abs(train_scale.numpy() - evaluator.scale_).tolist(),
    }


def inspect_hdf5(path: Path) -> tuple[dict, np.ndarray]:
    with h5py.File(path, "r") as h5:
        datasets = {
            name: {"shape": list(item.shape), "dtype": str(item.dtype)}
            for name, item in h5.items()
            if isinstance(item, h5py.Dataset)
        }
        if "action" not in h5 or "ep_len" not in h5 or "ep_offset" not in h5:
            raise ValueError("Archive lacks required action/episode datasets")
        episode_lengths = h5["ep_len"][:]
        episode_offsets = h5["ep_offset"][:]
        if len(episode_lengths) != len(episode_offsets):
            raise ValueError("Episode length and offset arrays disagree")
        if len(episode_offsets) and not np.array_equal(
            episode_offsets, np.cumsum(np.r_[0, episode_lengths[:-1]])
        ):
            raise ValueError("Episode offsets are not contiguous cumulative lengths")
        actions = h5["action"][:]
        metadata = {
            "datasets": datasets,
            "attributes": {key: str(value) for key, value in h5.attrs.items()},
            "episode_count": int(len(episode_lengths)),
            "episode_length_min": int(episode_lengths.min()),
            "episode_length_max": int(episode_lengths.max()),
            "episode_length_sum": int(episode_lengths.sum()),
            "action_alignment": "row action leaves the same-row observation (collector left shift)",
            "simulator_lineage": "not present in archive metadata unless listed in attributes",
        }
    return metadata, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/quentinll-lewm-pusht") / DATASET_REVISION
    parser.add_argument("--archive", type=Path, default=base / "pusht_expert_train.h5.zst")
    parser.add_argument("--h5", type=Path, default=base / "pusht_expert_train.h5")
    parser.add_argument("--output", type=Path, default=Path("outputs/model001/action-stats.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    if not args.archive.exists():
        raise FileNotFoundError("Pinned archive missing; rerun with --download")
    if not args.h5.exists():
        args.h5.parent.mkdir(parents=True, exist_ok=True)
        extract_archive(args.archive, args.h5)
    elif (
        args.archive.stat().st_size != ARCHIVE_BYTES or file_sha256(args.archive) != ARCHIVE_SHA256
    ):
        raise ValueError("Training archive size or SHA-256 does not match the pinned artifact")
    metadata, actions = inspect_hdf5(args.h5)
    report = {
        "dataset_revision": DATASET_REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "h5_sha256": file_sha256(args.h5),
        "metadata": metadata,
        **compute_action_statistics(actions),
        "inference_scaler": "evaluation_scaler",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
