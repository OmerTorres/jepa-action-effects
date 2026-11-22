"""Shared paths, provenance and frozen-model helpers for the NEXT-001 stages."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import seal, sealed, write_json
from jepa_action_effects.pilot_collection import load_family, sha256_file

CHECKPOINT = Path("checkpoints/lewm-pusht")
STATS = Path("outputs/model001/action-stats.json")
PILOT = Path("outputs/pilot001")
READOUT = Path("outputs/readout001")
FRESH = READOUT / "collection"
ARCHIVE_H5 = Path(
    "data/quentinll-lewm-pusht/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5"
)
MANIFESTS = Path("docs/manifests")
DELTA = 5.12

__all__ = [
    "ARCHIVE_H5",
    "CHECKPOINT",
    "DELTA",
    "FRESH",
    "MANIFESTS",
    "PILOT",
    "READOUT",
    "STATS",
    "Stopwatch",
    "collection_manifest",
    "development_ids",
    "encode_frames",
    "family_path",
    "load_family",
    "load_model",
    "predict_family",
    "runtime_info",
    "seal",
    "sealed",
    "source_hashes",
    "write_json",
]


def source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash every existing path; absent optional inputs are recorded explicitly."""
    return {key: (sha256_file(path) if path.exists() else "absent") for key, path in paths.items()}


def runtime_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "precision": "float32",
    }


class Stopwatch:
    """Record elapsed stage time in a small JSON ledger (no hard cap)."""

    def __init__(self, root: Path):
        self.path = root / "runtime.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"stages": {}}

    def run(self, stage: str, function, *args, **kwargs):
        started = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            self.data["stages"].setdefault(stage, []).append(time.monotonic() - started)
            write_json(self.path, self.data)


def load_model():
    """Return the frozen checkpoint, evaluation action scaler and cache fingerprint."""
    from jepa_action_effects.model_inference import (
        ActionScaler,
        build_frozen_model,
        model_fingerprint,
    )

    model = build_frozen_model(CHECKPOINT)
    scaler = ActionScaler.from_json(STATS)
    return model, scaler, model_fingerprint(CHECKPOINT, scaler)


def encode_frames(model, frames: np.ndarray, raw_cls: bool = False) -> dict[str, np.ndarray]:
    """Encode uint8 THWC frames to projected (and optionally raw) CLS, float32 (T,192)."""
    import torch

    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import encode_real

    pixels = preprocess_pixels(np.ascontiguousarray(frames))
    result = {"projected": encode_real(model, pixels.unsqueeze(0)).numpy()[0]}
    if raw_cls:
        with torch.inference_mode():
            hidden = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state
        result["raw_cls"] = hidden[:, 0].detach().cpu().numpy()
    return result


def predict_family(model, scaler, family: dict) -> np.ndarray:
    """Two-step predictions (6,2,192) from observed history and candidate controls only."""
    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import predict_future

    return predict_future(
        model,
        preprocess_pixels(family["history_frames"]).unsqueeze(0),
        family["warmup_controls"][-10:][None],
        family["candidate_controls"][None],
        scaler,
    ).numpy()[0]


def collection_manifest(root: Path) -> dict:
    path = root / "collection-manifest.json"
    if sha256_file(path) != (root / "collection-manifest.sha256").read_text().strip():
        raise ValueError(f"Collection manifest hash mismatch: {root}")
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete" or manifest["selected_horizon"] != 2:
        raise ValueError(f"Expected a complete two-step collection: {root}")
    return manifest


def family_path(identifier: str, roots: dict[str, Path], manifests: dict[str, dict]) -> Path:
    """Resolve `source:family-id` against a collection manifest and verify its hash."""
    source, family_id = identifier.split(":", 1)
    record = manifests[source]["family_files"][family_id]
    path = roots[source] / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Family artifact hash mismatch: {identifier}")
    return path


def development_ids(pilot: dict, fresh: dict) -> tuple[list[str], list[str]]:
    """READOUT-001 development families, plus the opened confirmation IDs to exclude."""
    from jepa_action_effects.readout_recovery import _development_ids, _partition_ids, _tiers

    parts = _partition_ids(pilot, fresh)
    _tiers(parts)
    return _development_ids(parts), sorted(parts["confirmation"])
