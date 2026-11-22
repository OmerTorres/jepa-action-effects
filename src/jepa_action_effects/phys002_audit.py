"""PHYS-002 post-confirmation audit (exploratory; read-only on sealed PHYS-002 outputs).

    python -m jepa_action_effects.phys002_audit run

Checks whether the PHYS-002 shrinkage could be a data, alignment or probe artifact:

- **alignment:** decoded predicted/real displacement at steps 1 and 2 against true
  displacement at steps 1 and 2, plus a readout-free latent slope;
- **probe size:** the same MLP recipe refit on 2k-50k READOUT-002 frames;
- **archive control:** the same predictor and frozen probe on in-distribution archive
  windows with true states, split by whether the block moved during the history.

Nothing here is sealed or used for a decision: the reserved families were already
opened when this was written. Results go to `outputs/phys002_audit/` and a manifest.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    ARCHIVE_H5,
    DELTA,
    MANIFESTS,
    collection_manifest,
    encode_frames,
    family_path,
    load_family,
    load_model,
    runtime_info,
    source_hashes,
    write_json,
)

ROOT = Path("outputs/phys002_audit")
PHYS = Path("outputs/phys002")
READOUT2 = Path("outputs/readout002")
PROBE_SIZES = (2000, 5000, 10000, 20000, 50000)
PROBE_SEED = 20261005
ARCHIVE_SEED = 777
ARCHIVE_WINDOWS = 600
ARCHIVE_ATTEMPT_CAP = 5000
REST_LIMIT = 1.0  # history block motion (units) below which a window starts at rest
MAGNITUDE_BINS = ((DELTA, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 400.0))
BOOTSTRAP = 1000


# --------------------------------------------------------------------------- pure helpers
def slope(predicted: np.ndarray, true: np.ndarray) -> float | None:
    """Least-squares slope through the origin of predicted on true 2-D displacements."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true, dtype=np.float64).reshape(-1, 2)
    energy = float(np.sum(true * true))
    return None if energy <= 0 else float(np.sum(predicted * true) / energy)


def bootstrap_interval(
    groups: Sequence[Any], statistic: Callable[[list], float | None], seed: int
) -> list[float] | None:
    """Percentile 95% interval, resampling whole groups (families or windows)."""
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        value = statistic([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if value is not None:
            draws.append(value)
    return np.percentile(draws, (2.5, 97.5)).tolist() if draws else None


def pooled_slope(rows: Sequence[dict], predicted: str, true: str = "true") -> float | None:
    if not rows:
        return None
    return slope(
        np.concatenate([row[predicted] for row in rows]),
        np.concatenate([row[true] for row in rows]),
    )


def alignment_matrix(families: Sequence[dict]) -> dict[str, float | None]:
    """Slope of decoded source step k on true step j, over moved candidates.

    Each family holds `true` (C,2,2), `pred`/`real` decoded (C,2,2) and `start` (2,).
    A one-step offset shows up as a step-k slope that matches true step k-1.
    """
    result = {}
    for source in ("pred", "real"):
        for k in (0, 1):
            for j in (0, 1):
                decoded, true = [], []
                for family in families:
                    moved = np.linalg.norm(family["true"][:, 1], axis=1) > DELTA
                    decoded.append((family[source][:, k] - family["start"])[moved])
                    true.append(family["true"][:, j][moved])
                result[f"{source}_step{k + 1}_vs_true{j + 1}"] = slope(
                    np.concatenate(decoded), np.concatenate(true)
                )
    return result


def latent_statistics(families: Sequence[dict]) -> dict[str, float]:
    """Readout-free step-2 slope of (ẑ−z_t) on (z−z_t), and predictor/persistence MSE."""
    inner = energy = predicted_mse = persistence_mse = 0.0
    for family in families:
        real_change = family["z_real"][:, 1] - family["z_start"]
        predicted_change = family["z_pred"][:, 1] - family["z_start"]
        inner += float(np.sum(predicted_change * real_change))
        energy += float(np.sum(real_change * real_change))
        predicted_mse += float(np.mean((family["z_pred"][:, 1] - family["z_real"][:, 1]) ** 2))
        persistence_mse += float(np.mean(real_change**2))
    return {
        "latent_slope": inner / energy,
        "pred_over_persistence_mse": predicted_mse / persistence_mse,
    }


def binned_slopes(predicted: np.ndarray, true: np.ndarray) -> dict[str, dict]:
    magnitude = np.linalg.norm(true, axis=1)
    rows = {}
    for low, high in MAGNITUDE_BINS:
        mask = (magnitude > low) & (magnitude <= high)
        rows[f"{low:g}-{high:g}"] = {
            "count": int(mask.sum()),
            "slope_pred": slope(predicted[mask], true[mask]) if mask.any() else None,
        }
    return rows


# --------------------------------------------------------------------------- study data
def _study_families(probe) -> list[dict]:
    from jepa_action_effects.phys002 import _roots

    config = json.loads((PHYS / "config.json").read_text())
    roots, manifests = _roots()
    items = [
        ("develop", identifier, family_path(identifier, roots, manifests))
        for identifier in config["development_ids"]
    ]
    reserved = collection_manifest(PHYS / "reserved")
    items += [
        ("reserved", f"reserved:{key}", PHYS / "reserved" / record["path"])
        for key, record in sorted(reserved["family_files"].items())
    ]
    families = []
    for phase, identifier, path in items:
        family = load_family(path)
        folder = "develop" if phase == "develop" else "confirm-reserved"
        saved = PHYS / folder / "families" / f"{identifier.replace(':', '__')}.npz"
        with np.load(saved, allow_pickle=False) as arrays:
            real = arrays["real"].astype(np.float64)
            predicted = arrays["predicted"].astype(np.float64)
        history_block = family["history_states"][:, 2:4].astype(np.float64)
        z_real = real[3:].reshape(6, 2, 192)
        families.append(
            {
                "phase": phase,
                "cohort": "challenge" if "challenge" in identifier else "ordinary",
                "history_motion": float(np.linalg.norm(history_block[-1] - history_block[0])),
                "true": family["future_states"][:, :, 2:4].astype(np.float64) - history_block[-1],
                "z_start": real[2],
                "z_real": z_real,
                "z_pred": predicted,
                "start": probe(real[2:3])[0],
                "real": probe(z_real.reshape(-1, 192)).reshape(6, 2, 2),
                "pred": probe(predicted.reshape(-1, 192)).reshape(6, 2, 2),
            }
        )
    return families


def _step2_rows(families: Sequence[dict], decode=None) -> list[dict]:
    """Per-family moved step-2 displacements, optionally re-decoded with another probe."""
    rows = []
    for family in families:
        true = family["true"][:, 1]
        moved = np.linalg.norm(true, axis=1) > DELTA
        if decode is None:
            start, pred, real = family["start"], family["pred"][:, 1], family["real"][:, 1]
        else:
            start = decode(family["z_start"][None])[0]
            pred, real = decode(family["z_pred"][:, 1]), decode(family["z_real"][:, 1])
        rows.append(
            {
                "pred": (pred - start)[moved],
                "real": (real - start)[moved],
                "true": true[moved],
                "real_error": float(np.mean(np.linalg.norm((real - start) - true, axis=1))),
            }
        )
    return rows


# --------------------------------------------------------------------------- stages
def alignment(families: Sequence[dict]) -> dict[str, Any]:
    result = {}
    for phase in ("develop", "reserved"):
        chosen = [f for f in families if f["phase"] == phase and f["cohort"] == "challenge"]
        cell = alignment_matrix(chosen)
        cell.update(latent_statistics(chosen))
        cell["latent_slope_ci95"] = bootstrap_interval(
            chosen, lambda rows: latent_statistics(rows)["latent_slope"], ARCHIVE_SEED
        )
        cell["families"] = len(chosen)
        result[phase] = cell
    challenge = [f for f in families if f["cohort"] == "challenge"]
    result["challenge_history_motion"] = {
        "median": float(np.median([f["history_motion"] for f in challenge])),
        "fraction_at_rest": float(np.mean([f["history_motion"] < REST_LIMIT for f in challenge])),
    }
    rows = _step2_rows([f for f in challenge if f["phase"] == "develop"])
    result["develop_binned"] = binned_slopes(
        np.concatenate([r["pred"] for r in rows]), np.concatenate([r["true"] for r in rows])
    )
    return result


def probe_size(families: Sequence[dict]) -> dict[str, Any]:
    from jepa_action_effects.readout_scale import _load_split, fit_mlp, mlp_predictor, targets

    x, labels, _ = _load_split(READOUT2, "train")
    x_val, labels_val, _ = _load_split(READOUT2, "validation")
    y, y_val = targets(labels), targets(labels_val)
    challenge = [f for f in families if f["cohort"] == "challenge"]
    result = {}
    for size in PROBE_SIZES:
        state, info = fit_mlp(x[:size], y[:size], x_val, y_val, seed=PROBE_SEED)
        full = mlp_predictor(state)
        rows = _step2_rows(challenge, lambda z, full=full: full(z)[:, :2])
        predicted, real = pooled_slope(rows, "pred"), pooled_slope(rows, "real")
        result[str(size)] = {
            "validation_block_mean": info["validation_block_mean"],
            "slope_pred": predicted,
            "slope_real": real,
            "ratio": predicted / real,
            "real_displacement_error": float(np.mean([r["real_error"] for r in rows])),
        }
        print("probe size", size, result[str(size)], flush=True)
    return result


def archive_control(probe) -> dict[str, Any]:
    import h5py

    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import predict_future

    model, scaler, _ = load_model()
    rng = np.random.default_rng(ARCHIVE_SEED)
    windows, attempts, skipped = [], 0, 0
    with h5py.File(ARCHIVE_H5, "r") as h5:
        offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
        eligible = np.flatnonzero(lengths >= 21)
        while len(windows) < ARCHIVE_WINDOWS and attempts < ARCHIVE_ATTEMPT_CAP:
            attempts += 1
            episode = int(rng.choice(eligible))
            start, length = int(offsets[episode]), int(lengths[episode])
            t = int(rng.integers(10, length - 10))
            indices = [start + t + k for k in (-10, -5, 0, 5, 10)]
            controls = h5["action"][start + t - 10 : start + t + 10].astype(np.float32)
            if np.any(np.abs(controls) > 1) or not np.isfinite(controls).all():
                skipped += 1
                continue
            states = np.stack([h5["state"][i] for i in indices]).astype(np.float64)
            frames = np.stack([h5["pixels"][i] for i in indices])
            predicted = predict_future(
                model,
                preprocess_pixels(frames[:3]).unsqueeze(0),
                controls[None, :10],
                controls[None, None, 10:],
                scaler,
            ).numpy()[0, 0]
            real = encode_frames(model, frames)["projected"]
            decoded_start = probe(real[2:3].astype(np.float64))[0]
            windows.append(
                {
                    "episode": episode,
                    "t": t,
                    "true": states[4:5, 2:4] - states[2, 2:4],
                    "pred": probe(predicted[-1:].astype(np.float64)) - decoded_start,
                    "real": probe(real[-1:].astype(np.float64)) - decoded_start,
                    "history_motion": float(np.linalg.norm(states[2, 2:4] - states[0, 2:4])),
                }
            )
    moved = [w for w in windows if np.linalg.norm(w["true"]) > DELTA]
    result: dict[str, Any] = {
        "windows": len(windows),
        "skipped_out_of_bounds": skipped,
        "moved_windows": len(moved),
        "moved_displacement_median": float(np.median([np.linalg.norm(w["true"]) for w in moved])),
    }
    for label, rows in (
        ("moved", moved),
        ("moved_block_at_rest", [w for w in moved if w["history_motion"] < REST_LIMIT]),
        ("moved_block_moving", [w for w in moved if w["history_motion"] >= REST_LIMIT]),
    ):
        result[label] = {
            "count": len(rows),
            "slope_pred": pooled_slope(rows, "pred"),
            "slope_pred_ci95": bootstrap_interval(rows, lambda r: pooled_slope(r, "pred"), 1),
            "slope_real": pooled_slope(rows, "real"),
            "slope_real_ci95": bootstrap_interval(rows, lambda r: pooled_slope(r, "real"), 2),
            "err_pred": float(np.mean([np.linalg.norm(w["pred"] - w["true"]) for w in rows]))
            if rows
            else None,
            "err_real": float(np.mean([np.linalg.norm(w["real"] - w["true"]) for w in rows]))
            if rows
            else None,
        }
    result["binned"] = binned_slopes(
        np.concatenate([w["pred"] for w in moved]), np.concatenate([w["true"] for w in moved])
    )
    return result


def run(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.readout_scale import load_block_probe

    probe, selection = load_block_probe(READOUT2)
    families = _study_families(probe)
    summary = {
        "task": "PHYS-002-AUDIT",
        "status": "exploratory (reserved families already opened)",
        "runtime": runtime_info(),
        "sources": source_hashes({"runner": Path(__file__), "uv_lock": Path("uv.lock")}),
        "readout_selection": selection["selected"],
        "alignment": alignment(families),
    }
    print(json.dumps(summary["alignment"], indent=1, default=str), flush=True)
    summary["probe_size"] = probe_size(families)
    summary["archive_control"] = archive_control(probe)
    print(json.dumps(summary["archive_control"], indent=1), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "phys002-audit-runtime.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
