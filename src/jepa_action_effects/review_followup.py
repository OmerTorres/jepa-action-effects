"""REVIEW-FIX-001: exploratory reanalysis and independent rerun after REVIEW-BRANCH-002.

    python -m jepa_action_effects.review_followup phys      # absolute endpoints, decomposition
    python -m jepa_action_effects.review_followup shift     # per-dose readout and decomposition
    python -m jepa_action_effects.review_followup archive   # episode-clustered audit interval
    python -m jepa_action_effects.review_followup rerun     # independent re-scoring
    python -m jepa_action_effects.review_followup report

Everything here is exploratory: the families were already opened (D026). Sealed
outputs are read, never written. Results go to `outputs/review_followup/` and
`docs/manifests/review-followup-runtime.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.followup_metrics import (
    absolute_endpoint_error,
    cluster_bootstrap,
    decompose,
    displacement_error,
    max_abs_difference,
)
from jepa_action_effects.next_common import (
    ARCHIVE_H5,
    DELTA,
    MANIFESTS,
    Stopwatch,
    collection_manifest,
    encode_frames,
    family_path,
    load_family,
    load_model,
    runtime_info,
    sealed,
    source_hashes,
    write_json,
)

ROOT = Path("outputs/review_followup")
PHYS = Path("outputs/phys002")
SHIFT2 = Path("outputs/shift002")
READOUT2 = Path("outputs/readout002")
SEED = 20261099
SAMPLES = 2000
SHIFT_RERUN = 30


def _probe():
    from jepa_action_effects.readout_scale import load_block_probe

    probe, _ = load_block_probe(READOUT2)
    return lambda z: probe(np.asarray(z, dtype=np.float64))


def _slope(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    t = np.array([r["true"] for r in rows])
    energy = float(np.sum(t * t))
    return None if energy <= 0 else float(np.sum(np.array([r[key] for r in rows]) * t) / energy)


def _ratio(rows: list[dict]) -> float | None:
    pred, real = _slope(rows, "pred"), _slope(rows, "real")
    return None if pred is None or not real else pred / real


def _decomposition(clusters: list[list[dict]], key: str, seed: int) -> dict[str, Any]:
    rows = [r for c in clusters for r in c]
    point = decompose(np.array([r[key] for r in rows]), np.array([r["true"] for r in rows]))
    point["length_ratio_pooled_ci95"] = cluster_bootstrap(
        clusters,
        lambda rs: decompose(np.array([r[key] for r in rs]), np.array([r["true"] for r in rs]))[
            "length_ratio_pooled"
        ],
        seed,
        SAMPLES,
    )["ci95"]
    point["abs_angle_mean_deg_ci95"] = cluster_bootstrap(
        clusters,
        lambda rs: decompose(np.array([r[key] for r in rs]), np.array([r["true"] for r in rs]))[
            "abs_angle_mean_deg"
        ],
        seed + 1,
        SAMPLES,
    )["ci95"]
    return point


def _slopes(clusters: list[list[dict]], seed: int) -> dict[str, Any]:
    return {
        "slope_pred": cluster_bootstrap(clusters, lambda r: _slope(r, "pred"), seed, SAMPLES),
        "slope_real": cluster_bootstrap(clusters, lambda r: _slope(r, "real"), seed + 1, SAMPLES),
        "ratio": cluster_bootstrap(clusters, _ratio, seed + 2, SAMPLES),
    }


# --------------------------------------------------------------------------- PHYS-002
def _phys_items() -> list[tuple[str, str, Path, Path]]:
    from jepa_action_effects.phys002 import _roots

    roots, manifests = _roots()
    config = sealed(PHYS / "config.json")
    items = [
        (
            "develop",
            identifier,
            family_path(identifier, roots, manifests),
            PHYS / "develop" / "families" / f"{identifier.replace(':', '__')}.npz",
        )
        for identifier in config["development_ids"]
    ]
    manifest = collection_manifest(PHYS / "reserved")
    for key, record in sorted(manifest["family_files"].items()):
        identifier = f"reserved:{key}"
        items.append(
            (
                "reserved",
                identifier,
                PHYS / "reserved" / record["path"],
                PHYS / "confirm-reserved" / "families" / f"{identifier.replace(':', '__')}.npz",
            )
        )
    return items


def phys(root: Path = ROOT) -> dict[str, Any]:
    """Absolute endpoint vs displacement error, and length/direction decomposition."""
    probe = _probe()
    families: dict[str, list[dict]] = {"develop": [], "reserved": []}
    for phase, identifier, path, saved in _phys_items():
        family = load_family(path)
        with np.load(saved, allow_pickle=False) as arrays:
            real, predicted = arrays["real"], arrays["predicted"]
        start = probe(real[2:3])[0]
        real_end = probe(real[3:].reshape(6, 2, 192)[:, -1])
        pred_end = probe(predicted[:, -1])
        true_end = family["future_states"][:, -1, 2:4].astype(np.float64)
        true_start = family["history_states"][-1, 2:4].astype(np.float64)
        moved = np.linalg.norm(true_end - true_start, axis=1) > DELTA
        families[phase].append(
            {
                "id": identifier,
                "cohort": "challenge" if "challenge" in identifier else "ordinary",
                "abs_pred": float(absolute_endpoint_error(pred_end, true_end).mean()),
                "abs_real": float(absolute_endpoint_error(real_end, true_end).mean()),
                "disp_pred": float(
                    displacement_error(pred_end, start, true_end, true_start).mean()
                ),
                "disp_real": float(
                    displacement_error(real_end, start, true_end, true_start).mean()
                ),
                "moved": [
                    {
                        "pred": (pred_end[i] - start).tolist(),
                        "real": (real_end[i] - start).tolist(),
                        "true": (true_end[i] - true_start).tolist(),
                    }
                    for i in np.flatnonzero(moved)
                ],
            }
        )
    result: dict[str, Any] = {}
    for phase, rows in families.items():
        for cohort in ("challenge", "pooled"):
            chosen = [r for r in rows if cohort == "pooled" or r["cohort"] == cohort]
            cell: dict[str, Any] = {"families": len(chosen)}
            for offset, key in enumerate(("abs_pred", "abs_real", "disp_pred", "disp_real")):
                cell[key] = cluster_bootstrap(
                    [[r] for r in chosen],
                    lambda rs, k=key: float(np.mean([r[k] for r in rs])),
                    SEED + offset,
                    SAMPLES,
                )
            clusters = [r["moved"] for r in chosen]
            cell["decompose_pred"] = _decomposition(clusters, "pred", SEED + 10)
            cell["decompose_real"] = _decomposition(clusters, "real", SEED + 20)
            cell.update(_slopes(clusters, SEED + 30))
            result[f"{phase}_{cohort}"] = cell
    write_json(root / "phys.json", result)
    return result


# --------------------------------------------------------------------------- SHIFT-002
def shift(root: Path = ROOT) -> dict[str, Any]:
    """Per dose × structure: readout slope, predictor slope, ratio and decomposition."""
    probe = _probe()
    collection = sealed(SHIFT2 / "collection.json")
    result: dict[str, Any] = {}
    for phase, folder in (("development", "develop"), ("reserved", "confirm")):
        cells: dict[tuple[str, str, str], list[list[dict]]] = {}
        for family_id, entry in sorted(collection["families"].items()):
            if entry["split"] != phase:
                continue
            family = load_family(SHIFT2 / entry["path"])
            with np.load(SHIFT2 / folder / f"{family_id}.npz", allow_pickle=False) as arrays:
                real, predicted = arrays["real"], arrays["predicted"]
            start = probe(real[2:3])[0]
            real_d = probe(real[3:]).reshape(6, 2, 2) - start
            pred_d = probe(predicted.reshape(-1, 192)).reshape(6, 2, 2) - start
            true = family["future_states"][:, :, 2:4] - family["history_states"][-1, 2:4]
            spec = family["metadata"]["spec"]
            per_family: dict[tuple[str, str, str], list[dict]] = {}
            for index, candidate in enumerate(family["metadata"]["candidate_ids"]):
                if np.linalg.norm(true[index, 1]) <= DELTA:
                    continue
                key = (entry["state"], spec[candidate]["label"], spec[candidate]["structure"])
                per_family.setdefault(key, []).append(
                    {
                        "pred": pred_d[index, 1].tolist(),
                        "real": real_d[index, 1].tolist(),
                        "true": true[index, 1].tolist(),
                    }
                )
            for key, rows in per_family.items():
                cells.setdefault(key, []).append(rows)
        for offset, (key, clusters) in enumerate(sorted(cells.items())):
            name = f"{phase}_{'_'.join(key)}"
            cell = {"families": len(clusters), **_slopes(clusters, SEED + 100 * offset)}
            cell["decompose_pred"] = _decomposition(clusters, "pred", SEED + 100 * offset + 10)
            result[name] = cell
    write_json(root / "shift.json", result)
    return result


# --------------------------------------------------------------------------- archive
def archive(root: Path = ROOT) -> dict[str, Any]:
    """Replay the audit's seeded archive sampling, then bootstrap by episode."""
    import h5py

    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import predict_future
    from jepa_action_effects.phys002_audit import (
        ARCHIVE_ATTEMPT_CAP,
        ARCHIVE_SEED,
        ARCHIVE_WINDOWS,
    )

    probe = _probe()
    model, scaler, _ = load_model()
    rng = np.random.default_rng(ARCHIVE_SEED)
    windows, attempts = [], 0
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
            decoded_start = probe(real[2:3])[0]
            windows.append(
                {
                    "episode": episode,
                    "true": (states[4, 2:4] - states[2, 2:4]).tolist(),
                    "pred": (probe(predicted[-1:])[0] - decoded_start).tolist(),
                    "real": (probe(real[-1:])[0] - decoded_start).tolist(),
                }
            )
    moved = [w for w in windows if np.linalg.norm(w["true"]) > DELTA]
    by_episode: dict[int, list[dict]] = {}
    for w in moved:
        by_episode.setdefault(w["episode"], []).append(w)
    result = {
        "windows": len(windows),
        "episodes": len({w["episode"] for w in windows}),
        "moved_windows": len(moved),
        "moved_episodes": len(by_episode),
        "window_bootstrap": _slopes([[w] for w in moved], SEED + 500),
        "episode_bootstrap": _slopes(list(by_episode.values()), SEED + 500),
        "decompose_pred": _decomposition(list(by_episode.values()), "pred", SEED + 510),
    }
    write_json(root / "archive.json", result)
    return result


# --------------------------------------------------------------------------- rerun
def rerun(root: Path = ROOT) -> dict[str, Any]:
    """Re-score sealed families from scratch and compare with the sealed records."""
    from jepa_action_effects.phys002 import _goals, _summaries
    from jepa_action_effects.phys002 import score_family as phys_score
    from jepa_action_effects.readout_scale import load_block_probe
    from jepa_action_effects.shift001 import score_family as shift_score

    model, scaler, _ = load_model()
    probe, _ = load_block_probe(READOUT2)
    skip = frozenset({"input_sha256", "verification_max_error"})
    manifest = collection_manifest(PHYS / "reserved")
    records, record_diff, array_diff = [], 0.0, 0.0
    for key in sorted(manifest["family_files"]):
        identifier = f"reserved:{key}"
        family = load_family(PHYS / "reserved" / manifest["family_files"][key]["path"])
        record, arrays = phys_score(identifier, family, _goals(manifest, key), model, scaler, probe)
        base = PHYS / "confirm-reserved" / "families" / f"reserved__{key}"
        stored = sealed(base.with_suffix(".json"))
        record_diff = max(record_diff, max_abs_difference(record, stored, skip))
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as saved:
            for name in ("real", "predicted"):
                array_diff = max(array_diff, float(np.max(np.abs(arrays[name] - saved[name]))))
        records.append(record)
    config = sealed(PHYS / "config.json")
    summary = _summaries(records, config["bootstrap_seed"])
    sealed_summary = sealed(PHYS / "confirm.json")["reserved"]["summary"]
    phys_result = {
        "families": len(records),
        "max_record_difference": record_diff,
        "max_embedding_difference": array_diff,
        "max_summary_difference": max_abs_difference(summary, sealed_summary),
        "criteria_rerun": summary["challenge"]["criteria"],
    }
    collection = sealed(SHIFT2 / "collection.json")
    reserved = sorted(k for k, v in collection["families"].items() if v["split"] == "reserved")
    chosen = [k for k in reserved if k.startswith("challenge")][: SHIFT_RERUN // 2] + [
        k for k in reserved if k.startswith("archive")
    ][: SHIFT_RERUN // 2]
    shift_diff = 0.0
    for family_id in chosen:
        entry = collection["families"][family_id]
        family = load_family(SHIFT2 / entry["path"])
        record, _ = shift_score(family_id, family, entry["state"], model, scaler, probe)
        stored = sealed(SHIFT2 / "confirm" / f"{family_id}.json")
        shift_diff = max(shift_diff, max_abs_difference(record, stored, skip))
    result = {
        "phys002_reserved": phys_result,
        "shift002_reserved_subset": {"families": len(chosen), "max_record_difference": shift_diff},
    }
    write_json(root / "rerun.json", result)
    return result


def report(root: Path = ROOT) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "task": "REVIEW-FIX-001",
        "status": "exploratory (families already opened; D026)",
        "runtime": runtime_info(),
        "sources": source_hashes(
            {
                "runner": Path(__file__),
                "metrics": Path(__file__).with_name("followup_metrics.py"),
                "uv_lock": Path("uv.lock"),
            }
        ),
    }
    for stage in ("phys", "shift", "archive", "rerun"):
        path = root / f"{stage}.json"
        summary[stage] = json.loads(path.read_text()) if path.exists() else None
    if summary["phys"]:
        for cell in summary["phys"].values():
            for key in ("decompose_pred", "decompose_real"):
                cell[key].pop("rows", None)
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "review-followup-runtime.json", summary)
    return {k: v for k, v in summary.items() if k in ("task", "status")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("phys", "shift", "archive", "rerun", "report"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    stages = {"phys": phys, "shift": shift, "archive": archive, "rerun": rerun, "report": report}
    result = Stopwatch(ROOT).run(args.stage, stages[args.stage], ROOT)
    print(json.dumps(result, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
