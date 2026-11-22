"""Real test examples for the course report (D039); display only, read-only.

    python -m jepa_action_effects.report_examples   # writes docs/manifests/report-examples.json
                                                     # and docs/figures/report-examples/*.png

Selects one designed family from the sealed SUPPORT-001 cohort and one from the TRAIN-002
confirmation cohort by the D039 rule: true block displacement closest to the cohort median
(at strength 1.00 and 0.60 respectively; ties go to the lowest family id). The rule reads
simulator outcomes only, never model errors. Both seals and every input hash are verified.
TRAIN-002 endpoints come from the saved confirmation records; SUPPORT-001 endpoints apply
the sealed primary readout to the saved predicted latents, as `support001_derived` does.
No world model is run, no sealed file is written and no estimate depends on the examples.
The manifest also carries a histogram of the demo command magnitudes in the SUPPORT-001
support map, so the report can show where the training demos lie.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import sealed, write_json
from jepa_action_effects.next_common import MANIFESTS
from jepa_action_effects.pilot_collection import load_family, sha256_file

OUT = MANIFESTS / "report-examples.json"
FRAMES = Path("docs/figures/report-examples")
SUPPORT_ROOT = Path("outputs/support001")
READOUT = "mlp-20261005"
SUPPORT_RULE = "m1.00"
TRAIN_RULE = "m0.60"
DESIGNED = "challenge"
HISTOGRAM_BINS = np.round(np.arange(0.0, 1.0001, 0.025), 3)
MOVED = 5.12  # a branch "moved" if the block moved more than 1% of the table width
ARMS = ("frozen", "balanced", "archive_weighted", "dose_omitted", "archive_only")

# Simulator geometry (`stable_worldmodel` PushT `add_tee` and the default agent scale).
BAR = ((-60.0, 30.0), (60.0, 30.0), (60.0, 0.0), (-60.0, 0.0))
STEM = ((-15.0, 30.0), (-15.0, 120.0), (15.0, 120.0), (15.0, 30.0))
AGENT_RADIUS = 15.0
WORLD = 512.0


class ExampleMismatch(ValueError):
    """An input does not match its recorded hash or the selection rule cannot be applied."""


def median_closest(values: dict[str, float]) -> str:
    """Id whose value is closest to the median of all values; ties go to the lowest id."""
    if not values:
        raise ExampleMismatch("No candidate families")
    median = float(np.median(list(values.values())))
    return min(sorted(values), key=lambda key: (abs(values[key] - median), key))


def tee_polygons(x: float, y: float, angle: float) -> list[np.ndarray]:
    """The T-block's bar and stem corners in world units (y down, as rendered)."""
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return [np.asarray(part) @ rotation.T + (x, y) for part in (BAR, STEM)]


def _floats(array: Any) -> Any:
    return np.asarray(array, dtype=float).tolist()


def _true_displacement(family: dict, label: str) -> float:
    branch = list(family["metadata"]["candidate_ids"]).index(label)
    start = np.asarray(family["history_states"], float)[2, 2:4]
    end = np.asarray(family["future_states"], float)[branch, -1, 2:4]
    return float(np.linalg.norm(end - start))


def branch_summary(states: dict, start: Any, ends: dict[str, Any]) -> dict[str, dict]:
    """Per strength: true block displacement, agent travel, and each decoded end's length and
    gain along the true direction (p·t/|t|², the one-branch projection gain; None if unmoved).
    """
    history = np.asarray(states["history_states"], float)
    future = np.asarray(states["future_states"], float)
    true = future[:, -1, 2:4] - history[2, 2:4]
    travel = np.linalg.norm(future[:, -1, 0:2] - history[2, 0:2], axis=1)
    out = {}
    for i, label in enumerate(states["labels"]):
        energy = float(true[i] @ true[i])
        row: dict[str, Any] = {"true": float(np.sqrt(energy)), "agent_travel": float(travel[i])}
        for name, end in ends.items():
            shift = np.asarray(end, float)[i] - np.asarray(start, float)
            gain = float(shift @ true[i] / energy) if np.sqrt(energy) > MOVED else None
            row[name] = {"length": float(np.linalg.norm(shift)), "gain": gain}
        out[label] = row
    return out


def arm_means(models: dict[str, dict]) -> dict[str, list]:
    """Seed-averaged decoded ends per fine-tuning arm (the released model has one entry)."""
    out = {}
    for arm in ARMS:
        ids = [m for m in models if m == arm or m.startswith(f"{arm}-")]
        if not ids:
            raise ExampleMismatch(f"No records for {arm}")
        out[arm] = _floats(np.mean([models[m]["pred_end"] for m in ids], axis=0))
    return out


def _verified_family(root: Path, entry: dict) -> dict:
    path = root / entry["relative_path"]
    if sha256_file(path) != entry["file_sha256"]:
        raise ExampleMismatch(f"{entry['id']} changed after admission")
    return load_family(path)


def _states(family: dict) -> dict[str, Any]:
    trajectories = np.asarray(family["block_trajectories"], float)
    return {
        "labels": list(family["metadata"]["candidate_ids"]),
        "history_states": _floats(family["history_states"]),
        "future_states": _floats(family["future_states"]),
        "block_trajectories": _floats(trajectories[:, ::5]),
        "mean_command": _floats(np.asarray(family["candidate_controls"], float).mean(axis=1)),
        "contacts": [bool(c) for c in family["contacts"]],
        "vertices_check": _vertex_error(family),
    }


def _vertex_error(family: dict) -> float | None:
    """Largest distance from our T geometry to the simulator's saved corners, if saved."""
    saved = family["metadata"].get("actual_block_vertices")
    if saved is None:
        return None
    x, y, angle = np.asarray(family["history_states"], float)[2, 2:5]
    ours = np.concatenate(tee_polygons(x, y, angle))
    distance = np.linalg.norm(ours[:, None] - np.asarray(saved)[None], axis=-1)
    error = float(distance.min(axis=1).max())
    if error > 1e-6:
        raise ExampleMismatch(f"T geometry differs from saved corners by {error:.3g}")
    return error


def _save_frame(frame: np.ndarray, name: str, frames: Path) -> dict[str, str]:
    from PIL import Image

    frames.mkdir(parents=True, exist_ok=True)
    path = frames / f"{name}.png"
    Image.fromarray(np.asarray(frame, np.uint8)).save(path, optimize=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def support_example(root: Path = SUPPORT_ROOT, frames: Path = FRAMES) -> dict[str, Any]:
    from jepa_action_effects import support001

    seal = support001.verify_seal(root)
    collection = sealed(root / "collection.json")
    cache = sealed(root / "cache.json")
    designed = {k: v for k, v in sorted(collection["families"].items()) if v["scene"] == DESIGNED}
    values = {
        k: _true_displacement(_verified_family(root, v), SUPPORT_RULE) for k, v in designed.items()
    }
    chosen = median_closest(values)
    family = _verified_family(root, designed[chosen])
    block, agent, digests = support001.decoders()
    if digests != seal["readouts"]:
        raise ExampleMismatch("Readouts changed after the SUPPORT-001 seal")
    predicted_path = root / "predicted" / f"{chosen}.npz"
    with np.load(predicted_path) as arrays:
        predicted = arrays["predicted"].astype(np.float64)
    real = support001._real_decodes(
        support001._real_latents(root, cache["records"][chosen]), block, agent
    )[READOUT]
    labels = list(family["metadata"]["candidate_ids"])
    states = _states(family)
    pred_end = block[READOUT](predicted[:, 1])
    shown = {
        f"future-{label}": _save_frame(
            family["future_frames"][labels.index(label), -1], f"{chosen}_{label}", frames
        )
        for label in ("m0.30", "m0.60", "m1.00")
    }
    return {
        "family": chosen,
        "cohort": "SUPPORT-001 designed families (sealed confirmation)",
        "rule": f"true block displacement at {SUPPORT_RULE} closest to the cohort median",
        "candidates": len(values),
        "median_true_displacement": float(np.median(list(values.values()))),
        "chosen_true_displacement": values[chosen],
        **states,
        "summary": branch_summary(
            states, real["start"], {"frozen": pred_end, "real_frame": real["real_end"]}
        ),
        "decoded": {
            "readout": READOUT,
            "start": _floats(real["start"]),
            "real_end": _floats(real["real_end"]),
            "pred_end": _floats(pred_end),
            "agent_start": _floats(real["agent_start"]),
            "agent_real_end": _floats(real["agent_real_end"]),
            "agent_pred_end": _floats(agent[READOUT](predicted[:, 1])),
        },
        "frames": {"history-2": _save_frame(family["history_frames"][2], f"{chosen}_now", frames)}
        | shown,
        "sources": {
            "seal_sha256": sha256_file(root / "seal.json"),
            "family_sha256": designed[chosen]["file_sha256"],
            "predicted_sha256": sha256_file(predicted_path),
            "cache_sha256": cache["records"][chosen]["sha256"],
            "readout_digest": digests[READOUT],
        },
    }


def train_example(frames: Path = FRAMES) -> dict[str, Any]:
    from jepa_action_effects.train002 import _model_ids, protocol, verify_seal
    from jepa_action_effects.train002_recompute import load_records

    p = protocol()
    payload = verify_seal(p)
    root = p.run_root
    collection = sealed(root / "confirmation" / "collection.json")
    designed = {k: v for k, v in sorted(collection["families"].items()) if v["scene"] == DESIGNED}
    values = {
        k: _true_displacement(_verified_family(root, v), TRAIN_RULE) for k, v in designed.items()
    }
    chosen = median_closest(values)
    family = _verified_family(root, designed[chosen])
    models = {}
    for model_id in _model_ids(payload["registry"]):
        record = next(
            r for r in load_records(p, payload, model_id, "family") if r["unit"] == chosen
        )
        decoded = record["decoded"][READOUT]
        models[model_id] = {key: _floats(v) for key, v in decoded.items() if key != "perm_end"}
        if model_id == "real-frames":
            models[model_id] |= {
                "true_start": _floats(record["true_start"]),
                "true_end": _floats(record["true_end"]),
            }
    labels = list(family["metadata"]["candidate_ids"])
    states = _states(family)
    means = arm_means(models)
    real = models["real-frames"]
    summary = branch_summary(states, real["start"], means | {"real_frame": real["real_end"]})
    saved = {
        f"history-{i}": _save_frame(family["history_frames"][i], f"{chosen}_history{i}", frames)
        for i in range(3)
    }
    saved |= {
        f"future-{label}": _save_frame(
            family["future_frames"][labels.index(label), -1], f"{chosen}_{label}", frames
        )
        for label in labels
    }
    return {
        "family": chosen,
        "cohort": "TRAIN-002 designed confirmation families (sealed)",
        "rule": f"true block displacement at {TRAIN_RULE} closest to the cohort median",
        "candidates": len(values),
        "median_true_displacement": float(np.median(list(values.values()))),
        "chosen_true_displacement": values[chosen],
        **states,
        "summary": summary,
        "arm_means": means,
        "models": models,
        "frames": saved,
        "sources": {
            "seal_sha256": sha256_file(root / "seal.json"),
            "family_sha256": designed[chosen]["file_sha256"],
            "readout_digest": payload["readouts"][READOUT],
        },
    }


def command_histogram(root: Path = SUPPORT_ROOT) -> dict[str, Any]:
    """Histogram of ten-control mean-command magnitudes over the sampled demo windows."""
    from jepa_action_effects import support001

    seal = support001.verify_seal(root)  # also checks map.npz against the seal
    with np.load(root / "map.npz") as arrays:
        magnitude = np.linalg.norm(np.asarray(arrays["a"], float), axis=1)
    counts, edges = np.histogram(magnitude, bins=HISTOGRAM_BINS)
    return {
        "windows": int(magnitude.size),
        "edges": _floats(edges),
        "counts": [int(c) for c in counts],
        "above_last_edge": int(np.sum(magnitude > edges[-1])),
        "map_npz_sha256": seal["map_npz_sha256"],
    }


def agent_in_view(root: Path = SUPPORT_ROOT) -> dict[str, Any]:
    """Share of SUPPORT-001 families whose agent ends fully inside the rendered table.

    Simulator states only. The kinematic agent is not stopped by the walls, so at high
    strengths it can leave the view, and its position can no longer be read from frames.
    """
    from jepa_action_effects import support001

    support001.verify_seal(root)
    collection = sealed(root / "collection.json")
    inside: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []
    for entry in (v for _, v in sorted(collection["families"].items())):
        family = _verified_family(root, entry)
        labels = list(family["metadata"]["candidate_ids"])
        end = np.asarray(family["future_states"], float)[:, -1, 0:2]
        fits = (end >= AGENT_RADIUS) & (end <= WORLD - AGENT_RADIUS)
        inside.setdefault(entry["scene"], []).append(fits.all(axis=1))
    return {
        scene: dict(zip(labels, np.mean(rows, axis=0).tolist(), strict=True))
        for scene, rows in sorted(inside.items())
    }


def build(out: Path = OUT, frames: Path = FRAMES) -> dict[str, Any]:
    payload = {
        "task": "REPORT-REDESIGN-001: real test examples for the course report (D039)",
        "note": " ".join(__doc__.split("\n\n")[2].split()),
        "geometry": {"bar": BAR, "stem": STEM, "agent_radius": AGENT_RADIUS, "world": WORLD},
        "support001": support_example(frames=frames),
        "train002": train_example(frames=frames),
        "demo_commands": command_histogram(),
        "agent_in_view": agent_in_view(),
    }
    write_json(out, payload)
    return {
        "support001": payload["support001"]["family"],
        "train002": payload["train002"]["family"],
        "frames": sum(len(payload[k]["frames"]) for k in ("support001", "train002")),
        "path": str(out),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    print(json.dumps(build()))


if __name__ == "__main__":
    main()
