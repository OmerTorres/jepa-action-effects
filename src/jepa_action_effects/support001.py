"""SUPPORT-001: archive support map and preregistered distortion-account test (D036).

    python -m jepa_action_effects.support001 map       # archive only, no model
    python -m jepa_action_effects.support001 fit       # opened TRAIN-001 frozen records
    python -m jepa_action_effects.support001 collect   # fresh six-dose families
    python -m jepa_action_effects.support001 cache     # real-frame latents only
    python -m jepa_action_effects.support001 gate      # real frames only
    python -m jepa_action_effects.support001 seal
    python -m jepa_action_effects.support001 confirm   # frozen predictor, frozen accounts
    python -m jepa_action_effects.support001 report

The collector mirrors TRAIN-001's family process (same proposals, admission and
identity rules) with six doses, because the sealed TRAIN-001 modules fix four.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.calib001_metrics import row_statistics, summarize_group
from jepa_action_effects.latent_audit import seal, sealed, write_json
from jepa_action_effects.next_common import (
    ARCHIVE_H5,
    DELTA,
    MANIFESTS,
    Stopwatch,
    encode_frames,
    load_family,
    load_model,
    predict_family,
    runtime_info,
)
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.support001_metrics import (
    ACCOUNTS,
    RADIUS,
    K,
    SupportMap,
    block_frame,
    decide,
    fit,
    pairwise,
    squared_errors,
)
from jepa_action_effects.train001_collect import DataIntegrityError
from jepa_action_effects.train001_protocol import READOUTS, SCENES, Protocol, magnitude_label

ROOT = Path("outputs/support001")
OUTPUTS = Path("outputs")
TRAIN001 = Path("outputs/train001")
TRAIN002 = Path("outputs/train002")
READOUT2 = Path("outputs/readout002")
MANIFEST = MANIFESTS / "support001-runtime.json"
DRAFT = Path("docs/SUPPORT-001_RESULT_DRAFT.md")
DOSES = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
FIT_DOSES = (0.15, 0.30, 0.45, 0.60)
OVER_RANGE = (0.80, 1.00)
GATE_DOSES = (0.60, 0.80, 1.00)
PER_SCENE = 150
SEED = 20261400
BOOTSTRAP_SEED = 20261422
DRAWS = 2000
STRIDE = 5
THRESHOLDS = Protocol()  # TRAIN-001 admission, identity and gate thresholds
STREAMS = {"challenge_proposals": 1, "challenge_order": 2, "archive": 3}


class GateFailure(RuntimeError):
    """A lineage, ordering or measurement prerequisite failed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _rng(stream: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, STREAMS[stream]]))


def _sources() -> dict[str, str]:
    here = Path(__file__).parent
    names = (
        "support001",
        "support001_metrics",
        "calib001_metrics",
        "train001_collect",
        "next_common",
    )
    return {name: sha256_file(here / f"{name}.py") for name in names}


def _h5():
    import h5py

    return h5py.File(ARCHIVE_H5, "r")


# --------------------------------------------------------------------------- map
def archive_windows(states: np.ndarray, actions: np.ndarray, offsets, lengths) -> dict:
    """Every stride-5 window with a 10-step history and a 10-step future."""
    rows = [
        int(start) + t
        for start, length in zip(offsets, lengths, strict=True)
        for t in range(10, int(length) - 10, STRIDE)  # t + 10 stays inside the episode
    ]
    index = np.asarray(rows)
    cumulative = np.concatenate([np.zeros((1, 2)), np.cumsum(actions, axis=0)])
    command = (cumulative[index + 10] - cumulative[index]) / 10.0
    return block_frame(
        states[index, 0:2],
        states[index - 5, 0:2],
        states[index, 2:4],
        states[index, 4],
        command,
        states[index + 10, 2:4] - states[index, 2:4],
    ) | {"index": index}


def support_map(root: Path = ROOT) -> dict[str, Any]:
    path = root / "map.npz"
    if path.exists():
        return json.loads((root / "map.json").read_text())
    with _h5() as h5:
        states = h5["state"][:].astype(np.float64)
        actions = h5["action"][:].astype(np.float64)
        offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
    windows = archive_windows(states, actions, offsets, lengths)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, s=windows["s"], a=windows["a"], d=windows["d"])
    magnitude = np.linalg.norm(windows["a"], axis=1)
    payload = {
        "windows": int(len(windows["s"])),
        "episodes": int(len(offsets)),
        "stride": STRIDE,
        "k": K,
        "radius": RADIUS,
        "command_magnitude_quantiles": dict(
            zip(
                ("p50", "p90", "p99", "p999"),
                np.quantile(magnitude, [0.5, 0.9, 0.99, 0.999]).tolist(),
                strict=True,
            )
        ),
        "map_sha256": sha256_file(path),
        "archive": str(ARCHIVE_H5),
    }
    write_json(root / "map.json", payload)
    return payload


def _load_map(root: Path) -> tuple[SupportMap, str]:
    payload = json.loads((root / "map.json").read_text())
    _check(sha256_file(root / "map.npz") == payload["map_sha256"], "Support map changed")
    with np.load(root / "map.npz") as arrays:
        return SupportMap(arrays["s"], arrays["a"], arrays["d"]), payload["map_sha256"]


# --------------------------------------------------------------------------- units
def family_units(family: dict, pred_end: np.ndarray, real: dict, labels: list[str]) -> dict:
    """Per-branch vectors for one family from decoded endpoints (block frame inputs).

    `pred_end` (B,2) and `real` {"start": (2,), "real_end": (B,2)} are block decodes.
    """
    history = np.asarray(family["history_states"], float)
    future = np.asarray(family["future_states"], float)
    controls = np.asarray(family["candidate_controls"], float)
    branches = len(labels)
    agent, previous = history[2, 0:2], history[1, 0:2]
    block, angle = history[2, 2:4], history[2, 4]
    command = controls.mean(axis=1)
    frame = block_frame(
        np.tile(agent, (branches, 1)),
        np.tile(previous, (branches, 1)),
        np.tile(block, (branches, 1)),
        np.full(branches, angle),
        command,
    )
    start = np.asarray(real["start"], float)
    return {
        "labels": labels,
        "t": future[:, -1, 2:4] - block,
        "p": np.asarray(pred_end, float) - start,
        "r": np.asarray(real["real_end"], float) - start,
        "s": frame["s"],
        "a": frame["a"],
        "angle": np.full(branches, angle),
        "magnitude": np.linalg.norm(command, axis=1),
    }


def _covariates(support: SupportMap, units: dict) -> dict:
    query = support.query(units["s"], units["a"], units["angle"])
    return {
        **units,
        "m": query["m"],
        "log_support": query["log_support"],
        "quantile": query["magnitude_quantile"],
    }


def _stack(rows: list[dict], keys) -> dict[str, np.ndarray]:
    return {key: np.concatenate([np.asarray(r[key]) for r in rows]) for key in keys}


def fit_stage(root: Path = ROOT) -> dict[str, Any]:
    """Fit the four accounts on opened TRAIN-001 frozen confirmation records."""
    path = root / "fit.json"
    if path.exists():
        return sealed(path)
    support, map_sha = _load_map(root)
    t1_seal = sealed(TRAIN001 / "seal.json")
    collection = sealed(TRAIN001 / "confirmation" / "collection.json")
    _check(
        sha256_file(TRAIN001 / "confirmation" / "collection.json")
        == t1_seal["file_sha256"]["collection_confirmation"],
        "TRAIN-001 confirmation collection changed",
    )
    records = TRAIN001 / "confirm" / "records"
    rows = []
    for unit, entry in sorted(collection["families"].items()):
        family_path = TRAIN001 / entry["relative_path"]
        _check(sha256_file(family_path) == entry["file_sha256"], f"{unit} family changed")
        family = load_family(family_path)
        frozen = json.loads((records / f"{unit}__frozen.json").read_text())
        real = json.loads((records / f"{unit}__real-frames.json").read_text())
        _check(
            frozen["fingerprint_sha256"] == t1_seal["fingerprints"]["frozen"], f"{unit} fingerprint"
        )
        labels = list(family["metadata"]["candidate_ids"])
        units = family_units(
            family, frozen["decoded"][READOUTS[0]]["pred_end"], real["decoded"][READOUTS[0]], labels
        )
        rows.append(_covariates(support, units) | {"scene": np.full(len(labels), entry["scene"])})
    data = _stack(rows, ("t", "p", "m", "log_support", "magnitude", "quantile", "scene", "labels"))
    fits = {
        name: fit(name, data["p"], data["t"], data["m"], data["log_support"], data["magnitude"])
        for name in ACCOUNTS
    }
    describe = {}
    for scene in SCENES:
        for dose in FIT_DOSES:
            rows_ = (data["scene"] == scene) & (data["labels"] == magnitude_label(dose))
            describe[f"{scene}:{magnitude_label(dose)}"] = {
                "n": int(rows_.sum()),
                "log_support_median": float(np.median(data["log_support"][rows_])),
                "magnitude_quantile_median": float(np.median(data["quantile"][rows_])),
                "typical_displacement_median": float(
                    np.median(np.linalg.norm(data["m"][rows_], axis=1))
                ),
                "true_displacement_median": float(
                    np.median(np.linalg.norm(data["t"][rows_], axis=1))
                ),
            }
    payload = {
        "label": "exploratory fit on opened TRAIN-001 frozen confirmation records (D036)",
        "readout": READOUTS[0],
        "units": int(len(data["t"])),
        "fits": fits,
        "covariates": describe,
        "map_sha256": map_sha,
        "train001_seal_sha256": sha256_file(TRAIN001 / "seal.json"),
    }
    seal(path, payload)
    return payload


# --------------------------------------------------------------------------- collection
def _candidates(direction: np.ndarray):
    from jepa_action_effects.simulator_replay import Candidate

    return [
        Candidate(magnitude_label(m), np.tile(direction * m, (10, 1)).astype(np.float32))
        for m in DOSES
    ]


def _save(directory: Path, family_id: str, recipe, candidates, replay, metadata: dict) -> dict:
    from jepa_action_effects.pilot_collection import save_family
    from jepa_action_effects.train001_collect import content_sha256, file_sha256

    labels = [magnitude_label(m) for m in DOSES]
    subset = type(replay)(
        family_id, replay.seed, replay.execution_order, {k: replay.branches[k] for k in labels}
    )
    path = directory / f"{family_id}.npz"
    save_family(path, recipe, [c for c in candidates if c.candidate_id in labels], subset, metadata)
    return {
        "id": family_id,
        "relative_path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha256(path),
        "content_sha256": content_sha256(load_family(path)),
    }


def _registry() -> dict[str, Any]:
    """Every earlier study, including TRAIN-001 (scanned) and TRAIN-002 (sealed collections)."""
    from jepa_action_effects.train001_collect import exclusion_registry

    registry = exclusion_registry(OUTPUTS, ROOT)
    seeds, episodes = set(registry["seeds"]), set(registry["episodes"])
    histories = set(registry["history_sha256"])
    recipes: set[str] = set()
    for relative in ("train_windows/collection.json", "confirmation/collection.json"):
        collection = sealed(TRAIN002 / relative)
        for record in [*collection["families"].values(), *collection["windows"].values()]:
            if record.get("scene") == "challenge" and record.get("seed") is not None:
                seeds.add(int(record["seed"]))
            if record.get("episode") is not None:
                episodes.add(int(record["episode"]))
            histories.add(record["history_sha256"])
            recipes.add(record["recipe_sha256"])
    return {
        "seeds": sorted(seeds),
        "episodes": sorted(episodes),
        "history_sha256": sorted(histories),
        "recipes": sorted(recipes),
        "sources": {**registry["sources"], "train002": "sealed collections"},
    }


def collect(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.pilot_collection import PilotConfig, propose_challenge
    from jepa_action_effects.shift001 import _archive_window, safe_goal
    from jepa_action_effects.shift001_metrics import is_rest_window
    from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family
    from jepa_action_effects.train001_collect import (
        _admit_common,
        archive_direction,
        archive_identity,
        array_sha256,
        identity_passes,
        recipe_sha256,
    )

    path = root / "collection.json"
    if path.exists():
        return sealed(path)
    _check((root / "fit.json").exists(), "Fit and seal the accounts before collecting")
    registry = _registry()
    write_json(root / "exclusions.json", registry)
    directory = root / "families"
    if directory.exists():
        for stale in directory.glob("*.npz"):
            stale.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    forbidden = set(registry["seeds"])
    seen_recipes = set(registry["recipes"])
    seen_histories = set(registry["history_sha256"])
    log: dict[str, int] = {}

    def count(reason: str) -> None:
        log[reason] = log.get(reason, 0) + 1

    families: dict[str, dict] = {}
    config = PilotConfig(master_seed=SEED)
    proposals, order_rng = _rng("challenge_proposals"), _rng("challenge_order")
    attempt = 0
    while sum(r["scene"] == "challenge" for r in families.values()) < PER_SCENE:
        _check(attempt < 30000, f"Challenge attempt cap reached; {log}")
        proposal, reason = propose_challenge(config, attempt, proposals)
        attempt += 1
        if proposal is None:
            count(f"challenge:{reason}")
            continue
        recipe, _, metadata = proposal
        recipe_hash = recipe_sha256(recipe.seed, recipe.warmup_controls, recipe.reset_options)
        if recipe.seed in forbidden or recipe_hash in seen_recipes:
            count("challenge:used")
            continue
        toward = np.asarray(metadata["approach_toward"], float)
        candidates = _candidates(toward)
        order = [c.candidate_id for c in candidates]
        order_rng.shuffle(order)
        try:
            replay = run_family(recipe, candidates, order)
        except RuntimeError:
            count("challenge:replay_error")
            continue
        moved = _admit_common(replay, THRESHOLDS, log, "challenge")
        if moved is None:
            continue
        history_hash = array_sha256(next(iter(replay.branches.values())).history_frames)
        if history_hash in seen_histories:
            count("challenge:duplicate_content")
            continue
        n = sum(r["scene"] == "challenge" for r in families.values())
        family_id = f"s1-challenge-{n:03d}"
        record = _save(
            directory,
            family_id,
            recipe,
            candidates,
            replay,
            {
                **metadata,
                "cohort": "challenge",
                "moved": moved,
                "direction": toward.tolist(),
                "recipe_sha256": recipe_hash,
            },
        )
        record |= {
            "scene": "challenge",
            "seed": int(recipe.seed),
            "episode": None,
            "recipe_sha256": recipe_hash,
            "history_sha256": history_hash,
            "moved": moved,
        }
        forbidden.add(int(recipe.seed))
        seen_recipes.add(recipe_hash)
        seen_histories.add(history_hash)
        families[family_id] = record
    log["challenge:attempts"] = attempt

    rng = _rng("archive")
    with _h5() as h5:
        lengths = h5["ep_len"][:]
        available = lengths >= 21
        available[[e for e in registry["episodes"] if e < len(available)]] = False
        attempt = 0
        while sum(r["scene"] == "archive" for r in families.values()) < PER_SCENE:
            _check(attempt < 80000, f"Archive attempt cap reached; {log}")
            attempt += 1
            episode = int(rng.choice(np.flatnonzero(available)))
            t = int(rng.integers(10, int(lengths[episode]) - 10))
            window = _archive_window(h5, episode, t)
            if window is None:
                count("archive:out_of_bounds")
                continue
            states, controls = window["states"], window["controls"]
            if not is_rest_window(states[:11, 2:4], THRESHOLDS.rest_limit):
                count("archive:block_not_at_rest")
                continue
            direction = archive_direction(controls[10:20], THRESHOLDS.direction_min_norm)
            if direction is None:
                count("archive:near_zero_direction")
                continue
            start_state = states[0].tolist()
            reset_options = {"state": start_state, "goal_state": safe_goal(start_state)}
            recipe_hash = recipe_sha256(0, controls[:10], reset_options)
            if recipe_hash in seen_recipes:
                count("archive:duplicate_recipe")
                continue
            recipe = ReplayRecipe(f"s1-archive-{attempt:05d}", 0, controls[:10], reset_options)
            candidates = [*_candidates(direction), Candidate("recorded", controls[10:20])]
            order = [c.candidate_id for c in candidates]
            rng.shuffle(order)
            try:
                replay = run_family(recipe, candidates, order)
            except RuntimeError:
                count("archive:replay_error")
                continue
            if not replay.branches["recorded"].completed:
                count("archive:incomplete_branch")
                continue
            identity = archive_identity(
                replay.branches["recorded"], states, h5["pixels"][window["start"] + t]
            )
            if not identity_passes(identity, THRESHOLDS):
                count("archive:identity_failed")
                continue
            moved = _admit_common(replay, THRESHOLDS, log, "archive")
            if moved is None:
                continue
            moved.pop("recorded", None)
            history_hash = array_sha256(replay.branches["recorded"].history_frames)
            if history_hash in seen_histories:
                count("archive:duplicate_content")
                continue
            n = sum(r["scene"] == "archive" for r in families.values())
            family_id = f"s1-archive-{n:03d}"
            record = _save(
                directory,
                family_id,
                recipe,
                candidates,
                replay,
                {
                    "cohort": "archive",
                    "moved": moved,
                    "episode": episode,
                    "t": t,
                    "direction": direction.tolist(),
                    "identity": identity,
                    "recipe_sha256": recipe_hash,
                },
            )
            record |= {
                "scene": "archive",
                "seed": 0,
                "episode": episode,
                "t": t,
                "recipe_sha256": recipe_hash,
                "history_sha256": history_hash,
                "moved": moved,
            }
            available[episode] = False
            seen_recipes.add(recipe_hash)
            seen_histories.add(history_hash)
            families[family_id] = record
        log["archive:attempts"] = attempt
    seeds = sorted(r["seed"] for r in families.values() if r["scene"] == "challenge")
    write_json(root / "forbidden-seeds.json", {"seeds": seeds})
    payload = {"families": families, "rejections": dict(sorted(log.items())), "doses": list(DOSES)}
    seal(path, payload)
    return {"families": len(families), "rejections": payload["rejections"]}


# --------------------------------------------------------------------------- cache
def _family(root: Path, entry: dict) -> dict:
    path = root / entry["relative_path"]
    if sha256_file(path) != entry["file_sha256"]:
        raise DataIntegrityError(f"{entry['id']} changed after admission")
    return load_family(path)


def cache(root: Path = ROOT) -> dict[str, Any]:
    """Encoded real history and future frames; no predictor is run."""
    manifest = root / "cache.json"
    if manifest.exists():
        return sealed(manifest)
    collection = sealed(root / "collection.json")
    model, _, fingerprint = load_model()
    records = {}
    for family_id, entry in sorted(collection["families"].items()):
        family = _family(root, entry)
        frames = np.concatenate(
            [family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3)]
        )
        z = encode_frames(model, frames)["projected"].astype(np.float32)
        path = root / "cache" / f"{family_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, real=z)
        records[family_id] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
    payload = {"fingerprint": fingerprint, "records": records}
    seal(manifest, payload)
    return {"families": len(records)}


def _real_latents(root: Path, record: dict) -> np.ndarray:
    path = root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise DataIntegrityError(f"Cache changed: {path.name}")
    with np.load(path) as arrays:
        return arrays["real"].astype(np.float64)


def decoders() -> tuple[dict, dict, dict]:
    """Block decoders (three MLPs), agent decoders (same MLPs, outputs 4-5) and digests."""
    from jepa_action_effects.readout_scale import mlp_predictor
    from jepa_action_effects.train001 import readout_decoders

    block, digests = readout_decoders(READOUT2)
    agent = {}
    for name in READOUTS:
        with np.load(READOUT2 / "models" / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        state = {
            "network": {k[4:]: v for k, v in arrays.items() if k.startswith("net.")},
            **{k: arrays[k] for k in ("x_mean", "x_scale", "y_mean", "y_scale")},
        }
        agent[name] = (lambda f: lambda x: f(np.asarray(x, np.float64))[:, 4:6])(
            mlp_predictor(state)
        )
    return block, agent, digests


def _real_decodes(real: np.ndarray, block, agent) -> dict:
    current, future = real[2:3], real[3:].reshape(len(DOSES), 2, -1)[:, 1]
    return {
        name: {
            "start": block[name](current)[0],
            "real_end": block[name](future),
            "agent_start": agent[name](current)[0],
            "agent_real_end": agent[name](future),
        }
        for name in block
    }


def gate(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.train001_metrics import cell_gate

    path = root / "gate.json"
    if path.exists():
        return sealed(path)
    _check(not (root / "seal.json").exists(), "Gate must precede the seal")
    collection, manifest = sealed(root / "collection.json"), sealed(root / "cache.json")
    block, agent, digests = decoders()
    true, real, scenes = [], [], []
    for family_id, entry in sorted(collection["families"].items()):
        family = _family(root, entry)
        decodes = _real_decodes(_real_latents(root, manifest["records"][family_id]), block, agent)[
            READOUTS[0]
        ]
        start = np.asarray(family["history_states"])[2, 2:4]
        true.append(np.asarray(family["future_states"])[:, -1, 2:4] - start)
        real.append(decodes["real_end"] - decodes["start"])
        scenes.append(entry["scene"])
    true, real, scenes = np.stack(true), np.stack(real), np.asarray(scenes)
    cells = {}
    for offset, (scene, dose) in enumerate((s, d) for s in SCENES for d in GATE_DOSES):
        b = DOSES.index(dose)
        rows = scenes == scene
        cells[f"{scene}:{magnitude_label(dose)}"] = cell_gate(
            true[rows, b],
            real[rows, b],
            np.linalg.norm(real[rows, b] - true[rows, b], axis=1),
            THRESHOLDS,
            BOOTSTRAP_SEED + 50 + offset,
        )
    payload = {
        "label": "real frames only; computed before any predictor scoring",
        "cells": cells,
        "readouts": digests,
        "cache_sha256": sha256_file(root / "cache.json"),
    }
    seal(path, payload)
    return {cell: value["passed"] for cell, value in cells.items()}


SEALED = ("map.json", "fit.json", "collection.json", "cache.json", "gate.json")


def seal_stage(root: Path = ROOT) -> dict[str, Any]:
    path = root / "seal.json"
    if path.exists():
        return sealed(path)
    for name in SEALED:
        _check((root / name).exists(), f"Missing {name}")
    _, _, digests = decoders()
    _check(sealed(root / "gate.json")["readouts"] == digests, "Readouts changed since gate")
    payload = {
        "task": "SUPPORT-001 (D036)",
        "sources": _sources(),
        "file_sha256": {name: sha256_file(root / name) for name in SEALED},
        "map_npz_sha256": sha256_file(root / "map.npz"),
        "readouts": digests,
        "sealed_at": time.time(),
    }
    seal(path, payload)
    return {"sealed": True}


def verify_seal(root: Path = ROOT) -> dict[str, Any]:
    payload = sealed(root / "seal.json")
    _check(payload["sources"] == _sources(), "Source changed after the seal")
    for name, digest in payload["file_sha256"].items():
        _check(sha256_file(root / name) == digest, f"{name} changed after the seal")
    _check(sha256_file(root / "map.npz") == payload["map_npz_sha256"], "map.npz changed")
    return payload


# --------------------------------------------------------------------------- confirm
def _cells(units_by_family: list[dict], scenes: np.ndarray, readout: str, kind: str) -> dict:
    """calib001 sufficient statistics per scene × dose (block or agent)."""
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for family, scene in zip(units_by_family, scenes, strict=True):
        rows = row_statistics(family[f"{kind}_p"], family[f"{kind}_r"], family[f"{kind}_t"], DELTA)
        unit = {f"{label}": rows[i] for i, label in enumerate(family["labels"])}
        groups.setdefault(scene, []).append(unit)
    out = {}
    for offset, (scene, units) in enumerate(sorted(groups.items())):
        names = sorted({n for u in units for n in u})
        sums = np.stack([[u[n] for n in names] for u in units])
        out[scene] = summarize_group(names, sums, DRAWS, BOOTSTRAP_SEED + 10 + offset)["cells"]
    return {f"{readout}:{kind}": out}


def confirm(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.model_inference import ActionScaler
    from jepa_action_effects.next_common import STATS
    from jepa_action_effects.train001_metrics import paired_bootstrap

    payload = verify_seal(root)
    summary_path = root / "summary.json"
    if summary_path.exists():
        return sealed(summary_path)
    write_json(
        root / "opened.json", {"seal_sha256": sha256_file(root / "seal.json"), "at": time.time()}
    )
    collection, manifest = sealed(root / "collection.json"), sealed(root / "cache.json")
    fits = sealed(root / "fit.json")["fits"]
    gates = sealed(root / "gate.json")["cells"]
    support, _ = _load_map(root)
    block, agent, digests = decoders()
    _check(digests == payload["readouts"], "Readouts changed after the seal")
    model, _, _ = load_model()
    scaler = ActionScaler.from_json(STATS)
    labels = [magnitude_label(d) for d in DOSES]
    (root / "predicted").mkdir(exist_ok=True)
    per_readout = {name: [] for name in READOUTS}
    scenes = []
    for family_id, entry in sorted(collection["families"].items()):
        family = _family(root, entry)
        _check(list(family["metadata"]["candidate_ids"]) == labels, f"{family_id} branch order")
        predicted = predict_family(model, scaler, family).astype(np.float64)
        np.savez_compressed(root / "predicted" / f"{family_id}.npz", predicted=predicted)
        real = _real_latents(root, manifest["records"][family_id])
        decodes = _real_decodes(real, block, agent)
        history = np.asarray(family["history_states"], float)
        agent_true = np.tile(history[2, 0:2], (len(DOSES), 1))
        agent_future = np.asarray(family["future_states"], float)[:, -1, 0:2]
        for name in READOUTS:
            units = family_units(family, block[name](predicted[:, 1]), decodes[name], labels)
            units = _covariates(support, units)
            units |= {
                "block_p": units["p"],
                "block_r": units["r"],
                "block_t": units["t"],
                "agent_p": agent[name](predicted[:, 1]) - decodes[name]["agent_start"],
                "agent_r": decodes[name]["agent_real_end"] - decodes[name]["agent_start"],
                "agent_t": agent_future - agent_true,
            }
            per_readout[name].append(units)
        scenes.append(entry["scene"])
    scenes = np.asarray(scenes)
    scene_index = np.asarray([SCENES.index(s) for s in scenes])
    over = [DOSES.index(d) for d in OVER_RANGE]
    gated = {
        scene: [b for b in over if gates[f"{scene}:{magnitude_label(DOSES[b])}"]["passed"]]
        for scene in SCENES
    }
    result: dict[str, Any] = {"families": int(len(scenes)), "gated_branches": gated, "readouts": {}}
    for name in READOUTS:
        families = per_readout[name]
        errors = {acc: [] for acc in ACCOUNTS}
        in_range = {acc: [] for acc in ACCOUNTS}
        for units in families:
            e = squared_errors(
                fits, units["p"], units["t"], units["m"], units["log_support"], units["magnitude"]
            )
            for acc in ACCOUNTS:
                in_range[acc].append(float(np.mean(e[acc][: DOSES.index(0.60) + 1])))
                errors[acc].append(e[acc])
        keep = np.asarray([bool(gated[s]) for s in scenes])
        over_errors = {
            acc: np.asarray(
                [
                    float(np.mean(err[gated[s]])) if gated[s] else np.nan
                    for err, s in zip(errors[acc], scenes, strict=True)
                ]
            )
            for acc in ACCOUNTS
        }
        block_result: dict[str, Any] = {
            "mean_over_range": {acc: float(np.nanmean(v)) for acc, v in over_errors.items()},
            "mean_in_range": {acc: float(np.mean(v)) for acc, v in in_range.items()},
        }
        if keep.any():
            boot = paired_bootstrap(
                pairwise({a: v[keep] for a, v in over_errors.items()}),
                scene_index[keep],
                DRAWS,
                BOOTSTRAP_SEED,
            )
            block_result["over_range_bootstrap"] = boot
            block_result["decision"] = decide(boot)
        else:
            block_result["decision"] = {"verdict": "not_assessable", "winners": []}
        block_result["in_range_bootstrap"] = paired_bootstrap(
            pairwise({a: np.asarray(v) for a, v in in_range.items()}),
            scene_index,
            DRAWS,
            BOOTSTRAP_SEED + 1,
        )
        block_result["dose_response"] = _cells(families, scenes, name, "block")
        block_result["agent_probe"] = _cells(families, scenes, name, "agent")
        block_result["covariates"] = {
            f"{scene}:{label}": {
                "log_support_median": float(
                    np.median(
                        [
                            u["log_support"][i]
                            for u, s in zip(families, scenes, strict=True)
                            if s == scene
                        ]
                    )
                ),
                "magnitude_quantile_median": float(
                    np.median(
                        [
                            u["quantile"][i]
                            for u, s in zip(families, scenes, strict=True)
                            if s == scene
                        ]
                    )
                ),
            }
            for scene in SCENES
            for i, label in enumerate(labels)
        }
        result["readouts"][name] = block_result
    primary = result["readouts"][READOUTS[0]]
    result["verdict"] = primary["decision"]["verdict"]
    result["readout_agreement"] = (
        len({result["readouts"][r]["decision"]["verdict"] for r in READOUTS}) == 1
    )
    seal(summary_path, result)
    return {"verdict": result["verdict"], "agreement": result["readout_agreement"]}


def report(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.support001_report import draft

    payload = verify_seal(root)
    summary = sealed(root / "summary.json")
    manifest = {
        "task": "SUPPORT-001 (D036)",
        "status": "complete",
        "map": json.loads((root / "map.json").read_text()),
        "fit": sealed(root / "fit.json"),
        "gate": sealed(root / "gate.json")["cells"],
        "collection_rejections": sealed(root / "collection.json")["rejections"],
        "summary": summary,
        "seal_sha256": sha256_file(root / "seal.json"),
        "summary_sha256": sha256_file(root / "summary.json"),
        "sources": payload["sources"],
        "runtime": runtime_info(),
    }
    DRAFT.write_text(draft(manifest))
    write_json(MANIFEST, manifest)
    return {"verdict": summary["verdict"]}


STAGES = {
    "map": support_map,
    "fit": fit_stage,
    "collect": collect,
    "cache": cache,
    "gate": gate,
    "seal": seal_stage,
    "confirm": confirm,
    "report": report,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "all"])
    stage = parser.parse_args(argv).stage
    ROOT.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch(ROOT)
    for name in STAGES if stage == "all" else [stage]:
        try:
            result = watch.run(name, STAGES[name])
        except (GateFailure, DataIntegrityError) as error:
            print(json.dumps({"stage": name, "status": "stopped", "detail": str(error)}))
            sys.exit(2)
        print(
            json.dumps({"stage": name, "status": "completed", "result": result}, default=str)[:2000]
        )


if __name__ == "__main__":
    main()
