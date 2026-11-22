"""TRAIN-001 collection: exclusions, archive weights, families, ordinary windows.

Admission and integrity are separate (D028):

- A *proposal* (challenge seed, archive episode/window) that fails a rule is rejected
  and counted by reason. This never stops the study.
- An *admitted* artifact that later violates its identity, split or hash contract is a
  `DataIntegrityError`, as are an exhausted attempt cap and a sparse weight bin.

Identities are source-specific: challenge families by reset seed and recipe hash;
archive families and windows by archive episode, window start and recipe hash (every
archive replay uses simulator seed 0). Every family also records a content hash.
Selection uses simulator/archive outcomes only; nothing here touches a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.train001_budget import Budget, NullBudget
from jepa_action_effects.train001_protocol import (
    MAGNITUDES,
    MOVE_REQUIRED,
    SCENES,
    Protocol,
    magnitude_label,
)

STREAMS = {
    "weights": 0,
    ("development", "challenge_proposals"): 1,
    ("development", "challenge_order"): 2,
    ("development", "archive"): 3,
    ("development", "windows"): 4,
    ("development", "partition"): 5,
    ("confirmation", "challenge_proposals"): 11,
    ("confirmation", "challenge_order"): 12,
    ("confirmation", "archive"): 13,
    ("confirmation", "windows"): 14,
    ("confirmation", "partition"): 15,
}
PHASE_TAG = {"development": "dev", "confirmation": "conf"}
FORBIDDEN_SEED_FILES = ("phys002", "planning001", "planning002", "shift001", "shift002")
FAMILY_SCAN_ROOTS = (
    "pilot001",
    "readout001",
    "phys002",
    "planning001",
    "planning002",
    "shift001",
    "shift001-prerevision",
    "shift002",
)
MODEL001_EPISODES = tuple(range(32))


class DataIntegrityError(RuntimeError):
    """An admitted artifact or the fixed cohort contract is violated."""


# --------------------------------------------------------------------------- hashing
def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{values.dtype.str}{values.shape}".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def recipe_sha256(seed: int, warmup: np.ndarray, reset_options: dict) -> str:
    return canonical_sha256(
        {
            "seed": int(seed),
            "warmup": np.asarray(warmup, np.float32).tolist(),
            "reset_options": reset_options,
        }
    )


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash every non-metadata array of a saved family or window."""
    return canonical_sha256(
        {key: array_sha256(value) for key, value in sorted(arrays.items()) if key != "metadata"}
    )


def file_sha256(path: Path) -> str:
    from jepa_action_effects.pilot_collection import sha256_file

    return sha256_file(path)


def raw_frame_mae(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute difference on the raw uint8 scale (0-255), as in SHIFT-002."""
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise TypeError("Identity frames must be raw uint8 images")
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean())


def rng_for(protocol: Protocol, key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([protocol.collection_seed, STREAMS[key]]))


def _count(log: dict[str, int], reason: str) -> None:
    log[reason] = log.get(reason, 0) + 1


# --------------------------------------------------------------------------- exclusions
def _load_npz_metadata(path: Path) -> dict | None:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            return None
        metadata = json.loads(str(archive["metadata"].item()))
        history = archive["history_frames"] if "history_frames" in archive.files else None
    if not isinstance(metadata, dict):
        return None
    metadata["_history_sha256"] = array_sha256(history) if history is not None else None
    return metadata


def exclusion_registry(outputs: Path, current_root: Path) -> dict[str, Any]:
    """Seeds, archive episodes and history-frame hashes used by every earlier study.

    Also includes any other TRAIN-001 root (for example the smoke run) so that its
    families are excluded from the study. Known sources must be readable.
    """
    seeds: set[int] = set()
    episodes: set[int] = set(MODEL001_EPISODES)
    histories: set[str] = set()
    sources: dict[str, dict[str, int]] = {"model001_smoke_episodes": {"episodes": 32}}
    for study in FORBIDDEN_SEED_FILES:
        path = outputs / study / "forbidden-seeds.json"
        values = json.loads(path.read_text())["seeds"]
        seeds.update(int(seed) for seed in values)
        sources[f"{study}/forbidden-seeds.json"] = {"seeds": len(values)}
    for study in ("shift001", "shift002"):
        config = json.loads((outputs / study / "config.json").read_text())
        values = config["excluded_archive_episodes"]
        episodes.update(int(episode) for episode in values)
        sources[f"{study}/config.json"] = {"episodes": len(values)}
    dist = outputs / "dist001"
    identity = json.loads((dist / "identity.json").read_text())
    gap = json.loads((dist / "gap.json").read_text())
    dist_episodes = {*identity["episodes"], *(row["episode"] for row in gap["rows"])}
    episodes.update(int(episode) for episode in dist_episodes)
    sources["dist001"] = {"episodes": len(dist_episodes)}
    scan_roots = [outputs / name for name in FAMILY_SCAN_ROOTS]
    scan_roots += [
        path
        for path in sorted(outputs.glob("train001*"))
        if path.is_dir() and path.resolve() != current_root.resolve() and "budget" not in path.name
    ]
    for root in scan_roots:
        if not root.exists():
            raise DataIntegrityError(f"Known exclusion source missing: {root}")
        counts = {"families": 0, "seeds": 0, "episodes": 0}
        for path in sorted(root.rglob("*.npz")):
            if "cache" in path.parts:
                continue
            metadata = _load_npz_metadata(path)
            if metadata is None:
                continue
            counts["families"] += 1
            if "seed" in metadata and metadata.get("cohort") != "archive":
                seeds.add(int(metadata["seed"]))
                counts["seeds"] += 1
            if "episode" in metadata:
                episodes.add(int(metadata["episode"]))
                counts["episodes"] += 1
            for episode, _ in metadata.get("borrowed", []):
                episodes.add(int(episode))
            if metadata["_history_sha256"]:
                histories.add(metadata["_history_sha256"])
        weights = root / "development" / "weights.json"
        if weights.exists():
            values = json.loads(weights.read_text())["episodes"]
            episodes.update(int(episode) for episode in values)
            counts["episodes"] += len(values)
        sources[str(root.relative_to(outputs))] = counts
    return {
        "seeds": sorted(seeds),
        "episodes": sorted(episodes),
        "history_sha256": sorted(histories),
        "sources": sources,
    }


# --------------------------------------------------------------------------- weights
def window_norms(controls: np.ndarray, window: int = 10) -> tuple[np.ndarray, int]:
    """Mean per-control L2 norm of non-overlapping windows from the episode start."""
    values = np.asarray(controls, dtype=np.float64)
    count = len(values) // window
    blocks = values[: count * window].reshape(count, window, 2)
    valid = np.all(np.isfinite(blocks), axis=(1, 2)) & np.all(np.abs(blocks) <= 1, axis=(1, 2))
    norms = np.linalg.norm(blocks[valid], axis=2).mean(axis=1)
    return norms, int(np.sum(~valid))


def bin_weights(norms: np.ndarray, edges: Iterable[float], min_count: int) -> dict[str, Any]:
    edges = tuple(float(edge) for edge in edges)
    bins = np.digitize(np.asarray(norms, dtype=np.float64), edges, right=False)
    counts = np.bincount(bins, minlength=len(edges) + 1).astype(int)
    if np.any(counts < min_count):
        raise DataIntegrityError(
            f"Archive weight bins {counts.tolist()} have fewer than {min_count} windows"
        )
    weights = counts / counts.sum()
    return {"edges": list(edges), "counts": counts.tolist(), "weights": weights.tolist()}


def weight_sample(h5, rng: np.random.Generator, available: np.ndarray, protocol: Protocol):
    eligible = np.flatnonzero(available)
    if len(eligible) < protocol.weight_episodes:
        raise DataIntegrityError("Not enough eligible archive episodes for the weight sample")
    episodes = np.sort(rng.choice(eligible, size=protocol.weight_episodes, replace=False))
    offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
    norms, skipped = [], 0
    for episode in episodes:
        start, length = int(offsets[episode]), int(lengths[episode])
        values, invalid = window_norms(h5["action"][start : start + length], protocol.weight_window)
        norms.append(values)
        skipped += invalid
    norms_array = np.concatenate(norms)
    result = bin_weights(norms_array, protocol.weight_edges, protocol.weight_min_bin)
    available[episodes] = False
    return {
        **result,
        "episodes": [int(episode) for episode in episodes],
        "windows": int(len(norms_array)),
        "skipped_windows": skipped,
        "norm_quantiles": np.quantile(norms_array, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
    }


# --------------------------------------------------------------------------- families
def moved_labels(replay, delta: float) -> dict[str, bool]:
    from jepa_action_effects.pilot_collection import _states_from_steps

    moved = {}
    for key, branch in replay.branches.items():
        history, future, _ = _states_from_steps(branch)
        moved[key] = bool(np.linalg.norm(future[-1, 2:4] - history[-1, 2:4]) > delta)
    return moved


def histories_identical(replay) -> bool:
    branches = list(replay.branches.values())
    return all(
        np.array_equal(branch.history_frames, branches[0].history_frames) for branch in branches
    )


def _magnitude_candidates(direction: np.ndarray):
    from jepa_action_effects.simulator_replay import Candidate

    return [
        Candidate(magnitude_label(m), np.tile(direction * m, (10, 1)).astype(np.float32))
        for m in MAGNITUDES
    ]


def _save(
    directory: Path, family_id: str, recipe, candidates, replay, metadata: dict
) -> dict[str, Any]:
    from jepa_action_effects.pilot_collection import load_family, save_family

    labels = [magnitude_label(m) for m in MAGNITUDES]
    subset = type(replay)(
        family_id, replay.seed, replay.execution_order, {k: replay.branches[k] for k in labels}
    )
    path = directory / f"{family_id}.npz"
    save_family(path, recipe, [c for c in candidates if c.candidate_id in labels], subset, metadata)
    arrays = load_family(path)
    return {
        "id": family_id,
        "path": str(path),
        "file_sha256": file_sha256(path),
        "content_sha256": content_sha256(arrays),
    }


def _admit_common(replay, protocol: Protocol, log: dict, prefix: str) -> dict[str, bool] | None:
    if not all(branch.completed for branch in replay.branches.values()):
        _count(log, f"{prefix}:incomplete_branch")
        return None
    if not histories_identical(replay):
        _count(log, f"{prefix}:history_mismatch")
        return None
    moved = moved_labels(replay, protocol.delta)
    if not all(moved[magnitude_label(m)] for m in MOVE_REQUIRED):
        _count(log, f"{prefix}:movement_criterion")
        return None
    return moved


def collect_challenge(
    directory: Path,
    phase: str,
    protocol: Protocol,
    forbidden_seeds: set[int],
    seen: dict[str, set[str]],
    log: dict[str, int],
    budget: Budget,
) -> list[dict[str, Any]]:
    from jepa_action_effects.pilot_collection import PilotConfig, propose_challenge
    from jepa_action_effects.simulator_replay import run_family

    config = PilotConfig(master_seed=protocol.collection_seed)
    proposals = rng_for(protocol, (phase, "challenge_proposals"))
    order_rng = rng_for(protocol, (phase, "challenge_order"))
    target = protocol.target(phase)
    accepted: list[dict[str, Any]] = []
    attempts = 0
    while len(accepted) < target:
        if attempts >= protocol.attempt_caps["challenge"]:
            raise DataIntegrityError(
                f"Challenge attempt cap reached with {len(accepted)}/{target} families; {log}"
            )
        attempt = attempts
        attempts += 1
        budget.unit("proposal")
        proposal, reason = propose_challenge(config, attempt, proposals)
        if proposal is None:
            _count(log, f"challenge:{reason}")
            continue
        recipe, _, metadata = proposal
        if recipe.seed in forbidden_seeds:
            _count(log, "challenge:reset_seed_used")
            continue
        recipe_hash = recipe_sha256(recipe.seed, recipe.warmup_controls, recipe.reset_options)
        if recipe_hash in seen["recipe"]:
            _count(log, "challenge:duplicate_recipe")
            continue
        toward = np.asarray(metadata["approach_toward"], dtype=np.float64)
        candidates = _magnitude_candidates(toward)
        order = [c.candidate_id for c in candidates]
        order_rng.shuffle(order)
        try:
            replay = run_family(recipe, candidates, order)
        except RuntimeError:
            _count(log, "challenge:replay_error")
            continue
        moved = _admit_common(replay, protocol, log, "challenge")
        if moved is None:
            continue
        history_hash = array_sha256(next(iter(replay.branches.values())).history_frames)
        if history_hash in seen["history"]:
            _count(log, "challenge:duplicate_content")
            continue
        family_id = f"challenge-{PHASE_TAG[phase]}-{len(accepted):03d}"
        record = _save(
            directory,
            family_id,
            recipe,
            candidates,
            replay,
            {
                **metadata,
                "cohort": "challenge",
                "phase": phase,
                "moved": moved,
                "direction": toward.tolist(),
                "recipe_sha256": recipe_hash,
            },
        )
        record.update(
            {
                "scene": "challenge",
                "seed": int(recipe.seed),
                "episode": None,
                "t": None,
                "attempt": attempt,
                "recipe_sha256": recipe_hash,
                "history_sha256": history_hash,
                "moved": moved,
            }
        )
        forbidden_seeds.add(int(recipe.seed))
        seen["recipe"].add(recipe_hash)
        seen["history"].add(history_hash)
        accepted.append(record)
    log["challenge:attempts"] = attempts
    return accepted


def archive_identity(reference, states: np.ndarray, archive_frame: np.ndarray) -> dict[str, float]:
    """Replay identity of the recorded-future branch against the archive window."""
    from jepa_action_effects.pilot_collection import _states_from_steps

    history, future, _ = _states_from_steps(reference)
    position = float(
        np.max(
            np.linalg.norm(
                history[:, :4].reshape(3, 2, 2) - states[[0, 5, 10], :4].reshape(3, 2, 2), axis=2
            )
        )
    )
    return {
        "position": position,
        "frame_raw_mae": raw_frame_mae(reference.history_frames[-1], archive_frame),
        "own_future": float(np.linalg.norm(future[-1, 2:4] - states[20, 2:4])),
    }


def identity_passes(identity: dict[str, float], protocol: Protocol) -> bool:
    return (
        identity["position"] <= protocol.identity_position
        and identity["frame_raw_mae"] <= protocol.identity_frame_raw_mae
        and identity["own_future"] <= protocol.identity_own_future
    )


def archive_direction(controls: np.ndarray, minimum: float) -> np.ndarray | None:
    """Normalized mean of the recorded ten-control future, or None if near zero."""
    mean = np.asarray(controls, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < minimum:
        return None
    return mean / norm


def collect_archive(
    directory: Path,
    phase: str,
    protocol: Protocol,
    h5,
    available: np.ndarray,
    seen: dict[str, set[str]],
    log: dict[str, int],
    budget: Budget,
) -> list[dict[str, Any]]:
    from jepa_action_effects.shift001 import _archive_window, safe_goal
    from jepa_action_effects.shift001_metrics import is_rest_window
    from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family

    rng = rng_for(protocol, (phase, "archive"))
    lengths = h5["ep_len"][:]
    target = protocol.target(phase)
    accepted: list[dict[str, Any]] = []
    attempts = 0
    while len(accepted) < target:
        if attempts >= protocol.attempt_caps["archive"]:
            raise DataIntegrityError(
                f"Archive attempt cap reached with {len(accepted)}/{target} families; {log}"
            )
        attempt = attempts
        attempts += 1
        budget.unit("proposal")
        pool = np.flatnonzero(available)
        if len(pool) == 0:
            raise DataIntegrityError("Archive episode pool exhausted")
        episode = int(rng.choice(pool))
        t = int(rng.integers(10, int(lengths[episode]) - 10))
        window = _archive_window(h5, episode, t)
        if window is None:
            _count(log, "archive:out_of_bounds")
            continue
        states, controls = window["states"], window["controls"]
        if not is_rest_window(states[:11, 2:4], protocol.rest_limit):
            _count(log, "archive:block_not_at_rest")
            continue
        direction = archive_direction(controls[10:20], protocol.direction_min_norm)
        if direction is None:
            _count(log, "archive:near_zero_direction")
            continue
        start_state = states[0].tolist()
        reset_options = {"state": start_state, "goal_state": safe_goal(start_state)}
        recipe_hash = recipe_sha256(0, controls[:10], reset_options)
        if recipe_hash in seen["recipe"]:
            _count(log, "archive:duplicate_recipe")
            continue
        recipe = ReplayRecipe(
            f"archive-{PHASE_TAG[phase]}-{attempt:05d}", 0, controls[:10], reset_options
        )
        candidates = [*_magnitude_candidates(direction), Candidate("recorded", controls[10:20])]
        order = [c.candidate_id for c in candidates]
        rng.shuffle(order)
        try:
            replay = run_family(recipe, candidates, order)
        except RuntimeError:
            _count(log, "archive:replay_error")
            continue
        if not replay.branches["recorded"].completed:
            _count(log, "archive:incomplete_branch")
            continue
        archive_frame = h5["pixels"][window["start"] + t]
        identity = archive_identity(replay.branches["recorded"], states, archive_frame)
        if not identity_passes(identity, protocol):
            _count(log, "archive:identity_failed")
            continue
        moved = _admit_common(replay, protocol, log, "archive")
        if moved is None:
            continue
        moved.pop("recorded", None)
        history_hash = array_sha256(replay.branches["recorded"].history_frames)
        if history_hash in seen["history"]:
            _count(log, "archive:duplicate_content")
            continue
        family_id = f"archive-{PHASE_TAG[phase]}-{len(accepted):03d}"
        record = _save(
            directory,
            family_id,
            recipe,
            candidates,
            replay,
            {
                "cohort": "archive",
                "phase": phase,
                "moved": moved,
                "episode": episode,
                "t": t,
                "direction": direction.tolist(),
                "direction_norm": float(np.linalg.norm(controls[10:20].mean(axis=0))),
                "identity": identity,
                "recipe_sha256": recipe_hash,
            },
        )
        record.update(
            {
                "scene": "archive",
                "seed": 0,
                "episode": episode,
                "t": t,
                "attempt": attempt,
                "recipe_sha256": recipe_hash,
                "history_sha256": history_hash,
                "moved": moved,
                "identity": identity,
            }
        )
        available[episode] = False
        seen["recipe"].add(recipe_hash)
        seen["history"].add(history_hash)
        accepted.append(record)
    log["archive:attempts"] = attempts
    return accepted


def collect_windows(
    directory: Path,
    phase: str,
    protocol: Protocol,
    h5,
    available: np.ndarray,
    log: dict[str, int],
    budget: Budget,
) -> list[dict[str, Any]]:
    """Ordinary archive windows read directly from the archive, no movement filter."""
    rng = rng_for(protocol, (phase, "windows"))
    offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
    target = protocol.ordinary_target(phase)
    accepted: list[dict[str, Any]] = []
    attempts = 0
    while len(accepted) < target:
        if attempts >= protocol.attempt_caps["ordinary"]:
            raise DataIntegrityError(f"Ordinary-window attempt cap reached; {log}")
        attempts += 1
        budget.unit("proposal")
        episode = int(rng.choice(np.flatnonzero(available)))
        start, length = int(offsets[episode]), int(lengths[episode])
        t = int(rng.integers(10, length - 10))
        controls = h5["action"][start + t - 10 : start + t + 10].astype(np.float32)
        if np.any(np.abs(controls) > 1) or not np.isfinite(controls).all():
            _count(log, "ordinary:out_of_bounds")
            continue
        indices = [start + t + k for k in (-10, -5, 0, 5, 10)]
        states = np.stack([h5["state"][i] for i in indices]).astype(np.float64)
        frames = np.stack([h5["pixels"][i] for i in indices])
        window_id = f"ordinary-{PHASE_TAG[phase]}-{len(accepted):03d}"
        path = directory / f"{window_id}.npz"
        arrays = {"frames": frames, "controls": controls, "states": states}
        metadata = {"cohort": "ordinary", "phase": phase, "episode": episode, "t": t}
        _atomic_npz(path, arrays, metadata)
        available[episode] = False
        accepted.append(
            {
                "id": window_id,
                "scene": "ordinary",
                "path": str(path),
                "file_sha256": file_sha256(path),
                "content_sha256": content_sha256(arrays),
                "history_sha256": array_sha256(frames[:3]),
                "recipe_sha256": canonical_sha256({"episode": episode, "t": t}),
                "episode": episode,
                "t": t,
                "seed": None,
            }
        )
    log["ordinary:attempts"] = attempts
    return accepted


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, metadata=np.array(json.dumps(metadata)), **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_window(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files if key != "metadata"}
        result["metadata"] = json.loads(str(archive["metadata"].item()))
    return result


# --------------------------------------------------------------------------- partitions
def assign_partitions(
    records: list[dict[str, Any]], phase: str, protocol: Protocol, rng: np.random.Generator
) -> None:
    """Seeded per-scene permutation into train/validation (development) or confirmation."""
    for scene in SCENES:
        members = [record for record in records if record["scene"] == scene]
        order = rng.permutation(len(members))
        for rank, index in enumerate(order):
            record = members[index]
            if phase == "confirmation":
                record["partition"], record["rank"] = "confirmation", rank
                continue
            if rank < protocol.train_per_scene:
                record["partition"], record["rank"] = "train", rank
                record["pilot"] = rank < protocol.pilot_train_per_scene
                record["overfit"] = rank < protocol.overfit_families_per_scene
            else:
                record["partition"] = "validation"
                record["rank"] = rank - protocol.train_per_scene
                record["pilot"] = record["rank"] < protocol.pilot_validation_per_scene


# --------------------------------------------------------------------------- integrity
def verify_integrity(
    collections: list[dict[str, Any]],
    registry: dict[str, Any],
    root: Path | None = None,
) -> dict[str, int]:
    """Post-admission contract: unique identities, disjoint from all prior use.

    Raises `DataIntegrityError` on any overlap, duplicate or (with `root`) changed bytes.
    """
    ids: set[str] = set()
    seeds: set[int] = set()
    episodes: set[int] = set()
    recipes: set[str] = set()
    histories: set[str] = set()
    prior_seeds = set(registry["seeds"])
    prior_episodes = set(registry["episodes"])
    prior_histories = set(registry["history_sha256"])
    count = 0

    def claim(pool: set, value, label: str) -> None:
        if value in pool:
            raise DataIntegrityError(f"Duplicate {label}: {value}")
        pool.add(value)

    for collection in collections:
        for episode in collection.get("weights", {}).get("episodes", []):
            if episode in prior_episodes:
                raise DataIntegrityError(f"Weight episode {episode} used by an earlier study")
            claim(episodes, int(episode), "archive episode")
        for record in [*collection["families"].values(), *collection["windows"].values()]:
            count += 1
            claim(ids, record["id"], "record id")
            claim(recipes, record["recipe_sha256"], "recipe hash")
            claim(histories, record["history_sha256"], "history content hash")
            if record["history_sha256"] in prior_histories:
                raise DataIntegrityError(f"{record['id']} duplicates an earlier family")
            if record["scene"] == "challenge":
                if record["seed"] in prior_seeds:
                    raise DataIntegrityError(f"{record['id']} reuses an earlier reset seed")
                claim(seeds, int(record["seed"]), "reset seed")
            else:
                if record["episode"] in prior_episodes:
                    raise DataIntegrityError(f"{record['id']} reuses an earlier archive episode")
                claim(episodes, int(record["episode"]), "archive episode")
            if root is not None:
                path = root / record["relative_path"]
                if not path.exists() or file_sha256(path) != record["file_sha256"]:
                    raise DataIntegrityError(f"{record['id']} bytes changed or missing")
    return {"records": count, "episodes": len(episodes), "seeds": len(seeds)}


def expected_counts(phase: str, protocol: Protocol) -> dict[str, int]:
    if phase == "development":
        return {
            "challenge:train": protocol.train_per_scene,
            "challenge:validation": protocol.validation_per_scene,
            "archive:train": protocol.train_per_scene,
            "archive:validation": protocol.validation_per_scene,
            "ordinary": protocol.ordinary_validation,
        }
    return {
        "challenge:confirmation": protocol.confirmation_per_scene,
        "archive:confirmation": protocol.confirmation_per_scene,
        "ordinary": protocol.ordinary_confirmation,
    }


def observed_counts(collection: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in collection["families"].values():
        key = f"{record['scene']}:{record['partition']}"
        counts[key] = counts.get(key, 0) + 1
    counts["ordinary"] = len(collection["windows"])
    return counts


# --------------------------------------------------------------------------- stage
def collect_phase(
    phase: str,
    protocol: Protocol,
    root: Path,
    registry: dict[str, Any],
    h5,
    prior: list[dict[str, Any]],
    budget: Budget | None = None,
) -> dict[str, Any]:
    """Collect one phase deterministically. A restart clears the unsealed phase directory."""
    budget = budget or NullBudget()
    directory = root / phase
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "families").mkdir(parents=True)
    (directory / "windows").mkdir(parents=True)
    forbidden = set(registry["seeds"])
    seen = {"recipe": set(), "history": set(registry["history_sha256"])}
    available = h5["ep_len"][:] >= 21
    available[[e for e in registry["episodes"] if e < len(available)]] = False
    for collection in prior:
        available[collection.get("weights", {}).get("episodes", [])] = False
        for record in [*collection["families"].values(), *collection["windows"].values()]:
            if record.get("seed") is not None and record["scene"] == "challenge":
                forbidden.add(int(record["seed"]))
            if record.get("episode") is not None:
                available[int(record["episode"])] = False
            seen["recipe"].add(record["recipe_sha256"])
            seen["history"].add(record["history_sha256"])
    log: dict[str, int] = {}
    weights = None
    if phase == "development":
        weights = weight_sample(h5, rng_for(protocol, "weights"), available, protocol)
        payload = {**weights}
        (directory / "weights.json").write_text(json.dumps(payload, sort_keys=True, indent=2))
    families = collect_challenge(
        directory / "families", phase, protocol, forbidden, seen, log, budget
    )
    families += collect_archive(
        directory / "families", phase, protocol, h5, available, seen, log, budget
    )
    windows = collect_windows(directory / "windows", phase, protocol, h5, available, log, budget)
    assign_partitions(families, phase, protocol, rng_for(protocol, (phase, "partition")))
    for record in [*families, *windows]:
        record["relative_path"] = str(Path(record.pop("path")).relative_to(root))
        if record["scene"] == "ordinary":
            record["partition"] = "validation" if phase == "development" else "confirmation"
    collection = {
        "phase": phase,
        "families": {record["id"]: record for record in families},
        "windows": {record["id"]: record for record in windows},
        "rejections": dict(sorted(log.items())),
    }
    if weights is not None:
        collection["weights"] = weights
    counts = observed_counts(collection)
    if counts != expected_counts(phase, protocol):
        raise DataIntegrityError(f"Cohort counts {counts} differ from the protocol")
    collection["counts"] = counts
    collection["integrity"] = verify_integrity([*prior, collection], registry, root)
    return collection
