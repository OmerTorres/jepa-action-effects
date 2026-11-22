"""Simulator-only generation and persistence for the bounded PILOT-001 cohort."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.simulator_replay import (
    Candidate,
    InstrumentedPushT,
    ReplayRecipe,
    assert_dynamic_states_equal,
    run_family,
)

MASTER_SEED = 20260926
FAR_GOAL = [50.0, 50.0, 450.0, 450.0, 0.0, 0.0, 0.0]
GOAL_GRID = tuple((float(x), float(y)) for x in (128, 256, 384) for y in (128, 256, 384))


@dataclass(frozen=True)
class PilotConfig:
    schema_version: int = 1
    master_seed: int = MASTER_SEED
    challenge_target: int = 80
    ordinary_target: int = 40
    challenge_attempt_cap: int = 800
    ordinary_attempt_cap: int = 400
    initial_horizon: int = 2
    workspace_width: float = 512.0
    replay_epsilon: float = 0.0
    challenge_block_min: float = 175.0
    challenge_block_max: float = 337.0
    challenge_clearance_min: float = 2.0
    challenge_clearance_max: float = 10.0
    challenge_lateral_inside: float = 10.0
    challenge_lateral_outside: float = 15.0
    challenge_push_magnitude: float = 0.6
    challenge_perturbations: tuple[float, ...] = (-0.2, -0.1, 0.1, 0.2)
    ordinary_action_bound: float = 0.35
    moving_warmup_fraction: float = 0.5
    challenge_split_counts: tuple[int, int, int] = (40, 20, 20)
    ordinary_split_counts: tuple[int, int, int] = (20, 10, 10)
    horizon_policy: str = "adaptive"
    forbidden_reset_seeds: tuple[int, ...] = ()

    @property
    def delta(self) -> float:
        return max(0.01 * self.workspace_width, 10.0 * self.replay_epsilon)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_payload(config: PilotConfig) -> dict[str, Any]:
    result = asdict(config)
    result["delta"] = config.delta
    result["seed_streams"] = {
        name: int(sequence.generate_state(1)[0])
        for name, sequence in zip(
            ("proposals", "splits", "execution_order", "goals", "diagnostics", "bootstrap"),
            np.random.SeedSequence(config.master_seed).spawn(6),
            strict=True,
        )
    }
    result["generator_version"] = "pilot001-v1"
    return result


def _rngs(config: PilotConfig) -> dict[str, np.random.Generator]:
    names = ("proposals", "splits", "execution_order", "goals", "diagnostics", "bootstrap")
    return {
        name: np.random.default_rng(sequence)
        for name, sequence in zip(
            names, np.random.SeedSequence(config.master_seed).spawn(len(names)), strict=True
        )
    }


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def _actual_block_vertices(block_position: np.ndarray, angle: float) -> np.ndarray:
    """Read the proposal boundary from the actual configured Pymunk polygons."""
    env = InstrumentedPushT()
    try:
        env.reset(
            seed=0,
            options={
                "state": [100.0, 100.0, *block_position, angle, 0.0, 0.0],
                "goal_state": FAR_GOAL,
            },
        )
        vertices = []
        for shape in env.block.shapes:
            if hasattr(shape, "get_vertices"):
                vertices.extend(env.block.local_to_world(vertex) for vertex in shape.get_vertices())
        result = np.asarray([(point.x, point.y) for point in vertices], dtype=np.float64)
        if result.shape != (8, 2):
            raise RuntimeError(f"Expected two four-vertex block polygons, got {result.shape}")
        return result
    finally:
        env.close()


def _constant(vector: np.ndarray, blocks: int = 2) -> np.ndarray:
    return np.tile(np.asarray(vector, dtype=np.float32), (5 * blocks, 1))


def propose_challenge(config: PilotConfig, attempt: int, rng: np.random.Generator):
    block = rng.uniform(config.challenge_block_min, config.challenge_block_max, size=2)
    angle = float(rng.uniform(-math.pi, math.pi))
    rotation = _rotation(angle)
    toward = rotation @ np.array([0.0, -1.0])
    outward = -toward
    lateral = rotation @ np.array([1.0, 0.0])
    vertices = _actual_block_vertices(block, angle)
    offsets = vertices - block
    support_outward = float(np.max(offsets @ outward))
    support_lateral = float(np.max(offsets @ lateral))
    clearance = float(rng.uniform(config.challenge_clearance_min, config.challenge_clearance_max))
    lateral_offset = support_lateral + float(
        rng.uniform(-config.challenge_lateral_inside, config.challenge_lateral_outside)
    )
    agent = block + outward * (support_outward + 15.0 + clearance) + lateral * lateral_offset
    if np.any(agent < 22.0) or np.any(agent > 490.0):
        return None, "agent_outside_safe_workspace"
    moving = bool(rng.random() < config.moving_warmup_fraction)
    warmup_vector = toward * (0.08 if moving else 0.0)
    actions = [
        ("hold", np.zeros(2)),
        ("push", toward * config.challenge_push_magnitude),
        *[
            (f"lateral_{index}", toward * config.challenge_push_magnitude + lateral * offset)
            for index, offset in enumerate(config.challenge_perturbations)
        ],
    ]
    recipe = ReplayRecipe(
        f"challenge-{attempt:04d}",
        int(rng.integers(0, 2**31 - 1)),
        _constant(warmup_vector),
        {"state": [*agent, *block, angle, 0.0, 0.0], "goal_state": FAR_GOAL},
    )
    metadata = {
        "cohort": "challenge",
        "attempt": attempt,
        "moving_warmup": moving,
        "actual_block_vertices": vertices.tolist(),
        "approach_toward": toward.tolist(),
        "lateral_axis": lateral.tolist(),
        "support_outward": support_outward,
        "support_lateral": support_lateral,
        "clearance": clearance,
        "lateral_offset": lateral_offset,
    }
    return (
        recipe,
        [Candidate(name, _constant(vector)) for name, vector in actions],
        metadata,
    ), None


def propose_ordinary(config: PilotConfig, attempt: int, rng: np.random.Generator):
    bound = config.ordinary_action_bound
    warmup_blocks = rng.uniform(-bound, bound, size=(2, 2)).astype(np.float32)
    warmup = np.repeat(warmup_blocks, 5, axis=0)
    candidates = []
    for index in range(6):
        blocks = rng.uniform(-bound, bound, size=(2, 2)).astype(np.float32)
        candidates.append(Candidate(f"ordinary_{index}", np.repeat(blocks, 5, axis=0)))
    recipe = ReplayRecipe(
        f"ordinary-{attempt:04d}",
        int(rng.integers(0, 2**31 - 1)),
        warmup,
        {"goal_state": FAR_GOAL},
    )
    return recipe, candidates, {"cohort": "ordinary", "attempt": attempt}


def _branch_contact(branch) -> bool:
    return any(
        step.agent_block_contact for step in branch.physics_steps if step.phase == "candidate"
    )


def _challenge_boundary_pair(candidates: list[Candidate], replay) -> tuple[str, str] | None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ids = sorted(by_id)
    for left_index, left in enumerate(ids):
        if left == "hold":
            continue
        for right in ids[left_index + 1 :]:
            if right == "hold":
                continue
            action_distance = float(
                np.max(np.linalg.norm(by_id[left].controls - by_id[right].controls, axis=1))
            )
            if action_distance <= 0.2 + 1e-7 and _branch_contact(
                replay.branches[left]
            ) != _branch_contact(replay.branches[right]):
                return left, right
    return None


def _states_from_steps(branch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reset = [step for step in branch.physics_steps if step.phase == "reset"][-1]
    warmup = [step for step in branch.physics_steps if step.phase == "warmup"]
    history_steps = (reset, warmup[49], warmup[99])
    history = np.asarray(
        [
            [*step.agent["position"], *step.block["position"], step.block["angle"]]
            for step in history_steps
        ],
        dtype=np.float64,
    )
    candidate_steps = [step for step in branch.physics_steps if step.phase == "candidate"]
    native = [candidate_steps[index] for index in range(49, len(candidate_steps), 50)]
    future = np.asarray(
        [
            [*step.agent["position"], *step.block["position"], step.block["angle"]]
            for step in native
        ],
        dtype=np.float64,
    )
    trajectory = np.asarray([step.block["position"] for step in candidate_steps], dtype=np.float64)
    return history, future, trajectory


def save_family(
    path: Path,
    recipe: ReplayRecipe,
    candidates: list[Candidate],
    replay,
    metadata: dict[str, Any],
) -> None:
    ids = sorted(replay.branches)
    reference = replay.branches[ids[0]]
    history_states, _, _ = _states_from_steps(reference)
    future_states = []
    trajectories = []
    contacts = []
    for candidate_id in ids:
        _, future, trajectory = _states_from_steps(replay.branches[candidate_id])
        future_states.append(future)
        trajectories.append(trajectory)
        contacts.append(_branch_contact(replay.branches[candidate_id]))
    payload = {
        **metadata,
        "family_id": replay.family_id,
        "seed": replay.seed,
        "execution_order": list(replay.execution_order),
        "candidate_ids": ids,
        "reset_options": recipe.reset_options,
        "completed": {key: replay.branches[key].completed for key in ids},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        np.savez_compressed(
            temporary,
            metadata=np.array(json.dumps(payload, sort_keys=True)),
            warmup_controls=recipe.warmup_controls,
            candidate_controls=np.stack([by_id[key].controls for key in ids]),
            history_frames=reference.history_frames,
            history_states=history_states,
            future_frames=np.stack([replay.branches[key].future_frames for key in ids]),
            future_states=np.stack(future_states),
            block_trajectories=np.stack(trajectories),
            contacts=np.asarray(contacts, dtype=bool),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_family(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files if key != "metadata"}
        result["metadata"] = json.loads(str(archive["metadata"].item()))
    return result


def _split_membership(
    accepted: dict[str, list[str]], rng: np.random.Generator, config: PilotConfig
) -> dict[str, str]:
    membership: dict[str, str] = {}
    counts = {
        "challenge": tuple(
            zip(("fit", "validation", "development"), config.challenge_split_counts, strict=True)
        ),
        "ordinary": tuple(
            zip(("fit", "validation", "development"), config.ordinary_split_counts, strict=True)
        ),
    }
    for cohort, allocations in counts.items():
        ids = list(accepted[cohort])
        rng.shuffle(ids)
        cursor = 0
        for split, count in allocations:
            for family_id in ids[cursor : cursor + count]:
                membership[family_id] = split
            cursor += count
    return membership


def _goals(family_ids: list[str], rng: np.random.Generator) -> dict[str, list[list[float]]]:
    grid = np.asarray(GOAL_GRID)
    return {
        family_id: grid[rng.choice(len(grid), size=4, replace=False)].tolist()
        for family_id in sorted(family_ids)
    }


def _meaningful_family(path: Path, delta: float) -> bool:
    family = load_family(path)
    endpoints = family["future_states"][:, -1, 2:4]
    return any(
        np.linalg.norm(endpoints[left] - endpoints[right]) > delta
        for left in range(len(endpoints))
        for right in range(left + 1, len(endpoints))
    )


def collect(output_dir: Path, config: PilotConfig | None = None) -> dict[str, Any]:
    config = config or PilotConfig()
    if (
        sum(config.challenge_split_counts) != config.challenge_target
        or sum(config.ordinary_split_counts) != config.ordinary_target
    ):
        raise ValueError("Split allocations must sum to cohort targets")
    if config.horizon_policy not in {"adaptive", "fixed_two"}:
        raise ValueError("Unknown horizon policy")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = config_payload(config)
    config_hash = sha256_bytes(canonical_json(payload))
    config_path = output_dir / "config.json"
    if config_path.exists():
        existing_hash = sha256_bytes(canonical_json(json.loads(config_path.read_text())))
        if existing_hash != config_hash:
            raise ValueError("Existing pilot configuration does not match requested configuration")
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rngs = _rngs(config)
    accepted: dict[str, list[str]] = {"challenge": [], "ordinary": []}
    rejected: list[dict[str, Any]] = []
    attempts = {"challenge": 0, "ordinary": 0}
    progress_path = output_dir / "collection-progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["config_sha256"] != config_hash:
            raise ValueError("Interrupted collection configuration hash mismatch")
        accepted = progress["accepted"]
        rejected = progress["rejections"]
        attempts = progress["attempts"]
        for name, state in progress["rng_states"].items():
            rngs[name].bit_generator.state = state
    forbidden_seeds = set(config.forbidden_reset_seeds)
    for accepted_ids in accepted.values():
        for family_id in accepted_ids:
            stored = load_family(output_dir / "families" / f"{family_id}.npz")
            forbidden_seeds.add(int(stored["metadata"]["seed"]))

    def save_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "config_sha256": config_hash,
                "accepted": accepted,
                "rejections": rejected,
                "attempts": attempts,
                "rng_states": {name: rng.bit_generator.state for name, rng in rngs.items()},
            },
        )

    save_progress()

    targets = {"challenge": config.challenge_target, "ordinary": config.ordinary_target}
    caps = {"challenge": config.challenge_attempt_cap, "ordinary": config.ordinary_attempt_cap}
    for cohort in ("challenge", "ordinary"):
        for attempt in range(attempts[cohort], caps[cohort]):
            if len(accepted[cohort]) >= targets[cohort]:
                break
            attempts[cohort] += 1
            if cohort == "challenge":
                proposal, reason = propose_challenge(config, attempt, rngs["proposals"])
                if proposal is None:
                    rejected.append({"cohort": cohort, "attempt": attempt, "reason": reason})
                    save_progress()
                    continue
                recipe, candidates, metadata = proposal
            else:
                recipe, candidates, metadata = propose_ordinary(config, attempt, rngs["proposals"])
            if recipe.seed in forbidden_seeds:
                rejected.append(
                    {"cohort": cohort, "attempt": attempt, "reason": "reset_seed_overlap"}
                )
                save_progress()
                continue
            order = [candidate.candidate_id for candidate in candidates]
            rngs["execution_order"].shuffle(order)
            try:
                replay = run_family(recipe, candidates, order)
            except RuntimeError as error:
                rejected.append(
                    {"cohort": cohort, "attempt": attempt, "reason": f"replay_error:{error}"}
                )
                save_progress()
                continue
            if not all(branch.completed for branch in replay.branches.values()):
                rejected.append({"cohort": cohort, "attempt": attempt, "reason": "incomplete"})
                save_progress()
                continue
            if cohort == "challenge":
                boundary = _challenge_boundary_pair(candidates, replay)
                if boundary is None:
                    rejected.append(
                        {
                            "cohort": cohort,
                            "attempt": attempt,
                            "reason": "no_close_contact_boundary",
                        }
                    )
                    save_progress()
                    continue
                metadata["accepted_boundary_pair"] = list(boundary)
            family_id = f"{cohort}-{len(accepted[cohort]):03d}"
            recipe = ReplayRecipe(
                family_id, recipe.seed, recipe.warmup_controls, recipe.reset_options
            )
            replay = type(replay)(family_id, replay.seed, replay.execution_order, replay.branches)
            save_family(
                output_dir / "families" / f"{family_id}.npz", recipe, candidates, replay, metadata
            )
            accepted[cohort].append(family_id)
            forbidden_seeds.add(recipe.seed)
            save_progress()

    family_ids = accepted["challenge"] + accepted["ordinary"]
    splits = (
        _split_membership(accepted, rngs["splits"], config)
        if all(len(accepted[key]) == targets[key] for key in targets)
        else {}
    )
    goals = _goals(family_ids, rngs["goals"])
    meaningful = sum(
        _meaningful_family(output_dir / "families" / f"{family_id}.npz", config.delta)
        for family_id in family_ids
    )
    fraction = meaningful / len(family_ids) if family_ids else 0.0
    selected_horizon = 2 if config.horizon_policy == "fixed_two" or fraction >= 0.2 else 4
    selected_paths = {
        family_id: output_dir / "families" / f"{family_id}.npz" for family_id in family_ids
    }
    four_step_incomplete: list[str] = []
    if selected_horizon == 4:
        for family_id in family_ids:
            stored = load_family(selected_paths[family_id])
            metadata = stored["metadata"]
            recipe = ReplayRecipe(
                family_id,
                int(metadata["seed"]),
                stored["warmup_controls"],
                metadata["reset_options"],
            )
            candidates = [
                Candidate(candidate_id, np.concatenate((controls, controls)))
                for candidate_id, controls in zip(
                    metadata["candidate_ids"], stored["candidate_controls"], strict=True
                )
            ]
            replay = run_family(recipe, candidates, metadata["execution_order"])
            if not all(branch.completed for branch in replay.branches.values()):
                four_step_incomplete.append(family_id)
            selected_path = output_dir / "families-h4" / f"{family_id}.npz"
            save_family(selected_path, recipe, candidates, replay, metadata)
            selected_paths[family_id] = selected_path
    audit_ids = []
    for cohort in ("challenge", "ordinary"):
        chosen = rngs["diagnostics"].choice(
            accepted[cohort], size=min(6, len(accepted[cohort])), replace=False
        )
        audit_ids.extend(str(value) for value in chosen)
    audit = replay_audit(output_dir, audit_ids)
    (output_dir / "replay-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    animation_ids = audit_ids[: min(4, len(audit_ids))]
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    for family_id in animation_ids:
        stored = load_family(selected_paths[family_id])
        candidate_index = 0
        if stored["metadata"]["cohort"] == "challenge":
            contacted = np.flatnonzero(stored["contacts"])
            if len(contacted):
                candidate_index = int(contacted[0])
        frames = [*stored["history_frames"], *stored["future_frames"][candidate_index]]
        rendered = []
        for frame in frames:
            image = Image.fromarray(frame.copy())
            draw = ImageDraw.Draw(image)
            color = "red" if stored["contacts"][candidate_index] else "blue"
            draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=color, width=4)
            rendered.append(image)
        rendered[0].save(
            diagnostic_dir / f"{family_id}.gif",
            save_all=True,
            append_images=rendered[1:],
            duration=350,
            loop=0,
        )
    manifest = {
        "schema_version": 1,
        "status": "complete"
        if all(len(accepted[key]) == targets[key] for key in targets)
        else "quota_shortfall",
        "config_sha256": config_hash,
        "accepted": accepted,
        "attempts": attempts,
        "rejections": rejected,
        "splits": splits,
        "goals": goals,
        "delta": config.delta,
        "two_step_meaningful_family_count": meaningful,
        "two_step_meaningful_family_fraction": fraction,
        "selected_horizon": selected_horizon,
        "four_step_incomplete_families": four_step_incomplete,
        "elapsed_seconds": time.perf_counter() - started,
        "replay_audit": audit,
        "diagnostic_animation_families": animation_ids,
        "family_files": {
            family_id: {
                "path": str(selected_paths[family_id].relative_to(output_dir)),
                "sha256": sha256_file(selected_paths[family_id]),
            }
            for family_id in family_ids
        },
    }
    manifest_path = output_dir / "collection-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "collection-manifest.sha256").write_text(sha256_file(manifest_path) + "\n")
    progress_path.unlink(missing_ok=True)
    return manifest


def verify_collection(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "collection-manifest.json"
    expected = (output_dir / "collection-manifest.sha256").read_text().strip()
    if sha256_file(manifest_path) != expected:
        raise ValueError("Collection manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Collection is not complete: {manifest['status']}")
    for family_id, record in manifest["family_files"].items():
        path = output_dir / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Family artifact hash mismatch: {family_id}")
    return manifest


def replay_audit(output_dir: Path, family_ids: list[str]) -> dict[str, Any]:
    results = []
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    for family_id in family_ids:
        stored = load_family(output_dir / "families" / f"{family_id}.npz")
        metadata = stored["metadata"]
        recipe = ReplayRecipe(
            family_id,
            int(metadata["seed"]),
            stored["warmup_controls"],
            metadata["reset_options"],
        )
        candidates = [
            Candidate(candidate_id, controls)
            for candidate_id, controls in zip(
                metadata["candidate_ids"], stored["candidate_controls"], strict=True
            )
        ]
        forward = run_family(recipe, candidates, metadata["execution_order"])
        reverse = run_family(recipe, candidates, list(reversed(metadata["execution_order"])))
        for candidate_id in metadata["candidate_ids"]:
            left, right = forward.branches[candidate_id], reverse.branches[candidate_id]
            np.testing.assert_array_equal(left.history_frames, right.history_frames)
            np.testing.assert_array_equal(left.future_frames, right.future_frames)
            np.testing.assert_array_equal(left.observations, right.observations)
            assert_dynamic_states_equal(left.physics_steps, right.physics_steps)
        contact_branch = next(
            (
                forward.branches[candidate_id]
                for candidate_id in metadata["candidate_ids"]
                if _branch_contact(forward.branches[candidate_id])
            ),
            None,
        )
        overlay_path = None
        if contact_branch is not None:
            image = Image.fromarray(contact_branch.history_frames[-1].copy())
            draw = ImageDraw.Draw(image)
            for step in contact_branch.physics_steps:
                if step.phase != "candidate":
                    continue
                for event in step.events:
                    if not event.agent_block:
                        continue
                    for point in event.points:
                        x, y = point["point_a"]
                        x, y = x * 224 / 512, y * 224 / 512
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
            overlay_path = diagnostic_dir / f"{family_id}-contact-overlay.png"
            image.save(overlay_path)
        results.append(
            {
                "family_id": family_id,
                "passed": True,
                "contact_overlay": (
                    None if overlay_path is None else str(overlay_path.relative_to(output_dir))
                ),
            }
        )
    return {"family_count": len(results), "all_passed": True, "families": results}
