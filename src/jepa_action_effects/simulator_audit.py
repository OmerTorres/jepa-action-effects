"""Run the SIM-001 deterministic replay and pair-specific contact validation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from jepa_action_effects.interface_audit import SWM_REVISION, make_env, sha256
from jepa_action_effects.simulator_replay import (
    STATE_ATOL,
    BranchResult,
    Candidate,
    ReplayRecipe,
    _body_state,
    assert_dynamic_states_equal,
    branch_to_dict,
    recipe_to_dict,
    run_family,
)

ORDERING_SEED = 20260926
AUDIT_SEEDS = tuple(range(16))
FAR_GOAL = [50.0, 50.0, 450.0, 450.0, 0.0, 0.0, 0.0]


def _constant(x: float, y: float, length: int = 10) -> np.ndarray:
    return np.tile(np.array([x, y], dtype=np.float32), (length, 1))


def diagnostic_candidates() -> list[Candidate]:
    return [
        Candidate("hold", _constant(0.0, 0.0)),
        Candidate("right", _constant(0.35, 0.0)),
        Candidate("left", _constant(-0.35, 0.0)),
        Candidate("up", _constant(0.0, -0.35)),
        Candidate("down", _constant(0.0, 0.35)),
        Candidate("diagonal", _constant(0.25, -0.25)),
    ]


def diagnostic_fixtures() -> dict[str, tuple[ReplayRecipe, Candidate]]:
    zero = _constant(0.0, 0.0)
    return {
        "direct_push": (
            ReplayRecipe(
                "fixture-direct-push",
                99,
                zero,
                {"state": [256, 380, 256, 256, 0, 0, 0], "goal_state": FAR_GOAL},
            ),
            Candidate("push", _constant(0.0, -0.6)),
        ),
        "near_miss": (
            ReplayRecipe(
                "fixture-near-miss",
                99,
                zero,
                {"state": [150, 180, 256, 256, 0, 0, 0], "goal_state": FAR_GOAL},
            ),
            Candidate("miss", _constant(0.6, 0.0)),
        ),
        "sustained_contact": (
            ReplayRecipe(
                "fixture-sustained-contact",
                99,
                zero,
                {"state": [256, 360, 256, 256, 0, 0, 0], "goal_state": FAR_GOAL},
            ),
            Candidate("sustain", _constant(0.0, -0.25)),
        ),
        "block_wall_only": (
            ReplayRecipe(
                "fixture-block-wall-only",
                99,
                zero,
                {"state": [100, 100, 470, 256, 0, 0, 0], "goal_state": FAR_GOAL},
            ),
            Candidate("hold", zero),
        ),
        "moving_rotating_branch": (
            ReplayRecipe(
                "fixture-moving-rotating",
                99,
                _constant(0.0, -0.6),
                {"state": [256, 380, 256, 256, 0, 0, 0], "goal_state": FAR_GOAL},
            ),
            Candidate("coast", zero),
        ),
    }


def _events_signature(branch: BranchResult) -> list[tuple[Any, ...]]:
    return [
        (
            step.phase,
            step.control_index,
            step.substep_index,
            tuple(
                (
                    event.shape_a,
                    event.shape_b,
                    event.agent_block,
                    event.contact_point_count,
                    event.normal,
                    tuple(
                        (point["point_a"], point["point_b"], point["distance"])
                        for point in event.points
                    ),
                )
                for event in step.events
            ),
        )
        for step in branch.physics_steps
    ]


def _max_numeric_difference(left: BranchResult, right: BranchResult) -> tuple[float, float]:
    values: list[float] = []
    positions: list[float] = []
    for actual, expected in zip(left.physics_steps, right.physics_steps, strict=True):
        for body in ("agent", "block"):
            a = getattr(actual, body)
            b = getattr(expected, body)
            for key in ("position", "angle", "velocity", "angular_velocity", "force", "torque"):
                values.append(float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))))
            positions.append(float(np.linalg.norm(np.asarray(a["position"]) - b["position"])))
    return max(values, default=0.0), max(positions, default=0.0)


def _compare_branches(left: BranchResult, right: BranchResult) -> tuple[float, float]:
    assert_dynamic_states_equal(left.physics_steps, right.physics_steps)
    np.testing.assert_array_equal(left.history_frames, right.history_frames)
    np.testing.assert_array_equal(left.history_controls, right.history_controls)
    np.testing.assert_array_equal(left.future_frames, right.future_frames)
    np.testing.assert_allclose(left.observations, right.observations, atol=STATE_ATOL, rtol=0)
    np.testing.assert_allclose(left.rewards, right.rewards, atol=STATE_ATOL, rtol=0)
    np.testing.assert_array_equal(left.upstream_contacts, right.upstream_contacts)
    np.testing.assert_array_equal(left.terminated, right.terminated)
    np.testing.assert_array_equal(left.truncated, right.truncated)
    if left.completed != right.completed or left.completed_controls != right.completed_controls:
        raise AssertionError("Branch completion differs")
    if _events_signature(left) != _events_signature(right):
        raise AssertionError("Contact identity, timestamp, or geometry differs")
    return _max_numeric_difference(left, right)


def _reference_rollout(recipe: ReplayRecipe, candidate: Candidate) -> dict[str, Any]:
    """Observe the unmodified base environment without replacing its collision callback."""
    env = make_env()
    physics: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=recipe.seed, options=recipe.reset_options)
        original_step = env.space.step

        def observed_step(dt):
            original_step(dt)
            physics.append({"agent": _body_state(env.agent), "block": _body_state(env.block)})

        env.space.step = observed_step
        frames = {0: env.render().copy()}
        for index, action in enumerate(recipe.warmup_controls):
            observation, _, terminated, truncated, _ = env.step(action.copy())
            if (index + 1) % 5 == 0:
                frames[index + 1] = env.render().copy()
            if terminated or truncated:
                raise RuntimeError("Reference warm-up ended early")
        history = np.stack([frames[len(recipe.warmup_controls) - offset] for offset in (10, 5, 0)])
        observations = []
        rewards = []
        contacts = []
        terminated_flags = []
        truncated_flags = []
        future = []
        for index, action in enumerate(candidate.controls):
            observation, reward, terminated, truncated, info = env.step(action.copy())
            observations.append(observation["state"].copy())
            rewards.append(reward)
            contacts.append(info["n_contacts"])
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            if (index + 1) % 5 == 0:
                future.append(env.render().copy())
            if terminated or truncated:
                break
        return {
            "physics": physics,
            "history": history,
            "future": np.stack(future),
            "observations": np.stack(observations),
            "rewards": np.asarray(rewards),
            "contacts": np.asarray(contacts),
            "terminated": np.asarray(terminated_flags),
            "truncated": np.asarray(truncated_flags),
        }
    finally:
        env.close()


def _check_reference_parity(
    recipe: ReplayRecipe, candidate: Candidate, branch: BranchResult
) -> float:
    reference = _reference_rollout(recipe, candidate)
    instrumented = [step for step in branch.physics_steps if step.phase != "reset"]
    if len(reference["physics"]) != len(instrumented):
        raise AssertionError("Instrumented and base physics-step counts differ")
    maximum = 0.0
    for ref, actual in zip(reference["physics"], instrumented, strict=True):
        for body in ("agent", "block"):
            for field_name in (
                "position",
                "angle",
                "velocity",
                "angular_velocity",
                "force",
                "torque",
            ):
                delta = float(
                    np.max(
                        np.abs(
                            np.asarray(ref[body][field_name])
                            - actual.__getattribute__(body)[field_name]
                        )
                    )
                )
                maximum = max(maximum, delta)
                if delta > STATE_ATOL:
                    raise AssertionError(f"Instrumentation changed {body}.{field_name}")
    np.testing.assert_array_equal(reference["history"], branch.history_frames)
    np.testing.assert_array_equal(reference["future"], branch.future_frames)
    np.testing.assert_allclose(
        reference["observations"], branch.observations, atol=STATE_ATOL, rtol=0
    )
    np.testing.assert_allclose(reference["rewards"], branch.rewards, atol=STATE_ATOL, rtol=0)
    np.testing.assert_array_equal(reference["contacts"], branch.upstream_contacts)
    np.testing.assert_array_equal(reference["terminated"], branch.terminated)
    np.testing.assert_array_equal(reference["truncated"], branch.truncated)
    return maximum


def _save_contact_example(output_dir: Path, name: str, branch: BranchResult) -> dict[str, str]:
    raw = branch.future_frames[-1]
    raw_path = output_dir / f"{name}-model-frame.png"
    annotated_path = output_dir / f"{name}-annotated.png"
    Image.fromarray(raw).save(raw_path)
    annotated = Image.fromarray(raw.copy())
    draw = ImageDraw.Draw(annotated)
    scale = raw.shape[1] / 512
    for step in branch.physics_steps:
        if step.phase != "candidate":
            continue
        for event in step.events:
            if not event.agent_block:
                continue
            for point in event.points:
                for key in ("point_a", "point_b"):
                    x, y = point[key]
                    px, py = x * scale, (512 - y) * scale
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 0, 0))
    annotated.save(annotated_path)
    return {
        "model_compatible_frame": str(raw_path),
        "annotated_diagnostic": str(annotated_path),
    }


def run_audit(output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = diagnostic_candidates()
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    permutation = list(candidate_ids)
    np.random.default_rng(ORDERING_SEED).shuffle(permutation)
    maximum_numeric = 0.0
    epsilon_sim = 0.0
    seed_summaries = []
    for seed in AUDIT_SEEDS:
        warmup = np.concatenate((_constant(0.1, 0.0, 5), _constant(0.0, 0.1, 5)))
        recipe = ReplayRecipe(f"seed-{seed:02d}", seed, warmup, {"goal_state": FAR_GOAL})
        forward = run_family(recipe, candidates)
        variants = [
            run_family(recipe, candidates),
            run_family(recipe, candidates, list(reversed(candidate_ids))),
            run_family(recipe, candidates, permutation),
        ]
        for variant in variants:
            for candidate_id in candidate_ids:
                difference, positional = _compare_branches(
                    forward.branches[candidate_id], variant.branches[candidate_id]
                )
                maximum_numeric = max(maximum_numeric, difference)
                epsilon_sim = max(epsilon_sim, positional)
        seed_summaries.append(
            {
                "seed": seed,
                "completed_candidates": sum(
                    branch.completed for branch in forward.branches.values()
                ),
                "agent_block_contact_candidates": sum(
                    bool(branch.agent_block_contact_mask.any())
                    for branch in forward.branches.values()
                ),
            }
        )

    fixtures_payload: dict[str, Any] = {}
    fixture_branches: dict[str, BranchResult] = {}
    for name, (recipe, candidate) in diagnostic_fixtures().items():
        branch = run_family(recipe, [candidate]).branches[candidate.candidate_id]
        fixture_branches[name] = branch
        candidate_steps = [step for step in branch.physics_steps if step.phase == "candidate"]
        wall_substeps = sum(
            any(
                (event.shape_a.startswith("wall:") or event.shape_b.startswith("wall:"))
                and not event.agent_block
                for event in step.events
            )
            for step in candidate_steps
        )
        fixtures_payload[name] = {
            "recipe": recipe_to_dict(recipe),
            "candidate_id": candidate.candidate_id,
            "agent_block_contact_substeps": int(branch.agent_block_contact_mask.sum()),
            "wall_contact_substeps": wall_substeps,
            "trace": branch_to_dict(branch),
        }

    direct = fixture_branches["direct_push"]
    near = fixture_branches["near_miss"]
    sustained = fixture_branches["sustained_contact"]
    wall = fixture_branches["block_wall_only"]
    moving = fixture_branches["moving_rotating_branch"]
    block_shapes = {
        shape
        for step in sustained.physics_steps
        for event in step.events
        if event.agent_block
        for shape in (event.shape_a, event.shape_b)
        if shape.startswith("block:")
    }
    first_candidate = next(step for step in moving.physics_steps if step.phase == "candidate")
    moving_at_branch = bool(
        np.linalg.norm(first_candidate.block["velocity"]) > 0
        and abs(first_candidate.block["angular_velocity"]) > 0
    )
    reference_maximum = _check_reference_parity(
        diagnostic_fixtures()["sustained_contact"][0],
        diagnostic_fixtures()["sustained_contact"][1],
        sustained,
    )
    candidate_substeps = [step for step in sustained.physics_steps if step.phase == "candidate"]
    substep_counts = {
        control: sum(step.control_index == control for step in candidate_substeps)
        for control in range(10)
    }
    gates = {
        "duplicate_replay": maximum_numeric <= STATE_ATOL,
        "branch_order_independence": epsilon_sim <= STATE_ATOL,
        "contact_identities_and_timestamps_exact": True,
        "instrumentation_matches_base": reference_maximum <= STATE_ATOL,
        "ten_physics_substeps_per_control": set(substep_counts.values()) == {10},
        "reset_steps_separate": [step.phase for step in sustained.physics_steps[:2]]
        == ["reset", "reset"],
        "history_timing": sustained.history_frames.shape == (3, 224, 224, 3)
        and sustained.history_controls.shape == (10, 2),
        "direct_push_contact": bool(direct.agent_block_contact_mask.any()),
        "near_miss_no_contact": not bool(near.agent_block_contact_mask.any()),
        "sustained_contact": int(sustained.agent_block_contact_mask.sum()) >= 50,
        "both_block_polygons_identified": block_shapes == {"block:0", "block:1"},
        "block_wall_without_agent_block": not bool(wall.agent_block_contact_mask.any())
        and fixtures_payload["block_wall_only"]["wall_contact_substeps"] > 0,
        "moving_rotating_branch_point": moving_at_branch,
        "all_seed_candidates_complete": all(
            item["completed_candidates"] == 6 for item in seed_summaries
        ),
    }
    examples = {
        name: _save_contact_example(output_dir, name, fixture_branches[name])
        for name in ("direct_push", "near_miss", "sustained_contact", "block_wall_only")
    }
    report = {
        "scope": "SIM-001 simulator-only replay/contact validation; no model inference",
        "simulator_revision": SWM_REVISION,
        "state_atol": STATE_ATOL,
        "ordering_seed": ORDERING_SEED,
        "audit_seeds": list(AUDIT_SEEDS),
        "candidate_ids": candidate_ids,
        "permuted_order": permutation,
        "maximum_numeric_discrepancy": maximum_numeric,
        "epsilon_sim": epsilon_sim,
        "instrumentation_reference_maximum_discrepancy": reference_maximum,
        "seed_summaries": seed_summaries,
        "fixtures": fixtures_payload,
        "examples": examples,
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    packages = ["stable-worldmodel", "numpy", "pymunk", "pygame", "gymnasium"]
    manifest = {
        "task": "SIM-001",
        "status": "passed" if report["passed"] else "failed",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
        "uv_lock_sha256": sha256(Path("uv.lock")),
        "simulator_revision": SWM_REVISION,
        "configuration": {
            "state_atol": STATE_ATOL,
            "ordering_seed": ORDERING_SEED,
            "audit_seeds": list(AUDIT_SEEDS),
            "candidate_ids": candidate_ids,
        },
        "results": {
            "passed": report["passed"],
            "gates": gates,
            "maximum_numeric_discrepancy": maximum_numeric,
            "epsilon_sim": epsilon_sim,
            "instrumentation_reference_maximum_discrepancy": reference_maximum,
        },
        "artifacts": {
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "examples": {
                name: {
                    kind: {"path": path, "sha256": sha256(Path(path))}
                    for kind, path in paths.items()
                }
                for name, paths in examples.items()
            },
        },
        "limitations": [
            "Validated only on the recorded locked runtime; "
            "cross-platform determinism is unproven.",
            "Observable body dynamics are compared; Pymunk solver caches are neither "
            "exposed nor cloned.",
            "Fixtures validate labels and replay mechanics, not natural-regime prevalence.",
            "Historical training-time simulator compatibility remains unproven.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"SIM-001 audit gates failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sim001"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/manifests/sim001-runtime.json"))
    args = parser.parse_args()
    report = run_audit(args.output_dir, args.manifest)
    print(
        f"SIM-001 audit passed in {report['elapsed_seconds']:.2f}s; "
        f"epsilon_sim={report['epsilon_sim']:.3g}"
    )


if __name__ == "__main__":
    main()
