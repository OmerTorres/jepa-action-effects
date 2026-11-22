"""Run the EVAL-001 analytic fixtures and a small SIM-001 integration check."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.evaluation import (
    BranchPrediction,
    BranchTruth,
    EvaluationConfig,
    FamilyTruth,
    Goal,
    evaluate_family,
)
from jepa_action_effects.evaluation_replay import family_truth_from_replay
from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family

CONFIG = EvaluationConfig(
    delta=5.12,
    workspace_width=512.0,
    sim_epsilon=0.0,
    horizon_native_steps=2,
)
FAR_GOAL = [50.0, 50.0, 450.0, 450.0, 0.0, 0.0, 0.0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _controls(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.tile(np.array([x, y], dtype=np.float32), (10, 1))


def _analytic_truth(*, zero_effect: bool = False) -> FamilyTruth:
    endpoint_a = np.zeros(2) if zero_effect else np.array([10.0, 0.0])
    return FamilyTruth(
        "analytic-zero" if zero_effect else "analytic-effect",
        np.zeros(2),
        np.zeros(2),
        (
            BranchTruth("a", endpoint_a, endpoint_a, True),
            BranchTruth("b", np.zeros(2), np.zeros(2), False),
        ),
        2,
        split="synthetic",
        group_id="eval001-fixtures",
    )


def _predictions(truth: FamilyTruth, transform) -> tuple[BranchPrediction, ...]:
    return tuple(
        BranchPrediction(branch.candidate_id, transform(branch.block_endpoint))
        for branch in truth.branches
    )


def _evaluate(truth: FamilyTruth, predictions: tuple[BranchPrediction, ...], method: str):
    return evaluate_family(
        truth,
        predictions,
        (Goal("origin", np.zeros(2)),),
        CONFIG,
        method=method,
        goal_provenance="fixed analytic origin",
    )


def run_audit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    truth = _analytic_truth()
    fixtures = {
        "oracle": _evaluate(truth, _predictions(truth, lambda value: value), "oracle"),
        "persistence": _evaluate(truth, _predictions(truth, lambda _: np.zeros(2)), "persistence"),
        "doubled_effect": _evaluate(
            truth, _predictions(truth, lambda value: 2 * value), "doubled_effect"
        ),
        "reversed_effect": _evaluate(
            truth, _predictions(truth, lambda value: -value), "reversed_effect"
        ),
    }
    zero_truth = _analytic_truth(zero_effect=True)
    fixtures["spurious_motion"] = _evaluate(
        zero_truth,
        (
            BranchPrediction("a", np.array([10.0, 0.0])),
            BranchPrediction("b", np.zeros(2)),
        ),
        "spurious_motion",
    )
    fixtures["positive_regret"] = _evaluate(
        truth,
        (
            BranchPrediction("a", np.zeros(2)),
            BranchPrediction("b", np.array([10.0, 0.0])),
        ),
        "positive_regret",
    )

    observed = {
        name: {
            "effect_error": evaluation.pair_metrics[0].effect_error,
            "direction_error_radians": evaluation.pair_metrics[0].direction_error_radians,
            "regret": evaluation.goal_metrics[0].regret,
        }
        for name, evaluation in fixtures.items()
    }
    expected = {
        "oracle": {"effect_error": 0.0, "regret": 0.0},
        "persistence": {"effect_error": 10.0},
        "doubled_effect": {"effect_error": 10.0},
        "reversed_effect": {"effect_error": 20.0, "direction_error_radians": np.pi},
        "spurious_motion": {"effect_error": 10.0},
        "positive_regret": {"regret": 10.0},
    }
    for fixture, values in expected.items():
        for metric, value in values.items():
            np.testing.assert_allclose(observed[fixture][metric], value, atol=1e-12, rtol=0)

    recipe = ReplayRecipe(
        "eval001-replay-integration",
        99,
        _controls(),
        {"state": [256, 400, 256, 256, 0, 0, 0], "goal_state": FAR_GOAL},
    )
    replay = run_family(
        recipe,
        (Candidate("push", _controls(0, -1)), Candidate("hold", _controls())),
    )
    replay_truth = family_truth_from_replay(
        replay, horizon_native_steps=2, split="development", group_id=recipe.family_id
    )
    contact_by_candidate = {
        branch.candidate_id: branch.contacted for branch in replay_truth.branches
    }
    if contact_by_candidate != {"hold": False, "push": True}:
        raise AssertionError(f"unexpected integration contacts: {contact_by_candidate}")

    report = {
        "task": "EVAL-001",
        "status": "passed",
        "evidence_kind": "analytic mock predictors and simulator integration; no checkpoint",
        "configuration": {
            "workspace_width": CONFIG.workspace_width,
            "sim_epsilon": CONFIG.sim_epsilon,
            "delta": CONFIG.delta,
            "delta_formula": "max(0.01 * workspace_width, 10 * sim_epsilon)",
            "horizon_native_steps": CONFIG.horizon_native_steps,
            "pair_policy": "all unordered pairs, canonical candidate ID order",
            "threshold_boundary": "negligible and stationary use <= delta",
            "informative_goal": "true candidate cost spread > delta",
            "tie_policy": "canonical candidate ID",
        },
        "analytic_fixtures": observed,
        "replay_integration": {
            "family_id": replay_truth.family_id,
            "eligible": replay_truth.eligible,
            "contact_by_candidate": contact_by_candidate,
            "trajectory_resolution": "physics_substep",
        },
        "limitations": [
            "Mock predictors validate calculations but provide no model-performance evidence.",
            "The physical relevance of delta and the two-step horizon remain provisional.",
            "Candidate generation, goal-grid sampling, readouts, and bootstrap inference "
            "are downstream.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    source_paths = [
        root / "src/jepa_action_effects/evaluation.py",
        root / "src/jepa_action_effects/evaluation_replay.py",
        root / "src/jepa_action_effects/evaluation_audit.py",
        root / "tests/test_evaluation.py",
        root / "tests/test_evaluation_replay.py",
    ]
    manifest = {
        "task": "EVAL-001",
        "status": "passed",
        "base_git_head": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "stable_worldmodel": version("stable-worldmodel"),
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "source_sha256": {str(path.relative_to(root)): _sha256(path) for path in source_paths},
        "report": "outputs/eval001/report.json",
        "configuration": report["configuration"],
        "limitations": report["limitations"],
    }
    return report, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/eval001/report.json"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/manifests/eval001-runtime.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    report, manifest = run_audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
