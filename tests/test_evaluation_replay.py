"""SIM-001 to EVAL-001 integration tests."""

from dataclasses import replace

import numpy as np

from jepa_action_effects.evaluation_replay import family_truth_from_replay
from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family


def _controls(x=0.0, y=0.0):
    return np.tile(np.array([x, y], dtype=np.float32), (10, 1))


def test_replay_adapter_aligns_endpoints_and_excludes_warmup_and_wall_contact():
    recipe = ReplayRecipe(
        "eval-adapter",
        99,
        _controls(),
        {
            "state": [256, 400, 256, 256, 0, 0, 0],
            "goal_state": [50, 50, 450, 450, 0, 0, 0],
        },
    )
    replay = run_family(
        recipe,
        [Candidate("push", _controls(0, -1.0)), Candidate("hold", _controls())],
    )
    truth = family_truth_from_replay(replay, horizon_native_steps=2, split="development")
    by_id = {branch.candidate_id: branch for branch in truth.branches}
    assert truth.eligible
    assert by_id["push"].contacted
    assert not by_id["hold"].contacted
    assert by_id["push"].trajectory_resolution == "physics_substep"
    np.testing.assert_allclose(
        by_id["push"].block_endpoint,
        replay.branches["push"].observations[-1, 2:4],
        atol=1e-10,
        rtol=0,
    )

    incomplete = replace(replay.branches["hold"], completed=False)
    replay.branches["hold"] = incomplete
    excluded = family_truth_from_replay(replay, horizon_native_steps=2)
    assert not excluded.eligible
    assert excluded.exclusion_reasons == ("incomplete_candidates:hold",)

    wall_recipe = ReplayRecipe(
        "wall-only",
        99,
        _controls(),
        {
            "state": [100, 100, 470, 256, 0, 0, 0],
            "goal_state": [50, 50, 450, 450, 0, 0, 0],
        },
    )
    wall_replay = run_family(wall_recipe, [Candidate("hold", _controls())])
    assert any(
        event.shape_a.startswith("wall:") or event.shape_b.startswith("wall:")
        for step in wall_replay.branches["hold"].physics_steps
        if step.phase == "candidate"
        for event in step.events
    )
    wall_truth = family_truth_from_replay(wall_replay, horizon_native_steps=2)
    assert not wall_truth.branches[0].contacted


def test_replay_adapter_marks_horizon_mismatch_ineligible():
    recipe = ReplayRecipe(
        "short-horizon",
        99,
        _controls(),
        {"goal_state": [50, 50, 450, 450, 0, 0, 0]},
    )
    short_controls = np.zeros((5, 2), dtype=np.float32)
    replay = run_family(recipe, [Candidate("short", short_controls)])
    truth = family_truth_from_replay(replay, horizon_native_steps=2)
    assert not truth.eligible
    assert truth.exclusion_reasons == ("horizon_mismatch_candidates:short",)
