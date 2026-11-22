"""Regression checks for SIM-001 replay and substep contact instrumentation."""

from dataclasses import replace

import numpy as np
import pytest

from jepa_action_effects.simulator_replay import (
    Candidate,
    InstrumentedPushT,
    ReplayRecipe,
    assert_dynamic_states_equal,
    run_family,
)


def _controls(x=0.0, y=0.0, length=10):
    return np.tile(np.array([x, y], dtype=np.float32), (length, 1))


def _contact_recipe(family_id="contact"):
    return ReplayRecipe(
        family_id,
        99,
        _controls(),
        {
            "state": [256, 360, 256, 256, 0, 0, 0],
            "goal_state": [50, 50, 450, 450, 0, 0, 0],
        },
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ReplayRecipe("x", 0, np.zeros((9, 2), np.float32)), "warmup"),
        (lambda: Candidate("x", np.zeros((6, 2), np.float32)), "candidate"),
        (lambda: Candidate("x", np.full((5, 2), np.nan, np.float32)), "finite"),
        (lambda: Candidate("x", np.full((5, 2), 1.01, np.float32)), "never clipped"),
    ],
)
def test_replay_inputs_are_rejected_without_clipping(factory, message):
    with pytest.raises((ValueError, TypeError), match=message):
        factory()


def test_candidate_accepts_four_native_steps():
    candidate = Candidate("four-step", _controls(length=20))
    assert candidate.controls.shape == (20, 2)


def test_each_candidate_gets_full_reset_warmup_and_order_is_irrelevant():
    recipe = _contact_recipe()
    candidates = [
        Candidate("push", _controls(0, -0.25)),
        Candidate("hold", _controls()),
    ]
    forward = run_family(recipe, candidates)
    reverse = run_family(recipe, candidates, ["hold", "push"])
    for candidate_id in ("push", "hold"):
        left = forward.branches[candidate_id]
        right = reverse.branches[candidate_id]
        np.testing.assert_array_equal(left.history_frames, right.history_frames)
        np.testing.assert_array_equal(left.observations, right.observations)
        assert_dynamic_states_equal(left.physics_steps, right.physics_steps)
        assert len([step for step in left.physics_steps if step.phase == "reset"]) == 2
        assert len([step for step in left.physics_steps if step.phase == "warmup"]) == 100
        assert len([step for step in left.physics_steps if step.phase == "candidate"]) == 100


def test_early_termination_is_explicit_and_stops_rollout(monkeypatch):
    original_eval_state = InstrumentedPushT.eval_state

    def candidate_only_termination(self, goal_state, current_state):
        if self._phase == "candidate":
            return True, 0.0
        return original_eval_state(self, goal_state, current_state)

    monkeypatch.setattr(InstrumentedPushT, "eval_state", candidate_only_termination)
    recipe = ReplayRecipe(
        "early-end",
        5,
        _controls(),
        {"goal_state": [50, 50, 450, 450, 0, 0, 0]},
    )
    branch = run_family(recipe, [Candidate("hold", _controls())]).branches["hold"]
    assert not branch.completed
    assert branch.completed_controls == 1
    assert branch.expected_controls == 10
    assert branch.terminated.tolist() == [True]
    assert len(branch.agent_block_contact_mask) == 10


def test_upstream_callback_and_pair_specific_callback_survive_resets():
    env = InstrumentedPushT()
    try:
        for repetition in range(2):
            env.reset(
                seed=99,
                options={
                    "state": [256, 360, 256, 256, 0, 0, 0],
                    "goal_state": [50, 50, 450, 450, 0, 0, 0],
                },
            )
            for index in range(10):
                env.set_record_context("candidate", index)
                _, _, _, _, info = env.step(np.array([0, -0.25], dtype=np.float32))
            candidate_steps = [step for step in env.physics_steps if step.phase == "candidate"]
            assert info["n_contacts"] > 0
            assert any(step.agent_block_contact for step in candidate_steps), repetition
    finally:
        env.close()


def test_dynamic_comparison_detects_velocity_mismatch_when_pose_matches():
    branch = run_family(_contact_recipe(), [Candidate("hold", _controls(0, -0.25))]).branches[
        "hold"
    ]
    changed_block = dict(branch.physics_steps[-1].block)
    changed_block["velocity"] = (
        changed_block["velocity"][0] + 1e-4,
        changed_block["velocity"][1],
    )
    changed = list(branch.physics_steps)
    changed[-1] = replace(changed[-1], block=changed_block)
    with pytest.raises(AssertionError, match="block.velocity"):
        assert_dynamic_states_equal(branch.physics_steps, changed)


def test_contact_identity_includes_both_t_polygons_and_excludes_wall_only():
    contact = run_family(_contact_recipe(), [Candidate("push", _controls(0, -0.25))]).branches[
        "push"
    ]
    block_shapes = {
        shape
        for step in contact.physics_steps
        for event in step.events
        if event.agent_block
        for shape in (event.shape_a, event.shape_b)
        if shape.startswith("block:")
    }
    assert block_shapes == {"block:0", "block:1"}

    wall_recipe = ReplayRecipe(
        "wall",
        99,
        _controls(),
        {
            "state": [100, 100, 470, 256, 0, 0, 0],
            "goal_state": [50, 50, 450, 450, 0, 0, 0],
        },
    )
    wall = run_family(wall_recipe, [Candidate("hold", _controls())]).branches["hold"]
    assert not wall.agent_block_contact_mask.any()
    assert any(
        event.shape_a.startswith("wall:") or event.shape_b.startswith("wall:")
        for step in wall.physics_steps
        if step.phase == "candidate"
        for event in step.events
    )
