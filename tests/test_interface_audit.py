"""Behavioral checks for ENV-001 timing and input isolation, not replay validation."""

import numpy as np
import pytest

from jepa_action_effects.interface_audit import make_env, ordinary_transition, preprocess_pixels


def test_control_step_has_ten_physics_substeps_and_relative_target(monkeypatch):
    env = make_env()
    try:
        env.reset(seed=42)
        start = np.array(env.agent.position)
        original_step = env.space.step
        timesteps = []

        def counted_step(dt):
            timesteps.append(dt)
            original_step(dt)

        monkeypatch.setattr(env.space, "step", counted_step)
        env.step(np.array([0.2, 0.0], dtype=np.float32))
        assert timesteps == [0.01] * 10
        assert start[0] < env.agent.position.x < start[0] + 20
        assert env.agent.position.y == pytest.approx(start[1])
    finally:
        env.close()


def test_history_excludes_future_target_and_preserves_action_order():
    sample = ordinary_transition()
    np.testing.assert_array_equal(
        sample["context_pixels"][0], preprocess_pixels(sample["frames"][:3])
    )
    np.testing.assert_array_equal(
        sample["target_pixels"][0], preprocess_pixels(sample["frames"][3:])
    )
    assert sample["context_pixels"].shape == (1, 3, 3, 224, 224)
    assert sample["target_pixels"].shape == (1, 1, 3, 224, 224)
    expected_history = np.repeat([[0.2, 0.0], [0.0, 0.2]], 5, axis=0).reshape(2, 10)
    np.testing.assert_allclose(sample["action_history_raw"], expected_history)
    np.testing.assert_allclose(sample["candidate_actions_raw"], [[-0.2, 0.0] * 5])
    # The target is a later physical transition, not a duplicate context frame.
    assert not np.array_equal(sample["states"][2], sample["states"][3])


def test_rgb_scaling_and_channel_order():
    frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    frames[..., 0] = 255
    actual = preprocess_pixels(frames)[0, :, 0, 0].numpy()
    np.testing.assert_allclose(
        actual, (np.array([1, 0, 0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225], rtol=1e-6
    )
    with pytest.raises(ValueError, match="uint8 RGB"):
        preprocess_pixels(frames.astype(float))
