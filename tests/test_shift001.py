"""Model-free fixtures for SHIFT-001 action construction, metrics and decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.shift001_metrics import (
    block_residuals,
    constant_version,
    family_record,
    is_rest_window,
    ratio,
    summarize_shift,
    varying_version,
)


def _pool(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks = rng.uniform(-0.8, 0.8, size=(2000, 5, 2))
    return block_residuals(blocks), np.linalg.norm(blocks, axis=-1).mean(axis=-1)


def test_residuals_are_zero_mean_and_varying_preserves_block_means() -> None:
    residuals, magnitudes = _pool()
    assert np.allclose(residuals.mean(axis=1), 0.0)
    constant = np.repeat(np.array([[0.0, -0.6], [0.3, 0.1]]), 5, axis=0)
    varying = varying_version(constant, residuals, magnitudes, np.random.default_rng(1))
    assert varying.shape == (10, 2)
    assert np.all(np.abs(varying) <= 1.0)
    assert np.allclose(varying.reshape(2, 5, 2).mean(axis=1), [[0.0, -0.6], [0.3, 0.1]])
    assert np.allclose(constant_version(varying), constant)
    assert not np.allclose(varying, constant)
    with pytest.raises(ValueError):
        varying_version(varying, residuals, magnitudes, np.random.default_rng(1))


def test_rest_window_filter() -> None:
    still = np.tile([[200.0, 300.0]], (11, 1)) + np.linspace(0, 0.5, 11)[:, None]
    assert is_rest_window(still)
    moving = still.copy()
    moving[-1] += [3.0, 0.0]
    assert not is_rest_window(moving)


SPEC = {
    **{
        f"{label}-{k}": {"structure": k, "label": label}
        for label in ("push", "lateral", "small")
        for k in ("constant", "varying")
    },
    **{
        f"{label}-{k}": {"structure": k, "label": label}
        for label in ("own", "b1", "b2")
        for k in ("constant", "varying")
    },
}


def _record(identifier, state, scales, rng, noise=0.5):
    """scales: {(label, structure): predicted/true factor}; real frames are near-exact."""
    labels = ("push", "lateral", "small") if state == "challenge" else ("own", "b1", "b2")
    ids = sorted(f"{label}-{k}" for label in labels for k in ("constant", "varying"))
    true = np.zeros((6, 2, 2))
    predicted = np.zeros_like(true)
    for index, key in enumerate(ids):
        label, structure = key.split("-")
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        step1 = direction * rng.uniform(10, 25)
        true[index] = [step1, 2 * step1]
        predicted[index] = true[index] * scales.get((label, structure), 1.0)
    real = true + rng.normal(scale=noise, size=true.shape)
    return family_record(identifier, state, ids, SPEC, true, predicted, real, [True] * 6)


def _records(challenge_scales, archive_scales, count=40):
    rng = np.random.default_rng(3)
    return [_record(f"challenge-{i}", "challenge", challenge_scales, rng) for i in range(count)] + [
        _record(f"archive-{i}", "archive", archive_scales, rng) for i in range(count)
    ]


def test_ratio_is_one_for_an_oracle_and_tracks_shrinkage() -> None:
    records = _records({}, {})
    assert ratio(records, "challenge", "constant") == pytest.approx(1.0, abs=0.02)
    shrunk = _records({("push", "constant"): 0.5}, {})
    assert ratio(shrunk, "challenge", "constant", {"push"}) == pytest.approx(0.5, abs=0.02)


def test_structure_caused_shrinkage_is_detected() -> None:
    constant_only = {(label, "constant"): 0.55 for label in ("push", "lateral", "small")}
    summary = summarize_shift(_records(constant_only, {}), seed=1, samples=300)
    criteria = summary["criteria"]
    assert criteria["passed"]
    assert criteria["structure"] == "supported"
    assert criteria["state"] == "not_established"
    assert criteria["magnitude"] == "not_established"


def test_state_caused_shrinkage_is_detected() -> None:
    everywhere = {
        (label, k): 0.55 for label in ("push", "lateral", "small") for k in ("constant", "varying")
    }
    criteria = summarize_shift(_records(everywhere, {}), seed=1, samples=300)["criteria"]
    assert criteria["passed"]
    assert criteria["state"] == "supported"
    assert criteria["structure"] == "not_established"


def test_failed_manipulation_check_makes_everything_inconclusive() -> None:
    criteria = summarize_shift(_records({}, {}), seed=1, samples=300)["criteria"]
    assert not criteria["manipulation_checks"]["challenge_constant_push_shrinks"]
    assert not criteria["passed"]
    assert {criteria[k] for k in ("structure", "magnitude", "state")} == {
        "inconclusive_manipulation_check"
    }


def test_safe_goal_reset_reproduces_start_without_termination() -> None:
    from jepa_action_effects.pilot_collection import _states_from_steps
    from jepa_action_effects.shift001 import safe_goal
    from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family

    start = [150.0, 150.0, 300.0, 250.0, 0.4, 0.0, 0.0]
    goal = safe_goal(start)
    assert np.linalg.norm(np.subtract(goal[2:4], start[2:4])) > 150
    assert np.linalg.norm(np.subtract(goal[0:2], start[0:2])) > 150
    recipe = ReplayRecipe(
        "archive-test", 0, np.zeros((10, 2)), {"state": start, "goal_state": goal}
    )
    candidates = [Candidate("own-constant", np.zeros((10, 2)))]
    replay = run_family(recipe, candidates, ["own-constant"])
    history, future, _ = _states_from_steps(replay.branches["own-constant"])
    assert np.allclose(history[0], start[:5], atol=1e-6)
    assert np.allclose(future[-1, 2:5], start[2:5], atol=1e-6)  # block stays at rest
