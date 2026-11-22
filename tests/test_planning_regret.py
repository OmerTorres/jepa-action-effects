"""Model-free fixtures for H3-PLAN-001 regret metrics and the goal-image renderer."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.planning_metrics import family_record, goal_regrets, summarize_planning


def _setup(rng):
    block_end = rng.uniform(150, 350, size=(6, 2))
    # A latent where distance tracks block position, so a perfect predictor plans well.
    real = np.concatenate([block_end / 100.0, np.zeros((6, 190))], axis=1)
    return block_end, real


def test_goal_regrets_oracle_random_and_informative_flag() -> None:
    rng = np.random.default_rng(0)
    block_end, real = _setup(rng)
    goal = block_end[3] + 1.0
    goal_latent = np.concatenate([goal / 100.0, np.zeros(190)])
    row = goal_regrets(block_end, goal, real, real, goal_latent)
    assert row["real_latent"] == pytest.approx(0.0)
    assert row["jepa"] == pytest.approx(0.0)
    cost = np.linalg.norm(block_end - goal, axis=1)
    assert row["random"] == pytest.approx(cost.mean() - cost.min())
    assert row["informative"] == (cost.max() - cost.min() > 5.12)
    same = goal_regrets(np.full((6, 2), 200.0), goal, real, real, goal_latent)
    assert not same["informative"] and same["random"] == pytest.approx(0.0)


def _records(shrink, count=60):
    rng = np.random.default_rng(1)
    records = []
    for index in range(count):
        block_end, real = _setup(rng)
        start = block_end.mean(axis=0)
        predicted = real.copy()
        predicted[:, :2] = (start + (block_end - start) * shrink) / 100.0
        goals = [list(rng.uniform(120, 380, size=2)) for _ in range(4)]
        latents = np.stack([np.concatenate([np.asarray(g) / 100.0, np.zeros(190)]) for g in goals])
        records.append(
            family_record(
                f"challenge-{index}",
                "challenge",
                block_end,
                goals,
                [True] * 4,
                predicted,
                real,
                latents,
            )
        )
    return records


def test_perfect_predictor_adds_no_regret() -> None:
    summary = summarize_planning(_records(1.0), seed=1, samples=300)
    assert summary["jepa_minus_real_all"]["estimate"] == pytest.approx(0.0)
    assert summary["criteria"]["passed"]
    assert summary["criteria"]["h3"] == "not_established"


def test_noisy_predictor_adds_regret() -> None:
    rng = np.random.default_rng(2)
    records = []
    for index in range(60):
        block_end, real = _setup(rng)
        predicted = real.copy()
        predicted[:, :2] += rng.normal(scale=0.6, size=(6, 2))
        goals = [list(rng.uniform(120, 380, size=2)) for _ in range(4)]
        latents = np.stack([np.concatenate([np.asarray(g) / 100.0, np.zeros(190)]) for g in goals])
        records.append(
            family_record(
                f"challenge-{index}",
                "challenge",
                block_end,
                goals,
                [True] * 4,
                predicted,
                real,
                latents,
            )
        )
    summary = summarize_planning(records, seed=1, samples=300)
    assert summary["criteria"]["h3"] == "supported"


def test_goal_image_places_the_block_at_the_goal() -> None:
    from jepa_action_effects.planning_regret import goal_images
    from jepa_action_effects.readout_scale import _new_env, _place

    start = np.array([100.0, 100.0, 300.0, 300.0, 0.3])
    images, valid = goal_images(start, [[256.0, 256.0], [300.0, 300.0]])
    assert images.shape == (2, 224, 224, 3)
    assert valid == [True, True]
    assert not np.array_equal(images[0], images[1])
    env = _new_env()
    try:
        _place(env, np.array([100.0, 100.0, 256.0, 256.0, 0.3]))
        assert np.array_equal(env.render(), images[0])
    finally:
        env.close()
