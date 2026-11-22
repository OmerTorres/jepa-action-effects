"""Fixtures for H3-PLAN-002 agent-matched goal images and regret."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.planning002_metrics import (
    choice_histogram,
    family_record,
    matched_goal_regrets,
)
from jepa_action_effects.planning_metrics import summarize_planning


def _latent(agent, block, agent_weight=5.0):
    """Toy latent dominated by the agent; zero-padded to 192 dimensions."""
    return np.concatenate(
        [np.asarray(agent) * agent_weight / 100.0, np.asarray(block) / 100.0, np.zeros(188)]
    )


def _family(rng):
    agent_end = rng.uniform(100, 400, size=(6, 2))
    agent_end[0] = [250.0, 250.0]  # candidate 0 ("hold") keeps the agent at its start
    block_end = rng.uniform(150, 350, size=(6, 2))
    real = np.stack([_latent(a, b) for a, b in zip(agent_end, block_end, strict=True)])
    return agent_end, block_end, real


def test_matched_goal_images_remove_the_agent_confound() -> None:
    rng = np.random.default_rng(0)
    agent_end, block_end, real = _family(rng)
    goal = block_end[4] + 1.0
    # Start-pose goal image (H3-PLAN-001): agent term dominates and favours candidate 0.
    start_goal = _latent([250.0, 250.0], goal)
    assert int(np.argmin(np.linalg.norm(real - start_goal, axis=1))) == 0
    # Agent-matched goal images: the cost reduces to block distance.
    matched = np.stack([_latent(a, goal) for a in agent_end])
    row = matched_goal_regrets(block_end, goal, real, real, matched)
    assert row["choice"]["real_latent"] == 4
    assert row["real_latent"] == pytest.approx(0.0)


def _records(noise, count=60):
    rng = np.random.default_rng(1)
    records = []
    for index in range(count):
        agent_end, block_end, real = _family(rng)
        predicted = real.copy()
        predicted[:, 2:4] += rng.normal(scale=noise, size=(6, 2))
        goals = [list(rng.uniform(120, 380, size=2)) for _ in range(4)]
        latents = np.stack([np.stack([_latent(a, g) for a in agent_end]) for g in goals])
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


def test_oracle_adds_no_regret_and_noise_adds_regret() -> None:
    oracle = summarize_planning(_records(0.0), seed=1, samples=300)
    assert oracle["criteria"]["passed"]
    assert oracle["jepa_minus_real_all"]["estimate"] == pytest.approx(0.0)
    assert oracle["criteria"]["h3"] == "not_established"
    noisy = summarize_planning(_records(0.6), seed=1, samples=300)
    assert noisy["criteria"]["h3"] == "supported"
    counts = choice_histogram(_records(0.0), "real_latent")
    assert sum(counts) == 60 * 4 and max(counts) < 0.5 * sum(counts)


def test_invalid_goal_is_excluded_and_shapes_are_checked() -> None:
    rng = np.random.default_rng(2)
    agent_end, block_end, real = _family(rng)
    goals = [[200.0, 200.0], [300.0, 300.0]]
    latents = np.stack([np.stack([_latent(a, g) for a in agent_end]) for g in goals])
    record = family_record("c", "challenge", block_end, goals, [True, False], real, real, latents)
    assert record["valid_goals"] == 1
    with pytest.raises(ValueError):
        family_record("c", "challenge", block_end, goals, [True, True], real, real, latents[:, :3])


def test_matched_goal_images_differ_only_by_block_when_agents_match() -> None:
    from jepa_action_effects.planning002 import matched_goal_images

    finals = np.array([[100.0, 100.0, 300.0, 300.0, 0.3], [100.0, 100.0, 320.0, 280.0, 0.3]])
    images, valid = matched_goal_images(finals, [[256.0, 256.0]])
    assert images.shape == (1, 2, 224, 224, 3)
    assert valid == [True]
    # Same agent, same angle, block at the same goal: identical images.
    assert np.array_equal(images[0, 0], images[0, 1])
    moved = finals.copy()
    moved[1, :2] = [400.0, 120.0]
    images, _ = matched_goal_images(moved, [[256.0, 256.0]])
    assert not np.array_equal(images[0, 0], images[0, 1])
