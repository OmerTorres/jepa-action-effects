"""Adapt validated SIM-001 replay traces to checkpoint-independent evaluation truth."""

from __future__ import annotations

import numpy as np

from jepa_action_effects.evaluation import BranchTruth, FamilyTruth
from jepa_action_effects.simulator_replay import STATE_ATOL, FamilyResult


def family_truth_from_replay(
    family: FamilyResult,
    *,
    horizon_native_steps: int,
    split: str | None = None,
    group_id: str | None = None,
    atol: float = STATE_ATOL,
) -> FamilyTruth:
    """Build truth while preserving incomplete branches as family exclusions."""
    if horizon_native_steps <= 0:
        raise ValueError("horizon_native_steps must be positive")
    if not family.branches:
        raise ValueError("replay family must contain branches")

    current_blocks: list[np.ndarray] = []
    current_agents: list[np.ndarray] = []
    branches: list[BranchTruth] = []
    incomplete: list[str] = []
    wrong_horizon: list[str] = []
    expected_controls = 5 * horizon_native_steps
    for candidate_id in sorted(family.branches):
        branch = family.branches[candidate_id]
        if branch.candidate_id != candidate_id:
            raise ValueError("branch dictionary key and candidate_id differ")
        warmup_steps = [step for step in branch.physics_steps if step.phase == "warmup"]
        candidate_steps = [step for step in branch.physics_steps if step.phase == "candidate"]
        if not warmup_steps or not candidate_steps:
            raise ValueError(f"branch {candidate_id} lacks warm-up or candidate physics steps")
        current_blocks.append(np.asarray(warmup_steps[-1].block["position"], dtype=np.float64))
        current_agents.append(np.asarray(warmup_steps[-1].agent["position"], dtype=np.float64))
        endpoint_block = np.asarray(candidate_steps[-1].block["position"], dtype=np.float64)
        endpoint_agent = np.asarray(candidate_steps[-1].agent["position"], dtype=np.float64)
        if len(branch.observations):
            np.testing.assert_allclose(
                endpoint_agent,
                branch.observations[-1, 0:2],
                atol=atol,
                rtol=0,
                err_msg=f"agent endpoint misaligned for {candidate_id}",
            )
            np.testing.assert_allclose(
                endpoint_block,
                branch.observations[-1, 2:4],
                atol=atol,
                rtol=0,
                err_msg=f"block endpoint misaligned for {candidate_id}",
            )
        branches.append(
            BranchTruth(
                candidate_id,
                endpoint_block,
                endpoint_agent,
                any(step.agent_block_contact for step in candidate_steps),
                np.asarray([step.block["position"] for step in candidate_steps]),
                "physics_substep",
            )
        )
        if not branch.completed:
            incomplete.append(candidate_id)
        if branch.expected_controls != expected_controls:
            wrong_horizon.append(candidate_id)

    reference_block = current_blocks[0]
    reference_agent = current_agents[0]
    for candidate_id, block, agent in zip(
        sorted(family.branches), current_blocks, current_agents, strict=True
    ):
        np.testing.assert_allclose(
            block,
            reference_block,
            atol=atol,
            rtol=0,
            err_msg=f"branch-point block mismatch for {candidate_id}",
        )
        np.testing.assert_allclose(
            agent,
            reference_agent,
            atol=atol,
            rtol=0,
            err_msg=f"branch-point agent mismatch for {candidate_id}",
        )

    exclusion_reasons = []
    if incomplete:
        exclusion_reasons.append("incomplete_candidates:" + ",".join(sorted(incomplete)))
    if wrong_horizon:
        exclusion_reasons.append("horizon_mismatch_candidates:" + ",".join(sorted(wrong_horizon)))
    return FamilyTruth(
        family.family_id,
        reference_block,
        reference_agent,
        tuple(branches),
        horizon_native_steps,
        split,
        group_id,
        eligible=not exclusion_reasons,
        exclusion_reasons=tuple(exclusion_reasons),
    )
