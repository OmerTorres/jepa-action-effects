"""Pure measurements and sealed decision rules for SHIFT-001 (states × action structure).

Every family has six candidates organised as three matched pairs. The two members of
a pair share per-block mean controls: `constant` tiles the block mean, `varying` adds
a zero-mean archive residual. Two state types (challenge layouts and archive rest
windows) are compared between families.

The shrinkage ratio of a cell is R = slope_pred / slope_real on moved candidates
(true step-2 block displacement > delta). Both slopes are pooled least squares
through the origin against the same true displacements, so
R = Σ(pred·true) / Σ(real·true).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DELTA = 5.12
FLOOR_LIMIT = 0.25
SHRINK_UPPER = 0.8  # challenge constant push must reproduce PHYS-002 shrinkage
FAITHFUL_LOWER = 0.9  # archive own-future varying must reproduce the audit
MIN_FAMILIES = 30
RESIDUAL_TOLERANCE = 0.1
STATES = ("challenge", "archive")
STRUCTURES = ("constant", "varying")
# Cell statistics use these pairs by default. The challenge `lateral` pair is excluded:
# its +0.2 sideways offset steers the agent past the block, so it moves the block in
# almost no family (0 constant / 9 varying of 160 at collection) and would unbalance
# the constant and varying cells. It is still recorded per candidate.
PRIMARY_LABELS = {"challenge": frozenset({"push", "small"}), "archive": None}


# --------------------------------------------------------------------------- actions
def block_residuals(blocks: np.ndarray) -> np.ndarray:
    """(N,5,2) control blocks minus their own block means (zero-mean per block)."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (5, 2):
        raise ValueError("Blocks must have shape (N,5,2)")
    return values - values.mean(axis=1, keepdims=True)


def constant_version(controls: np.ndarray) -> np.ndarray:
    """Replace each 5-control block by its mean (same mean command, zero spread)."""
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) % 5:
        raise ValueError("Controls must have shape (5k,2)")
    blocks = values.reshape(-1, 5, 2)
    return np.repeat(blocks.mean(axis=1), 5, axis=0)


def varying_version(
    constant: np.ndarray,
    residuals: np.ndarray,
    magnitudes: np.ndarray,
    rng: np.random.Generator,
    tolerance: float = RESIDUAL_TOLERANCE,
    tries: int = 200,
) -> np.ndarray:
    """Add one archive residual per block, drawn from blocks of similar magnitude.

    Block means are preserved exactly; draws that leave [-1, 1] are redrawn.
    """
    base = np.asarray(constant, dtype=np.float64).reshape(-1, 5, 2)
    if not np.allclose(base, base.mean(axis=1, keepdims=True)):
        raise ValueError("Base controls must be constant within each block")
    result = []
    for block in base:
        magnitude = float(np.linalg.norm(block[0]))
        pool = np.flatnonzero(np.abs(np.asarray(magnitudes) - magnitude) <= tolerance)
        if len(pool) == 0:
            raise ValueError(f"No archive residuals near magnitude {magnitude:.3f}")
        for _ in range(tries):
            candidate = block + residuals[int(rng.choice(pool))]
            if np.all(np.abs(candidate) <= 1.0):
                result.append(candidate)
                break
        else:
            raise ValueError("Could not draw an in-bounds residual")
    return np.concatenate(result)


def is_rest_window(block_positions: np.ndarray, limit: float = 1.0) -> bool:
    """True if the block stays within `limit` units of its first position."""
    positions = np.asarray(block_positions, dtype=np.float64)
    return bool(np.max(np.linalg.norm(positions - positions[0], axis=1)) < limit)


# --------------------------------------------------------------------------- records
def family_record(
    identifier: str,
    state: str,
    candidate_ids: Sequence[str],
    spec: dict[str, dict],
    true: np.ndarray,
    predicted: np.ndarray,
    real: np.ndarray,
    contacts: Sequence[bool],
    delta: float = DELTA,
) -> dict[str, Any]:
    """Per-candidate sufficient statistics; displacements are (C,2 steps,2)."""
    if state not in STATES:
        raise ValueError(f"Unknown state type {state}")
    true, predicted, real = (np.asarray(a, dtype=np.float64) for a in (true, predicted, real))
    count = len(candidate_ids)
    if not (true.shape == predicted.shape == real.shape == (count, 2, 2)):
        raise ValueError("Displacements must be (candidates, 2 steps, 2)")
    rows = []
    for index, candidate_id in enumerate(candidate_ids):
        t2, p2, r2 = true[index, 1], predicted[index, 1], real[index, 1]
        t1, p1, r1 = true[index, 0], predicted[index, 0], real[index, 0]
        magnitude = float(np.linalg.norm(t2))
        rows.append(
            {
                "id": candidate_id,
                "structure": spec[candidate_id]["structure"],
                "label": spec[candidate_id]["label"],
                "contact": bool(contacts[index]),
                "moved": magnitude > delta,
                "true_magnitude": magnitude,
                "pred_inner": float(p2 @ t2),
                "real_inner": float(r2 @ t2),
                "energy": float(t2 @ t2),
                "real_error": float(np.linalg.norm(r2 - t2)),
                "pred_error": float(np.linalg.norm(p2 - t2)),
                "pred_inner_step1": float(p1 @ t1),
                "real_inner_step1": float(r1 @ t1),
            }
        )
    return {"id": identifier, "state": state, "candidates": rows}


def _select(records, state, structure=None, labels=None):
    if labels is None:
        labels = PRIMARY_LABELS[state]
    for record in records:
        if record["state"] != state:
            continue
        for row in record["candidates"]:
            if not row["moved"]:
                continue
            if structure is not None and row["structure"] != structure:
                continue
            if labels is not None and row["label"] not in labels:
                continue
            yield record["id"], row


def ratio(records, state, structure=None, labels=None, step: int = 2) -> float | None:
    suffix = "" if step == 2 else "_step1"
    rows = [row for _, row in _select(records, state, structure, labels)]
    real = sum(row[f"real_inner{suffix}"] for row in rows)
    if not rows or real <= 0:
        return None
    return sum(row[f"pred_inner{suffix}"] for row in rows) / real


def slope(records, state, structure=None, labels=None, source: str = "pred") -> float | None:
    rows = [row for _, row in _select(records, state, structure, labels)]
    energy = sum(row["energy"] for row in rows)
    return None if energy <= 0 else sum(row[f"{source}_inner"] for row in rows) / energy


def floor_ratio(records, state, structure=None) -> float | None:
    rows = [row for _, row in _select(records, state, structure)]
    if not rows:
        return None
    median = float(np.median([row["true_magnitude"] for row in rows]))
    return float(np.mean([row["real_error"] for row in rows]) / median)


def moved_families(records, state, structure) -> int:
    return len({family for family, _ in _select(records, state, structure)})


def _difference(left, right):
    return None if left is None or right is None else left - right


STATISTICS = {
    **{
        f"R_{state}_{structure}": (lambda r, s=state, k=structure: ratio(r, s, k))
        for state in STATES
        for structure in STRUCTURES
    },
    **{
        f"R1_{state}_{structure}": (lambda r, s=state, k=structure: ratio(r, s, k, step=1))
        for state in STATES
        for structure in STRUCTURES
    },
    **{
        f"slope_{source}_{state}_{structure}": (
            lambda r, s=state, k=structure, src=source: slope(r, s, k, source=src)
        )
        for source in ("pred", "real")
        for state in STATES
        for structure in STRUCTURES
    },
    **{
        f"floor_{state}_{structure}": (lambda r, s=state, k=structure: floor_ratio(r, s, k))
        for state in STATES
        for structure in STRUCTURES
    },
    "R_challenge_constant_push": lambda r: ratio(r, "challenge", "constant", {"push"}),
    "R_challenge_constant_small": lambda r: ratio(r, "challenge", "constant", {"small"}),
    "R_challenge_varying_small": lambda r: ratio(r, "challenge", "varying", {"small"}),
    "R_archive_varying_own": lambda r: ratio(r, "archive", "varying", {"own"}),
    "dS_challenge": lambda r: _difference(
        ratio(r, "challenge", "varying"), ratio(r, "challenge", "constant")
    ),
    "dS_archive": lambda r: _difference(
        ratio(r, "archive", "varying"), ratio(r, "archive", "constant")
    ),
    "dM_challenge": lambda r: _difference(
        ratio(r, "challenge", "constant", {"small"}), ratio(r, "challenge", "constant", {"push"})
    ),
    "dT_varying": lambda r: _difference(
        ratio(r, "archive", "varying"), ratio(r, "challenge", "varying", {"small"})
    ),
}


def summarize_shift(records: Sequence[dict], seed: int, samples: int = 2000) -> dict[str, Any]:
    """Estimates and 95% intervals; families are resampled within each state type."""
    records = list(records)
    pools = [[r for r in records if r["state"] == state] for state in STATES]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in STATISTICS}
    for _ in range(samples):
        chosen = []
        for pool in pools:
            if pool:
                chosen.extend(pool[i] for i in rng.integers(0, len(pool), len(pool)))
        for name, statistic in STATISTICS.items():
            value = statistic(chosen)
            if value is not None and np.isfinite(value):
                draws[name].append(float(value))
    summary: dict[str, Any] = {}
    for name, statistic in STATISTICS.items():
        estimate = statistic(records)
        values = draws[name]
        summary[name] = {
            "estimate": None if estimate is None else float(estimate),
            "ci95": np.percentile(values, (2.5, 97.5)).tolist()
            if len(values) >= samples // 2
            else None,
        }
    summary["moved_families"] = {
        f"{state}_{structure}": moved_families(records, state, structure)
        for state in STATES
        for structure in STRUCTURES
    }
    summary["families"] = {state: len(pool) for state, pool in zip(STATES, pools, strict=True)}
    summary["criteria"] = apply_shift_criteria(summary)
    return summary


# --------------------------------------------------------------------------- rules
def apply_shift_criteria(summary: dict[str, Any]) -> dict[str, Any]:
    """Sealed SHIFT-001 rules. Manipulation checks gate every factor verdict."""

    def interval(name):
        return summary[name]["ci95"]

    def lower_above(name, bound=0.0):
        ci = interval(name)
        return ci is not None and ci[0] > bound

    shrink = interval("R_challenge_constant_push")
    faithful = interval("R_archive_varying_own")
    floors = [summary[f"floor_{s}_{k}"]["estimate"] for s in STATES for k in STRUCTURES]
    checks = {
        "challenge_constant_push_shrinks": shrink is not None and shrink[1] < SHRINK_UPPER,
        "archive_own_future_faithful": faithful is not None
        and (faithful[0] <= 1.0 <= faithful[1] or faithful[0] > FAITHFUL_LOWER),
        "floors_below_limit": all(value is not None and value < FLOOR_LIMIT for value in floors),
        "enough_moved_families": all(
            count >= MIN_FAMILIES for count in summary["moved_families"].values()
        ),
    }
    passed = all(checks.values())
    if not passed:
        status = "inconclusive_manipulation_check"
        return {
            "manipulation_checks": checks,
            "passed": False,
            "structure": status,
            "magnitude": status,
            "state": status,
        }
    verdict = {
        "structure": "supported" if lower_above("dS_challenge") else "not_established",
        "magnitude": "supported" if lower_above("dM_challenge") else "not_established",
        "state": "supported" if lower_above("dT_varying") else "not_established",
    }
    return {"manipulation_checks": checks, "passed": True, **verdict}
