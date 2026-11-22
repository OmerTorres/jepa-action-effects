"""Pure measurements and sealed decision rules for SHIFT-002 (push-magnitude dose-response).

SHIFT-001 development (exploratory) showed no within-block-structure effect but a large
magnitude effect: a 0.21 push was over-predicted and a 0.6 push under-predicted. It
also failed its effect-floor check, because small pushes move the block little and
archive-scene frames read out with about 12 units of noise. SHIFT-002 tests the
magnitude hypothesis on fresh families.

- **Challenge families:** constant pushes at 0.15, 0.30, 0.45 and 0.60, plus varying
  versions at 0.30 and 0.60.
- **Archive rest windows:** the window's own future controls scaled by 1.0, 1.5 and
  2.0, each constant and varying.

The shrinkage ratio of a cell is R = Σ(pred·true) / Σ(real·true) over moved
candidates (see `shift001_metrics`). Readout noise is zero-mean in both inner
products, so R is judged by a readout-bias gate (slope_real ≈ 1) rather than by the
effect-floor ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

CHALLENGE_LEVELS = (0.15, 0.30, 0.45, 0.60)
CHALLENGE_VARYING = (0.30, 0.60)
ARCHIVE_SCALES = (1.0, 1.5, 2.0)
STATES = ("challenge", "archive")
STRUCTURES = ("constant", "varying")
HIGH_SHRINK_UPPER = 1.0  # the 0.60 push must show some shrinkage for a dose-response test
FAITHFUL_LOWER = 0.9
READOUT_BAND = (0.85, 1.15)
MIN_FAMILIES = 30
EQUIVALENCE = 0.1


def challenge_label(level: float) -> str:
    return f"m{level:.2f}"


def archive_label(scale: float) -> str:
    return f"x{scale:.1f}"


LOW = frozenset(challenge_label(level) for level in (0.15, 0.30))
HIGH = frozenset({challenge_label(0.60)})
MATCHED = frozenset(challenge_label(level) for level in CHALLENGE_VARYING)


def _rows(records, state, structure=None, labels=None):
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
    rows = [row for _, row in _rows(records, state, structure, labels)]
    real = sum(row[f"real_inner{suffix}"] for row in rows)
    if not rows or real <= 0:
        return None
    return sum(row[f"pred_inner{suffix}"] for row in rows) / real


def slope_real(records, state, structure=None, labels=None) -> float | None:
    rows = [row for _, row in _rows(records, state, structure, labels)]
    energy = sum(row["energy"] for row in rows)
    return None if energy <= 0 else sum(row["real_inner"] for row in rows) / energy


def moved_families(records, state, structure=None, labels=None) -> int:
    return len({family for family, _ in _rows(records, state, structure, labels)})


def _difference(left, right):
    return None if left is None or right is None else left - right


def _statistics() -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for level in CHALLENGE_LEVELS:
        label = {challenge_label(level)}
        stats[f"R_challenge_{challenge_label(level)}_constant"] = lambda r, lab=label: ratio(
            r, "challenge", "constant", lab
        )
    for level in CHALLENGE_VARYING:
        label = {challenge_label(level)}
        stats[f"R_challenge_{challenge_label(level)}_varying"] = lambda r, lab=label: ratio(
            r, "challenge", "varying", lab
        )
    for scale in ARCHIVE_SCALES:
        label = {archive_label(scale)}
        stats[f"R_archive_{archive_label(scale)}"] = lambda r, lab=label: ratio(
            r, "archive", None, lab
        )
        for structure in STRUCTURES:
            stats[f"R_archive_{archive_label(scale)}_{structure}"] = (
                lambda r, lab=label, k=structure: ratio(r, "archive", k, lab)
            )
    stats["R_challenge_low_constant"] = lambda r: ratio(r, "challenge", "constant", LOW)
    stats["R_challenge_high_constant"] = lambda r: ratio(r, "challenge", "constant", HIGH)
    stats["R1_challenge_high_constant"] = lambda r: ratio(r, "challenge", "constant", HIGH, step=1)
    stats["dM_challenge"] = lambda r: _difference(
        ratio(r, "challenge", "constant", LOW), ratio(r, "challenge", "constant", HIGH)
    )
    stats["dM_archive"] = lambda r: _difference(
        ratio(r, "archive", None, {archive_label(1.0)}),
        ratio(r, "archive", None, {archive_label(2.0)}),
    )
    stats["dS_challenge"] = lambda r: _difference(
        ratio(r, "challenge", "varying", MATCHED), ratio(r, "challenge", "constant", MATCHED)
    )
    stats["dS_archive"] = lambda r: _difference(
        ratio(r, "archive", "varying"), ratio(r, "archive", "constant")
    )
    for state in STATES:
        for structure in STRUCTURES:
            stats[f"slope_real_{state}_{structure}"] = lambda r, s=state, k=structure: slope_real(
                r, s, k
            )
    return stats


STATISTICS = _statistics()
GATE_FAMILY_CELLS = {
    "challenge_low_constant": ("challenge", "constant", LOW),
    "challenge_high_constant": ("challenge", "constant", HIGH),
    "archive_x1.0": ("archive", None, {archive_label(1.0)}),
    "archive_x2.0": ("archive", None, {archive_label(2.0)}),
}


def describe(records: Sequence[dict]) -> dict[str, Any]:
    """Per state and label: moved candidates, median true displacement, real-frame error."""
    table: dict[str, Any] = {}
    for state in STATES:
        labels = sorted(
            {row["label"] for r in records if r["state"] == state for row in r["candidates"]}
        )
        for label in labels:
            for structure in STRUCTURES:
                rows = [row for _, row in _rows(records, state, structure, {label})]
                if rows:
                    table[f"{state}_{label}_{structure}"] = {
                        "moved": len(rows),
                        "median_true_displacement": float(
                            np.median([x["true_magnitude"] for x in rows])
                        ),
                        "mean_real_error": float(np.mean([x["real_error"] for x in rows])),
                        "mean_pred_error": float(np.mean([x["pred_error"] for x in rows])),
                    }
    return table


def summarize_dose(records: Sequence[dict], seed: int, samples: int = 2000) -> dict[str, Any]:
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
        name: moved_families(records, *cell) for name, cell in GATE_FAMILY_CELLS.items()
    }
    summary["families"] = {state: len(pool) for state, pool in zip(STATES, pools, strict=True)}
    summary["describe"] = describe(records)
    summary["criteria"] = apply_dose_criteria(summary)
    return summary


def _structure_verdict(interval) -> str:
    if interval is None:
        return "inconclusive"
    low, high = interval
    if -EQUIVALENCE < low and high < EQUIVALENCE:
        return f"no_effect_within_{EQUIVALENCE}"
    if low > 0:
        return "varying_less_shrunk"
    if high < 0:
        return "varying_more_shrunk"
    return "inconclusive"


def apply_dose_criteria(summary: dict[str, Any]) -> dict[str, Any]:
    """Sealed SHIFT-002 rules. Manipulation checks gate every verdict."""

    def ci(name):
        return summary[name]["ci95"]

    high = ci("R_challenge_high_constant")
    faithful = ci("R_archive_x1.0_varying")
    readout = [ci(f"slope_real_{s}_{k}") for s in STATES for k in STRUCTURES]
    checks = {
        "challenge_high_shrinks": high is not None and high[1] < HIGH_SHRINK_UPPER,
        "archive_unit_faithful": faithful is not None
        and (faithful[0] <= 1.0 <= faithful[1] or faithful[0] > FAITHFUL_LOWER),
        "readout_unbiased": all(
            band is not None and READOUT_BAND[0] < band[0] and band[1] < READOUT_BAND[1]
            for band in readout
        ),
        "enough_moved_families": all(
            count >= MIN_FAMILIES for count in summary["moved_families"].values()
        ),
    }
    if not all(checks.values()):
        status = "inconclusive_manipulation_check"
        return {
            "manipulation_checks": checks,
            "passed": False,
            "magnitude_challenge": status,
            "magnitude_archive": status,
            "structure_challenge": status,
            "structure_archive": status,
        }

    def positive(name):
        interval = ci(name)
        return "supported" if interval is not None and interval[0] > 0 else "not_established"

    return {
        "manipulation_checks": checks,
        "passed": True,
        "magnitude_challenge": positive("dM_challenge"),
        "magnitude_archive": positive("dM_archive"),
        "structure_challenge": _structure_verdict(ci("dS_challenge")),
        "structure_archive": _structure_verdict(ci("dS_archive")),
    }
