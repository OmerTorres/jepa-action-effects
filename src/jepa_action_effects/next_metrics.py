"""Pure NEXT-001 measurements: action support, boundary pairs, floor-relative errors.

Every function here is deterministic numpy code with no simulator or model access,
so the behavioral tests can pin down the statistics before any Mac run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

import numpy as np

DELTA = 5.12
BOUNDARY_TOLERANCE = 0.2
REGIMES = ("neither", "exactly_one", "both")
FLOOR_RATIO_LIMIT = 0.25


# --------------------------------------------------------------------------- actions
def block_features(blocks: np.ndarray) -> dict[str, np.ndarray]:
    """Mean control magnitude and within-block spread for (...,5,2) control blocks."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.shape[-2:] != (5, 2):
        raise ValueError("Control blocks must end in shape (5,2)")
    magnitude = np.linalg.norm(values, axis=-1).mean(axis=-1)
    differences = values[..., :, None, :] - values[..., None, :, :]
    spread = np.linalg.norm(differences, axis=-1).max(axis=(-1, -2))
    return {"magnitude": magnitude, "spread": spread}


def archive_blocks(actions: np.ndarray, offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Non-overlapping 5-control blocks from each episode start; NaN blocks dropped."""
    actions = np.asarray(actions)
    blocks = []
    for offset, length in zip(np.asarray(offsets), np.asarray(lengths), strict=True):
        count = int(length) // 5
        if count:
            blocks.append(actions[int(offset) : int(offset) + 5 * count].reshape(count, 5, 2))
    if not blocks:
        raise ValueError("No complete five-control archive blocks")
    result = np.concatenate(blocks).astype(np.float64)
    return result[np.isfinite(result).all(axis=(1, 2))]


def percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right") / len(
        ordered
    )


def support_summary(
    archive: np.ndarray, study: np.ndarray, seed: int, sample: int = 200_000
) -> dict[str, Any]:
    """Compare study control blocks with the archive's block distribution.

    Reports marginal feature percentiles, joint tail mass (archive blocks at least as
    large *and* at least as constant), and 10-D nearest-neighbour distances relative
    to archive-to-archive distances on a disjoint archive subsample.
    """
    from sklearn.neighbors import NearestNeighbors

    archive = np.asarray(archive, dtype=np.float64)
    study = np.asarray(study, dtype=np.float64)
    if archive.ndim != 3 or study.ndim != 3 or len(archive) < 10 or not len(study):
        raise ValueError("Expected nonempty (N,5,2) archive and study blocks")
    reference = block_features(archive)
    observed = block_features(study)
    quantiles = [1, 5, 25, 50, 75, 95, 99, 99.9]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(archive))
    held = order[: min(2000, len(archive) // 2)]
    bank = order[len(held) : len(held) + min(sample, len(archive) - len(held))]
    flat = archive.reshape(len(archive), 10)
    search = NearestNeighbors(n_neighbors=1).fit(flat[bank])
    archive_nn = search.kneighbors(flat[held])[0][:, 0]
    study_nn = search.kneighbors(study.reshape(len(study), 10))[0][:, 0]
    joint = np.asarray(
        [
            np.mean((reference["magnitude"] >= magnitude) & (reference["spread"] <= spread + 1e-12))
            for magnitude, spread in zip(observed["magnitude"], observed["spread"], strict=True)
        ]
    )
    nn_p99 = float(np.percentile(archive_nn, 99))
    return {
        "archive_blocks": int(len(archive)),
        "study_blocks": int(len(study)),
        "archive_magnitude_percentiles": dict(
            zip(
                map(str, quantiles),
                np.percentile(reference["magnitude"], quantiles).tolist(),
                strict=True,
            )
        ),
        "archive_spread_percentiles": dict(
            zip(
                map(str, quantiles),
                np.percentile(reference["spread"], quantiles).tolist(),
                strict=True,
            )
        ),
        "study_magnitude_rank": _describe(
            percentile_rank(reference["magnitude"], observed["magnitude"])
        ),
        "study_fraction_above_archive_p99_magnitude": float(
            np.mean(observed["magnitude"] > np.percentile(reference["magnitude"], 99))
        ),
        "study_joint_tail_mass": _describe(joint),
        "study_fraction_joint_tail_below_1e-3": float(np.mean(joint < 1e-3)),
        "archive_nn_distance": _describe(archive_nn),
        "study_nn_distance": _describe(study_nn),
        "study_fraction_nn_beyond_archive_p99": float(np.mean(study_nn > nn_p99)),
    }


def _describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


# --------------------------------------------------------------------------- pairs
def boundary_pairs(
    controls: np.ndarray, candidate_ids: Sequence[str], tolerance: float = BOUNDARY_TOLERANCE
) -> list[tuple[int, int]]:
    """Designed near-boundary pairs: max per-control distance <= tolerance, `hold` excluded."""
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(candidate_ids) or values.shape[-1] != 2:
        raise ValueError("Controls must be (candidates, controls, 2) aligned with IDs")
    pairs = []
    for i, j in combinations(range(len(candidate_ids)), 2):
        if "hold" in (candidate_ids[i], candidate_ids[j]):
            continue
        distance = float(np.max(np.linalg.norm(values[i] - values[j], axis=1)))
        if distance <= tolerance + 1e-7:
            pairs.append((i, j))
    return pairs


def regime(contacts: Sequence[bool], i: int, j: int) -> str:
    return REGIMES[int(bool(contacts[i])) + int(bool(contacts[j]))]


def latent_pair_totals(pairs: Sequence[dict], keep: Callable[[dict], bool]) -> dict[str, float]:
    """Sum LATENT-001 pair terms. Pair `A` equals mean(Δẑ·Δz) and `real_spread` mean(Δz²),
    so sum(A)/sum(real_spread) is the least-squares slope of predicted on real differences."""
    selected = [row for row in pairs if keep(row)]
    return {
        "count": len(selected),
        "inner": float(sum(row["A"] for row in selected)),
        "energy": float(sum(row["real_spread"] for row in selected)),
        "predicted_energy": float(sum(row["predicted_spread"] for row in selected)),
    }


# --------------------------------------------------------------------------- PHYS-002
def family_displacements(
    decoded_prediction: np.ndarray,
    decoded_real: np.ndarray,
    decoded_start: np.ndarray,
    true_end: np.ndarray,
    true_start: np.ndarray,
    contacts: Sequence[bool],
    designed_pairs: Sequence[tuple[int, int]],
    goals: Sequence[Sequence[float]] = (),
    delta: float = DELTA,
) -> dict[str, Any]:
    """Floor-relative block displacement/effect errors for one six-candidate family.

    The same frozen readout decodes predicted and real futures; displacements are
    taken relative to the decoded current frame, so a constant readout bias cancels.
    """
    pred = np.asarray(decoded_prediction, dtype=np.float64)
    real = np.asarray(decoded_real, dtype=np.float64)
    start = np.asarray(decoded_start, dtype=np.float64)
    end = np.asarray(true_end, dtype=np.float64)
    origin = np.asarray(true_start, dtype=np.float64)
    count = len(end)
    if pred.shape != (count, 2) or real.shape != (count, 2) or start.shape != (2,):
        raise ValueError("Decoded endpoints must be (candidates,2) and start (2,)")
    if origin.shape != (2,) or len(contacts) != count:
        raise ValueError("True start/contact labels are misaligned")
    true_disp = end - origin
    pred_disp = pred - start
    real_disp = real - start
    err_pred = np.linalg.norm(pred_disp - true_disp, axis=1)
    err_real = np.linalg.norm(real_disp - true_disp, axis=1)
    moved = np.linalg.norm(true_disp, axis=1) > delta
    contact = np.asarray(contacts, dtype=bool)

    def slope_terms(mask: np.ndarray) -> dict[str, float]:
        return {
            "pred_inner": float(np.sum(pred_disp[mask] * true_disp[mask])),
            "real_inner": float(np.sum(real_disp[mask] * true_disp[mask])),
            "energy": float(np.sum(true_disp[mask] ** 2)),
            "count": int(np.sum(mask)),
        }

    pair_rows = []
    designed = {tuple(sorted(pair)) for pair in designed_pairs}
    for i, j in combinations(range(count), 2):
        true_effect = end[i] - end[j]
        e_pred = float(np.linalg.norm((pred[i] - pred[j]) - true_effect))
        e_real = float(np.linalg.norm((real[i] - real[j]) - true_effect))
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "regime": regime(contact, i, j),
                "designed": (i, j) in designed,
                "true_effect": float(np.linalg.norm(true_effect)),
                "meaningful": bool(np.linalg.norm(true_effect) > delta),
                "e_pred": e_pred,
                "e_real": e_real,
                "D": e_pred - e_real,
            }
        )

    def pair_mean(keep: Callable[[dict], bool]) -> dict[str, Any] | None:
        rows = [row for row in pair_rows if keep(row)]
        if not rows:
            return None
        return {
            "count": len(rows),
            **{
                key: float(np.mean([row[key] for row in rows])) for key in ("e_pred", "e_real", "D")
            },
            "effects": [row["true_effect"] for row in rows],
        }

    regret = []
    for goal in goals:
        target = np.asarray(goal, dtype=np.float64)
        cost = np.linalg.norm(end - target, axis=1)
        best = float(np.min(cost))
        regret.append(
            {
                "jepa": float(cost[int(np.argmin(np.linalg.norm(pred - target, axis=1)))] - best),
                "real_frame": float(
                    cost[int(np.argmin(np.linalg.norm(real - target, axis=1)))] - best
                ),
                "random": float(np.mean(cost) - best),
            }
        )

    return {
        "endpoint": {
            "err_pred": float(np.mean(err_pred)),
            "err_real": float(np.mean(err_real)),
            "D": float(np.mean(err_pred - err_real)),
            "moved_count": int(np.sum(moved)),
            "moved_D": float(np.mean((err_pred - err_real)[moved])) if moved.any() else None,
        },
        "slope": {
            "moved": slope_terms(moved),
            "moved_contact": slope_terms(moved & contact),
            "moved_no_contact": slope_terms(moved & ~contact),
        },
        "boundary": {
            name: pair_mean(lambda row, name=name: row["designed"] and row["regime"] == name)
            for name in REGIMES
        },
        "boundary_all": pair_mean(lambda row: row["designed"]),
        "boundary_meaningful": pair_mean(lambda row: row["designed"] and row["meaningful"]),
        "meaningful_all": pair_mean(lambda row: row["meaningful"]),
        "regret": {
            key: float(np.mean([row[key] for row in regret]))
            for key in ("jepa", "real_frame", "random")
        }
        if regret
        else None,
        "pairs": pair_rows,
    }


# --------------------------------------------------------------------------- statistics
def bootstrap(
    records: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float | None],
    seed: int,
    samples: int = 2000,
) -> dict[str, Any]:
    """Family bootstrap preserving per-cohort counts; statistics may return None."""
    records = list(records)
    estimate = statistic(records) if records else None
    cohorts = sorted({row["cohort"] for row in records})
    pools = [[row for row in records if row["cohort"] == name] for name in cohorts]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples if records else 0):
        chosen = []
        for pool in pools:
            chosen.extend(pool[index] for index in rng.integers(0, len(pool), len(pool)))
        value = statistic(chosen)
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    interval = np.percentile(draws, (2.5, 97.5)).tolist() if len(draws) >= samples // 2 else None
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci95": interval,
        "families": len(records),
        "valid_draws": len(draws),
    }


def mean_of(path: Sequence[str]) -> Callable[[Sequence[dict]], float | None]:
    """Family-weighted mean of a nested field, skipping families where it is None."""

    def statistic(rows: Sequence[dict]) -> float | None:
        values = [_get(row, path) for row in rows]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    return statistic


def paired_contrast(
    left: Sequence[str], right: Sequence[str]
) -> Callable[[Sequence[dict]], float | None]:
    """Mean within-family difference, using only families where both fields exist."""

    def statistic(rows: Sequence[dict]) -> float | None:
        values = [
            _get(row, left) - _get(row, right)
            for row in rows
            if _get(row, left) is not None and _get(row, right) is not None
        ]
        return float(np.mean(values)) if values else None

    return statistic


def ratio_slope(
    group: str, numerator: str, minus: str | None = None
) -> Callable[[Sequence[dict]], float | None]:
    """Pooled least-squares slope sum(inner)/sum(energy); optionally a slope difference."""

    def statistic(rows: Sequence[dict]) -> float | None:
        terms = [row["slope"][group] for row in rows]
        energy = sum(term["energy"] for term in terms)
        if energy <= 0:
            return None
        value = sum(term[numerator] for term in terms) / energy
        if minus is not None:
            value -= sum(term[minus] for term in terms) / energy
        return float(value)

    return statistic


def floor_ratio(rows: Sequence[dict], field: str = "boundary_meaningful") -> float | None:
    """Family-mean real-frame effect error divided by the median true effect size."""
    cells = [row[field] for row in rows if row.get(field) is not None]
    if not cells:
        return None
    effects = [value for cell in cells for value in cell["effects"]]
    median = float(np.median(effects))
    return None if median <= 0 else float(np.mean([cell["e_real"] for cell in cells]) / median)


def _get(row: dict, path: Sequence[str]):
    value: Any = row
    for key in path:
        if value is None:
            return None
        value = value[key]
    return value


def apply_criteria(summary: dict[str, dict]) -> dict[str, str]:
    """D021 decision rules on bootstrap summaries; the floor gates every claim."""
    ratio = summary["floor_ratio"]["estimate"]
    floor_ok = ratio is not None and ratio < FLOOR_RATIO_LIMIT

    def lower_above_zero(name: str) -> bool:
        interval = summary[name]["ci95"]
        return interval is not None and interval[0] > 0

    def upper_below(name: str, bound: float) -> bool:
        interval = summary[name]["ci95"]
        return interval is not None and interval[1] < bound

    if not floor_ok:
        status = "inconclusive_readout_floor"
        return {"floor": "failed", "physical_failure": status, "shrinkage": status, "h1": status}
    return {
        "floor": "passed",
        "physical_failure": "supported" if lower_above_zero("D_boundary") else "not_established",
        "shrinkage": "supported"
        if upper_below("slope_pred", 1.0) and upper_below("slope_pred_minus_real", 0.0)
        else "not_established",
        "h1": "supported" if lower_above_zero("h1_contrast") else "not_established",
    }


def summarize_phys(records: Sequence[dict], seed: int, samples: int = 2000) -> dict[str, Any]:
    """All PHYS-002 primary and secondary quantities with family-bootstrap intervals."""
    spec: dict[str, Callable[[Sequence[dict]], float | None]] = {
        "D_boundary": mean_of(("boundary_all", "D")),
        "D_boundary_meaningful": mean_of(("boundary_meaningful", "D")),
        "D_exactly_one": mean_of(("boundary", "exactly_one", "D")),
        "D_both": mean_of(("boundary", "both", "D")),
        "h1_contrast": paired_contrast(("boundary", "exactly_one", "D"), ("boundary", "both", "D")),
        "D_meaningful_all_pairs": mean_of(("meaningful_all", "D")),
        "e_real_boundary_meaningful": mean_of(("boundary_meaningful", "e_real")),
        "endpoint_err_pred": mean_of(("endpoint", "err_pred")),
        "endpoint_err_real": mean_of(("endpoint", "err_real")),
        "endpoint_D_moved": mean_of(("endpoint", "moved_D")),
        "slope_pred": ratio_slope("moved", "pred_inner"),
        "slope_real": ratio_slope("moved", "real_inner"),
        "slope_pred_minus_real": ratio_slope("moved", "pred_inner", "real_inner"),
        "slope_pred_contact": ratio_slope("moved_contact", "pred_inner"),
        "slope_pred_no_contact": ratio_slope("moved_no_contact", "pred_inner"),
        "regret_jepa": mean_of(("regret", "jepa")),
        "regret_real_frame": mean_of(("regret", "real_frame")),
        "regret_random": mean_of(("regret", "random")),
        "floor_ratio": floor_ratio,
    }
    result = {}
    for offset, (name, statistic) in enumerate(spec.items()):
        result[name] = bootstrap(records, statistic, seed + offset, samples)
    result["criteria"] = apply_criteria(result)
    return result
