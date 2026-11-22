"""Pure, family-level projected-latent correspondence measurements."""

from __future__ import annotations

from itertools import combinations

import numpy as np

REGIMES = ("neither", "exactly_one", "both")


def cost_matrix(predicted: np.ndarray, real: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if predicted.shape != (6, 192) or real.shape != (6, 192):
        raise ValueError("Expected six projected CLS endpoints of width 192")
    if not np.isfinite(predicted).all() or not np.isfinite(real).all():
        raise ValueError("Nonfinite projected embedding")
    return np.mean((predicted[:, None] - real[None]) ** 2, axis=2)


def scores(cost: np.ndarray, start: np.ndarray, real: np.ndarray) -> dict[str, float]:
    cost = np.asarray(cost, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if cost.shape != (6, 6) or start.shape != (192,) or real.shape != (6, 192):
        raise ValueError("Latent score shapes changed")
    matched = float(np.trace(cost) / 6)
    wrong = float((cost.sum() - np.trace(cost)) / 30)
    persistence = float(np.mean((real - start) ** 2))
    return {
        "L": matched,
        "W": wrong,
        "A": wrong - matched,
        "P": persistence,
        "B": persistence - matched,
    }


def pair_rows(
    cost: np.ndarray,
    contacts: np.ndarray,
    block: np.ndarray,
    agent: np.ndarray,
    start_agent: np.ndarray,
    motion: np.ndarray,
    predicted: np.ndarray,
    real: np.ndarray,
    delta: float = 5.12,
) -> list[dict]:
    rows = []
    for i, j in combinations(range(6), 2):
        count = int(contacts[i]) + int(contacts[j])
        rows.append(
            {
                "i": i,
                "j": j,
                "regime": REGIMES[count],
                "effect": "meaningful"
                if np.linalg.norm(block[i] - block[j]) > delta
                else "negligible",
                "motion": [bool(motion[i]), bool(motion[j])],
                "A": float((cost[i, j] + cost[j, i] - cost[i, i] - cost[j, j]) / 2),
                "predicted_spread": float(np.mean((predicted[i] - predicted[j]) ** 2)),
                "real_spread": float(np.mean((real[i] - real[j]) ** 2)),
                "agent_displacement": float(
                    (
                        np.linalg.norm(agent[i] - start_agent)
                        + np.linalg.norm(agent[j] - start_agent)
                    )
                    / 2
                ),
            }
        )
    return rows


def endpoint_subset(
    cost: np.ndarray, start: np.ndarray, real: np.ndarray, pairs: list[dict]
) -> dict:
    ids = sorted({index for pair in pairs for index in (pair["i"], pair["j"])})
    if not ids:
        raise ValueError("No participating endpoints")
    matched = float(np.mean(np.diag(cost)[ids]))
    persistence = float(np.mean((real[ids] - start) ** 2))
    return {
        "families": 1,
        "endpoints": len(ids),
        "L": matched,
        "P": persistence,
        "B": persistence - matched,
    }


def summarize(records: list[dict], seed: int, budget_check=lambda: None) -> dict:
    if not records:
        return {"status": "incomplete", "families": 0}
    rng = np.random.default_rng(seed)
    groups: dict[str, dict[str, dict[str, float]]] = {}
    motion_groups = ("motion_neither", "motion_exactly_one", "motion_both")
    for record in records:
        cohort = record["cohort"]
        values = {name: record["scores"][name] for name in ("L", "W", "A", "P", "B")}
        groups[record["id"]] = {"overall": values, cohort: values}
        for regime in (*REGIMES, *motion_groups):
            pairs = [
                row
                for row in record["pairs"]
                if (
                    row["regime"] == regime
                    if regime in REGIMES
                    else "motion_" + REGIMES[sum(row["motion"])] == regime
                )
            ]
            if pairs:
                endpoints = record["endpoint_strata"][regime]
                groups[record["id"]][regime] = {
                    "A": float(np.mean([row["A"] for row in pairs])),
                    "L": endpoints["L"],
                    "P": endpoints["P"],
                    "B": endpoints["B"],
                }
        for effect in ("meaningful", "negligible"):
            pairs = [row for row in record["pairs"] if row["effect"] == effect]
            if pairs:
                groups[record["id"]][effect] = {"A": float(np.mean([row["A"] for row in pairs]))}
    result = {}
    for group in (
        "overall",
        "challenge",
        "ordinary",
        *REGIMES,
        *motion_groups,
        "meaningful",
        "negligible",
    ):
        selected = [record for record in records if group in groups[record["id"]]]
        if not selected:
            result[group] = {"status": "unassessable", "families": 0}
            continue
        entries = [groups[record["id"]][group] for record in selected]
        metrics = {}
        for name in entries[0]:
            values = np.asarray([entry[name] for entry in entries])
            draws = []
            if group == "overall":
                challenge = np.asarray(
                    [idx for idx, item in enumerate(selected) if item["cohort"] == "challenge"]
                )
                ordinary = np.asarray(
                    [idx for idx, item in enumerate(selected) if item["cohort"] == "ordinary"]
                )
                for _ in range(2000):
                    budget_check()
                    index = np.concatenate(
                        (rng.choice(challenge, len(challenge)), rng.choice(ordinary, len(ordinary)))
                    )
                    draws.append(float(values[index].mean()))
            else:
                for _ in range(2000):
                    budget_check()
                    draws.append(float(values[rng.integers(len(values), size=len(values))].mean()))
            low, high = np.percentile(draws, (2.5, 97.5))
            state = (
                "unassessable"
                if len(selected) < 10
                else ("positive" if low > 0 else "negative" if high < 0 else "inconclusive")
                if name in ("A", "B")
                else "descriptive"
            )
            metrics[name] = {
                "mean": float(values.mean()),
                "ci95": [float(low), float(high)],
                "state": state,
            }
        result[group] = {
            "status": "assessable" if len(selected) >= 10 else "unassessable",
            "families": len(selected),
            "metrics": metrics,
        }
    contrasts = {}
    for left, right in combinations(REGIMES, 2):
        ids = [
            record["id"]
            for record in records
            if left in groups[record["id"]] and right in groups[record["id"]]
        ]
        values = np.asarray([groups[key][left]["A"] - groups[key][right]["A"] for key in ids])
        if not len(values):
            contrasts[f"{left}_minus_{right}"] = {"status": "unassessable", "families": 0}
            continue
        draws = []
        for _ in range(2000):
            budget_check()
            draws.append(float(values[rng.integers(len(values), size=len(values))].mean()))
        contrasts[f"{left}_minus_{right}"] = {
            "status": "assessable" if len(values) >= 10 else "unassessable",
            "families": len(values),
            "mean": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(draws, (2.5, 97.5))],
        }
    return {
        "status": "complete",
        "families": len(records),
        "groups": result,
        "contrasts": contrasts,
    }
