"""Behavioral checks for LATENT-001 metrics, integrity, and budget."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from jepa_action_effects import latent_audit as audit
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.latent_metrics import (
    cost_matrix,
    endpoint_subset,
    pair_rows,
    scores,
    summarize,
)


def vectors():
    real = np.zeros((6, 192), dtype=np.float64)
    real[:, 0] = np.arange(6) * 10
    return real


def test_oracle_permutation_persistence_and_pair_equivalence():
    real = vectors()
    cost = cost_matrix(real, real)
    value = scores(cost, np.zeros(192), real)
    assert value["L"] == 0
    assert value["A"] > 0
    assert value["B"] > 0
    derangements = [p for p in permutations(range(6)) if all(i != p[i] for i in range(6))]
    assert len(derangements) == 265
    assert np.mean(
        [np.mean([cost[i, p[i]] for i in range(6)]) for p in derangements]
    ) == pytest.approx(value["W"])
    rows = pair_rows(
        cost,
        np.zeros(6, dtype=bool),
        np.zeros((6, 2)),
        np.zeros((6, 2)),
        np.zeros(2),
        np.zeros(6, dtype=bool),
        real,
        real,
    )
    assert np.mean([item["A"] for item in rows]) == pytest.approx(value["A"])
    swapped = cost_matrix(real[::-1], real)
    assert scores(swapped, np.zeros(192), real)["A"] < 0
    constant = cost_matrix(np.repeat(real[:1], 6, axis=0), real)
    assert scores(constant, np.zeros(192), real)["A"] == pytest.approx(0, abs=1e-12)
    assert (
        scores(
            cost_matrix(np.zeros_like(real), np.zeros_like(real)),
            np.zeros(192),
            np.zeros_like(real),
        )["A"]
        == 0
    )
    row_col = cost + np.arange(6)[:, None] + np.arange(6)[None, :]
    assert scores(row_col, np.zeros(192), real)["A"] == pytest.approx(value["A"])


def test_endpoint_dedup_and_family_equal_summary():
    real = vectors()
    cost = cost_matrix(real, real)
    pairs = [{"i": 0, "j": 1}, {"i": 0, "j": 2}]
    endpoint = endpoint_subset(cost, np.zeros(192), real, pairs)
    assert endpoint["endpoints"] == 3
    assert endpoint["P"] == pytest.approx(np.mean(real[:3] ** 2))
    records = [
        {
            "id": "a",
            "cohort": "challenge",
            "scores": {"L": 0, "W": 0, "A": 1, "P": 0, "B": 1},
            "pairs": [
                {"regime": "neither", "effect": "negligible", "motion": [False, False], "A": 1}
            ]
            * 15,
            "endpoint_strata": {
                "neither": {"L": 0, "P": 0, "B": 0},
                "motion_neither": {"L": 0, "P": 0, "B": 0},
            },
        },
        {
            "id": "b",
            "cohort": "ordinary",
            "scores": {"L": 0, "W": 0, "A": 3, "P": 0, "B": 3},
            "pairs": [
                {"regime": "neither", "effect": "negligible", "motion": [False, False], "A": 3}
            ],
            "endpoint_strata": {
                "neither": {"L": 0, "P": 0, "B": 0},
                "motion_neither": {"L": 0, "P": 0, "B": 0},
            },
        },
    ]
    report = summarize(records, 7)
    assert report["groups"]["overall"]["metrics"]["A"]["mean"] == 2
    assert report["groups"]["neither"]["metrics"]["A"]["mean"] == 2
    assert report["groups"]["neither"]["status"] == "unassessable"


def test_partition_exclusion_and_sealed_mismatch(tmp_path):
    pilot = {
        "splits": {f"challenge-{i}": "fit" for i in range(40)}
        | {f"ordinary-{i}": "fit" for i in range(20)}
    }
    fresh = {
        "splits": {f"challenge-{i}": "fit" for i in range(120)}
        | {f"ordinary-{i}": "fit" for i in range(60)}
    }
    fresh["splits"].update({f"challenge-v{i}": "validation" for i in range(40)})
    fresh["splits"].update({f"ordinary-v{i}": "validation" for i in range(20)})
    fresh["splits"].update({f"challenge-d{i}": "development" for i in range(40)})
    fresh["splits"].update({f"ordinary-d{i}": "development" for i in range(20)})
    ids, excluded = audit._selected(pilot, fresh)
    assert len(ids) == 60 and len(excluded) == 60 and not set(ids) & set(excluded)
    path = tmp_path / "sealed.json"
    audit.seal(path, {"value": 1})
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="mismatch"):
        audit.sealed(path)


def test_budget_resume_and_exhaustion(tmp_path, monkeypatch):
    audit.seal(tmp_path / "audit.json", {"audit_seconds": 0.0})
    times = iter((100.0, 101.0))
    monkeypatch.setattr(audit.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(audit.time, "time", lambda: 100.0)
    budget = audit.Budget(tmp_path, "same")
    budget.begin("evaluate")
    budget.finish("complete")
    assert budget.data["used_seconds"] == 1
    with pytest.raises(ValueError, match="signature"):
        audit.Budget(tmp_path, "changed")
    budget.data["used_seconds"] = 1740
    audit.write_json(budget.path, budget.data)
    with pytest.raises(TimeoutError):
        audit.Budget(tmp_path, "same").check()


def test_projected_cache_only_and_future_input_isolation(tmp_path, monkeypatch):
    family = {
        "history_frames": np.zeros((3, 224, 224, 3), dtype=np.uint8),
        "future_frames": np.ones((6, 2, 224, 224, 3), dtype=np.uint8),
        "warmup_controls": np.zeros((10, 2), dtype=np.float32),
        "candidate_controls": np.zeros((6, 10, 2), dtype=np.float32),
        "future_states": np.full((6, 2, 5), 999, dtype=np.float32),
    }
    fingerprint = {"test": "fixed"}
    previous = LatentCache(tmp_path / "previous")
    current = LatentCache(tmp_path / "current")
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    exact = {
        **fingerprint,
        "representation": "projected",
        "source": "new",
        "family_id": "challenge-1",
    }
    key = previous.key("real", {"frames": frames}, exact)
    previous.write("real", key, np.zeros((1, 15, 192), dtype=np.float32), exact)
    monkeypatch.setattr(
        audit, "encode_real", lambda *_: pytest.fail("cache hit should avoid encoding")
    )
    real = audit._load_real("new:challenge-1", family, previous, fingerprint, current, None)
    assert real.shape == (15, 192)
    assert not list((tmp_path / "current").rglob("*.npz"))

    def fake_predict(model, context, warmup, candidate, scaler):
        assert context.shape == (1, 3, 3, 224, 224)
        assert warmup.shape == (1, 10, 2) and candidate.shape == (1, 6, 10, 2)
        assert not np.any(candidate == 999)
        return np.zeros((1, 6, 2, 192), dtype=np.float32)

    class TensorLike:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    monkeypatch.setattr(
        audit, "preprocess_pixels", lambda value: __import__("torch").zeros((3, 3, 224, 224))
    )
    monkeypatch.setattr(audit, "predict_future", lambda *args: TensorLike(fake_predict(*args)))
    prediction = audit._predict("new:challenge-1", family, current, fingerprint, None, None)
    assert prediction.shape == (6, 2, 192)
    assert len(list((tmp_path / "current").rglob("*.npz"))) == 1


def test_frame_target_alignment_in_family_record():
    real = np.zeros((15, 192), dtype=np.float32)
    for candidate in range(6):
        real[3 + 2 * candidate + 1, 0] = candidate + 1
    prediction = np.zeros((6, 2, 192), dtype=np.float32)
    prediction[:, -1, 0] = np.arange(1, 7)
    family = {
        "history_states": np.zeros((3, 5)),
        "future_states": np.zeros((6, 2, 5)),
        "block_trajectories": np.zeros((6, 1, 2)),
        "contacts": np.zeros(6, dtype=bool),
        "future_frames": np.stack([np.full((2, 1, 1, 3), i, dtype=np.uint8) for i in range(6)]),
        "metadata": {"candidate_ids": [str(i) for i in range(6)]},
    }
    result = audit._record("new:ordinary-000", family, real, prediction)
    assert result["scores"]["L"] == 0
    assert all(result["cost"][i][i] == 0 for i in range(6))
