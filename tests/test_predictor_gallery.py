"""Behavioral controls for the sealed rendered-pose comparison."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects import predictor_gallery as runner
from jepa_action_effects.predictor_gallery import _validate_ids
from jepa_action_effects.predictor_gallery_metrics import (
    FIELDS,
    aggregate,
    candidate_scores,
    derangement_expectation,
    endpoint_metrics,
    family_mean,
    latent_summary,
    participants,
)


def _bank(position: float) -> np.ndarray:
    values = np.zeros((5, 192), dtype=np.float32)
    values[:, 0] = position + np.array([0, -3, -2, 2, 3])
    return values


def test_oracle_constant_and_swapped_sequences() -> None:
    predictions = np.zeros((6, 192), dtype=np.float32)
    predictions[:, 0] = np.arange(6) * 10
    bank = _bank(20)
    correct = endpoint_metrics(candidate_scores(predictions, predictions[0], bank, 1e-12), 2)
    assert correct["matched_accuracy"] == 1
    assert correct["action_advantage"] > 0
    reversed_scores = endpoint_metrics(
        candidate_scores(predictions[::-1].copy(), predictions[0], bank, 1e-12), 2
    )
    assert reversed_scores["matched_accuracy"] == 0
    constant = np.zeros_like(predictions)
    constant_scores = endpoint_metrics(candidate_scores(constant, constant[0], bank, 1e-12), 2)
    assert constant_scores["action_advantage"] == 0
    assert constant_scores["persistence_advantage"] == 0


def test_ties_and_identical_futures_do_not_create_action_advantage() -> None:
    bank = np.ones((5, 192), dtype=np.float32)
    predictions = np.ones((6, 192), dtype=np.float32)
    score = endpoint_metrics(candidate_scores(predictions, predictions[0], bank, 1e-12), 0)
    assert score["matched_accuracy"] == 0
    assert score["action_advantage"] == 0


def test_all_265_derangements_equal_off_diagonal_mean() -> None:
    matrix = np.random.default_rng(9).normal(size=(6, 6))
    expected = (matrix.sum() - np.trace(matrix)) / 30
    assert derangement_expectation(matrix) == pytest.approx(expected)


def test_endpoint_dedup_and_sparse_regimes() -> None:
    labels = [True, True, False, False, False, False]
    assert participants(labels, "both") == {0, 1}
    assert participants(labels, "neither") == {2, 3, 4, 5}
    assert participants([True] * 6, "neither") == set()
    predictions = np.zeros((6, 192), dtype=np.float32)
    rows = [
        {"eligible": True, **candidate_scores(predictions, predictions[0], _bank(i), 1e-12)}
        for i in range(6)
    ]
    assert family_mean(rows, participants(labels, "both"))["count"] == 2
    assert family_mean(rows, set()) is None


def test_family_weighting_and_fixed_cohort_bootstrap() -> None:
    rows = [
        {"cohort": "challenge", "summary": {"count": 6, **{name: 1.0 for name in FIELDS}}},
        {"cohort": "ordinary", "summary": {"count": 1, **{name: 3.0 for name in FIELDS}}},
    ]
    pooled = aggregate(rows, 7)["pooled"]
    assert pooled["metrics"]["matched_margin"]["mean"] == 2.0
    assert pooled["metrics"]["matched_margin"]["ci95"] == [2.0, 2.0]
    assert pooled["endpoints"] == 7
    latent = [
        {"cohort": "challenge", "latent": {key: 1.0 for key in ("L", "W", "A", "P", "B")}},
        {"cohort": "ordinary", "latent": {key: 3.0 for key in ("L", "W", "A", "P", "B")}},
    ]
    assert latent_summary(latent, 7)["pooled"]["A"]["mean"] == 2.0


def test_partition_counts_reject_overlap_or_bad_cohort() -> None:
    valid = [f"new:challenge-{i}" for i in range(40)] + [f"new:ordinary-{i}" for i in range(20)]
    _validate_ids(valid, 40, 20)
    with pytest.raises(ValueError):
        _validate_ids(valid[:-1] + [valid[0]], 40, 20)
    with pytest.raises(ValueError):
        _validate_ids(valid, 20, 40)


def test_audit_excludes_confirmation_payloads_and_rejects_changed_source(
    tmp_path, monkeypatch
) -> None:
    if not (runner.PILOT / "collection-manifest.json").exists():
        pytest.skip("ignored PILOT-001 collection artifacts are not present")
    monkeypatch.setattr(
        runner,
        "load_family",
        lambda *_: pytest.fail("audit opened a family payload"),
    )
    sealed = runner.audit(tmp_path)
    assert len(sealed["config"]["confirmation_ids"]) == 60
    assert not (tmp_path / "confirmation-opened.json").exists()
    monkeypatch.setattr(runner, "_sources", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source or configuration mismatch"):
        runner.audit(tmp_path)


def test_confirmation_requires_seal_before_opening(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.confirm(tmp_path)
    assert not (tmp_path / "confirmation-opened.json").exists()
