"""Model-free fixtures for the SHIFT-002 dose-response metrics and decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from jepa_action_effects.shift001_metrics import family_record
from jepa_action_effects.shift002_metrics import (
    ARCHIVE_SCALES,
    CHALLENGE_LEVELS,
    CHALLENGE_VARYING,
    archive_label,
    challenge_label,
    ratio,
    summarize_dose,
)


def _candidates(state):
    if state == "challenge":
        keys = [(challenge_label(m), "constant", m) for m in CHALLENGE_LEVELS]
        keys += [(challenge_label(m), "varying", m) for m in CHALLENGE_VARYING]
    else:
        keys = [(archive_label(s), k, s) for s in ARCHIVE_SCALES for k in ("constant", "varying")]
    return keys


def _record(identifier, state, factor, rng, readout_gain=1.0):
    """factor(level) gives predicted/true; real frames decode true*readout_gain + noise."""
    keys = _candidates(state)
    ids = [f"{label}-{structure}" for label, structure, _ in keys]
    spec = {
        f"{label}-{structure}": {"structure": structure, "label": label}
        for label, structure, _ in keys
    }
    true = np.zeros((6, 2, 2))
    predicted = np.zeros_like(true)
    for index, (_, _, level) in enumerate(keys):
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        step1 = direction * (8 + 30 * level) * rng.uniform(0.8, 1.2)
        true[index] = [step1, 2 * step1]
        predicted[index] = true[index] * factor(level)
    real = true * readout_gain + rng.normal(scale=0.5, size=true.shape)
    return family_record(identifier, state, ids, spec, true, predicted, real, [True] * 6)


def _records(challenge_factor, archive_factor, count=40, readout_gain=1.0):
    rng = np.random.default_rng(5)
    return [
        _record(f"challenge-{i}", "challenge", challenge_factor, rng, readout_gain)
        for i in range(count)
    ] + [
        _record(f"archive-{i}", "archive", archive_factor, rng, readout_gain) for i in range(count)
    ]


def test_ratio_matches_the_level_factor() -> None:
    records = _records(lambda m: 1.3 - m, lambda s: 1.0)
    assert ratio(records, "challenge", "constant", {"m0.60"}) == pytest.approx(0.7, abs=0.02)
    assert ratio(records, "challenge", "constant", {"m0.15"}) == pytest.approx(1.15, abs=0.02)


def test_magnitude_driven_shrinkage_is_supported_and_structure_is_equivalent() -> None:
    summary = summarize_dose(
        _records(lambda m: 1.3 - m, lambda s: 1.3 - 0.3 * s), seed=1, samples=300
    )
    criteria = summary["criteria"]
    assert criteria["passed"]
    assert criteria["magnitude_challenge"] == "supported"
    assert criteria["magnitude_archive"] == "supported"
    assert criteria["structure_challenge"].startswith("no_effect_within")


def test_no_magnitude_effect_fails_the_high_push_check() -> None:
    criteria = summarize_dose(_records(lambda m: 1.0, lambda s: 1.0), seed=1, samples=300)[
        "criteria"
    ]
    assert not criteria["manipulation_checks"]["challenge_high_shrinks"]
    assert criteria["magnitude_challenge"] == "inconclusive_manipulation_check"


def test_biased_readout_fails_the_readout_gate() -> None:
    records = _records(lambda m: 1.3 - m, lambda s: 1.0, readout_gain=0.7)
    criteria = summarize_dose(records, seed=1, samples=300)["criteria"]
    assert not criteria["manipulation_checks"]["readout_unbiased"]
    assert not criteria["passed"]
