"""REPORT-002: saved-record recomputation compares whole summaries exactly."""

from __future__ import annotations

import math

from jepa_action_effects.train002_recompute import differences


def test_differences_finds_value_key_and_length_changes():
    left = {"a": {"estimate": 1.0, "ci": [0.5, 1.5]}, "v": "supported", "n": 3}
    assert differences(left, left) == []
    changed = {"a": {"estimate": 1.0 + 1e-12, "ci": [0.5]}, "v": "not", "extra": 1, "n": 3}
    found = differences(left, changed)
    assert "/a/ci: length 2 != 1" in found
    assert any(item.startswith("/a/estimate") for item in found)
    assert "/v: 'supported' != 'not'" in found
    assert "/extra: missing" in found
    assert differences(
        left, changed | {"a": {"estimate": 1.0 + 1e-12, "ci": [0.5, 1.5]}}, tolerance=1e-9
    )[:1] == ["/extra: missing"]


def test_differences_treats_matching_nans_as_equal_and_bools_strictly():
    assert differences({"x": math.nan}, {"x": math.nan}) == []
    assert differences({"x": True}, {"x": 1}) != []
