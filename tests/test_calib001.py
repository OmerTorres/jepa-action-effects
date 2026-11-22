"""CALIB-001 runner guards that need no sealed artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jepa_action_effects import calib001


def test_compare_requires_matching_presence_and_tolerance() -> None:
    assert calib001._compare(None, None)["passed"]
    assert not calib001._compare(None, 0.5)["passed"]
    assert not calib001._compare(0.5, None)["passed"]
    assert calib001._compare(0.5, 0.5 + 1e-12)["passed"]
    assert not calib001._compare(0.5, 0.5 + 1e-6)["passed"]


def test_stack_fills_missing_cells_with_zero_statistics() -> None:
    row = np.arange(len(calib001.STATISTICS), dtype=float)
    names, sums = calib001._stack([{"a": row}, {"a": row, "b": 2 * row}])
    assert names == ["a", "b"]
    assert sums.shape == (2, 2, len(calib001.STATISTICS))
    assert np.all(sums[0, 1] == 0.0)


def test_failed_reproduction_writes_no_manifest_or_tables(tmp_path, monkeypatch) -> None:
    root = tmp_path / "calib001"
    root.mkdir()
    for name in ("phys", "shift", "train"):
        (root / f"{name}.json").write_text(json.dumps({}))
    manifests, draft = tmp_path / "manifests", tmp_path / "CALIB-001_RESULT_DRAFT.md"
    monkeypatch.setattr(calib001, "MANIFESTS", manifests)
    monkeypatch.setattr(calib001, "DRAFT", draft)
    monkeypatch.setattr(
        calib001, "reproduction", lambda results: {"passed": False, "checks": {}, "count": 0}
    )
    with pytest.raises(SystemExit):
        calib001.report(root)
    assert (root / "reproduction-failed.json").exists()
    assert not (root / "report.json").exists()
    assert not manifests.exists()
    assert not draft.exists()
