"""ROBUST-001 (D034): kNN readout, off-manifold score, lineage and reproduction guards."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import jepa_action_effects.robust001 as robust
from jepa_action_effects.robust001_metrics import (
    KnnReadout,
    max_pairwise,
    offmanifold_threshold,
    select_k,
    spearman,
)


def _grid(n: int = 20):
    """Latents on a 2-D grid embedded in 4-D; labels are the grid coordinates."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    labels = np.column_stack([xs.ravel(), ys.ravel()])
    latents = np.column_stack([labels, np.zeros((len(labels), 2))])
    return latents, labels


def test_knn_recovers_bank_points_and_averages_neighbours():
    latents, labels = _grid()
    exact = KnnReadout(latents, labels, k=1)
    np.testing.assert_allclose(exact.predict(latents[:7]), labels[:7])
    np.testing.assert_allclose(exact.nearest_distance(latents[:7]), 0.0)
    # Halfway between two grid points along x: two neighbours average to the midpoint.
    query = np.array([[3.5, 4.0, 0.0, 0.0]])
    np.testing.assert_allclose(KnnReadout(latents, labels, k=2).predict(query), [[3.5, 4.0]])


def test_knn_clamps_beyond_the_bank_unlike_linear_extrapolation():
    # The documented failure mode: a query far outside the bank decodes to the edge.
    latents, labels = _grid(10)
    far = np.array([[40.0, 5.0, 0.0, 0.0]])
    prediction = KnnReadout(latents, labels, k=1).predict(far)
    assert prediction[0, 0] == pytest.approx(9.0)
    assert KnnReadout(latents, labels, k=1).nearest_distance(far)[0] == pytest.approx(31.0)


def test_select_k_prefers_averaging_under_label_noise_and_ties_to_smaller_k():
    rng = np.random.default_rng(0)
    latents, labels = _grid(30)
    noisy = labels + rng.normal(0, 2.0, labels.shape)
    queries = latents[::7] + rng.normal(0, 0.01, latents[::7].shape)
    result = select_k(latents, noisy, queries, labels[::7], grid=(1, 4, 16))
    assert result["k"] > 1
    assert result["validation_mean_error"][1] > result["validation_mean_error"][result["k"]]
    # Without noise and exact queries every k is not equal, but k = 1 is exact.
    exact = select_k(latents, labels, latents[:50], labels[:50], grid=(1, 2))
    assert exact["k"] == 1 and exact["validation_mean_error"][1] == 0.0


def test_offmanifold_threshold_and_spread_and_spearman():
    assert offmanifold_threshold(np.arange(101.0)) == pytest.approx(95.0)
    with pytest.raises(ValueError):
        offmanifold_threshold(np.array([]))
    points = np.array([[[0, 0], [3, 4], [0, 1]], [[1, 1], [1, 1], [1, 1]]], float)
    np.testing.assert_allclose(max_pairwise(points), [5.0, 0.0])
    assert spearman(np.arange(5.0), np.arange(5.0) ** 2) == pytest.approx(1.0)
    assert spearman(np.ones(5), np.arange(5.0)) is None


def _shards(root, split, count, value):
    root.mkdir(parents=True, exist_ok=True)
    latents, labels = _grid(6)
    full = np.column_stack([np.zeros((len(labels), 2)), labels, np.zeros(len(labels))])
    for index in range(count):
        np.savez(
            root / f"{split}-{index:03d}.npz",
            projected=np.pad(latents, ((0, 0), (0, 188))) + value,
            labels=full,
        )


def test_knn_selection_is_refused_after_a_shard_changes(tmp_path, monkeypatch):
    shards = tmp_path / "shards"
    _shards(shards, "train", 2, 0.0)
    _shards(shards, "validation", 1, 0.0)
    monkeypatch.setattr(robust, "SHARDS", shards)
    root = tmp_path / "robust"
    root.mkdir()
    selection = {
        "k": 1,
        "train_shards": robust._hashes(robust._shard_files("train")),
        "validation_shards": robust._hashes(robust._shard_files("validation")),
    }
    (root / "knn.json").write_text(json.dumps(selection))
    readout, _ = robust._knn_readout(root)
    assert readout.k == 1 and len(readout.bank) == 72
    _shards(shards, "train", 1, 1.0)  # rewrite train-000 with different content
    with pytest.raises(robust.GateFailure, match="Training shards changed"):
        robust._knn_readout(root)


def _cell(value, ci=(0.0, 1.0)):
    return {key: {"estimate": value, "ci95": list(ci)} for key in robust.GATE_KEYS}


def test_reproduction_passes_identical_cells_and_names_a_changed_one():
    reference = {
        "shift": {"reserved:challenge": {"cells": {"mlp|m0.60|constant": _cell(0.9)}}},
        "phys": {"reserved:challenge": {"cells": {"mlp": _cell(0.5)}}},
    }
    results = {name: {"groups": copy.deepcopy(groups)} for name, groups in reference.items()}
    # Our outputs may hold extra cells (kNN, on-manifold); only shared ones are checked.
    results["shift"]["groups"]["reserved:challenge"]["cells"]["knn|m0.60|constant"] = _cell(7.0)
    passed = robust.reproduction(results, reference)
    assert passed["passed"] and passed["checks"] == 2 * 4 * 3
    results["phys"]["groups"]["reserved:challenge"]["cells"]["mlp"]["alignment"]["ci95"][1] += 1e-6
    failed = robust.reproduction(results, reference)
    assert not failed["passed"]
    assert failed["failures"] == ["phys/reserved:challenge/mlp/alignment"]
    # An interval present on one side only is a failure, not a skip.
    results = {name: {"groups": copy.deepcopy(groups)} for name, groups in reference.items()}
    results["phys"]["groups"]["reserved:challenge"]["cells"]["mlp"]["b_rev"]["ci95"] = None
    assert not robust.reproduction(results, reference)["passed"]


def _contrasts(values):
    """SHIFT results whose dose contrasts take the given upper CI bound per instrument."""
    groups = {}
    for scene, dose in (("challenge", "m0.60-m0.30"), ("archive", "x2.0-x1.0")):
        contrasts = {}
        for name, high in values.items():
            for estimand in ("energy_ratio", "b_rev", "alignment"):
                for suffix in ("", f"|{robust.ON}"):
                    contrasts[f"{name}|{estimand}|{dose}{suffix}"] = {
                        "estimate": high - 0.1,
                        "ci95": [high - 0.2, high],
                    }
        groups[f"reserved:{scene}"] = {"contrasts": contrasts}
    return {"shift": {"groups": groups}}


def test_verdict_rules_require_every_instrument_and_discount_knn_only_trends():
    names = ["mlp-a", "ridge", robust.KNN]
    robust_all = robust.verdicts(_contrasts(dict.fromkeys(names, -0.05)), names)
    assert robust_all["instrument_robust_trend"] and robust_all["on_manifold"] == "trend_survives"
    knn_only = robust.verdicts(_contrasts({"mlp-a": 0.3, "ridge": 0.3, robust.KNN: -0.05}), names)
    assert not knn_only["instrument_robust_trend"] and not knn_only["trend_without_knn"]
    no_knn = robust.verdicts(_contrasts({"mlp-a": -0.05, "ridge": -0.05, robust.KNN: 0.3}), names)
    assert not no_knn["instrument_robust_trend"] and no_knn["trend_without_knn"]
    assert no_knn["on_manifold"] == "inconclusive"


def test_failed_reproduction_writes_no_report(tmp_path, monkeypatch):
    root = tmp_path / "robust"
    root.mkdir()
    inputs = {"knn": "k" * 64, "readouts": {"mlp": "m" * 64}}
    for name in ("shift", "phys"):
        groups = {"reserved:challenge": {"cells": {"mlp": _cell(0.5)}, "contrasts": {}}}
        (root / f"{name}.json").write_text(json.dumps({"groups": groups, "inputs": inputs}))
    (root / "knn.json").write_text("{}")
    reference = {
        name: {"reserved:challenge": {"cells": {"mlp": _cell(0.6)}}} for name in ("shift", "phys")
    }
    monkeypatch.setattr(robust, "_instruments", lambda r: ({"mlp": None}, None, inputs))
    monkeypatch.setattr(robust, "_calib_reference", lambda: reference)
    monkeypatch.setattr(robust, "DRAFT", tmp_path / "draft.md")
    monkeypatch.setattr(robust, "MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(robust.GateFailure, match="Reproduction failed"):
        robust.report(root)
    assert not (tmp_path / "draft.md").exists() and not (tmp_path / "manifest.json").exists()


def test_report_refuses_outputs_made_with_another_knn(tmp_path, monkeypatch):
    root = tmp_path / "robust"
    root.mkdir()
    for name in ("shift", "phys"):
        stale = {"knn": "0" * 64, "readouts": {"mlp": "m" * 64}}
        (root / f"{name}.json").write_text(json.dumps({"groups": {}, "inputs": stale}))
    (root / "knn.json").write_text("{}")
    current = {"knn": "1" * 64, "readouts": {"mlp": "m" * 64}}
    monkeypatch.setattr(robust, "_instruments", lambda r: ({"mlp": None}, None, current))
    with pytest.raises(robust.GateFailure, match="used another kNN"):
        robust.report(root)
