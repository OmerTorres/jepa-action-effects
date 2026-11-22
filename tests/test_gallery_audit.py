"""Behavioral fixtures for GALLERY-001 geometry, metrics and provenance."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from jepa_action_effects import gallery_audit as audit
from jepa_action_effects.gallery_metrics import (
    family_metrics,
    group_summary,
    query_score,
    weighted_quantile,
)
from jepa_action_effects.gallery_render import (
    directions,
    geometry,
    load_rendered,
    render_candidate,
    save_rendered,
    set_block_pose,
)
from jepa_action_effects.latent_audit import FRESH, manifest, seal, sealed, write_json
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.pilot_collection import load_family
from jepa_action_effects.simulator_replay import InstrumentedPushT


def _vectors() -> np.ndarray:
    result = np.zeros((5, 192), dtype=np.float32)
    result[1:, 0] = [10, -10, 20, -20]
    return result


def test_nonidentical_queries_oracle_constant_and_ties():
    bank = _vectors()
    query = bank[0].copy()
    query[0] = 1
    oracle = query_score(query, bank, 1e-12)
    assert oracle["correct"] and oracle["ratio"] > 0.5
    assert oracle["d_true"] > 0  # No self-matching in the representation gate.
    constant = query_score(np.zeros(192), np.zeros((5, 192)), 1e-12)
    assert not constant["correct"] and constant["tie"] and constant["collapsed"]
    almost = query_score(bank[0], bank, 1e6)
    assert not almost["correct"] and almost["tie"]
    with pytest.raises(ValueError):
        query_score(np.zeros(191), bank, 1e-12)


def test_family_weighting_quantile_and_sparse_groups():
    easy = {"queries": [{"correct": True, "ratio": 1.0}] * 2}
    hard = {"queries": [{"correct": False, "ratio": -1.0}] * 2}
    one = family_metrics([easy] * 6)
    two = family_metrics([hard])
    result = group_summary(
        [{"cohort": "challenge", "metrics": one}, {"cohort": "ordinary", "metrics": two}],
        7,
        bootstrap=100,
    )
    assert result["accuracy"] == 0.5
    assert result["ratio_p05"] == -1.0
    assert group_summary([], 7)["status"] == "unassessable"
    assert weighted_quantile([1, 3], [9, 1], 0.5) == 1
    assert family_metrics([{"queries": []}]) is None
    with pytest.raises(TimeoutError):
        group_summary(
            [{"cohort": "challenge", "metrics": one}],
            7,
            bootstrap=200,
            check_budget=lambda: (_ for _ in ()).throw(TimeoutError()),
        )


def test_pair_regime_deduplicates_endpoints_and_marks_sparse():
    candidate = {"eligible": True, "queries": [{"correct": True, "ratio": 0.8}] * 2}
    rows = [
        {
            "cohort": "challenge",
            "contacts": [False, False, False, True, True, True],
            "motion": [False] * 6,
            "scales": {"negative_10.24": {"candidates": [candidate] * 6}},
        }
    ]
    result = audit._regime_rows(rows, "negative_10.24", 1, "contacts", lambda: None)
    assert result["neither"]["candidates"] == 3
    assert result["both"]["candidates"] == 3
    assert result["exactly_one"]["candidates"] == 6
    assert all(row["status"] == "unassessable" for row in result.values())


def test_partition_selection_deterministic_and_excludes_confirmation():
    tuning = [f"new:challenge-{index}" for index in range(40)] + [
        f"new:ordinary-{index}" for index in range(20)
    ]
    verification = tuning[:6] + tuning[40:46]
    selected = audit._choose(tuning, verification, 20260930)
    assert selected == audit._choose(tuning, verification, 20260930)
    assert len(selected) == 30 and len(set(selected)) == 30
    assert len([key for key in selected if "challenge" in key]) == 20
    assert set(verification) <= set(selected)
    with pytest.raises(ValueError):
        audit._choose(tuning, verification[:11] + ["new:confirmation-0"], 20260930)


def test_directions_and_cache_fingerprint_separation(tmp_path: Path):
    first = directions(3, "family", "query")
    assert np.array_equal(first, directions(3, "family", "query"))
    assert not np.array_equal(first, directions(3, "family", "negative"))
    cache = LatentCache(tmp_path)
    image = np.zeros((224, 224, 3), np.uint8)
    primary = cache.key("real", {"frames": image}, {"representation": "projected"})
    diagnostic = cache.key("real", {"frames": image}, {"representation": "raw"})
    assert primary != diagnostic
    assert "candidate_controls" not in inspect.signature(audit._encode).parameters


def test_rendered_artifact_mismatch_rejected(tmp_path: Path):
    image = np.zeros((224, 224, 3), np.uint8)
    path = tmp_path / "sample.npz"
    from jepa_action_effects.latent_cache import array_digest

    save_rendered(path, {"images_sha256": {"true": array_digest(image)}}, {"true": image})
    load_rendered(path)
    save_rendered(path, {"images_sha256": {"true": "wrong"}}, {"true": image})
    with pytest.raises(ValueError, match="hash mismatch"):
        load_rendered(path)
    seal(tmp_path / "config.json", {"x": 1})
    (tmp_path / "config.json").write_text('{"x":2}')
    with pytest.raises(ValueError, match="mismatch"):
        sealed(tmp_path / "config.json")


def test_real_replay_pose_order_geometry_and_restoration():
    if not (FRESH / "collection-manifest.json").exists():
        pytest.skip("ignored READOUT-001 collection artifacts are not present")
    source = manifest(FRESH)
    family = load_family(FRESH / source["family_files"]["challenge-041"]["path"])
    env = InstrumentedPushT()
    try:
        meta, images = render_candidate(
            env,
            family,
            1,
            "new:challenge-041",
            {"templates": 1, "queries": 2},
        )
        assert np.array_equal(images["true"], family["future_frames"][1, -1])
        assert len(meta["queries"]) == 2
        assert all(len(value["selected"]) <= 4 for value in meta["scales"].values())
        original = (tuple(env.block.position), float(env.block.angle))
        before = env.render().copy()
        set_block_pose(env, original[1], (1.0, 1.0))
        assert geometry(env)[0] is False
        set_block_pose(env, original[1], tuple(env.agent.position))
        assert geometry(env)[0] is False
        set_block_pose(env, original[1], original[0])
        assert np.array_equal(env.render(), before)
    finally:
        env.close()


def test_incomplete_report_and_budget_recovery(tmp_path: Path, monkeypatch):
    source = audit.audit
    monkeypatch.setattr(audit, "audit", lambda root: {"config_sha256": "same"})
    seal(tmp_path / "audit.json", {"audit_seconds": 0.0})
    seal(tmp_path / "config.json", {"ids": []})
    result = audit.report(tmp_path)
    assert result["status"] == "incomplete" and result["expected_candidates"] == 180
    ledger = audit.Budget(tmp_path, "same" + audit.sha256_file(tmp_path / "audit.json"))
    ledger.data["used_seconds"] = 1740
    write_json(ledger.path, ledger.data)
    with pytest.raises(TimeoutError):
        audit.Budget(tmp_path, ledger.data["signature"]).check()
    monkeypatch.setattr(audit, "audit", source)


def test_render_resume_validates_completed_candidates(tmp_path: Path, monkeypatch):
    from jepa_action_effects.latent_cache import array_digest

    identifier = "new:ordinary-0"
    payload = tmp_path / "family.npz"
    payload.write_bytes(b"fixed-family-artifact")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    family = {
        "future_frames": np.zeros((6, 2, 224, 224, 3), dtype=np.uint8),
        "candidate_controls": np.zeros((6, 10, 2), dtype=np.float32),
        "metadata": {"candidate_ids": [str(index) for index in range(6)]},
    }
    seal(tmp_path / "audit.json", {"audit_seconds": 0.0})
    seal(
        tmp_path / "config.json",
        {
            "ids": [identifier],
            "family_hashes": {identifier: audit.sha256_file(payload)},
            "example_ids": [],
        },
    )
    for index in range(6):
        save_rendered(
            audit._render_path(tmp_path, identifier, index),
            {
                "identifier": identifier,
                "candidate_index": index,
                "input_sha256": audit.sha256_file(payload),
                "audit_sha256": audit.sha256_file(tmp_path / "audit.json"),
                "images_sha256": {"true": array_digest(image)},
            },
            {"true": image},
        )

    class Ledger:
        def begin(self, _stage):
            pass

        def check(self):
            pass

        def finish(self, _status):
            pass

    class Environment:
        def close(self):
            pass

    monkeypatch.setattr(audit, "audit", lambda _root: {})
    monkeypatch.setattr(audit, "_ledger", lambda _root: Ledger())
    monkeypatch.setattr(audit, "InstrumentedPushT", Environment)
    monkeypatch.setattr(audit, "manifest", lambda _root: {})
    monkeypatch.setattr(audit, "_family_path", lambda *_: payload)
    monkeypatch.setattr(audit, "load_family", lambda _path: family)
    monkeypatch.setattr(
        audit, "render_candidate", lambda *_: pytest.fail("resume recomputed a completed candidate")
    )
    assert audit.render(tmp_path)["status"] == "complete"
    damaged = audit._render_path(tmp_path, identifier, 0)
    damaged.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Invalid rendered gallery"):
        audit.render(tmp_path)
