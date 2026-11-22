"""TRAIN-001 admission vs integrity, weights, identities and partitions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jepa_action_effects.train001_collect import (
    DataIntegrityError,
    _admit_common,
    archive_direction,
    assign_partitions,
    bin_weights,
    collect_challenge,
    identity_passes,
    raw_frame_mae,
    verify_integrity,
    window_norms,
)
from jepa_action_effects.train001_protocol import Protocol

EMPTY = {"seeds": [], "episodes": [], "history_sha256": []}


def test_bin_weights_normalize_and_reject_sparse_bins():
    norms = np.concatenate([np.full(3, 0.1), np.full(2, 0.3), np.full(4, 0.4), np.full(1, 0.6)])
    result = bin_weights(norms, (0.225, 0.375, 0.525), 1)
    assert result["counts"] == [3, 2, 4, 1]
    assert result["weights"] == pytest.approx([0.3, 0.2, 0.4, 0.1])
    # Edges belong to the upper bin.
    assert bin_weights([0.225, 0.375, 0.525, 0.0], (0.225, 0.375, 0.525), 1)["counts"] == [1] * 4
    with pytest.raises(DataIntegrityError, match="fewer than 2"):
        bin_weights(norms, (0.225, 0.375, 0.525), 2)


def test_window_norms_use_nonoverlapping_valid_windows():
    controls = np.zeros((25, 2))
    controls[:10] = [0.3, 0.4]  # norm 0.5
    controls[10:20] = [0.0, 0.2]
    controls[12] = [1.5, 0.0]  # out of bounds invalidates the second window
    norms, skipped = window_norms(controls)
    assert norms.tolist() == pytest.approx([0.5])
    assert skipped == 1


def test_archive_direction_rejects_near_zero_means():
    assert archive_direction(np.tile([0.02, 0.02], (10, 1)), 0.05) is None
    direction = archive_direction(np.tile([0.3, -0.4], (10, 1)), 0.05)
    assert direction == pytest.approx([0.6, -0.8])


def test_raw_frame_mae_uses_uint8_scale():
    left = np.full((4, 4, 3), 100, np.uint8)
    assert raw_frame_mae(left, left + 1) == pytest.approx(1.0)
    protocol = Protocol()
    identity = {"position": 0.1, "frame_raw_mae": raw_frame_mae(left, left + 1), "own_future": 0.2}
    assert identity_passes(identity, protocol)  # 1.0 raw == 1/255 normalized: passes
    identity["frame_raw_mae"] = raw_frame_mae(left, left + 2)
    assert not identity_passes(identity, protocol)
    with pytest.raises(TypeError):
        raw_frame_mae(left.astype(np.float32) / 255, left.astype(np.float32) / 255)


def _branch(completed=True, frame=0):
    return SimpleNamespace(
        completed=completed, history_frames=np.full((3, 2, 2, 3), frame, np.uint8)
    )


def test_admission_rejections_are_counted_not_raised(monkeypatch):
    import jepa_action_effects.train001_collect as collect

    protocol = Protocol()
    log: dict[str, int] = {}
    labels = ["m0.15", "m0.30", "m0.45", "m0.60"]
    incomplete = SimpleNamespace(branches={k: _branch(k != "m0.45") for k in labels})
    assert _admit_common(incomplete, protocol, log, "archive") is None
    mismatch = SimpleNamespace(branches={k: _branch(frame=i) for i, k in enumerate(labels)})
    assert _admit_common(mismatch, protocol, log, "archive") is None
    static = SimpleNamespace(branches={k: _branch() for k in labels})
    monkeypatch.setattr(collect, "moved_labels", lambda replay, delta: dict.fromkeys(labels, False))
    assert _admit_common(static, protocol, log, "archive") is None
    assert log == {
        "archive:incomplete_branch": 1,
        "archive:history_mismatch": 1,
        "archive:movement_criterion": 1,
    }


def test_attempt_cap_is_a_fatal_integrity_failure(monkeypatch):
    import jepa_action_effects.pilot_collection as pilot_collection

    monkeypatch.setattr(pilot_collection, "propose_challenge", lambda *a: (None, "unsafe"))
    protocol = Protocol(attempt_caps={"challenge": 5, "archive": 5, "ordinary": 5})
    log: dict[str, int] = {}
    from jepa_action_effects.train001_budget import NullBudget

    with pytest.raises(DataIntegrityError, match="attempt cap"):
        collect_challenge(
            None,
            "development",
            protocol,
            set(),
            {"recipe": set(), "history": set()},
            log,
            NullBudget(),
        )
    assert log["challenge:unsafe"] == 5


def _record(identifier, scene, seed=None, episode=None, recipe=None, history=None):
    return {
        "id": identifier,
        "scene": scene,
        "seed": seed,
        "episode": episode,
        "recipe_sha256": recipe or f"r-{identifier}",
        "history_sha256": history or f"h-{identifier}",
        "file_sha256": "x",
        "relative_path": f"{identifier}.npz",
    }


def _collection(families, windows=(), weights=None):
    result = {
        "families": {r["id"]: r for r in families},
        "windows": {r["id"]: r for r in windows},
    }
    if weights is not None:
        result["weights"] = {"episodes": weights}
    return result


def test_integrity_accepts_archive_families_sharing_simulator_seed_zero():
    families = [
        _record("archive-dev-000", "archive", seed=0, episode=10),
        _record("archive-dev-001", "archive", seed=0, episode=11),
        _record("challenge-dev-000", "challenge", seed=123),
    ]
    windows = [_record("ordinary-dev-000", "ordinary", episode=12)]
    summary = verify_integrity([_collection(families, windows, [13, 14])], EMPTY)
    assert summary == {"records": 4, "episodes": 5, "seeds": 1}


@pytest.mark.parametrize(
    ("second", "registry", "message"),
    [
        (
            _collection([_record("archive-conf-000", "archive", episode=10)]),
            EMPTY,
            "archive episode",
        ),
        (_collection([_record("challenge-conf-000", "challenge", seed=123)]), EMPTY, "reset seed"),
        (
            _collection([_record("c1", "challenge", seed=5, recipe="r-challenge-dev-000")]),
            EMPTY,
            "recipe",
        ),
        (
            _collection([_record("c2", "challenge", seed=6, history="h-challenge-dev-000")]),
            EMPTY,
            "content",
        ),
        (_collection([_record("challenge-dev-000", "challenge", seed=7)]), EMPTY, "record id"),
        (_collection([], weights=[10]), EMPTY, "archive episode"),
        (
            _collection([_record("c3", "challenge", seed=99)]),
            {**EMPTY, "seeds": [99]},
            "earlier reset",
        ),
        (
            _collection([_record("a3", "archive", episode=50)]),
            {**EMPTY, "episodes": [50]},
            "earlier archive",
        ),
        (
            _collection([_record("a4", "archive", episode=51, history="old")]),
            {**EMPTY, "history_sha256": ["old"]},
            "earlier family",
        ),
    ],
)
def test_integrity_rejects_overlaps_duplicates_and_prior_use(second, registry, message):
    first = _collection(
        [
            _record("archive-dev-000", "archive", seed=0, episode=10),
            _record("challenge-dev-000", "challenge", seed=123),
        ]
    )
    with pytest.raises(DataIntegrityError, match=message):
        verify_integrity([first, second], registry)


def test_integrity_detects_changed_bytes(tmp_path):
    path = tmp_path / "a.npz"
    path.write_bytes(b"one")
    record = _record("a", "archive", episode=1)
    record["relative_path"] = "a.npz"
    from jepa_action_effects.pilot_collection import sha256_file

    record["file_sha256"] = sha256_file(path)
    verify_integrity([_collection([record])], EMPTY, tmp_path)
    path.write_bytes(b"two")
    with pytest.raises(DataIntegrityError, match="bytes changed"):
        verify_integrity([_collection([record])], EMPTY, tmp_path)


def test_partitions_split_each_scene_and_mark_pilot_subsets():
    protocol = Protocol(
        train_per_scene=5,
        validation_per_scene=2,
        pilot_train_per_scene=3,
        pilot_validation_per_scene=1,
        overfit_families_per_scene=2,
    )
    records = [{"scene": s, "id": f"{s}{i}"} for s in ("challenge", "archive") for i in range(7)]
    assign_partitions(records, "development", protocol, np.random.default_rng(0))
    for scene in ("challenge", "archive"):
        rows = [r for r in records if r["scene"] == scene]
        assert sum(r["partition"] == "train" for r in rows) == 5
        assert sum(r["partition"] == "validation" for r in rows) == 2
        assert sum(r.get("pilot", False) and r["partition"] == "train" for r in rows) == 3
        assert sum(r.get("overfit", False) for r in rows) == 2
        assert sum(r.get("pilot", False) and r["partition"] == "validation" for r in rows) == 1
