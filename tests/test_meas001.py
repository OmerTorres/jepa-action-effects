"""MEAS-001 runner guards on synthetic data (no simulator search, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_action_effects import meas001
from jepa_action_effects.next_common import seal, sealed
from jepa_action_effects.pose_search import SearchConfig
from jepa_action_effects.train001_budget import Budget, NullBudget

CHALLENGE = ["m0.15-constant", "m0.30-constant", "m0.45-constant", "m0.60-constant",
             "m0.30-varying", "m0.60-varying"]  # fmt: skip
ARCHIVE = ["x1.0-constant", "x1.0-varying", "x1.5-constant", "x1.5-varying", "x2.0-constant",
           "x2.0-varying"]  # fmt: skip


def _family_npz(
    path: Path, scene: str, seed: int, episode: int | None, state: float | None = None
) -> None:
    candidates = CHALLENGE if scene == "challenge" else ARCHIVE
    spec = {c: {"label": c.split("-")[0], "structure": c.split("-")[1]} for c in candidates}
    metadata = {
        "seed": seed,
        "episode": episode,
        "candidate_ids": candidates,
        "spec": spec,
        "reset_options": {
            "state": [
                float(seed if state is None else state),
                float(episode or 0),
                2.0,
                3.0,
                0.5,
                0.0,
                0.0,
            ]  # fmt: skip
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        warmup_controls=np.zeros((10, 2), np.float32),
        future_states=np.full((6, 2, 5), 7.0),  # a label that query inputs must never read
    )


@pytest.fixture()
def shift(tmp_path, monkeypatch):
    root = tmp_path / "shift002"
    families = {}
    for scene, offset in (("archive", 0), ("challenge", 1000)):
        for index in range(30):
            family_id = f"{scene}-{index:03d}"
            path = root / "families" / f"{family_id}.npz"
            seed = 0 if scene == "archive" else offset + index  # archive resets: seed 0
            _family_npz(path, scene, seed, index if scene == "archive" else None)
            latents = np.arange(15 * 192, dtype=np.float64).reshape(15, 192) + index
            (root / "develop").mkdir(parents=True, exist_ok=True)
            np.savez(
                root / "develop" / f"{family_id}.npz",
                real=latents,
                predicted=-latents[3:].reshape(6, 2, 192),
            )
            families[family_id] = {
                "path": f"families/{family_id}.npz",
                "sha256": meas001.sha256_file(path),
                "split": "development",
                "state": scene,
            }
    seal(root / "collection.json", {"families": families, "rejections": {}, "status": "complete"})
    monkeypatch.setattr(meas001, "SHIFT2", root)
    return root


def test_membership_is_deterministic_balanced_and_disjoint(shift, tmp_path) -> None:
    out = tmp_path / "meas"
    counts = meas001.select(out)
    assert counts["families"] == 48 and counts["real_queries"] == 216
    assert counts["calibration_queries"] == 108
    assert meas001.select(out) == counts  # re-sealing identical content is allowed
    membership = sealed(out / "membership.json")
    for scene in meas001.SCENES:
        for partition in ("calibration", "assessment"):
            members = [
                f
                for f, v in membership["families"].items()
                if v["scene"] == scene and v["partition"] == partition
            ]
            assert len(members) == 12
    assert len(membership["roles"]["benchmark"]) == 2
    assert len(membership["roles"]["controls"]) == 6
    assert len(membership["roles"]["predictions"]) == 16


def test_selection_rejects_aliases(shift, tmp_path) -> None:
    # Every challenge family gets the same seed: seeded resets must then be rejected.
    for index in range(30):
        name = f"challenge-{index:03d}"
        _family_npz(shift / "families" / f"{name}.npz", "challenge", 7, None, state=1000 + index)
    with pytest.raises(ValueError, match="seed"):
        meas001.select(tmp_path / "meas")


def test_archive_constant_seed_is_not_an_alias(shift, tmp_path) -> None:
    membership_counts = meas001.select(tmp_path / "meas")  # archive seeds are all 0
    assert membership_counts["families"] == 48


def test_disjointness_check() -> None:
    engineering = {"g": {"seed": 1, "reset_state": [1.0, 2.0], "episode": None}}
    assert meas001.check_disjoint(
        engineering, {"m": {"seed": 2, "reset_state": [3.0], "episode": 4}}
    )
    with pytest.raises(meas001.DisjointnessError):
        meas001.check_disjoint(
            engineering, {"m": {"seed": 1, "reset_state": [3.0], "episode": None}}
        )
    with pytest.raises(meas001.DisjointnessError):
        meas001.check_disjoint(
            {"g": {"seed": 9, "reset_state": [0.0], "episode": 4}},
            {"m": {"seed": 2, "reset_state": [3.0], "episode": 4}},
        )


def test_query_inputs_read_no_labels(shift, tmp_path, monkeypatch) -> None:
    meas001.select(tmp_path / "meas")
    membership = sealed(tmp_path / "meas" / "membership.json")
    monkeypatch.setattr(meas001, "load_family", lambda *a, **k: pytest.fail("labels loaded"))
    original = meas001.read_npz

    def guarded(path, keys):
        assert not any(k.startswith("future") for k in keys)
        return original(path, keys)

    monkeypatch.setattr(meas001, "read_npz", guarded)
    start = next(q for q in membership["queries"] if q["query"] == "start")
    endpoint = next(q for q in membership["queries"] if q["query"] == "m0.60")
    context, target = meas001.query_inputs(membership, start)
    offset = int(start["family"].split("-")[1])
    assert target[0] == 2 * 192 + offset and len(context.warmup_controls) == 10
    _, target = meas001.query_inputs(membership, endpoint)
    row = 3 + 2 * endpoint["candidate"] + 1
    assert target[0] == row * 192 + int(endpoint["family"].split("-")[1])
    _, predicted = meas001.query_inputs(membership, endpoint, kind="predicted")
    assert predicted[0] == -target[0]


def _frozen(root: Path) -> None:
    seal(
        root / "config.json",
        {
            "membership_sha256": meas001.sha256_file(root / "membership.json"),
            "sources": meas001.source_hashes(),
            "search": SearchConfig().to_dict(),
            "geometry": {"tolerance": 1.0, "agent_walls": False},
        },
    )


def test_lineage_and_freeze_refusals(shift, tmp_path) -> None:
    root = tmp_path / "meas"
    meas001.select(root)
    with pytest.raises(meas001.LineageError):
        meas001.require_frozen(root)
    digest = seal(root / "config-proposed.json", {"status": "not_ready", "sources": {}})
    with pytest.raises(meas001.LineageError):
        meas001.freeze(root, "0" * 64)
    with pytest.raises(meas001.LineageError):
        meas001.freeze(root, digest)  # not freeze_ready
    _frozen(root)
    assert meas001.require_frozen(root)["geometry"]["agent_walls"] is False
    membership = json.loads((root / "membership.json").read_text())
    membership["roles"]["benchmark"] = []
    seal(root / "membership.json", membership)
    with pytest.raises(meas001.LineageError):
        meas001.require_frozen(root)


def test_assessment_stays_closed_after_failed_calibration(shift, tmp_path) -> None:
    root = tmp_path / "meas"
    meas001.select(root)
    _frozen(root)
    seal(root / "calibration.json", {"tau": None, "status": "failed"})
    seal(root / "controls.json", {"records": [], "known_failure": {"passed": True}})
    with pytest.raises(meas001.LineageError):
        meas001.assess(root)
    with pytest.raises(meas001.LineageError):
        meas001.controls(root)


def test_predictions_are_skipped_unless_allowed(shift, tmp_path, monkeypatch) -> None:
    root = tmp_path / "meas"
    meas001.select(root)
    _frozen(root)
    seal(root / "scores.json", {"predictions_allowed": False})

    def sentinel(*args, **kwargs):
        raise AssertionError("prediction latents loaded after a failed gate")

    # MEAS-REVIEW-001 F2: a failed gate must never read saved predictor latents.
    monkeypatch.setattr(meas001, "query_inputs", sentinel)
    monkeypatch.setattr(meas001, "_load_model", sentinel)
    assert meas001.predictions(root)["status"] == "skipped"


def test_failed_integrity_writes_no_report(shift, tmp_path, monkeypatch) -> None:
    root = tmp_path / "meas"
    meas001.select(root)
    _frozen(root)
    seal(root / "scores.json", {"integrity_passed": False})
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(meas001, "MANIFESTS", manifests)
    with pytest.raises(SystemExit):
        meas001.report(root)
    assert not manifests.exists()


def test_seal_once_refuses_different_content(tmp_path) -> None:
    path = tmp_path / "record.json"
    meas001._seal_once(path, {"a": 1})
    assert meas001._seal_once(path, {"a": 1}) == {"a": 1}
    with pytest.raises(meas001.LineageError):
        meas001._seal_once(path, {"a": 2})


def test_budget_resume_charges_crashed_sessions_by_stage(tmp_path) -> None:
    now = [1000.0]
    clock = lambda: now[0]  # noqa: E731
    first = Budget(tmp_path, limit_seconds=4 * 3600, clock=clock, watchdog=False)
    first.start("engineer", "x")
    now[0] += 120.0
    first.heartbeat(force=True)
    first._lock_handle.close()  # the process dies without finish(); the OS releases the lock
    now[0] += 600.0
    second = Budget(tmp_path, limit_seconds=4 * 3600, clock=clock, watchdog=False)
    second.start("engineer", "x")
    now[0] += 30.0
    charged = meas001._stage_seconds(second, "engineer")
    assert charged == pytest.approx(120.0 + 60.0 + 30.0)  # last heartbeat + one interval
    assert meas001._stage_seconds(second, "calibrate") == 0.0
    second.finish("complete")


class _Bowl:
    def __init__(self):
        self.calls = 0

    def valid(self, poses):
        return np.ones(len(poses), dtype=bool)

    def evaluate(self, poses):
        self.calls += len(poses)
        return 1e-3 * np.sum((poses[:, :4] - [100.0, 300.0, 250.0, 260.0]) ** 2, axis=1)


def test_stored_inversions_are_reused_and_verified(tmp_path) -> None:
    objective = _Bowl()
    config = SearchConfig(local_starts=2, local_evaluations=320)
    path = tmp_path / "q"
    first = meas001._run_target(path, objective, config, 1, "q", NullBudget(), {"kind": "test"})
    calls = objective.calls
    second = meas001._run_target(path, objective, config, 1, "q", NullBudget(), {"kind": "test"})
    assert objective.calls == calls and first == second
    inversion, _ = meas001.load_inversion(path)
    assert inversion.trace.evaluations == first["evaluations"]
    np.savez_compressed(path.with_suffix(".npz"), poses=np.zeros((1, 5)))
    with pytest.raises(meas001.LineageError):
        meas001.load_inversion(path)


@pytest.mark.parametrize(
    "change",
    ["query", "config", "seed", "target", "inputs"],
)
def test_stored_inversions_with_a_different_binding_are_refused(tmp_path, change) -> None:
    # MEAS-REVIEW-001 F1: an intact file is not evidence for a different request.
    objective = _Bowl()
    objective.target = np.zeros(4)
    config = SearchConfig(local_starts=2, local_evaluations=320)
    path = tmp_path / "q"
    meas001._run_target(path, objective, config, 1, "q", NullBudget(), {}, {"family": "a"})
    calls = objective.calls
    request = {"query": "q", "config": config, "seed": 1, "inputs": {"family": "a"}}
    if change == "query":
        request["query"] = "other"
    elif change == "config":
        request["config"] = SearchConfig(local_starts=4, local_evaluations=160)
    elif change == "seed":
        request["seed"] = 2
    elif change == "target":
        objective.target = np.ones(4)
    else:
        request["inputs"] = {"family": "b"}
    with pytest.raises(meas001.LineageError, match="different binding"):
        meas001._run_target(
            path,
            objective,
            request["config"],
            request["seed"],
            request["query"],
            NullBudget(),
            {},
            request["inputs"],
        )
    assert objective.calls == calls  # nothing was recomputed or overwritten


def test_legacy_results_without_a_binding_are_refused(tmp_path) -> None:
    objective = _Bowl()
    config = SearchConfig(local_starts=2, local_evaluations=320)
    path = tmp_path / "legacy"
    meas001._run_target(path, objective, config, 1, "q", NullBudget(), {})
    payload = json.loads(path.with_suffix(".json").read_text())
    payload.pop("binding")
    path.with_suffix(".json").write_text(json.dumps(payload))
    with pytest.raises(meas001.LineageError):
        meas001._run_target(path, objective, config, 1, "q", NullBudget(), {})


def test_changed_family_files_are_refused_after_freeze(shift, tmp_path) -> None:
    root = tmp_path / "meas"
    meas001.select(root)
    _frozen(root)
    membership = sealed(root / "membership.json")
    assert meas001.require_frozen(root)
    family_id, family = next(iter(membership["families"].items()))
    query = next(q for q in membership["queries"] if q["family"] == family_id)
    with np.load(family["path"], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["warmup_controls"] = arrays["warmup_controls"] + 0.01  # silent content change
    np.savez(family["path"], **arrays)
    with pytest.raises(meas001.LineageError, match="Family file changed"):
        meas001.require_frozen(root)
    with pytest.raises(meas001.LineageError, match="Family file changed"):
        meas001.query_inputs(membership, query)
    with pytest.raises(meas001.LineageError, match="Family file changed"):
        meas001._labels(membership, query)


def test_known_failure_controls_never_accept() -> None:
    result = meas001.known_failure_controls()
    assert result["passed"]
    assert "accepted" not in result["outcomes"].values()


def test_engineering_diagnostics_from_saved_traces(tmp_path) -> None:
    root = tmp_path / "meas"
    path = root / "engineering" / "A" / "S" / "fam__start"
    config = SearchConfig(local_starts=2, local_evaluations=320)
    payload = meas001._run_target(path, _Bowl(), config, 1, "q", NullBudget(), {"kind": "test"})
    record = {
        "id": "fam:start",
        "true_block": [250.0, 260.0],
        "restart_distance": 0.1,
        "true_pose_residual": 0.0,
    }
    result = {"phases": {"A:S": {"records": [record], "tau": 0.01}}}
    (root / "engineering" / "result.json").write_text(json.dumps(result))
    diagnostics = meas001.engineering_diagnostics(root)["A:S"]
    assert diagnostics["targets"] == 1 and payload["evaluations"] > 0
    assert diagnostics["median_best_error"] < 1.0
    curve = diagnostics["residual_by_block_distance"]
    assert curve[0]["median_best_residual"] <= curve[-1]["median_best_residual"]
