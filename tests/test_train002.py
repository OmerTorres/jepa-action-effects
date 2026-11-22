"""TRAIN-002 (D035): estimands, verdict rules, window tensors and seal guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

import jepa_action_effects.train002 as train002
from jepa_action_effects.latent_audit import seal
from jepa_action_effects.model_inference import ActionScaler
from jepa_action_effects.train001 import GateFailure
from jepa_action_effects.train001_protocol import READOUTS
from jepa_action_effects.train002_metrics import (
    arm_average,
    arm_of,
    contrasts_at,
    generic_verdict,
    primary_verdict,
    summarize,
    transfer_fraction,
)


def test_arm_mapping_and_seed_average():
    assert arm_of("frozen") == "frozen"
    assert arm_of("archive_only-20261111") == "archive_only"
    assert arm_of("archive_weighted-20261112") == "archive_weighted"
    with pytest.raises(ValueError):
        arm_of("unknown-1")
    averaged = arm_average({"balanced-1": np.array([1.0, 3.0]), "balanced-2": np.array([3.0, 5.0])})
    np.testing.assert_allclose(averaged["balanced"], [2.0, 4.0])


def test_transfer_fraction_and_contrasts_on_known_errors():
    scenes = np.array([0] * 20 + [1] * 20)
    frozen, balanced, omitted = np.full(40, 20.0), np.full(40, 10.0), np.full(40, 15.0)
    result = transfer_fraction(frozen, omitted, balanced, scenes, 200, 1)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci"] == pytest.approx([0.5, 0.5])
    errors = {
        a: np.tile(v[:, None], (1, 4))
        for a, v in (("frozen", frozen), ("balanced", balanced), ("dose_omitted", omitted))
    }
    out = contrasts_at(errors, 3)
    np.testing.assert_allclose(out["dose_omitted_minus_balanced"], 5.0)
    np.testing.assert_allclose(out["frozen_minus_dose_omitted"], 5.0)
    assert "frozen_minus_archive_only" not in out  # arm absent


def test_verdict_rules_respect_the_gate():
    assert primary_verdict([0.5, 2.0], True) == "held_out_dose_deficit_supported"
    assert primary_verdict([-2.0, -0.5], True) == "omission_better_supported"
    assert primary_verdict([-0.5, 2.0], True) == "held_out_dose_deficit_not_established"
    assert primary_verdict([0.5, 2.0], False) == "not_assessable"
    assert generic_verdict([0.1, 3.0], True) == "generic_finetuning_gain_supported"
    assert generic_verdict([-0.1, 3.0], True) == "generic_finetuning_gain_not_established"
    assert generic_verdict([0.1, 3.0], False) == "not_assessable"


def _records(offsets: dict[str, float], families: int = 30, windows: int = 12, seed: int = 0):
    """Synthetic aligned records: each model's 0.60 endpoint is off by a known amount."""
    rng = np.random.default_rng(seed)
    real, models = [], {m: [] for m in offsets}
    for index in range(families):
        scene = "challenge" if index < families // 2 else "archive"
        start = rng.uniform(150, 350, 2)
        direction = np.array([1.0, 0.0])
        end = np.stack([start + direction * d for d in (2.0, 12.0, 24.0, 36.0)])
        decoded = {r: {"start": start.tolist(), "real_end": end.tolist()} for r in READOUTS}
        unit = f"f{index:03d}"
        real.append(
            {
                "unit": unit,
                "scene": scene,
                "true_start": start.tolist(),
                "true_end": end.tolist(),
                "decoded": decoded,
            }
        )
        for model, offset in offsets.items():
            pred = end.copy()
            pred[3] = start + direction * 36.0 * (1 - offset / 36.0)  # shortened by `offset`
            pred[:3] -= direction * 0.0
            models[model].append(
                {
                    "unit": unit,
                    "latent_mse": np.full((4, 2), 0.1 + offset / 100).tolist(),
                    "decoded": {
                        r: {"pred_end": pred.tolist(), "perm_end": pred.tolist()} for r in READOUTS
                    },
                }
            )
    window_real = [
        {
            "unit": f"w{i}",
            "true_end": [[200.0, 200.0]],
            "decoded": {r: {"real_end": [[201.0, 200.0]]} for r in READOUTS},
        }
        for i in range(windows)
    ]
    window_models = {
        m: [
            {
                "unit": f"w{i}",
                "decoded": {r: {"pred_end": [[200.0 + o / 10, 200.0]]} for r in READOUTS},
            }
            for i in range(windows)
        ]
        for m, o in offsets.items()
    }
    return real, models, window_real, window_models


GATES = {"cells": {"challenge:m0.60": {"passed": True}, "archive:m0.60": {"passed": True}}}


def test_summary_recovers_a_held_out_dose_deficit_and_generic_gain():
    offsets = {
        "frozen": 14.0,
        "archive_weighted-1": 4.0,
        "balanced-1": 3.0,
        "dose_omitted-1": 9.0,
        "archive_only-1": 13.0,
    }
    summary = summarize(*_records(offsets), READOUTS, GATES, 200, 7, 8, 5.12, 1.28)
    primary = summary["readouts"][READOUTS[0]]
    assert primary["bootstrap"]["dose_omitted_minus_balanced"]["estimate"] == pytest.approx(6.0)
    assert summary["primary_verdict"] == "held_out_dose_deficit_supported"
    assert summary["generic_verdict"] == "generic_finetuning_gain_supported"
    assert primary["transfer_fraction"]["estimate"] == pytest.approx(5.0 / 11.0)
    assert primary["per_seed_dose_omitted_minus_balanced"] == {"dose_omitted-1": pytest.approx(6.0)}
    ratio = primary["slope_ratios"]["dose_omitted"]["challenge:m0.60"]["ratio"]
    assert ratio == pytest.approx(27.0 / 36.0)
    assert summary["readout_agreement"]["dose_omitted_minus_balanced"]
    failed = {"cells": {"challenge:m0.60": {"passed": True}, "archive:m0.60": {"passed": False}}}
    blocked = summarize(*_records(offsets), READOUTS, failed, 50, 7, 8, 5.12, 1.28)
    assert blocked["primary_verdict"] == "not_assessable"


def test_window_tensors_are_single_branch_archive_families():
    scaler = ActionScaler(mean=(0.0, 0.0), scale=(0.2, 0.2))
    records = [
        {
            "id": f"w{i}",
            "z": np.random.default_rng(i).normal(size=(5, 192)).astype(np.float32),
            "controls": np.zeros((20, 2), np.float32),
        }
        for i in range(3)
    ]
    data = train002.window_tensors(records, scaler)
    assert tuple(data.history.shape) == (3, 3, 192)
    assert tuple(data.targets.shape) == (3, 1, 2, 192)
    assert tuple(data.actions.shape) == (3, 1, 4, 10)
    assert set(data.scenes.tolist()) == {1}
    np.testing.assert_allclose(data.targets[0, 0].numpy(), records[0]["z"][3:5])


def test_optimization_gate_requires_a_falling_loss():
    falling = [[u, 1.0 / u, 0.0] for u in range(10, 2001, 10)]
    assert train002.optimization_gate(falling)["passed"]
    flat = [[u, 1.0, 0.0] for u in range(10, 2001, 10)]
    assert not train002.optimization_gate(flat)["passed"]


def test_reused_checkpoints_must_match_the_train001_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(train002, "TRAIN001", tmp_path)
    registry = {}
    for arm in train002.REUSED_ARMS:
        path = tmp_path / f"{arm}.pt"
        path.write_bytes(arm.encode())
        from jepa_action_effects.pilot_collection import sha256_file

        registry[f"{arm}-1"] = {
            "checkpoint": path.name,
            "checkpoint_sha256": sha256_file(path),
            "run_sha256": "r",
        }
    t1 = {"registry": registry}
    assert len(train002.reused_runs(t1, (1,))) == 2
    (tmp_path / "balanced.pt").write_bytes(b"tampered")
    with pytest.raises(GateFailure, match="balanced-1 checkpoint changed"):
        train002.reused_runs(t1, (1,))


def test_verify_seal_refuses_changed_inputs_sources_and_checkpoints(tmp_path, monkeypatch):
    from jepa_action_effects.pilot_collection import sha256_file

    root = tmp_path / "train002"
    for relative in train002.SEALED_FILES.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": relative}))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    sources = {"train002": "a" * 64}
    monkeypatch.setattr(train002, "_sources", lambda: dict(sources))
    payload = {
        "sources": sources,
        "file_sha256": {k: sha256_file(root / v) for k, v in train002.SEALED_FILES.items()},
        "registry": {
            "m-1": {
                "kind": "trained",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        },
    }
    seal(root / "seal.json", payload)
    p = train002.Protocol(root=str(root))
    assert train002.verify_seal(p)["registry"]
    (root / "gate.json").write_text("{}")
    with pytest.raises(GateFailure, match="gate.json changed"):
        train002.verify_seal(p)
    (root / "gate.json").write_text(json.dumps({"name": "gate.json"}))
    checkpoint.write_bytes(b"other")
    with pytest.raises(GateFailure, match="m-1"):
        train002.verify_seal(p)
    checkpoint.write_bytes(b"weights")
    sources["train002"] = "b" * 64
    with pytest.raises(GateFailure, match="Source changed"):
        train002.verify_seal(p)
