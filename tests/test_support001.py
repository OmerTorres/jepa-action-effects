"""SUPPORT-001 (D036): block-frame geometry, support map, account recovery and guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

import jepa_action_effects.support001 as support
from jepa_action_effects.latent_audit import seal
from jepa_action_effects.support001_metrics import (
    ACCOUNTS,
    SupportMap,
    block_frame,
    decide,
    fit,
    pairwise,
    predict,
    rotate,
    squared_errors,
)
from jepa_action_effects.train001_metrics import paired_bootstrap


def test_block_frame_is_rotation_invariant():
    agent, previous = np.array([[10.0, 0.0]]), np.array([[8.0, 0.0]])
    block, command = np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]])
    base = block_frame(agent, previous, block, np.array([0.0]), command)
    angle = np.array([np.pi / 3])
    turned = block_frame(
        rotate(agent, angle), rotate(previous, angle), block, angle, rotate(command, angle)
    )
    np.testing.assert_allclose(turned["s"], base["s"], atol=1e-12)
    np.testing.assert_allclose(turned["a"], base["a"], atol=1e-12)


def test_support_map_counts_command_support_and_typical_displacement():
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.normal(size=(n, 4))
    a = np.column_stack([rng.uniform(0, 0.5, n), np.zeros(n)])  # commands only up to 0.5
    d = np.tile([3.0, 0.0], (n, 1))
    m = SupportMap(s, a, d, k=200)
    query = m.query(
        np.zeros((2, 4)), np.array([[0.25, 0.0], [0.9, 0.0]]), np.array([0.0, np.pi / 2])
    )
    assert query["count"][0] > 20 and query["count"][1] == 0
    assert query["log_support"][1] < query["log_support"][0]
    np.testing.assert_allclose(query["m"][0], [3.0, 0.0])
    np.testing.assert_allclose(query["m"][1], [0.0, 3.0], atol=1e-12)  # rotated to world
    assert query["magnitude_quantile"][1] == 1.0


def _synthetic(account: str, params, n: int = 600, seed: int = 1):
    rng = np.random.default_rng(seed)
    magnitude = rng.uniform(0.15, 1.0, n)
    t = rng.normal(size=(n, 2)) * (30 * magnitude)[:, None]
    m = rng.normal(size=(n, 2)) * 10
    log_support = np.log(np.exp(-6 * magnitude) + 1e-3)
    p = predict(account, params, t, m, log_support, magnitude) + rng.normal(0, 0.5, (n, 2))
    return p, t, m, log_support, magnitude


@pytest.mark.parametrize(
    ("account", "params"),
    [("A", [1.2, -0.6]), ("B", [4.0, 0.8]), ("C", [0.55])],
)
def test_each_account_is_recovered_and_selected_on_its_own_data(account, params):
    data = _synthetic(account, params)
    fits = {name: fit(name, *data) for name in ACCOUNTS}
    np.testing.assert_allclose(fits[account]["params"], params, rtol=0.05, atol=0.05)
    fresh = _synthetic(account, params, seed=2)
    errors = squared_errors(fits, *fresh)
    scenes = np.repeat([0, 1], len(fresh[0]) // 2)
    boot = paired_bootstrap(pairwise(errors), scenes, 300, 3)
    assert decide(boot)["verdict"] == f"account_{account}_established"


def test_decide_reports_no_account_when_intervals_overlap():
    boot = {f"{x}_minus_{y}": {"ci": [-1.0, 1.0]} for x in ACCOUNTS for y in ACCOUNTS if x != y}
    assert decide(boot) == {"verdict": "no_account_established", "winners": []}


def test_archive_windows_stay_inside_episodes_and_average_ten_controls():
    lengths = np.array([40, 31])
    offsets = np.array([0, 40])
    total = lengths.sum()
    states = np.zeros((total, 7))
    states[:, 2] = np.arange(total)  # block x advances one unit per step
    actions = np.tile([0.2, 0.0], (total, 1))
    windows = support.archive_windows(states, actions, offsets, lengths)
    index = windows["index"]
    episode = np.searchsorted(offsets, index, side="right") - 1
    assert np.all(index + 10 < offsets[episode] + lengths[episode])
    assert np.all(index - 10 >= offsets[episode])
    np.testing.assert_allclose(windows["a"], np.tile([0.2, 0.0], (len(index), 1)))
    np.testing.assert_allclose(windows["d"][:, 0], 10.0)


def test_family_units_use_block_frame_and_decoded_start():
    family = {
        "history_states": np.array([[0, 0, 5, 5, 0], [1, 0, 5, 5, 0], [2, 0, 5, 5, 0]], float),
        "future_states": np.tile(np.array([[[0, 0, 6, 5, 0], [0, 0, 7, 5, 0]]], float), (2, 1, 1)),
        "candidate_controls": np.stack(
            [np.tile([0.3, 0.0], (10, 1)), np.tile([0.6, 0.0], (10, 1))]
        ),
    }
    real = {"start": [5.5, 5.0], "real_end": [[7.5, 5.0], [7.5, 5.0]]}
    units = support.family_units(
        family, np.array([[6.5, 5.0], [7.0, 5.0]]), real, ["m0.30", "m0.60"]
    )
    np.testing.assert_allclose(units["t"], [[2.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(units["p"], [[1.0, 0.0], [1.5, 0.0]])
    np.testing.assert_allclose(units["magnitude"], [0.3, 0.6])
    np.testing.assert_allclose(units["s"][0], [-3.0, -5.0, 1.0, 0.0])


def test_verify_seal_refuses_changes(tmp_path, monkeypatch):
    from jepa_action_effects.pilot_collection import sha256_file

    for name in support.SEALED:
        (tmp_path / name).write_text(json.dumps({"name": name}))
    (tmp_path / "map.npz").write_bytes(b"map")
    monkeypatch.setattr(support, "_sources", lambda: {"support001": "a"})
    seal(
        tmp_path / "seal.json",
        {
            "sources": {"support001": "a"},
            "file_sha256": {n: sha256_file(tmp_path / n) for n in support.SEALED},
            "map_npz_sha256": sha256_file(tmp_path / "map.npz"),
        },
    )
    assert support.verify_seal(tmp_path)
    (tmp_path / "fit.json").write_text("{}")
    with pytest.raises(support.GateFailure, match="fit.json changed"):
        support.verify_seal(tmp_path)
