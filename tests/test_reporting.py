"""REPORT-001 reporting: registry rendering, missing-field failure and ledger output."""

from __future__ import annotations

import json

import pytest

from jepa_action_effects import reporting
from jepa_action_effects.reporting import MissingField, Number, claims_md, macros_tex, numbers


def _manifest(tmp_path):
    payload = {"a": {"b": {"estimate": -0.5612, "ci95": [-0.7, -0.4]}}, "n": 1888}
    (tmp_path / "m.json").write_text(json.dumps(payload))
    return tmp_path


def test_render_values_and_intervals(tmp_path):
    root = _manifest(tmp_path)
    registry = (
        Number("slope", "m.json", ("a", "b"), 3, "ci", "claim"),
        Number("count", "m.json", ("n",), 0, "value", "count"),
    )
    values = numbers(registry, root)
    assert values == {"slope": "-0.561\\,[-0.700,\\,-0.400]", "count": "1888"}
    tex = macros_tex(values)
    assert "\\newcommand{\\slope}{\\ensuremath{-0.561\\,[-0.700,\\,-0.400]}}" in tex
    assert "\\newcommand{\\slopePt}{\\ensuremath{-0.561}}" in tex
    assert "countPt" not in tex
    ledger = claims_md(values, registry, root)
    assert "`m.json` → `a/b`" in ledger and "-0.561 [-0.700, -0.400]" in ledger


def test_missing_fields_and_bad_macros_fail_loudly(tmp_path):
    root = _manifest(tmp_path)
    with pytest.raises(MissingField, match="a/missing"):
        numbers((Number("x", "m.json", ("a", "missing")),), root)
    with pytest.raises(ValueError, match="letters only"):
        numbers((Number("x1", "m.json", ("n",)),), root)
    with pytest.raises(ValueError, match="Duplicate"):
        numbers((Number("x", "m.json", ("n",)), Number("x", "m.json", ("n",))), root)


def test_registry_macros_are_unique_letters():
    names = [n.macro for n in reporting.REGISTRY]
    assert len(names) == len(set(names))
    assert all(name.isalpha() for name in names)


@pytest.mark.skipif(
    not (reporting.MANIFESTS / "robust001-runtime.json").exists(), reason="tracked manifests"
)
def test_every_registry_path_exists_in_the_tracked_manifests():
    values = numbers()
    assert set(values) == {n.macro for n in reporting.REGISTRY}


def test_support_curves_use_account_projection_slopes_and_skip_undefined_cells():
    leaf = {"estimate": 0.7, "ci95": [0.6, 0.8]}
    gate = {"slope_real_moved": {"ci95": [0.9, 1.1]}, "b_rev": leaf}
    cells = {f"m{d:.2f}": {"b_rev": leaf} for d in reporting.DOSES}
    agent = {f"m{d:.2f}": gate for d in reporting.DOSES}
    readout = "mlp-20261005"
    summary = {
        "dose_response": {f"{readout}:block": {"challenge": cells, "archive": cells}},
        "agent_probe": {f"{readout}:agent": {"challenge": agent, "archive": agent}},
    }
    payload = {"summary": {"readouts": {readout: summary}}}
    slopes = {f"m{d:.2f}": {"A": 0.9, "B": 0.64, "C": 1.0} for d in reporting.DOSES}
    slopes["m0.15"] = {"A": None, "B": None, "C": None}
    accounts = {"readout": readout, "account_slopes": {"challenge": slopes, "archive": slopes}}
    curves = reporting.support_curves(payload, accounts)
    assert curves["archive"][0]["B"] is None
    assert curves["archive"][3]["B"] == 0.64
    with pytest.raises(ValueError, match="Account slopes"):
        reporting.support_curves(payload, {**accounts, "readout": "ridge"})


def _curve_payload(arms=None, shift=0):
    arms = arms or [arm for arm, _, _ in reporting.CURVE_ARMS]
    runs = {}
    for arm in arms:
        for seed, offset in ((1, 0.0), (2, 0.2)):
            runs[f"{arm}-{seed}"] = {
                "arm": arm,
                "train_curve": [[u + shift * seed, 1.0 + offset + u] for u in (1, 2, 3)],
                "validation_curve": [[250, 0.5 + offset], [500, 0.4 + offset]],
            }
    return {"runs": runs}


def test_seed_mean_curves_average_seeds_and_smooth():
    curves = reporting.seed_mean_curves(_curve_payload(), "validation_curve")
    updates, values = curves["balanced"]
    assert updates == [250, 500] and values == pytest.approx([0.6, 0.5])
    updates, values = reporting.seed_mean_curves(_curve_payload(), "train_curve", smooth=2)[
        "archive_only"
    ]
    assert updates == [2, 3] and values == pytest.approx([2.6, 3.6])


def test_seed_mean_curves_fail_on_missing_arm_or_misaligned_seeds():
    with pytest.raises(MissingField, match="no runs for"):
        reporting.seed_mean_curves(_curve_payload(arms=["balanced"]), "train_curve")
    with pytest.raises(ValueError, match="different updates"):
        reporting.seed_mean_curves(_curve_payload(shift=1), "train_curve")


@pytest.mark.skipif(
    not (reporting.MANIFESTS / "report-examples.json").exists(), reason="tracked manifests"
)
def test_report_figures_build_from_tracked_manifests(tmp_path):
    assert set(reporting.FIGURES) == {"teaser", "method", "readout", "robustness", "finetuning"}
    for name in reporting.FIGURES:
        assert reporting.FIGURES[name](tmp_path).exists()


def test_tee_outline_matches_the_simulator_geometry():
    import numpy as np

    from jepa_action_effects.report_examples import tee_polygons

    outline = reporting.tee_outline(250.0, 240.0, 0.7)
    bar, stem = tee_polygons(250.0, 240.0, 0.7)
    corners = np.vstack([bar, stem])
    # Every outline vertex is a bar or stem corner, or where the stem meets the bar.
    for vertex in outline:
        assert np.min(np.linalg.norm(corners - vertex, axis=1)) < 1e-9


def test_count_percent_and_list_index_rendering(tmp_path):
    payload = {"n": 2336736, "share": 0.4066, "pair": {"ci95": [1.0, 5.59]}}
    (tmp_path / "m.json").write_text(json.dumps(payload))
    registry = (
        Number("rows", "m.json", ("n",), 0, "count"),
        Number("share", "m.json", ("share",), 0, "percent"),
        Number("upper", "m.json", ("pair", "ci95", "1"), 1),
    )
    assert numbers(registry, tmp_path) == {
        "rows": "2{,}336{,}736",
        "share": "41\\%",
        "upper": "5.6",
    }
