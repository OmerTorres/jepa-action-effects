"""CALIB-001: calibration-side reanalysis of opened records (exploratory, D030).

    python -m jepa_action_effects.calib001 phys     # PHYS-002 development and reserved
    python -m jepa_action_effects.calib001 shift    # SHIFT-002 development and reserved
    python -m jepa_action_effects.calib001 train    # TRAIN-001 confirmation and windows
    python -m jepa_action_effects.calib001 report   # reproduction gate, manifest, tables
    python -m jepa_action_effects.calib001 all

Sealed outputs are read, never written, and no model is run. PHYS-002 and SHIFT-002
use their saved latents; TRAIN-001 uses its saved decoded endpoints. Results go to
`outputs/calib001/`, the manifest to `docs/manifests/calib001-runtime.json` and the
generated tables to `docs/CALIB-001_RESULT_DRAFT.md`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.calib001_metrics import (
    K_VALUES,
    STATISTICS,
    row_statistics,
    summarize_group,
)
from jepa_action_effects.next_common import (
    DELTA,
    MANIFESTS,
    Stopwatch,
    load_family,
    runtime_info,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file

ROOT = Path("outputs/calib001")
PHYS = Path("outputs/phys002")
SHIFT2 = Path("outputs/shift002")
TRAIN = Path("outputs/train001")
READOUT2 = Path("outputs/readout002")
DRAFT = Path("docs/CALIB-001_RESULT_DRAFT.md")
PRIMARY = "mlp-20261005"
DRAWS = 2000
SEED = 20261201
TOLERANCE = 1e-9
SHIFT_FOLDERS = {"development": "develop", "reserved": "confirm"}
CONDITIONS = ("frozen", "archive_weighted", "balanced")
COMPACT = ("b_rev", "b_cal_k0", "b_cal_k1")
MOVED_ONLY = ("b_rev_moved", "ratio_moved", "slope_real_moved")
Decoder = Callable[[np.ndarray], np.ndarray]


def _decoders() -> tuple[dict[str, Decoder], dict[str, str]]:
    """READOUT-002: the three MLPs (primary first) and the linear ridge readout."""
    from jepa_action_effects.readout_scale import RidgeProbe
    from jepa_action_effects.train001 import readout_decoders

    decoders, digests = readout_decoders(READOUT2)
    path = READOUT2 / "models" / "ridge.npz"
    with np.load(path, allow_pickle=False) as archive:
        ridge = RidgeProbe.from_arrays({key: archive[key] for key in archive.files})
    decoders["ridge"] = lambda x: ridge.predict(np.asarray(x, np.float64))[:, :2]
    digests["ridge"] = sha256_file(path)
    return decoders, digests


def _vectors(real, predicted, history_states, future_states, decode: Decoder):
    """Step-2 decoded predicted, decoded real and true block displacements, each (6, 2)."""
    future = real[3:].reshape(6, 2, -1)
    start = decode(real[2:3])[0]
    true_end = np.asarray(future_states, float)[:, 1, 2:4]
    true_start = np.asarray(history_states, float)[-1, 2:4]
    return decode(predicted[:, 1]) - start, decode(future[:, 1]) - start, true_end - true_start


def _latent(real, predicted) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Readout-free changes of the projected latent from the current frame, per step."""
    future = real[3:].reshape(6, 2, -1)
    current = real[2]
    return {
        step: (predicted[:, step - 1] - current, future[:, step - 1] - current) for step in (1, 2)
    }


def _saved_latents(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return arrays["real"].astype(np.float64), arrays["predicted"].astype(np.float64)


def _stack(units: list[dict[str, np.ndarray]]) -> tuple[list[str], np.ndarray]:
    names = sorted({name for unit in units for name in unit})
    zero = np.zeros(len(STATISTICS))
    return names, np.stack([[unit.get(name, zero) for name in names] for unit in units])


# --------------------------------------------------------------------------- PHYS-002
def phys(root: Path = ROOT) -> dict[str, Any]:
    """One cell per cohort and readout: all six candidates of every family."""
    from jepa_action_effects.review_followup import _phys_items

    decoders, _ = _decoders()
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for phase, identifier, path, saved in _phys_items():
        family = load_family(path)
        real, predicted = _saved_latents(saved)
        latent = _latent(real, predicted)
        unit = {}
        for name, decode in decoders.items():
            vectors = _vectors(
                real, predicted, family["history_states"], family["future_states"], decode
            )
            unit[name] = row_statistics(*vectors, DELTA, latent).sum(axis=0)
        cohort = "challenge" if "challenge" in identifier else "ordinary"
        groups.setdefault(f"{phase}:{cohort}", []).append(unit)
    result = {}
    for offset, (key, units) in enumerate(sorted(groups.items())):
        names, sums = _stack(units)
        result[key] = summarize_group(names, sums, DRAWS, SEED + offset)
    write_json(root / "phys.json", result)
    return {key: value["units"] for key, value in result.items()}


# --------------------------------------------------------------------------- SHIFT-002
def _shift_contrasts(state: str, readouts) -> list[tuple[str, str, str, str]]:
    high, low, structure = (
        ("m0.60", "m0.30", "constant") if state == "challenge" else ("x2.0", "x1.0", "pooled")
    )
    return [
        (
            f"{name}|{key}|{high}-{low}",
            f"{name}|{high}|{structure}",
            f"{name}|{low}|{structure}",
            key,
        )
        for name in readouts
        for key in ("b_cal_k1", "b_rev", "b_cal_k0", "energy_ratio", "alignment")
    ]


def shift(root: Path = ROOT) -> dict[str, Any]:
    """Cells per readout × dose label × structure, plus structures pooled."""
    decoders, _ = _decoders()
    collection = sealed(SHIFT2 / "collection.json")
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for family_id, entry in sorted(collection["families"].items()):
        family = load_family(SHIFT2 / entry["path"])
        real, predicted = _saved_latents(
            SHIFT2 / SHIFT_FOLDERS[entry["split"]] / f"{family_id}.npz"
        )
        latent = _latent(real, predicted)
        spec = family["metadata"]["spec"]
        unit: dict[str, np.ndarray] = {}
        for name, decode in decoders.items():
            vectors = _vectors(
                real, predicted, family["history_states"], family["future_states"], decode
            )
            rows = row_statistics(*vectors, DELTA, latent)
            for index, candidate in enumerate(family["metadata"]["candidate_ids"]):
                label, structure = spec[candidate]["label"], spec[candidate]["structure"]
                for cell in (f"{name}|{label}|{structure}", f"{name}|{label}|pooled"):
                    unit[cell] = unit.get(cell, 0.0) + rows[index]
        groups.setdefault(f"{entry['split']}:{entry['state']}", []).append(unit)
    result = {}
    for offset, (key, units) in enumerate(sorted(groups.items())):
        names, sums = _stack(units)
        contrasts = _shift_contrasts(key.split(":")[1], decoders)
        result[key] = summarize_group(names, sums, DRAWS, SEED + 100 + offset, contrasts)
    write_json(root / "shift.json", result)
    return {key: value["units"] for key, value in result.items()}


# --------------------------------------------------------------------------- TRAIN-001
def _train_contrasts(group: str, readouts) -> list[tuple[str, str, str, str]]:
    if group == "ordinary":
        return [
            (f"{PRIMARY}|b_cal_k1|frozen-{c}", f"{PRIMARY}|frozen", f"{PRIMARY}|{c}", "b_cal_k1")
            for c in CONDITIONS[1:]
        ]
    trends = [
        (f"{r}|{c}|{key}|m0.60-m0.30", f"{r}|{c}|m0.60", f"{r}|{c}|m0.30", key)
        for r in readouts
        for c in CONDITIONS
        for key in ("b_cal_k1", "b_rev", "b_cal_k0", "energy_ratio", "alignment")
    ]
    tuned = [
        (
            f"{PRIMARY}|b_cal_k1|{label}|frozen-{c}",
            f"{PRIMARY}|frozen|{label}",
            f"{PRIMARY}|{c}|{label}",
            "b_cal_k1",
        )
        for label in ("m0.45", "m0.60")
        for c in CONDITIONS[1:]
    ]
    return trends + tuned


def train(root: Path = ROOT) -> dict[str, Any]:
    """Cells per readout × condition × magnitude; fine-tuned seeds pooled within families."""
    from jepa_action_effects.train001 import READOUTS
    from jepa_action_effects.train001_protocol import MAGNITUDES, magnitude_label

    collection = sealed(TRAIN / "confirmation" / "collection.json")
    registry = sealed(TRAIN / "seal.json")["registry"]
    models = sorted(model for model, entry in registry.items() if entry["kind"] != "real")
    records = TRAIN / "confirm" / "records"
    labels = [magnitude_label(m) for m in MAGNITUDES]

    def load(unit: str, model: str) -> dict:
        return json.loads((records / f"{unit}__{model}.json").read_text())

    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for kind in ("families", "windows"):
        for unit_id in sorted(collection[kind]):
            real = load(unit_id, "real-frames")
            true = np.asarray(real["true_end"], float) - np.asarray(real["true_start"], float)
            unit: dict[str, np.ndarray] = {}
            for model in models:
                condition = model.split("-")[0]
                record = load(unit_id, model)
                for readout in READOUTS:
                    start = np.asarray(real["decoded"][readout]["start"], float)
                    rows = row_statistics(
                        np.asarray(record["decoded"][readout]["pred_end"], float) - start,
                        np.asarray(real["decoded"][readout]["real_end"], float) - start,
                        true,
                        DELTA,
                    )
                    for index, row in enumerate(rows):
                        cell = f"{readout}|{condition}"
                        if kind == "families":
                            cell += f"|{labels[index]}"
                        unit[cell] = unit.get(cell, 0.0) + row
            group = real["scene"] if kind == "families" else "ordinary"
            groups.setdefault(group, []).append(unit)
    result = {}
    for offset, (key, units) in enumerate(sorted(groups.items())):
        names, sums = _stack(units)
        contrasts = _train_contrasts(key, READOUTS)
        result[key] = summarize_group(names, sums, DRAWS, SEED + 200 + offset, contrasts)
    write_json(root / "train.json", result)
    return {key: value["units"] for key, value in result.items()}


# --------------------------------------------------------------------------- report
def _compare(ours: float | None, reference: float | None) -> dict[str, Any]:
    if ours is None or reference is None:
        passed = ours is None and reference is None
        difference = None
    else:
        difference = abs(ours - reference)
        passed = difference <= TOLERANCE
    return {"ours": ours, "sealed": reference, "difference": difference, "passed": passed}


def reproduction(results: dict[str, Any]) -> dict[str, Any]:
    """Sealed moved-only values that CALIB-001 must reproduce before interpretation."""
    checks: dict[str, Any] = {}
    phys_sealed = sealed(PHYS / "confirm.json")["reserved"]["summary"]
    for cohort in ("challenge", "ordinary"):
        cell = results["phys"][f"reserved:{cohort}"]["cells"][PRIMARY]
        for ours, theirs in (("b_rev_moved", "slope_pred"), ("slope_real_moved", "slope_real")):
            checks[f"phys002:{cohort}:{theirs}"] = _compare(
                cell[ours]["estimate"], phys_sealed[cohort][theirs]["estimate"]
            )
    shift_sealed = sealed(SHIFT2 / "confirm.json")["summary"]
    for state in ("challenge", "archive"):
        for name, cell in results["shift"][f"reserved:{state}"]["cells"].items():
            readout, label, structure = name.split("|")
            suffix = "" if structure == "pooled" else f"_{structure}"
            key = f"R_{state}_{label}{suffix}"
            if readout == PRIMARY and key in shift_sealed:
                checks[f"shift002:{key}"] = _compare(
                    cell["ratio_moved"]["estimate"], shift_sealed[key]["estimate"]
                )
    diagnostics = sealed(TRAIN / "confirm" / "summary.json")["readouts"][PRIMARY]["diagnostics"]
    for key, value in diagnostics["frozen"].items():
        scene, label = key.split(":")
        cell = results["train"][scene]["cells"][f"{PRIMARY}|frozen|{label}"]
        checks[f"train001:frozen:{key}:ratio"] = _compare(
            cell["ratio_moved"]["estimate"], value["slope_ratio_pred_over_real"]
        )
    differences = [c["difference"] for c in checks.values() if c["difference"] is not None]
    return {
        "checks": checks,
        "count": len(checks),
        "max_difference": max(differences, default=None),
        "passed": all(c["passed"] for c in checks.values()),
    }


def _compact(results: dict[str, Any]) -> dict[str, Any]:
    """Primary-readout cells in full; other readouts keep the headline slopes only.

    Moved-only estimands exist to reproduce sealed values and are kept in the
    reproduction block instead.
    """
    compact: dict[str, Any] = {}
    for dataset, groups in results.items():
        for group, payload in groups.items():
            cells = {}
            for name, cell in payload["cells"].items():
                if name.startswith(f"{PRIMARY}|") or name == PRIMARY:
                    cells[name] = {k: v for k, v in cell.items() if k not in MOVED_ONLY}
                else:
                    cells[name] = {
                        **{key: cell[key] for key in ("n", "n_moved", *COMPACT)},
                        "class_k1": cell["class_k1"],
                        "class_k0": cell["class_k0"],
                    }
            compact[f"{dataset}/{group}"] = {
                "units": payload["units"],
                "cells": cells,
                "contrasts": payload["contrasts"],
            }
    return compact


def _fmt(entry: dict[str, Any] | None, digits: int = 3) -> str:
    if not entry or entry.get("estimate") is None:
        return "—"
    text = f"{entry['estimate']:.{digits}f}"
    if entry.get("ci95"):
        low, high = entry["ci95"]
        text += f" [{low:.{digits}f}, {high:.{digits}f}]"
    return text


def _show(name: str) -> str:
    """Cell names contain '|', which would split a markdown table cell."""
    return name.replace("|", " · ")


def _cell_table(cells: dict[str, Any], names: list[str], latent: bool) -> list[str]:
    header = (
        "| Cell | n / moved | b_rev | scalar gain b_cal (k=0) | class (k=0) | energy ratio P/T"
        " | alignment ρ | b_cal k=1 (sensitivity) |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    if latent:
        header += " latent b_cal step 1 | latent b_cal step 2 |"
        rule += " --- | --- |"
    lines = [header, rule]
    for name in names:
        cell = cells[name]
        row = (
            f"| {_show(name.split('|', 1)[-1]) if '|' in name else name} | {cell['n']} / "
            f"{cell['n_moved']} | {_fmt(cell['b_rev'])} | {_fmt(cell['b_cal_k0'])} | "
            f"{cell['class_k0']} | {_fmt(cell['energy_ratio'])} | {_fmt(cell['alignment'])} | "
            f"{_fmt(cell['b_cal_k1'])} |"
        )
        if latent:
            row += f" {_fmt(cell['z1_b_cal'])} | {_fmt(cell['z2_b_cal'])} |"
        lines.append(row)
    return lines


def draft(results: dict[str, Any], checks: dict[str, Any]) -> str:
    lines = [
        "# CALIB-001 — generated tables (exploratory, D030; labels corrected by D031)",
        "",
        "Generated by `python -m jepa_action_effects.calib001 report`; do not edit by hand.",
        "Interpretation is in CALIB-001_RESULT.md. All families were already opened.",
        "",
        f"Reproduction gate: **{'passed' if checks['passed'] else 'FAILED'}** "
        f"({checks['count']} sealed values; max difference {checks['max_difference']:.2e}).",
        "",
        "- b_rev = C/T = Σp·t/Σt·t (projection slope, all candidates in the design cell).",
        "- Scalar gain b_cal = C/P = Σp·t/Σp·p: optimal through-origin factor from decoded",
        "  predictions to outcomes. Necessary, not sufficient, for physical calibration; it",
        "  cannot separate predictor bias from latent uncertainty through a nonlinear decoder.",
        "- Energy ratio P/T (magnitude) and alignment ρ = C/sqrt(PT) (direction), uncorrected:",
        "  b_rev = ρ·sqrt(P/T) and b_cal = ρ/sqrt(P/T). Readout error inflates P.",
        "- b_cal k=1 subtracts the real-frame error energy from P: a sensitivity analysis under",
        "  an additive error orthogonal to prediction and outcome, not recovered truth.",
        "- Latent columns: readout-free scalar gain of the projected latent change.",
        f"- Primary readout `{PRIMARY}`; 95% family-bootstrap intervals, {DRAWS} draws.",
        "",
    ]
    train = results["train"]
    for scene in ("challenge", "archive", "ordinary"):
        cells = train[scene]["cells"]
        names = [n for n in sorted(cells) if n.startswith(f"{PRIMARY}|")]
        names.sort(key=lambda n: (CONDITIONS.index(n.split("|")[1]), n))
        lines += [f"## TRAIN-001 confirmation — {scene}", "", *_cell_table(cells, names, False), ""]
    for phase in ("reserved", "development"):
        for state in ("challenge", "archive"):
            cells = results["shift"][f"{phase}:{state}"]["cells"]
            names = [n for n in sorted(cells) if n.startswith(f"{PRIMARY}|")]
            lines += [f"## SHIFT-002 {phase} — {state}", "", *_cell_table(cells, names, True), ""]
    for key in sorted(results["phys"]):
        cells = results["phys"][key]["cells"]
        lines += [f"## PHYS-002 {key}", "", *_cell_table(cells, [PRIMARY], True), ""]
    lines += [
        "## Paired contrasts",
        "",
        "| Group | Contrast | Estimate [95% CI] |",
        "| --- | --- | --- |",
    ]
    for dataset in ("train", "shift"):
        for group, payload in sorted(results[dataset].items()):
            for label, contrast in payload["contrasts"].items():
                if label.startswith(f"{PRIMARY}|"):
                    lines.append(f"| {dataset}/{group} | {_show(label)} | {_fmt(contrast)} |")
    lines += [
        "",
        "## Readout sensitivity (scalar gain b_cal, k=0)",
        "",
        "| Group | Cell | " + " | ".join(_readouts(results)) + " |",
        "| --- | --- |" + " --- |" * len(_readouts(results)),
    ]
    for dataset, group, suffix in _key_cells():
        cells = results[dataset][group]["cells"]
        values = [
            _fmt(cells.get(f"{r}|{suffix}" if suffix else r, {}).get("b_cal_k0"))
            for r in _readouts(results)
        ]
        lines.append(
            f"| {dataset}/{group} | {_show(suffix) or 'all'} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        f"## Error-model sensitivity (b_cal, `{PRIMARY}`; k>0 assumes orthogonal error)",
        "",
        "| Group | Cell | " + " | ".join(f"k={k:g}" for k in K_VALUES) + " |",
        "| --- | --- |" + " --- |" * len(K_VALUES),
    ]
    for dataset, group, suffix in _key_cells():
        cell = results[dataset][group]["cells"].get(f"{PRIMARY}|{suffix}" if suffix else PRIMARY)
        if cell:
            values = [_fmt(cell[f"b_cal_k{k:g}"]) for k in K_VALUES]
            lines.append(
                f"| {dataset}/{group} | {_show(suffix) or 'all'} | " + " | ".join(values) + " |"
            )
    lines += [
        "",
        "## Spurious predicted motion where the block does not move",
        "",
        "Static excess = mean(|p|² − |r|²) over candidates without block motion (units²): the",
        "predicted displacement's squared length beyond the real-frame readout's. Calibration",
        "slopes are undefined in these cells because t ≡ 0.",
        "",
        "| Group | Cell | static candidates | static excess | real-frame noise rms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, group, suffix in _static_cells():
        cell = results[dataset][group]["cells"][f"{PRIMARY}|{suffix}"]
        lines.append(
            f"| {dataset}/{group} | {_show(suffix)} | {cell['n'] - cell['n_moved']} | "
            f"{_fmt(cell['static_excess'], 1)} | {_fmt(cell['noise_rms'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def _readouts(results: dict[str, Any]) -> list[str]:
    names = {n.split("|")[0] for n in results["shift"]["reserved:challenge"]["cells"]}
    return [PRIMARY, *sorted(names - {PRIMARY})]


def _static_cells() -> list[tuple[str, str, str]]:
    return [
        ("train", "challenge", "frozen|m0.15"),
        ("train", "challenge", "archive_weighted|m0.15"),
        ("train", "challenge", "balanced|m0.15"),
        ("shift", "reserved:challenge", "m0.15|constant"),
        ("shift", "development:challenge", "m0.15|constant"),
    ]


def _key_cells() -> list[tuple[str, str, str]]:
    return [
        ("train", "challenge", "frozen|m0.30"),
        ("train", "challenge", "frozen|m0.60"),
        ("train", "archive", "frozen|m0.30"),
        ("train", "archive", "frozen|m0.60"),
        ("train", "challenge", "balanced|m0.60"),
        ("train", "archive", "balanced|m0.60"),
        ("shift", "reserved:challenge", "m0.30|constant"),
        ("shift", "reserved:challenge", "m0.60|constant"),
        ("shift", "reserved:archive", "x1.0|pooled"),
        ("shift", "reserved:archive", "x2.0|pooled"),
        ("phys", "reserved:challenge", ""),
        ("phys", "reserved:ordinary", ""),
    ]


def report(root: Path = ROOT) -> dict[str, Any]:
    results = {
        name: json.loads((root / f"{name}.json").read_text()) for name in ("phys", "shift", "train")
    }
    checks = reproduction(results)
    if not checks["passed"]:
        # Nothing is interpreted or published unless the sealed values are reproduced.
        write_json(root / "reproduction-failed.json", checks)
        raise SystemExit("CALIB-001 reproduction gate failed; manifest and tables not written")
    _, digests = _decoders()
    summary = {
        "task": "CALIB-001",
        "status": "exploratory (all families already opened; D030)",
        "definitions": {
            "b_rev": "sum(p.t)/sum(t.t) over all candidates in a design cell",
            "b_cal_k": "sum(p.t)/(sum(p.p) - k*sum(|r-t|^2)), k in "
            + ", ".join(f"{k:g}" for k in K_VALUES),
            "classification": "class_k0 primary (D031); class_k1 is a sensitivity label",
            "magnitude_direction": "energy_ratio = P/T, alignment = C/sqrt(P*T), uncorrected",
            "undefined": "b_cal needs sum(t.t) > 0 and a positive corrected denominator",
            "static_excess": "mean(|p|^2 - |r|^2) over candidates without block motion",
            "latent": "z{step}_b_cal = sum(dz_pred.dz_real)/sum(dz_pred.dz_pred), projected space",
            "draws": DRAWS,
            "delta": DELTA,
        },
        "runtime": runtime_info(),
        "sources": source_hashes(
            {
                "runner": Path(__file__),
                "metrics": Path(__file__).with_name("calib001_metrics.py"),
                "uv_lock": Path("uv.lock"),
            }
        ),
        "inputs": {
            "phys002_confirm": sha256_file(PHYS / "confirm.json"),
            "shift002_confirm": sha256_file(SHIFT2 / "confirm.json"),
            "shift002_collection": sha256_file(SHIFT2 / "collection.json"),
            "train001_summary": sha256_file(TRAIN / "confirm" / "summary.json"),
            "train001_collection": sha256_file(TRAIN / "confirmation" / "collection.json"),
            "readouts": digests,
        },
        "outputs": {name: sha256_file(root / f"{name}.json") for name in results},
        "reproduction": checks,
        "results": _compact(results),
    }
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "calib001-runtime.json", summary)
    DRAFT.write_text(draft(results, checks))
    return {"reproduction_passed": checks["passed"], "max_difference": checks["max_difference"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("phys", "shift", "train", "report", "all"))
    args = parser.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    stages = {"phys": phys, "shift": shift, "train": train, "report": report}
    watch = Stopwatch(ROOT)
    for stage in stages if args.stage == "all" else [args.stage]:
        result = watch.run(stage, stages[stage], ROOT)
        print(stage, json.dumps(result, default=str))


if __name__ == "__main__":
    main()
