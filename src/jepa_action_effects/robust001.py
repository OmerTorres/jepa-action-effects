"""ROBUST-001: multi-instrument re-decoding of opened predicted latents (exploratory, D034).

    python -m jepa_action_effects.robust001 knn      # choose k on READOUT-002 validation
    python -m jepa_action_effects.robust001 shift    # SHIFT-002 development and reserved
    python -m jepa_action_effects.robust001 phys     # PHYS-002 development and reserved
    python -m jepa_action_effects.robust001 report   # reproduction gate, manifest, tables
    python -m jepa_action_effects.robust001 all

No model is run and no sealed file is written. Saved latents are decoded by five
instruments (three READOUT-002 MLPs, ridge, and a kNN readout over the READOUT-002
training bank). Family files are verified against their collection hashes; every
consumed latent, shard and model file is hashed. The report stage first reproduces every
CALIB-001 cell for the four shared instruments and writes nothing if that fails.
Results go to `outputs/robust001/`, the manifest to
`docs/manifests/robust001-runtime.json` and tables to `docs/ROBUST-001_RESULT_DRAFT.md`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.calib001 import DRAWS as CALIB_DRAWS
from jepa_action_effects.calib001 import (
    PHYS,
    READOUT2,
    SHIFT2,
    SHIFT_FOLDERS,
    _decoders,
    _latent,
    _saved_latents,
    _shift_contrasts,
    _stack,
    _vectors,
)
from jepa_action_effects.calib001 import ROOT as CALIB_ROOT
from jepa_action_effects.calib001 import SEED as CALIB_SEED
from jepa_action_effects.calib001_metrics import row_statistics, summarize_group
from jepa_action_effects.next_common import (
    DELTA,
    MANIFESTS,
    Stopwatch,
    collection_manifest,
    family_path,
    load_family,
    runtime_info,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.robust001_metrics import (
    K_GRID,
    QUANTILE,
    KnnReadout,
    max_pairwise,
    offmanifold_threshold,
    select_k,
    spearman,
)

ROOT = Path("outputs/robust001")
SHARDS = READOUT2 / "shards"
MANIFEST = MANIFESTS / "robust001-runtime.json"
CALIB_MANIFEST = MANIFESTS / "calib001-runtime.json"
DRAFT = Path("docs/ROBUST-001_RESULT_DRAFT.md")
PRIMARY = "mlp-20261005"
KNN = "knn"
SPREAD = (PRIMARY, "ridge", KNN)
GATE_KEYS = ("b_rev", "energy_ratio", "alignment", "b_cal_k0")
TOLERANCE = 1e-9
ON = "on"
Decoder = Callable[[np.ndarray], np.ndarray]


class GateFailure(RuntimeError):
    """Lineage or reproduction failed; nothing is reported."""


# --------------------------------------------------------------------------- inputs
def _shard_files(split: str) -> list[Path]:
    files = sorted(SHARDS.glob(f"{split}-*.npz"))
    if not files:
        raise GateFailure(f"No READOUT-002 {split} shards in {SHARDS}")
    return files


def _load_shards(files: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    latents, labels = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            latents.append(archive["projected"].astype(np.float64))
            labels.append(archive["labels"][:, 2:4].astype(np.float64))  # block x, y
    return np.concatenate(latents), np.concatenate(labels)


def _hashes(files: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in files}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


# --------------------------------------------------------------------------- kNN
def knn(root: Path = ROOT) -> dict[str, Any]:
    """Choose k on the validation shards before any study latent is decoded."""
    train_files, validation_files = _shard_files("train"), _shard_files("validation")
    bank, labels = _load_shards(train_files)
    queries, truth = _load_shards(validation_files)
    selection = select_k(bank, labels, queries, truth, K_GRID)
    decoders, digests = _decoders()
    reference = {
        name: float(np.mean(np.linalg.norm(decode(queries) - truth, axis=1)))
        for name, decode in decoders.items()
    }
    payload = {
        **selection,
        "grid": list(K_GRID),
        "bank_size": len(bank),
        "validation_size": len(queries),
        "train_shards": _hashes(train_files),
        "validation_shards": _hashes(validation_files),
        "readout_sha256": digests,
        "validation_mean_error_existing": reference,
    }
    write_json(root / "knn.json", payload)
    return {"k": selection["k"], "errors": selection["validation_mean_error"], **reference}


def _knn_readout(root: Path) -> tuple[KnnReadout, dict[str, Any]]:
    """The selected kNN readout; refuses a selection made from different shards."""
    path = root / "knn.json"
    _require(path.exists(), "Run the knn stage first")
    selection = json.loads(path.read_text())
    train_files = _shard_files("train")
    _require(selection["train_shards"] == _hashes(train_files), "Training shards changed")
    _require(
        selection["validation_shards"] == _hashes(_shard_files("validation")),
        "Validation shards changed",
    )
    bank, labels = _load_shards(train_files)
    return KnnReadout(bank, labels, int(selection["k"])), selection


def _instruments(root: Path) -> tuple[dict[str, Decoder], KnnReadout, dict[str, Any]]:
    decoders, digests = _decoders()
    readout, selection = _knn_readout(root)
    _require(selection["readout_sha256"] == digests, "READOUT-002 models changed since knn")
    decoders[KNN] = readout.predict
    return decoders, readout, {"readouts": digests, "knn": sha256_file(root / "knn.json")}


# --------------------------------------------------------------------------- shared
def _step2(real: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and real step-2 latents of the six candidates, each (6, 192)."""
    return predicted[:, 1], real[3:].reshape(6, 2, -1)[:, 1]


def _family_rows(family, real, predicted, decoders) -> dict[str, np.ndarray]:
    latent = _latent(real, predicted)
    return {
        name: row_statistics(
            *_vectors(real, predicted, family["history_states"], family["future_states"], decode),
            DELTA,
            latent,
        )
        for name, decode in decoders.items()
    }


def _spread(decoders: dict[str, Decoder], latents: np.ndarray) -> np.ndarray:
    return max_pairwise(np.stack([decoders[name](latents) for name in SPREAD], axis=1))


def _thresholds(passes: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[np.ndarray]] = {}
    for item in passes:
        groups.setdefault(item["group"], []).append(item["d_real"])
    return {g: offmanifold_threshold(np.concatenate(v)) for g, v in sorted(groups.items())}


def _summaries(groups, contrasts_for, seed: int) -> dict[str, Any]:
    result = {}
    for offset, (key, units) in enumerate(sorted(groups.items())):
        names, sums = _stack(units)
        result[key] = summarize_group(names, sums, CALIB_DRAWS, seed + offset, contrasts_for(key))
    return result


def _describe(candidates: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """Flag rates, distances and instrument spread by group and cell (descriptive)."""
    cells: dict[str, list[dict]] = {}
    for row in candidates:
        cells.setdefault("|".join(str(row[k]) for k in keys), []).append(row)
    out = {}
    for name, rows in sorted(cells.items()):
        d_pred = np.array([r["d_pred"] for r in rows])
        spread_pred = np.array([r["spread_pred"] for r in rows])
        out[name] = {
            "candidates": len(rows),
            "offmanifold_rate": float(np.mean([r["flag"] for r in rows])),
            "d_pred_median": float(np.median(d_pred)),
            "d_real_median": float(np.median([r["d_real"] for r in rows])),
            "spread_pred_median": float(np.median(spread_pred)),
            "spread_real_median": float(np.median([r["spread_real"] for r in rows])),
            "spearman_d_pred_spread_pred": spearman(d_pred, spread_pred),
        }
    return out


# --------------------------------------------------------------------------- SHIFT-002
def _shift_all_contrasts(key: str, decoders) -> list[tuple[str, str, str, str]]:
    state = key.split(":")[1]
    base = _shift_contrasts(state, decoders)
    on = [
        (f"{label}|{ON}", f"{left}|{ON}", f"{right}|{ON}", estimand)
        for label, left, right, estimand in base
        if estimand in ("b_rev", "energy_ratio", "alignment")
    ]
    return base + on


def shift(root: Path = ROOT) -> dict[str, Any]:
    decoders, readout, digests = _instruments(root)
    collection = sealed(SHIFT2 / "collection.json")
    passes, latent_hashes = [], {}
    for family_id, entry in sorted(collection["families"].items()):
        path = SHIFT2 / entry["path"]
        _require(sha256_file(path) == entry["sha256"], f"Family hash mismatch: {family_id}")
        saved = SHIFT2 / SHIFT_FOLDERS[entry["split"]] / f"{family_id}.npz"
        latent_hashes[family_id] = sha256_file(saved)
        real, predicted = _saved_latents(saved)
        step_pred, step_real = _step2(real, predicted)
        passes.append(
            {
                "family_id": family_id,
                "group": f"{entry['split']}:{entry['state']}",
                "family": load_family(path),
                "real": real,
                "predicted": predicted,
                "d_pred": readout.nearest_distance(step_pred),
                "d_real": readout.nearest_distance(step_real),
                "spread_pred": _spread(decoders, step_pred),
                "spread_real": _spread(decoders, step_real),
            }
        )
    thresholds = _thresholds(passes)
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    candidates = []
    for item in passes:
        family = item["family"]
        rows = _family_rows(family, item["real"], item["predicted"], decoders)
        flags = item["d_pred"] > thresholds[item["group"]]
        spec = family["metadata"]["spec"]
        unit: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(family["metadata"]["candidate_ids"]):
            label, structure = spec[candidate]["label"], spec[candidate]["structure"]
            names = [f"{label}|{structure}", f"{label}|pooled"]
            for name, statistics in rows.items():
                for cell in names:
                    key = f"{name}|{cell}"
                    unit[key] = unit.get(key, 0.0) + statistics[index]
                    if not flags[index]:
                        unit[f"{key}|{ON}"] = unit.get(f"{key}|{ON}", 0.0) + statistics[index]
            candidates.append(
                {
                    "group": item["group"],
                    "label": label,
                    "structure": structure,
                    "d_pred": float(item["d_pred"][index]),
                    "d_real": float(item["d_real"][index]),
                    "flag": bool(flags[index]),
                    "spread_pred": float(item["spread_pred"][index]),
                    "spread_real": float(item["spread_real"][index]),
                }
            )
        groups.setdefault(item["group"], []).append(unit)
    result = {
        "groups": _summaries(
            groups, lambda key: _shift_all_contrasts(key, decoders), CALIB_SEED + 100
        ),
        "thresholds": thresholds,
        "descriptive": _describe(candidates, ("group", "label")),
        "inputs": {"collection": sha256_file(SHIFT2 / "collection.json"), **digests},
        "latent_sha256": latent_hashes,
    }
    write_json(root / "shift.json", result)
    return {key: value["units"] for key, value in result["groups"].items()}


# --------------------------------------------------------------------------- PHYS-002
def _phys_items() -> list[tuple[str, str, Path, Path]]:
    """As review_followup._phys_items, but every family file is hash-verified."""
    from jepa_action_effects.phys002 import _roots

    roots, manifests = _roots()
    config = sealed(PHYS / "config.json")
    items = [
        (
            "develop",
            identifier,
            family_path(identifier, roots, manifests),
            PHYS / "develop" / "families" / f"{identifier.replace(':', '__')}.npz",
        )
        for identifier in config["development_ids"]
    ]
    manifest = collection_manifest(PHYS / "reserved")
    for key, record in sorted(manifest["family_files"].items()):
        identifier = f"reserved:{key}"
        path = PHYS / "reserved" / record["path"]
        _require(sha256_file(path) == record["sha256"], f"Family hash mismatch: {identifier}")
        items.append(
            (
                "reserved",
                identifier,
                path,
                PHYS / "confirm-reserved" / "families" / f"{identifier.replace(':', '__')}.npz",
            )
        )
    return items


def phys(root: Path = ROOT) -> dict[str, Any]:
    decoders, readout, digests = _instruments(root)
    passes, latent_hashes = [], {}
    for phase, identifier, path, saved in _phys_items():
        latent_hashes[identifier] = sha256_file(saved)
        real, predicted = _saved_latents(saved)
        step_pred, step_real = _step2(real, predicted)
        cohort = "challenge" if "challenge" in identifier else "ordinary"
        passes.append(
            {
                "group": f"{phase}:{cohort}",
                "family": load_family(path),
                "real": real,
                "predicted": predicted,
                "d_pred": readout.nearest_distance(step_pred),
                "d_real": readout.nearest_distance(step_real),
                "spread_pred": _spread(decoders, step_pred),
                "spread_real": _spread(decoders, step_real),
            }
        )
    thresholds = _thresholds(passes)
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    candidates = []
    for item in passes:
        rows = _family_rows(item["family"], item["real"], item["predicted"], decoders)
        flags = item["d_pred"] > thresholds[item["group"]]
        unit = {}
        for name, statistics in rows.items():
            unit[name] = statistics.sum(axis=0)
            unit[f"{name}|{ON}"] = statistics[~flags].sum(axis=0)
        for index in range(len(flags)):
            candidates.append(
                {
                    "group": item["group"],
                    "d_pred": float(item["d_pred"][index]),
                    "d_real": float(item["d_real"][index]),
                    "flag": bool(flags[index]),
                    "spread_pred": float(item["spread_pred"][index]),
                    "spread_real": float(item["spread_real"][index]),
                }
            )
        groups.setdefault(item["group"], []).append(unit)
    result = {
        "groups": _summaries(groups, lambda key: (), CALIB_SEED),
        "thresholds": thresholds,
        "descriptive": _describe(candidates, ("group",)),
        "inputs": digests,
        "latent_sha256": latent_hashes,
    }
    write_json(root / "phys.json", result)
    return {key: value["units"] for key, value in result["groups"].items()}


# --------------------------------------------------------------------------- report
def _calib_reference() -> dict[str, Any]:
    """CALIB-001 outputs, verified against the hashes in its tracked manifest."""
    recorded = json.loads(CALIB_MANIFEST.read_text())["outputs"]
    reference = {}
    for name in ("shift", "phys"):
        path = CALIB_ROOT / f"{name}.json"
        _require(sha256_file(path) == recorded[name], f"CALIB-001 {name} output changed")
        reference[name] = json.loads(path.read_text())
    return reference


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def reproduction(results: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Every shared CALIB-001 cell (estimate and interval) must be reproduced."""
    count, failures, largest = 0, [], 0.0
    for dataset in ("shift", "phys"):
        for group, payload in reference[dataset].items():
            ours = results[dataset]["groups"][group]["cells"]
            for cell, values in payload["cells"].items():
                for key in GATE_KEYS:
                    theirs, mine = values[key], ours[cell][key]
                    pairs = [(mine["estimate"], theirs["estimate"])]
                    if (mine["ci95"] is None) != (theirs["ci95"] is None):
                        pairs.append((None, 0.0))  # one interval defined, the other not
                    elif mine["ci95"] is not None:
                        pairs += list(zip(mine["ci95"], theirs["ci95"], strict=True))
                    for a, b in pairs:
                        count += 1
                        if not _close(a, b):
                            failures.append(f"{dataset}/{group}/{cell}/{key}")
                        elif a is not None:
                            largest = max(largest, abs(float(a) - float(b)))
    return {
        "checks": count,
        "failures": sorted(set(failures)),
        "max_difference": largest,
        "passed": not failures,
    }


def _contrast(results, group: str, label: str) -> dict[str, Any] | None:
    entry = results["shift"]["groups"][group]["contrasts"].get(label)
    return None if entry is None else {"estimate": entry["estimate"], "ci95": entry["ci95"]}


def _negative(entry: dict[str, Any] | None) -> bool:
    return bool(entry and entry["ci95"] and entry["ci95"][1] < 0)


def verdicts(results: dict[str, Any], instruments: Iterable[str]) -> dict[str, Any]:
    """D034 rules on the reserved SHIFT-002 dose contrasts."""
    cells = {
        "challenge": ("reserved:challenge", "m0.60-m0.30"),
        "archive": ("reserved:archive", "x2.0-x1.0"),
    }
    table: dict[str, dict[str, Any]] = {}
    for scene, (group, dose) in cells.items():
        for name in instruments:
            for estimand in ("energy_ratio", "b_rev", "alignment"):
                for stratum, suffix in (("all", ""), (ON, f"|{ON}")):
                    label = f"{name}|{estimand}|{dose}{suffix}"
                    table.setdefault(f"{scene}:{estimand}:{stratum}", {})[name] = _contrast(
                        results, group, label
                    )
    names = list(instruments)
    others = [n for n in names if n != KNN]

    def all_negative(key: str, among: list[str]) -> bool:
        return all(_negative(table[key][n]) for n in among)

    robust = all(all_negative(f"{s}:energy_ratio:all", names) for s in cells)
    survives = all(all_negative(f"{s}:energy_ratio:{ON}", names) for s in cells)
    return {
        "contrasts": table,
        "instrument_robust_trend": robust,
        "trend_without_knn": all(all_negative(f"{s}:energy_ratio:all", others) for s in cells),
        "on_manifold": "trend_survives" if survives else "inconclusive",
    }


def _fmt(entry: dict[str, Any] | None, digits: int = 3) -> str:
    if not entry or entry.get("estimate") is None:
        return "—"
    text = f"{entry['estimate']:.{digits}f}"
    if entry.get("ci95"):
        low, high = entry["ci95"]
        text += f" [{low:.{digits}f}, {high:.{digits}f}]"
    return text


def draft(results: dict[str, Any], checks: dict[str, Any], decision: dict[str, Any]) -> str:
    knn_selection = results["knn"]
    lines = [
        "# ROBUST-001 generated tables (exploratory, D034)",
        "",
        "Generated by `python -m jepa_action_effects.robust001 report`; do not edit.",
        "",
        f"Reproduction gate: {checks['checks']} CALIB-001 values, "
        f"max difference {checks['max_difference']:.3g}, passed {checks['passed']}.",
        "",
        "## kNN readout (validation mean block error, world units)",
        "",
        "| k | error |",
        "| ---: | ---: |",
        *(f"| {k} | {v:.3f} |" for k, v in knn_selection["validation_mean_error"].items()),
        "",
        f"Selected k = {knn_selection['k']}. Existing instruments on the same validation "
        "frames: "
        + ", ".join(
            f"{n} {v:.3f}" for n, v in knn_selection["validation_mean_error_existing"].items()
        )
        + ".",
        "",
        "## Reserved SHIFT-002 dose contrasts (high − low), per instrument",
        "",
        "| Scene | Estimand | Stratum | " + " | ".join(results["instruments"]) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in results["instruments"]) + " |",
    ]
    for key, row in decision["contrasts"].items():
        scene, estimand, stratum = key.split(":")
        values = " | ".join(_fmt(row[n]) for n in results["instruments"])
        lines.append(f"| {scene} | {estimand} | {stratum} | {values} |")
    lines += [
        "",
        f"Instrument-robust energy-ratio trend (all five): {decision['instrument_robust_trend']}. "
        f"Without kNN: {decision['trend_without_knn']}. "
        f"On-manifold stratum: {decision['on_manifold']}.",
        "",
        "## Reserved SHIFT-002 energy ratio by dose, per instrument",
        "",
    ]
    for group, structure in (("reserved:challenge", "constant"), ("reserved:archive", "pooled")):
        cells = results["shift"]["groups"][group]["cells"]
        labels = sorted({c.split("|")[1] for c in cells if c.startswith(f"{PRIMARY}|")})
        lines += [
            f"**{group}** ({structure})",
            "",
            "| Instrument | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join("---" for _ in labels) + " |",
        ]
        for name in results["instruments"]:
            row = [
                _fmt(cells.get(f"{name}|{lab}|{structure}", {}).get("energy_ratio"), 2)
                for lab in labels
            ]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
        lines.append("")
    lines += [
        "## Off-manifold flags and instrument spread (descriptive)",
        "",
        "| Dataset | Cell | n | flagged | d₁ pred / real (median) | spread pred / real (median) "
        "| Spearman(d₁, spread) |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for dataset in ("shift", "phys"):
        for cell, value in results[dataset]["descriptive"].items():
            rho = value["spearman_d_pred_spread_pred"]
            lines.append(
                f"| {dataset} | {cell} | {value['candidates']} | "
                f"{value['offmanifold_rate']:.2f} | "
                f"{value['d_pred_median']:.3f} / {value['d_real_median']:.3f} | "
                f"{value['spread_pred_median']:.2f} / {value['spread_real_median']:.2f} | "
                f"{'—' if rho is None else f'{rho:.2f}'} |"
            )
    lines += [
        "",
        "## PHYS-002 cells, per instrument (all candidates; b_rev, energy ratio, alignment)",
        "",
        "| Group | Instrument | stratum | b_rev | energy ratio | alignment |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group, payload in results["phys"]["groups"].items():
        for name in results["instruments"]:
            for stratum, key in (("all", name), (ON, f"{name}|{ON}")):
                cell = payload["cells"][key]
                lines.append(
                    f"| {group} | {name} | {stratum} | {_fmt(cell['b_rev'])} | "
                    f"{_fmt(cell['energy_ratio'])} | {_fmt(cell['alignment'])} |"
                )
    return "\n".join(lines) + "\n"


def _sources() -> dict[str, str]:
    here = Path(__file__).parent
    return source_hashes(
        {
            "robust001": here / "robust001.py",
            "robust001_metrics": here / "robust001_metrics.py",
            "calib001": here / "calib001.py",
            "calib001_metrics": here / "calib001_metrics.py",
        }
    )


def report(root: Path = ROOT) -> dict[str, Any]:
    results = {name: json.loads((root / f"{name}.json").read_text()) for name in ("shift", "phys")}
    results["knn"] = json.loads((root / "knn.json").read_text())
    decoders, _, digests = _instruments(root)
    for name in ("shift", "phys"):
        _require(results[name]["inputs"]["knn"] == digests["knn"], f"{name} used another kNN")
        _require(results[name]["inputs"]["readouts"] == digests["readouts"], f"{name} readouts")
    results["instruments"] = list(decoders)
    checks = reproduction(results, _calib_reference())
    if not checks["passed"]:
        raise GateFailure(f"Reproduction failed: {checks['failures'][:5]}")
    decision = verdicts(results, results["instruments"])
    payload = {
        "task": "ROBUST-001 (D034)",
        "label": "Exploratory re-decoding of opened records; no model run, no sealed file written",
        "status": "complete",
        "reproduction": checks,
        "knn": {
            k: results["knn"][k]
            for k in ("k", "validation_mean_error", "validation_mean_error_existing", "bank_size")
        },
        "offmanifold_quantile": QUANTILE,
        "thresholds": {name: results[name]["thresholds"] for name in ("shift", "phys")},
        "verdicts": decision,
        "descriptive": {name: results[name]["descriptive"] for name in ("shift", "phys")},
        "inputs": {
            "readouts": digests["readouts"],
            "knn_selection": digests["knn"],
            "train_shards": results["knn"]["train_shards"],
            "shift002_collection": results["shift"]["inputs"]["collection"],
            "latent_files": {
                "shift002": len(results["shift"]["latent_sha256"]),
                "phys002": len(results["phys"]["latent_sha256"]),
            },
        },
        "outputs": {name: sha256_file(root / f"{name}.json") for name in ("knn", "shift", "phys")},
        "sources": _sources(),
        "runtime": runtime_info(),
    }
    DRAFT.write_text(draft(results, checks, decision))
    write_json(MANIFEST, payload)
    return {
        "reproduction": checks["passed"],
        **{k: decision[k] for k in decision if k != "contrasts"},
    }


STAGES = {"knn": knn, "shift": shift, "phys": phys, "report": report}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "all"])
    stage = parser.parse_args(argv).stage
    ROOT.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch(ROOT)
    for name in STAGES if stage == "all" else [stage]:
        try:
            result = watch.run(name, STAGES[name])
        except GateFailure as error:
            print(json.dumps({"stage": name, "failed": str(error)}))
            raise SystemExit(1) from error
        print(json.dumps({"stage": name, "result": result}, default=str))


if __name__ == "__main__":
    main()
