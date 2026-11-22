"""PHYS-002: frozen predictor against the real-frame readout floor, plus shrinkage.

    python -m jepa_action_effects.phys002 audit [--in-support-push 0.35]
    python -m jepa_action_effects.phys002 collect   # fresh reserved families, simulator only
    python -m jepa_action_effects.phys002 develop   # 300 READOUT development families
    python -m jepa_action_effects.phys002 seal      # freeze code, config, readout, criteria
    python -m jepa_action_effects.phys002 confirm   # score reserved families exactly once
    python -m jepa_action_effects.phys002 report

The same frozen READOUT-002 block probe decodes real and predicted futures of the same
branches. Displacements are measured from the decoded current frame, so a constant
readout bias cancels. Decision rules (D021) live in `next_metrics.apply_criteria` and
are sealed with the code before any reserved family is scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    FRESH,
    MANIFESTS,
    PILOT,
    Stopwatch,
    collection_manifest,
    development_ids,
    encode_frames,
    family_path,
    load_family,
    load_model,
    predict_family,
    runtime_info,
    seal,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.next_metrics import (
    FLOOR_RATIO_LIMIT,
    boundary_pairs,
    family_displacements,
    summarize_phys,
)
from jepa_action_effects.pilot_collection import PilotConfig, sha256_file
from jepa_action_effects.pilot_collection import collect as collect_families

ROOT = Path("outputs/phys002")
READOUT2 = Path("outputs/readout002")
DIST = Path("outputs/dist001")
MASTER_SEED = 20261010
RESERVED = {"challenge": 100, "ordinary": 50}
VARIANT_CHALLENGE = 60
BOOTSTRAP = 2000


def _sources() -> dict[str, str]:
    here = Path(__file__)
    return source_hashes(
        {
            "runner": here,
            "metrics": here.with_name("next_metrics.py"),
            "common": here.with_name("next_common.py"),
            "readout": here.with_name("readout_scale.py"),
            "inference": here.with_name("model_inference.py"),
            "collection": here.with_name("pilot_collection.py"),
            "replay": here.with_name("simulator_replay.py"),
            "uv_lock": Path("uv.lock"),
        }
    )


def _roots() -> tuple[dict[str, Path], dict[str, dict]]:
    roots = {"pilot": PILOT, "new": FRESH}
    return roots, {key: collection_manifest(path) for key, path in roots.items()}


def _cohort(identifier: str) -> str:
    return "challenge" if identifier.split(":", 1)[1].startswith("challenge") else "ordinary"


def _reserved_config(seeds: list[int], push: float | None = None) -> PilotConfig:
    if push is None:
        return PilotConfig(
            master_seed=MASTER_SEED,
            challenge_target=RESERVED["challenge"],
            ordinary_target=RESERVED["ordinary"],
            challenge_attempt_cap=2000,
            ordinary_attempt_cap=1000,
            challenge_split_counts=(0, 0, RESERVED["challenge"]),
            ordinary_split_counts=(0, 0, RESERVED["ordinary"]),
            horizon_policy="fixed_two",
            forbidden_reset_seeds=tuple(sorted(seeds)),
        )
    return PilotConfig(
        master_seed=MASTER_SEED + 1,
        challenge_target=VARIANT_CHALLENGE,
        ordinary_target=0,
        challenge_attempt_cap=4000,
        ordinary_attempt_cap=0,
        challenge_push_magnitude=float(push),
        challenge_split_counts=(0, 0, VARIANT_CHALLENGE),
        ordinary_split_counts=(0, 0, 0),
        horizon_policy="fixed_two",
        forbidden_reset_seeds=tuple(sorted(seeds)),
    )


def audit(root: Path = ROOT, in_support_push: float | None = None) -> dict[str, Any]:
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources():
            raise ValueError("PHYS-002 source changed after audit; start a new version")
        return old
    identity = DIST / "identity.json"
    if identity.exists() and json.loads(identity.read_text()).get("rendering_mismatch_flag"):
        raise RuntimeError("DIST-001 flagged a rendering mismatch: pause and re-scope (NEXT-001)")
    selection = sealed(READOUT2 / "selection.json")
    roots, manifests = _roots()
    dev, excluded = development_ids(manifests["pilot"], manifests["new"])
    if len(dev) != 300 or set(dev) & set(excluded):
        raise ValueError("Development/opened-confirmation membership changed")
    seeds = sorted(
        {
            int(load_family(roots[source] / record["path"])["metadata"]["seed"])
            for source, manifest in manifests.items()
            for record in manifest["family_files"].values()
        }
    )
    rng = np.random.default_rng(MASTER_SEED)
    verification = []
    for cohort in ("challenge", "ordinary"):
        pool = sorted(key for key in dev if _cohort(key) == cohort)
        verification.extend(sorted(rng.choice(pool, size=6, replace=False).tolist()))
    config = {
        "schema": "phys002-v1",
        "master_seed": MASTER_SEED,
        "development_ids": dev,
        "excluded_opened_confirmation_ids": excluded,
        "verification_ids": verification,
        "reserved_counts": RESERVED,
        "in_support_variant": None
        if in_support_push is None
        else {"push_magnitude": in_support_push, "challenge": VARIANT_CHALLENGE},
        "forbidden_reset_seed_count": len(seeds),
        "forbidden_reset_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
        "readout_selection": selection,
        "readout_selection_sha256": sha256_file(READOUT2 / "selection.json"),
        "horizon_native_steps": 2,
        "delta": 5.12,
        "boundary_tolerance": 0.2,
        "floor_ratio_limit": FLOOR_RATIO_LIMIT,
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": MASTER_SEED + 7,
        "criteria": "next_metrics.apply_criteria (D021)",
    }
    write_json(root / "forbidden-seeds.json", {"seeds": seeds})
    result = {"status": "passed", "sources": _sources(), "config": config}
    result["config_sha256"] = seal(root / "config.json", config)
    seal(root / "audit.json", result)
    return result


def collect(root: Path = ROOT) -> dict[str, Any]:
    """Collect reserved families by simulator outcomes only; nothing is scored here."""
    config = sealed(root / "config.json")
    seeds = json.loads((root / "forbidden-seeds.json").read_text())["seeds"]
    main = collect_families(root / "reserved", _reserved_config(seeds))
    result = {"reserved": {"status": main["status"], "families": len(main["family_files"])}}
    variant = config["in_support_variant"]
    if variant is not None:
        used = seeds + [
            int(load_family(root / "reserved" / r["path"])["metadata"]["seed"])
            for r in main["family_files"].values()
        ]
        extra = collect_families(
            root / "reserved-insupport", _reserved_config(used, variant["push_magnitude"])
        )
        result["in_support"] = {"status": extra["status"], "families": len(extra["family_files"])}
    write_json(root / "collect.json", result)
    return result


def _goals(manifest: dict, family_id: str) -> list:
    return manifest.get("goals", {}).get(family_id, [])


def score_family(
    identifier: str, family: dict, goals: list, model, scaler, probe
) -> tuple[dict, dict[str, np.ndarray]]:
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    real = encode_frames(model, frames)["projected"]
    predicted = predict_family(model, scaler, family)
    if real.shape != (15, 192) or predicted.shape != (6, 2, 192):
        raise ValueError(f"Unexpected embedding alignment for {identifier}")
    ids = family["metadata"]["candidate_ids"]
    record = family_displacements(
        probe(predicted[:, -1].astype(np.float64)),
        probe(real[3:].reshape(6, 2, 192)[:, -1].astype(np.float64)),
        probe(real[2:3].astype(np.float64))[0],
        family["future_states"][:, -1, 2:4],
        family["history_states"][-1, 2:4],
        family["contacts"].astype(bool).tolist(),
        boundary_pairs(family["candidate_controls"], ids),
        goals,
    )
    return {"id": identifier, "cohort": _cohort(identifier), **record}, {
        "real": real,
        "predicted": predicted,
    }


def _verify(family: dict, predicted: np.ndarray, model, scaler) -> float:
    from jepa_action_effects.latent_audit import _verification

    return _verification(family, predicted, model, scaler)


def _score_all(
    root: Path,
    phase: str,
    items: list[tuple[str, Path, list]],
    verification: set[str],
    model,
    scaler,
    probe,
) -> list[dict]:
    records = []
    for identifier, path, goals in items:
        target = root / phase / "families" / f"{identifier.replace(':', '__')}.json"
        if target.exists():
            records.append(sealed(target))
            continue
        family = load_family(path)
        record, arrays = score_family(identifier, family, goals, model, scaler, probe)
        record["input_sha256"] = sha256_file(path)
        record["verification_max_error"] = (
            _verify(family, arrays["predicted"], model, scaler)
            if identifier in verification
            else None
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".npz"), **arrays)
        seal(target, record)
        records.append(record)
    return records


def develop(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.readout_scale import load_block_probe

    config = sealed(root / "config.json")
    if sealed(root / "audit.json")["sources"] != _sources():
        raise ValueError("Source changed since audit")
    model, scaler, fingerprint = load_model()
    probe, selection = load_block_probe(READOUT2)
    if selection != config["readout_selection"]:
        raise ValueError("READOUT-002 selection changed since audit")
    roots, manifests = _roots()
    items = []
    for identifier in config["development_ids"]:
        source, family_id = identifier.split(":", 1)
        items.append(
            (
                identifier,
                family_path(identifier, roots, manifests),
                _goals(manifests[source], family_id),
            )
        )
    records = _score_all(
        root, "develop", items, set(config["verification_ids"]), model, scaler, probe
    )
    summary = _summaries(records, config["bootstrap_seed"])
    result = {
        "status": "complete",
        "phase": "development (exploratory; READOUT-001 families already used)",
        "families": len(records),
        "fingerprint": fingerprint,
        "verification_max_error": max(
            r["verification_max_error"] for r in records if r["verification_max_error"] is not None
        ),
        "summary": summary,
    }
    seal(root / "develop.json", result)
    return result


def _summaries(records: list[dict], seed: int) -> dict[str, Any]:
    return {
        "pooled": summarize_phys(records, seed, BOOTSTRAP),
        "challenge": summarize_phys(
            [r for r in records if r["cohort"] == "challenge"], seed, BOOTSTRAP
        ),
        "ordinary": summarize_phys(
            [r for r in records if r["cohort"] == "ordinary"], seed, BOOTSTRAP
        ),
    }


def freeze(root: Path = ROOT) -> dict[str, Any]:
    developed = sealed(root / "develop.json")
    if developed["status"] != "complete":
        raise ValueError("Development must complete before sealing")
    collected = json.loads((root / "collect.json").read_text())
    if collected["reserved"]["status"] != "complete":
        raise ValueError("Reserved collection incomplete")
    reserved = collection_manifest(root / "reserved")
    body = {
        "sources": _sources(),
        "config_sha256": sha256_file(root / "config.json"),
        "audit_sha256": sha256_file(root / "audit.json"),
        "develop_sha256": sha256_file(root / "develop.json"),
        "readout_selection_sha256": sha256_file(READOUT2 / "selection.json"),
        "reserved_manifest_sha256": sha256_file(root / "reserved" / "collection-manifest.json"),
        "reserved_ids": sorted(reserved["family_files"]),
        "decision_rules": "next_metrics.apply_criteria; floor ratio < 0.25 gates every claim",
    }
    target = root / "seal.json"
    if target.exists():
        if sealed(target) != body:
            raise ValueError("Existing seal differs; create a new protocol version")
        return body
    seal(target, body)
    return body


def confirm(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.readout_scale import load_block_probe

    frozen = sealed(root / "seal.json")
    if frozen["sources"] != _sources():
        raise ValueError("Source changed after seal; confirmation refused")
    config = sealed(root / "config.json")
    marker = root / "confirmation-opened.json"
    seal_hash = sha256_file(root / "seal.json")
    if marker.exists():
        if json.loads(marker.read_text())["seal_sha256"] != seal_hash:
            raise ValueError("Confirmation was opened under a different seal")
        if (root / "confirm.json").exists():
            return sealed(root / "confirm.json")
    else:
        write_json(marker, {"seal_sha256": seal_hash, "note": "reserved families opened"})
    model, scaler, fingerprint = load_model()
    probe, _ = load_block_probe(READOUT2)
    result: dict[str, Any] = {"status": "complete", "fingerprint": fingerprint}
    groups = [("reserved", root / "reserved")]
    if config["in_support_variant"] is not None and (root / "reserved-insupport").exists():
        groups.append(("in_support", root / "reserved-insupport"))
    for name, directory in groups:
        try:
            manifest = collection_manifest(directory)
        except ValueError as error:
            result[name] = {"status": "unavailable", "reason": str(error)}
            continue
        ids = sorted(manifest["family_files"])
        verification = set()
        for cohort in ("challenge", "ordinary"):
            pool = [key for key in ids if key.startswith(cohort)]
            verification.update(pool[: min(6, len(pool))])
        items = [
            (
                f"{name}:{key}",
                directory / manifest["family_files"][key]["path"],
                _goals(manifest, key),
            )
            for key in ids
        ]
        records = _score_all(
            root,
            f"confirm-{name}",
            items,
            {f"{name}:{key}" for key in verification},
            model,
            scaler,
            probe,
        )
        summary = _summaries(records, config["bootstrap_seed"])
        result[name] = {
            "families": len(records),
            "verification_max_error": max(
                (
                    r["verification_max_error"]
                    for r in records
                    if r["verification_max_error"] is not None
                ),
                default=None,
            ),
            "summary": summary,
            "criteria_challenge": summary["challenge"]["criteria"],
        }
    seal(root / "confirm.json", result)
    return result


def _table(summary: dict, keys: tuple[str, ...]) -> list[str]:
    lines = [
        "| Quantity | Estimate | 95% family bootstrap CI | Families |",
        "| --- | ---: | --- | ---: |",
    ]
    for key in keys:
        cell = summary[key]
        estimate = "n/a" if cell["estimate"] is None else f"{cell['estimate']:.4f}"
        interval = "n/a" if cell["ci95"] is None else "[{:.4f}, {:.4f}]".format(*cell["ci95"])
        lines.append(f"| {key} | {estimate} | {interval} | {cell['families']} |")
    return lines


KEY_ROWS = (
    "floor_ratio",
    "D_boundary",
    "D_exactly_one",
    "D_both",
    "h1_contrast",
    "slope_pred",
    "slope_real",
    "slope_pred_minus_real",
    "slope_pred_contact",
    "slope_pred_no_contact",
    "endpoint_err_pred",
    "endpoint_err_real",
    "regret_jepa",
    "regret_real_frame",
    "regret_random",
)


def report(root: Path = ROOT) -> dict[str, Any]:
    def maybe(name: str):
        path = root / name
        return sealed(path) if path.exists() else None

    developed, confirmed = maybe("develop.json"), maybe("confirm.json")
    summary = {
        "task": "PHYS-002",
        "runtime": runtime_info(),
        "sources": _sources(),
        "config_sha256": sha256_file(root / "config.json"),
        "seal": maybe("seal.json"),
        "develop": developed,
        "confirm": confirmed,
        "stage_seconds": json.loads((root / "runtime.json").read_text())["stages"]
        if (root / "runtime.json").exists()
        else None,
    }
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "phys002-runtime.json", summary)
    lines = ["# PHYS-002 result (auto-generated draft)", ""]
    for label, block in (
        ("Development (exploratory)", developed),
        ("Reserved confirmation", confirmed),
    ):
        if block is None:
            lines += [f"## {label}", "", "Not run.", ""]
            continue
        source = block["summary"] if "summary" in block else block["reserved"]["summary"]
        lines += [f"## {label}: challenge cohort", ""]
        lines += _table(source["challenge"], KEY_ROWS)
        lines += ["", f"Criteria: `{source['challenge']['criteria']}`", ""]
        lines += [f"## {label}: pooled slope and endpoint rows", ""]
        lines += _table(source["pooled"], ("slope_pred", "slope_real", "endpoint_err_pred"))
        lines += [""]
    (root / "RESULT.md").write_text("\n".join(lines) + "\n")
    Path("docs/PHYS-002_RESULT_DRAFT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("audit", "collect", "develop", "seal", "confirm", "report")
    )
    parser.add_argument("--in-support-push", type=float, default=None)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch(ROOT)
    if args.stage == "audit":
        result = watch.run("audit", audit, ROOT, args.in_support_push)
    else:
        stages = {
            "collect": collect,
            "develop": develop,
            "seal": freeze,
            "confirm": confirm,
            "report": report,
        }
        result = watch.run(args.stage, stages[args.stage], ROOT)
    print(json.dumps(result, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
