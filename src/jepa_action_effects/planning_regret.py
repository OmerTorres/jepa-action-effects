"""H3-PLAN-001: does prediction error cost reward under the model's own planning objective?

    python -m jepa_action_effects.planning_regret audit
    python -m jepa_action_effects.planning_regret collect   # fresh families, simulator only
    python -m jepa_action_effects.planning_regret develop   # PHYS-002 development families
    python -m jepa_action_effects.planning_regret seal
    python -m jepa_action_effects.planning_regret confirm   # opens reserved families once
    python -m jepa_action_effects.planning_regret report

Each candidate is scored by the distance from its final latent to the latent of a goal
image. The goal image is rendered with the validated render-only pose setter: block at
the grid goal, start block angle, agent at its start position. The predicted-latent
chooser is compared with the same chooser on real future latents, and with random
choice. Rules are in `planning_metrics.apply_planning_criteria`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    MANIFESTS,
    Stopwatch,
    collection_manifest,
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
from jepa_action_effects.pilot_collection import PilotConfig, sha256_file
from jepa_action_effects.planning_metrics import family_record, summarize_planning

ROOT = Path("outputs/planning001")
PHYS = Path("outputs/phys002")
SHIFT2 = Path("outputs/shift002")
MASTER_SEED = 20261030
RESERVED = {"challenge": 100, "ordinary": 50}
BOOTSTRAP = 2000


def _sources() -> dict[str, str]:
    here = Path(__file__)
    return source_hashes(
        {
            "runner": here,
            "metrics": here.with_name("planning_metrics.py"),
            "next_metrics": here.with_name("next_metrics.py"),
            "common": here.with_name("next_common.py"),
            "readout": here.with_name("readout_scale.py"),
            "gallery": here.with_name("gallery_render.py"),
            "inference": here.with_name("model_inference.py"),
            "collection": here.with_name("pilot_collection.py"),
            "replay": here.with_name("simulator_replay.py"),
            "uv_lock": Path("uv.lock"),
        }
    )


def _cohort(identifier: str) -> str:
    return "challenge" if "challenge" in identifier else "ordinary"


def _prior_seeds() -> list[int]:
    """Every reset seed used by PILOT/READOUT, PHYS-002, SHIFT-001 and SHIFT-002."""
    seeds = set(json.loads((SHIFT2 / "forbidden-seeds.json").read_text())["seeds"])
    collection = sealed(SHIFT2 / "collection.json")
    for entry in collection["families"].values():
        if entry["state"] == "challenge":
            seeds.add(int(load_family(SHIFT2 / entry["path"])["metadata"]["seed"]))
    return sorted(seeds)


def _reserved_config(seeds: list[int]) -> PilotConfig:
    return PilotConfig(
        master_seed=MASTER_SEED,
        challenge_target=RESERVED["challenge"],
        ordinary_target=RESERVED["ordinary"],
        challenge_attempt_cap=2000,
        ordinary_attempt_cap=1000,
        challenge_split_counts=(0, 0, RESERVED["challenge"]),
        ordinary_split_counts=(0, 0, RESERVED["ordinary"]),
        horizon_policy="fixed_two",
        forbidden_reset_seeds=tuple(seeds),
    )


def audit(root: Path = ROOT) -> dict[str, Any]:
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources():
            raise ValueError("H3-PLAN-001 source changed after audit; start a new version")
        return old
    phys_config = sealed(PHYS / "config.json")
    seeds = _prior_seeds()
    config = {
        "schema": "planning001-v1",
        "master_seed": MASTER_SEED,
        "reserved_counts": RESERVED,
        "development_ids": phys_config["development_ids"],
        "goal_image": "block at grid goal, start block angle, agent at start; render-only setter",
        "cost": "L2 distance between final projected latent and goal-image latent",
        "forbidden_reset_seed_count": len(seeds),
        "forbidden_reset_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": MASTER_SEED + 7,
        "criteria": "planning_metrics.apply_planning_criteria",
    }
    write_json(root / "forbidden-seeds.json", {"seeds": seeds})
    result = {
        "status": "passed",
        "sources": _sources(),
        "config_sha256": seal(root / "config.json", config),
    }
    seal(root / "audit.json", result)
    return result


def collect(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.pilot_collection import collect as collect_families

    sealed(root / "config.json")
    seeds = json.loads((root / "forbidden-seeds.json").read_text())["seeds"]
    manifest = collect_families(root / "reserved", _reserved_config(seeds))
    result = {"status": manifest["status"], "families": len(manifest["family_files"])}
    write_json(root / "collect.json", result)
    return result


def goal_images(start: np.ndarray, goals: list) -> tuple[np.ndarray, list[bool]]:
    """Render one goal image per grid goal; `valid` is False on any geometry overlap."""
    from jepa_action_effects.readout_scale import _new_env, _place

    env = _new_env()
    frames, valid = [], []
    try:
        for goal in goals:
            pose = np.array([start[0], start[1], goal[0], goal[1], start[4]], dtype=np.float64)
            valid.append(bool(_place(env, pose)))
            frames.append(env.render().copy())
    finally:
        env.close()
    return np.stack(frames), valid


def score_family(identifier: str, family: dict, goals: list, model, scaler) -> tuple[dict, dict]:
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    real = encode_frames(model, frames)["projected"].astype(np.float64)
    predicted = predict_family(model, scaler, family).astype(np.float64)
    if real.shape != (15, 192) or predicted.shape != (6, 2, 192):
        raise ValueError(f"Unexpected embedding alignment for {identifier}")
    images, valid = goal_images(family["history_states"][-1], goals)
    goal_latents = encode_frames(model, images)["projected"].astype(np.float64)
    record = family_record(
        identifier,
        _cohort(identifier),
        family["future_states"][:, -1, 2:4],
        goals,
        valid,
        predicted[:, -1],
        real[3:].reshape(6, 2, 192)[:, -1],
        goal_latents,
    )
    return record, {"goal_latents": goal_latents, "goal_valid": np.asarray(valid)}


def _score_all(root: Path, phase: str, items: list[tuple[str, Path, list]], model, scaler):
    records = []
    for identifier, path, goals in items:
        target = root / phase / f"{identifier.replace(':', '__')}.json"
        if target.exists():
            records.append(sealed(target))
            continue
        family = load_family(path)
        record, arrays = score_family(identifier, family, goals, model, scaler)
        record["input_sha256"] = sha256_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".npz"), **arrays)
        seal(target, record)
        records.append(record)
    return records


def develop(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.phys002 import _goals, _roots

    config = sealed(root / "config.json")
    if sealed(root / "audit.json")["sources"] != _sources():
        raise ValueError("Source changed since audit")
    model, scaler, fingerprint = load_model()
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
    records = _score_all(root, "develop", items, model, scaler)
    result = {
        "status": "complete",
        "phase": "development (exploratory; PHYS-002 development families)",
        "fingerprint": fingerprint,
        "summary": _summaries(records, config["bootstrap_seed"]),
    }
    seal(root / "develop.json", result)
    return result


def _summaries(records: list[dict], seed: int) -> dict[str, Any]:
    return {
        "pooled": summarize_planning(records, seed, BOOTSTRAP),
        "challenge": summarize_planning(
            [r for r in records if r["cohort"] == "challenge"], seed, BOOTSTRAP
        ),
        "ordinary": summarize_planning(
            [r for r in records if r["cohort"] == "ordinary"], seed, BOOTSTRAP
        ),
        "goals": {
            "valid": int(sum(r["valid_goals"] for r in records)),
            "informative": int(sum(r["informative_goals"] for r in records)),
            "families_without_valid_goal": int(sum(r["all"] is None for r in records)),
        },
    }


def freeze(root: Path = ROOT) -> dict[str, Any]:
    developed = sealed(root / "develop.json")
    if not developed["summary"]["pooled"]["criteria"]["passed"]:
        raise ValueError("Development checks failed; redesign before opening reserved families")
    collected = json.loads((root / "collect.json").read_text())
    if collected["status"] != "complete":
        raise ValueError("Reserved collection incomplete")
    reserved = collection_manifest(root / "reserved")
    body = {
        "sources": _sources(),
        "config_sha256": sha256_file(root / "config.json"),
        "audit_sha256": sha256_file(root / "audit.json"),
        "develop_sha256": sha256_file(root / "develop.json"),
        "reserved_manifest_sha256": sha256_file(root / "reserved" / "collection-manifest.json"),
        "reserved_ids": sorted(reserved["family_files"]),
        "decision_rules": "planning_metrics.apply_planning_criteria on pooled reserved families",
    }
    target = root / "seal.json"
    if target.exists():
        if sealed(target) != body:
            raise ValueError("Existing seal differs; create a new protocol version")
        return body
    seal(target, body)
    return body


def confirm(root: Path = ROOT) -> dict[str, Any]:
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
    manifest = collection_manifest(root / "reserved")
    model, scaler, fingerprint = load_model()
    items = [
        (
            f"reserved:{key}",
            root / "reserved" / manifest["family_files"][key]["path"],
            manifest["goals"][key],
        )
        for key in frozen["reserved_ids"]
    ]
    records = _score_all(root, "confirm", items, model, scaler)
    result = {
        "status": "complete",
        "fingerprint": fingerprint,
        "summary": _summaries(records, config["bootstrap_seed"]),
    }
    seal(root / "confirm.json", result)
    return result


ROWS = (
    "regret_jepa_all",
    "regret_real_latent_all",
    "regret_random_all",
    "jepa_minus_real_all",
    "real_minus_random_all",
    "jepa_minus_random_all",
    "regret_jepa_informative",
    "regret_real_latent_informative",
    "regret_random_informative",
    "jepa_minus_real_informative",
)


def _table(summary: dict) -> list[str]:
    lines = ["| Quantity | Estimate | 95% family bootstrap CI |", "| --- | ---: | --- |"]
    for key in ROWS:
        cell = summary[key]
        estimate = "n/a" if cell["estimate"] is None else f"{cell['estimate']:.3f}"
        interval = "n/a" if cell["ci95"] is None else "[{:.3f}, {:.3f}]".format(*cell["ci95"])
        lines.append(f"| {key} | {estimate} | {interval} |")
    return lines


def report(root: Path = ROOT) -> dict[str, Any]:
    def maybe(name: str):
        path = root / name
        return sealed(path) if path.exists() else None

    developed, confirmed = maybe("develop.json"), maybe("confirm.json")
    summary = {
        "task": "H3-PLAN-001",
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
    write_json(MANIFESTS / "planning001-runtime.json", summary)
    lines = ["# H3-PLAN-001 result (auto-generated draft)", ""]
    for label, block in (
        ("Development (exploratory)", developed),
        ("Reserved confirmation", confirmed),
    ):
        lines += [f"## {label}", ""]
        if block is None:
            lines += ["Not run.", ""]
            continue
        for cohort in ("pooled", "challenge", "ordinary"):
            cell = block["summary"][cohort]
            lines += [f"### {cohort} ({cell['families']} families)", ""]
            lines += _table(cell)
            lines += ["", f"Criteria: `{cell['criteria']}`", ""]
        lines += [f"Goals: `{block['summary']['goals']}`", ""]
    Path("docs/H3-PLAN-001_RESULT_DRAFT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("audit", "collect", "develop", "seal", "confirm", "report")
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    stages = {
        "audit": audit,
        "collect": collect,
        "develop": develop,
        "seal": freeze,
        "confirm": confirm,
        "report": report,
    }
    result = Stopwatch(ROOT).run(args.stage, stages[args.stage], ROOT)
    print(json.dumps(result, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
