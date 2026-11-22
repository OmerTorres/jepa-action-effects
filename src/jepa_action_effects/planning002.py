"""H3-PLAN-002: latent-planning regret with agent-matched goal images.

    python -m jepa_action_effects.planning002 audit
    python -m jepa_action_effects.planning002 collect   # fresh families, simulator only
    python -m jepa_action_effects.planning002 develop   # PHYS-002 development families
    python -m jepa_action_effects.planning002 seal      # needs passing development checks
    python -m jepa_action_effects.planning002 confirm   # opens reserved families once
    python -m jepa_action_effects.planning002 report

H3-PLAN-001 failed its latent-cost check: a start-pose agent in the goal image made the
cost reward `hold`. Each candidate now gets its own goal image, with its realized final
agent position and block angle and the block at the grid goal (D025). Rules are
`planning_metrics.apply_planning_criteria`, unchanged from D024.
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
from jepa_action_effects.planning002_metrics import choice_histogram, family_record
from jepa_action_effects.planning_regret import _cohort, _summaries, _table

ROOT = Path("outputs/planning002")
PHYS = Path("outputs/phys002")
PLAN1 = Path("outputs/planning001")
MASTER_SEED = 20261050
RESERVED = {"challenge": 100, "ordinary": 50}
BOOTSTRAP_SEED = MASTER_SEED + 7


def _sources() -> dict[str, str]:
    here = Path(__file__)
    return source_hashes(
        {
            "runner": here,
            "metrics": here.with_name("planning002_metrics.py"),
            "planning_metrics": here.with_name("planning_metrics.py"),
            "planning_regret": here.with_name("planning_regret.py"),
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


def _prior_seeds() -> list[int]:
    """All seeds forbidden for H3-PLAN-001 plus its (unopened) reserved families."""
    seeds = set(json.loads((PLAN1 / "forbidden-seeds.json").read_text())["seeds"])
    manifest = collection_manifest(PLAN1 / "reserved")
    for record in manifest["family_files"].values():
        seeds.add(int(load_family(PLAN1 / "reserved" / record["path"])["metadata"]["seed"]))
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
            raise ValueError("H3-PLAN-002 source changed after audit; start a new version")
        return old
    seeds = _prior_seeds()
    config = {
        "schema": "planning002-v1",
        "master_seed": MASTER_SEED,
        "reserved_counts": RESERVED,
        "development_ids": sealed(PHYS / "config.json")["development_ids"],
        "goal_image": (
            "per candidate: agent at the candidate's realized final position, block at the "
            "grid goal with the candidate's realized final angle; render-only setter; a goal "
            "counts only if all six images are geometry-valid"
        ),
        "cost": "L2 distance between final projected latent and its candidate's goal latent",
        "forbidden_reset_seed_count": len(seeds),
        "forbidden_reset_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "criteria": "planning_metrics.apply_planning_criteria (as D024)",
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


def matched_goal_images(final_states: np.ndarray, goals: list) -> tuple[np.ndarray, list[bool]]:
    """(G,C,224,224,3) images; final_states is (C,5) = agent xy, block xy, angle."""
    from jepa_action_effects.readout_scale import _new_env, _place

    env = _new_env()
    images, valid = [], []
    try:
        for goal in goals:
            row, ok = [], True
            for state in final_states:
                pose = np.array([state[0], state[1], goal[0], goal[1], state[4]], dtype=np.float64)
                ok = bool(_place(env, pose)) and ok
                row.append(env.render().copy())
            images.append(np.stack(row))
            valid.append(ok)
    finally:
        env.close()
    return np.stack(images), valid


def score_family(identifier: str, family: dict, goals: list, model, scaler) -> tuple[dict, dict]:
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    real = encode_frames(model, frames)["projected"].astype(np.float64)
    predicted = predict_family(model, scaler, family).astype(np.float64)
    if real.shape != (15, 192) or predicted.shape != (6, 2, 192):
        raise ValueError(f"Unexpected embedding alignment for {identifier}")
    final_states = family["future_states"][:, -1]
    images, valid = matched_goal_images(final_states, goals)
    goal_latents = (
        encode_frames(model, images.reshape(-1, 224, 224, 3))["projected"]
        .astype(np.float64)
        .reshape(len(goals), 6, 192)
    )
    record = family_record(
        identifier,
        _cohort(identifier),
        final_states[:, 2:4],
        goals,
        valid,
        predicted[:, -1],
        real[3:].reshape(6, 2, 192)[:, -1],
        goal_latents,
    )
    return record, {
        "goal_latents": goal_latents.astype(np.float32),
        "goal_valid": np.asarray(valid),
    }


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


def _phase_result(records: list[dict], fingerprint: dict) -> dict[str, Any]:
    by_cohort = {
        cohort: [r for r in records if r["cohort"] == cohort]
        for cohort in ("challenge", "ordinary")
    }
    return {
        "status": "complete",
        "fingerprint": fingerprint,
        "summary": _summaries(records, BOOTSTRAP_SEED),
        "choices": {
            cohort: {
                chooser: choice_histogram(rows, chooser) for chooser in ("jepa", "real_latent")
            }
            for cohort, rows in by_cohort.items()
        },
    }


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
    result = {"phase": "development (exploratory; PHYS-002 development families)"}
    result.update(_phase_result(records, fingerprint))
    seal(root / "develop.json", result)
    return result


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
    result = _phase_result(records, fingerprint)
    seal(root / "confirm.json", result)
    return result


def report(root: Path = ROOT) -> dict[str, Any]:
    def maybe(name: str):
        path = root / name
        return sealed(path) if path.exists() else None

    developed, confirmed = maybe("develop.json"), maybe("confirm.json")
    summary = {
        "task": "H3-PLAN-002",
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
    write_json(MANIFESTS / "planning002-runtime.json", summary)
    lines = ["# H3-PLAN-002 result (auto-generated draft)", ""]
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
        lines += [
            f"Goals: `{block['summary']['goals']}`",
            "",
            f"Choice histograms (candidate index): `{block['choices']}`",
            "",
        ]
    Path("docs/H3-PLAN-002_RESULT_DRAFT.md").write_text("\n".join(lines) + "\n")
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
