"""SHIFT-002: does predictor shrinkage grow with push magnitude?

    python -m jepa_action_effects.shift002 audit
    python -m jepa_action_effects.shift002 collect   # simulator/archive only, no model
    python -m jepa_action_effects.shift002 develop
    python -m jepa_action_effects.shift002 seal      # needs passing manipulation checks
    python -m jepa_action_effects.shift002 confirm   # opens reserved families once
    python -m jepa_action_effects.shift002 report

SHIFT-001 development suggested push magnitude, not within-block structure, drives the
PHYS-002 shrinkage. SHIFT-001 could not be sealed, and its reserved families stay
unopened. This version uses fresh seeds and archive episodes disjoint from every
earlier stage. Rules are in `shift002_metrics.apply_dose_criteria`.
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
    load_family,
    load_model,
    runtime_info,
    seal,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.shift001 import (
    _archive_window,
    _dist_episodes,
    _exploratory_episodes,
    _h5,
    _phase_ids,
    _residual_pool,
    _score,
    safe_goal,
)
from jepa_action_effects.shift001_metrics import (
    DELTA,
    constant_version,
    is_rest_window,
    varying_version,
)
from jepa_action_effects.shift002_metrics import (
    ARCHIVE_SCALES,
    CHALLENGE_LEVELS,
    CHALLENGE_VARYING,
    archive_label,
    challenge_label,
    summarize_dose,
)

ROOT = Path("outputs/shift002")
SHIFT1 = Path("outputs/shift001")
READOUT2 = Path("outputs/readout002")
MASTER_SEED = 20261040
TARGETS = {
    "challenge": {"development": 60, "reserved": 120},
    "archive": {"development": 60, "reserved": 120},
}
ATTEMPT_CAPS = {"challenge": 4000, "archive": 60000}
REST_LIMIT = 1.0
IDENTITY_POSITION = 0.5
IDENTITY_FRAME = 1.0
OWN_FUTURE_LIMIT = 1.0
MAX_OWN_CONTROL = 1.0 / max(ARCHIVE_SCALES)  # scaled controls stay within [-1, 1]
BOOTSTRAP = 2000


def _sources() -> dict[str, str]:
    here = Path(__file__)
    return source_hashes(
        {
            "runner": here,
            "metrics": here.with_name("shift002_metrics.py"),
            "shift001": here.with_name("shift001.py"),
            "shift001_metrics": here.with_name("shift001_metrics.py"),
            "common": here.with_name("next_common.py"),
            "readout": here.with_name("readout_scale.py"),
            "inference": here.with_name("model_inference.py"),
            "collection": here.with_name("pilot_collection.py"),
            "replay": here.with_name("simulator_replay.py"),
            "uv_lock": Path("uv.lock"),
        }
    )


def _shift001_exclusions() -> tuple[list[int], list[int]]:
    """Reset seeds and archive episodes used by SHIFT-001 (both collections are identical)."""
    config = sealed(SHIFT1 / "config.json")
    collection = sealed(SHIFT1 / "collection.json")
    seeds = set(json.loads((SHIFT1 / "forbidden-seeds.json").read_text())["seeds"])
    episodes = set(config["excluded_archive_episodes"])
    for entry in collection["families"].values():
        metadata = load_family(SHIFT1 / entry["path"])["metadata"]
        if entry["state"] == "challenge":
            seeds.add(int(metadata["seed"]))
        else:
            episodes.add(int(metadata["episode"]))
            episodes.update(int(episode) for episode, _ in metadata["borrowed"])
    return sorted(seeds), sorted(episodes)


def audit(root: Path = ROOT) -> dict[str, Any]:
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources():
            raise ValueError("SHIFT-002 source changed after audit; start a new version")
        return old
    selection = sealed(READOUT2 / "selection.json")
    seeds, episodes = _shift001_exclusions()
    with _h5() as h5:
        episodes = sorted({*episodes, *_dist_episodes(), *_exploratory_episodes(h5)})
    config = {
        "schema": "shift002-v1",
        "master_seed": MASTER_SEED,
        "targets": TARGETS,
        "attempt_caps": ATTEMPT_CAPS,
        "challenge_levels": CHALLENGE_LEVELS,
        "challenge_varying": CHALLENGE_VARYING,
        "archive_scales": ARCHIVE_SCALES,
        "max_own_control": MAX_OWN_CONTROL,
        "rest_limit": REST_LIMIT,
        "identity_limits": {
            "position": IDENTITY_POSITION,
            "frame_mean_abs": IDENTITY_FRAME,
            "own_future": OWN_FUTURE_LIMIT,
        },
        "delta": DELTA,
        "forbidden_reset_seed_count": len(seeds),
        "forbidden_reset_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
        "excluded_archive_episodes": episodes,
        "readout_selection": selection,
        "readout_selection_sha256": sha256_file(READOUT2 / "selection.json"),
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": MASTER_SEED + 7,
        "criteria": "shift002_metrics.apply_dose_criteria",
    }
    write_json(root / "forbidden-seeds.json", {"seeds": seeds})
    result = {
        "status": "passed",
        "sources": _sources(),
        "config_sha256": seal(root / "config.json", config),
    }
    seal(root / "audit.json", result)
    return result


def _moved(replay) -> dict[str, bool]:
    from jepa_action_effects.pilot_collection import _states_from_steps

    moved = {}
    for key, branch in replay.branches.items():
        history, future, _ = _states_from_steps(branch)
        moved[key] = bool(np.linalg.norm(future[-1, 2:4] - history[-1, 2:4]) > DELTA)
    return moved


def _count(log: dict, reason: str) -> None:
    log[reason] = log.get(reason, 0) + 1


def _collect_challenge(root: Path, seeds: set[int], residuals, magnitudes, log: dict) -> list[str]:
    from jepa_action_effects.pilot_collection import (
        PilotConfig,
        _constant,
        _rngs,
        propose_challenge,
        save_family,
    )
    from jepa_action_effects.simulator_replay import Candidate, run_family

    config = PilotConfig(master_seed=MASTER_SEED)
    rngs = _rngs(config)
    residual_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 1]))
    target = sum(TARGETS["challenge"].values())
    accepted: list[str] = []
    attempt = 0
    for attempt in range(ATTEMPT_CAPS["challenge"]):
        if len(accepted) >= target:
            break
        proposal, reason = propose_challenge(config, attempt, rngs["proposals"])
        if proposal is None:
            _count(log, reason)
            continue
        recipe, _, metadata = proposal
        if recipe.seed in seeds:
            _count(log, "reset_seed_overlap")
            continue
        toward = np.asarray(metadata["approach_toward"])
        spec: dict[str, dict] = {}
        candidates = []
        for level in CHALLENGE_LEVELS:
            label = challenge_label(level)
            constant = _constant(toward * level).astype(np.float64)
            spec[f"{label}-constant"] = {"structure": "constant", "label": label, "level": level}
            candidates.append(Candidate(f"{label}-constant", constant))
            if level in CHALLENGE_VARYING:
                varying = varying_version(constant, residuals, magnitudes, residual_rng)
                spec[f"{label}-varying"] = {"structure": "varying", "label": label, "level": level}
                candidates.append(Candidate(f"{label}-varying", varying))
        order = [c.candidate_id for c in candidates]
        rngs["execution_order"].shuffle(order)
        try:
            replay = run_family(recipe, candidates, order)
        except RuntimeError:
            _count(log, "replay_error")
            continue
        if not all(branch.completed for branch in replay.branches.values()):
            _count(log, "incomplete")
            continue
        moved = _moved(replay)
        low = moved["m0.15-constant"] or moved["m0.30-constant"]
        if not (moved["m0.60-constant"] and low):
            _count(log, "high_or_low_push_static")
            continue
        family_id = f"challenge-{len(accepted):03d}"
        replay = type(replay)(family_id, replay.seed, replay.execution_order, replay.branches)
        save_family(
            root / "families" / f"{family_id}.npz",
            recipe,
            candidates,
            replay,
            {**metadata, "cohort": "challenge", "spec": spec, "moved": moved},
        )
        seeds.add(recipe.seed)
        accepted.append(family_id)
    log["challenge_attempts"] = attempt + 1
    return accepted


def _collect_archive(root: Path, excluded: set[int], log: dict) -> list[str]:
    from jepa_action_effects.pilot_collection import _states_from_steps, save_family
    from jepa_action_effects.simulator_replay import Candidate, ReplayRecipe, run_family

    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 2]))
    target = sum(TARGETS["archive"].values())
    accepted: list[str] = []
    with _h5() as h5:
        lengths = h5["ep_len"][:]
        available = lengths >= 21
        available[sorted(excluded)] = False
        attempt = 0
        for attempt in range(ATTEMPT_CAPS["archive"]):
            if len(accepted) >= target:
                break
            episode = int(rng.choice(np.flatnonzero(available)))
            t = int(rng.integers(10, int(lengths[episode]) - 10))
            window = _archive_window(h5, episode, t)
            if window is None:
                _count(log, "out_of_bounds")
                continue
            states, own = window["states"], window["controls"][10:20]
            if not is_rest_window(states[:11, 2:4], REST_LIMIT):
                _count(log, "block_not_at_rest")
                continue
            if np.linalg.norm(states[20, 2:4] - states[10, 2:4]) <= DELTA:
                _count(log, "own_future_static")
                continue
            if np.max(np.abs(own)) > MAX_OWN_CONTROL:
                _count(log, "own_future_too_large_to_scale")
                continue
            spec: dict[str, dict] = {}
            candidates = []
            for scale in ARCHIVE_SCALES:
                label = archive_label(scale)
                for structure, controls in (
                    ("constant", constant_version(own) * scale),
                    ("varying", own * scale),
                ):
                    key = f"{label}-{structure}"
                    spec[key] = {"structure": structure, "label": label, "level": scale}
                    candidates.append(Candidate(key, controls))
            start_state = states[0].tolist()
            recipe = ReplayRecipe(
                f"archive-{attempt:05d}",
                0,
                window["controls"][:10],
                {"state": start_state, "goal_state": safe_goal(start_state)},
            )
            order = [c.candidate_id for c in candidates]
            rng.shuffle(order)
            try:
                replay = run_family(recipe, candidates, order)
            except RuntimeError:
                _count(log, "replay_error")
                continue
            if not all(branch.completed for branch in replay.branches.values()):
                _count(log, "incomplete")
                continue
            reference = replay.branches["x1.0-varying"]
            history, future, _ = _states_from_steps(reference)
            position_error = float(
                np.max(
                    np.linalg.norm(
                        history[:, :4].reshape(3, 2, 2) - states[[0, 5, 10], :4].reshape(3, 2, 2),
                        axis=2,
                    )
                )
            )
            archive_frame = h5["pixels"][window["start"] + t].astype(np.int16)
            frame_error = float(
                np.abs(reference.history_frames[-1].astype(np.int16) - archive_frame).mean()
            )
            own_error = float(np.linalg.norm(future[-1, 2:4] - states[20, 2:4]))
            if (
                position_error > IDENTITY_POSITION
                or frame_error > IDENTITY_FRAME
                or own_error > OWN_FUTURE_LIMIT
            ):
                _count(log, "identity_failed")
                continue
            moved = _moved(replay)
            unit = moved["x1.0-constant"] or moved["x1.0-varying"]
            double = moved["x2.0-constant"] or moved["x2.0-varying"]
            if not (unit and double):
                _count(log, "unit_or_double_static")
                continue
            family_id = f"archive-{len(accepted):03d}"
            replay = type(replay)(family_id, replay.seed, replay.execution_order, replay.branches)
            metadata = {
                "cohort": "archive",
                "spec": spec,
                "moved": moved,
                "episode": episode,
                "t": t,
                "identity": {
                    "position": position_error,
                    "frame": frame_error,
                    "own_future": own_error,
                },
            }
            save_family(
                root / "families" / f"{family_id}.npz", recipe, candidates, replay, metadata
            )
            available[episode] = False
            accepted.append(family_id)
    log["archive_attempts"] = attempt + 1
    return accepted


def collect(root: Path = ROOT) -> dict[str, Any]:
    """Build every family from simulator/archive outcomes only; nothing is scored."""
    if (root / "collection.json").exists():
        return sealed(root / "collection.json")
    config = sealed(root / "config.json")
    seeds = set(json.loads((root / "forbidden-seeds.json").read_text())["seeds"])
    with _h5() as h5:
        residuals, magnitudes = _residual_pool(h5)
    log: dict[str, int] = {}
    ids = {
        "challenge": _collect_challenge(root, seeds, residuals, magnitudes, log),
        "archive": _collect_archive(root, set(config["excluded_archive_episodes"]), log),
    }
    split_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 3]))
    families = {}
    for state, accepted in ids.items():
        wanted = TARGETS[state]
        if len(accepted) < sum(wanted.values()):
            raise RuntimeError(f"Only {len(accepted)} {state} families; see rejection log {log}")
        order = split_rng.permutation(len(accepted))
        for rank, index in enumerate(order):
            family_id = accepted[index]
            path = Path("families") / f"{family_id}.npz"
            families[family_id] = {
                "path": str(path),
                "sha256": sha256_file(root / path),
                "state": state,
                "split": "development" if rank < wanted["development"] else "reserved",
            }
    result = {"status": "complete", "families": families, "rejections": log}
    seal(root / "collection.json", result)
    return result


def develop(root: Path = ROOT) -> dict[str, Any]:
    from jepa_action_effects.readout_scale import load_block_probe

    config = sealed(root / "config.json")
    if sealed(root / "audit.json")["sources"] != _sources():
        raise ValueError("Source changed since audit")
    collection = sealed(root / "collection.json")
    model, scaler, fingerprint = load_model()
    probe, selection = load_block_probe(READOUT2)
    if selection != config["readout_selection"]:
        raise ValueError("READOUT-002 selection changed since audit")
    ids = _phase_ids(collection, "development")
    records = _score(root, "develop", ids, collection, model, scaler, probe)
    summary = summarize_dose(records, config["bootstrap_seed"], BOOTSTRAP)
    result = {
        "status": "complete",
        "phase": "development (exploratory)",
        "families": len(records),
        "fingerprint": fingerprint,
        "verification_max_error": max(
            r["verification_max_error"] for r in records if r["verification_max_error"] is not None
        ),
        "summary": summary,
    }
    seal(root / "develop.json", result)
    return result


def freeze(root: Path = ROOT) -> dict[str, Any]:
    developed = sealed(root / "develop.json")
    if not developed["summary"]["criteria"]["passed"]:
        raise ValueError(
            "Development manipulation checks failed; redesign before opening reserved families"
        )
    collection = sealed(root / "collection.json")
    body = {
        "sources": _sources(),
        "config_sha256": sha256_file(root / "config.json"),
        "audit_sha256": sha256_file(root / "audit.json"),
        "develop_sha256": sha256_file(root / "develop.json"),
        "collection_sha256": sha256_file(root / "collection.json"),
        "readout_selection_sha256": sha256_file(READOUT2 / "selection.json"),
        "reserved_ids": _phase_ids(collection, "reserved"),
        "decision_rules": (
            "shift002_metrics.apply_dose_criteria; manipulation checks gate every verdict"
        ),
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
    collection = sealed(root / "collection.json")
    model, scaler, fingerprint = load_model()
    probe, _ = load_block_probe(READOUT2)
    records = _score(root, "confirm", frozen["reserved_ids"], collection, model, scaler, probe)
    summary = summarize_dose(records, config["bootstrap_seed"], BOOTSTRAP)
    result = {
        "status": "complete",
        "families": len(records),
        "fingerprint": fingerprint,
        "verification_max_error": max(
            r["verification_max_error"] for r in records if r["verification_max_error"] is not None
        ),
        "summary": summary,
    }
    seal(root / "confirm.json", result)
    return result


def _table(summary: dict) -> list[str]:
    lines = ["| Quantity | Estimate | 95% family bootstrap CI |", "| --- | ---: | --- |"]
    for key, cell in summary.items():
        if not isinstance(cell, dict) or "estimate" not in cell:
            continue
        estimate = "n/a" if cell["estimate"] is None else f"{cell['estimate']:.4f}"
        interval = "n/a" if cell["ci95"] is None else "[{:.4f}, {:.4f}]".format(*cell["ci95"])
        lines.append(f"| {key} | {estimate} | {interval} |")
    return lines


def report(root: Path = ROOT) -> dict[str, Any]:
    def maybe(name: str):
        path = root / name
        return sealed(path) if path.exists() else None

    collection = maybe("collection.json")
    developed, confirmed = maybe("develop.json"), maybe("confirm.json")
    summary = {
        "task": "SHIFT-002",
        "runtime": runtime_info(),
        "sources": _sources(),
        "config_sha256": sha256_file(root / "config.json"),
        "collection": None
        if collection is None
        else {"rejections": collection["rejections"], "families": len(collection["families"])},
        "seal": maybe("seal.json"),
        "develop": developed,
        "confirm": confirmed,
        "stage_seconds": json.loads((root / "runtime.json").read_text())["stages"]
        if (root / "runtime.json").exists()
        else None,
    }
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "shift002-runtime.json", summary)
    lines = ["# SHIFT-002 result (auto-generated draft)", ""]
    for label, block in (
        ("Development (exploratory)", developed),
        ("Reserved confirmation", confirmed),
    ):
        lines += [f"## {label}", ""]
        if block is None:
            lines += ["Not run.", ""]
            continue
        lines += _table(block["summary"])
        lines += [
            "",
            f"Moved families per gate cell: `{block['summary']['moved_families']}`",
            "",
            f"Criteria: `{block['summary']['criteria']}`",
            "",
        ]
    Path("docs/SHIFT-002_RESULT_DRAFT.md").write_text("\n".join(lines) + "\n")
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
