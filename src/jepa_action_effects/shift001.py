"""SHIFT-001: which factor makes the predictor shrink challenge-family block motion?

    python -m jepa_action_effects.shift001 audit     # freeze config, seeds, exclusions
    python -m jepa_action_effects.shift001 collect   # simulator/archive only, no model
    python -m jepa_action_effects.shift001 develop   # score development families
    python -m jepa_action_effects.shift001 seal      # needs passing manipulation checks
    python -m jepa_action_effects.shift001 confirm   # opens reserved families once
    python -m jepa_action_effects.shift001 report

Two state types (designed challenge layouts; archive windows whose block is at rest)
cross two action structures (constant 5-control blocks; the same block means plus
zero-mean archive residuals). Challenge families also carry a magnitude contrast
(push 0.6 vs 0.21). The frozen READOUT-002 probe decodes real and predicted futures,
and the rules in `shift001_metrics.apply_shift_criteria` are sealed with the code
before the reserved families are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    ARCHIVE_H5,
    MANIFESTS,
    Stopwatch,
    collection_manifest,
    encode_frames,
    load_family,
    load_model,
    predict_family,
    runtime_info,
    seal,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.shift001_metrics import (
    DELTA,
    block_residuals,
    constant_version,
    family_record,
    is_rest_window,
    summarize_shift,
    varying_version,
)

ROOT = Path("outputs/shift001")
READOUT2 = Path("outputs/readout002")
PHYS = Path("outputs/phys002")
DIST = Path("outputs/dist001")
MASTER_SEED = 20261020
TARGETS = {
    "challenge": {"development": 60, "reserved": 100},
    "archive": {"development": 60, "reserved": 100},
}
ATTEMPT_CAPS = {"challenge": 3000, "archive": 30000}
PUSH, LATERAL, SMALL = 0.6, 0.2, 0.21  # SMALL = archive median block magnitude (DIST-001)
REST_LIMIT = 1.0
IDENTITY_POSITION = 0.5
IDENTITY_FRAME = 1.0  # mean absolute uint8 difference
OWN_FUTURE_LIMIT = 1.0
BOOTSTRAP = 2000
VERIFY_PER_STATE = 6


def _sources() -> dict[str, str]:
    here = Path(__file__)
    return source_hashes(
        {
            "runner": here,
            "metrics": here.with_name("shift001_metrics.py"),
            "common": here.with_name("next_common.py"),
            "readout": here.with_name("readout_scale.py"),
            "inference": here.with_name("model_inference.py"),
            "collection": here.with_name("pilot_collection.py"),
            "replay": here.with_name("simulator_replay.py"),
            "uv_lock": Path("uv.lock"),
        }
    )


# --------------------------------------------------------------------------- archive
def _h5():
    import h5py

    return h5py.File(ARCHIVE_H5, "r")


def _exploratory_episodes(h5) -> list[int]:
    """Episodes drawn by the PHYS-002 audits (seed 777), replayed without the model."""
    rng = np.random.default_rng(777)
    offsets, lengths = h5["ep_offset"][:], h5["ep_len"][:]
    eligible = np.flatnonzero(lengths >= 21)
    drawn, moved, tries = set(), 0, 0
    while moved < 600 and tries < 5000:
        tries += 1
        episode = int(rng.choice(eligible))
        start, length = int(offsets[episode]), int(lengths[episode])
        t = int(rng.integers(10, length - 10))
        drawn.add(episode)
        controls = h5["action"][start + t - 10 : start + t + 10]
        if np.any(np.abs(controls) > 1) or not np.isfinite(controls).all():
            continue
        states = h5["state"][[start + t, start + t + 10]]
        if np.linalg.norm(states[1, 2:4] - states[0, 2:4]) > DELTA:
            moved += 1
    return sorted(drawn)


def _dist_episodes() -> list[int]:
    identity = json.loads((DIST / "identity.json").read_text())
    gap = json.loads((DIST / "gap.json").read_text())
    return sorted({*identity["episodes"], *(row["episode"] for row in gap["rows"])})


GOAL_CORNERS = ((128.0, 128.0), (128.0, 384.0), (384.0, 128.0), (384.0, 384.0))


def safe_goal(state: list[float]) -> list[float]:
    """A goal pose that neither overlaps the walls nor can be reached in the window.

    `goal_state = state` terminates on the first step (success needs agent and block
    within 20 units), and a wall-overlapping goal such as FAR_GOAL kicks the block.
    The goal block and goal agent go to the two interior corners farthest from their
    start positions, so the goal physics step has no collisions.
    """
    corners = np.asarray(GOAL_CORNERS)
    block = corners[int(np.argmax(np.linalg.norm(corners - state[2:4], axis=1)))]
    remaining = corners[~np.all(corners == block, axis=1)]
    agent = remaining[int(np.argmax(np.linalg.norm(remaining - state[0:2], axis=1)))]
    return [*agent.tolist(), *block.tolist(), 0.0, 0.0, 0.0]


def _residual_pool(h5) -> tuple[np.ndarray, np.ndarray]:
    from jepa_action_effects.next_metrics import archive_blocks, block_features

    blocks = archive_blocks(h5["action"][:], h5["ep_offset"][:], h5["ep_len"][:])
    blocks = blocks[np.all(np.abs(blocks) <= 1, axis=(1, 2))]
    return block_residuals(blocks), block_features(blocks)["magnitude"]


# --------------------------------------------------------------------------- stages
def audit(root: Path = ROOT) -> dict[str, Any]:
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources():
            raise ValueError("SHIFT-001 source changed after audit; start a new version")
        return old
    selection = sealed(READOUT2 / "selection.json")
    prior = json.loads((PHYS / "forbidden-seeds.json").read_text())["seeds"]
    reserved = collection_manifest(PHYS / "reserved")
    prior += [
        int(load_family(PHYS / "reserved" / r["path"])["metadata"]["seed"])
        for r in reserved["family_files"].values()
    ]
    with _h5() as h5:
        excluded = sorted({*_dist_episodes(), *_exploratory_episodes(h5)})
    seeds = sorted(set(prior))
    config = {
        "schema": "shift001-v1",
        "master_seed": MASTER_SEED,
        "targets": TARGETS,
        "attempt_caps": ATTEMPT_CAPS,
        "challenge_pairs": {"push": PUSH, "lateral": [PUSH, LATERAL], "small": SMALL},
        "archive_pairs": ["own", "b1", "b2"],
        "rest_limit": REST_LIMIT,
        "identity_limits": {
            "position": IDENTITY_POSITION,
            "frame_mean_abs": IDENTITY_FRAME,
            "own_future": OWN_FUTURE_LIMIT,
        },
        "delta": DELTA,
        "forbidden_reset_seed_count": len(seeds),
        "forbidden_reset_seeds_sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
        "excluded_archive_episodes": excluded,
        "readout_selection": selection,
        "readout_selection_sha256": sha256_file(READOUT2 / "selection.json"),
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": MASTER_SEED + 7,
        "criteria": "shift001_metrics.apply_shift_criteria",
    }
    write_json(root / "forbidden-seeds.json", {"seeds": seeds})
    result = {
        "status": "passed",
        "sources": _sources(),
        "config_sha256": seal(root / "config.json", config),
    }
    seal(root / "audit.json", result)
    return result


def _accept_moved(replay, spec: dict[str, dict]) -> tuple[bool, dict[str, bool]]:
    from jepa_action_effects.pilot_collection import _states_from_steps

    moved = {}
    for key, branch in replay.branches.items():
        history, future, _ = _states_from_steps(branch)
        moved[key] = bool(np.linalg.norm(future[-1, 2:4] - history[-1, 2:4]) > DELTA)
    ok = all(
        any(moved[k] for k, s in spec.items() if s["structure"] == structure)
        for structure in ("constant", "varying")
    )
    return ok, moved


def _pair_candidates(label: str, varying: np.ndarray, constant: np.ndarray, spec: dict):
    from jepa_action_effects.simulator_replay import Candidate

    spec[f"{label}-constant"] = {"structure": "constant", "label": label}
    spec[f"{label}-varying"] = {"structure": "varying", "label": label}
    return [Candidate(f"{label}-constant", constant), Candidate(f"{label}-varying", varying)]


def _collect_challenge(root: Path, seeds: set[int], residuals, magnitudes, log: dict) -> list[str]:
    from jepa_action_effects.pilot_collection import (
        PilotConfig,
        _constant,
        _rngs,
        propose_challenge,
        save_family,
    )
    from jepa_action_effects.simulator_replay import run_family

    config = PilotConfig(master_seed=MASTER_SEED)
    rngs = _rngs(config)
    residual_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 1]))
    target = sum(TARGETS["challenge"].values())
    accepted: list[str] = []
    for attempt in range(ATTEMPT_CAPS["challenge"]):
        if len(accepted) >= target:
            break
        proposal, reason = propose_challenge(config, attempt, rngs["proposals"])
        if proposal is None:
            log[reason] = log.get(reason, 0) + 1
            continue
        recipe, _, metadata = proposal
        if recipe.seed in seeds:
            log["reset_seed_overlap"] = log.get("reset_seed_overlap", 0) + 1
            continue
        toward = np.asarray(metadata["approach_toward"])
        lateral = np.asarray(metadata["lateral_axis"])
        bases = {
            "push": toward * PUSH,
            "lateral": toward * PUSH + lateral * LATERAL,
            "small": toward * SMALL,
        }
        spec: dict[str, dict] = {}
        candidates = []
        for label, vector in bases.items():
            constant = _constant(vector).astype(np.float64)
            varying = varying_version(constant, residuals, magnitudes, residual_rng)
            candidates += _pair_candidates(label, varying, constant, spec)
        order = [c.candidate_id for c in candidates]
        rngs["execution_order"].shuffle(order)
        try:
            replay = run_family(recipe, candidates, order)
        except RuntimeError:
            log["replay_error"] = log.get("replay_error", 0) + 1
            continue
        if not all(branch.completed for branch in replay.branches.values()):
            log["incomplete"] = log.get("incomplete", 0) + 1
            continue
        ok, moved = _accept_moved(replay, spec)
        if not ok:
            log["no_moved_in_both_structures"] = log.get("no_moved_in_both_structures", 0) + 1
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


def _archive_window(h5, episode: int, t: int) -> dict | None:
    start = int(h5["ep_offset"][episode])
    controls = h5["action"][start + t - 10 : start + t + 10].astype(np.float64)
    if np.any(np.abs(controls) > 1) or not np.isfinite(controls).all():
        return None
    states = h5["state"][start + t - 10 : start + t + 11].astype(np.float64)
    return {"episode": episode, "t": t, "start": start, "controls": controls, "states": states}


def _collect_archive(root: Path, excluded: set[int], residuals, magnitudes, log: dict) -> list[str]:
    from jepa_action_effects.pilot_collection import _states_from_steps, save_family
    from jepa_action_effects.simulator_replay import ReplayRecipe, run_family

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
            pool = np.flatnonzero(available)
            episode = int(rng.choice(pool))
            t = int(rng.integers(10, int(lengths[episode]) - 10))
            window = _archive_window(h5, episode, t)
            if window is None:
                log["out_of_bounds"] = log.get("out_of_bounds", 0) + 1
                continue
            states = window["states"]
            if not is_rest_window(states[:11, 2:4], REST_LIMIT):
                log["block_not_at_rest"] = log.get("block_not_at_rest", 0) + 1
                continue
            if np.linalg.norm(states[20, 2:4] - states[10, 2:4]) <= DELTA:
                log["own_future_static"] = log.get("own_future_static", 0) + 1
                continue
            borrowed = []
            while len(borrowed) < 2:
                other = int(rng.choice(pool))
                if other == episode or other in {b["episode"] for b in borrowed}:
                    continue
                candidate = _archive_window(
                    h5, other, int(rng.integers(10, int(lengths[other]) - 10))
                )
                if candidate is not None:
                    borrowed.append(candidate)
            spec: dict[str, dict] = {}
            candidates = []
            for label, source in (("own", window), ("b1", borrowed[0]), ("b2", borrowed[1])):
                varying = source["controls"][10:20]
                candidates += _pair_candidates(label, varying, constant_version(varying), spec)
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
                log["replay_error"] = log.get("replay_error", 0) + 1
                continue
            if not all(branch.completed for branch in replay.branches.values()):
                log["incomplete"] = log.get("incomplete", 0) + 1
                continue
            reference = replay.branches["own-varying"]
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
                log["identity_failed"] = log.get("identity_failed", 0) + 1
                continue
            ok, moved = _accept_moved(replay, spec)
            if not ok:
                log["no_moved_in_both_structures"] = log.get("no_moved_in_both_structures", 0) + 1
                continue
            family_id = f"archive-{len(accepted):03d}"
            replay = type(replay)(family_id, replay.seed, replay.execution_order, replay.branches)
            metadata = {
                "cohort": "archive",
                "spec": spec,
                "moved": moved,
                "episode": episode,
                "t": t,
                "borrowed": [[b["episode"], b["t"]] for b in borrowed],
                "identity": {
                    "position": position_error,
                    "frame": frame_error,
                    "own_future": own_error,
                },
            }
            save_family(
                root / "families" / f"{family_id}.npz", recipe, candidates, replay, metadata
            )
            available[[episode, *(b["episode"] for b in borrowed)]] = False
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
        "archive": _collect_archive(
            root, set(config["excluded_archive_episodes"]), residuals, magnitudes, log
        ),
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


# --------------------------------------------------------------------------- scoring
def score_family(identifier: str, family: dict, state: str, model, scaler, probe):
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    real = encode_frames(model, frames)["projected"].astype(np.float64)
    predicted = predict_family(model, scaler, family).astype(np.float64)
    if real.shape != (15, 192) or predicted.shape != (6, 2, 192):
        raise ValueError(f"Unexpected embedding alignment for {identifier}")
    start = probe(real[2:3])[0]
    decoded_real = probe(real[3:]).reshape(6, 2, 2) - start
    decoded_pred = probe(predicted.reshape(-1, 192)).reshape(6, 2, 2) - start
    true = family["future_states"][:, :, 2:4] - family["history_states"][-1, 2:4]
    record = family_record(
        identifier,
        state,
        family["metadata"]["candidate_ids"],
        family["metadata"]["spec"],
        true,
        decoded_pred,
        decoded_real,
        family["contacts"].astype(bool).tolist(),
    )
    return record, {"real": real, "predicted": predicted}


def _score(
    root: Path, phase: str, ids: list[str], collection: dict, model, scaler, probe
) -> list[dict]:
    from jepa_action_effects.latent_audit import _verification

    verify = set()
    for state in ("challenge", "archive"):
        verify.update(
            [key for key in ids if collection["families"][key]["state"] == state][:VERIFY_PER_STATE]
        )
    records = []
    for family_id in ids:
        target = root / phase / f"{family_id}.json"
        if target.exists():
            records.append(sealed(target))
            continue
        entry = collection["families"][family_id]
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Family artifact changed: {family_id}")
        family = load_family(path)
        record, arrays = score_family(family_id, family, entry["state"], model, scaler, probe)
        record["verification_max_error"] = (
            _verification(family, arrays["predicted"].astype(np.float32), model, scaler)
            if family_id in verify
            else None
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".npz"), **arrays)
        seal(target, record)
        records.append(record)
    return records


def _phase_ids(collection: dict, split: str) -> list[str]:
    return sorted(k for k, v in collection["families"].items() if v["split"] == split)


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
    records = _score(
        root, "develop", _phase_ids(collection, "development"), collection, model, scaler, probe
    )
    summary = summarize_shift(records, config["bootstrap_seed"], BOOTSTRAP)
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
            "shift001_metrics.apply_shift_criteria; manipulation checks gate every verdict"
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
    summary = summarize_shift(records, config["bootstrap_seed"], BOOTSTRAP)
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


KEY_ROWS = (
    "R_challenge_constant",
    "R_challenge_varying",
    "R_archive_constant",
    "R_archive_varying",
    "R_challenge_constant_push",
    "R_challenge_constant_small",
    "R_challenge_varying_small",
    "R_archive_varying_own",
    "dS_challenge",
    "dS_archive",
    "dM_challenge",
    "dT_varying",
    "R1_challenge_constant",
    "R1_archive_varying",
    "floor_challenge_constant",
    "floor_challenge_varying",
    "floor_archive_constant",
    "floor_archive_varying",
)


def _table(summary: dict) -> list[str]:
    lines = ["| Quantity | Estimate | 95% family bootstrap CI |", "| --- | ---: | --- |"]
    for key in KEY_ROWS:
        cell = summary[key]
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
        "task": "SHIFT-001",
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
    write_json(MANIFESTS / "shift001-runtime.json", summary)
    lines = ["# SHIFT-001 result (auto-generated draft)", ""]
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
            f"Moved families per cell: `{block['summary']['moved_families']}`",
            "",
            f"Criteria: `{block['summary']['criteria']}`",
            "",
        ]
    Path("docs/SHIFT-001_RESULT_DRAFT.md").write_text("\n".join(lines) + "\n")
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
