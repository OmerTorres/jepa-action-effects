"""MEAS-001: bounded feasibility of abstaining joint-pose inversion (D032, D033).

Stages, in order. The engineering pass (D033) runs only the first four:

    select           metadata-only membership of the 48 SHIFT-002 development families
    engineer-select  declare GALLERY-001 engineering families; verify disjointness
    engineer         develop and validate search settings on the engineering set (charged)
    propose          freeze-ready configuration proposal and runtime projection
    freeze           (after review) seal the reviewed proposal; requires its SHA-256
    benchmark        throughput check on two calibration families; may stop, never tune
    calibrate        ε and τ from calibration queries only; sealed before assessment
    controls         24 synthetic targets plus known-failure checks
    assess           the 24 assessment families, once
    score            validator scoring with labels, integrity checks and gates
    predictions      optional 16 saved frozen-predictor latents, only if D032 §7 cells pass
    report           manifest and tables; nothing is written after an integrity failure

Search code receives only a `SceneContext` and a target latent; labels are read only by
validator functions (`_labels`, `score`, engineering scoring). One four-hour ledger at
`outputs/meas001-budget/` charges every stage, including failed and resumed ones.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.meas001_metrics import (
    ENGINEERING,
    GATES,
    LOW_MOTION,
    decision,
    displacement_cell,
    endpoint_cell,
    engineering_criteria,
    predictions_allowed,
)
from jepa_action_effects.next_common import (
    MANIFESTS,
    collection_manifest,
    load_family,
    runtime_info,
    seal,
    sealed,
    write_json,
)
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.pose_render import (
    AGENT_WALL_CHECKS,
    PENETRATION_TOLERANCE,
    SceneContext,
    SceneRenderer,
)
from jepa_action_effects.pose_search import (
    BLOCK,
    CalibrationFailure,
    Inversion,
    SearchConfig,
    SearchResult,
    Trace,
    calibrate_tau,
    classify,
    invert,
    seed_sequence,
)
from jepa_action_effects.train001_budget import Budget, BudgetStop, session_seconds

ROOT = Path("outputs/meas001")
BUDGET_ROOT = Path("outputs/meas001-budget")
SHIFT2 = Path("outputs/shift002")
GALLERY = Path("outputs/gallery001")
FRESH = Path("outputs/readout001/collection")
CHECKPOINT = Path("checkpoints/lewm-pusht")
LIMIT_SECONDS = 4 * 3600.0
ENGINEERING_LIMIT = 3600.0
SEEDS = {
    "selection": 2026100101,
    "search": 2026100102,
    "engineering": 2026100103,
    "controls": 2026100104,
}
SCENES = ("archive", "challenge")
PER_SCENE = 12
ENDPOINTS = {
    "challenge": (
        ("m0.15", "constant", "low_motion"),
        ("m0.30", "constant", "endpoint"),
        ("m0.45", "constant", "endpoint"),
        ("m0.60", "constant", "endpoint"),
    ),
    "archive": (
        ("x1.0", "varying", "endpoint"),
        ("x1.5", "varying", "endpoint"),
        ("x2.0", "varying", "endpoint"),
    ),
}
PREDICTION_DOSES = {"challenge": ("m0.30", "m0.60"), "archive": ("x1.0", "x2.0")}
ENGINEERING_SETTINGS = {
    "S1-4x160": SearchConfig(),
    "S2-2x320": SearchConfig(local_starts=2, local_evaluations=320),
    "S3-512+2x192": SearchConfig(coarse=512, local_starts=2, local_evaluations=192),
}
ENGINEERING_SETTINGS_V2 = {
    # D037: one setting, to fit the remaining engineering sub-cap with phase B.
    "V2-S2r300": SearchConfig(
        version=2, local_starts=2, local_evaluations=320, refine_evaluations=300
    ),
}
ACCOUNTING_CORRECTION_SECONDS = 120.0  # disclosed, conservative (MEAS-REVIEW-001)
ENGINEERING_PER_COHORT = 6  # three for setting selection (A), three for validation (B)
WORKLOAD_TARGETS = 256  # 216 real + 24 synthetic + 16 optional prediction targets
HEADROOM = 1.5
ENCODE_BATCH = 64
SOURCES = (
    "meas001.py",
    "meas001_metrics.py",
    "pose_search.py",
    "pose_render.py",
    "gallery_render.py",
    "simulator_replay.py",
    "model_inference.py",
    "interface_audit.py",
    "next_common.py",
    "train001_budget.py",
)


class LineageError(RuntimeError):
    """A sealed input, source or configuration changed after it was recorded."""


class DisjointnessError(RuntimeError):
    """Engineering and MEAS-001 families share a reset family or archive episode."""


class EngineeringCap(RuntimeError):
    """The next engineering target would exceed the 60-minute engineering sub-cap."""


# --------------------------------------------------------------------------- helpers
def _seal_once(path: Path, payload: dict) -> dict:
    """Write-once seal: an existing record must be identical, never silently replaced."""
    payload = json.loads(json.dumps(payload, sort_keys=True))
    if path.exists():
        if sealed(path) != payload:
            raise LineageError(f"{path} already sealed with different content")
        return payload
    seal(path, payload)
    return payload


def _hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else "absent"


def source_hashes() -> dict[str, str]:
    here = Path(__file__).parent
    hashes = {name: _hash(here / name) for name in SOURCES}
    hashes["uv_lock"] = _hash(Path("uv.lock"))
    hashes["checkpoint_weights"] = _hash(CHECKPOINT / "weights.pt")
    hashes["checkpoint_config"] = _hash(CHECKPOINT / "config.json")
    return hashes


def read_npz(path: Path, keys: Iterable[str]) -> dict[str, Any]:
    """Read only the named members; NPZ members load lazily, so others stay untouched."""
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for key in keys:
            value = archive[key]
            result[key] = json.loads(str(value.item())) if key == "metadata" else value
    return result


def _state_key(state: Iterable[float] | None) -> tuple[float, ...] | None:
    """Exact reset state, or None for a seeded random reset (identified by its seed)."""
    return None if state is None else tuple(round(float(x), 6) for x in list(state)[:5])


def _stage_seconds(budget: Budget, prefix: str) -> float:
    ledger = budget.load()
    now = budget.clock()
    total = 0.0
    for session in ledger["sessions"]:
        if session["stage"].startswith(prefix):
            active = budget.session is not None and session["id"] == budget.session["id"]
            total += session_seconds(session, now if active else None)
    return total


def _budget() -> Budget:
    return Budget(BUDGET_ROOT, limit_seconds=LIMIT_SECONDS)


def _load_model():
    import torch

    from jepa_action_effects.next_common import load_model

    torch.set_num_threads(4)
    model, _, fingerprint = load_model()
    return model, fingerprint


def encode(model, frames: np.ndarray) -> np.ndarray:
    """float32 projected CLS for uint8 frames, returned in float64 for reductions."""
    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import encode_real

    out = []
    for start in range(0, len(frames), ENCODE_BATCH):
        chunk = np.ascontiguousarray(frames[start : start + ENCODE_BATCH])
        out.append(encode_real(model, preprocess_pixels(chunk).unsqueeze(0)).numpy()[0])
    return np.concatenate(out).astype(np.float64)


class RealObjective:
    """Render-and-encode residual to one target latent; sees no labels."""

    def __init__(self, renderer: SceneRenderer, model, target: np.ndarray):
        self.renderer = renderer
        self.model = model
        self.target = np.asarray(target, dtype=np.float64)

    def valid(self, poses: np.ndarray) -> np.ndarray:
        return self.renderer.check(poses)[0]

    def evaluate(self, poses: np.ndarray) -> np.ndarray:
        latents = encode(self.model, self.renderer.render(poses))
        return np.mean((latents - self.target) ** 2, axis=1)


def make_objective(renderer: SceneRenderer, model, target: np.ndarray):
    return RealObjective(renderer, model, target)


def encoder_floor(model, frames: np.ndarray) -> float:
    """ε: largest per-frame MSE between forward and reversed/duplicated batch encodings."""
    forward = encode(model, frames)
    backward = encode(model, frames[::-1].copy())[::-1]
    single = np.stack([encode(model, frames[i : i + 1])[0] for i in range(min(8, len(frames)))])
    mse = max(
        float(np.max(np.mean((forward - backward) ** 2, axis=1))),
        float(np.max(np.mean((forward[: len(single)] - single) ** 2, axis=1))),
    )
    return max(mse, 1e-12)


def save_inversion(path: Path, inversion: Inversion, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = inversion.trace.arrays()
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    payload = {
        **extra,
        "trace_sha256": sha256_file(path.with_suffix(".npz")),
        "evaluations": inversion.trace.evaluations,
        "invalid_attempts": inversion.trace.invalid_attempts,
        "searches": [
            {
                "best_pose": None if s.best_pose is None else s.best_pose.tolist(),
                "best_residual": s.best_residual if math.isfinite(s.best_residual) else None,
                "converged": s.converged,
                "coarse_valid": s.coarse_valid,
            }
            for s in inversion.searches
        ],
        "nuisance": inversion.nuisance,
        "config": inversion.config.to_dict(),
        "resolution": inversion.resolution,
    }
    write_json(path.with_suffix(".json"), payload)


def load_inversion(path: Path) -> tuple[Inversion, dict]:
    payload = json.loads(path.with_suffix(".json").read_text())
    if sha256_file(path.with_suffix(".npz")) != payload["trace_sha256"]:
        raise LineageError(f"Trace changed: {path}")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        trace = Trace(
            poses=list(arrays["poses"]),
            residuals=[float(v) for v in arrays["residuals"]],
            phases=[int(v) for v in arrays["phases"]],
            searches=[int(v) for v in arrays["searches"]],
            cached=[bool(v) for v in arrays["cached"]],
            evaluations=payload["evaluations"],
            invalid_attempts=payload["invalid_attempts"],
        )
    searches = [
        SearchResult(
            None if s["best_pose"] is None else np.asarray(s["best_pose"], float),
            math.inf if s["best_residual"] is None else float(s["best_residual"]),
            bool(s["converged"]),
            int(s["coarse_valid"]),
        )
        for s in payload["searches"]
    ]
    config = SearchConfig.from_dict(payload["config"])
    inversion = Inversion(searches, payload["nuisance"], trace, config, payload.get("resolution"))
    return inversion, payload


SEARCH_SOURCES = ("pose_search.py", "pose_render.py", "gallery_render.py", "model_inference.py")


def binding(objective, config: SearchConfig, master: int, query_id: str, inputs: dict) -> dict:
    """Identity of one inversion (MEAS-REVIEW-001 F1): reuse requires an exact match."""
    import hashlib

    target = getattr(objective, "target", None)
    here = Path(__file__).parent
    return json.loads(
        json.dumps(
            {
                "query_id": query_id,
                "config": config.to_dict(),
                "master": int(master),
                "target_sha256": None
                if target is None
                else hashlib.sha256(np.ascontiguousarray(target, np.float64).tobytes()).hexdigest(),
                "inputs": inputs,
                "sources": {name: _hash(here / name) for name in SEARCH_SOURCES},
                "checkpoint": _hash(CHECKPOINT / "weights.pt"),
            },
            sort_keys=True,
        )
    )


def _run_target(
    path: Path,
    objective,
    config: SearchConfig,
    master: int,
    query_id: str,
    budget: Budget,
    extra: dict,
    inputs: dict | None = None,
) -> dict:
    """Invert one target unless a result with the identical binding exists.

    A stored result with a different (or missing) binding is refused, never reused.
    """
    expected = binding(objective, config, master, query_id, inputs or {})
    if path.with_suffix(".json").exists():
        payload = load_inversion(path)[1]
        if payload.get("binding") != expected:
            raise LineageError(f"Stored inversion has a different binding: {path}")
        return payload
    budget.unit("target")
    started = time.monotonic()
    inversion = invert(objective, config, master, query_id)
    seconds = time.monotonic() - started
    budget.record_unit("target", seconds)
    budget.heartbeat()
    save_inversion(
        path, inversion, {**extra, "query_id": query_id, "seconds": seconds, "binding": expected}
    )
    return load_inversion(path)[1]


# --------------------------------------------------------------------------- select
def _shift_family_path(entry: dict) -> Path:
    return SHIFT2 / entry["path"]


def select(root: Path = ROOT) -> dict:
    """Membership from metadata only; no frame, latent or label is read."""
    collection = sealed(SHIFT2 / "collection.json")
    development = {k: v for k, v in collection["families"].items() if v["split"] == "development"}
    rng = np.random.default_rng(SEEDS["selection"])
    families: dict[str, dict] = {}
    queries: list[dict] = []
    roles: dict[str, list] = {"benchmark": [], "controls": [], "predictions": []}
    for scene in SCENES:
        ids = sorted(k for k, v in development.items() if v["state"] == scene)
        order = [ids[i] for i in rng.permutation(len(ids))]
        if len(order) < 2 * PER_SCENE:
            raise ValueError(f"Not enough independent {scene} development families")
        for partition, chosen in (
            ("calibration", order[:PER_SCENE]),
            ("assessment", order[PER_SCENE : 2 * PER_SCENE]),
        ):
            for family_id in chosen:
                entry = development[family_id]
                path = _shift_family_path(entry)
                metadata = read_npz(path, ["metadata"])["metadata"]
                spec = metadata["spec"]
                index = {
                    (spec[c]["label"], spec[c]["structure"]): i
                    for i, c in enumerate(metadata["candidate_ids"])
                }
                families[family_id] = {
                    "scene": scene,
                    "partition": partition,
                    "path": str(path),
                    "file_sha256": entry["sha256"],
                    "latents": str(SHIFT2 / "develop" / f"{family_id}.npz"),
                    "latents_sha256": _hash(SHIFT2 / "develop" / f"{family_id}.npz"),
                    "seed": int(metadata["seed"]),
                    "episode": metadata.get("episode"),
                    "reset_state": _state_key(metadata["reset_options"].get("state")),
                }
                queries.append({"id": f"{family_id}:start", "family": family_id, "query": "start"})
                for label, structure, role in ENDPOINTS[scene]:
                    if (label, structure) not in index:
                        raise ValueError(f"{family_id} lacks candidate {label}/{structure}")
                    queries.append(
                        {
                            "id": f"{family_id}:{label}",
                            "family": family_id,
                            "query": label,
                            "candidate": index[(label, structure)],
                            "role": role,
                        }
                    )
        calibration, assessment = order[:PER_SCENE], order[PER_SCENE : 2 * PER_SCENE]
        roles["benchmark"].append(calibration[0])
        roles["controls"].extend(calibration[:3])
        for family_id in assessment[:4]:
            for label in PREDICTION_DOSES[scene]:
                roles["predictions"].append(f"{family_id}:{label}")
    # Reset-family identity: the exact reset state, plus the archive episode. Archive scenes
    # reset from archive states with a constant seed (0), so a seed identifies a family only
    # for seeded resets (no episode).
    keys = [
        ("reset_state", [tuple(f["reset_state"]) for f in families.values() if f["reset_state"]]),
        ("episode", [f["episode"] for f in families.values() if f["episode"] is not None]),
        ("seed", [f["seed"] for f in families.values() if f["episode"] is None]),
    ]
    for name, values in keys:
        if len(values) != len(set(values)):
            raise ValueError(f"Selected families share a {name}; aliases would cross partitions")
    payload = {
        "task": "MEAS-001",
        "seed": SEEDS["selection"],
        "collection_sha256": sha256_file(SHIFT2 / "collection.json"),
        "families": families,
        "queries": queries,
        "roles": roles,
        "counts": {
            "families": len(families),
            "real_queries": len(queries),
            "calibration_queries": sum(
                families[q["family"]]["partition"] == "calibration" for q in queries
            ),
        },
    }
    return _seal_once(root / "membership.json", payload)["counts"]


# --------------------------------------------------------------------------- engineering
def _gallery_family(identifier: str) -> tuple[Path, str]:
    manifest = collection_manifest(FRESH)
    key = identifier.split(":", 1)[1]
    return FRESH / manifest["family_files"][key]["path"], key.split("-")[0]


def check_disjoint(engineering: dict[str, dict], measured: dict[str, dict]) -> dict:
    """No shared reset seed, reset state or archive episode (D033)."""
    report = {}

    def values(group: dict[str, dict], name: str) -> set:
        found = set()
        for v in group.values():
            if v.get(name) is None or (name == "seed" and v.get("episode") is not None):
                continue  # archive-derived resets share a constant seed
            found.add(tuple(v[name]) if isinstance(v[name], list) else v[name])
        return found

    for name in ("seed", "reset_state", "episode"):
        left, right = values(engineering, name), values(measured, name)
        shared = sorted(map(str, left & right))
        report[name] = {"engineering": len(left), "measured": len(right), "shared": shared}
        if shared:
            raise DisjointnessError(f"Engineering and MEAS-001 families share {name}: {shared}")
    return report


def engineer_select(root: Path = ROOT) -> dict:
    """Declare the engineering set, targets and rules before any engineering encode."""
    membership = sealed(root / "membership.json")
    gallery = sealed(GALLERY / "config.json")
    engineering: dict[str, dict] = {}
    for identifier in gallery["ids"]:
        path, cohort = _gallery_family(identifier)
        if sha256_file(path) != gallery["family_hashes"][identifier]:
            raise LineageError(f"GALLERY-001 family changed: {identifier}")
        metadata = read_npz(path, ["metadata"])["metadata"]
        engineering[identifier] = {
            "cohort": cohort,
            "path": str(path),
            "file_sha256": gallery["family_hashes"][identifier],
            "seed": int(metadata["seed"]),
            "episode": metadata.get("episode"),
            "reset_state": _state_key(metadata["reset_options"].get("state")),
            "candidate_ids": metadata["candidate_ids"],
        }
    disjoint = check_disjoint(engineering, membership["families"])
    rng = np.random.default_rng(SEEDS["engineering"])
    phases: dict[str, list[dict]] = {"A": [], "B": []}
    for cohort in ("challenge", "ordinary"):
        ids = sorted(k for k, v in engineering.items() if v["cohort"] == cohort)
        order = [ids[i] for i in rng.permutation(len(ids))][:ENGINEERING_PER_COHORT]
        if len(order) < ENGINEERING_PER_COHORT:
            raise ValueError(f"Not enough {cohort} engineering families")
        for position, identifier in enumerate(order):
            candidates = engineering[identifier]["candidate_ids"]
            if cohort == "challenge":
                laterals = [c for c in candidates if c.startswith("lateral")]
                chosen = ["push", laterals[int(rng.integers(len(laterals)))]]
            else:
                picks = rng.choice(len(candidates), size=2, replace=False)
                chosen = [candidates[int(i)] for i in sorted(picks)]
            phase = "A" if position < ENGINEERING_PER_COHORT // 2 else "B"
            phases[phase].append(
                {
                    "family": identifier,
                    "cohort": cohort,
                    "endpoints": [{"candidate": candidates.index(c), "id": c} for c in chosen],
                }
            )
    spec = {
        "task": "MEAS-001 engineering (D033)",
        "membership_sha256": sha256_file(root / "membership.json"),
        "families": {k: engineering[k] for k in sorted(engineering)},
        "disjointness": disjoint,
        "phases": phases,
        "settings": {name: config.legacy_dict() for name, config in ENGINEERING_SETTINGS.items()},
        "geometry": {"tolerance": PENETRATION_TOLERANCE, "agent_walls": AGENT_WALL_CHECKS},
        "criteria": ENGINEERING,
        "selection_rule": "settings passing all phase-A criteria; highest minimum cohort "
        "coverage, then lowest mean seconds per target. Phase B validates the choice with "
        "phase-A τ.",
        "synthetic": "one geometry-valid pose per family: observed start pose with block "
        "translated U(5, 30) units, rotated U(-0.5, 0.5) rad, agent translated U(10, 60) units",
        "sub_cap_seconds": ENGINEERING_LIMIT,
        "seed": SEEDS["engineering"],
        "predictions_used": False,
    }
    payload = _seal_once(root / "engineering" / "spec.json", spec)
    return {
        "disjointness": payload["disjointness"],
        "phases": {k: len(v) for k, v in phases.items()},
    }


def _synthetic_pose(renderer: SceneRenderer, start: np.ndarray, seed) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    for _ in range(50):
        pose = np.array(start, dtype=float)
        angle = rng.uniform(0, 2 * math.pi)
        pose[2:4] += rng.uniform(5, 30) * np.array([math.cos(angle), math.sin(angle)])
        pose[4] = (pose[4] + rng.uniform(-0.5, 0.5)) % (2 * math.pi)
        angle = rng.uniform(0, 2 * math.pi)
        pose[0:2] += rng.uniform(10, 60) * np.array([math.cos(angle), math.sin(angle)])
        if renderer.check(pose[None])[0][0]:
            return pose
    return None


def _prepare_engineering_family(item: dict, model, spec: dict) -> dict:
    """Targets, labels and integrity for one engineering family (labels are validator-side)."""
    from jepa_action_effects.gallery_render import replay_endpoint
    from jepa_action_effects.simulator_replay import InstrumentedPushT

    started = time.monotonic()
    meta = spec["families"][item["family"]]
    family = load_family(Path(meta["path"]))
    renderer = SceneRenderer(
        SceneContext.from_family(family, item["cohort"]),
        spec["geometry"]["tolerance"],
        spec["geometry"]["agent_walls"],
    )
    integrity = {
        "start_rgb": bool(np.array_equal(renderer.baseline_rgb, family["history_frames"][-1]))
    }
    env = InstrumentedPushT()
    try:
        for endpoint in item["endpoints"]:
            replay_endpoint(env, family, endpoint["candidate"])  # raises on any mismatch
    finally:
        env.close()
    integrity["endpoint_replay"] = True
    frames = [family["history_frames"][-1]]
    labels = [family["history_states"][-1][:5]]
    ids = [f"{item['family']}:start"]
    for endpoint in item["endpoints"]:
        frames.append(family["future_frames"][endpoint["candidate"], 1])
        labels.append(family["future_states"][endpoint["candidate"], 1][:5])
        ids.append(f"{item['family']}:{endpoint['id']}")
    kinds = ["real"] * len(frames)
    synthetic = _synthetic_pose(
        renderer,
        np.asarray(family["history_states"][-1][:5], float),
        seed_sequence(SEEDS["engineering"], item["family"], 9),
    )
    if synthetic is not None:
        frames.append(renderer.render(synthetic[None])[0])
        labels.append(synthetic)
        ids.append(f"{item['family']}:synthetic")
        kinds.append("synthetic")
    frames = np.stack(frames)
    latents = encode(model, frames)
    labels = [np.asarray(label, float) for label in labels]
    # Validator diagnostics: render the labelled poses by assignment and compare.
    valid, reasons, _ = renderer.check(np.stack(labels))
    rendered = renderer.render(np.stack(labels))
    true_pose_residual = np.mean((encode(model, rendered) - latents) ** 2, axis=1)
    mismatch = [int(np.any(rendered[i] != frames[i], axis=2).sum()) for i in range(len(frames))]
    return {
        "renderer": renderer,
        "targets": [
            {
                "id": ids[i],
                "kind": kinds[i],
                "scene": item["cohort"],
                "latent": latents[i],
                "true_pose": labels[i],
                "true_pose_valid": bool(valid[i]),
                "true_pose_reason": reasons[i],
                "true_pose_residual": float(true_pose_residual[i]),
                "assignment_pixel_mismatch": mismatch[i],
            }
            for i in range(len(ids))
        ],
        "epsilon": encoder_floor(model, frames),
        "integrity": integrity,
        "setup_seconds": time.monotonic() - started,
    }


def _score(payload: dict, target: dict, tau: float) -> dict:
    inversion, _ = load_inversion(Path(payload["path"]))
    result = classify(inversion, tau)
    best = result.get("best_pose")
    record = {
        "id": target["id"],
        "scene": target["scene"],
        "kind": target["kind"],
        "status": result["status"],
        "converged": result["converged"],
        "restart_distance": result.get("restart_distance"),
        "near_optimal_diameter": result.get("near_optimal_diameter"),
        "best_residual": result["best_residual"],
        "true_block": target["true_pose"][BLOCK].tolist(),
        "true_pose_valid": target["true_pose_valid"],
        "true_pose_residual": target["true_pose_residual"],
        "assignment_pixel_mismatch": target["assignment_pixel_mismatch"],
        "best_block": None if best is None else list(best[2:4]),
        "seconds": payload["seconds"],
        "evaluations": payload["evaluations"],
    }
    if result["status"] == "accepted":
        record["block_center"] = result["block_center"]
    return record


def best_residual(path: Path) -> float:
    """Best residual used by the τ rule; ∞ if none.

    Version 1: overall best (searches and nuisance). Version 2 (D037): the reported pose,
    i.e. the best independent-search result.
    """
    inversion = load_inversion(path)[0]
    return inversion.reported[1] if inversion.config.version >= 2 else inversion.best[1]


def family_maxima(rows: Iterable[tuple[str, str, float]]) -> dict[str, list[float]]:
    """(scene, family, best residual) rows → per-scene lists of per-family maxima."""
    maxima: dict[str, dict[str, float]] = {}
    for scene, family, value in rows:
        current = maxima.setdefault(scene, {}).get(family, -math.inf)
        maxima[scene][family] = max(current, value)
    return {scene: list(values.values()) for scene, values in maxima.items()}


def _engineering_tau(payloads: list[dict], targets: dict[str, dict], epsilon: float) -> float:
    rows = []
    for payload in payloads:
        target = targets[payload["query_id"].split("engineering:", 1)[1]]
        if target["kind"] == "real":
            family = target["id"].rsplit(":", 1)[0]
            rows.append((target["scene"], family, best_residual(Path(payload["path"]))))
    return calibrate_tau(family_maxima(rows), epsilon)


def engineer(root: Path = ROOT, folder: str = "engineering") -> dict:
    """Charged engineering validation (D033): choose settings on A, validate on B.

    `folder` is "engineering" (v1, D033) or "engineering-v2" (D037); both share the
    cumulative 60-minute engineering sub-cap.
    """
    spec = sealed(root / folder / "spec.json")
    if spec["membership_sha256"] != sha256_file(root / "membership.json"):
        raise LineageError("Membership changed after the engineering spec was sealed")
    budget = _budget()
    budget.start("engineer" if folder == "engineering" else f"engineer-{folder}", str(root))
    status = "complete"
    prepared: dict[str, dict] = {}
    result: dict[str, Any] = {"phases": {}, "status": "incomplete"}
    try:
        model, fingerprint = _load_model()
        result["model_fingerprint"] = fingerprint

        seconds: list[float] = []

        def run_phase(phase: str, name: str, tau: float | None) -> dict:
            config = SearchConfig.from_dict(spec["settings"][name])
            payloads, targets, epsilons, setups = [], {}, [], []
            for item in spec["phases"][phase]:
                if item["family"] not in prepared:
                    prepared[item["family"]] = _prepare_engineering_family(item, model, spec)
                family = prepared[item["family"]]
                epsilons.append(family["epsilon"])
                setups.append(family["setup_seconds"])
                for target in family["targets"]:
                    used = _stage_seconds(budget, "engineer")
                    estimate = float(np.mean(seconds)) if seconds else 30.0
                    if used + estimate > ENGINEERING_LIMIT:
                        raise EngineeringCap(f"{used:.0f} s used; next target ~{estimate:.0f} s")
                    targets[target["id"]] = target
                    path = root / folder / phase / name / target["id"].replace(":", "__")
                    payload = _run_target(
                        path,
                        make_objective(family["renderer"], model, target["latent"]),
                        config,
                        SEEDS["engineering"],
                        f"engineering:{target['id']}",
                        budget,
                        {"phase": phase, "setting": name},
                        {
                            "family_sha256": spec["families"][item["family"]]["file_sha256"],
                            "target": target["id"],
                            "kind": target["kind"],
                        },
                    )
                    payloads.append({**payload, "path": str(path)})
                    seconds.append(payload["seconds"])
            epsilon = max(epsilons)
            phase_tau = _engineering_tau(payloads, targets, epsilon) if tau is None else tau
            records = [
                _score(p, targets[p["query_id"].split("engineering:", 1)[1]], phase_tau)
                for p in payloads
            ]
            return {
                "setting": name,
                "tau": phase_tau,
                "tau_source": "own phase" if tau is None else "phase A",
                "epsilon": epsilon,
                "criteria": engineering_criteria(records),
                "seconds_per_target": float(np.mean([r["seconds"] for r in records])),
                "evaluations_per_target": float(np.mean([r["evaluations"] for r in records])),
                "setup_seconds_per_family": float(np.mean(setups)),
                "records": records,
            }

        for name in spec["settings"]:
            result["phases"][f"A:{name}"] = run_phase("A", name, None)
        passing = [
            (name, r)
            for name, r in ((k.split(":", 1)[1], v) for k, v in result["phases"].items())
            if r["criteria"]["passed"]
        ]
        if passing:
            passing.sort(
                key=lambda item: (
                    -min(c["coverage"] for c in item[1]["criteria"]["cohorts"].values()),
                    item[1]["seconds_per_target"],
                )
            )
            selected = passing[0][0]
            result["selected"] = selected
            result["phases"][f"B:{selected}"] = run_phase(
                "B", selected, result["phases"][f"A:{selected}"]["tau"]
            )
            validated = result["phases"][f"B:{selected}"]["criteria"]["passed"]
            result["status"] = "passed" if validated else "failed_validation"
        else:
            result["selected"] = None
            result["status"] = "failed_selection"
    except (EngineeringCap, BudgetStop) as error:
        status = "budget_stop"
        result["status"] = "budget_stop"
        result["reason"] = str(error)
    except BaseException:
        status = "failed"
        raise
    finally:
        for family in prepared.values():
            family["renderer"].close()
        result["engineering_seconds"] = _stage_seconds(budget, "engineer")
        budget.finish(status)
        result["budget_used_seconds"] = budget.used()
        write_json(root / folder / "result.json", result)
    return {k: result[k] for k in ("status", "engineering_seconds") if k in result} | {
        "selected": result.get("selected")
    }


def engineer_select_v2(root: Path = ROOT) -> dict:
    """D037 spec: the sealed v1 engineering set (re-verified) with the v2 search and rules."""
    v1 = sealed(root / "engineering" / "spec.json")
    if v1["membership_sha256"] != sha256_file(root / "membership.json"):
        raise LineageError("Membership changed since the v1 engineering spec")
    for identifier, family in v1["families"].items():
        if sha256_file(Path(family["path"])) != family["file_sha256"]:
            raise LineageError(f"Engineering family changed: {identifier}")
    spec = {
        **{k: v1[k] for k in ("membership_sha256", "families", "disjointness", "phases")},
        "task": "MEAS-001 engineering version 2 (D037)",
        "v1_spec_sha256": sha256_file(root / "engineering" / "spec.json"),
        "settings": {name: config.to_dict() for name, config in ENGINEERING_SETTINGS_V2.items()},
        "geometry": v1["geometry"],
        "criteria": ENGINEERING,
        "rules": {
            "epsilon": v1.get("rules", {}).get("epsilon", "as v1"),
            "tau_fit": "larger scene-specific 95th percentile of per-family maximum *reported* "
            "(independent-search) residuals over calibration real queries; floor ε",
            "tau_amb": "per query, label-free: least residual increase among 8 block offsets "
            "of 1.28 units around the reported pose (pose_search.resolution)",
            "classification": "pose_search.classify, SearchConfig.version 2",
        },
        "selection_rule": "phase A develops the single declared setting; phase B validates it "
        "once with phase-A τ_fit; stop if either fails",
        "phase_note": "phase A families were inspected in v1 and are development data; phase "
        "B families were rendered in the GALLERY-001 audit but are unused for search",
        "sub_cap_seconds": ENGINEERING_LIMIT,
        "seed": SEEDS["engineering"],
        "predictions_used": False,
    }
    payload = _seal_once(root / "engineering-v2" / "spec.json", spec)
    return {
        "settings": list(payload["settings"]),
        "phases": {k: len(v) for k, v in payload["phases"].items()},
    }


def account_correction(root: Path = ROOT) -> dict:
    """Charge the disclosed, previously uncharged profiling once (MEAS-REVIEW-001)."""
    budget = _budget()
    ledger = budget.load()
    if any(s["stage"] == "engineer-accounting-correction" for s in ledger["sessions"]):
        return {"status": "already charged"}
    now = budget.clock()
    ledger["sessions"].append(
        {
            "id": f"engineer-accounting-correction-{len(ledger['sessions']):04d}",
            "stage": "engineer-accounting-correction",
            "run_root": str(root),
            "started": now,
            "last_heartbeat": now + ACCOUNTING_CORRECTION_SECONDS,
            "ended": now + ACCOUNTING_CORRECTION_SECONDS,
            "status": "correction",
            "note": "conservative charge for implementation profiling (~1 min) and the "
            "uncharged v1 integrity checks; disclosed in MEAS-001_ENGINEERING.md",
        }
    )
    budget._save(ledger)
    return {"status": "charged", "seconds": ACCOUNTING_CORRECTION_SECONDS}


# --------------------------------------------------------------------------- propose / freeze
DISTANCE_BINS = (0.0, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, math.inf)


def engineering_diagnostics(root: Path = ROOT, folder: str = "engineering") -> dict:
    """Failure analysis from saved engineering traces and labels (D033 permits labels for
    debugging engineering runs). No rendering or encoding.

    The distance table summarizes best *visited* residuals per bin across differing
    targets; it is not a controlled landscape sweep and cannot establish identifiability
    (MEAS-REVIEW-001 F4)."""
    engineering = json.loads((root / folder / "result.json").read_text())
    result: dict[str, Any] = {}
    for name, phase in engineering["phases"].items():
        stage, setting = name.split(":", 1)
        rows, curve = [], {i: [] for i in range(len(DISTANCE_BINS) - 1)}
        for record in phase["records"]:
            path = root / folder / stage / setting / record["id"].replace(":", "__")
            inversion, _ = load_inversion(path)
            truth = np.asarray(record["true_block"], float)
            pose, best = inversion.reported if inversion.config.version >= 2 else inversion.best
            arrays = inversion.trace.arrays()
            finite = np.isfinite(arrays["residuals"])
            distance = np.linalg.norm(arrays["poses"][finite][:, BLOCK] - truth, axis=1)
            for i in curve:
                mask = (distance >= DISTANCE_BINS[i]) & (distance < DISTANCE_BINS[i + 1])
                if mask.any():
                    curve[i].append(float(np.min(arrays["residuals"][finite][mask])))
            rows.append(
                {
                    "converged": [s.converged for s in inversion.searches],
                    "restart_distance": record["restart_distance"],
                    "best_error": float(np.linalg.norm(pose[BLOCK] - truth)),
                    "best_residual": best,
                    "true_pose_residual": record["true_pose_residual"],
                }
            )
        errors = np.array([r["best_error"] for r in rows])
        result[name] = {
            "targets": len(rows),
            "both_converged": int(sum(all(r["converged"]) for r in rows)),
            "median_restart_distance": float(np.median([r["restart_distance"] for r in rows])),
            "median_best_error": float(np.median(errors)),
            "best_error_within": {
                str(limit): int(np.sum(errors <= limit)) for limit in (1.28, 2.56, 5.12, 20.48)
            },
            "median_best_residual": float(np.median([r["best_residual"] for r in rows])),
            "median_true_pose_residual": float(np.median([r["true_pose_residual"] for r in rows])),
            "reached_true_basin": int(
                sum(r["best_residual"] <= 1.5 * r["true_pose_residual"] + 1e-9 for r in rows)
            ),
            "better_than_truth_but_far": int(
                sum(
                    r["best_residual"] < r["true_pose_residual"] and r["best_error"] > 5.12
                    for r in rows
                )
            ),
            "tau": phase["tau"],
            "residual_by_block_distance": [
                {
                    "distance": [DISTANCE_BINS[i], DISTANCE_BINS[i + 1]],
                    "targets": len(values),
                    "median_best_residual": float(np.median(values)) if values else None,
                }
                for i, values in curve.items()
            ],
        }
    return result


def propose(root: Path = ROOT, folder: str = "engineering") -> dict:
    """Freeze-ready proposal: settings, seeds, membership, rules, gates and projection."""
    spec = sealed(root / folder / "spec.json")
    engineering = json.loads((root / folder / "result.json").read_text())
    reasons = []
    if engineering["status"] != "passed":
        reasons.append(f"engineering {engineering['status']}")
    selected = engineering.get("selected")
    phases = [v for k, v in engineering["phases"].items() if selected and k.endswith(selected)]
    per_target = float(np.mean([p["seconds_per_target"] for p in phases])) if phases else None
    setup = float(np.mean([p["setup_seconds_per_family"] for p in phases])) if phases else None
    budget = _budget()
    remaining = budget.remaining() - budget.shutdown
    projection = None
    if per_target is not None:
        projection = HEADROOM * (WORKLOAD_TARGETS * per_target + 48 * setup)
        if projection > remaining:
            reasons.append(f"projected {projection / 60:.0f} min exceeds remaining")
    proposal = {
        "task": "MEAS-001",
        "status": "freeze_ready" if not reasons else "not_ready",
        "reasons": reasons,
        "membership_sha256": sha256_file(root / "membership.json"),
        "engineering_spec_sha256": sha256_file(root / folder / "spec.json"),
        "engineering_folder": folder,
        "search": spec["settings"][selected] if selected else None,
        "selected_setting": selected,
        "geometry": spec["geometry"],
        "seeds": SEEDS,
        "rules": spec.get(
            "rules",
            {
                "epsilon": "max MSE between forward, reversed and single-frame encodings of "
                "each family's observed frame and renders; floor 1e-12",
                "tau": "larger scene-specific 95th percentile of per-family maximum best "
                "residuals over calibration real queries; floor ε (pose_search.calibrate_tau)",
                "classification": "pose_search.classify",
            },
        ),
        "gates": GATES,
        "low_motion_cells": LOW_MOTION,
        "budget": {
            "limit_seconds": LIMIT_SECONDS,
            "used_seconds": budget.used(),
            "remaining_seconds": remaining,
            "engineering_seconds": engineering.get("engineering_seconds"),
        },
        "projection": {
            "targets": WORKLOAD_TARGETS,
            "seconds_per_target": per_target,
            "setup_seconds_per_family": setup,
            "headroom": HEADROOM,
            "projected_seconds": projection,
        },
        "sources": source_hashes(),
        "runtime": runtime_info(),
    }
    proposal_path = (
        root / "config-proposed.json"
        if folder == "engineering"
        else root / folder / "proposal.json"
    )
    digest = seal(proposal_path, proposal)
    summary = {
        "task": "MEAS-001 engineering pass (D033)",
        "proposal_sha256": digest,
        "proposal": proposal,
        "engineering": {k: v for k, v in engineering.items() if k != "phases"}
        | {
            "phases": {
                name: {k: v for k, v in phase.items() if k != "records"}
                for name, phase in engineering["phases"].items()
            }
        },
        "disjointness": spec["disjointness"],
        "spec_sha256": proposal["engineering_spec_sha256"],
        "diagnostics": engineering_diagnostics(root, folder),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    name = "meas001-engineering.json" if folder == "engineering" else f"meas001-{folder}.json"
    write_json(MANIFESTS / name, summary)
    return {"status": proposal["status"], "reasons": reasons, "proposal_sha256": digest}


def freeze(root: Path = ROOT, proposal_sha256: str = "", folder: str = "engineering") -> dict:
    """After review: seal the exact reviewed proposal as the frozen configuration."""
    path = (
        root / "config-proposed.json"
        if folder == "engineering"
        else root / folder / "proposal.json"
    )
    if sha256_file(path) != proposal_sha256:
        raise LineageError("The proposal differs from the reviewed SHA-256")
    proposal = sealed(path)
    if proposal["status"] != "freeze_ready":
        raise LineageError("Only a freeze_ready proposal can be frozen")
    if proposal["sources"] != source_hashes():
        raise LineageError("Sources changed after the proposal")
    return _seal_once(root / "config.json", {**proposal, "frozen_from": proposal_sha256})


def require_frozen(root: Path = ROOT) -> dict:
    """Lineage check run by every post-freeze stage and on every resume."""
    config_path = root / "config.json"
    if not config_path.exists():
        raise LineageError("MEAS-001 is not frozen; run `freeze` after review")
    config = sealed(config_path)
    if config["membership_sha256"] != sha256_file(root / "membership.json"):
        raise LineageError("Membership changed after freeze")
    if config["sources"] != source_hashes():
        raise LineageError("Sources changed after freeze")
    membership = sealed(root / "membership.json")
    for family_id, family in membership["families"].items():
        if sha256_file(Path(family["path"])) != family["file_sha256"]:
            raise LineageError(f"Family file changed after freeze: {family_id}")
        if _hash(Path(family["latents"])) != family["latents_sha256"]:
            raise LineageError(f"Saved latents changed after freeze: {family_id}")
    return config


def _verified_family(membership: dict, family_id: str) -> Path:
    family = membership["families"][family_id]
    path = Path(family["path"])
    if sha256_file(path) != family["file_sha256"]:
        raise LineageError(f"Family file changed: {family_id}")
    return path


# --------------------------------------------------------------------------- MEAS queries
def query_inputs(
    membership: dict, query: dict, kind: str = "real"
) -> tuple[SceneContext, np.ndarray]:
    """Scene context and target latent only. Saved latents: start at index 2, then each
    candidate's two future steps (3 + 2·candidate + step). No label is read."""
    family = membership["families"][query["family"]]
    if sha256_file(Path(family["latents"])) != family["latents_sha256"]:
        raise LineageError(f"Saved latents changed: {query['family']}")
    observed = read_npz(
        _verified_family(membership, query["family"]), ["metadata", "warmup_controls"]
    )
    context = SceneContext.from_family(observed, family["scene"])
    key = "real" if kind == "real" else "predicted"
    latents = read_npz(Path(family["latents"]), [key])[key].astype(np.float64)
    if kind == "predicted":
        return context, latents[query["candidate"], 1]
    if query["query"] == "start":
        return context, latents[2]
    return context, latents[3 + 2 * query["candidate"] + 1]


def _labels(membership: dict, query: dict) -> dict:
    """Validator only: simulator pose and replay-identity checks for one query."""
    from jepa_action_effects.gallery_render import replay_endpoint
    from jepa_action_effects.simulator_replay import InstrumentedPushT

    family = load_family(_verified_family(membership, query["family"]))
    if query["query"] == "start":
        pose = np.asarray(family["history_states"][-1][:5], float)
        renderer = SceneRenderer(SceneContext.from_family(family, "validator"))
        try:
            identity = bool(np.array_equal(renderer.baseline_rgb, family["history_frames"][-1]))
            valid = bool(renderer.check(pose[None])[0][0])
        finally:
            renderer.close()
        return {"pose": pose, "replay_identity": identity, "valid": valid}
    env = InstrumentedPushT()
    try:
        replay_endpoint(env, family, query["candidate"])
        identity = True
    except ValueError:
        identity = False
    finally:
        env.close()
    pose = np.asarray(family["future_states"][query["candidate"], 1][:5], float)
    renderer = SceneRenderer(SceneContext.from_family(family, "validator"))
    try:
        valid = bool(renderer.check(pose[None])[0][0])
    finally:
        renderer.close()
    return {"pose": pose, "replay_identity": identity, "valid": valid}


def _invert_queries(
    root: Path,
    config: dict,
    queries: list[dict],
    folder: str,
    budget: Budget,
    model,
    kind: str = "real",
) -> list[dict]:
    membership = sealed(root / "membership.json")
    search = SearchConfig.from_dict(config["search"])
    payloads = []
    for query in queries:
        context, target = query_inputs(membership, query, kind)
        renderer = SceneRenderer(context, **_geometry(config))
        try:
            path = root / folder / query["id"].replace(":", "__")
            family = membership["families"][query["family"]]
            payload = _run_target(
                path,
                make_objective(renderer, model, target),
                search,
                SEEDS["search"],
                f"{folder}:{query['id']}",
                budget,
                {"kind": kind, "family": query["family"], "query": query["query"]},
                {
                    "family_sha256": family["file_sha256"],
                    "latents_sha256": family["latents_sha256"],
                    "kind": kind,
                },
            )
        finally:
            renderer.close()
        payloads.append({**payload, "path": str(path)})
    return payloads


def _geometry(config: dict) -> dict:
    return {
        "tolerance": config["geometry"]["tolerance"],
        "agent_walls": config["geometry"]["agent_walls"],
    }


def _partition(membership: dict, partition: str) -> list[dict]:
    return [
        q
        for q in membership["queries"]
        if membership["families"][q["family"]]["partition"] == partition
    ]


def _session(stage: str, root: Path, work: Callable[[Budget], dict]) -> dict:
    budget = _budget()
    budget.start(stage, str(root))
    status = "complete"
    try:
        return work(budget)
    except BudgetStop:
        status = "budget_stop"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        budget.finish(status)


def benchmark(root: Path = ROOT) -> dict:
    """Throughput on the two benchmark calibration families; a stop, never a tuning signal."""
    config = require_frozen(root)
    membership = sealed(root / "membership.json")
    queries = [q for q in membership["queries"] if q["family"] in membership["roles"]["benchmark"]]

    def work(budget: Budget) -> dict:
        model, _ = _load_model()
        payloads = _invert_queries(root, config, queries, "calibration", budget, model)
        per_target = float(np.mean([p["seconds"] for p in payloads]))
        remaining_targets = WORKLOAD_TARGETS - len(payloads)
        projected = HEADROOM * per_target * remaining_targets
        remaining = budget.remaining() - budget.shutdown
        result = {
            "seconds_per_target": per_target,
            "remaining_targets": remaining_targets,
            "projected_seconds": projected,
            "remaining_seconds": remaining,
            "status": "continue" if projected <= remaining else "stop_infeasible",
        }
        _seal_once(root / "benchmark.json", result)
        if result["status"] != "continue":
            raise SystemExit("MEAS-001 projected workload exceeds the budget; stop (D032)")
        return result

    return _session("benchmark", root, work)


def calibrate(root: Path = ROOT) -> dict:
    """ε and τ from calibration queries only; no labels; sealed before assessment."""
    config = require_frozen(root)
    if sealed(root / "benchmark.json")["status"] != "continue":
        raise LineageError("Benchmark did not clear the budget check")
    membership = sealed(root / "membership.json")
    queries = _partition(membership, "calibration")

    def work(budget: Budget) -> dict:
        model, _ = _load_model()
        payloads = _invert_queries(root, config, queries, "calibration", budget, model)
        epsilons = []
        for family_id, family in membership["families"].items():
            if family["partition"] != "calibration":
                continue
            context = SceneContext.from_family(
                read_npz(Path(family["path"]), ["metadata", "warmup_controls"]), family["scene"]
            )
            renderer = SceneRenderer(context, **_geometry(config))
            try:
                rng = np.random.default_rng(seed_sequence(SEEDS["search"], family_id, 99))
                search = SearchConfig.from_dict(config["search"])
                lower, upper = np.array(search.lower), np.array(search.upper)
                poses = lower + rng.random((64, 5)) * (upper - lower)
                poses = poses[renderer.check(poses)[0]][:15]
                frames = np.concatenate([renderer.baseline_rgb[None], renderer.render(poses)])
                epsilons.append(encoder_floor(model, frames))
            finally:
                renderer.close()
        epsilon = max(epsilons)
        best, rows = {}, []
        for payload in payloads:
            value = best_residual(Path(payload["path"]))
            best[payload["query_id"]] = value if math.isfinite(value) else None
            scene = membership["families"][payload["family"]]["scene"]
            rows.append((scene, payload["family"], value))
        maxima = family_maxima(rows)
        try:
            tau = calibrate_tau(maxima, epsilon)
            status = "sealed"
        except CalibrationFailure as error:
            tau, status = None, f"failed: {error}"
        maxima = {k: [v if math.isfinite(v) else None for v in vs] for k, vs in maxima.items()}
        result = {
            "epsilon": epsilon,
            "tau": tau,
            "status": status,
            "family_maxima": maxima,
            "best": best,
        }
        _seal_once(root / "calibration.json", result)
        if tau is None:
            raise SystemExit("Threshold calibration failed; assessment stays closed (D032)")
        return {"epsilon": epsilon, "tau": tau}

    return _session("calibrate", root, work)


def known_failure_controls() -> dict:
    """Analytic objectives that must not yield a confident output (D032 §6)."""

    class Flat:
        def valid(self, poses):
            return np.ones(len(poses), dtype=bool)

        def evaluate(self, poses):
            return np.full(len(poses), 0.5)

    class TwoMinima(Flat):
        def evaluate(self, poses):
            first = np.sum((poses[:, :4] - [100, 300, 250, 260]) ** 2, axis=1)
            second = np.sum((poses[:, :4] - [100, 300, 310, 260]) ** 2, axis=1)
            return 1e-3 * np.minimum(first, second)

    tiny = SearchConfig(coarse=64, local_starts=1, local_evaluations=12, nuisance_evaluations=12)
    outcomes = {
        "flat": classify(invert(Flat(), SearchConfig(), 1, "flat"), 1e-3)["status"],
        "two_minima": classify(invert(TwoMinima(), SearchConfig(), 1, "two"), 0.05)["status"],
        "failed_search": classify(invert(TwoMinima(), tiny, 1, "tiny"), 10.0)["status"],
    }
    return {"outcomes": outcomes, "passed": all(v != "accepted" for v in outcomes.values())}


def controls(root: Path = ROOT) -> dict:
    """24 synthetic targets on six calibration families; recorded, never ground truth for
    mixtures. Requires sealed calibration."""
    config = require_frozen(root)
    calibration = sealed(root / "calibration.json")
    if calibration["tau"] is None:
        raise LineageError("Calibration failed; controls and assessment stay closed")
    membership = sealed(root / "membership.json")
    families = membership["roles"]["controls"]
    search = SearchConfig.from_dict(config["search"])

    def work(budget: Budget) -> dict:
        model, _ = _load_model()
        rng = np.random.default_rng(SEEDS["controls"])
        genuine: list[tuple[str, SceneRenderer, np.ndarray, np.ndarray]] = []
        records = []
        try:
            for family_id in families:
                family = membership["families"][family_id]
                observed = read_npz(Path(family["path"]), ["metadata", "warmup_controls"])
                renderer = SceneRenderer(
                    SceneContext.from_family(observed, family["scene"]), **_geometry(config)
                )
                lower, upper = np.array(search.lower), np.array(search.upper)
                for _ in range(200):
                    pose = lower + rng.random(5) * (upper - lower)
                    if renderer.check(pose[None])[0][0]:
                        break
                else:
                    raise RuntimeError(f"No valid control pose for {family_id}")
                latent = encode(model, renderer.render(pose[None]))[0]
                genuine.append((family_id, renderer, pose, latent))
            separation = float(
                np.median(
                    [
                        np.sqrt(np.mean((a[3] - b[3]) ** 2))
                        for a in genuine
                        for b in genuine
                        if a is not b
                    ]
                )
            )
            targets = []
            for family_id, renderer, pose, latent in genuine:
                targets.append((f"{family_id}:genuine", renderer, latent, {"pose": pose.tolist()}))
            shifts = [(2, 10.24), (4, math.radians(20.0)), (0, 30.0), (3, 20.48)]
            for (family_id, renderer, pose, latent), (axis, delta) in zip(
                genuine[:4], shifts, strict=True
            ):
                other = pose.copy()
                other[axis] += delta
                if not renderer.check(other[None])[0][0]:
                    other[axis] -= 2 * delta
                other_latent = encode(model, renderer.render(other[None]))[0]
                for weight in (0.25, 0.5, 0.75):
                    mixture = weight * latent + (1 - weight) * other_latent
                    targets.append(
                        (
                            f"{family_id}:mixture-{axis}-{weight}",
                            renderer,
                            mixture,
                            {"pose_a": pose.tolist(), "pose_b": other.tolist(), "weight": weight},
                        )
                    )
            for index, (family_id, renderer, pose, latent) in enumerate(genuine):
                scale = (0.25, 0.5)[index % 2] * separation
                direction = rng.standard_normal(latent.shape)
                direction /= np.sqrt(np.mean(direction**2))
                targets.append(
                    (
                        f"{family_id}:perturbed-{scale:.4f}",
                        renderer,
                        latent + scale * direction,
                        {"pose": pose.tolist(), "rms_scale": scale},
                    )
                )
            for target_id, renderer, latent, truth in targets:
                path = root / "controls" / target_id.replace(":", "__")
                _run_target(
                    path,
                    make_objective(renderer, model, latent),
                    search,
                    SEEDS["controls"],
                    f"controls:{target_id}",
                    budget,
                    {"truth": truth},
                )
                inversion, _ = load_inversion(path)
                records.append(
                    {"id": target_id, "truth": truth, **classify(inversion, calibration["tau"])}
                )
        finally:
            for _, renderer, _, _ in genuine:
                renderer.close()
        result = {"records": records, "known_failure": known_failure_controls()}
        _seal_once(root / "controls.json", json.loads(json.dumps(result, default=float)))
        return {"targets": len(records), "known_failure_passed": result["known_failure"]["passed"]}

    return _session("controls", root, work)


def assess(root: Path = ROOT) -> dict:
    """The 24 assessment families, once; τ and controls must already be sealed."""
    config = require_frozen(root)
    calibration = sealed(root / "calibration.json")
    sealed(root / "controls.json")
    if calibration["tau"] is None:
        raise LineageError("Calibration failed; assessment stays closed")
    if (root / "assessment-complete.json").exists():
        raise LineageError("Assessment already ran; a rerun needs a new version (D033)")
    membership = sealed(root / "membership.json")

    def work(budget: Budget) -> dict:
        model, _ = _load_model()
        payloads = _invert_queries(
            root, config, _partition(membership, "assessment"), "assessment", budget, model
        )
        result = {"queries": len(payloads), "tau": calibration["tau"]}
        _seal_once(root / "assessment-complete.json", result)
        return result

    return _session("assess", root, work)


def score(root: Path = ROOT) -> dict:
    """Validator scoring with labels, integrity checks and the D032 gates (charged)."""
    return _session("score", root, lambda budget: _score_stage(root))


def _score_stage(root: Path) -> dict:
    config = require_frozen(root)
    calibration = sealed(root / "calibration.json")
    controls_result = sealed(root / "controls.json")
    complete = (root / "assessment-complete.json").exists()
    membership = sealed(root / "membership.json")
    integrity = {"replay": True, "sources": config["sources"] == source_hashes()}
    records: dict[str, dict] = {}
    for query in _partition(membership, "assessment"):
        path = root / "assessment" / query["id"].replace(":", "__")
        if not path.with_suffix(".json").exists():
            continue
        inversion, _ = load_inversion(path)
        result = classify(inversion, calibration["tau"])
        labels = _labels(membership, query)
        integrity["replay"] = integrity["replay"] and labels["replay_identity"]
        best = result.get("best_pose")
        records[query["id"]] = {
            "family": query["family"],
            "scene": membership["families"][query["family"]]["scene"],
            "query": query["query"],
            "status": result["status"],
            "true_block": labels["pose"][BLOCK].tolist(),
            "best_block": None if best is None else list(best[2:4]),
            "unsupported_true_geometry": not labels["valid"],
            **({"block_center": result["block_center"]} if result["status"] == "accepted" else {}),
        }
    endpoint_cells, displacement_cells = {}, {}
    for scene in SCENES:
        families = sorted(
            f
            for f, v in membership["families"].items()
            if v["scene"] == scene and v["partition"] == "assessment"
        )
        rows = [r for r in records.values() if r["scene"] == scene]
        starts = [r for r in rows if r["query"] == "start"]
        endpoint_cells[f"{scene}:start"] = endpoint_cell(starts, families)
        for label, _, role in ENDPOINTS[scene]:
            ends = [r for r in rows if r["query"] == label]
            cell = f"{scene}:{label}"
            summary = endpoint_cell(ends, families)
            moving = displacement_cell(starts, ends, families)
            if role == "low_motion":
                endpoint_cells.setdefault("_low_motion", {})[cell] = {
                    **summary,
                    "displacement": moving,
                }
                continue
            endpoint_cells[cell] = summary
            displacement_cells[cell] = moving
    low_motion = endpoint_cells.pop("_low_motion", {})
    integrity_passed = all(integrity.values())
    outcome = decision(
        endpoint_cells,
        displacement_cells,
        integrity_passed,
        controls_result["known_failure"]["passed"],
        complete,
    )
    result = {
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "endpoint_cells": endpoint_cells,
        "displacement_cells": displacement_cells,
        "low_motion": low_motion,
        "decision": outcome,
        "predictions_allowed": predictions_allowed(
            outcome,
            endpoint_cells,
            displacement_cells,
            integrity_passed,
            complete,
            controls_result["known_failure"]["passed"],
        ),
        "records": records,
    }
    _seal_once(root / "scores.json", json.loads(json.dumps(result, default=float)))
    return {"decision": outcome, "integrity_passed": integrity_passed}


def predictions(root: Path = ROOT) -> dict:
    """Optional D032 §7 check on 16 saved frozen-predictor latents; skipped unless allowed."""
    config = require_frozen(root)
    scores = sealed(root / "scores.json")
    membership = sealed(root / "membership.json")
    if not scores["predictions_allowed"]:
        return _seal_once(
            root / "predictions.json",
            {"status": "skipped", "reason": "D032 §7 cells did not all pass"},
        )
    queries = []
    for item in membership["roles"]["predictions"]:
        family, label = item.rsplit(":", 1)
        query = next(q for q in membership["queries"] if q["id"] == item)
        queries.append({**query, "id": f"{family}:{label}:predicted", "family": family})

    def work(budget: Budget) -> dict:
        model, _ = _load_model()
        payloads = _invert_queries(
            root, config, queries, "predictions", budget, model, kind="predicted"
        )
        tau = sealed(root / "calibration.json")["tau"]
        records = []
        for payload in payloads:
            inversion, _ = load_inversion(Path(payload["path"]))
            records.append({"id": payload["query_id"], **classify(inversion, tau)})
        result = {"status": "complete", "records": records}
        _seal_once(root / "predictions.json", json.loads(json.dumps(result, default=float)))
        return {"targets": len(records)}

    return _session("predictions", root, work)


def report(root: Path = ROOT) -> dict:
    """Manifest and draft; refuses to write anything unless integrity passed."""
    config = require_frozen(root)
    scores = sealed(root / "scores.json")
    if not scores["integrity_passed"]:
        raise SystemExit("MEAS-001 integrity failed; no report written")
    budget = _budget()
    summary = {
        "task": "MEAS-001",
        "status": "feasibility study (D032, D033); not a physical-mechanism result",
        "config_sha256": sha256_file(root / "config.json"),
        "decision": scores["decision"],
        "endpoint_cells": scores["endpoint_cells"],
        "displacement_cells": scores["displacement_cells"],
        "low_motion": scores["low_motion"],
        "calibration": {
            k: v
            for k, v in sealed(root / "calibration.json").items()
            if k in ("epsilon", "tau", "status")
        },
        "controls": sealed(root / "controls.json"),
        "predictions": sealed(root / "predictions.json")
        if (root / "predictions.json").exists()
        else None,
        "budget": {"used_seconds": budget.used(), "limit_seconds": LIMIT_SECONDS},
        "sources": config["sources"],
        "runtime": runtime_info(),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "meas001-runtime.json", json.loads(json.dumps(summary, default=float)))
    return {"decision": scores["decision"]}


STAGES: dict[str, Callable[..., dict]] = {
    "select": select,
    "engineer-select": engineer_select,
    "engineer": engineer,
    "propose": propose,
    "account-correction": account_correction,
    "engineer-select-v2": engineer_select_v2,
    "engineer-v2": lambda root: engineer(root, "engineering-v2"),
    "propose-v2": lambda root: propose(root, "engineering-v2"),
    "benchmark": benchmark,
    "calibrate": calibrate,
    "controls": controls,
    "assess": assess,
    "score": score,
    "predictions": predictions,
    "report": report,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=[*STAGES, "freeze"])
    parser.add_argument("--proposal-sha256", default="")
    args = parser.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze":
        result = freeze(ROOT, args.proposal_sha256)
        print(json.dumps({"frozen": True, "status": result["status"]}))
        return
    print(json.dumps(STAGES[args.stage](ROOT), default=str, indent=2)[:4000])


if __name__ == "__main__":
    main()
