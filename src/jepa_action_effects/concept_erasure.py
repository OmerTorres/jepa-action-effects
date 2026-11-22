"""ERASURE-001 staged selective linear concept-erasure feasibility study."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from jepa_action_effects.concept_erasure_metrics import (
    LinearEraser,
    bootstrap_gain,
    family_weights,
    fit_leace,
    interval,
    random_eraser,
    score_probe,
    select_probe,
)
from jepa_action_effects.interface_audit import CONFIG_SHA256, WEIGHTS_SHA256
from jepa_action_effects.latent_cache import LatentCache, array_digest
from jepa_action_effects.pilot_collection import (
    canonical_json,
    load_family,
    sha256_bytes,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.readout_recovery import _partition_ids, _tiers

ROOT = Path("outputs/erasure001")
PILOT = Path("outputs/pilot001")
READOUT = Path("outputs/readout001")
FRESH = READOUT / "collection"
CHECKPOINT = Path("checkpoints/lewm-pusht")
MASTER_SEED = 20261002
EXPERIMENT_LIMIT = 3480.0
TOTAL_LIMIT = 3600.0
RTOL = 1e-10
CONCEPTS = {"agent": slice(0, 2), "block": slice(2, 4)}


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def seal(path: Path, payload: Any) -> str:
    digest = write_json(path, payload)
    path.with_name(path.name + ".sha256").write_text(digest + "\n")
    return digest


def sealed(path: Path) -> Any:
    digest_path = path.with_name(path.name + ".sha256")
    if sha256_file(path) != digest_path.read_text().strip():
        raise ValueError(f"Sealed artifact mismatch: {path}")
    return json.loads(path.read_text())


def _sources() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "metrics": Path(__file__).with_name("concept_erasure_metrics.py"),
        "plan": Path("docs/ERASURE-001_PLAN.md"),
        "lock": Path("uv.lock"),
        "model_inference": Path(__file__).with_name("model_inference.py"),
        "readout_recovery": Path(__file__).with_name("readout_recovery.py"),
        "pilot_manifest": PILOT / "collection-manifest.json",
        "fresh_manifest": FRESH / "collection-manifest.json",
        "weights": CHECKPOINT / "weights.pt",
        "checkpoint_config": CHECKPOINT / "config.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _streams() -> dict[str, int]:
    names = ("partition", "control", "nonlinear", "verification", "bootstrap")
    return {
        name: int(child.generate_state(1)[0])
        for name, child in zip(names, np.random.SeedSequence(MASTER_SEED).spawn(5), strict=True)
    }


def _partition(pilot: dict, fresh: dict) -> dict[str, list[str]]:
    parts = _partition_ids(pilot, fresh)
    fit = _tiers(parts)[240]
    tuning = parts["tuning"]
    excluded = parts["confirmation"]
    rng = np.random.default_rng(_streams()["partition"])
    calibration, assessment = [], []
    for cohort, expected in (("challenge", 40), ("ordinary", 20)):
        ids = sorted(item for item in tuning if item.split(":", 1)[1].startswith(cohort))
        if len(ids) != expected:
            raise ValueError("Tuning cohort membership changed")
        shuffled = np.asarray(ids)[rng.permutation(len(ids))].tolist()
        calibration.extend(shuffled[: expected // 2])
        assessment.extend(shuffled[expected // 2 :])
    groups = {
        "fit": sorted(fit),
        "calibration": sorted(calibration),
        "assessment": sorted(assessment),
        "excluded_confirmation": sorted(excluded),
    }
    flat = (
        groups["fit"]
        + groups["calibration"]
        + groups["assessment"]
        + groups["excluded_confirmation"]
    )
    if len(flat) != len(set(flat)) or list(map(len, groups.values())) != [240, 30, 30, 60]:
        raise ValueError("ERASURE-001 partitions overlap or have wrong counts")
    return groups


def _family_path(identifier: str, pilot: dict, fresh: dict) -> Path:
    source, family_id = identifier.split(":", 1)
    root, manifest = (PILOT, pilot) if source == "pilot" else (FRESH, fresh)
    return root / manifest["family_files"][family_id]["path"]


def _cache_index() -> dict[str, Path]:
    result = {}
    for path in sorted((READOUT / "latent-cache" / "real").glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["manifest"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid upstream cache entry: {path}") from error
        fingerprint = metadata.get("fingerprint", {})
        identifier = metadata.get("identifiers", {}).get("identifier")
        if fingerprint.get("representation") == "projected" and identifier:
            if identifier in result:
                raise ValueError(f"Duplicate projected cache identity: {identifier}")
            result[identifier] = path
    return result


def _read_projected(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["manifest"].item()))
    cache = LatentCache(path.parent.parent)
    values, _ = cache.read("real", metadata["key"], metadata["fingerprint"])
    if values.shape != (1, 15, 192):
        raise ValueError("Expected cached projected real embeddings with shape (1,15,192)")
    return values[0, :3].astype(np.float64)


def _frame_digest(frame: np.ndarray) -> str:
    return array_digest(np.asarray(frame, dtype=np.uint8))


def _deduplicated(identifier: str, family: dict, embeddings: np.ndarray) -> list[dict[str, Any]]:
    seen = {}
    rows = []
    for index, (frame, state, embedding) in enumerate(
        zip(family["history_frames"], family["history_states"], embeddings, strict=True)
    ):
        digest = _frame_digest(frame)
        if digest in seen:
            previous = rows[seen[digest]]
            if not np.array_equal(previous["state"], state):
                raise ValueError(f"Conflicting labels for identical history frame in {identifier}")
            continue
        seen[digest] = len(rows)
        rows.append(
            {
                "identifier": identifier,
                "frame_index": index,
                "frame_digest": digest,
                "state": np.asarray(state, dtype=np.float64),
                "embedding": np.asarray(embedding, dtype=np.float64),
            }
        )
    return rows


def _load_rows(groups: dict, pilot: dict, fresh: dict) -> tuple[dict[str, list[dict]], dict]:
    index = _cache_index()
    rows = {}
    global_frames: dict[str, tuple[str, str, np.ndarray]] = {}
    cross_partition = []
    counts = {}
    for partition in ("fit", "calibration", "assessment"):
        rows[partition] = []
        for identifier in groups[partition]:
            if identifier not in index:
                raise ValueError(f"Missing projected upstream cache entry for {identifier}")
            family = load_family(_family_path(identifier, pilot, fresh))
            family_rows = _deduplicated(identifier, family, _read_projected(index[identifier]))
            for row in family_rows:
                digest = row["frame_digest"]
                if digest in global_frames:
                    old_partition, old_id, old_state = global_frames[digest]
                    if old_partition != partition:
                        cross_partition.append([old_partition, old_id, partition, identifier])
                    if not np.array_equal(old_state, row["state"]):
                        raise ValueError("Identical history frame has conflicting state labels")
                else:
                    global_frames[digest] = (partition, identifier, row["state"])
                rows[partition].append(row)
        counts[partition] = len(rows[partition])
    if cross_partition:
        raise ValueError(f"Cross-partition identical frames detected: {cross_partition[:3]}")
    return rows, {"deduplicated_rows": counts, "cross_partition_duplicate_count": 0}


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([row["embedding"] for row in rows]),
        np.stack([row["state"] for row in rows]),
        np.asarray([row["identifier"] for row in rows]),
    )


def audit(root: Path = ROOT) -> dict:
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "audit.json").exists():
        result = sealed(root / "audit.json")
        if result["sources"] != _sources() or result["config"] != sealed(root / "config.json"):
            raise ValueError("Audited source or configuration changed")
        return result
    pilot, fresh = verify_collection(PILOT), verify_collection(FRESH)
    if (
        sha256_file(CHECKPOINT / "weights.pt") != WEIGHTS_SHA256
        or sha256_file(CHECKPOINT / "config.json") != CONFIG_SHA256
    ):
        raise ValueError("Checkpoint identity changed")
    groups = _partition(pilot, fresh)
    config = {
        "task": "ERASURE-001",
        "master_seed": MASTER_SEED,
        "seed_streams": _streams(),
        "representation": "projected_cls_192",
        "history_frames_only": True,
        "leace_rtol": RTOL,
        "ridge_alphas": [0.01, 0.1, 1, 10, 100, 1000],
        "random_controls": 10,
        "shuffled_controls": 10,
        "bootstrap_resamples": 2000,
        "gates": {"accessibility": 0.50, "removal": 0.10, "retention": 0.90},
        "nonlinear": {
            "hidden": 128,
            "seeds": [
                int(value)
                for value in np.random.SeedSequence(_streams()["nonlinear"]).generate_state(3)
            ],
            "maximum_epochs": 1000,
            "patience": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "budget_seconds": {"experiment": EXPERIMENT_LIMIT, "total": TOTAL_LIMIT},
        "groups": groups,
    }
    seal(root / "config.json", config)
    rows, duplicate_report = _load_rows(groups, pilot, fresh)
    verification_rng = np.random.default_rng(_streams()["verification"])
    verification = []
    for cohort in ("challenge", "ordinary"):
        eligible = [
            item
            for item in groups["fit"] + groups["calibration"] + groups["assessment"]
            if item.split(":", 1)[1].startswith(cohort)
        ]
        verification.extend(sorted(verification_rng.choice(eligible, 6, replace=False).tolist()))
    result = {
        "status": "passed",
        "sources": _sources(),
        "config": config,
        "checkpoint": {"weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256},
        "upstream": {
            "pilot_manifest_sha256": sha256_file(PILOT / "collection-manifest.json"),
            "fresh_manifest_sha256": sha256_file(FRESH / "collection-manifest.json"),
            "readout_cache_entries": len(_cache_index()),
        },
        "verification_family_ids": sorted(verification),
        "duplicate_report": duplicate_report,
        "audit_seconds": time.monotonic() - started,
    }
    seal(root / "audit.json", result)
    return result


class Budget:
    def __init__(self, root: Path, signature: str):
        self.path = root / "runtime.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["signature"] != signature:
                raise ValueError("Runtime signature mismatch")
            running = self.data.pop("running", None)
            if running:
                self.data["used_seconds"] += max(0.0, time.time() - running["started"])
                self.data["interruptions"].append(running["stage"])
                write_json(self.path, self.data)
        else:
            self.data = {
                "signature": signature,
                "used_seconds": sealed(root / "audit.json")["audit_seconds"],
                "stages": {},
                "interruptions": [],
            }
            write_json(self.path, self.data)
        self.started = None

    def check(self, reporting: bool = False) -> None:
        active = 0.0 if self.started is None else time.monotonic() - self.started
        limit = TOTAL_LIMIT if reporting else EXPERIMENT_LIMIT
        if self.data["used_seconds"] + active >= limit:
            raise TimeoutError("ERASURE-001 cumulative runtime budget exhausted")

    def begin(self, stage: str) -> None:
        self.check(stage == "report")
        self.stage = stage
        self.started = time.monotonic()
        self.data["running"] = {"stage": stage, "started": time.time()}
        write_json(self.path, self.data)

    def finish(self, status: str) -> None:
        elapsed = time.monotonic() - self.started
        self.data["used_seconds"] += elapsed
        self.data["stages"].setdefault(self.stage, []).append(
            {"status": status, "seconds": elapsed}
        )
        self.data.pop("running", None)
        self.started = None
        write_json(self.path, self.data)


def _signature(root: Path) -> str:
    audit_result = sealed(root / "audit.json")
    return sha256_bytes(
        canonical_json(
            {"audit": sha256_file(root / "audit.json"), "sources": audit_result["sources"]}
        )
    )


def _eraser_arrays(eraser: LinearEraser) -> dict[str, np.ndarray]:
    return {
        "mean": eraser.mean,
        "matrix": eraser.matrix,
        "removed_rank": np.asarray(eraser.removed_rank),
        "covariance_rank": np.asarray(eraser.covariance_rank),
        "rtol": np.asarray(eraser.rtol),
    }


def _load_eraser(path: Path) -> LinearEraser:
    with np.load(path, allow_pickle=False) as data:
        return LinearEraser(
            data["mean"],
            data["matrix"],
            int(data["removed_rank"]),
            int(data["covariance_rank"]),
            float(data["rtol"]),
        )


def fit(root: Path = ROOT) -> dict:
    audit_result = sealed(root / "audit.json")
    signature = _signature(root)
    report_path = root / "fit.json"
    if report_path.exists():
        result = sealed(report_path)
        if result["signature"] != signature:
            raise ValueError("Fit signature mismatch")
        return result
    budget = Budget(root, signature)
    budget.begin("fit")
    try:
        pilot, fresh = verify_collection(PILOT), verify_collection(FRESH)
        rows, _ = _load_rows(audit_result["config"]["groups"], pilot, fresh)
        x, states, ids = _arrays(rows["fit"])
        maps = root / "erasers"
        maps.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "signature": signature,
            "status": "complete",
            "concepts": {},
            "controls": {},
        }
        control_rng = np.random.default_rng(_streams()["control"])
        for concept, target_slice in CONCEPTS.items():
            eraser, diagnostics = fit_leace(x, states[:, target_slice], ids, RTOL)
            path = maps / f"{concept}.npz"
            np.savez_compressed(path, **_eraser_arrays(eraser))
            perturbation = np.linalg.norm(eraser.transform(x) - x, axis=1)
            result["concepts"][concept] = {
                **diagnostics,
                "path": str(path),
                "sha256": sha256_file(path),
                "mean_perturbation": float(np.sum(perturbation * family_weights(ids))),
                "maximum_perturbation": float(np.max(perturbation)),
            }
            result["controls"][concept] = {"random": [], "shuffled": []}
            for index in range(10):
                random_map = random_eraser(
                    eraser.mean, x.shape[1], eraser.removed_rank, control_rng
                )
                random_path = maps / f"{concept}-random-{index:02}.npz"
                np.savez_compressed(random_path, **_eraser_arrays(random_map))
                result["controls"][concept]["random"].append(
                    {"path": str(random_path), "sha256": sha256_file(random_path)}
                )
            family_list = np.unique(ids)
            cohorts = np.asarray([item.split(":", 1)[1].split("-", 1)[0] for item in family_list])
            for index in range(10):
                mapping = {}
                for cohort in ("challenge", "ordinary"):
                    members = family_list[cohorts == cohort]
                    donors = members[control_rng.permutation(len(members))]
                    mapping.update(dict(zip(members.tolist(), donors.tolist(), strict=True)))
                donor_labels = np.empty((len(states), 2), dtype=np.float64)
                for family in family_list:
                    recipient_rows = np.flatnonzero(ids == family)
                    source_rows = np.flatnonzero(ids == mapping[family])
                    # Histories contain one to three unique frames. Preserve temporal recipes.
                    # Resample donor sequences at relative positions when lengths differ.
                    picks = np.rint(
                        np.linspace(0, len(source_rows) - 1, len(recipient_rows))
                    ).astype(int)
                    donor_labels[recipient_rows] = states[source_rows[picks], target_slice]
                shuffled, diagnostics = fit_leace(x, donor_labels, ids, RTOL)
                shuffled_path = maps / f"{concept}-shuffled-{index:02}.npz"
                np.savez_compressed(shuffled_path, **_eraser_arrays(shuffled))
                result["controls"][concept]["shuffled"].append(
                    {
                        "path": str(shuffled_path),
                        "sha256": sha256_file(shuffled_path),
                        "removed_rank": shuffled.removed_rank,
                        "cross_covariance_after_fro": diagnostics["cross_covariance_after_fro"],
                    }
                )
        seal(report_path, result)
        budget.finish("complete")
        return result
    except Exception:
        budget.finish("failed")
        raise


class NonlinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, values):
        return self.network(values)


def _nonlinear(
    name: str,
    eraser: LinearEraser | None,
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    seeds: list[int],
    root: Path,
    budget: Budget,
) -> list[dict]:
    train_x, train_states, train_ids = datasets["fit"]
    calibration_x, calibration_states, calibration_ids = datasets["calibration"]
    assessment_x, assessment_states, assessment_ids = datasets["assessment"]
    transform = (lambda value: value) if eraser is None else eraser.transform
    train_x = transform(train_x)
    calibration_x = transform(calibration_x)
    assessment_x = transform(assessment_x)
    weights_np = family_weights(train_ids)
    feature_mean = np.sum(train_x * weights_np[:, None], axis=0)
    feature_scale = np.sqrt(np.sum((train_x - feature_mean) ** 2 * weights_np[:, None], axis=0))
    feature_scale[feature_scale == 0] = 1
    targets = train_states[:, :4]
    target_mean = np.sum(targets * weights_np[:, None], axis=0)
    target_scale = np.sqrt(np.sum((targets - target_mean) ** 2 * weights_np[:, None], axis=0))
    target_scale[target_scale == 0] = 1
    train_tensor = torch.tensor((train_x - feature_mean) / feature_scale, dtype=torch.float32)
    target_tensor = torch.tensor((targets - target_mean) / target_scale, dtype=torch.float32)
    weights = torch.tensor(weights_np, dtype=torch.float32)
    calibration_tensor = torch.tensor(
        (calibration_x - feature_mean) / feature_scale, dtype=torch.float32
    )
    output = []
    for seed in seeds:
        budget.check()
        torch.manual_seed(seed)
        model = NonlinearProbe()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_epoch, best_state, stale, curve = float("inf"), 0, None, 0, []
        for epoch in range(1, 1001):
            model.train()
            optimizer.zero_grad()
            loss = (torch.mean((model(train_tensor) - target_tensor) ** 2, dim=1) * weights).sum()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                prediction = model(calibration_tensor).numpy() * target_scale + target_mean
            score = float(
                np.mean(
                    [
                        np.mean(
                            (
                                prediction[calibration_ids == family]
                                - calibration_states[calibration_ids == family, :4]
                            )
                            ** 2
                        )
                        for family in np.unique(calibration_ids)
                    ]
                )
            )
            if score < best - 1e-4:
                best, best_epoch = score, epoch
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                curve.append(
                    {"epoch": epoch, "train_loss": float(loss.detach()), "calibration_mse": score}
                )
                budget.check()
            if stale >= 100:
                break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            prediction = (
                model(
                    torch.tensor((assessment_x - feature_mean) / feature_scale, dtype=torch.float32)
                ).numpy()
                * target_scale
                + target_mean
            )
        scores = {}
        for concept, target_slice in CONCEPTS.items():
            constant = np.sum(train_states[:, target_slice] * weights_np[:, None], axis=0)
            selected_prediction = prediction[:, target_slice]
            dummy = type(
                "ArrayProbe",
                (),
                {"predict": lambda self, _, value=selected_prediction: value},
            )()
            scores[concept] = score_probe(
                dummy, assessment_x, assessment_states[:, target_slice], assessment_ids, constant
            )
        curve_path = root / "nonlinear" / f"{name}-{seed}.curve.json"
        write_json(curve_path, {"curve": curve})
        model_path = root / "nonlinear" / f"{name}-{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        output.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs": epoch,
                "best_calibration_mse": best,
                "assessment": scores,
                "curve": str(curve_path),
                "model": str(model_path),
            }
        )
    return output


def validate(root: Path = ROOT) -> dict:
    audit_result, fit_result = sealed(root / "audit.json"), sealed(root / "fit.json")
    signature = _signature(root)
    if fit_result["signature"] != signature:
        raise ValueError("Fit and audit signatures differ")
    report_path = root / "validation.json"
    if report_path.exists():
        result = sealed(report_path)
        if result["signature"] != signature:
            raise ValueError("Validation signature mismatch")
        return result
    budget = Budget(root, signature)
    budget.begin("validate")
    opening = {
        "signature": signature,
        "opened_at_unix": time.time(),
        "assessment_ids": audit_result["config"]["groups"]["assessment"],
        "fit_sha256": sha256_file(root / "fit.json"),
    }
    seal(root / "assessment-opening.json", opening)
    try:
        pilot, fresh = verify_collection(PILOT), verify_collection(FRESH)
        rows, _ = _load_rows(audit_result["config"]["groups"], pilot, fresh)
        datasets = {name: _arrays(value) for name, value in rows.items()}
        train_x, train_states, train_ids = datasets["fit"]
        calibration_x, calibration_states, calibration_ids = datasets["calibration"]
        assessment_x, assessment_states, assessment_ids = datasets["assessment"]
        transformations: dict[str, LinearEraser | None] = {"identity": None}
        for concept in CONCEPTS:
            transformations[concept] = _load_eraser(Path(fit_result["concepts"][concept]["path"]))
            for kind in ("random", "shuffled"):
                for index, item in enumerate(fit_result["controls"][concept][kind]):
                    transformations[f"{concept}_{kind}_{index:02}"] = _load_eraser(
                        Path(item["path"])
                    )
        result: dict[str, Any] = {
            "status": "complete",
            "signature": signature,
            "linear": {},
            "nonlinear": {},
            "gates": {},
        }
        predictions = {}
        bootstrap_seed = _streams()["bootstrap"]
        for transform_index, (name, eraser) in enumerate(transformations.items()):
            budget.check()
            apply = (lambda value: value) if eraser is None else eraser.transform
            result["linear"][name] = {
                "transformation": {
                    "removed_rank": 0 if eraser is None else eraser.removed_rank,
                    "fit_mean_perturbation": 0.0
                    if eraser is None
                    else float(
                        np.sum(
                            np.linalg.norm(eraser.transform(train_x) - train_x, axis=1)
                            * family_weights(train_ids)
                        )
                    ),
                }
            }
            predictions[name] = {}
            for concept_index, (concept, target_slice) in enumerate(CONCEPTS.items()):
                probe, alpha_scores = select_probe(
                    apply(train_x),
                    train_states[:, target_slice],
                    train_ids,
                    apply(calibration_x),
                    calibration_states[:, target_slice],
                    calibration_ids,
                )
                constant = np.sum(
                    train_states[:, target_slice] * family_weights(train_ids)[:, None], axis=0
                )
                concept_result = {
                    "selected_alpha": probe.alpha,
                    "calibration_alpha_scores": alpha_scores,
                    "assessment": {},
                }
                for cohort in ("overall", "challenge", "ordinary"):
                    mask = (
                        np.ones(len(assessment_ids), dtype=bool)
                        if cohort == "overall"
                        else np.asarray(
                            [item.split(":", 1)[1].startswith(cohort) for item in assessment_ids]
                        )
                    )
                    ids_subset = assessment_ids[mask]
                    x_subset = apply(assessment_x[mask])
                    y_subset = assessment_states[mask, target_slice]
                    scored = score_probe(probe, x_subset, y_subset, ids_subset, constant)
                    pred = probe.predict(x_subset)
                    boot = bootstrap_gain(
                        pred,
                        y_subset,
                        ids_subset,
                        constant,
                        bootstrap_seed
                        + transform_index * 100
                        + concept_index * 10
                        + (0 if cohort == "overall" else 1 if cohort == "challenge" else 2),
                        strata=np.asarray(
                            [item.split(":", 1)[1].split("-", 1)[0] for item in ids_subset]
                        )
                        if cohort == "overall"
                        else None,
                    )
                    scored["gain_interval"] = interval(boot)
                    scored["family_count"] = int(len(np.unique(ids_subset)))
                    concept_result["assessment"][cohort] = scored
                predictions[name][concept] = probe.predict(apply(assessment_x))
                result["linear"][name][concept] = concept_result
        for erased in CONCEPTS:
            other = "block" if erased == "agent" else "agent"
            result["gates"][erased] = {}
            for cohort in ("overall", "challenge", "ordinary"):
                identity_target = result["linear"]["identity"][erased]["assessment"][cohort]
                identity_other = result["linear"]["identity"][other]["assessment"][cohort]
                erased_target = result["linear"][erased][erased]["assessment"][cohort]
                erased_other = result["linear"][erased][other]["assessment"][cohort]
                access_intervals = [
                    identity_target["gain_interval"],
                    identity_other["gain_interval"],
                ]
                removal_interval = erased_target["gain_interval"]
                # Paired retention bootstrap: use the same resampled families for both gains.
                mask = (
                    np.ones(len(assessment_ids), dtype=bool)
                    if cohort == "overall"
                    else np.asarray(
                        [item.split(":", 1)[1].startswith(cohort) for item in assessment_ids]
                    )
                )
                target = assessment_states[mask, CONCEPTS[other]]
                ids_subset = assessment_ids[mask]
                constant = np.sum(
                    train_states[:, CONCEPTS[other]] * family_weights(train_ids)[:, None], axis=0
                )
                bootstrap_strata = (
                    np.asarray([item.split(":", 1)[1].split("-", 1)[0] for item in ids_subset])
                    if cohort == "overall"
                    else None
                )
                boot_identity = bootstrap_gain(
                    predictions["identity"][other][mask],
                    target,
                    ids_subset,
                    constant,
                    bootstrap_seed
                    + 700
                    + (0 if cohort == "overall" else 1 if cohort == "challenge" else 2),
                    strata=bootstrap_strata,
                )
                boot_erased = bootstrap_gain(
                    predictions[erased][other][mask],
                    target,
                    ids_subset,
                    constant,
                    bootstrap_seed
                    + 700
                    + (0 if cohort == "overall" else 1 if cohort == "challenge" else 2),
                    strata=bootstrap_strata,
                )
                retention_values = (
                    np.asarray(boot_erased) - 0.9 * np.asarray(boot_identity)
                ).tolist()
                retention_interval = interval(retention_values)
                support = identity_target["family_count"] >= 10

                def status_for(test: bool | None) -> str:
                    return "unassessable" if test is None else "pass" if test else "fail"

                accessibility = (
                    None
                    if any(value is None for value in access_intervals)
                    else all(value[0] >= 0.5 for value in access_intervals)
                )
                removal = None if removal_interval is None else removal_interval[1] <= 0.1
                retention = None if retention_interval is None else retention_interval[0] >= 0
                numerical = fit_result["concepts"][erased][
                    "idempotence_max_abs"
                ] <= 1e-6 and fit_result["concepts"][erased]["cross_covariance_after_fro"] <= max(
                    1e-8, fit_result["concepts"][erased]["cross_covariance_before_fro"] * 1e-6
                )
                statuses = {
                    "accessibility": status_for(accessibility),
                    "removal": status_for(removal),
                    "retention": status_for(retention),
                    "support": status_for(support),
                    "numerical": status_for(numerical),
                }
                statuses["overall"] = (
                    "pass"
                    if all(value == "pass" for value in statuses.values())
                    else "unassessable"
                    if any(value == "unassessable" for value in statuses.values())
                    else "fail"
                )
                result["gates"][erased][cohort] = {
                    "status": statuses,
                    "identity_target_gain": identity_target["gain"],
                    "identity_other_gain": identity_other["gain"],
                    "erased_target_gain": erased_target["gain"],
                    "erased_other_gain": erased_other["gain"],
                    "retention_difference_interval": retention_interval,
                }
        seeds = audit_result["config"]["nonlinear"]["seeds"]
        result["nonlinear"]["identity"] = _nonlinear(
            "identity", None, datasets, seeds, root, budget
        )
        for concept in CONCEPTS:
            result["nonlinear"][concept] = _nonlinear(
                concept, transformations[concept], datasets, seeds, root, budget
            )
        seal(report_path, result)
        budget.finish("complete")
        return result
    except TimeoutError:
        budget.finish("budget_exhausted")
        incomplete = {
            "status": "budget_exhausted",
            "signature": signature,
            "assessment_opened": True,
        }
        seal(report_path, incomplete)
        return incomplete
    except Exception:
        budget.finish("failed")
        raise


def report(root: Path = ROOT) -> dict:
    audit_result, fit_result, validation = (
        sealed(root / "audit.json"),
        sealed(root / "fit.json"),
        sealed(root / "validation.json"),
    )
    signature = _signature(root)
    budget = Budget(root, signature)
    budget.begin("report")
    try:
        if audit_result["sources"] != _sources():
            raise ValueError("Audited source changed before reporting")
        if validation["status"] != "complete":
            recommendation = "incomplete"
            nonlinear_target_recovery = {}
            control_summary = {}
        else:
            passed = {
                concept: all(
                    validation["gates"][concept][cohort]["status"]["overall"] == "pass"
                    for cohort in ("overall", "challenge", "ordinary")
                )
                for concept in CONCEPTS
            }
            nonlinear_target_recovery = {
                concept: {
                    "minimum_gain_across_seeds": min(
                        item["assessment"][concept]["gain"]
                        for item in validation["nonlinear"][concept]
                    ),
                    "maximum_gain_across_seeds": max(
                        item["assessment"][concept]["gain"]
                        for item in validation["nonlinear"][concept]
                    ),
                }
                for concept in CONCEPTS
            }
            substantial_nonlinear_recovery = any(
                value["minimum_gain_across_seeds"] >= 0.5
                for value in nonlinear_target_recovery.values()
            )
            if substantial_nonlinear_recovery:
                recommendation = "stop_erasure_route"
            elif all(passed.values()):
                recommendation = "plan_predictor_intervention"
            elif any(passed.values()):
                recommendation = "narrow_to_passing_eraser"
            else:
                recommendation = "stop_erasure_route"
            control_summary = {}
            for concept in CONCEPTS:
                control_summary[concept] = {}
                for kind in ("random", "shuffled"):
                    names = sorted(
                        name
                        for name in validation["linear"]
                        if name.startswith(f"{concept}_{kind}_")
                    )
                    gains = [
                        validation["linear"][name][concept]["assessment"]["overall"]["gain"]
                        for name in names
                    ]
                    distortions = [
                        validation["linear"][name]["transformation"]["fit_mean_perturbation"]
                        for name in names
                    ]
                    control_summary[concept][kind] = {
                        "count": len(names),
                        "target_gain_range": [float(min(gains)), float(max(gains))],
                        "mean_fit_perturbation_range": [
                            float(min(distortions)),
                            float(max(distortions)),
                        ],
                    }
        result = {
            "task": "ERASURE-001",
            "status": validation["status"],
            "signature": signature,
            "recommendation": recommendation,
            "assessment_is_fresh_confirmation": False,
            "excluded_confirmation_family_count": len(
                audit_result["config"]["groups"]["excluded_confirmation"]
            ),
            "concept_diagnostics": fit_result["concepts"],
            "gates": validation.get("gates", {}),
            "nonlinear": validation.get("nonlinear", {}),
            "nonlinear_target_recovery": nonlinear_target_recovery,
            "control_summary": control_summary,
            "runtime": budget.data,
            "limitations": [
                "Continuous LEACE removes fitted linear cross-covariance, "
                "not all nonlinear position information.",
                "Assessment families were used in earlier studies and are not fresh confirmation.",
                "No predictor behavior or physical competence was evaluated.",
                "Historical training-simulator identity remains unproven.",
            ],
        }
        seal(root / "report.json", result)
        budget.finish("complete")
        result["runtime"] = budget.data
        seal(root / "report.json", result)
        return result
    except Exception:
        budget.finish("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "fit", "validate", "report"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"audit": audit, "fit": fit, "validate": validate, "report": report}[args.stage](
        args.output
    )
    print(json.dumps({"stage": args.stage, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
