"""LATENT-001: bounded projected-CLS candidate correspondence audit."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from jepa_action_effects.interface_audit import (
    CONFIG_SHA256,
    WEIGHTS_SHA256,
    preprocess_pixels,
)
from jepa_action_effects.latent_cache import LatentCache, array_digest
from jepa_action_effects.latent_metrics import (
    REGIMES,
    cost_matrix,
    endpoint_subset,
    pair_rows,
    scores,
    summarize,
)
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    encode_real,
    model_fingerprint,
    predict_future,
)
from jepa_action_effects.pilot_collection import load_family, sha256_file
from jepa_action_effects.readout_recovery import _partition_ids, _tiers

ROOT = Path("outputs/latent001")
PILOT = Path("outputs/pilot001")
READOUT = Path("outputs/readout001")
FRESH = READOUT / "collection"
CHECKPOINT = Path("checkpoints/lewm-pusht")
STATS = Path("outputs/model001/action-stats.json")
EXPERIMENT_LIMIT = 1740.0
TOTAL_LIMIT = 1800.0


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def seal(path: Path, payload: dict) -> str:
    digest = write_json(path, payload)
    path.with_name(path.name + ".sha256").write_text(digest + "\n")
    return digest


def sealed(path: Path) -> dict:
    digest_path = path.with_name(path.name + ".sha256")
    if not digest_path.exists():
        digest_path = path.with_suffix(".sha256")
    digest = digest_path.read_text().strip()
    if sha256_file(path) != digest:
        raise ValueError(f"Sealed artifact mismatch: {path}")
    return json.loads(path.read_text())


def manifest(root: Path) -> dict:
    path = root / "collection-manifest.json"
    if sha256_file(path) != (root / "collection-manifest.sha256").read_text().strip():
        raise ValueError("Upstream collection manifest mismatch")
    result = json.loads(path.read_text())
    if result["status"] != "complete" or result["selected_horizon"] != 2:
        raise ValueError("Expected completed two-step upstream collection")
    return result


def sources() -> dict:
    paths = {
        "metrics": Path(__file__).with_name("latent_metrics.py"),
        "runner": Path(__file__),
        "plan": Path("docs/LATENT-001_PLAN.md"),
        "uv_lock": Path("uv.lock"),
        "weights": CHECKPOINT / "weights.pt",
        "checkpoint_config": CHECKPOINT / "config.json",
        "action_stats": STATS,
        "pilot_manifest": PILOT / "collection-manifest.json",
        "fresh_manifest": FRESH / "collection-manifest.json",
        "readout_fit": READOUT / "fit-report.json",
        "model_inference": Path(__file__).with_name("model_inference.py"),
        "preprocessing": Path(__file__).with_name("interface_audit.py"),
        "cache": Path(__file__).with_name("latent_cache.py"),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


class Budget:
    def __init__(self, root: Path, signature: str):
        self.path = root / "runtime.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["signature"] != signature:
                raise ValueError("Runtime signature mismatch")
            running = self.data.pop("running", None)
            if running:
                self.data["used_seconds"] += max(0, time.time() - running["started"])
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

    def check(self, report: bool = False) -> None:
        elapsed = 0.0 if self.started is None else time.monotonic() - self.started
        if self.data["used_seconds"] + elapsed >= (TOTAL_LIMIT if report else EXPERIMENT_LIMIT):
            raise TimeoutError("LATENT-001 cumulative runtime budget exhausted")

    def begin(self, stage: str) -> None:
        if stage != "report":
            self.check()
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


def _selected(pilot: dict, fresh: dict) -> tuple[list[str], list[str]]:
    parts = _partition_ids(pilot, fresh)
    _tiers(parts)
    ids = parts["tuning"]
    excluded = parts["confirmation"]
    if len(ids) != 60 or len(excluded) != 60 or set(ids) & set(excluded):
        raise ValueError("Tuning or confirmation membership changed")
    if sum(key.startswith("new:challenge") for key in ids) != 40:
        raise ValueError("Challenge allocation changed")
    return ids, excluded


def _selection(ids: list[str], seed: int, per_cohort: int) -> list[str]:
    rng = np.random.default_rng(seed)
    selected = []
    for cohort in ("challenge", "ordinary"):
        eligible = sorted(key for key in ids if key.split(":", 1)[1].startswith(cohort))
        selected.extend(sorted(rng.choice(eligible, size=per_cohort, replace=False).tolist()))
    return selected


def audit(root: Path = ROOT) -> dict:
    started = time.monotonic()
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != sources() or sealed(root / "config.json") != old["config"]:
            raise ValueError("Audited source or configuration changed")
        return old
    pilot, fresh = manifest(PILOT), manifest(FRESH)
    if (
        sha256_file(CHECKPOINT / "weights.pt") != WEIGHTS_SHA256
        or sha256_file(CHECKPOINT / "config.json") != CONFIG_SHA256
    ):
        raise ValueError("Checkpoint identity changed")
    fit = sealed(READOUT / "fit-report.json")
    if fit["status"] != "complete" or fit["confirmation_opened"]:
        raise ValueError("Readout fit provenance is invalid")
    ids, excluded = _selected(pilot, fresh)
    family_hashes = {}
    for identifier in ids:
        key = identifier.split(":", 1)[1]
        if fresh["splits"][key] != "validation":
            raise ValueError("Expected tuning-only family")
        entry = fresh["family_files"][key]
        path = FRESH / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Tuning family hash mismatch: {identifier}")
        family = load_family(path)
        if (
            family["history_frames"].shape != (3, 224, 224, 3)
            or family["future_frames"].shape != (6, 2, 224, 224, 3)
            or family["candidate_controls"].shape != (6, 10, 2)
            or family["future_states"].shape != (6, 2, 5)
            or family["contacts"].shape != (6,)
            or len(family["metadata"]["candidate_ids"]) != 6
            or len(set(family["metadata"]["candidate_ids"])) != 6
            or not all(family["metadata"]["completed"].values())
        ):
            raise ValueError(f"Tuning family shape or candidate identity changed: {identifier}")
        if not np.isfinite(family["candidate_controls"]).all():
            raise ValueError("Nonfinite candidate controls")
        family_hashes[identifier] = entry["sha256"]
    seeds = np.random.SeedSequence(20260929).spawn(3)
    streams = [int(value.generate_state(1)[0]) for value in seeds]
    config = {
        "schema": "latent001-v1",
        "master_seed": 20260929,
        "seeds": {"verification": streams[0], "illustrations": streams[1], "bootstrap": streams[2]},
        "ids": ids,
        "excluded_ids": excluded,
        "verification_ids": _selection(ids, streams[0], 6),
        "illustration_ids": _selection(ids, streams[1], 2),
        "family_hashes": family_hashes,
        "horizon": 2,
        "width": 192,
        "delta": 5.12,
        "bootstrap": 2000,
        "experiment_limit_seconds": EXPERIMENT_LIMIT,
        "total_limit_seconds": TOTAL_LIMIT,
    }
    source_hashes = sources()
    digest = seal(root / "config.json", config)
    result = {
        "status": "passed",
        "sources": source_hashes,
        "config": config,
        "config_sha256": digest,
        "audit_seconds": time.monotonic() - started,
    }
    seal(root / "audit.json", result)
    return result


def _load_real(
    identifier: str, family: dict, cache: LatentCache, fingerprint: dict, out: LatentCache, model
) -> np.ndarray:
    source, key_id = identifier.split(":", 1)
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    exact = {**fingerprint, "representation": "projected", "source": source, "family_id": key_id}
    key = cache.key("real", {"frames": frames}, exact)
    existing = cache.root / "real" / f"{key}.npz"
    if existing.exists():
        real, _ = cache.read("real", key, exact)
    else:
        real = encode_real(model, preprocess_pixels(frames).unsqueeze(0)).numpy()
        out.write("real", key, real, exact, {"identifier": identifier})
    if real.shape != (1, 15, 192):
        raise ValueError("Real projected CLS alignment changed")
    return real[0]


def _predict(
    identifier: str, family: dict, cache: LatentCache, fingerprint: dict, model, scaler
) -> np.ndarray:
    key = cache.key(
        "prediction",
        {
            "history_frames": family["history_frames"],
            "warmup_controls": family["warmup_controls"],
            "candidate_controls": family["candidate_controls"],
        },
        {**fingerprint, "identifier": identifier},
    )
    exact = {**fingerprint, "identifier": identifier}
    if (cache.root / "prediction" / f"{key}.npz").exists():
        result, _ = cache.read("prediction", key, exact)
    else:
        result = predict_future(
            model,
            preprocess_pixels(family["history_frames"]).unsqueeze(0),
            family["warmup_controls"][None],
            family["candidate_controls"][None],
            scaler,
        ).numpy()
        cache.write("prediction", key, result, exact, {"identifier": identifier})
    if result.shape != (1, 6, 2, 192):
        raise ValueError("Prediction shape or horizon changed")
    return result[0]


def _verification(family: dict, prediction: np.ndarray, model, scaler) -> float:
    pixels = preprocess_pixels(family["history_frames"]).unsqueeze(0)
    arguments = (model, pixels, family["warmup_controls"][None])
    again = predict_future(*arguments, family["candidate_controls"][None], scaler).numpy()[0]
    reversed_result = predict_future(
        *arguments, family["candidate_controls"][None, ::-1].copy(), scaler
    ).numpy()[0, ::-1]
    discrepancy = float(
        max(np.max(np.abs(prediction - again)), np.max(np.abs(prediction - reversed_result)))
    )
    if discrepancy > 1e-6:
        raise ValueError("Duplicate/reversed candidate prediction mismatch")
    return discrepancy


def _record(identifier: str, family: dict, real: np.ndarray, prediction: np.ndarray) -> dict:
    start = real[2].astype(np.float64)
    future = real[3:].reshape(6, 2, 192)[:, -1].astype(np.float64)
    predicted = prediction[:, -1].astype(np.float64)
    cost = cost_matrix(predicted, future)
    block = family["future_states"][:, -1, 2:4].astype(np.float64)
    agent = family["future_states"][:, -1, :2].astype(np.float64)
    start_agent = family["history_states"][-1, :2].astype(np.float64)
    start_block = family["history_states"][-1, 2:4].astype(np.float64)
    contacts = family["contacts"].astype(bool)
    trajectory = family["block_trajectories"].astype(np.float64)
    motion = np.max(np.linalg.norm(trajectory - start_block, axis=2), axis=1) > 5.12
    rows = pair_rows(cost, contacts, block, agent, start_agent, motion, predicted, future)
    tolerance = 1e-8 * max(1.0, float(np.mean(future**2)))
    spreads = np.asarray([row["real_spread"] for row in rows])
    return {
        "id": identifier,
        "cohort": "challenge"
        if identifier.split(":", 1)[1].startswith("challenge")
        else "ordinary",
        "candidate_ids": family["metadata"]["candidate_ids"],
        "scores": scores(cost, start, future),
        "cost": cost.tolist(),
        "pairs": rows,
        "endpoint_strata": {
            regime: endpoint_subset(
                cost,
                start,
                future,
                [
                    row
                    for row in rows
                    if (
                        row["regime"] == regime
                        if regime in REGIMES
                        else "motion_" + REGIMES[sum(row["motion"])] == regime
                    )
                ],
            )
            for regime in (*REGIMES, "motion_neither", "motion_exactly_one", "motion_both")
            if any(
                (
                    row["regime"] == regime
                    if regime in REGIMES
                    else "motion_" + REGIMES[sum(row["motion"])] == regime
                )
                for row in rows
            )
        },
        "contact_count": int(contacts.sum()),
        "moving_count": int(motion.sum()),
        "real_spread_mean": float(spreads.mean()),
        "predicted_spread_mean": float(np.mean([row["predicted_spread"] for row in rows])),
        "collapsed_pairs": int((spreads <= tolerance).sum()),
        "identical_frame_pairs": int(
            sum(
                array_digest(family["future_frames"][i, -1])
                == array_digest(family["future_frames"][j, -1])
                for i in range(6)
                for j in range(i + 1, 6)
            )
        ),
        "numerical_tolerance": tolerance,
        "agent_displacement_mean": float(np.mean(np.linalg.norm(agent - start_agent, axis=1))),
    }


def evaluate(root: Path = ROOT) -> dict:
    audited = audit(root)
    config = sealed(root / "config.json")
    signature = audited["config_sha256"] + sha256_file(root / "audit.json")
    budget = Budget(root, signature)
    budget.begin("evaluate")
    status = "complete"
    try:
        fresh = manifest(FRESH)
        scaler = ActionScaler.from_json(STATS)
        model = build_frozen_model(CHECKPOINT)
        fingerprint = {
            **model_fingerprint(CHECKPOINT, scaler),
            "pilot_manifest": audited["sources"]["pilot_manifest"],
            "recovery_manifest": audited["sources"]["fresh_manifest"],
        }
        old_cache, new_cache = (
            LatentCache(READOUT / "latent-cache"),
            LatentCache(root / "latent-cache"),
        )
        verification = {}
        for identifier in config["ids"]:
            budget.check()
            key_id = identifier.split(":", 1)[1]
            path = FRESH / fresh["family_files"][key_id]["path"]
            if sha256_file(path) != config["family_hashes"][identifier]:
                raise ValueError("Family changed after audit")
            record_path = root / "families" / f"{key_id}.json"
            if record_path.exists():
                record = sealed(record_path)
                if (
                    record["id"] != identifier
                    or record["input_sha256"] != config["family_hashes"][identifier]
                ):
                    raise ValueError("Family result provenance mismatch")
                continue
            family = load_family(path)
            real = _load_real(identifier, family, old_cache, fingerprint, new_cache, model)
            predicted = _predict(identifier, family, new_cache, fingerprint, model, scaler)
            if identifier in config["verification_ids"]:
                verification[identifier] = _verification(family, predicted, model, scaler)
            record = _record(identifier, family, real, predicted)
            record["input_sha256"] = config["family_hashes"][identifier]
            record["real_embedding_sha256"] = array_digest(real)
            record["prediction_sha256"] = array_digest(predicted)
            record["verification_max_error"] = verification.get(identifier)
            seal(record_path, record)
        records = [
            sealed(root / "families" / f"{identifier.split(':', 1)[1]}.json")
            for identifier in config["ids"]
        ]
        if len(records) != 60:
            raise ValueError("LATENT-001 family count incomplete")
        aggregate = summarize(records, config["seeds"]["bootstrap"], budget.check)
        result = {
            "status": "complete",
            "audit_sha256": sha256_file(root / "audit.json"),
            "family_count": len(records),
            "verification_count": sum(
                record["verification_max_error"] is not None for record in records
            ),
            "verification_max_error": max(
                record["verification_max_error"]
                for record in records
                if record["verification_max_error"] is not None
            ),
            "aggregate": aggregate,
        }
        if result["verification_count"] != 12:
            raise ValueError("Verification subset incomplete")
        seal(root / "evaluation.json", result)
        return result
    except TimeoutError:
        status = "incomplete_budget"
        return {
            "status": status,
            "completed_families": len(list((root / "families").glob("*.json"))),
        }
    except Exception:
        status = "failed_integrity"
        raise
    finally:
        budget.finish(status)


def report(root: Path = ROOT) -> dict:
    audited = audit(root)
    budget = Budget(root, audited["config_sha256"] + sha256_file(root / "audit.json"))
    budget.begin("report")
    status = "complete"
    try:
        config = sealed(root / "config.json")
        records = {}
        for identifier in config["ids"]:
            path = root / "families" / f"{identifier.split(':', 1)[1]}.json"
            if path.exists():
                records[identifier] = sealed(path)
        evaluation = (
            sealed(root / "evaluation.json") if (root / "evaluation.json").exists() else None
        )
        complete = (
            evaluation is not None and evaluation["status"] == "complete" and len(records) == 60
        )
        report_data = {
            "status": "complete" if complete else "incomplete",
            "families": len(records),
            "evaluation_sha256": sha256_file(root / "evaluation.json") if complete else None,
            "aggregate": evaluation["aggregate"] if complete else None,
            "illustrations": {
                key: records[key]["scores"] for key in config["illustration_ids"] if key in records
            },
            "coverage": {
                "contacts": sum(item["contact_count"] for item in records.values()),
                "moving": sum(item["moving_count"] for item in records.values()),
                "collapsed_pairs": sum(item["collapsed_pairs"] for item in records.values()),
                "identical_frame_pairs": sum(
                    item["identical_frame_pairs"] for item in records.values()
                ),
                "pairs": sum(len(item["pairs"]) for item in records.values()),
                "mean_agent_displacement": float(
                    np.mean([item["agent_displacement_mean"] for item in records.values()])
                )
                if records
                else None,
            },
            "runtime_pre_finish": budget.data,
            "limitations": [
                "Exploratory reuse of READOUT tuning families; no untouched confirmation.",
                "Global CLS correspondence can reflect agent motion; "
                "block physical accuracy unassessable.",
                "Historical training-simulator identity unproven.",
            ],
        }
        seal(root / "report.json", report_data)
        lines = [
            "# LATENT-001 — projected-CLS action correspondence",
            "",
            f"Status: **{report_data['status']}**, {len(records)}/60 families.",
            "",
        ]
        if complete:
            lines += [
                "| Group | Families | A mean [95% interval] | B mean [95% interval] |",
                "| --- | ---: | ---: | ---: |",
            ]
            for name, group in evaluation["aggregate"]["groups"].items():
                metrics = group.get("metrics", {})

                def cell(key, data=metrics):
                    value = data.get(key)
                    return (
                        "—"
                        if value is None
                        else (
                            f"{value['mean']:.5f} "
                            f"[{value['ci95'][0]:.5f}, {value['ci95'][1]:.5f}] "
                            f"({value['state']})"
                        )
                    )

                lines.append(f"| {name} | {group['families']} | {cell('A')} | {cell('B')} |")
            lines += [
                "",
                "Pairwise contact contrasts and raw matrices: report.json and families/*.json.",
                "",
            ]
        lines += ["Preselected examples:", ""]
        for key, item in report_data["illustrations"].items():
            lines.append(f"- {key}: A={item['A']:.5f}, B={item['B']:.5f}.")
        lines += [
            "",
            "Physical block competence remains unassessable because all prior "
            "readout gates failed.",
            "",
        ]
        (root / "REPORT.md").write_text("\n".join(lines))
        return report_data
    except TimeoutError:
        status = "incomplete_budget"
        return {"status": status}
    finally:
        budget.finish(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "evaluate", "report"))
    args = parser.parse_args()
    started = time.monotonic()
    result = {"audit": audit, "evaluate": evaluate, "report": report}[args.stage]()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": result["status"],
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
