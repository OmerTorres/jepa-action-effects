"""Run PILOT-001 collection, frozen inference/readouts, and final reporting."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa_action_effects.evaluation import (
    BranchPrediction,
    BranchTruth,
    EvaluationConfig,
    FamilyEvaluation,
    FamilyTruth,
    Goal,
    aggregate_families,
    evaluate_family,
)
from jepa_action_effects.interface_audit import preprocess_pixels, sha256
from jepa_action_effects.latent_cache import LatentCache, array_digest
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    encode_real,
    model_fingerprint,
    predict_future,
)
from jepa_action_effects.pilot_collection import (
    PilotConfig,
    collect,
    load_family,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.pilot_readouts import (
    NONLINEAR_SEEDS,
    fit_nonlinear,
    readout_gate,
    select_ridge,
)

DEFAULT_OUTPUT = Path("outputs/pilot001")


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _load_all(output_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        family_id: load_family(output_dir / record["path"])
        for family_id, record in manifest["family_files"].items()
    }


def _cache_embeddings(
    output_dir: Path,
    families: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    checkpoint_dir: Path,
    stats_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], dict[str, Any]]:
    scaler = ActionScaler.from_json(stats_path)
    model = build_frozen_model(checkpoint_dir)
    fingerprint = {
        **model_fingerprint(checkpoint_dir, scaler),
        "pilot_config_sha256": manifest["config_sha256"],
        "collection_manifest_sha256": sha256_file(output_dir / "collection-manifest.json"),
        "horizon": manifest["selected_horizon"],
    }
    cache = LatentCache(output_dir / "latent-cache")
    encoded: dict[str, dict[str, np.ndarray]] = {}
    timing = {
        "real_encoding_seconds": 0.0,
        "prediction_seconds": 0.0,
        "real_cache_hits": 0,
        "real_cache_misses": 0,
        "prediction_cache_hits": 0,
        "prediction_cache_misses": 0,
    }
    for family_id in sorted(families):
        family = families[family_id]
        history = family["history_frames"]
        future = family["future_frames"]
        real_frames = np.concatenate((history, future.reshape(-1, *future.shape[2:])), axis=0)
        real_key = cache.key("real", {"frames": real_frames}, fingerprint)
        real_path = cache.root / "real" / f"{real_key}.npz"
        if real_path.exists():
            real, _ = cache.read("real", real_key, fingerprint)
            timing["real_cache_hits"] += 1
        else:
            started = time.perf_counter()
            real = encode_real(model, preprocess_pixels(real_frames).unsqueeze(0)).numpy()
            timing["real_encoding_seconds"] += time.perf_counter() - started
            timing["real_cache_misses"] += 1
            cache.write("real", real_key, real, fingerprint, {"family_id": family_id})
        controls = family["candidate_controls"]
        prediction_key = cache.key(
            "prediction",
            {
                "history_frames": history,
                "past_controls": family["warmup_controls"],
                "candidate_controls": controls,
            },
            fingerprint,
        )
        prediction_path = cache.root / "prediction" / f"{prediction_key}.npz"
        if prediction_path.exists():
            prediction, _ = cache.read("prediction", prediction_key, fingerprint)
            timing["prediction_cache_hits"] += 1
        else:
            started = time.perf_counter()
            prediction = predict_future(
                model,
                preprocess_pixels(history).unsqueeze(0),
                family["warmup_controls"][None],
                controls[None],
                scaler,
            ).numpy()
            timing["prediction_seconds"] += time.perf_counter() - started
            timing["prediction_cache_misses"] += 1
            cache.write(
                "prediction", prediction_key, prediction, fingerprint, {"family_id": family_id}
            )
        horizon = int(manifest["selected_horizon"])
        encoded[family_id] = {
            "history": real[0, :3],
            "future": real[0, 3:].reshape(6, horizon, 192),
            "prediction": prediction[0],
        }
    return encoded, timing, fingerprint


def _training_data(
    families: dict[str, dict[str, Any]],
    embeddings: dict[str, dict[str, np.ndarray]],
    splits: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_embeddings, all_states, all_ids = [], [], []
    for family_id in sorted(families):
        if splits[family_id] != "fit":
            continue
        family = families[family_id]
        frames = np.concatenate(
            (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
        )
        states = np.concatenate((family["history_states"], family["future_states"].reshape(-1, 5)))
        latent = np.concatenate(
            (embeddings[family_id]["history"], embeddings[family_id]["future"].reshape(-1, 192))
        )
        seen: set[str] = set()
        for frame, state, value in zip(frames, states, latent, strict=True):
            digest = array_digest(frame)
            if digest in seen:
                continue
            seen.add(digest)
            all_embeddings.append(value)
            all_states.append(state)
            all_ids.append(family_id)
    return np.asarray(all_embeddings), np.asarray(all_states), np.asarray(all_ids)


def _validation_data(
    families: dict[str, dict[str, Any]],
    embeddings: dict[str, dict[str, np.ndarray]],
    splits: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        family_id: (embeddings[family_id]["future"][:, -1], family["future_states"][:, -1])
        for family_id, family in families.items()
        if splits[family_id] == "validation"
    }


def _linear_gate(model, validation, families, delta):
    errors, contacts = {}, {}
    for family_id, (embeddings, states) in validation.items():
        predicted = model.predict(embeddings)["block"]
        errors[family_id] = np.linalg.norm(predicted - states[:, 2:4], axis=1)
        contacts[family_id] = families[family_id]["contacts"]
    return readout_gate(errors, contacts, delta)


def _truth(family_id, family, manifest) -> FamilyTruth:
    metadata = family["metadata"]
    branches = tuple(
        BranchTruth(
            candidate_id,
            family["future_states"][index, -1, 2:4],
            family["future_states"][index, -1, :2],
            bool(family["contacts"][index]),
            family["block_trajectories"][index],
            "physics_substep",
        )
        for index, candidate_id in enumerate(metadata["candidate_ids"])
    )
    completed = all(metadata["completed"].values())
    return FamilyTruth(
        family_id,
        family["history_states"][-1, 2:4],
        family["history_states"][-1, :2],
        branches,
        int(manifest["selected_horizon"]),
        split=manifest["splits"][family_id],
        group_id=family_id,
        eligible=completed,
        exclusion_reasons=() if completed else ("incomplete_four_step_branch",),
    )


def _predictions(method: str, truth: FamilyTruth, family, embedded, readout):
    if method == "jepa":
        decoded = readout.predict(embedded["prediction"].reshape(-1, 192))
        block = decoded["block"].reshape(6, -1, 2)
        agent = decoded["agent"].reshape(6, -1, 2)
    elif method == "real_frame_readout":
        decoded = readout.predict(embedded["future"].reshape(-1, 192))
        block = decoded["block"].reshape(6, -1, 2)
        agent = decoded["agent"].reshape(6, -1, 2)
    elif method == "oracle":
        block = family["future_states"][:, :, 2:4]
        agent = family["future_states"][:, :, :2]
    elif method == "persistence":
        horizon = family["future_states"].shape[1]
        block = np.tile(truth.current_block_position, (6, horizon, 1))
        agent = np.tile(truth.current_agent_position, (6, horizon, 1))
    else:
        raise ValueError(f"Unknown method: {method}")
    return tuple(
        BranchPrediction(
            candidate_id,
            block[index, -1],
            agent[index, -1],
            block[index],
            "native_model_step",
        )
        for index, candidate_id in enumerate(family["metadata"]["candidate_ids"])
    )


def _bootstrap(evaluations: list[FamilyEvaluation], seed: int, samples: int = 2000):
    eligible = [value for value in evaluations if value.evaluation_eligible]
    metrics = {"endpoint_error": [], "effect_error": [], "regret": []}
    for evaluation in eligible:
        metrics["endpoint_error"].append(
            float(np.mean([value.block_endpoint_error for value in evaluation.branch_metrics]))
        )
        metrics["effect_error"].append(
            float(np.mean([value.effect_error for value in evaluation.pair_metrics]))
        )
        metrics["regret"].append(
            float(np.mean([value.regret for value in evaluation.goal_metrics]))
        )
    rng = np.random.default_rng(seed)
    result = {}
    for name, values_list in metrics.items():
        values = np.asarray(values_list)
        if len(values) == 0:
            result[name] = {"mean": None, "ci95": None, "family_count": 0}
            continue
        draws = np.mean(values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1)
        result[name] = {
            "mean": float(np.mean(values)),
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "family_count": len(values),
        }
    return result


def _family_metric(evaluation: FamilyEvaluation, name: str) -> float:
    if name == "endpoint_error":
        return float(np.mean([value.block_endpoint_error for value in evaluation.branch_metrics]))
    if name == "effect_error":
        return float(np.mean([value.effect_error for value in evaluation.pair_metrics]))
    if name == "regret":
        return float(np.mean([value.regret for value in evaluation.goal_metrics]))
    raise ValueError(name)


def _paired_bootstrap(
    left: list[FamilyEvaluation],
    right: list[FamilyEvaluation],
    seed: int,
    samples: int = 2000,
) -> dict[str, Any]:
    left_by_id = {value.family_id: value for value in left if value.evaluation_eligible}
    right_by_id = {value.family_id: value for value in right if value.evaluation_eligible}
    family_ids = sorted(set(left_by_id) & set(right_by_id))
    rng = np.random.default_rng(seed)
    result = {"family_count": len(family_ids), "sign": "jepa_minus_persistence"}
    for metric in ("endpoint_error", "effect_error", "regret"):
        differences = np.asarray(
            [
                _family_metric(left_by_id[family_id], metric)
                - _family_metric(right_by_id[family_id], metric)
                for family_id in family_ids
            ]
        )
        if len(differences) == 0:
            result[metric] = {"mean": None, "ci95": None}
            continue
        draws = np.mean(
            differences[rng.integers(0, len(differences), size=(samples, len(differences)))],
            axis=1,
        )
        result[metric] = {
            "mean": float(np.mean(differences)),
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        }
    return result


def _coverage(evaluations: list[FamilyEvaluation]) -> dict[str, Any]:
    eligible = [value for value in evaluations if value.evaluation_eligible]
    branches = [metric for value in eligible for metric in value.branch_metrics]
    pairs = [metric for value in eligible for metric in value.pair_metrics]
    goals = [metric for value in eligible for metric in value.goal_metrics]
    return {
        "eligible_family_count": len(eligible),
        "branch_count": len(branches),
        "contact_branch_count": sum(metric.contacted for metric in branches),
        "no_contact_stationary_branch_count": sum(
            metric.true_stationary_no_contact is True for metric in branches
        ),
        "no_contact_moving_branch_count": sum(
            metric.true_stationary_no_contact is False for metric in branches
        ),
        "meaningful_pair_count": sum(metric.effect_label == "meaningful" for metric in pairs),
        "negligible_pair_count": sum(metric.effect_label == "negligible" for metric in pairs),
        "informative_goal_count": sum(metric.informative for metric in goals),
        "goal_count": len(goals),
    }


def evaluate(
    output_dir: Path,
    checkpoint_dir: Path = Path("checkpoints/lewm-pusht"),
    stats_path: Path = Path("outputs/model001/action-stats.json"),
) -> dict[str, Any]:
    started = time.perf_counter()
    previous_path = output_dir / "evaluation-report.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    manifest = verify_collection(output_dir)
    families = _load_all(output_dir, manifest)
    embedded, inference_timing, fingerprint = _cache_embeddings(
        output_dir, families, manifest, checkpoint_dir, stats_path
    )
    train_embeddings, train_states, train_ids = _training_data(
        families, embedded, manifest["splits"]
    )
    validation = _validation_data(families, embedded, manifest["splits"])
    fit_started = time.perf_counter()
    readout, alpha_scores = select_ridge(train_embeddings, train_states, train_ids, validation)
    ridge_gate = _linear_gate(readout, validation, families, manifest["delta"])
    np.savez_compressed(output_dir / "ridge-readouts.npz", **readout.arrays())
    nonlinear_reports = []
    nonlinear_all_passed = False
    if not ridge_gate["passed"]:
        for seed in NONLINEAR_SEEDS:
            model, target_mean, target_scale, training_report = fit_nonlinear(
                train_embeddings,
                train_states,
                train_ids,
                validation,
                readout.feature_mean,
                readout.feature_scale,
                seed,
            )
            errors = {}
            contacts = {}
            with torch.inference_mode():
                for family_id, (values, states) in validation.items():
                    inputs = torch.tensor(
                        (values - readout.feature_mean) / readout.feature_scale,
                        dtype=torch.float32,
                    )
                    block = model(inputs)[:, :2].numpy() * target_scale[:2] + target_mean[:2]
                    errors[family_id] = np.linalg.norm(block - states[:, 2:4], axis=1)
                    contacts[family_id] = families[family_id]["contacts"]
            gate = readout_gate(errors, contacts, manifest["delta"])
            nonlinear_reports.append({**training_report, "gate": gate})
            torch.save(model.state_dict(), output_dir / f"nonlinear-readout-{seed}.pt")
            np.savez_compressed(
                output_dir / f"nonlinear-readout-{seed}-scaling.npz",
                feature_mean=readout.feature_mean,
                feature_scale=readout.feature_scale,
                target_mean=target_mean,
                target_scale=target_scale,
            )
        nonlinear_all_passed = all(report["gate"]["passed"] for report in nonlinear_reports)
    fitting_seconds = time.perf_counter() - fit_started

    config = EvaluationConfig(manifest["delta"], 512.0, 0.0, int(manifest["selected_horizon"]))
    evaluations: dict[str, dict[str, list[FamilyEvaluation]]] = {
        cohort: {method: [] for method in ("jepa", "persistence", "oracle", "real_frame_readout")}
        for cohort in ("challenge", "ordinary")
    }
    raw = []
    for family_id in sorted(families):
        if manifest["splits"][family_id] != "development":
            continue
        family = families[family_id]
        truth = _truth(family_id, family, manifest)
        goals = tuple(
            Goal(f"goal-{index}", position)
            for index, position in enumerate(manifest["goals"][family_id])
        )
        cohort = family["metadata"]["cohort"]
        for method in evaluations[cohort]:
            result = evaluate_family(
                truth,
                _predictions(method, truth, family, embedded[family_id], readout),
                goals,
                config,
                method=method,
                goal_provenance="fixed_3x3_grid_without_replacement_before_model_inference",
            )
            evaluations[cohort][method].append(result)
            raw.append(result.to_dict())
    bootstrap_seed = int(
        np.random.SeedSequence(PilotConfig().master_seed).spawn(6)[5].generate_state(1)[0]
    )
    summaries = {}
    paired_contrasts = {}
    coverage = {}
    for cohort, methods in evaluations.items():
        summaries[cohort] = {}
        for method, values in methods.items():
            summaries[cohort][method] = {
                "aggregate": aggregate_families(tuple(values)),
                "bootstrap_2000": _bootstrap(values, bootstrap_seed),
            }
        paired_contrasts[cohort] = _paired_bootstrap(
            methods["jepa"], methods["persistence"], bootstrap_seed
        )
        coverage[cohort] = _coverage(methods["oracle"])
    model001 = json.loads(Path("outputs/model001/report.json").read_text())
    current_timing = {
        **inference_timing,
        "readout_fitting": fitting_seconds,
        "total_evaluation": time.perf_counter() - started,
    }
    if inference_timing["real_cache_misses"] or inference_timing["prediction_cache_misses"]:
        cold_timing = current_timing
    elif previous and previous.get("cold_timing_seconds"):
        cold_timing = previous["cold_timing_seconds"]
    else:
        cold_timing = None
    report = {
        "schema_version": 1,
        "collection_manifest_sha256": sha256_file(output_dir / "collection-manifest.json"),
        "model_fingerprint": fingerprint,
        "readout": {
            "training_family_count": len(set(train_ids.tolist())),
            "training_sample_count_after_within_family_deduplication": len(train_ids),
            "selected_alpha": readout.alpha,
            "alpha_validation_family_mean_errors": alpha_scores,
            "ridge_gate": ridge_gate,
            "nonlinear_attempted": bool(nonlinear_reports),
            "nonlinear_seeds": nonlinear_reports,
            "nonlinear_all_seeds_passed": nonlinear_all_passed,
        },
        "development_evaluation": summaries,
        "paired_contrasts": paired_contrasts,
        "development_coverage": coverage,
        "raw_family_evaluations": raw,
        "ordinary_transition_latent_check": model001["simulator_smoke"],
        "timing_seconds": current_timing,
        "cold_timing_seconds": cold_timing,
        "artifact_bytes_after_evaluation": _artifact_bytes(output_dir),
    }
    path = output_dir / "evaluation-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "evaluation-report.sha256").write_text(sha256_file(path) + "\n")
    return report


def _coverage_pass(report: dict[str, Any]) -> tuple[bool, dict[str, int]]:
    counts = {}
    for cohort in ("challenge", "ordinary"):
        strata = report["development_evaluation"][cohort]["jepa"]["aggregate"]["contact_strata"]
        for stratum, value in strata.items():
            counts[f"{cohort}:{stratum}"] = int(value["family_count"])
    central = [
        counts["challenge:exactly_one"],
        counts["challenge:both"],
        counts["ordinary:neither"],
    ]
    return all(value >= 10 for value in central), counts


def _write_plots(output_dir: Path, evaluation: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gate = evaluation["readout"]["ridge_gate"]
    regimes = ["overall", "neither", "exactly_one", "both"]
    means = [gate[name]["mean_error"] for name in regimes]
    tails = [gate[name]["p95_error"] for name in regimes]
    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(regimes))
    axis.bar(x - 0.18, means, 0.36, label="mean")
    axis.bar(x + 0.18, tails, 0.36, label="p95")
    axis.axhline(gate["thresholds"]["mean"], color="C0", linestyle="--", label="mean gate")
    axis.axhline(gate["thresholds"]["p95"], color="C1", linestyle=":", label="p95 gate")
    axis.set_xticks(x, regimes)
    axis.set_ylabel("block-position error (workspace units)")
    axis.set_title("Validation real-frame ridge readout")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(plot_dir / "readout-gates.png", dpi=160)
    plt.close(figure)

    methods = ["jepa", "persistence", "real_frame_readout"]
    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    width = 0.36
    for offset, cohort in ((-width / 2, "challenge"), (width / 2, "ordinary")):
        values = [
            evaluation["development_evaluation"][cohort][method]["aggregate"][
                "mean_family_block_endpoint_error"
            ]
            for method in methods
        ]
        axis.bar(x + offset, values, width, label=cohort)
    axis.set_xticks(x, ["JEPA", "persistence", "real frame"])
    axis.set_ylabel("family-mean endpoint error")
    axis.set_title("Exploratory development endpoints")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "development-endpoint-errors.png", dpi=160)
    plt.close(figure)


def physical_competence_status(
    readout_passed: bool, jepa_error: float | None, persistence_error: float | None
) -> str:
    if not readout_passed or jepa_error is None or persistence_error is None:
        return "unassessable"
    return "pass" if jepa_error < persistence_error else "fail"


def report(output_dir: Path) -> dict[str, Any]:
    manifest = verify_collection(output_dir)
    evaluation_path = output_dir / "evaluation-report.json"
    expected = (output_dir / "evaluation-report.sha256").read_text().strip()
    if sha256_file(evaluation_path) != expected:
        raise ValueError("Evaluation report hash mismatch")
    evaluation = json.loads(evaluation_path.read_text())
    _write_plots(output_dir, evaluation)
    replay_passed = bool(manifest["replay_audit"]["all_passed"])
    ridge_readout_passed = bool(evaluation["readout"]["ridge_gate"]["passed"])
    readout_passed = bool(
        ridge_readout_passed or evaluation["readout"]["nonlinear_all_seeds_passed"]
    )
    coverage_passed, coverage_counts = _coverage_pass(evaluation)
    latent = evaluation["ordinary_transition_latent_check"]
    latent_passed = latent["latent_mse"] < latent["persistence_mse"]
    ordinary = evaluation["development_evaluation"]["ordinary"]
    jepa_error = ordinary["jepa"]["aggregate"]["mean_family_block_endpoint_error"]
    persistence_error = ordinary["persistence"]["aggregate"]["mean_family_block_endpoint_error"]
    physical_competence = physical_competence_status(
        ridge_readout_passed, jepa_error, persistence_error
    )
    collection_passed = (
        manifest["status"] == "complete" and not manifest["four_step_incomplete_families"]
    )
    evaluation_timing = evaluation["cold_timing_seconds"] or evaluation["timing_seconds"]
    projected_main_seconds = (
        (manifest["elapsed_seconds"] + evaluation_timing["total_evaluation"]) * 7.5 * 1.5
    )
    runtime_passed = projected_main_seconds <= 8 * 3600
    gates = {
        "collection": collection_passed,
        "replay": replay_passed,
        "readout": readout_passed,
        "central_comparison_coverage": coverage_passed,
        "ordinary_latent_better_than_persistence": latent_passed,
        "ordinary_physical_endpoint_better_than_persistence": physical_competence,
        "projected_main_runtime_with_50pct_contingency_under_8h": runtime_passed,
    }
    if (
        not replay_passed
        or not readout_passed
        or not latent_passed
        or physical_competence != "pass"
    ):
        decision = "stop"
    elif all(gates.values()):
        decision = "proceed"
    else:
        decision = "narrow"
    limitations = []
    if not readout_passed:
        limitations.append("Bounded real-frame readout diagnosis did not pass the prescribed gate.")
    if not coverage_passed:
        limitations.append(
            "At least one central development contact comparison has fewer than 10 families."
        )
    if physical_competence == "unassessable":
        limitations.append("Ordinary physical competence is unassessable with the failed readout.")
    elif physical_competence == "fail":
        limitations.append("Decoded JEPA endpoints did not beat ordinary block persistence.")
    limitations.extend(
        [
            "Pilot findings are exploratory; no locked test families were examined.",
            "The historical training-simulator identity remains unproven.",
            "One checkpoint does not support claims about training-seed variability.",
        ]
    )
    final = {
        "decision": decision,
        "gates": gates,
        "coverage_family_counts": coverage_counts,
        "selected_horizon": manifest["selected_horizon"],
        "delta": manifest["delta"],
        "projected_main_runtime_seconds_with_50pct_contingency": projected_main_seconds,
        "artifact_bytes": _artifact_bytes(output_dir),
        "limitations": limitations,
    }
    (output_dir / "final-report.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    ridge_gate_label = "PASS" if evaluation["readout"]["ridge_gate"]["passed"] else "FAIL"
    projected_hours = projected_main_seconds / 3600
    lines = [
        "# PILOT-001 feasibility report",
        "",
        f"**Decision: {decision.upper()}**",
        "",
        f"Collected 120 families at a {manifest['selected_horizon']}-step horizon with "
        f"delta={manifest['delta']:.3f}. The two-step meaningful-effect rate was "
        f"{manifest['two_step_meaningful_family_fraction']:.1%}.",
        "",
        "## Decision gates",
        "",
        *[
            f"- {label} — {name}"
            for name, passed in gates.items()
            for label in [
                ("PASS" if passed else "FAIL") if isinstance(passed, bool) else passed.upper()
            ]
        ],
        "",
        "## Measured development results",
        "",
        f"- Ridge alpha: {evaluation['readout']['selected_alpha']}",
        f"- Ridge readout gate: {ridge_gate_label}",
        f"- Ordinary JEPA endpoint error: {jepa_error:.4f}",
        f"- Ordinary persistence endpoint error: {persistence_error:.4f}",
        f"- Projected 900-family runtime with contingency: {projected_hours:.2f} hours",
        f"- Pilot artifact storage: {_artifact_bytes(output_dir) / 1024**2:.1f} MiB",
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in limitations],
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    return final


def provenance(output_dir: Path) -> dict[str, Any]:
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv_lock_sha256": sha256(Path("uv.lock")),
        "checkpoint_weights_sha256": sha256(Path("checkpoints/lewm-pusht/weights.pt")),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "evaluate", "report", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args.output_dir)
    if args.stage in ("evaluate", "all"):
        evaluate(args.output_dir)
    if args.stage in ("report", "all"):
        result = report(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
    (args.output_dir / "runtime-provenance.json").write_text(
        json.dumps(provenance(args.output_dir), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
