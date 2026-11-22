"""READOUT-001 audit and bounded readout recovery."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from jepa_action_effects.interface_audit import preprocess_pixels
from jepa_action_effects.latent_cache import array_digest
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    manual_predict_future,
    predict_future,
)
from jepa_action_effects.pilot import (
    _cache_embeddings,
    _linear_gate,
    _load_all,
    _training_data,
    _validation_data,
    physical_competence_status,
)
from jepa_action_effects.pilot_collection import (
    Candidate,
    PilotConfig,
    ReplayRecipe,
    _actual_block_vertices,
    _states_from_steps,
    load_family,
    run_family,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.pilot_readouts import (
    LinearReadouts,
    NonlinearReadout,
)
from jepa_action_effects.readout_recovery import fit_recovery

PILOT_ROOT = Path("outputs/pilot001")
OUTPUT_ROOT = Path("outputs/readout001")
CHECKPOINT = Path("checkpoints/lewm-pusht")
STATS = Path("outputs/model001/action-stats.json")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_audit(root: Path) -> dict:
    path = root / "audit.json"
    digest_path = root / "audit.sha256"
    if sha256_file(path) != digest_path.read_text().strip():
        raise ValueError("READOUT-001 audit manifest hash mismatch")
    return json.loads(path.read_text())


def _load_original_ridge(path: Path) -> LinearReadouts:
    with np.load(path, allow_pickle=False) as archive:
        return LinearReadouts(
            *(
                np.asarray(archive[name])
                for name in (
                    "feature_mean",
                    "feature_scale",
                    "alpha",
                    "block_coef",
                    "block_intercept",
                    "agent_coef",
                    "agent_intercept",
                    "orientation_coef",
                    "orientation_intercept",
                )
            )
        )


def _duplicate_conflicts(families: dict, splits: dict) -> dict:
    seen = {}
    duplicates = 0
    conflicts = []
    block_conflicts = 0
    agent_conflicts = 0
    for family_id in sorted(families):
        if splits[family_id] != "fit":
            continue
        family = families[family_id]
        frames = np.concatenate(
            (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
        )
        states = np.concatenate((family["history_states"], family["future_states"].reshape(-1, 5)))
        for index, (frame, state) in enumerate(zip(frames, states, strict=True)):
            key = array_digest(frame)
            if key in seen:
                duplicates += 1
                old_id, old_index, old_state = seen[key]
                block_difference = float(np.max(np.abs(old_state[2:4] - state[2:4])))
                agent_difference = float(np.max(np.abs(old_state[:2] - state[:2])))
                angle_difference = abs(
                    float((old_state[4] - state[4] + np.pi) % (2 * np.pi) - np.pi)
                )
                if block_difference > 1e-6:
                    block_conflicts += 1
                if agent_difference > 1e-6:
                    agent_conflicts += 1
                if max(block_difference, agent_difference, angle_difference) > 1e-6:
                    conflicts.append(
                        {
                            "first": [old_id, old_index],
                            "second": [family_id, index],
                            "block_difference": block_difference,
                            "agent_difference": agent_difference,
                            "circular_angle_difference": angle_difference,
                        }
                    )
            else:
                seen[key] = (family_id, index, state)
    return {
        "unique_frames": len(seen),
        "duplicate_frames": duplicates,
        "block_conflict_count": block_conflicts,
        "agent_conflict_count": agent_conflicts,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:50],
    }


def _alignment_error(direct_history: np.ndarray, direct_future: np.ndarray, stored: dict) -> float:
    history = stored["history_states"]
    future = stored["future_states"][0]
    position_error = max(
        float(np.max(np.abs(direct_history[:, :4] - history[:, :4]))),
        float(np.max(np.abs(direct_future[:, :4] - future[:, :4]))),
    )
    angle_error = max(
        float(np.max(np.abs((direct_history[:, 4] - history[:, 4] + np.pi) % (2 * np.pi) - np.pi))),
        float(np.max(np.abs((direct_future[:, 4] - future[:, 4] + np.pi) % (2 * np.pi) - np.pi))),
    )
    return max(position_error, angle_error)


def _tiny_fit(x: np.ndarray, states: np.ndarray, output: Path) -> dict:
    indices = np.random.default_rng(20260927).choice(len(x), size=32, replace=False)
    selected_x = x[indices].astype(np.float64)
    selected_y = states[indices, 2:4].astype(np.float64)
    design = np.column_stack((selected_x, np.ones(32)))
    coefficients, _, rank, singular = np.linalg.lstsq(design, selected_y, rcond=None)
    residuals = np.linalg.norm(design @ coefficients - selected_y, axis=1)
    mean = selected_x.mean(axis=0)
    scale = selected_x.std(axis=0)
    scale[scale == 0] = 1
    target_mean = selected_y.mean(axis=0)
    target_scale = selected_y.std(axis=0)
    target_scale[target_scale == 0] = 1
    torch.manual_seed(20260927)
    network = NonlinearReadout()
    inputs = torch.tensor((selected_x - mean) / scale, dtype=torch.float32)
    labels = torch.tensor((selected_y - target_mean) / target_scale, dtype=torch.float32)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    curve = []
    for epoch in range(1001):
        network.train()
        optimizer.zero_grad()
        prediction = network(inputs)[:, :2]
        loss = torch.mean((prediction - labels) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            network.eval()
            with torch.inference_mode():
                decoded = network(inputs)[:, :2].numpy() * target_scale + target_mean
            curve.append(
                {
                    "epoch": epoch,
                    "mean_error": float(np.mean(np.linalg.norm(decoded - selected_y, axis=1))),
                }
            )
    _write_json(
        output / "tiny-curve.json", {"curve": curve, "selected_fit_indices": indices.tolist()}
    )
    return {
        "sample_count": 32,
        "least_squares_rank": int(rank),
        "condition_number_nonzero_singular_values": float(singular[0] / singular[-1]),
        "least_squares_mean_error": float(np.mean(residuals)),
        "least_squares_max_error": float(np.max(residuals)),
        "nonlinear_final_mean_error": curve[-1]["mean_error"],
        "nonlinear_curve": "tiny-curve.json",
    }


def audit(pilot_root: Path = PILOT_ROOT, output: Path = OUTPUT_ROOT) -> dict:
    started = time.perf_counter()
    manifest = verify_collection(pilot_root)
    old_evaluation = json.loads((pilot_root / "evaluation-report.json").read_text())
    if (
        sha256_file(pilot_root / "evaluation-report.json")
        != (pilot_root / "evaluation-report.sha256").read_text().strip()
    ):
        raise ValueError("PILOT-001 evaluation report hash mismatch")
    families = _load_all(pilot_root, manifest)
    selected = [
        family_id
        for cohort in ("challenge", "ordinary")
        for split in ("fit", "validation", "development")
        for family_id in sorted(
            key for key in families if key.startswith(cohort) and manifest["splits"][key] == split
        )[:2]
    ]
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    frame_mismatches = []
    state_mismatches = []
    for family_id in selected:
        stored = families[family_id]
        metadata = stored["metadata"]
        recipe = ReplayRecipe(
            family_id, int(metadata["seed"]), stored["warmup_controls"], metadata["reset_options"]
        )
        candidates = [
            Candidate(key, controls)
            for key, controls in zip(
                metadata["candidate_ids"], stored["candidate_controls"], strict=True
            )
        ]
        replay = run_family(recipe, candidates, metadata["execution_order"])
        if list(replay.execution_order) != metadata["execution_order"]:
            raise AssertionError("Candidate execution order mismatch")
        for index, candidate_id in enumerate(metadata["candidate_ids"]):
            branch = replay.branches[candidate_id]
            frame_mismatches.append(
                int(not np.array_equal(branch.history_frames, stored["history_frames"]))
            )
            frame_mismatches.append(
                int(not np.array_equal(branch.future_frames, stored["future_frames"][index]))
            )
            state_mismatches.append(
                float(
                    np.max(
                        np.abs(
                            branch.observations[4::5, :4] - stored["future_states"][index, :, :4]
                        )
                    )
                )
            )
            state_mismatches.append(
                float(
                    np.max(
                        np.abs(
                            (
                                branch.observations[4::5, 4]
                                - stored["future_states"][index, :, 4]
                                + np.pi
                            )
                            % (2 * np.pi)
                            - np.pi
                        )
                    )
                )
            )
        image = Image.fromarray(stored["future_frames"][0, -1].copy())
        draw = ImageDraw.Draw(image)
        state = stored["future_states"][0, -1]
        for position, color in ((state[:2], "yellow"), (state[2:4], "red")):
            x, y = np.asarray(position) * 224 / 512
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        image.save(overlays / f"{family_id}.png")
    embedded, _, fingerprint = _cache_embeddings(pilot_root, families, manifest, CHECKPOINT, STATS)
    train_x, train_states, train_ids = _training_data(families, embedded, manifest["splits"])
    validation = _validation_data(families, embedded, manifest["splits"])
    ridge = _load_original_ridge(pilot_root / "ridge-readouts.npz")
    gate = _linear_gate(ridge, validation, families, manifest["delta"])
    previous = old_evaluation["readout"]["ridge_gate"]
    reproduced = all(
        np.isclose(gate[regime][metric], previous[regime][metric], atol=1e-9)
        for regime in ("overall", "neither", "exactly_one", "both")
        for metric in ("mean_error", "p95_error")
    )
    train_errors = {}
    for family_id in sorted(families):
        if manifest["splits"][family_id] == "fit":
            latent = np.concatenate(
                (embedded[family_id]["history"], embedded[family_id]["future"].reshape(-1, 192))
            )
            states = np.concatenate(
                (
                    families[family_id]["history_states"],
                    families[family_id]["future_states"].reshape(-1, 5),
                )
            )
            train_errors[family_id] = float(
                np.mean(np.linalg.norm(ridge.predict(latent)["block"] - states[:, 2:4], axis=1))
            )
    model = build_frozen_model(CHECKPOINT)
    pixels = preprocess_pixels(families[selected[0]]["history_frames"]).unsqueeze(0)
    with torch.inference_mode():
        raw_cls = model.encoder(
            pixels.reshape(3, 3, 224, 224), interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        manual_projected = model.projector(raw_cls).reshape(1, 3, 192)
        encoded_projected = model.encode({"pixels": pixels.clone()})["emb"]
    projection_error = float(torch.max(torch.abs(manual_projected - encoded_projected)))
    scaler = ActionScaler.from_json(STATS)
    sample = families[selected[0]]
    predicted = predict_future(
        model, pixels, sample["warmup_controls"][None], sample["candidate_controls"][None], scaler
    )
    independent = manual_predict_future(
        model, pixels, sample["warmup_controls"][None], sample["candidate_controls"][None], scaler
    )
    rollout_error = float(torch.max(torch.abs(predicted - independent)))
    duplicates = _duplicate_conflicts(families, manifest["splits"])
    tiny = _tiny_fit(train_x, train_states, output)
    result = {
        "status": "passed"
        if not any(frame_mismatches)
        and max(state_mismatches) <= 1e-10
        and reproduced
        and projection_error <= 1e-6
        and rollout_error <= 1e-6
        and not duplicates["block_conflict_count"]
        and tiny["least_squares_max_error"] <= 1e-6
        else "blocked",
        "pilot_manifest_sha256": sha256_file(pilot_root / "collection-manifest.json"),
        "pilot_evaluation_sha256": sha256_file(pilot_root / "evaluation-report.json"),
        "pilot_ridge_sha256": sha256_file(pilot_root / "ridge-readouts.npz"),
        "fingerprint": fingerprint,
        "replayed_family_ids": selected,
        "frame_mismatch_count": sum(frame_mismatches),
        "maximum_state_alignment_error": max(state_mismatches),
        "projected_cls_agreement_max_error": projection_error,
        "manual_rollout_agreement_max_error": rollout_error,
        "ridge_validation_reproduced": reproduced,
        "ridge_train_family_mean_error": float(np.mean(list(train_errors.values()))),
        "ridge_validation_gate": gate,
        "duplicates": duplicates,
        "tiny_fit": tiny,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "audit.json", result)
    (output / "audit.sha256").write_text(sha256_file(output / "audit.json") + "\n")
    return result


def collect_recovery(pilot_root: Path = PILOT_ROOT, output: Path = OUTPUT_ROOT) -> dict:
    audited = _require_audit(output)
    if audited["status"] != "passed":
        raise ValueError("Measurement audit must pass before collection")
    pilot_manifest = verify_collection(pilot_root)
    pilot_seeds = tuple(
        int(load_family(pilot_root / record["path"])["metadata"]["seed"])
        for record in pilot_manifest["family_files"].values()
    )
    config = PilotConfig(
        master_seed=20260927,
        challenge_target=200,
        ordinary_target=100,
        challenge_attempt_cap=2000,
        ordinary_attempt_cap=1000,
        challenge_split_counts=(120, 40, 40),
        ordinary_split_counts=(60, 20, 20),
        horizon_policy="fixed_two",
        forbidden_reset_seeds=pilot_seeds,
    )
    from jepa_action_effects.pilot_collection import collect as collect_families

    manifest = collect_families(output / "collection", config)
    if manifest["status"] != "complete":
        return {"status": "quota_shortfall", "attempts": manifest["attempts"]}
    if manifest["selected_horizon"] != 2:
        raise AssertionError("Recovery horizon changed")
    result = {
        "status": "complete",
        "pilot_manifest_sha256": audited["pilot_manifest_sha256"],
        "audit_sha256": sha256_file(output / "audit.json"),
        "collection_manifest_sha256": sha256_file(
            output / "collection" / "collection-manifest.json"
        ),
        "attempts": manifest["attempts"],
        "accepted": {key: len(value) for key, value in manifest["accepted"].items()},
        "splits": {
            split: sum(value == split for value in manifest["splits"].values())
            for split in ("fit", "validation", "development")
        },
        "overlap_seed_count": len(
            set(pilot_seeds)
            & {
                int(load_family(output / "collection" / record["path"])["metadata"]["seed"])
                for record in manifest["family_files"].values()
            }
        ),
    }
    _write_json(output / "collection-summary.json", result)
    return result


def report_recovery(output: Path = OUTPUT_ROOT) -> dict:
    audit_result = _require_audit(output)
    collected = json.loads((output / "collection-summary.json").read_text())
    verify_collection(output / "collection")
    if collected["collection_manifest_sha256"] != sha256_file(
        output / "collection" / "collection-manifest.json"
    ):
        raise ValueError("Recovery collection manifest hash mismatch")
    fit_path = output / "fit-report.json"
    if sha256_file(fit_path) != (output / "fit-report.sha256").read_text().strip():
        raise ValueError("READOUT-001 fit report hash mismatch")
    fit = json.loads(fit_path.read_text())
    if collected["audit_sha256"] != sha256_file(output / "audit.json"):
        raise ValueError("Recovery collection does not match audited pilot")
    pilot_manifest = verify_collection(PILOT_ROOT)
    pilot_families = _load_all(PILOT_ROOT, pilot_manifest)
    pilot_latents, _, _ = _cache_embeddings(
        PILOT_ROOT, pilot_families, pilot_manifest, CHECKPOINT, STATS
    )
    original_ridge = _load_original_ridge(PILOT_ROOT / "ridge-readouts.npz")
    original_breakdown = {}
    for split in ("fit", "validation"):
        endpoint_data = {
            key: (pilot_latents[key]["future"][:, -1], family["future_states"][:, -1])
            for key, family in pilot_families.items()
            if pilot_manifest["splits"][key] == split
        }
        original_breakdown[split] = {}
        for cohort in ("all", "challenge", "ordinary"):
            selected_data = {
                key: value
                for key, value in endpoint_data.items()
                if cohort == "all" or key.startswith(cohort)
            }
            original_breakdown[split][cohort] = _linear_gate(
                original_ridge, selected_data, pilot_families, pilot_manifest["delta"]
            )
    _write_json(output / "original-ridge-breakdown.json", original_breakdown)
    geometry_checks = []
    geometry_root = output / "overlays-geometry"
    geometry_root.mkdir(exist_ok=True)
    for family_id in audit_result["replayed_family_ids"]:
        stored = pilot_families[family_id]
        metadata = stored["metadata"]
        recipe = ReplayRecipe(
            family_id, int(metadata["seed"]), stored["warmup_controls"], metadata["reset_options"]
        )
        candidates = [
            Candidate(key, controls)
            for key, controls in zip(
                metadata["candidate_ids"], stored["candidate_controls"], strict=True
            )
        ]
        replay = run_family(recipe, candidates, metadata["execution_order"])
        branch = replay.branches[metadata["candidate_ids"][0]]
        direct_history, direct_future, _ = _states_from_steps(branch)
        alignment_error = _alignment_error(direct_history, direct_future, stored)
        if alignment_error > 1e-10:
            raise ValueError(f"Frame/label timestamp alignment mismatch: {family_id}")
        state = stored["future_states"][0, -1]
        image = Image.fromarray(stored["future_frames"][0, -1].copy())
        draw = ImageDraw.Draw(image)
        vertices = _actual_block_vertices(state[2:4], float(state[4])) * (224 / 512)
        for polygon in (vertices[:4], vertices[4:]):
            draw.polygon([tuple(point) for point in polygon], outline="red", width=2)
        agent = state[:2] * (224 / 512)
        draw.ellipse(
            (agent[0] - 5, agent[1] - 5, agent[0] + 5, agent[1] + 5), outline="yellow", width=2
        )
        image.save(geometry_root / f"{family_id}.png")
        geometry_checks.append(
            {
                "family_id": family_id,
                "maximum_state_alignment_error": alignment_error,
            }
        )
    _write_json(output / "geometry-alignment.json", {"checks": geometry_checks})
    old_evaluation = json.loads((PILOT_ROOT / "evaluation-report.json").read_text())
    ordinary = old_evaluation["development_evaluation"]["ordinary"]
    pilot_correction = {
        "version": "readout001-correction-1",
        "pilot_final_report_sha256": sha256_file(PILOT_ROOT / "final-report.json"),
        "pilot_evaluation_sha256": sha256_file(PILOT_ROOT / "evaluation-report.json"),
        "ridge_readout": "pass" if old_evaluation["readout"]["ridge_gate"]["passed"] else "fail",
        "nonlinear_diagnostic": (
            "pass" if old_evaluation["readout"]["nonlinear_all_seeds_passed"] else "fail"
        ),
        "ordinary_physical_competence": physical_competence_status(
            bool(old_evaluation["readout"]["ridge_gate"]["passed"]),
            ordinary["jepa"]["aggregate"]["mean_family_block_endpoint_error"],
            ordinary["persistence"]["aggregate"]["mean_family_block_endpoint_error"],
        ),
        "explanation": (
            "Pilot endpoint results were decoded by ridge; a nonlinear diagnostic "
            "cannot validate those ridge-decoded endpoints."
        ),
    }
    _write_json(output / "pilot-report-correction.json", pilot_correction)
    cells = []
    for size, views in fit["results"].items():
        for view, methods in views.items():
            ridge = methods["ridge"]
            cells.append(
                {
                    "fit_families": int(size),
                    "representation": view,
                    "model": "ridge",
                    "validation_mean": ridge["gate"]["overall"]["mean_error"],
                    "validation_p95": ridge["gate"]["overall"]["p95_error"],
                    "passed": ridge["gate"]["passed"],
                }
            )
            for nonlinear in methods.get("nonlinear", []):
                cells.append(
                    {
                        "fit_families": int(size),
                        "representation": view,
                        "model": f"nonlinear-{nonlinear['seed']}",
                        "validation_mean": nonlinear["gate"]["overall"]["mean_error"],
                        "validation_p95": nonlinear["gate"]["overall"]["p95_error"],
                        "passed": nonlinear["gate"]["passed"],
                        "best_epoch": nonlinear["best_epoch"],
                    }
                )
    confirmation_opened = fit["confirmation_opened"]
    selection = fit["selection"]["selection"]
    if fit["status"] != "complete":
        recommendation = "finish_bounded_fit"
    elif selection and fit["selection"].get("confirmation_passed"):
        recommendation = "exploratory_physical_rerun"
    else:
        recommendation = "narrow_or_revise_measurement"
    result = {
        "task": "READOUT-001",
        "status": fit["status"],
        "recommendation": recommendation,
        "audit_status": audit_result["status"],
        "original_ridge_validation_reproduced": audit_result["ridge_validation_reproduced"],
        "original_ridge_train_family_mean": audit_result["ridge_train_family_mean_error"],
        "original_ridge_validation_mean": audit_result["ridge_validation_gate"]["overall"][
            "mean_error"
        ],
        "original_ridge_breakdown_sha256": sha256_file(output / "original-ridge-breakdown.json"),
        "geometry_alignment_sha256": sha256_file(output / "geometry-alignment.json"),
        "pilot_report_correction_sha256": sha256_file(output / "pilot-report-correction.json"),
        "identical_frame_agent_conflicts": audit_result["duplicates"]["agent_conflict_count"],
        "identical_frame_block_conflicts": audit_result["duplicates"]["block_conflict_count"],
        "new_families": sum(collected["accepted"].values()),
        "confirmation_opened": confirmation_opened,
        "selection": selection,
        "cells": cells,
        "elapsed_fit_seconds": fit["elapsed_seconds"],
        "unfinished_cells": fit.get("unfinished_cells", []),
        "limitations": [
            "A failed probe does not prove that block state is absent from the representation.",
            "The historical training-simulator identity remains unproven.",
            "No main-study locked test labels were inspected.",
        ],
    }
    _write_json(output / "final-report.json", result)
    lines = [
        "# READOUT-001 diagnostic",
        "",
        f"Recommendation: **{recommendation.replace('_', ' ')}**.",
        "",
        "The original ridge validation mean error "
        f"({result['original_ridge_validation_mean']:.2f}) "
        "was reproduced; its fit-family mean error was "
        f"{result['original_ridge_train_family_mean']:.2f}.",
        f"The audit found {result['identical_frame_agent_conflicts']} ambiguous agent labels "
        f"and {result['identical_frame_block_conflicts']} conflicting block labels "
        "among identical frames.",
        "",
        "| Fit families | Representation | Readout | Tuning mean | Tuning p95 | Gate |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {cell['fit_families']} | {cell['representation']} | {cell['model']} | "
        f"{cell['validation_mean']:.2f} | {cell['validation_p95']:.2f} | "
        f"{'pass' if cell['passed'] else 'fail'} |"
        for cell in cells
    )
    lines.extend(
        ["", f"Confirmation labels opened: **{'yes' if confirmation_opened else 'no'}**.", ""]
    )
    lines.append(
        "Original ridge fit/validation errors by cohort and contact regime: "
        "`original-ridge-breakdown.json`."
    )
    if confirmation_opened:
        lines.append(f"Confirmation passed: **{fit['selection']['confirmation_passed']}**.")
    else:
        lines.append("No projected readout passed tuning; confirmation labels remain reserved.")
    lines.extend(["", *[f"- {item}" for item in result["limitations"]], ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "collect", "fit", "report"))
    args = parser.parse_args()
    if args.stage == "audit":
        result = audit()
    elif args.stage == "collect":
        result = collect_recovery()
    elif args.stage == "fit":
        result = fit_recovery()
    elif args.stage == "report":
        result = report_recovery()
    else:
        result = {"status": "pending", "stage": args.stage}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
