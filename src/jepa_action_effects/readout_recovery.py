"""Nested development readout study with reserved one-time confirmation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa_action_effects.interface_audit import preprocess_pixels
from jepa_action_effects.latent_cache import LatentCache
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    encode_real,
    model_fingerprint,
)
from jepa_action_effects.pilot import _training_data, _validation_data
from jepa_action_effects.pilot_collection import (
    canonical_json,
    load_family,
    sha256_bytes,
    sha256_file,
    verify_collection,
)
from jepa_action_effects.pilot_readouts import (
    NONLINEAR_SEEDS,
    NonlinearReadout,
    readout_gate,
    select_ridge,
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _partition_ids(pilot: dict, fresh: dict) -> dict[str, list[str]]:
    result = {}
    for name, manifest, split in (
        ("pilot_fit", pilot, "fit"),
        ("new_fit", fresh, "fit"),
        ("tuning", fresh, "validation"),
        ("confirmation", fresh, "development"),
    ):
        ids = sorted(key for key, value in manifest["splits"].items() if value == split)
        result[name] = [f"{'pilot' if name == 'pilot_fit' else 'new'}:{value}" for value in ids]
    return result


def _tiers(parts: dict[str, list[str]]) -> dict[int, list[str]]:
    def by_cohort(ids, cohort):
        return [value for value in ids if value.split(":", 1)[1].startswith(cohort)]

    old = parts["pilot_fit"]
    new = parts["new_fit"]
    all_ids = [key for values in parts.values() for key in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Recovery partitions overlap")
    if len(old) != 60 or len(new) != 180:
        raise ValueError("Recovery fit cohorts have wrong counts")
    if (len(by_cohort(old, "challenge")), len(by_cohort(old, "ordinary"))) != (40, 20):
        raise ValueError("Pilot fit cohort ratio changed")
    for partition, counts in (
        (new, (120, 60)),
        (parts["tuning"], (40, 20)),
        (parts["confirmation"], (40, 20)),
    ):
        if (
            len(by_cohort(partition, "challenge")),
            len(by_cohort(partition, "ordinary")),
        ) != counts:
            raise ValueError("Recovery partition cohort ratio changed")
    return {
        60: old,
        120: old + by_cohort(new, "challenge")[:40] + by_cohort(new, "ordinary")[:20],
        240: old + new,
    }


def _development_ids(parts: dict[str, list[str]]) -> list[str]:
    return sorted(set(parts["pilot_fit"] + parts["new_fit"] + parts["tuning"]))


def _resume_progress(path: Path, signature: str) -> dict:
    if not path.exists():
        return {"results": {}, "selection": None}
    progress = json.loads(path.read_text())
    if progress.get("signature") != signature:
        raise ValueError("Recovery progress input or code signature mismatch")
    return progress


def _load_encoded_family(
    identifier: str,
    pilot_root: Path,
    fresh_root: Path,
    pilot_manifest: dict,
    fresh_manifest: dict,
    cache: LatentCache,
    fingerprint: dict,
    model,
) -> tuple[dict, dict[str, np.ndarray]]:
    source, family_id = identifier.split(":", 1)
    root, manifest = (
        (pilot_root, pilot_manifest) if source == "pilot" else (fresh_root, fresh_manifest)
    )
    family = load_family(root / manifest["family_files"][family_id]["path"])
    frames = np.concatenate(
        (family["history_frames"], family["future_frames"].reshape(-1, 224, 224, 3))
    )
    arrays = {}
    for kind in ("projected", "raw_cls"):
        cache_fp = {**fingerprint, "representation": kind, "source": source, "family_id": family_id}
        key = cache.key("real", {"frames": frames}, cache_fp)
        path = cache.root / "real" / f"{key}.npz"
        if path.exists():
            encoded, _ = cache.read("real", key, cache_fp)
        else:
            pixels = preprocess_pixels(frames)
            with torch.inference_mode():
                if kind == "projected":
                    encoded = encode_real(model, pixels.unsqueeze(0)).numpy()
                else:
                    encoded = (
                        model.encoder(pixels, interpolate_pos_encoding=True)
                        .last_hidden_state[:, 0]
                        .unsqueeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            cache.write("real", key, encoded, cache_fp, {"identifier": identifier})
        arrays[kind] = {
            "history": encoded[0, :3],
            "future": encoded[0, 3:].reshape(6, 2, 192),
        }
    return family, arrays


def _gate(predictor, validation, families, delta: float) -> dict:
    errors = {}
    contacts = {}
    for identifier, (latent, states) in validation.items():
        prediction = predictor(latent)
        errors[identifier] = np.linalg.norm(prediction - states[:, 2:4], axis=1)
        contacts[identifier] = families[identifier]["contacts"]
    return readout_gate(errors, contacts, delta)


def _nonlinear_fit(
    x: np.ndarray,
    states: np.ndarray,
    family_ids: np.ndarray,
    validation: dict,
    families: dict,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    seed: int,
    output: Path,
    deadline: float,
) -> dict:
    torch.manual_seed(seed)
    model = NonlinearReadout()
    labels = np.column_stack(
        (states[:, 2:4], states[:, :2], np.sin(states[:, 4]), np.cos(states[:, 4]))
    )
    ids, counts = np.unique(family_ids, return_counts=True)
    count_map = dict(zip(ids.tolist(), counts.tolist(), strict=True))
    sample_weights = np.asarray([1.0 / count_map[key] for key in family_ids])
    target_mean = np.average(labels, axis=0, weights=sample_weights)
    target_scale = np.sqrt(np.average((labels - target_mean) ** 2, axis=0, weights=sample_weights))
    target_scale[target_scale == 0] = 1.0
    features = torch.tensor((x - feature_mean) / feature_scale, dtype=torch.float32)
    targets = torch.tensor((labels - target_mean) / target_scale, dtype=torch.float32)
    weights = torch.tensor(sample_weights / sample_weights.sum(), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    curves = []
    best = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    timed_out = False
    for epoch in range(1, 3001):
        model.train()
        optimizer.zero_grad()
        loss = (((model(features) - targets) ** 2).mean(dim=1) * weights).sum()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            fit_block = model(features)[:, :2].numpy() * target_scale[:2] + target_mean[:2]
        train_error = float(np.mean(np.linalg.norm(fit_block - states[:, 2:4], axis=1)))

        def predict(latent):
            inp = torch.tensor((latent - feature_mean) / feature_scale, dtype=torch.float32)
            with torch.inference_mode():
                return model(inp)[:, :2].numpy() * target_scale[:2] + target_mean[:2]

        tune_error = float(
            np.mean(
                [
                    np.mean(np.linalg.norm(predict(latent) - labels_[:, 2:4], axis=1))
                    for latent, labels_ in validation.values()
                ]
            )
        )
        if tune_error < best - 1e-4:
            best = tune_error
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            curves.append(
                {"epoch": epoch, "train_mean_error": train_error, "tuning_mean_error": tune_error}
            )
            if time.monotonic() >= deadline:
                timed_out = True
                break
        if stale >= 200:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    _json(output.with_suffix(".curve.json"), {"curve": curves})
    torch.save(model.state_dict(), output.with_suffix(".pt"))
    np.savez_compressed(
        output.with_suffix(".scaling.npz"),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    gate = _gate(predict, validation, families, 5.12)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_executed": epoch,
        "best_tuning_mean_error": best,
        "gate": gate,
        "timed_out": timed_out,
        "curve": str(output.with_suffix(".curve.json")),
    }


def fit_recovery(
    pilot_root: Path = Path("outputs/pilot001"),
    output: Path = Path("outputs/readout001"),
) -> dict:
    started = time.monotonic()
    deadline = started + 7200
    audit = json.loads((output / "audit.json").read_text())
    if (
        sha256_file(output / "audit.json") != (output / "audit.sha256").read_text().strip()
        or audit["status"] != "passed"
    ):
        raise ValueError("Passed audit manifest required")
    summary = json.loads((output / "collection-summary.json").read_text())
    fresh_root = output / "collection"
    pilot_manifest = verify_collection(pilot_root)
    fresh_manifest = verify_collection(fresh_root)
    if summary["collection_manifest_sha256"] != sha256_file(
        fresh_root / "collection-manifest.json"
    ):
        raise ValueError("Recovery collection manifest hash mismatch")
    parts = _partition_ids(pilot_manifest, fresh_manifest)
    tiers = _tiers(parts)
    pilot_seeds = {
        int(load_family(pilot_root / record["path"])["metadata"]["seed"])
        for record in pilot_manifest["family_files"].values()
    }
    fresh_seeds = [
        int(load_family(fresh_root / record["path"])["metadata"]["seed"])
        for record in fresh_manifest["family_files"].values()
    ]
    if len(fresh_seeds) != len(set(fresh_seeds)) or pilot_seeds.intersection(fresh_seeds):
        raise ValueError("Pilot/recovery reset recipe overlap")
    model = build_frozen_model(Path("checkpoints/lewm-pusht"))
    scaler = ActionScaler.from_json(Path("outputs/model001/action-stats.json"))
    fingerprint = {
        **model_fingerprint(Path("checkpoints/lewm-pusht"), scaler),
        "pilot_manifest": audit["pilot_manifest_sha256"],
        "recovery_manifest": summary["collection_manifest_sha256"],
    }
    signature = sha256_bytes(
        canonical_json(
            {
                "fingerprint": fingerprint,
                "recovery_code": sha256_file(Path(__file__)),
                "readout_code": sha256_file(Path(__file__).with_name("pilot_readouts.py")),
                "protocol": "READOUT-001-weighted-v2",
            }
        )
    )
    progress_path = output / "fit-progress.json"
    progress = _resume_progress(progress_path, signature)
    completed_path = output / "fit-report.json"
    completed_hash = output / "fit-report.sha256"
    if completed_path.exists() and completed_hash.exists():
        if sha256_file(completed_path) != completed_hash.read_text().strip():
            raise ValueError("Recovery fit report hash mismatch")
        completed = json.loads(completed_path.read_text())
        if completed.get("signature") == signature and completed["status"] == "complete":
            return completed
    cache = LatentCache(output / "latent-cache")
    families = {}
    embeddings = {"projected": {}, "raw_cls": {}}
    # Confirmation families are deliberately absent from this load.
    for identifier in _development_ids(parts):
        family, views = _load_encoded_family(
            identifier,
            pilot_root,
            fresh_root,
            pilot_manifest,
            fresh_manifest,
            cache,
            fingerprint,
            model,
        )
        families[identifier] = family
        for view in embeddings:
            embeddings[view][identifier] = views[view]
    results = progress["results"]
    selection = progress["selection"]
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for size, fit_ids in tiers.items():
        results.setdefault(str(size), {})
        all_ids = fit_ids + parts["tuning"]
        splits = {key: ("fit" if key in fit_ids else "validation") for key in all_ids}
        subset = {key: families[key] for key in all_ids}
        for view in ("projected", "raw_cls"):
            view_embeddings = {key: embeddings[view][key] for key in all_ids}
            x, states, family_ids = _training_data(subset, view_embeddings, splits)
            validation = _validation_data(subset, view_embeddings, splits)
            ridge, alpha_scores = select_ridge(x, states, family_ids, validation)
            gate = _gate(
                lambda latent, fitted=ridge: fitted.predict(latent)["block"],
                validation,
                subset,
                5.12,
            )
            path = models_dir / f"{size}-{view}-ridge.npz"
            np.savez_compressed(path, **ridge.arrays())
            cell = {
                "training_sample_count": len(x),
                "training_family_count": len(fit_ids),
                "tuning_family_count": len(validation),
                "selected_alpha": ridge.alpha,
                "alpha_scores": alpha_scores,
                "gate": gate,
                "model_path": str(path),
            }
            results[str(size)].setdefault(view, {})["ridge"] = cell
            _json(
                progress_path, {"signature": signature, "results": results, "selection": selection}
            )
            if view == "projected" and gate["passed"] and selection is None:
                selection = {"size": size, "model": "ridge", "paths": [str(path)]}
            if view == "projected" and not gate["passed"]:
                nonlinear = results[str(size)][view].get("nonlinear", [])
                for seed in NONLINEAR_SEEDS:
                    existing = next((item for item in nonlinear if item["seed"] == seed), None)
                    prefix = models_dir / f"{size}-projected-nonlinear-{seed}"
                    saved = all(
                        prefix.with_suffix(suffix).exists()
                        for suffix in (".pt", ".curve.json", ".scaling.npz")
                    )
                    if existing is not None and not existing["timed_out"] and saved:
                        continue
                    if time.monotonic() >= deadline:
                        break
                    fitted = _nonlinear_fit(
                        x,
                        states,
                        family_ids,
                        validation,
                        subset,
                        ridge.feature_mean,
                        ridge.feature_scale,
                        seed,
                        prefix,
                        deadline,
                    )
                    nonlinear = [item for item in nonlinear if item["seed"] != seed]
                    nonlinear.append(fitted)
                    results[str(size)][view]["nonlinear"] = nonlinear
                    _json(
                        progress_path,
                        {"signature": signature, "results": results, "selection": selection},
                    )
                if (
                    len(nonlinear) == 3
                    and all(item["gate"]["passed"] and not item["timed_out"] for item in nonlinear)
                    and selection is None
                ):
                    selection = {
                        "size": size,
                        "model": "nonlinear",
                        "paths": [
                            str(models_dir / f"{size}-projected-nonlinear-{seed}")
                            for seed in NONLINEAR_SEEDS
                        ],
                    }
        if time.monotonic() >= deadline:
            break
    unfinished_cells = []
    for size in (60, 120, 240):
        for view in ("projected", "raw_cls"):
            if "ridge" not in results.get(str(size), {}).get(view, {}):
                unfinished_cells.append(f"{size}-{view}-ridge")
        projected = results.get(str(size), {}).get("projected", {})
        if projected.get("ridge", {}).get("gate", {}).get("passed") is False:
            by_seed = {entry["seed"]: entry for entry in projected.get("nonlinear", [])}
            for seed in NONLINEAR_SEEDS:
                if seed not in by_seed or by_seed[seed]["timed_out"]:
                    unfinished_cells.append(f"{size}-projected-nonlinear-{seed}")
    selection_manifest = {
        "selection": selection,
        "pilot_manifest_sha256": audit["pilot_manifest_sha256"],
        "recovery_manifest_sha256": summary["collection_manifest_sha256"],
        "result_status": (
            "budget_exhausted"
            if unfinished_cells
            else "selected"
            if selection
            else "no_passing_projected_readout"
        ),
    }
    if selection:
        selected_paths = (
            [Path(selection["paths"][0])]
            if selection["model"] == "ridge"
            else [
                Path(f"{prefix}{suffix}")
                for prefix in selection["paths"]
                for suffix in (".pt", ".scaling.npz")
            ]
        )
        selection_manifest["model_artifact_sha256"] = {
            str(path): sha256_file(path) for path in selected_paths
        }
    _json(output / "selection.json", selection_manifest)
    selection_manifest["sha256"] = sha256_file(output / "selection.json")
    if selection and not unfinished_cells and time.monotonic() < deadline:
        confirmation = {}
        # Open confirmation only after selection and its hash have been saved.
        for identifier in parts["confirmation"]:
            family, views = _load_encoded_family(
                identifier,
                pilot_root,
                fresh_root,
                pilot_manifest,
                fresh_manifest,
                cache,
                fingerprint,
                model,
            )
            confirmation[identifier] = (family, views["projected"])
        confirmation_families = {key: value[0] for key, value in confirmation.items()}
        validation = {
            key: (value[1]["future"][:, -1], value[0]["future_states"][:, -1])
            for key, value in confirmation.items()
        }
        gates = []
        if selection["model"] == "ridge":
            from jepa_action_effects.readout_diagnostic import _load_original_ridge

            readout = _load_original_ridge(Path(selection["paths"][0]))
            gates.append(
                _gate(
                    lambda latent: readout.predict(latent)["block"],
                    validation,
                    confirmation_families,
                    5.12,
                )
            )
        else:
            for prefix in selection["paths"]:
                network = NonlinearReadout()
                network.load_state_dict(
                    torch.load(prefix + ".pt", map_location="cpu", weights_only=True)
                )
                network.eval()
                with np.load(prefix + ".scaling.npz") as arrays:
                    mean, scale = arrays["feature_mean"], arrays["feature_scale"]
                    target_mean, target_scale = arrays["target_mean"], arrays["target_scale"]

                def predictor(
                    latent,
                    fitted=network,
                    feature_mean=mean,
                    feature_scale=scale,
                    label_mean=target_mean,
                    label_scale=target_scale,
                ):
                    inp = torch.tensor((latent - feature_mean) / feature_scale, dtype=torch.float32)
                    with torch.inference_mode():
                        return fitted(inp)[:, :2].numpy() * label_scale[:2] + label_mean[:2]

                gates.append(_gate(predictor, validation, confirmation_families, 5.12))
        selection_manifest["confirmation_gates"] = gates
        selection_manifest["confirmation_passed"] = all(gate["passed"] for gate in gates)
    result = {
        "status": "complete" if not unfinished_cells else "budget_exhausted",
        "signature": signature,
        "results": results,
        "selection": selection_manifest,
        "confirmation_opened": "confirmation_gates" in selection_manifest,
        "unfinished_cells": unfinished_cells,
        "elapsed_seconds": time.monotonic() - started,
    }
    _json(output / "fit-report.json", result)
    (output / "fit-report.sha256").write_text(sha256_file(output / "fit-report.json") + "\n")
    return result
