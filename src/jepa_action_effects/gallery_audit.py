"""GALLERY-001: bounded simulator and frozen-encoder feasibility audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from jepa_action_effects.gallery_metrics import family_metrics, group_summary, query_score
from jepa_action_effects.gallery_render import (
    load_rendered,
    render_candidate,
    save_rendered,
)
from jepa_action_effects.interface_audit import (
    CONFIG_SHA256,
    WEIGHTS_SHA256,
    preprocess_pixels,
)
from jepa_action_effects.latent_audit import (
    Budget,
    _selected,
    manifest,
    seal,
    sealed,
)
from jepa_action_effects.latent_cache import LatentCache, array_digest
from jepa_action_effects.model_inference import (
    ActionScaler,
    build_frozen_model,
    encode_real,
    model_fingerprint,
)
from jepa_action_effects.pilot_collection import load_family, sha256_file
from jepa_action_effects.simulator_replay import InstrumentedPushT

ROOT = Path("outputs/gallery001")
PILOT = Path("outputs/pilot001")
FRESH = Path("outputs/readout001/collection")
READOUT_CACHE = Path("outputs/readout001/latent-cache")
CHECKPOINT = Path("checkpoints/lewm-pusht")
STATS = Path("outputs/model001/action-stats.json")
LATENT_CONFIG = Path("outputs/latent001/config.json")
SCALES = (10.24, 20.48)


def _sources() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "renderer": Path(__file__).with_name("gallery_render.py"),
        "metrics": Path(__file__).with_name("gallery_metrics.py"),
        "simulator_replay": Path(__file__).with_name("simulator_replay.py"),
        "model_inference": Path(__file__).with_name("model_inference.py"),
        "preprocessing": Path(__file__).with_name("interface_audit.py"),
        "cache": Path(__file__).with_name("latent_cache.py"),
        "plan": Path("docs/GALLERY-001_PLAN.md"),
        "lock": Path("uv.lock"),
        "checkpoint_weights": CHECKPOINT / "weights.pt",
        "checkpoint_config": CHECKPOINT / "config.json",
        "scaler": STATS,
        "pilot_manifest": PILOT / "collection-manifest.json",
        "fresh_manifest": FRESH / "collection-manifest.json",
        "readout_fit": Path("outputs/readout001/fit-report.json"),
        "latent_config": LATENT_CONFIG,
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _choose(tuning: list[str], verification: list[str], seed: int) -> list[str]:
    if len(verification) != 12 or len(set(verification)) != 12:
        raise ValueError("Expected twelve distinct preselected verification families")
    if set(verification) - set(tuning):
        raise ValueError("Verification family outside tuning partition")
    rng = np.random.default_rng(seed)
    selected = list(verification)
    for cohort, additional, expected in (("challenge", 14, 6), ("ordinary", 4, 6)):
        prior = [key for key in selected if key.split(":", 1)[1].startswith(cohort)]
        if len(prior) != expected:
            raise ValueError("Verification cohort balance changed")
        remaining = sorted(
            key for key in tuning if key.split(":", 1)[1].startswith(cohort) and key not in selected
        )
        selected.extend(rng.choice(remaining, size=additional, replace=False).tolist())
    return sorted(selected)


def _family_path(identifier: str, fresh: dict) -> Path:
    key = identifier.split(":", 1)[1]
    return FRESH / fresh["family_files"][key]["path"]


def audit(root: Path = ROOT) -> dict:
    start = time.monotonic()
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources() or sealed(root / "config.json") != old["config"]:
            raise ValueError("GALLERY source or configuration changed after audit")
        return old
    pilot, fresh = manifest(PILOT), manifest(FRESH)
    if (
        sha256_file(CHECKPOINT / "weights.pt") != WEIGHTS_SHA256
        or sha256_file(CHECKPOINT / "config.json") != CONFIG_SHA256
    ):
        raise ValueError("Checkpoint identity changed")
    tuning, confirmation = _selected(pilot, fresh)
    latent = sealed(LATENT_CONFIG)
    if set(latent["ids"]) != set(tuning) or set(latent["excluded_ids"]) != set(confirmation):
        raise ValueError("LATENT membership or confirmation exclusion changed")
    if not (set(latent["verification_ids"]) <= set(tuning)):
        raise ValueError("Verification families are outside tuning membership")
    fit = sealed(Path("outputs/readout001/fit-report.json"))
    if fit["status"] != "complete" or fit["confirmation_opened"]:
        raise ValueError("READOUT provenance or confirmation status changed")
    names = ("membership", "templates", "queries", "examples", "bootstrap")
    children = np.random.SeedSequence(20260930).spawn(len(names))
    seeds = {
        name: int(child.generate_state(1)[0]) for name, child in zip(names, children, strict=True)
    }
    ids = _choose(tuning, latent["verification_ids"], seeds["membership"])
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) & set(confirmation):
        raise ValueError("Gallery selection is not 30 disjoint tuning families")
    family_hashes = {}
    for identifier in ids:
        key = identifier.split(":", 1)[1]
        if fresh["splits"][key] != "validation":
            raise ValueError("Selected family is not in tuning partition")
        entry = fresh["family_files"][key]
        path = _family_path(identifier, fresh)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Selected family artifact hash mismatch")
        family_hashes[identifier] = entry["sha256"]
    examples_rng = np.random.default_rng(seeds["examples"])
    examples = []
    for cohort in ("challenge", "ordinary"):
        cohort_ids = [key for key in ids if key.split(":", 1)[1].startswith(cohort)]
        examples.extend(sorted(examples_rng.choice(cohort_ids, size=2, replace=False).tolist()))
    config = {
        "schema": "gallery001-v1",
        "master_seed": 20260930,
        "seeds": seeds,
        "ids": ids,
        "verification_ids": latent["verification_ids"],
        "example_ids": examples,
        "excluded_ids": confirmation,
        "family_hashes": family_hashes,
        "scales": list(SCALES),
        "query_radius": 1.28,
        "candidate_count": 6,
        "negative_count": 4,
        "bootstrap": 2000,
        "experiment_limit_seconds": 1740,
        "total_limit_seconds": 1800,
    }
    hashes = _sources()
    config_hash = seal(root / "config.json", config)
    result = {
        "status": "passed",
        "sources": hashes,
        "config": config,
        "config_sha256": config_hash,
        "audit_seconds": time.monotonic() - start,
    }
    seal(root / "audit.json", result)
    return result


def _ledger(root: Path) -> Budget:
    audited = audit(root)
    return Budget(root, audited["config_sha256"] + sha256_file(root / "audit.json"))


def _render_path(root: Path, identifier: str, index: int) -> Path:
    return root / "renders" / f"{identifier.split(':', 1)[1]}__{index}.npz"


def _example(root: Path, identifier: str) -> None:
    path = _render_path(root, identifier, 0)
    meta, images = load_rendered(path)
    ordered = ["true", *meta["queries"], *meta["scales"]["negative_10.24"]["selected"]]
    ordered = ordered[:7]
    canvas = Image.new("RGB", (224 * len(ordered), 250), "white")
    draw = ImageDraw.Draw(canvas)
    for index, key in enumerate(ordered):
        canvas.paste(Image.fromarray(images[key]), (224 * index, 26))
        draw.text((224 * index + 4, 5), key, fill="black")
        for polygon in meta["footprints"][key]:
            outline = [(224 * index + x * 224 / 512, 26 + y * 224 / 512) for x, y in polygon]
            draw.line(outline, fill=(220, 35, 35), width=1)
        cx, cy = meta["agent_center"]
        radius = meta["agent_radius"]
        draw.ellipse(
            (
                224 * index + (cx - radius) * 224 / 512,
                26 + (cy - radius) * 224 / 512,
                224 * index + (cx + radius) * 224 / 512,
                26 + (cy + radius) * 224 / 512,
            ),
            outline=(0, 80, 220),
        )
    path = root / "examples" / f"{identifier.split(':', 1)[1]}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render(root: Path = ROOT) -> dict:
    audit(root)
    config = sealed(root / "config.json")
    ledger = _ledger(root)
    ledger.begin("render")
    status = "complete"
    env = InstrumentedPushT()
    try:
        fresh = manifest(FRESH)
        for identifier in config["ids"]:
            path = _family_path(identifier, fresh)
            if sha256_file(path) != config["family_hashes"][identifier]:
                raise ValueError("Selected family changed since audit")
            family = load_family(path)
            if (
                family["future_frames"].shape != (6, 2, 224, 224, 3)
                or family["candidate_controls"].shape != (6, 10, 2)
                or len(family["metadata"]["candidate_ids"]) != 6
            ):
                raise ValueError("Family frame, control, or candidate alignment changed")
            for index in range(6):
                ledger.check()
                output = _render_path(root, identifier, index)
                if output.exists():
                    meta, images = load_rendered(output)
                    if (
                        meta["identifier"] != identifier
                        or meta["candidate_index"] != index
                        or meta["input_sha256"] != config["family_hashes"][identifier]
                        or meta["audit_sha256"] != sha256_file(root / "audit.json")
                        or not np.array_equal(images["true"], family["future_frames"][index, -1])
                    ):
                        raise ValueError("Rendered candidate provenance mismatch")
                    continue
                meta, images = render_candidate(env, family, index, identifier, config["seeds"])
                meta["input_sha256"] = config["family_hashes"][identifier]
                meta["audit_sha256"] = sha256_file(root / "audit.json")
                save_rendered(output, meta, images)
            if identifier in config["example_ids"]:
                _example(root, identifier)
        paths = [
            _render_path(root, identifier, index)
            for identifier in config["ids"]
            for index in range(6)
        ]
        result = {
            "status": "complete",
            "audit_sha256": sha256_file(root / "audit.json"),
            "files": {str(path.relative_to(root)): sha256_file(path) for path in paths},
            "examples": config["example_ids"],
        }
        seal(root / "render-manifest.json", result)
        return result
    except TimeoutError:
        status = "incomplete_budget"
        return {
            "status": status,
            "completed_candidates": len(list((root / "renders").glob("*.npz"))),
        }
    except Exception:
        status = "failed_integrity"
        raise
    finally:
        env.close()
        ledger.finish(status)


def _encode(model, images: dict, fingerprint: dict, cache: LatentCache, key_id: str):
    names = sorted(images)
    pixels = np.stack([images[name] for name in names])
    exact = {**fingerprint, "representation": "projected", "source": "gallery", "candidate": key_id}
    key = cache.key("real", {"frames": pixels}, exact)
    cache_path = cache.root / "real" / f"{key}.npz"
    if cache_path.exists():
        latent, _ = cache.read("real", key, exact)
    else:
        latent = encode_real(model, preprocess_pixels(pixels).unsqueeze(0)).numpy()
        cache.write("real", key, latent, exact, {"names": names})
    if latent.shape != (1, len(names), 192):
        raise ValueError("Gallery projected CLS shape changed")
    return {name: latent[0, index] for index, name in enumerate(names)}, key, pixels, names


def _verify_encoding(
    model, pixels: np.ndarray, names: list[str], expected: dict
) -> tuple[float, float]:
    again = encode_real(model, preprocess_pixels(pixels).unsqueeze(0)).numpy()[0]
    reversed_values = encode_real(
        model, preprocess_pixels(pixels[::-1].copy()).unsqueeze(0)
    ).numpy()[0, ::-1]
    reference = np.stack([expected[name] for name in names])
    maximum = float(
        max(np.max(np.abs(reference - again)), np.max(np.abs(reference - reversed_values)))
    )
    mse = float(
        max(
            np.max(np.mean((reference - again) ** 2, axis=1)),
            np.max(np.mean((reference - reversed_values) ** 2, axis=1)),
        )
    )
    if maximum > 1e-6:
        raise ValueError("Duplicate/reversed gallery encoding exceeds 1e-6")
    return maximum, mse


def _score_record(meta: dict, embeddings: dict, eta: float) -> dict:
    result = {"scales": {}, "queries": len(meta["queries"])}
    for scale in SCALES:
        label = f"negative_{scale:g}"
        selected = meta["scales"][label]["selected"]
        eligible = len(selected) == 4 and len(meta["queries"]) == 2
        if not eligible:
            result["scales"][label] = {"eligible": False, "negative_count": len(selected)}
            continue
        bank = np.stack([embeddings["true"], *[embeddings[key] for key in selected]])
        queries = [query_score(embeddings[key], bank, eta) for key in meta["queries"]]
        self_distances = np.mean((bank - bank[0]) ** 2, axis=1)
        result["scales"][label] = {
            "eligible": True,
            "negative_count": 4,
            "queries": queries,
            "exact_self_correct": bool(np.min(self_distances[1:]) > eta),
            "template_distances": self_distances.tolist(),
        }
    return result


def encode(root: Path = ROOT) -> dict:
    audit(root)
    config = sealed(root / "config.json")
    rendered = sealed(root / "render-manifest.json")
    if rendered["status"] != "complete" or rendered["audit_sha256"] != sha256_file(
        root / "audit.json"
    ):
        raise ValueError("Render manifest is incomplete or mismatched")
    for name, digest in rendered["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Rendered file changed after manifest sealing")
    ledger = _ledger(root)
    ledger.begin("encode")
    status = "complete"
    try:
        model = build_frozen_model(CHECKPOINT)
        scaler = ActionScaler.from_json(STATS)
        fingerprint = {
            **model_fingerprint(CHECKPOINT, scaler),
            "pilot_manifest": sha256_file(PILOT / "collection-manifest.json"),
            "recovery_manifest": sha256_file(FRESH / "collection-manifest.json"),
            "gallery_audit": sha256_file(root / "audit.json"),
        }
        cache = LatentCache(root / "latent-cache")
        records = {}
        duplicate_max = 0.0
        duplicate_mse = 0.0
        for identifier in config["ids"]:
            for index in range(6):
                ledger.check()
                rendered_path = _render_path(root, identifier, index)
                meta, images = load_rendered(rendered_path)
                key_id = f"{identifier}:{index}"
                record_path = root / "encoded" / f"{identifier.split(':', 1)[1]}__{index}.json"
                if record_path.exists():
                    record = sealed(record_path)
                    if record["render_sha256"] != rendered["files"][
                        str(rendered_path.relative_to(root))
                    ] or record["audit_sha256"] != sha256_file(root / "audit.json"):
                        raise ValueError("Encoded candidate provenance mismatch")
                    records[key_id] = record
                    duplicate_max = max(duplicate_max, record["verification_max"] or 0.0)
                    duplicate_mse = max(duplicate_mse, record["verification_mse"] or 0.0)
                    continue
                embeddings, cache_key, pixels, names = _encode(
                    model, images, fingerprint, cache, key_id
                )
                if identifier in config["verification_ids"]:
                    maximum, mse = _verify_encoding(model, pixels, names, embeddings)
                    duplicate_max = max(duplicate_max, maximum)
                    duplicate_mse = max(duplicate_mse, mse)
                else:
                    maximum, mse = None, None
                record = {
                    "identifier": identifier,
                    "candidate_index": index,
                    "audit_sha256": sha256_file(root / "audit.json"),
                    "render_sha256": rendered["files"][str(rendered_path.relative_to(root))],
                    "cache_key": cache_key,
                    "embeddings_sha256": {
                        key: array_digest(value) for key, value in embeddings.items()
                    },
                    "verification_max": maximum,
                    "verification_mse": mse,
                }
                seal(record_path, record)
                records[key_id] = record
        eta = max(1e-12, 100 * duplicate_mse)
        result = {
            "status": "complete",
            "render_sha256": sha256_file(root / "render-manifest.json"),
            "fingerprint": fingerprint,
            "verification_max": duplicate_max,
            "verification_mse": duplicate_mse,
            "eta": eta,
            "records": {
                key: sha256_file(
                    root / "encoded" / f"{key.split(':', 1)[1].replace(':', '__')}.json"
                )
                for key in records
            },
        }
        if len(records) != 180:
            raise ValueError("Gallery encoding count incomplete")
        seal(root / "encode-manifest.json", result)
        return result
    except TimeoutError:
        status = "incomplete_budget"
        return {
            "status": status,
            "completed_candidates": len(list((root / "encoded").glob("*.json"))),
        }
    except Exception:
        status = "failed_integrity"
        raise
    finally:
        ledger.finish(status)


def _family_rows(root: Path, config: dict, encoded: dict) -> list[dict]:
    cache = LatentCache(root / "latent-cache")
    fresh = manifest(FRESH)
    rows = []
    for identifier in config["ids"]:
        family = {
            "identifier": identifier,
            "cohort": identifier.split(":", 1)[1].split("-", 1)[0],
            "scales": {},
        }
        stored = load_family(_family_path(identifier, fresh))
        family["contacts"] = stored["contacts"].astype(bool).tolist()
        start_block = stored["history_states"][-1, 2:4]
        family["motion"] = (
            np.max(np.linalg.norm(stored["block_trajectories"] - start_block, axis=2), axis=1)
            > 5.12
        ).tolist()
        candidate_rows = {f"negative_{scale:g}": [] for scale in SCALES}
        for index in range(6):
            render_path = _render_path(root, identifier, index)
            meta, images = load_rendered(render_path)
            record_path = root / "encoded" / f"{identifier.split(':', 1)[1]}__{index}.json"
            record = sealed(record_path)
            record_key = f"{identifier}:{index}"
            if record["render_sha256"] != sha256_file(render_path) or encoded["records"][
                record_key
            ] != sha256_file(record_path):
                raise ValueError("Rendered candidate changed since encoding")
            names = sorted(images)
            exact = {
                **encoded["fingerprint"],
                "representation": "projected",
                "source": "gallery",
                "candidate": f"{identifier}:{index}",
            }
            key = cache.key("real", {"frames": np.stack([images[name] for name in names])}, exact)
            if key != record["cache_key"]:
                raise ValueError("Gallery cache key mismatch")
            values, _ = cache.read("real", key, exact)
            embeddings = {name: values[0, j] for j, name in enumerate(names)}
            if any(
                array_digest(embeddings[name]) != record["embeddings_sha256"][name]
                for name in names
            ):
                raise ValueError("Gallery latent digest mismatch")
            scored = _score_record(meta, embeddings, encoded["eta"])
            for label in candidate_rows:
                candidate_rows[label].append(scored["scales"][label])
        for label, candidates in candidate_rows.items():
            family["scales"][label] = {
                "metrics": family_metrics(candidates),
                "eligible_candidates": sum(item["eligible"] for item in candidates),
                "candidates": candidates,
            }
        rows.append(family)
    return rows


def _regime_rows(rows: list[dict], scale: str, seed: int, kind: str, check_budget) -> dict:
    output = {}
    for regime in ("neither", "exactly_one", "both"):
        participants = []
        for family in rows:
            labels = family[kind]
            endpoint_ids = set()
            for i in range(6):
                for j in range(i + 1, 6):
                    if ("neither", "exactly_one", "both")[
                        int(labels[i]) + int(labels[j])
                    ] == regime:
                        endpoint_ids.update((i, j))
            selected = [family["scales"][scale]["candidates"][i] for i in sorted(endpoint_ids)]
            metrics = family_metrics(selected)
            if metrics is not None:
                participants.append({"cohort": family["cohort"], "metrics": metrics})
        summary = group_summary(participants, seed + len(regime) + len(kind), 2000, check_budget)
        summary["status"] = "assessable" if len(participants) >= 10 else "unassessable"
        output[regime] = summary
    return output


def report(root: Path = ROOT) -> dict:
    audit(root)
    config = sealed(root / "config.json")
    ledger = _ledger(root)
    ledger.begin("report")
    status = "complete"
    try:
        rendered = (
            sealed(root / "render-manifest.json")
            if (root / "render-manifest.json").exists()
            else None
        )
        encoded = (
            sealed(root / "encode-manifest.json")
            if (root / "encode-manifest.json").exists()
            else None
        )
        if rendered is None or encoded is None:
            status = "incomplete"
            result = {
                "status": status,
                "rendered_candidates": len(list((root / "renders").glob("*.npz"))),
                "encoded_candidates": len(list((root / "encoded").glob("*.json"))),
                "expected_candidates": 180,
                "audit_sha256": sha256_file(root / "audit.json"),
            }
            seal(root / "report.json", result)
            return result
        if encoded["render_sha256"] != sha256_file(root / "render-manifest.json"):
            raise ValueError("Encoding refers to a different render manifest")
        rows = _family_rows(root, config, encoded)
        scales = {}
        for scale in SCALES:
            label = f"negative_{scale:g}"
            groups = {}
            for cohort in ("challenge", "ordinary"):
                cohort_rows = [row for row in rows if row["cohort"] == cohort]
                group = [
                    {"cohort": cohort, "metrics": row["scales"][label]["metrics"]}
                    for row in cohort_rows
                ]
                summary = group_summary(
                    group,
                    config["seeds"]["bootstrap"] + int(scale),
                    2000,
                    lambda: ledger.check(True),
                )
                summary["eligible_endpoints"] = sum(
                    row["scales"][label]["eligible_candidates"] for row in cohort_rows
                )
                summary["total_endpoints"] = 6 * len(cohort_rows)
                summary["families_four_candidates"] = sum(
                    row["scales"][label]["eligible_candidates"] >= 4 for row in cohort_rows
                )
                summary["coverage_pass"] = (
                    summary["eligible_endpoints"] / summary["total_endpoints"] >= 0.8
                    and summary["families_four_candidates"] >= 10
                )
                summary["retrieval_pass"] = summary.get("accuracy", 0) >= 0.95
                summary["margin_pass"] = summary.get("ratio_p05", float("-inf")) >= 0.5
                groups[cohort] = summary
            scales[label] = {
                "groups": groups,
                "contact_regimes": _regime_rows(
                    rows,
                    label,
                    config["seeds"]["bootstrap"] + int(scale),
                    "contacts",
                    lambda: ledger.check(True),
                ),
                "motion_regimes": _regime_rows(
                    rows,
                    label,
                    config["seeds"]["bootstrap"] + int(scale),
                    "motion",
                    lambda: ledger.check(True),
                ),
                "exact_self_retrieval": {
                    cohort: {
                        "correct": sum(
                            item["exact_self_correct"]
                            for row in rows
                            if row["cohort"] == cohort
                            for item in row["scales"][label]["candidates"]
                            if item["eligible"]
                        ),
                        "eligible": sum(
                            item["eligible"]
                            for row in rows
                            if row["cohort"] == cohort
                            for item in row["scales"][label]["candidates"]
                        ),
                    }
                    for cohort in ("challenge", "ordinary")
                },
                "pass": all(
                    group["coverage_pass"] and group["retrieval_pass"] and group["margin_pass"]
                    for group in groups.values()
                ),
            }
        if all(value["pass"] for value in scales.values()):
            recommendation = "proceed_to_separate_predictor_plan_if_contribution_review_supports_it"
        elif any(value["pass"] for value in scales.values()):
            recommendation = "narrow"
        else:
            recommendation = "stop_gallery_measurement_route"
        result = {
            "status": "complete",
            "audit_sha256": sha256_file(root / "audit.json"),
            "render_sha256": sha256_file(root / "render-manifest.json"),
            "encode_sha256": sha256_file(root / "encode-manifest.json"),
            "families": len(rows),
            "candidates": 6 * len(rows),
            "eta": encoded["eta"],
            "verification_max": encoded["verification_max"],
            "scales": scales,
            "recommendation": recommendation,
            "family_rows": rows,
        }
        seal(root / "report.json", result)
        return result
    except TimeoutError:
        status = "incomplete_budget"
        return {"status": status}
    except Exception:
        status = "failed_integrity"
        raise
    finally:
        ledger.finish(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "render", "encode", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    result = {"audit": audit, "render": render, "encode": encode, "report": report}[
        arguments.stage
    ](arguments.root)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"family_rows", "scales", "files", "config", "sources", "records"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
