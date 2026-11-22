"""PREDICTOR-GALLERY-001: sealed, staged frozen-predictor gallery audit."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from jepa_action_effects.gallery_audit import (
    CHECKPOINT,
    FRESH,
    PILOT,
    STATS,
    _encode,
    _example,
    _family_path,
    _score_record,
    _verify_encoding,
)
from jepa_action_effects.gallery_metrics import family_metrics, group_summary, query_score
from jepa_action_effects.gallery_render import load_rendered, render_candidate, save_rendered
from jepa_action_effects.interface_audit import CONFIG_SHA256, WEIGHTS_SHA256
from jepa_action_effects.latent_audit import (
    Budget,
    _load_real,
    _predict,
    _selected,
    _selection,
    _verification,
    manifest,
    seal,
    sealed,
)
from jepa_action_effects.latent_cache import LatentCache, array_digest
from jepa_action_effects.latent_metrics import cost_matrix, scores
from jepa_action_effects.model_inference import ActionScaler, build_frozen_model, model_fingerprint
from jepa_action_effects.pilot_collection import load_family, sha256_file
from jepa_action_effects.predictor_gallery_metrics import (
    REGIMES,
    aggregate,
    candidate_scores,
    family_mean,
    latent_summary,
    participants,
)
from jepa_action_effects.simulator_replay import InstrumentedPushT

ROOT = Path("outputs/predictor_gallery001")
GALLERY = Path("outputs/gallery001")
READOUT_CACHE = Path("outputs/readout001/latent-cache")
SCALES = (10.24, 20.48)


def _sources() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "metrics": Path(__file__).with_name("predictor_gallery_metrics.py"),
        "gallery_runner": Path(__file__).with_name("gallery_audit.py"),
        "gallery_renderer": Path(__file__).with_name("gallery_render.py"),
        "gallery_metrics": Path(__file__).with_name("gallery_metrics.py"),
        "latent_runner": Path(__file__).with_name("latent_audit.py"),
        "latent_metrics": Path(__file__).with_name("latent_metrics.py"),
        "simulator": Path(__file__).with_name("simulator_replay.py"),
        "model": Path(__file__).with_name("model_inference.py"),
        "preprocessing": Path(__file__).with_name("interface_audit.py"),
        "cache": Path(__file__).with_name("latent_cache.py"),
        "lock": Path("uv.lock"),
        "weights": CHECKPOINT / "weights.pt",
        "checkpoint_config": CHECKPOINT / "config.json",
        "scaler": STATS,
        "pilot_manifest": PILOT / "collection-manifest.json",
        "readout_manifest": FRESH / "collection-manifest.json",
        "readout_fit": Path("outputs/readout001/fit-report.json"),
        "gallery_config": GALLERY / "config.json",
        "gallery_audit": GALLERY / "audit.json",
        "gallery_render": GALLERY / "render-manifest.json",
        "gallery_encode": GALLERY / "encode-manifest.json",
        "gallery_report": GALLERY / "report.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _cohort(identifier: str) -> str:
    return identifier.split(":", 1)[1].split("-", 1)[0]


def _validate_ids(ids: list[str], challenge: int, ordinary: int) -> None:
    if len(ids) != challenge + ordinary or len(set(ids)) != len(ids):
        raise ValueError("Family count or uniqueness changed")
    if (
        sum(_cohort(key) == "challenge" for key in ids) != challenge
        or sum(_cohort(key) == "ordinary" for key in ids) != ordinary
    ):
        raise ValueError("Cohort allocation changed")


def _record_path(root: Path, phase: str, identifier: str) -> Path:
    return root / phase / "families" / f"{identifier.split(':', 1)[1]}.json"


def _render_path(root: Path, identifier: str, index: int) -> Path:
    return root / "confirm" / "renders" / f"{identifier.split(':', 1)[1]}__{index}.npz"


def _encoded_path(root: Path, identifier: str, index: int) -> Path:
    return root / "confirm" / "encoded" / f"{identifier.split(':', 1)[1]}__{index}.json"


def audit(root: Path = ROOT) -> dict:
    start = time.monotonic()
    if (root / "audit.json").exists():
        old = sealed(root / "audit.json")
        if old["sources"] != _sources() or sealed(root / "config.json") != old["config"]:
            raise ValueError("PREDICTOR-GALLERY audit source or configuration mismatch")
        return old
    pilot, fresh = manifest(PILOT), manifest(FRESH)
    if (
        sha256_file(CHECKPOINT / "weights.pt") != WEIGHTS_SHA256
        or sha256_file(CHECKPOINT / "config.json") != CONFIG_SHA256
    ):
        raise ValueError("Checkpoint identity changed")
    tuning, confirmation = _selected(pilot, fresh)
    gallery_config = sealed(GALLERY / "config.json")
    gallery_audit = sealed(GALLERY / "audit.json")
    gallery_render = sealed(GALLERY / "render-manifest.json")
    gallery_encode = sealed(GALLERY / "encode-manifest.json")
    gallery_report = sealed(GALLERY / "report.json")
    ids = gallery_config["ids"]
    _validate_ids(ids, 20, 10)
    _validate_ids(confirmation, 40, 20)
    if set(ids) - set(tuning) or set(ids) & set(confirmation):
        raise ValueError("Development/confirmation partitions overlap or changed")
    if (
        gallery_audit["config"] != gallery_config
        or gallery_render["status"] != "complete"
        or gallery_encode["status"] != "complete"
        or gallery_report["status"] != "complete"
        or not all(gallery_report["scales"][f"negative_{scale:g}"]["pass"] for scale in SCALES)
    ):
        raise ValueError("GALLERY upstream gate or provenance is invalid")
    fit = sealed(Path("outputs/readout001/fit-report.json"))
    if fit["status"] != "complete" or fit["confirmation_opened"]:
        raise ValueError("READOUT confirmation provenance changed")
    names = ("templates", "queries", "verification", "examples", "bootstrap")
    streams = np.random.SeedSequence(20261001).spawn(len(names))
    seeds = {
        name: int(stream.generate_state(1)[0]) for name, stream in zip(names, streams, strict=True)
    }
    examples = _selection(confirmation, seeds["examples"], 2)
    verification = _selection(confirmation, seeds["verification"], 6)
    hashes = {
        identifier: fresh["family_files"][identifier.split(":", 1)[1]]["sha256"]
        for identifier in confirmation
    }
    if any(
        fresh["splits"][identifier.split(":", 1)[1]] != "development" for identifier in confirmation
    ):
        raise ValueError("Confirmation partition metadata changed")
    config = {
        "schema": "predictor-gallery001-v1",
        "master_seed": 20261001,
        "seeds": seeds,
        "development_ids": ids,
        "confirmation_ids": confirmation,
        "development_verification_ids": gallery_config["verification_ids"],
        "confirmation_verification_ids": verification,
        "development_examples": gallery_config["example_ids"],
        "confirmation_examples": examples,
        "confirmation_hashes": hashes,
        "scales": list(SCALES),
        "primary_scale": 10.24,
        "delta": 5.12,
        "eta": gallery_encode["eta"],
        "bootstrap": 2000,
        "experiment_limit_seconds": 1740,
        "total_limit_seconds": 1800,
    }
    digest = seal(root / "config.json", config)
    result = {
        "status": "passed",
        "sources": _sources(),
        "config": config,
        "config_sha256": digest,
        "audit_seconds": time.monotonic() - start,
    }
    seal(root / "audit.json", result)
    return result


def _ledger(root: Path) -> Budget:
    audited = audit(root)
    return Budget(root, audited["config_sha256"] + sha256_file(root / "audit.json"))


def _fingerprint(scaler: ActionScaler) -> dict:
    return {
        **model_fingerprint(CHECKPOINT, scaler),
        "pilot_manifest": sha256_file(PILOT / "collection-manifest.json"),
        "recovery_manifest": sha256_file(FRESH / "collection-manifest.json"),
    }


def _read_bank(
    root: Path, phase: str, identifier: str, index: int, config: dict, encoded: dict
) -> tuple[dict, dict]:
    if phase == "develop":
        base = GALLERY
        render_path = base / "renders" / f"{identifier.split(':', 1)[1]}__{index}.npz"
        encoded_path = base / "encoded" / f"{identifier.split(':', 1)[1]}__{index}.json"
    else:
        base = root / "confirm"
        render_path = _render_path(root, identifier, index)
        encoded_path = _encoded_path(root, identifier, index)
    meta, images = load_rendered(render_path)
    record = sealed(encoded_path)
    if (
        meta["identifier"] != identifier
        or meta["candidate_index"] != index
        or record["render_sha256"] != sha256_file(render_path)
    ):
        raise ValueError("Rendered candidate identity changed")
    if phase == "develop":
        if encoded["records"][f"{identifier}:{index}"] != sha256_file(encoded_path):
            raise ValueError("Development encoded record changed")
        exact = {
            **encoded["fingerprint"],
            "representation": "projected",
            "source": "gallery",
            "candidate": f"{identifier}:{index}",
        }
    else:
        if (
            record["audit_sha256"] != sha256_file(root / "audit.json")
            or record["input_sha256"] != config["confirmation_hashes"][identifier]
        ):
            raise ValueError("Confirmation encoding provenance changed")
        exact = {
            **encoded["fingerprint"],
            "representation": "projected",
            "source": "gallery",
            "candidate": f"{identifier}:{index}",
        }
    names = sorted(images)
    pixels = np.stack([images[name] for name in names])
    cache = LatentCache(base / "latent-cache")
    key = cache.key("real", {"frames": pixels}, exact)
    if key != record["cache_key"]:
        raise ValueError("Gallery cache identity mismatch")
    values, _ = cache.read("real", key, exact)
    embeddings = {name: values[0, j] for j, name in enumerate(names)}
    if any(array_digest(embeddings[name]) != record["embeddings_sha256"][name] for name in names):
        raise ValueError("Gallery embedding hash mismatch")
    return meta, embeddings


def _family_record(
    root: Path,
    phase: str,
    identifier: str,
    family: dict,
    predicted: np.ndarray,
    real: np.ndarray,
    config: dict,
    encoded: dict,
    verification: float | None,
) -> dict:
    if predicted.shape != (6, 2, 192) or real.shape != (15, 192):
        raise ValueError("Projected prediction or real alignment changed")
    terminal_real = real[3:].reshape(6, 2, 192)[:, -1]
    terminal_prediction = predicted[:, -1]
    start = real[2]
    labels = family["contacts"].astype(bool).tolist()
    start_block = family["history_states"][-1, 2:4]
    endpoint_block = family["future_states"][:, -1, 2:4]
    motion = (
        np.max(np.linalg.norm(family["block_trajectories"] - start_block, axis=2), axis=1) > 5.12
    ).tolist()
    agent_displacement = np.linalg.norm(
        family["future_states"][:, -1, :2] - family["history_states"][-1, :2], axis=1
    ).tolist()
    scale_rows = {}
    for scale in SCALES:
        label = f"negative_{scale:g}"
        endpoints = []
        for index in range(6):
            meta, embeddings = _read_bank(root, phase, identifier, index, config, encoded)
            if (
                meta["candidate_id"] != family["metadata"]["candidate_ids"][index]
                or array_digest(family["future_frames"][index, -1]) != meta["baseline_rgb_sha256"]
            ):
                raise ValueError("Candidate order or terminal RGB changed")
            selected = meta["scales"][label]["selected"]
            if len(selected) != 4 or len(meta["queries"]) != 2:
                endpoints.append({"eligible": False, "negative_count": len(selected)})
                continue
            bank = np.stack([embeddings["true"], *[embeddings[key] for key in selected]])
            scored = candidate_scores(terminal_prediction, start, bank, encoded["eta"])
            scored.update(
                {
                    "eligible": True,
                    "real": query_score(terminal_real[index], bank, encoded["eta"]),
                    "queries": [
                        query_score(embeddings[key], bank, encoded["eta"])
                        for key in meta["queries"]
                    ],
                }
            )
            endpoints.append(scored)
        regimes = {
            regime: family_mean(endpoints, participants(labels, regime)) for regime in REGIMES
        }
        motion_regimes = {
            regime: family_mean(endpoints, participants(motion, regime)) for regime in REGIMES
        }
        effect_groups = {"meaningful": set(), "negligible": set()}
        for i in range(6):
            for j in range(i + 1, 6):
                key = (
                    "meaningful"
                    if np.linalg.norm(endpoint_block[i] - endpoint_block[j]) > 5.12
                    else "negligible"
                )
                effect_groups[key].update((i, j))
        scale_rows[label] = {
            "endpoints": endpoints,
            "summary": family_mean(endpoints),
            "contact_regimes": regimes,
            "motion_regimes": motion_regimes,
            "effect_groups": {
                key: family_mean(endpoints, value) for key, value in effect_groups.items()
            },
        }
    latent = scores(cost_matrix(terminal_prediction, terminal_real), start, terminal_real)
    return {
        "id": identifier,
        "cohort": _cohort(identifier),
        "input_sha256": config["confirmation_hashes"][identifier]
        if phase == "confirm"
        else sealed(GALLERY / "config.json")["family_hashes"][identifier],
        "phase": phase,
        "prediction_sha256": array_digest(predicted),
        "real_sha256": array_digest(real),
        "verification_max": verification,
        "contact": labels,
        "motion": motion,
        "agent_displacement": agent_displacement,
        "block_endpoints": endpoint_block.tolist(),
        "latent": latent,
        "scales": scale_rows,
    }


def _phase_summary(records: list[dict], seed: int, check) -> dict:
    scales = {}
    for scale in SCALES:
        label = f"negative_{scale:g}"
        rows = [
            {"cohort": row["cohort"], "summary": row["scales"][label]["summary"]} for row in records
        ]
        grouped = aggregate(rows, seed + int(scale), check)
        strata = {}
        for category, names in (
            ("contact_regimes", REGIMES),
            ("motion_regimes", REGIMES),
            ("effect_groups", ("meaningful", "negligible")),
        ):
            strata[category] = {}
            for name in names:
                relevant = [
                    {"cohort": row["cohort"], "summary": row["scales"][label][category][name]}
                    for row in records
                ]
                strata[category][name] = aggregate(relevant, seed + int(scale) + len(name), check)
        controls = {}
        for cohort in ("challenge", "ordinary"):
            family_rows = [row for row in records if row["cohort"] == cohort]
            real_accuracy = []
            query_accuracy = []
            for row in family_rows:
                endpoints = [item for item in row["scales"][label]["endpoints"] if item["eligible"]]
                if endpoints:
                    real_accuracy.append(np.mean([item["real"]["correct"] for item in endpoints]))
                    query_accuracy.append(
                        np.mean(
                            [query["correct"] for item in endpoints for query in item["queries"]]
                        )
                    )
            controls[cohort] = {
                "families": len(real_accuracy),
                "real_accuracy": float(np.mean(real_accuracy)),
                "nonidentical_query_accuracy": float(np.mean(query_accuracy)),
            }
        scales[label] = {"groups": grouped, "strata": strata, "controls": controls}
    return {
        "scales": scales,
        "latent": latent_summary(records, seed + 100, check),
        "families": len(records),
        "agent_displacement_mean": float(
            np.mean([np.mean(row["agent_displacement"]) for row in records])
        ),
    }


def _predictor_example(root: Path, phase: str, record: dict) -> None:
    """Preselected candidate-zero gallery with scored latent distances."""
    identifier = record["id"]
    if phase == "develop":
        render_path = GALLERY / "renders" / f"{identifier.split(':', 1)[1]}__0.npz"
    else:
        render_path = _render_path(root, identifier, 0)
    meta, images = load_rendered(render_path)
    names = ["true", *meta["queries"], *meta["scales"]["negative_10.24"]["selected"]]
    score = record["scales"]["negative_10.24"]["endpoints"][0]
    canvas = Image.new("RGB", (224 * len(names), 280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        x = 224 * index
        canvas.paste(Image.fromarray(images[name]), (x, 25))
        draw.text((x + 3, 5), name, fill="black")
        for polygon in meta["footprints"][name]:
            points = [(x + px * 224 / 512, 25 + py * 224 / 512) for px, py in polygon]
            draw.line(points, fill=(220, 35, 35), width=1)
        cx, cy = meta["agent_center"]
        radius = meta["agent_radius"]
        draw.ellipse(
            (
                x + (cx - radius) * 224 / 512,
                25 + (cy - radius) * 224 / 512,
                x + (cx + radius) * 224 / 512,
                25 + (cy + radius) * 224 / 512,
            ),
            outline=(0, 80, 220),
        )
    if score["eligible"]:
        draw.text(
            (3, 253),
            "pred distances: "
            + ", ".join(f"{value:.3f}" for value in score["predictions"][0]["distances"]),
            fill="black",
        )
    output = root / phase / "examples" / f"{identifier.split(':', 1)[1]}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def develop(root: Path = ROOT) -> dict:
    audit(root)
    config = sealed(root / "config.json")
    gallery_encoded = sealed(GALLERY / "encode-manifest.json")
    if gallery_encoded["eta"] != config["eta"]:
        raise ValueError("Gallery numerical tolerance changed")
    ledger = _ledger(root)
    ledger.begin("develop")
    state = "complete"
    try:
        fresh = manifest(FRESH)
        scaler = ActionScaler.from_json(STATS)
        model = build_frozen_model(CHECKPOINT)
        fingerprint = _fingerprint(scaler)
        old = LatentCache(READOUT_CACHE)
        new = LatentCache(root / "latent-cache")
        for identifier in config["development_ids"]:
            ledger.check()
            path = _family_path(identifier, fresh)
            expected_hash = sealed(GALLERY / "config.json")["family_hashes"][identifier]
            if sha256_file(path) != expected_hash:
                raise ValueError("Development family hash changed")
            output = _record_path(root, "develop", identifier)
            if output.exists():
                existing = sealed(output)
                if existing["input_sha256"] != expected_hash:
                    raise ValueError("Development record mismatch")
                if identifier in config["development_examples"]:
                    _predictor_example(root, "develop", existing)
                continue
            family = load_family(path)
            real = _load_real(identifier, family, old, fingerprint, new, model)
            prediction = _predict(identifier, family, new, fingerprint, model, scaler)
            verification = (
                _verification(family, prediction, model, scaler)
                if identifier in config["development_verification_ids"]
                else None
            )
            record = _family_record(
                root,
                "develop",
                identifier,
                family,
                prediction,
                real,
                config,
                gallery_encoded,
                verification,
            )
            seal(
                output,
                record,
            )
            if identifier in config["development_examples"]:
                _predictor_example(root, "develop", record)
        records = [
            sealed(_record_path(root, "develop", identifier))
            for identifier in config["development_ids"]
        ]
        if sum(row["verification_max"] is not None for row in records) != 12:
            raise ValueError("Development verification incomplete")
        result = {
            "status": "complete",
            "families": 30,
            "records": {
                row["id"]: sha256_file(_record_path(root, "develop", row["id"])) for row in records
            },
            "summary": _phase_summary(records, config["seeds"]["bootstrap"], ledger.check),
        }
        seal(root / "develop.json", result)
        return result
    except TimeoutError:
        state = "incomplete_budget"
        return {
            "status": state,
            "completed_families": len(list((root / "develop" / "families").glob("*.json"))),
        }
    except Exception:
        state = "failed_integrity"
        raise
    finally:
        ledger.finish(state)


def freeze(root: Path = ROOT) -> dict:
    audited = audit(root)
    developed = sealed(root / "develop.json")
    if developed["status"] != "complete" or developed["families"] != 30:
        raise ValueError("Development analysis incomplete")
    for identifier, digest in developed["records"].items():
        if sha256_file(_record_path(root, "develop", identifier)) != digest:
            raise ValueError("Development record changed")
    body = {
        "status": "sealed",
        "audit_sha256": sha256_file(root / "audit.json"),
        "config_sha256": audited["config_sha256"],
        "develop_sha256": sha256_file(root / "develop.json"),
        "sources": audited["sources"],
        "confirmation_ids": audited["config"]["confirmation_ids"],
        "confirmation_hashes": audited["config"]["confirmation_hashes"],
        "criteria": {
            "primary_scale": 10.24,
            "challenge_primary": True,
            "bootstrap": 2000,
            "measurement_gates": {
                "coverage": 0.8,
                "families_four": 10,
                "query_accuracy": 0.95,
                "query_ratio_p05": 0.5,
            },
        },
    }
    target = root / "seal.json"
    if target.exists():
        if sealed(target) != body:
            raise ValueError("Frozen protocol mismatch")
    else:
        seal(target, body)
    return body


def _confirm_prepare(root: Path, config: dict, fresh: dict, model, scaler, ledger: Budget) -> dict:
    cache = LatentCache(root / "confirm" / "latent-cache")
    fingerprint = {**_fingerprint(scaler), "gallery_audit": sha256_file(root / "audit.json")}
    env = InstrumentedPushT()
    try:
        for identifier in config["confirmation_ids"]:
            path = _family_path(identifier, fresh)
            if sha256_file(path) != config["confirmation_hashes"][identifier]:
                raise ValueError("Confirmation family hash changed")
            family = load_family(path)
            for index in range(6):
                ledger.check()
                render_path = _render_path(root, identifier, index)
                if render_path.exists():
                    meta, images = load_rendered(render_path)
                    if (
                        meta["identifier"] != identifier
                        or meta["candidate_index"] != index
                        or meta["input_sha256"] != config["confirmation_hashes"][identifier]
                        or meta["audit_sha256"] != sha256_file(root / "audit.json")
                        or not np.array_equal(images["true"], family["future_frames"][index, -1])
                    ):
                        raise ValueError("Confirmation render mismatch")
                else:
                    meta, images = render_candidate(env, family, index, identifier, config["seeds"])
                    meta["input_sha256"] = config["confirmation_hashes"][identifier]
                    meta["audit_sha256"] = sha256_file(root / "audit.json")
                    save_rendered(render_path, meta, images)
                output = _encoded_path(root, identifier, index)
                if output.exists():
                    encoded_record = sealed(output)
                    if (
                        encoded_record["render_sha256"] != sha256_file(render_path)
                        or encoded_record["input_sha256"]
                        != config["confirmation_hashes"][identifier]
                    ):
                        raise ValueError("Confirmation encoding mismatch")
                    _read_bank(
                        root,
                        "confirm",
                        identifier,
                        index,
                        config,
                        {"fingerprint": fingerprint, "eta": config["eta"]},
                    )
                    continue
                embeddings, key, pixels, names = _encode(
                    model, images, fingerprint, cache, f"{identifier}:{index}"
                )
                maximum, mse = (
                    _verify_encoding(model, pixels, names, embeddings)
                    if identifier in config["confirmation_verification_ids"]
                    else (None, None)
                )
                seal(
                    output,
                    {
                        "identifier": identifier,
                        "candidate_index": index,
                        "input_sha256": config["confirmation_hashes"][identifier],
                        "audit_sha256": sha256_file(root / "audit.json"),
                        "render_sha256": sha256_file(render_path),
                        "cache_key": key,
                        "embeddings_sha256": {
                            name: array_digest(value) for name, value in embeddings.items()
                        },
                        "verification_max": maximum,
                        "verification_mse": mse,
                    },
                )
            if identifier in config["confirmation_examples"]:
                _example(root / "confirm", identifier)
        return {"fingerprint": fingerprint, "eta": config["eta"]}
    finally:
        env.close()


def _measurement_gate(root: Path, config: dict, encoded: dict, ledger: Budget) -> dict:
    verification = [
        sealed(_encoded_path(root, identifier, index))
        for identifier in config["confirmation_verification_ids"]
        for index in range(6)
    ]
    if len(verification) != 72 or any(
        row["verification_max"] is None or row["verification_max"] > 1e-6 for row in verification
    ):
        raise ValueError("Confirmation duplicate/reversed encoding check failed")
    rows = []
    for identifier in config["confirmation_ids"]:
        ledger.check()
        candidates = {f"negative_{scale:g}": [] for scale in SCALES}
        for index in range(6):
            meta, embeddings = _read_bank(root, "confirm", identifier, index, config, encoded)
            scored = _score_record(meta, embeddings, encoded["eta"])
            for label in candidates:
                candidates[label].append(scored["scales"][label])
        rows.append({"id": identifier, "cohort": _cohort(identifier), "candidates": candidates})
    result = {}
    for scale in SCALES:
        label = f"negative_{scale:g}"
        result[label] = {}
        for cohort in ("challenge", "ordinary"):
            subset = [row for row in rows if row["cohort"] == cohort]
            group = [
                {"cohort": cohort, "metrics": family_metrics(row["candidates"][label])}
                for row in subset
            ]
            summary = group_summary(
                group, config["seeds"]["bootstrap"] + int(scale), 2000, ledger.check
            )
            eligible = sum(item["eligible"] for row in subset for item in row["candidates"][label])
            families_four = sum(
                sum(item["eligible"] for item in row["candidates"][label]) >= 4 for row in subset
            )
            summary.update(
                {
                    "eligible_endpoints": eligible,
                    "total_endpoints": len(subset) * 6,
                    "families_four": families_four,
                }
            )
            summary["pass"] = (
                eligible / (len(subset) * 6) >= 0.8
                and families_four >= 10
                and summary.get("accuracy", 0) >= 0.95
                and summary.get("ratio_p05", -1) >= 0.5
            )
            result[label][cohort] = summary
    return {
        "status": "complete",
        "scales": result,
        "primary_pass": all(
            result["negative_10.24"][cohort]["pass"] for cohort in ("challenge", "ordinary")
        ),
    }


def confirm(root: Path = ROOT) -> dict:
    freeze(root)
    config = sealed(root / "config.json")
    seal_hash = sha256_file(root / "seal.json")
    marker = root / "confirmation-opened.json"
    if marker.exists():
        if sealed(marker)["seal_sha256"] != seal_hash:
            raise ValueError("Confirmation opening seal mismatch")
    else:
        seal(
            marker,
            {
                "seal_sha256": seal_hash,
                "opened_at_utc": datetime.now(timezone.utc).isoformat(),
                "family_count": 60,
            },
        )
    ledger = _ledger(root)
    ledger.begin("confirm")
    state = "complete"
    try:
        fresh = manifest(FRESH)
        scaler = ActionScaler.from_json(STATS)
        model = build_frozen_model(CHECKPOINT)
        encoded = _confirm_prepare(root, config, fresh, model, scaler, ledger)
        gate_path = root / "confirmation-gate.json"
        if gate_path.exists():
            gate = sealed(gate_path)
            if gate["status"] != "complete":
                raise ValueError("Confirmation gate incomplete")
        else:
            gate = _measurement_gate(root, config, encoded, ledger)
            seal(gate_path, gate)
        if not gate["primary_pass"]:
            result = {
                "status": "unassessable_measurement",
                "gate_sha256": sha256_file(gate_path),
                "families_rendered": 60,
                "families_scored": 0,
            }
            seal(root / "confirmation.json", result)
            return result
        old = LatentCache(READOUT_CACHE)
        new = LatentCache(root / "latent-cache")
        fingerprint = _fingerprint(scaler)
        for identifier in config["confirmation_ids"]:
            ledger.check()
            output = _record_path(root, "confirm", identifier)
            if output.exists():
                record = sealed(output)
                if (
                    record["input_sha256"] != config["confirmation_hashes"][identifier]
                    or record["phase"] != "confirm"
                ):
                    raise ValueError("Confirmation family record mismatch")
                if identifier in config["confirmation_examples"]:
                    _predictor_example(root, "confirm", record)
                continue
            path = _family_path(identifier, fresh)
            if sha256_file(path) != config["confirmation_hashes"][identifier]:
                raise ValueError("Confirmation family changed")
            family = load_family(path)
            real = _load_real(identifier, family, old, fingerprint, new, model)
            predicted = _predict(identifier, family, new, fingerprint, model, scaler)
            verification = (
                _verification(family, predicted, model, scaler)
                if identifier in config["confirmation_verification_ids"]
                else None
            )
            record = _family_record(
                root,
                "confirm",
                identifier,
                family,
                predicted,
                real,
                config,
                encoded,
                verification,
            )
            seal(
                output,
                record,
            )
            if identifier in config["confirmation_examples"]:
                _predictor_example(root, "confirm", record)
        records = [
            sealed(_record_path(root, "confirm", identifier))
            for identifier in config["confirmation_ids"]
        ]
        if len(records) != 60 or sum(row["verification_max"] is not None for row in records) != 12:
            raise ValueError("Confirmation family/verification count incomplete")
        result = {
            "status": "complete",
            "families": 60,
            "gate_sha256": sha256_file(gate_path),
            "records": {
                row["id"]: sha256_file(_record_path(root, "confirm", row["id"])) for row in records
            },
            "summary": _phase_summary(records, config["seeds"]["bootstrap"] + 1000, ledger.check),
        }
        seal(root / "confirmation.json", result)
        return result
    except TimeoutError:
        state = "incomplete_budget"
        return {
            "status": state,
            "completed_families": len(list((root / "confirm" / "families").glob("*.json"))),
        }
    except Exception:
        state = "failed_integrity"
        raise
    finally:
        ledger.finish(state)


def report(root: Path = ROOT) -> dict:
    audit(root)
    ledger = _ledger(root)
    ledger.begin("report")
    state = "complete"
    try:
        development = sealed(root / "develop.json") if (root / "develop.json").exists() else None
        confirmation = (
            sealed(root / "confirmation.json") if (root / "confirmation.json").exists() else None
        )
        complete = (
            development is not None
            and development["status"] == "complete"
            and confirmation is not None
            and confirmation["status"] == "complete"
        )
        outcome = "unassessable"
        central = None
        if complete:
            central = confirmation["summary"]["scales"]["negative_10.24"]["groups"]["challenge"]
            metrics = central["metrics"]
            checks = {
                "persistence_advantage": metrics["persistence_advantage"]["ci95"][0] > 0,
                "action_advantage": metrics["action_advantage"]["ci95"][0] > 0,
                "matched_margin": metrics["matched_margin"]["mean"] > 0,
                "accuracy": metrics["matched_accuracy"]["mean"] > 0.2,
            }
            outcome = "positive" if all(checks.values()) else "not_positive"
            central = {"metrics": metrics, "checks": checks, "families": central["families"]}
        result = {
            "status": "complete" if complete else "incomplete",
            "outcome": outcome,
            "central": central,
            "development_sha256": sha256_file(root / "develop.json") if development else None,
            "confirmation_sha256": sha256_file(root / "confirmation.json")
            if confirmation
            else None,
            "confirmation_opened": (root / "confirmation-opened.json").exists(),
            "runtime_seconds_before_report": ledger.data["used_seconds"],
        }
        seal(root / "report.json", result)
        return result
    except Exception:
        state = "failed_integrity"
        raise
    finally:
        ledger.finish(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "develop", "seal", "confirm", "report"))
    arguments = parser.parse_args()
    stage = {
        "audit": audit,
        "develop": develop,
        "seal": freeze,
        "confirm": confirm,
        "report": report,
    }[arguments.stage]
    print(json.dumps(stage(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
