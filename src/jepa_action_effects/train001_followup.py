"""TRAIN-001 review follow-up (TRAIN-001_REVIEW.md, D029): read-only over sealed outputs.

The sealed TRAIN-001 modules (`train001.sources()`) are historical and unchanged.
This module adds, without modifying any sealed artifact:

- **Contrast-level readout sensitivity.** The sealed `measurement_dependent` flag
  compares pooled verdict strings. Development eligibility uses the primary readout
  only, so once it fails every readout gets the same verdict and the flag cannot
  detect disagreement. This compares interval signs of each contrast, the ordinary
  outcome and the confirmation gates across readouts instead.
- **Training-lineage validation.** The sealed validator checks run/checkpoint bytes
  but not the run's recorded training collection/cache hashes, loss weights or
  optimization specification. `validate_lineage` checks them against the seal and the
  protocol; `validate_with_lineage` combines both for future sealed runs.
- **Exact report numbers** extracted from the sealed summary (no transcription).

    python -m jepa_action_effects.train001_followup
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jepa_action_effects.latent_audit import sealed, write_json
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001_metrics import interval_sign
from jepa_action_effects.train001_protocol import CONDITIONS, MAGNITUDES, READOUTS, Protocol

MANIFEST = Path("docs/manifests/train001-followup.json")
PRIMARY_CONTRASTS = (
    "balanced_minus_archive",
    "balanced_minus_frozen",
    "archive_minus_frozen",
    "calibrated_frozen_minus_frozen",
)
ORDINARY_CONTRASTS = ("balanced_minus_frozen", "archive_minus_frozen", "balanced_minus_archive")


class ValidationFailure(RuntimeError):
    """Combined sealed/lineage validation failed; no presentation or manifest is written."""

    def __init__(self, failed: list[dict[str, Any]]):
        super().__init__(f"{len(failed)} follow-up validation check(s) failed")
        self.failed = failed


# --------------------------------------------------------------------------- sensitivity
def _entry(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"estimate": value["estimate"], "ci": value["ci"], "sign": interval_sign(value["ci"])}


def _agreement(per_readout: Mapping[str, dict]) -> dict[str, Any]:
    signs = {readout: entry["sign"] for readout, entry in per_readout.items()}
    return {"per_readout": dict(per_readout), "signs_agree": len(set(signs.values())) == 1}


def contrast_sensitivity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compare each contrast's interval sign, ordinary outcome and gates across readouts."""
    readouts = summary["readouts"]
    quantities: dict[str, dict] = {}
    for name in PRIMARY_CONTRASTS:
        quantities[f"primary:{name}"] = _agreement(
            {r: _entry(b["primary"]["bootstrap"][name]) for r, b in readouts.items()}
        )
        if name in ("balanced_minus_archive", "balanced_minus_frozen"):
            for scene in ("challenge", "archive"):
                quantities[f"subgroup_exploratory:{scene}:{name}"] = _agreement(
                    {
                        r: _entry(b["primary"]["bootstrap"][name]["by_scene"][scene])
                        for r, b in readouts.items()
                    }
                )
    for name in ORDINARY_CONTRASTS:
        quantities[f"ordinary:{name}"] = _agreement(
            {r: _entry(b["ordinary"]["bootstrap"][name]) for r, b in readouts.items()}
        )
    outcomes = {r: b["ordinary"]["outcome"] for r, b in readouts.items()}
    gates = {
        r: {cell: gate["passed"] for cell, gate in b["measurement_gates"]["cells"].items()}
        for r, b in readouts.items()
    }
    disagreements = sorted(k for k, v in quantities.items() if not v["signs_agree"])
    return {
        "note": (
            "Development eligibility is computed once with the primary readout, so the "
            "pooled verdict string is identical across readouts by construction and is "
            "not a sensitivity test. Sensitivity is assessed per contrast (interval sign), "
            "per ordinary outcome and per confirmation gate."
        ),
        "quantities": quantities,
        "ordinary_outcome": {"per_readout": outcomes, "agree": len(set(outcomes.values())) == 1},
        "confirmation_gates": {
            "per_readout": gates,
            "agree": len({json.dumps(v, sort_keys=True) for v in gates.values()}) == 1,
        },
        "sign_disagreements": disagreements,
        "contrast_level_measurement_dependent": bool(disagreements)
        or len(set(outcomes.values())) > 1,
    }


# --------------------------------------------------------------------------- lineage
def _expected_weights(condition: str, development: Mapping[str, Any]) -> list[float]:
    if condition == "balanced":
        return [1.0 / len(MAGNITUDES)] * len(MAGNITUDES)
    if condition == "archive_weighted":
        return [float(w) for w in development["weights"]["weights"]]
    raise ValueError(condition)


def validate_lineage(root: Path, protocol: Protocol, base_weights_sha256: str) -> dict[str, Any]:
    """Each trained model's data lineage and optimization spec against seal and protocol."""
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "status": "passed" if ok else "failed", "detail": detail})

    seal = json.loads((root / "seal.json").read_text())
    development = json.loads((root / "development" / "collection.json").read_text())
    sealed_hashes = seal["file_sha256"]
    current = {
        "collection": sha256_file(root / "development" / "collection.json"),
        "cache": sha256_file(root / "cache" / "development.json"),
    }
    expected_models = {f"{c}-{s}" for c in CONDITIONS for s in protocol.training_seeds}
    trained = {k: v for k, v in seal["registry"].items() if v["kind"] == "trained"}
    check("lineage:trained_models", set(trained) == expected_models, sorted(trained))
    for model_id, entry in sorted(trained.items()):
        run = json.loads((root / entry["run"]).read_text())
        spec = run.get("spec", {})
        weights = _expected_weights(entry["condition"], development)
        problems = []
        for kind in ("collection", "cache"):
            recorded = run.get(f"{kind}_sha256")
            if recorded != sealed_hashes[f"{kind}_development"] or recorded != current[kind]:
                problems.append(f"{kind}_sha256")
        if run.get("base_weights_sha256") != base_weights_sha256:
            problems.append("base_weights_sha256")
        if [float(w) for w in run.get("weights", [])] != weights:
            problems.append("weights")
        expected_spec = {
            "condition": entry["condition"],
            "seed": entry["seed"],
            "learning_rate": seal["learning_rate"],
            "updates": protocol.updates,
            "weights": weights,
            "weight_decay": protocol.weight_decay,
            "gradient_clip": protocol.gradient_clip,
            "per_scene": protocol.families_per_scene_batch,
            "threads": protocol.threads,
            "checkpoint_updates": list(protocol.checkpoint_updates),
            "resume_every": protocol.resume_every,
            "curve_every": protocol.curve_every,
            "validation_every": protocol.validation_every,
        }
        for key, value in expected_spec.items():
            actual = spec.get(key)
            if isinstance(actual, (list, tuple)):
                actual = [float(x) if key == "weights" else x for x in actual]
            if actual != value:
                problems.append(f"spec.{key}")
        check(f"lineage:{model_id}", not problems, problems)
    return {"checks": checks, "valid": all(c["status"] == "passed" for c in checks)}


def validate_with_lineage(
    root: Path,
    protocol: Protocol,
    mode: str,
    current_sources: Mapping[str, str],
    current_fingerprint: Mapping[str, Any],
    base_weights_sha256: str,
) -> dict[str, Any]:
    """Sealed-run validation plus training lineage; required for future TRAIN runs."""
    from jepa_action_effects.train001_provenance import validate

    first = validate(
        root, protocol, mode, current_sources, current_fingerprint, base_weights_sha256
    )
    lineage = validate_lineage(root, protocol, base_weights_sha256)
    checks = first["checks"] + lineage["checks"]
    return {"checks": checks, "valid": first["valid"] and lineage["valid"]}


# --------------------------------------------------------------------------- report numbers
def exact_numbers(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Unrounded values the written reports cite (primary readout unless stated)."""
    primary = summary["readouts"][READOUTS[0]]
    latent = summary["secondary_terminal_latent_mse"]["contrasts"]
    return {
        "latent": {k: _entry(latent[k]) for k in ORDINARY_CONTRASTS},
        "primary": {k: _entry(primary["primary"]["bootstrap"][k]) for k in PRIMARY_CONTRASTS},
        "ordinary": {k: _entry(primary["ordinary"]["bootstrap"][k]) for k in ORDINARY_CONTRASTS},
    }


def smoke_seconds(ledger: Path = Path("outputs/train001-smoke-budget/ledger.json")) -> float | None:
    if not ledger.exists():
        return None
    sessions = json.loads(ledger.read_text())["sessions"]
    return float(sum(s["ended"] - s["started"] for s in sessions if s.get("ended")))


def run(protocol: Protocol | None = None) -> dict[str, Any]:
    from jepa_action_effects.interface_audit import WEIGHTS_SHA256
    from jepa_action_effects.train001 import frozen_fingerprint, sources
    from jepa_action_effects.train001_provenance import COMPLETE
    from jepa_action_effects.train001_report import regenerate_presentation

    started = time.monotonic()
    protocol = protocol or Protocol.production()
    root = protocol.run_root
    summary = sealed(root / "confirm" / "summary.json")
    validation = validate_with_lineage(
        root, protocol, COMPLETE, sources(), frozen_fingerprint(), WEIGHTS_SHA256
    )
    if not validation["valid"]:
        # TRAIN-REVIEW-003: refuse before any figure, table or manifest is replaced.
        raise ValidationFailure([c for c in validation["checks"] if c["status"] != "passed"])
    sensitivity = contrast_sensitivity(summary)
    presentation = regenerate_presentation(protocol, root, sensitivity)
    payload = {
        "task": "TRAIN-001 review follow-up (D029)",
        "label": "Read-only reanalysis of sealed outputs; no sealed artifact or verdict changed",
        "summary_sha256": sha256_file(root / "confirm" / "summary.json"),
        "seal_sha256": sha256_file(root / "seal.json"),
        "followup_source_sha256": sha256_file(Path(__file__)),
        "report_source_sha256": sha256_file(Path(__file__).with_name("train001_report.py")),
        "validation_with_lineage": {
            "valid": validation["valid"],
            "checks": len(validation["checks"]),
            "failed": [c for c in validation["checks"] if c["status"] != "passed"],
            "lineage": [c for c in validation["checks"] if c["check"].startswith("lineage:")],
        },
        "sealed_measurement_dependent_flag": summary["readout_sensitivity"],
        "contrast_sensitivity": sensitivity,
        "exact_numbers": exact_numbers(summary),
        "smoke_run_seconds_outside_production_ledger": smoke_seconds(),
        "presentation": presentation,
        "seconds": time.monotonic() - started,
    }
    write_json(MANIFEST, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    try:
        payload = run()
    except ValidationFailure as error:
        print(json.dumps({"valid_with_lineage": False, "failed": error.failed}, default=str))
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "valid_with_lineage": payload["validation_with_lineage"]["valid"],
                "sign_disagreements": payload["contrast_sensitivity"]["sign_disagreements"],
                "manifest": str(MANIFEST),
            }
        )
    )


if __name__ == "__main__":
    main()
