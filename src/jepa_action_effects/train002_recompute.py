"""Recompute the sealed TRAIN-002 summary from its saved per-unit records (read-only).

    python -m jepa_action_effects.train002_recompute

Needs the shared `outputs/train002/` root. Verifies the seal (sources, sealed files and
checkpoints), then loads all saved confirmation records with their SHA-256 sidecars,
unit/model identities, model fingerprints and readout hashes, calls
`train002_metrics.summarize` with the sealed gates and protocol bootstrap settings, and
compares the full result with `outputs/train002/confirm/summary.json` and the summary in
`docs/manifests/train002-runtime.json`.

Unlike `train002 confirm` on a completed root, this recomputes every estimate and
interval; it runs no model and writes nothing. Matching values establish computational
consistency with the saved records, not a fresh replication. Exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from jepa_action_effects.latent_audit import sealed
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001 import GateFailure
from jepa_action_effects.train001_protocol import READOUTS, Protocol
from jepa_action_effects.train002 import MANIFEST, _model_ids, protocol, verify_seal


def differences(left: Any, right: Any, path: str = "", tolerance: float = 0.0) -> list[str]:
    """Paths where two JSON-like trees differ (floats compared within `tolerance`)."""
    if isinstance(left, dict) and isinstance(right, dict):
        out = [f"{path}/{k}: missing" for k in sorted(set(left) ^ set(right))]
        for key in sorted(set(left) & set(right)):
            out += differences(left[key], right[key], f"{path}/{key}", tolerance)
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        out = []
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            out += differences(a, b, f"{path}[{i}]", tolerance)
        return out
    if isinstance(left, bool) or isinstance(right, bool):
        same = type(left) is type(right) and left == right
        return [] if same else [f"{path}: {left!r} != {right!r}"]
    numeric = (int, float)
    if isinstance(left, numeric) and isinstance(right, numeric):
        if isinstance(left, float) and isinstance(right, float):
            if math.isnan(left) and math.isnan(right):
                return []
        return [] if abs(left - right) <= tolerance else [f"{path}: {left!r} != {right!r}"]
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def load_records(p: Protocol, payload: dict[str, Any], model_id: str, kind: str) -> list[dict]:
    """Saved confirmation records for one model, in the order `train002 confirm` used."""
    root = p.run_root
    collection = sealed(root / "confirmation" / "collection.json")
    manifest = sealed(root / "cache" / "confirmation.json")
    units = sorted(collection["families"] if kind == "family" else collection["windows"])
    records_dir = root / "confirm" / "records"
    out = []
    for unit in units:
        path = records_dir / f"{unit}__{model_id}.json"
        digest = path.with_name(path.name + ".sha256").read_text().strip()
        if sha256_file(path) != digest:
            raise GateFailure(f"Record changed: {path.name}")
        record = json.loads(path.read_text())
        checks = {
            "unit": record["unit"] == unit,
            "model": record["model"] == model_id,
            "fingerprint": record["fingerprint_sha256"] == payload["fingerprints"][model_id],
            "input": record["input_sha256"] == manifest["records"][unit]["sha256"],
            "readouts": record["readout_sha256"] == payload["readouts"],
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise GateFailure(f"{path.name}: {', '.join(failed)} mismatch")
        out.append(record)
    return out


def recompute(p: Protocol | None = None) -> dict[str, Any]:
    from jepa_action_effects.train002_metrics import summarize

    p = p or protocol()
    payload = verify_seal(p)
    models = _model_ids(payload["registry"])
    predictors = [m for m in models if m != "real-frames"]
    families = {m: load_records(p, payload, m, "family") for m in models}
    windows = {m: load_records(p, payload, m, "window") for m in models}
    summary = summarize(
        families["real-frames"],
        {m: families[m] for m in predictors},
        windows["real-frames"],
        {m: windows[m] for m in predictors},
        READOUTS,
        sealed(p.run_root / "gate.json")["primary"],
        p.bootstrap_draws,
        p.primary_bootstrap_seed,
        p.ordinary_bootstrap_seed,
        p.delta,
        p.regression_tolerance,
    )
    count = sum(len(v) for v in families.values()) + sum(len(v) for v in windows.values())
    return {"summary": json.loads(json.dumps(summary)), "records": count, "models": len(models)}


def compare(result: dict[str, Any], p: Protocol | None = None) -> dict[str, Any]:
    p = p or protocol()
    sealed_summary = sealed(p.run_root / "confirm" / "summary.json")
    tracked = json.loads(MANIFEST.read_text())["summary"]
    against_sealed = differences(result["summary"], sealed_summary)
    against_tracked = differences(result["summary"], tracked)
    readout = result["summary"]["readouts"]["mlp-20261005"]
    return {
        "records": result["records"],
        "models": result["models"],
        "equals_sealed_summary": not against_sealed,
        "equals_tracked_manifest": not against_tracked,
        "differences": (against_sealed + against_tracked)[:20],
        "headline": {
            "primary_verdict": result["summary"]["primary_verdict"],
            "dose_omitted_minus_balanced": readout["bootstrap"]["dose_omitted_minus_balanced"],
            "transfer_fraction": readout["transfer_fraction"],
            "means_0.60": readout["means_0.60"],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    try:
        report = compare(recompute())
    except (GateFailure, FileNotFoundError) as error:
        print(json.dumps({"status": "failed", "detail": str(error)}))
        sys.exit(1)
    ok = report["equals_sealed_summary"] and report["equals_tracked_manifest"]
    print(json.dumps({"status": "match" if ok else "mismatch", **report}, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
