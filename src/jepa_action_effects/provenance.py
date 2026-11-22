"""Provenance validation for sealed runs (REVIEW-FIX-002, D026 finding 1).

The sealed runners record hashes in `seal.json`, but `confirm` only re-checked source
hashes, and cached per-family records were trusted without checking their inputs.
Checking a JSON file against its own sidecar is not enough: a producing stage can
rewrite both. This validator checks every recorded hash against the seal and the
files, the audit → develop → seal source chain, family bytes, the inference
fingerprint, and cached records.

Future sealed runners must call `require_valid` before opening or resuming
confirmation. Completed runners are historical and are not retrofitted (editing them
would break their source seals); `python -m jepa_action_effects.provenance` checks
them read-only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jepa_action_effects.pilot_collection import sha256_file

PASSED, FAILED, NOT_RECORDED = "passed", "failed", "not_recorded"
STRICT, LEGACY = "strict", "legacy"


class ProvenanceError(ValueError):
    """Raised when a sealed run fails validation."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    return sidecar.exists() and sidecar.read_text().strip() == sha256_file(path)


def validate_run(
    root: Path,
    current_sources: Mapping[str, str],
    family_index: Mapping[str, Path],
    record_dirs: list[Path],
    selection_path: Path | None = None,
    current_fingerprint: Mapping[str, Any] | None = None,
    mode: str = STRICT,
) -> dict[str, Any]:
    """Return {"checks": [...], "valid": bool}; never modifies the run.

    `family_index` maps each confirmation record id to its family artifact.
    `record_dirs` hold the cached per-family confirmation records.

    `mode="strict"` (new runs) fails any cached record lacking `input_sha256` or
    `fingerprint`. `mode="legacy"` (completed historical runs, which never recorded
    them) reports missing metadata as not recorded instead.
    """
    if mode not in (STRICT, LEGACY):
        raise ValueError(f"Unknown validation mode {mode!r}")
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool | None, detail: str = "") -> None:
        status = NOT_RECORDED if ok is None else (PASSED if ok else FAILED)
        checks.append({"check": name, "status": status, "detail": detail})

    seal_path, audit_path = root / "seal.json", root / "audit.json"
    for name in ("config.json", "audit.json", "develop.json", "seal.json"):
        check(f"sidecar:{name}", _sidecar_ok(root / name))
    seal, audit, develop = _load(seal_path), _load(audit_path), _load(root / "develop.json")

    # Source chain: audit == seal == current code.
    check("sources:audit_equals_seal", audit.get("sources") == seal.get("sources"))
    check("sources:current_equals_seal", dict(current_sources) == seal.get("sources"))

    # Every hash recorded in the seal against the files it names.
    for key, name in (
        ("config_sha256", "config.json"),
        ("audit_sha256", "audit.json"),
        ("develop_sha256", "develop.json"),
        ("collection_sha256", "collection.json"),
    ):
        if key in seal:
            check(f"seal:{key}", seal[key] == sha256_file(root / name))
    if "reserved_manifest_sha256" in seal:
        manifest = root / "reserved" / "collection-manifest.json"
        check(
            "seal:reserved_manifest_sha256",
            seal["reserved_manifest_sha256"] == sha256_file(manifest),
        )
    if "config_sha256" in audit:
        check("audit:config_sha256", audit["config_sha256"] == sha256_file(root / "config.json"))
    if selection_path is not None and "readout_selection_sha256" in seal:
        check(
            "seal:readout_selection_sha256",
            seal["readout_selection_sha256"] == sha256_file(selection_path),
        )
    if selection_path is not None:
        # The selection JSON names the probe artifact; hash the artifact's bytes too.
        selection = _load(selection_path)
        artifact = selection_path.parent / selection["path"]
        check(
            "probe:artifact_sha256",
            artifact.exists() and sha256_file(artifact) == selection["path_sha256"],
            str(artifact),
        )

    # Family bytes against their manifest entries.
    bad = [key for key, path in family_index.items() if not path.exists()]
    check(
        "families:present", not bad, f"missing {len(bad)}" if bad else f"{len(family_index)} files"
    )

    # Inference fingerprint: develop == confirm (== current, when supplied).
    confirm_path = root / "confirm.json"
    confirm = _load(confirm_path) if confirm_path.exists() else None
    if confirm is not None:
        check(
            "fingerprint:develop_equals_confirm",
            develop.get("fingerprint") == confirm.get("fingerprint"),
        )
    if current_fingerprint is not None:
        check(
            "fingerprint:current_equals_develop",
            dict(current_fingerprint) == develop.get("fingerprint"),
        )

    # Cached records: sidecar, a known family, the recorded input hash and fingerprint.
    unknown, stale, unverified, broken = [], [], 0, []
    foreign, unfingerprinted = [], 0
    for directory in record_dirs:
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".sha256"):
                continue
            if not _sidecar_ok(path):
                broken.append(path.name)
                continue
            record = _load(path)
            family = family_index.get(record.get("id"))
            if family is None or not family.exists():
                unknown.append(record.get("id"))
            elif "input_sha256" in record:
                if record["input_sha256"] != sha256_file(family):
                    stale.append(record["id"])
            else:
                unverified += 1
            if "fingerprint" in record:
                if record["fingerprint"] != develop.get("fingerprint"):
                    foreign.append(record.get("id"))
            else:
                unfingerprinted += 1
    check("records:sidecars", not broken, ", ".join(broken[:5]))
    check("records:known_family", not unknown, ", ".join(map(str, unknown[:5])))
    check("records:input_sha256", not stale, ", ".join(stale[:5]))
    check("records:fingerprint", not foreign, ", ".join(map(str, foreign[:5])))
    for name, missing, what in (
        ("records:input_sha256_recorded", unverified, "input hash"),
        ("records:fingerprint_recorded", unfingerprinted, "fingerprint"),
    ):
        if missing:
            detail = f"{missing} records carry no {what}"
            check(name, False if mode == STRICT else None, detail)
    return {"checks": checks, "valid": all(c["status"] != FAILED for c in checks)}


def require_valid(*args, **kwargs) -> dict[str, Any]:
    """Validate and raise ProvenanceError listing every failed check."""
    report = validate_run(*args, **kwargs)
    failed = [c for c in report["checks"] if c["status"] == FAILED]
    if failed:
        raise ProvenanceError("; ".join(f"{c['check']} {c['detail']}".strip() for c in failed))
    return report


# --------------------------------------------------------------------------- family indexes
def reserved_manifest_index(root: Path, prefix: str = "reserved:") -> dict[str, Path]:
    """Record id → family path for runs whose reserved families use a pilot manifest."""
    manifest = _load(root / "reserved" / "collection-manifest.json")
    index = {}
    for key, record in manifest["family_files"].items():
        path = root / "reserved" / record["path"]
        if sha256_file(path) != record["sha256"]:
            path = Path("/nonexistent/hash-mismatch") / key
        index[f"{prefix}{key}"] = path
    return index


def collection_index(root: Path, split: str = "reserved") -> dict[str, Path]:
    """Record id → family path for SHIFT-style `collection.json` runs."""
    collection = _load(root / "collection.json")
    index = {}
    for key, entry in collection["families"].items():
        if entry["split"] != split:
            continue
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            path = Path("/nonexistent/hash-mismatch") / key
        index[key] = path
    return index


def check_completed_runs() -> dict[str, Any]:
    """Read-only legacy-mode validation of PHYS-002, SHIFT-002 and H3-PLAN-002."""
    from jepa_action_effects import phys002, planning002, shift002
    from jepa_action_effects.next_common import load_model

    _, _, fingerprint = load_model()
    selection = Path("outputs/readout002/selection.json")
    runs = {
        "PHYS-002": (
            Path("outputs/phys002"),
            phys002._sources(),
            reserved_manifest_index(Path("outputs/phys002")),
            [Path("outputs/phys002/confirm-reserved/families")],
            selection,
        ),
        "SHIFT-002": (
            Path("outputs/shift002"),
            shift002._sources(),
            collection_index(Path("outputs/shift002")),
            [Path("outputs/shift002/confirm")],
            selection,
        ),
        "H3-PLAN-002": (
            Path("outputs/planning002"),
            planning002._sources(),
            reserved_manifest_index(Path("outputs/planning002")),
            [Path("outputs/planning002/confirm")],
            None,
        ),
    }
    return {
        name: validate_run(root, sources, index, dirs, sel, fingerprint, mode=LEGACY)
        for name, (root, sources, index, dirs, sel) in runs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/manifests/provenance-check.json")
    )
    args = parser.parse_args()
    from jepa_action_effects.next_common import write_json

    result = check_completed_runs()
    write_json(args.manifest, result)
    for name, report in result.items():
        summary = {c["check"]: c["status"] for c in report["checks"]}
        print(name, "valid" if report["valid"] else "INVALID", json.dumps(summary))


if __name__ == "__main__":
    main()
