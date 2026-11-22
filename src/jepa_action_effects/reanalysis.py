"""REANALYSIS-001: exploratory re-reading of saved PREDICTOR-GALLERY and LATENT outputs.

    python -m jepa_action_effects.reanalysis run

Reporting only. It reads sealed per-family JSON records and collection families,
never runs the model, and never rewrites historical artifacts. Every family used
here was already opened, so all results are exploratory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jepa_action_effects.next_common import (
    FRESH,
    MANIFESTS,
    collection_manifest,
    load_family,
    runtime_info,
    sealed,
    source_hashes,
    write_json,
)
from jepa_action_effects.next_metrics import (
    REGIMES,
    bootstrap,
    boundary_pairs,
    latent_pair_totals,
    mean_of,
    paired_contrast,
)
from jepa_action_effects.predictor_gallery_metrics import aggregate, family_mean

ROOT = Path("outputs/reanalysis001")
GALLERY_ROOT = Path("outputs/predictor_gallery001")
LATENT_ROOT = Path("outputs/latent001")
SCALES = ("negative_10.24", "negative_20.48")
SEED = 20261002


def _records(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No sealed family records under {directory}")
    return [sealed(path) for path in paths]


def gallery_motion_split(records: list[dict], seed: int = SEED) -> dict[str, Any]:
    """Matched/persistence/wrong gallery margins on block-moved vs block-static endpoints.

    `motion` is the saved per-candidate label: maximum block trajectory displacement
    from the branch point above 5.12. Persistence is correct by construction on
    static endpoints, so its contrast is only informative on moved ones.
    """
    result: dict[str, Any] = {}
    for scale in SCALES:
        result[scale] = {}
        for group, wanted in (("moved", True), ("static", False)):
            rows = []
            for record in records:
                indices = {i for i, moved in enumerate(record["motion"]) if bool(moved) == wanted}
                endpoints = record["scales"][scale]["endpoints"]
                rows.append(
                    {"cohort": record["cohort"], "summary": family_mean(endpoints, indices)}
                )
            result[scale][group] = aggregate(rows, seed)
    return result


def latent_family(record: dict, controls, candidate_ids: list[str]) -> dict[str, Any]:
    """Boundary-pair correspondence and latent slope terms for one LATENT-001 family."""
    designed = set(boundary_pairs(controls, candidate_ids))
    pairs = record["pairs"]

    def is_designed(row: dict) -> bool:
        return (row["i"], row["j"]) in designed

    boundary = {}
    for name in REGIMES:
        selected = [row["A"] for row in pairs if is_designed(row) and row["regime"] == name]
        boundary[name] = sum(selected) / len(selected) if selected else None
    totals = {
        "all": latent_pair_totals(pairs, lambda row: True),
        "designed": latent_pair_totals(pairs, is_designed),
        **{
            f"regime_{name}": latent_pair_totals(
                pairs, lambda row, name=name: row["regime"] == name
            )
            for name in REGIMES
        },
        **{
            f"designed_{name}": latent_pair_totals(
                pairs, lambda row, name=name: is_designed(row) and row["regime"] == name
            )
            for name in REGIMES
        },
    }
    return {
        "id": record["id"],
        "cohort": record["cohort"],
        "designed_pairs": len(designed),
        "boundary": boundary,
        "totals": totals,
    }


def _slope(group: str):
    def statistic(rows):
        energy = sum(row["totals"][group]["energy"] for row in rows)
        if energy <= 0:
            return None
        return sum(row["totals"][group]["inner"] for row in rows) / energy

    return statistic


def _spread_ratio(group: str):
    def statistic(rows):
        energy = sum(row["totals"][group]["energy"] for row in rows)
        return (
            None
            if energy <= 0
            else (sum(row["totals"][group]["predicted_energy"] for row in rows) / energy)
        )

    return statistic


def latent_summary(families: list[dict], seed: int = SEED) -> dict[str, Any]:
    groups = (
        ["all", "designed"] + [f"regime_{n}" for n in REGIMES] + [f"designed_{n}" for n in REGIMES]
    )
    summary: dict[str, Any] = {
        "boundary_A": {
            name: bootstrap(families, mean_of(("boundary", name)), seed + i)
            for i, name in enumerate(REGIMES)
        },
        "boundary_exactly_one_minus_both": bootstrap(
            families,
            paired_contrast(("boundary", "exactly_one"), ("boundary", "both")),
            seed + 10,
        ),
        "slope": {
            group: bootstrap(families, _slope(group), seed + 20 + i)
            for i, group in enumerate(groups)
        },
        "spread_ratio": {
            group: bootstrap(families, _spread_ratio(group), seed + 40 + i)
            for i, group in enumerate(groups)
        },
    }
    challenge = [row for row in families if row["cohort"] == "challenge"]
    summary["challenge_only_slope_designed"] = bootstrap(challenge, _slope("designed"), seed + 60)
    return summary


def run(root: Path = ROOT) -> dict[str, Any]:
    result: dict[str, Any] = {"task": "REANALYSIS-001", "status": "exploratory"}
    for phase in ("develop", "confirm"):
        directory = GALLERY_ROOT / phase / "families"
        if directory.exists():
            result[f"gallery_{phase}"] = gallery_motion_split(_records(directory))
    fresh = collection_manifest(FRESH)
    families = []
    for record in _records(LATENT_ROOT / "families"):
        family_id = record["id"].split(":", 1)[1]
        family = load_family(FRESH / fresh["family_files"][family_id]["path"])
        families.append(
            latent_family(record, family["candidate_controls"], family["metadata"]["candidate_ids"])
        )
    result["latent"] = latent_summary(families)
    result["latent_families"] = len(families)
    write_json(root / "families.json", {"families": families})
    write_json(root / "report.json", result)
    manifest = {
        **result,
        "runtime": runtime_info(),
        "sources": source_hashes(
            {
                "runner": Path(__file__),
                "metrics": Path(__file__).with_name("next_metrics.py"),
                "gallery_metrics": Path(__file__).with_name("predictor_gallery_metrics.py"),
            }
        ),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "reanalysis001-runtime.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    print(json.dumps(run(), indent=2, default=str)[:8000])


if __name__ == "__main__":
    main()
