"""Export the saved fine-tuning curves of the twelve TRAIN-001/TRAIN-002 runs (read-only).

    python -m jepa_action_effects.training_curves     # writes docs/manifests/training-curves.json

Reads each run's sealed `run.json` from `outputs/train001/runs` (balanced,
archive_weighted) and `outputs/train002/runs` (dose_omitted, archive_only), checks its
SHA-256 against the run hash in the TRAIN-002 model registry, and writes the curves to a
tracked manifest so `reporting` can plot them on a fresh clone. Training loss is the
arm's own weighted objective on its 64-rollout batch (logged every 10 updates), so its
level is not comparable across arms with different data or weights. Validation is
equal-weight terminal two-step latent MSE on the 120 TRAIN-001 validation families
(every 250 updates). No model is run and no sealed file is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jepa_action_effects.next_common import MANIFESTS
from jepa_action_effects.pilot_collection import sha256_file

ROOTS = {
    "balanced": Path("outputs/train001/runs"),
    "archive_weighted": Path("outputs/train001/runs"),
    "dose_omitted": Path("outputs/train002/runs"),
    "archive_only": Path("outputs/train002/runs"),
}
SEEDS = (20261111, 20261112, 20261113)
OUT = MANIFESTS / "training-curves.json"


class RunMismatch(ValueError):
    """A run record does not match its registered hash."""


def collect(
    registry: dict[str, Any], roots: dict[str, Path] = ROOTS, seeds=SEEDS
) -> dict[str, Any]:
    runs = {}
    for arm, root in roots.items():
        for seed in seeds:
            run_id = f"{arm}-{seed}"
            path = root / run_id / "run.json"
            expected = registry[run_id]["run_sha256"]
            if sha256_file(path) != expected:
                raise RunMismatch(f"{path} does not match registry hash {expected[:12]}")
            record = json.loads(path.read_text())
            runs[run_id] = {
                "arm": arm,
                "seed": seed,
                "run_sha256": expected,
                "updates": record["spec"]["updates"],
                "learning_rate": record["spec"]["learning_rate"],
                "train_curve": [[int(u), float(loss)] for u, loss, _ in record["curve"]],
                "grad_norm": [[int(u), float(norm)] for u, _, norm in record["curve"]],
                "validation_curve": [[int(u), float(v)] for u, v in record["validation_curve"]],
            }
    return {
        "task": "TRAIN-001/TRAIN-002 training curves (REPORT-002)",
        "note": __doc__.split("\n\n")[2].replace("\n", " "),
        "runs": runs,
    }


def build(out: Path = OUT) -> dict[str, Any]:
    registry = json.loads((MANIFESTS / "train002-runtime.json").read_text())["registry"]
    payload = collect(registry)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return {"runs": len(payload["runs"]), "path": str(out)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    print(json.dumps(build()))


if __name__ == "__main__":
    main()
