"""Derived SUPPORT-001 plotting summary (REPORT-FIX-001); not part of the sealed test.

    python -m jepa_action_effects.support001_derived

Runs after `support001 confirm` and reuses its saved predicted latents; no model is run.
It writes the account-implied projection slope per scene and dose, plus the archive
command-magnitude tail of the support map. Kept outside `support001.py` so the
sealed source hashes stay unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import sealed, write_json
from jepa_action_effects.next_common import MANIFESTS
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.support001 import (
    DOSES,
    ROOT,
    _covariates,
    _family,
    _load_map,
    _real_decodes,
    _real_latents,
    decoders,
    family_units,
    verify_seal,
)
from jepa_action_effects.support001_metrics import predict, slope
from jepa_action_effects.train001_protocol import SCENES, magnitude_label

MANIFEST = MANIFESTS / "support001-account-slopes.json"
READOUT = "mlp-20261005"
PLOTTED = ("A", "B", "C")


def account_slopes(fits: dict, units: list[dict]) -> dict[str, float | None]:
    """Projection slope Σp̂·t / Σ‖t‖² of each account's prediction, pooled over units."""
    t = np.concatenate([u["t"] for u in units])
    m = np.concatenate([u["m"] for u in units])
    ell = np.concatenate([u["log_support"] for u in units])
    magnitude = np.concatenate([u["magnitude"] for u in units])
    return {
        account: slope(predict(account, fits[account]["params"], t, m, ell, magnitude), t)
        for account in PLOTTED
    }


def command_tail(a: np.ndarray, threshold: float = 0.60) -> dict[str, Any]:
    magnitude = np.linalg.norm(np.asarray(a, float), axis=1)
    return {
        "windows": int(len(magnitude)),
        "max": float(magnitude.max()),
        "above_0.60": int(np.sum(magnitude > threshold)),
        "above_0.80": int(np.sum(magnitude > 0.80)),
    }


def run(root: Path = ROOT, out: Path = MANIFEST) -> dict[str, Any]:
    verify_seal(root)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("Run `support001 confirm` first")
    fits = sealed(root / "fit.json")["fits"]
    collection, manifest = sealed(root / "collection.json"), sealed(root / "cache.json")
    support, map_sha = _load_map(root)
    block, agent, digests = decoders()
    labels = [magnitude_label(d) for d in DOSES]
    cells: dict[str, list[list[dict]]] = {s: [[] for _ in DOSES] for s in SCENES}
    for family_id, entry in sorted(collection["families"].items()):
        family = _family(root, entry)
        with np.load(root / "predicted" / f"{family_id}.npz") as arrays:
            predicted = arrays["predicted"].astype(np.float64)
        real = _real_decodes(_real_latents(root, manifest["records"][family_id]), block, agent)
        units = _covariates(
            support, family_units(family, block[READOUT](predicted[:, 1]), real[READOUT], labels)
        )
        for i in range(len(DOSES)):
            cells[entry["scene"]][i].append(
                {key: units[key][i : i + 1] for key in ("t", "m", "log_support", "magnitude")}
            )
    slopes = {
        scene: {labels[i]: account_slopes(fits, rows) for i, rows in enumerate(cells[scene])}
        for scene in SCENES
    }
    with np.load(root / "map.npz") as arrays:
        tail = command_tail(arrays["a"])
    payload = {
        "task": "REPORT-FIX-001: derived SUPPORT-001 plotting summary",
        "note": "Account-implied projection slopes for the figure; not part of the sealed test.",
        "readout": READOUT,
        "readout_digest": digests[READOUT],
        "account_slopes": slopes,
        "command_tail": tail,
        "map_sha256": map_sha,
        "seal_sha256": sha256_file(root / "seal.json"),
        "summary_sha256": sha256_file(summary_path),
        "fit_sha256": sha256_file(root / "fit.json"),
    }
    write_json(out, payload)
    return payload


def main() -> None:
    print(json.dumps(run(), indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
