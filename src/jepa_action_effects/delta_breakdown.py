"""Report DELTA-001 tuning contact regimes separately for each cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jepa_action_effects.delta_probe import DirectRidge, measurement_gate
from jepa_action_effects.delta_readout import (
    ROOT,
    ExperimentBudget,
    _audit_required,
    _baseline_predict,
    _load_data,
    _model_context,
    _seal,
    _sealed,
    _selected_predictors,
    _sources,
)
from jepa_action_effects.pilot_collection import canonical_json, sha256_bytes, sha256_file


def main() -> None:
    root = ROOT
    audited = _audit_required(root)
    fit = _sealed(root / "fit-report.json")
    signature = sha256_bytes(canonical_json(_sources()))
    budget = ExperimentBudget(root, signature)
    budget.begin("cohort_regime_report")
    try:
        model, pilot, fresh, parts, fingerprint = _model_context(audited)
        tuning = _load_data(parts["tuning"], pilot, fresh, model, fingerprint)
        direct_path = root / "models" / "direct-ridge.npz"
        if sha256_file(direct_path) != fit["cells"]["direct_ridge"]["model_sha256"]:
            raise ValueError("Direct ridge model changed")
        with np.load(direct_path, allow_pickle=False) as archive:
            direct = DirectRidge(archive["scale"], archive["coef"], float(archive["alpha"]))
        predictors = {
            "zero": lambda start, future: np.zeros((6, 2)),
            "existing_ridge_difference": _baseline_predict,
            "direct_ridge": direct.predict,
        }
        for seed in (20260931, 20260932, 20260933):
            cell = fit["cells"][f"nonlinear_{seed}"]
            predictor = _selected_predictors({"method": "nonlinear", "models": [cell]})[0]
            predictors[f"nonlinear_{seed}"] = predictor
        result = {
            "fit_report_sha256": sha256_file(root / "fit-report.json"),
            "report_code_sha256": sha256_file(Path(__file__)),
            "family_partition": "same 60 tuning families; no confirmation family loaded",
            "models": {},
        }
        for name, predictor in predictors.items():
            result["models"][name] = {}
            for cohort in ("challenge", "ordinary"):
                subset = {
                    family_id: value
                    for family_id, value in tuning.items()
                    if family_id.split(":", 1)[1].startswith(cohort)
                }
                gate = measurement_gate(subset, predictor)
                result["models"][name][cohort] = {
                    key: gate["required"][key]
                    for key in (
                        "endpoint_overall",
                        "endpoint_neither",
                        "endpoint_exactly_one",
                        "endpoint_both",
                        "effect_overall",
                        "effect_neither",
                        "effect_exactly_one",
                        "effect_both",
                        "effect_meaningful",
                    )
                }
        _seal(root / "cohort-regime-breakdown.json", result)
    except Exception:
        budget.finish("error")
        raise
    budget.finish("complete")
    print(json.dumps({"status": "complete", "families": len(tuning)}, sort_keys=True))


if __name__ == "__main__":
    main()
