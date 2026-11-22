"""TRAIN-001 frozen protocol constants (docs/TRAIN-001_PLAN.md, D028).

The sealed `config.json` is generated from `Protocol.payload()`. The validator
regenerates it from this code and requires equality, so an edited JSON file with a
regenerated sidecar cannot redefine the protocol.

`Protocol.smoke()` is a tiny implementation-testing variant that writes to its own
root and ledger. It never produces study evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCENES = ("challenge", "archive")
MAGNITUDES = (0.15, 0.30, 0.45, 0.60)
HIGH_MAGNITUDES = (0.45, 0.60)
MOVE_REQUIRED = (0.30, 0.60)
CONDITIONS = ("archive_weighted", "balanced")
READOUTS = ("mlp-20261005", "mlp-20261004", "mlp-20261006")  # primary first (READOUT-002)
TRAINABLE_MODULES = ("predictor", "action_encoder", "pred_proj")
EXPECTED_TRAINABLE_PARAMETERS = 11_740_334


def magnitude_label(magnitude: float) -> str:
    return f"m{magnitude:.2f}"


@dataclass(frozen=True)
class Protocol:
    name: str = "train001"
    schema: str = "train001-v1"
    root: str = "outputs/train001"
    budget_root: str = "outputs/train001-budget"
    collection_seed: int = 20261100
    # cohort sizes, per scene
    train_per_scene: int = 300
    validation_per_scene: int = 60
    confirmation_per_scene: int = 120
    ordinary_validation: int = 60
    ordinary_confirmation: int = 120
    pilot_train_per_scene: int = 30
    pilot_validation_per_scene: int = 30
    # archive weight reference sample
    weight_episodes: int = 1000
    weight_edges: tuple[float, float, float] = (0.225, 0.375, 0.525)
    weight_min_bin: int = 100
    weight_window: int = 10
    # admission rules
    delta: float = 5.12
    direction_min_norm: float = 0.05
    rest_limit: float = 1.0
    identity_position: float = 0.5
    identity_frame_raw_mae: float = 1.0  # raw uint8 scale, as in SHIFT-002
    identity_own_future: float = 1.0
    attempt_caps: dict[str, int] = field(
        default_factory=lambda: {"challenge": 20000, "archive": 60000, "ordinary": 5000}
    )
    # training
    training_seeds: tuple[int, ...] = (20261111, 20261112, 20261113)
    learning_rates: tuple[float, ...] = (1e-5, 3e-5, 1e-4)
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    families_per_scene_batch: int = 8
    updates: int = 2000
    checkpoint_updates: tuple[int, ...] = (500, 1000, 1500, 2000)
    resume_every: int = 50
    curve_every: int = 10
    validation_every: int = 250
    threads: int = 4
    # pilot
    overfit_families_per_scene: int = 8
    overfit_updates: int = 500
    overfit_learning_rate: float = 1e-4
    overfit_check_every: int = 50
    overfit_required_reduction: float = 0.5
    sweep_updates: int = 250
    tie_relative_tolerance: float = 1e-12
    equivalence_tolerance: float = 1e-5
    contingency: float = 1.5
    # budget
    budget_hours: float = 24.0
    shutdown_allowance_seconds: float = 300.0
    # measurement gates and statistics
    min_moved_families: int = 50
    readout_band: tuple[float, float] = (0.85, 1.15)
    floor_limit: float = 0.25
    regression_tolerance: float = 1.28
    bootstrap_draws: int = 2000
    gate_bootstrap_seed: int = 20261121
    primary_bootstrap_seed: int = 20261122
    ordinary_bootstrap_seed: int = 20261123
    permutation_seed: int = 20261124

    # ------------------------------------------------------------------ variants
    @classmethod
    def production(cls) -> Protocol:
        return cls()

    @classmethod
    def smoke(cls) -> Protocol:
        """Tiny implementation check on real data; never study evidence."""
        return cls(
            name="train001-smoke",
            root="outputs/train001-smoke",
            budget_root="outputs/train001-smoke-budget",
            collection_seed=20261199,
            train_per_scene=8,
            validation_per_scene=4,
            confirmation_per_scene=4,
            ordinary_validation=4,
            ordinary_confirmation=4,
            pilot_train_per_scene=8,
            pilot_validation_per_scene=4,
            weight_episodes=60,
            weight_min_bin=1,
            training_seeds=(20261191, 20261192, 20261193),
            updates=20,
            checkpoint_updates=(10, 20),
            resume_every=5,
            curve_every=2,
            validation_every=10,
            overfit_updates=20,
            overfit_check_every=5,
            overfit_required_reduction=0.0,
            sweep_updates=6,
            min_moved_families=1,
            bootstrap_draws=50,
        )

    # ------------------------------------------------------------------ derived
    @property
    def run_root(self) -> Path:
        return Path(self.root)

    @property
    def budget_seconds(self) -> float:
        return self.budget_hours * 3600.0

    def target(self, phase: str) -> int:
        """Families per scene collected in one collection phase."""
        if phase == "development":
            return self.train_per_scene + self.validation_per_scene
        if phase == "confirmation":
            return self.confirmation_per_scene
        raise ValueError(phase)

    def ordinary_target(self, phase: str) -> int:
        return self.ordinary_validation if phase == "development" else self.ordinary_confirmation

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "scenes": SCENES,
                "magnitudes": MAGNITUDES,
                "high_magnitudes": HIGH_MAGNITUDES,
                "move_required": MOVE_REQUIRED,
                "conditions": CONDITIONS,
                "readouts": READOUTS,
                "trainable_modules": TRAINABLE_MODULES,
                "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
                "decision_rules": DECISION_RULES,
            }
        )
        return json.loads(json.dumps(result))

    def payload_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()


DECISION_RULES = {
    "primary": (
        "Per family: absolute terminal block-position error averaged over the 0.45 and "
        "0.60 branches, then over the three training seeds; contrasts balanced-archive, "
        "balanced-frozen, archive-frozen; scene means averaged with equal weight; 2,000 "
        "paired scene-stratified family bootstrap draws with seeds never resampled."
    ),
    "measurement_gate": (
        "Per scene x {0.45, 0.60} cell with the primary readout: >=50 observed moved "
        "families (true displacement > 5.12); real-frame projection-slope 95% CI on moved "
        "families strictly inside (0.85, 1.15); floor = mean real-frame displacement error "
        "over all accepted families / median true displacement over moved families < 0.25. "
        "Missing or degenerate statistics fail."
    ),
    "gate_policy": (
        "Development measurement-gate failure does not stop TRAIN-001 training or fresh "
        "latent evaluation. It makes the pooled physical hypothesis ineligible for "
        "confirmation; retain latent outcomes as secondary and physical subgroup "
        "comparisons as exploratory. Preserve thresholds, cohorts and readouts. Stop "
        "dependent execution only for implementation, optimization, data-integrity or "
        "budget failures. Report all gate outcomes explicitly, and do not promote a "
        "secondary result to replace the primary hypothesis."
    ),
    "verdicts": {
        "development_fail": "not_assessable_under_measurement_protocol",
        "confirmation_fail": "inconclusive_measurement_invalid_at_confirmation",
        "both_ci_below_zero": "balanced_benefit_supported",
        "required_ci_above_zero": "balanced_error_increase_supported",
        "otherwise": "physical_benefit_not_established",
    },
    "ordinary": {
        "upper_below_tolerance": "non_inferiority_within_tolerance_established",
        "lower_above_tolerance": "regression_beyond_tolerance_supported",
        "otherwise": "non_inferiority_not_established",
    },
    "secondary": "Terminal (step-2) latent MSE on the 0.45/0.60 branches, same contrasts.",
}
