# MEAS-001 runbook — abstaining joint-pose inversion

The protocol is in MEAS-001_PLAN.md (D032), as refined by D033. The code is in
`src/jepa_action_effects/{meas001,meas001_metrics,pose_search,pose_render}.py`.

Every stage below charges one four-hour ledger at `outputs/meas001-budget/ledger.json`,
including failed and interrupted runs. Stages resume: completed per-query traces are reused
after their hashes are checked. Outputs go to `outputs/meas001/` (ignored); tracked
evidence goes to `docs/manifests/`.

## Pass 1 — engineering (D033; done once, before review)

```bash
.venv/bin/python -m jepa_action_effects.meas001 select
```

Metadata only. Seals the 48-family membership, query list and roles. Nothing is encoded.

```bash
.venv/bin/python -m jepa_action_effects.meas001 engineer-select
```

Declares the GALLERY-001 engineering families and verifies disjointness. Seals
`engineering/spec.json` before any encode.

```bash
.venv/bin/python -m jepa_action_effects.meas001 engineer
```

Charged, with a 60-minute sub-cap. Phase A chooses among three predeclared settings;
phase B validates the choice using phase-A τ.

```bash
.venv/bin/python -m jepa_action_effects.meas001 propose
```

Writes `config-proposed.json` (`freeze_ready` or `not_ready`) and
`docs/manifests/meas001-engineering.json`.

**Stop here for review.** Pass 1 must not encode or invert the 48 families, compute τ, or
read saved predictor latents.

## Pass 2 — frozen protocol (after review)

```bash
.venv/bin/python -m jepa_action_effects.meas001 freeze --proposal-sha256 <reviewed hash>
```

Seals `config.json`. It refuses if the proposal, its status or any source file changed.

```bash
.venv/bin/python -m jepa_action_effects.meas001 benchmark
```

Inverts the first calibration family per scene. Stops if 1.5× the projected remaining
workload exceeds the budget. It never triggers tuning.

```bash
.venv/bin/python -m jepa_action_effects.meas001 calibrate
```

Calibration families only, with no labels. Seals ε and τ. A query without a finite
candidate fails calibration, and assessment stays closed.

```bash
.venv/bin/python -m jepa_action_effects.meas001 controls
```

Runs the 24 synthetic targets (6 genuine poses, 12 mixtures, 6 perturbed latents) plus the
analytic known-failure checks.

```bash
.venv/bin/python -m jepa_action_effects.meas001 assess
```

The 24 assessment families, once. A rerun needs a new version (D033).

```bash
.venv/bin/python -m jepa_action_effects.meas001 score
```

Validator labels, replay identity and the D032 gates. Seals `scores.json`.

```bash
.venv/bin/python -m jepa_action_effects.meas001 predictions
```

Inverts the 16 saved frozen-predictor latents only if every D032 §7 cell passed; otherwise
the check is skipped.

```bash
.venv/bin/python -m jepa_action_effects.meas001 report
```

Writes `docs/manifests/meas001-runtime.json`. It writes nothing if integrity failed.

If calibration shows that settings must change, stop that version. Its calibration
families become development data; if assessment has opened, its families are replaced
too (D033).

## Checks

```bash
make check && make test
```

Search behaviour has analytic fixtures (`tests/test_pose_search.py`). Geometry and
restore have simulator fixtures (`tests/test_pose_render.py`). Runner guards are covered
by `tests/test_meas001.py`, and gates by `tests/test_meas001_metrics.py`.

## Correction and version 2 (2026-10-01, MEAS-FIX-001, D037)

- **Ledger coverage.** The statement above that every stage charges the ledger was
  incorrect at `8ecd208`, for two reasons:
  - `score` performed replay and renderer checks outside a budget session;
  - implementation profiling was uncharged.

  `score` now runs inside a charged session. `account-correction` adds one disclosed,
  conservative 120 s session for the earlier uncharged profiling and integrity work. No
  cap was reset.
- **Version 2 commands** (the v1 artifacts are untouched):
  ```bash
  .venv/bin/python -m jepa_action_effects.meas001 account-correction
  .venv/bin/python -m jepa_action_effects.meas001 engineer-select-v2
  .venv/bin/python -m jepa_action_effects.meas001 engineer-v2
  .venv/bin/python -m jepa_action_effects.meas001 propose-v2
  ```
  - Traces go to `outputs/meas001/engineering-v2/`, and the proposal to
    `outputs/meas001/engineering-v2/proposal.json`.
  - The manifest is `docs/manifests/meas001-engineering-v2.json`.
  - A freeze would use `freeze` on that proposal after review. It is not automatic.
