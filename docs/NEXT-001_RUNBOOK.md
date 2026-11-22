# NEXT-001 runbook: Mac execution of DIST-001 → REANALYSIS-001 → READOUT-002 → PHYS-002

Run everything from the repository root on the Mac that already holds the checkpoint,
the extracted archive and the earlier `outputs/`. The code was written and tested
(without the model) in a cloud session. Each command is resumable: if one is
interrupted, rerun the same command.

## 0. Prerequisites and setup (about 5 minutes)

These paths must exist locally. They are all git-ignored:

- `checkpoints/lewm-pusht/{config.json,weights.pt}` (the pinned revision; hashes are
  checked on load)
- `outputs/model001/action-stats.json`
- `data/quentinll-lewm-pusht/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5`
- `outputs/pilot001/`, `outputs/readout001/`, `outputs/latent001/` and
  `outputs/predictor_gallery001/`

```sh
git fetch origin claude/youthful-clarke-akm28u
git checkout claude/youthful-clarke-akm28u
make setup && make check && make test
```

`make test` should report 0 failures. With the artifacts present, the three artifact
tests run instead of being skipped.

Every command below uses the same shell function:

```sh
run_stage() {
  env PYTHONWARNINGS=ignore UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python \
    uv run --locked --no-sync python -m "jepa_action_effects.$1" "${@:2}"
}
```

Correction (2026-09-29, REVIEW-BRANCH-002): the original version stored the prefix in
a scalar (`export RUN="env … -m jepa_action_effects"`) and ran `$RUN.<module> …`.
Default zsh does not split a scalar into words, so every command failed with exit 127.
The 2026-09-28 Mac run used an equivalent wrapper script instead.

## 1. DIST-001: distribution checks (about 5–10 minutes)

```sh
run_stage distribution_audit keys
run_stage distribution_audit actions
run_stage distribution_audit identity
run_stage distribution_audit gap
run_stage distribution_audit report
```

- `keys` records the archive's dataset names. If `state_key` or `pixels_key` is
  `null`, `identity` records itself as unavailable. `gap` still runs if pixels
  exist.
- **Stop rule.** If `outputs/dist001/report.json` says
  `"PAUSE Phase 3: rendering mismatch"`, stop after step 2 and report back. PHYS-002
  `audit` also refuses to run in that case.
- **In-support variant rule.** This is fixed now and uses archive statistics only.
  In `outputs/dist001/actions.json`, if
  `by_cohort.challenge.study_fraction_above_archive_p99_magnitude > 0.5`, set
  `PUSH` to the archive's 95th-percentile block magnitude
  (`all_study.archive_magnitude_percentiles["95"]`), rounded to two decimals.
  Otherwise leave `PUSH` unset.

## 2. REANALYSIS-001: exploratory, saved outputs only (about 1 minute)

```sh
run_stage reanalysis run
```

This reads sealed PREDICTOR-GALLERY and LATENT records and never runs the model. Its
numbers are exploratory because those families are already opened.

## 3. READOUT-002: readout on 60k frames (about 20–30 minutes)

```sh
run_stage readout_scale collect
run_stage readout_scale fit
run_stage readout_scale report
```

- `collect` writes 60 resumable shards, `sensitivity.npz` and `study-check.npz`.
- `fit` seals `outputs/readout002/selection.json`. That is the lowest validation
  block error among ridge and three MLP seeds, and it is frozen for PHYS-002.
- Look at `learning_curve_ridge` and `study_check` in the report. A large gap
  between validation error and study-check error means the random-pose readout
  transfers poorly to real study frames. Note it when reporting back.

## 4. PHYS-002: predictor against the readout floor (about 10–15 minutes)

```sh
run_stage phys002 audit                      # or: run_stage phys002 audit --in-support-push $PUSH
run_stage phys002 collect                    # 150 fresh reserved families (+60 variant), simulator only
run_stage phys002 develop                    # 300 development families
run_stage phys002 seal
run_stage phys002 confirm                    # opens the reserved families ONCE
run_stage phys002 report
```

- Do not edit any file under `src/` between `audit` and `confirm`. `seal` and
  `confirm` refuse to run if a source hash changes.
- If a stage fails for a code reason before `confirm`, stop and report back. Do not
  patch it locally. A fix means a new protocol version.
- `confirm` writes `confirmation-opened.json` before it scores anything. Never
  delete that file.

## 5. Commit the small results

```sh
make test && git diff --check
git add docs/manifests/dist001-runtime.json docs/manifests/reanalysis001-runtime.json \
        docs/manifests/readout002-runtime.json docs/manifests/phys002-runtime.json \
        docs/PHYS-002_RESULT_DRAFT.md
git commit -m "Record NEXT-001 Mac run results"
git push origin claude/youthful-clarke-akm28u
```

Keep `outputs/`, `data/` and `checkpoints/` out of Git. A follow-up session will
interpret the results and update STUDY_SYNTHESIS, TASKS, LOGBOOK and DECISIONS.
