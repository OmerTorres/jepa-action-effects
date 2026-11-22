# NEXT-001: plan after the external review

Status: proposed plan, 2026-09-28. Nothing has been implemented or run. Every stage
that touches the checkpoint, the archive or `outputs/` runs on Omer's Mac. Cloud
sessions write the code and tests.

## Context

`EXTERNAL_REVIEW_2026-09-28.md` found that the study's central question was never
answered. H1/H2 ask whether predicted block effects are wrong around contact, by
more than the readout can resolve. Two things blocked it:

- An absolute sub-pixel gate (1.28 units, which is 0.56 px) stopped every physical
  measurement.
- The readouts saw only about 240 block poses, and their learning curve was still
  falling.

The strongest remaining hypothesis is **shrinkage**: the predictor under-predicts
contact-induced block motion. The evidence is that predictions have 64% of the real
spread, and that the predicted block placement is no better than persistence.
Two confounds are still unmeasured: action support and simulator lineage.

**Goal for this round (through Phase 3):** answer H1/H2 with a readout that is
trained properly, a floor-relative criterion fixed in advance, and fresh reserved
families. ERASURE-001 is deferred.

**Execution split.** The Hugging Face connector can browse the Hub but cannot put
the 72 MB weights into the cloud container. The container's own download returns
HTTP 403. So all model and archive stages run on the Mac.

Cloud sessions write:

- the code;
- tests that use mocks instead of the model;
- `docs/NEXT-001_RUNBOOK.md`, which lists the exact Mac commands, the expected run
  times, and which small result files to commit. Large artifacts stay in the
  ignored `outputs/`.

Each stage follows the repository's existing pattern:

- a staged command-line interface (`python -m jepa_action_effects.<module> <stage>`);
- config, source hashes and family IDs sealed before scoring;
- resumable atomic writes;
- a tracked `docs/manifests/*-runtime.json`.

## Phase 0: repository hygiene (cloud, no model)

- Make two tests skip when `outputs/` is absent, following
  `tests/test_model_inference.py`. They currently fail:
  - `tests/test_gallery_audit.py::test_real_replay_pose_order_geometry_and_restoration`
  - `tests/test_predictor_gallery.py::test_audit_excludes_confirmation_payloads_and_rejects_changed_source`
- Record that `concept_erasure.py` and its tests are committed but have never run.
  ERASURE-001 is deferred (D021).

## Phase 1: cheap checks and reanalyses (Mac; code written in the cloud)

### 1a. DIST-001: distribution checks (new module `distribution_audit.py`)

1. **Action support.** Load the archive's `action` column, reusing
   `action_stats.py`.
   - Report per-coordinate percentiles, the magnitude of each 5-control block, and
     how constant each block is (largest difference between any two controls in
     it).
   - Compare these with the challenge candidates (constant 0.6 push plus a lateral
     offset) and the ordinary candidates (uniform in ±0.35) from
     `pilot_collection.propose_*`.
2. **Simulator identity.**
   - First list the HDF5 keys to see which state or proprioception fields exist.
   - For 20 preselected episodes, reset the pinned simulator to each recorded start
     state and replay the recorded actions.
   - Report the per-frame RGB difference and the projected-CLS distance to the
     archive frames.
   - If no state is recorded, fall back to comparing the static scene: background,
     goal overlay, agent radius and colours.
3. **Error gap.** On 200 archive windows, compare the predictor/persistence latent
   MSE ratio on the archive frames with the same windows replayed in the simulator.

**Stop rule:** if the replayed frames differ materially from the archive, report it
and pause Phase 3 so the question can be re-scoped.

### 1b. REANALYSIS-001: exploratory, reusing the saved `outputs/`

New module `reanalysis.py`, reporting only. It does not change any historical
artifact. These families are already opened, so everything here is labelled
exploratory.

- **PREDICTOR-GALLERY:** split endpoints by whether the block moved more than 5.12
  units. Report matched, persistence and wrong-action margins for each group, per
  cohort and per scale.
- **LATENT-001:** recompute the contact contrast using only the designed boundary
  pairs, meaning action distance ≤ 0.2 with `hold` excluded.
- **Latent shrinkage slope:** regress predicted pair differences `Δẑ_ij` on real
  pair differences `Δz_ij`, by contact regime.

## Phase 2: READOUT-002, a properly trained readout (Mac)

- New module `readout_scale.py`, with stages `collect | encode | fit | report`. It
  reuses:
  - `simulator_replay.InstrumentedPushT`;
  - `gallery_render.set_block_pose` (sets angle before position) and
    `gallery_render.geometry`;
  - `interface_audit.preprocess_pixels`;
  - `model_inference.build_frozen_model` and `encode_real`;
  - `latent_cache.LatentCache`;
  - weighting in the style of `pilot_readouts.fit_linear`.
- **Data.**
  - 60k single frames from seeded resets.
  - Block x, y and angle drawn uniformly over the valid workspace; agent position
    random but geometrically valid.
  - A separate 10k validation set on disjoint seeds.
  - Encoding takes about 15 CPU-minutes.
- **Probes.**
  - Fitted on projected CLS; unprojected CLS is a reference only.
  - Ridge regression, with alpha chosen on validation.
  - A 2×256 MLP trained with mini-batch Adam and early stopping until it
    converges, with 3 seeds.
- **Targets:** block x and y, block angle as sin/cos, and agent x and y.
- **Report:**
  - R², mean and 95th-percentile error;
  - a learning curve from 1k to 50k frames;
  - the smallest block shift, between 0.5 and 5 units, that the embedding reliably
    separates.
- This phase is descriptive, with no pass/fail gate. The probe is chosen (ridge or
  MLP, and which seed) by validation error alone, then frozen for Phase 3.

## Phase 3: PHYS-002, predictor against the readout floor, and shrinkage (Mac)

- **Development set:** the 300 READOUT/pilot development families already on the
  Mac. The 60 opened reserved families are excluded.
- **Reserved set:** 150 new families (100 challenge, 50 ordinary). Collect them with
  `pilot_collection` under a new master seed, record them before any scoring, and
  open them once.
- **Action-shift variant.** If DIST-001 finds action shift, add a second challenge
  set whose push size equals the archive's 90th percentile. Report it separately.
- **Measurement.**
  - Apply the frozen Phase 2 readout to the real future frames and to the predicted
    futures of the same branches, at the two-step horizon.
  - Decode displacements relative to `readout(z_t)`. This cancels the readout's
    constant bias and avoids comparing against oracle persistence.
- **Primary quantity: `D = err_pred − err_real`.** This is the family-level paired
  difference in block-displacement error, on the designed boundary pairs.
  - Compute it separately for pairs where contact differs and pairs where both
    candidates make contact.
  - The difference between those two groups is the H1 contrast.
- **Secondary quantities:**
  - the displacement slope (predicted against true, by total least squares),
    overall and by contact and by moved/static;
  - endpoint error;
  - regret against oracle, random and persistence-decoded choices, on the
    `PilotConfig` goal grid (H3).

  These reuse `evaluation.evaluate_family` and the bootstrap in
  `predictor_gallery_metrics.aggregate`, which resamples whole families and keeps
  the cohort mix (2,000 resamples).
- **Criteria.** These are sealed before the reserved families are opened, and are
  proposed in D021.
  - **Physical failure** is claimed only if both hold:
    - the lower bound of the 95% CI of `D` is above 0;
    - the real-frame mean displacement error is below 25% of the median meaningful
      effect.
  - **Shrinkage** is claimed if the upper bound of the slope's CI is below 1.
  - **H1** is supported if the CI of the contact-differs minus both-contact contrast
    in `D` lies entirely above 0.
  - If the real-frame floor condition fails, the result is inconclusive.
- **Integrity checks.**
  - Duplicate and reversed-order inference on 12 preselected families.
  - Check the seal hash before the reserved stage runs.

## Handoff and report-back

- `docs/NEXT-001_RUNBOOK.md` lists the Mac commands in order:
  1. `make setup`, `make check`, `make test`;
  2. the DIST-001 stages;
  3. the REANALYSIS-001 stages;
  4. the READOUT-002 stages;
  5. the PHYS-002 stages.

  It also gives the expected run times and the files to commit: the manifests plus
  short `docs/*_RESULT.md` drafts written automatically by each `report` stage.
- After the Mac results are pushed, a follow-up session interprets them and updates
  STUDY_SYNTHESIS, TASKS, LOGBOOK and DECISIONS.
- Out of scope for this round: the Jacobian decoder, planning-cost regret and the
  course report (Phases 4–5 in the review).

## Verification

**Cloud:** `make check` and `make test` pass with 0 failures on a fresh clone. New
tests that don't need the model:

- the slope estimator recovers 0.6 on synthetic shrunk data;
- the floor-relative criterion behaves correctly on oracle, noisy-readout and
  shrunk-predictor mocks;
- a readout fitted on a synthetic linear embedding recovers the targets;
- action-support percentiles are correct on a toy action array;
- the reanalysis splitters work on hand-built endpoint records;
- stage sealing and resume, following the patterns in `test_predictor_gallery.py`.

**Mac:** every stage writes its manifest, and the integrity checks above pass. The
runbook ends with `make test` and `git diff --check` before pushing.
