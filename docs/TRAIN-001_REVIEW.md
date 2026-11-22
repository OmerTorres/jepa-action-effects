# TRAIN-001 — implementation and result review

**Date:** 2026-09-30. **Base:** HEAD `6e091a6`, with the implementing agent's
uncommitted TRAIN-001 code, tests, documentation and artifacts present.

The completed numerical analysis reproduces. The training implementation preserves
the intended two-step autoregressive inputs and frozen visual/readout components,
and the pooled verdict respects the irreversible development measurement failure.
There are reporting corrections and two validator/diagnostic gaps to address.
This review did not retrain any model or modify experiment sources or sealed data.

## Findings

### [P2] Physical conclusions exceed the measurement and statistical verdicts

**Location:** `docs/TRAIN-001_RESULT.md:188`, including the following interpretation
paragraphs; physical figure labels in `train001_report.py`.

“Magnitude weighting did not matter physically” is stronger than the evidence.
The pooled physical hypothesis was not assessable, and the measured contrast
(-0.0915, interval [-0.5293, 0.3763]) was not tested for equivalence. An interval
spanning zero does not establish that weighting has no physical effect. Likewise,
the benefit of exposure to high-magnitude constant pushes was not isolated from
other consequences of additional fine-tuning by an archive-only or dose-omission
training control.

Use: **“No pooled balanced-over-archive physical benefit was established; the
descriptive decoded contrast was near zero. Both fine-tuned conditions reduced
decoded errors on the sampled constant-push distribution.”** Present the cause of
that adaptation as a hypothesis. Avoid claiming that 2% high-dose weight was
necessary/sufficient without an appropriate dose-omission comparison. The figures
should visibly mark physical contrasts as descriptive/exploratory because the
pooled hypothesis is ineligible; a detached plot currently reads as a primary
confirmatory result without that qualification.

### [P2] The sensitivity flag cannot detect disagreement after eligibility fails

**Location:** `src/jepa_action_effects/train001_metrics.py:484–488`.

`measurement_dependent` compares only pooled verdict strings. Every readout gets
the same primary-readout development eligibility. When that is false, all strings
are `not_assessable_under_measurement_protocol` regardless of the measured effects.
The flag is therefore necessarily false on this study, not evidence of robustness.

An in-memory counterexample changed only the alternative readout's balanced-model
decoded endpoints. Its balanced-minus-archive interval became
[1397.60, 1400.92], yet `measurement_dependent` remained false. No study record was
changed. This check used 20 bootstrap draws solely to demonstrate the logic defect.

Report protocol eligibility separately from sensitivity of contrasts, interval
signs and ordinary-case outcomes. In the actual artifacts, all readouts agree that
ordinary non-inferiority is unestablished, but the observed degradation against zero
is not uniformly supported: the ordinary balanced-minus-frozen interval for
`mlp-20261004` is [-0.1487, 1.8223], while the selected readout's is
[0.1108, 1.9925]. Replace the blanket “not measurement-dependent” sentence with the
specific comparisons that do and do not agree.

### [P2] Sealing does not check training-data lineage against the sealed inputs

**Location:** `src/jepa_action_effects/train001_provenance.py:168–181`.

The registry validates run/checkpoint bytes, seed, learning rate and eligibility,
but omits the run's recorded `collection_sha256` and `cache_sha256`. A checkpoint
trained on another development collection/cache can therefore be admitted into a
new seal even though the current development artifacts themselves pass validation.

Reproduction: in a temporary instance of the existing provenance test fixture,
set a training run's collection/cache hashes to foreign values, save its metadata,
and build the first confirmation seal. Entry validation returns `valid=True` with
no failed checks. This models admitting a stale run at seal creation, not changing
an already frozen production seal.

Before accepting checkpoints, compare those two run hashes with the sealed
development collection/cache hashes, and validate the run's loss weights and
optimization specification against the intended condition/protocol. Add fixtures
for stale training input hashes. **All six actual runs' collection/cache hashes
match**, and their saved weights agree with their conditions; this finding does
not show contamination in the completed results.

### [P2] The hand-written secondary interval differs from the saved result

**Location:** `docs/TRAIN-001_RESULT.md:113`; `docs/DECISIONS.md:727`; the matching
LOGBOOK entry.

The stated latent balanced-minus-archive result is -0.0070 [-0.0127, -0.0014].
Reaggregation of the saved confirmation records exactly reproduces the sealed
summary, whose result is:

**-0.0068376941 [-0.0127569325, -0.0011002090]**, or
**-0.00684 [-0.01276, -0.00110]** with consistent rounding.

The hand-written bounds are not simply a different rounding of this interval.
The sign and qualitative conclusion are unchanged. Correct the current result
document; append corrections to historical decision/logbook evidence. Generate
report numbers from the manifest to prevent further transcription drift.

## Verification and evidence

- `make check && make test`: passed; **233 tests, no skips**, 19 existing Gymnasium
  warnings, pytest duration 13.73 seconds.
- Read-only production `train001_provenance.validate(..., COMPLETE, ...)`: **51
  checks passed**, covering all 2,880 confirmation records.
- Re-ran `train001_metrics.summarize` from the saved family/window records for all
  seven predictors and all three readouts. The entire reconstructed confirmation
  summary **equals the saved summary exactly**.
- Recomputed inference for `archive-conf-000`, `archive-conf-119`,
  `challenge-conf-000`, `challenge-conf-119`, `ordinary-conf-000` and
  `ordinary-conf-119`, using the frozen checkpoint and all six trained checkpoints.
  Across 42 model/unit combinations, maximum decoded-coordinate difference was
  **6.062e-5 workspace units**, and maximum latent-MSE difference **5.216e-8**.
  Differences are tiny and consistent with changed inference batching; this was a
  subset verification, not a full independent reimplementation or retraining.
- Checked all six saved training collection/cache hashes against the production
  seal; all match. Saved archive/balanced weights are correct.
- Inspected all three generated figures. Layout and labels are legible; add the
  physical-ineligibility qualification directly to the physical plots/captions.
- Reviewed the differentiable rollout, training boundary, data partitions and
  implemented gate policy. No future-target input leak was found in the inspected
  prediction path.
- The two counterexamples above used temporary mock artifacts or in-memory copies.
  No production artifacts were mutated.

## Practical interpretation and next action

The defensible headline is a **small secondary latent-MSE advantage for balanced
weighting**, with **ordinary-case non-inferiority not established**. The physical
question stays unassessable under the frozen policy; physical contrasts remain
descriptive/exploratory. The study now includes substantial predictor training,
but instructor acceptance and publication scope remain separate judgments.

Correct reporting before REPORT-001 authoring. Preserve the frozen numerical
artifacts and original source snapshots. Any code fix touching seal-hashed modules
needs a versioned follow-up or explicit preserved-source reanalysis; do not rewrite
the completed seal to make checks pass. The findings alone do not justify retraining
the six models. Any revised confirmatory claim still requires a new protocol and
fresh families.

Additional disclosure: D028 excludes the real-data smoke run from the production
budget, whereas the original cap covered experimental execution. Production is far
below 24 hours, so this does not show a budget overrun; report smoke time separately
and avoid describing the 1.06-hour production ledger as all experimental work.


## Follow-up verification — TRAIN-REVIEW-003, 2026-09-30

Reviewed commit `3932604` against `68d9fbc`. **All four original findings are
addressed for the completed study.** Physical claims and both physical figures now
carry the appropriate descriptive qualification; the contrast-level sensitivity
analysis identifies the ordinary balanced-minus-frozen disagreement; all six runs
pass the new lineage checks; and the secondary latent interval is corrected.
D029 preserves the historical record and discloses smoke time separately.

### [P2] Stop report regeneration when follow-up validation fails

**Location:** `src/jepa_action_effects/train001_followup.py:211–215`, and the CLI
at lines 240–251.

`run()` obtains the combined validator result but does not enforce `valid` before
calling `regenerate_presentation` and writing the follow-up manifest. `main()` also
returns normally when that result is false. A hash/lineage failure that returns a
validation report therefore still permits replacement of the tracked figures and
table draft using rejected inputs, and a shell pipeline sees a successful command.
The manifest records the failure, but the generated presentation has no such guard.

Reproduction used `unittest.mock.patch` to return a failed collection-lineage check
and intercept both presentation regeneration and manifest writes. Calling `main([])`
returned normally, called each writer once, and printed `valid_with_lineage: false`.
This was a control-flow fixture, not an actual production validation failure; no
study or report output was changed by the reproduction.

Require a successful combined validation before any presentation/manifest mutation,
raise or exit nonzero on failure, and test that failed validation leaves existing
report artifacts unchanged. Keep this fix in the unsealed follow-up module.

### Verification evidence and disposition

- `make check && make test`: passed; **242 tests, no skips**, 19 existing Gymnasium
  warnings; pytest 13.09 seconds.
- Read-only `validate_with_lineage(..., COMPLETE, ...)`: **58/58 checks passed**.
- Recomputed `contrast_sensitivity(summary)` and `exact_numbers(summary)` exactly
  match the tracked follow-up manifest. The sole interval-sign disagreement is
  `ordinary:balanced_minus_frozen`.
- Production seal SHA-256:
  `e2cceaca9d7c451c78d70091c5acd37de130775aa8fe3ca8ff6e7427049a27c6`.
  Summary SHA-256:
  `01c1ad9e371c7844b1276d45be25ed22aa5ad294d3aeb477e91abbf8e09a5bef`.
  Both match the original manifest at `68d9fbc` and the new follow-up manifest.
- Original `train001-runtime.json` remains byte-identical to `68d9fbc`. Current
  follow-up/report source hashes match their recorded hashes; the combined validator
  verifies the sealed sources. Both relabelled figures were visually inspected.

The remaining runner bug does not invalidate the present results, which passed
validation. REPORT-001 authoring can proceed from the corrected evidence. Fix the
failure path before reusing the follow-up command. No retraining or new confirmation
is warranted by this review; this verification did not repeat the prior full summary
reaggregation or inference subset checks.
