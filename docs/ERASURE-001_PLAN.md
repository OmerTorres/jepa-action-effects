# ERASURE-001 — Bounded selective-erasure feasibility

Status: proposed implementation plan, 2026-09-28. No execution or frozen run yet.

## Objective and scope

Determine whether the frozen checkpoint's observed-history embeddings admit
selective removal of linearly accessible agent-position or block-position
information. Establish the validity of this intervention before testing how it
changes predictor behavior.

Use LEACE as the single erasure method. Do not run a search over erasure methods,
layers, spatial bins, architectures or data sizes. Use existing real-frame data,
the projected 192-dimensional CLS representation, Python 3.10.21, the locked CPU
runtime and frozen checkpoint. Model encoding is float32; covariance, erasure,
linear fitting and metric reductions are float64. No new collection, world-model
training, predictor-performance experiment, raw CLS analysis or main-study expansion.

The eventual mechanism question is whether agent and block information influence
action-conditioned prediction differently across contact conditions. ERASURE-001
answers only whether the proposed intervention can support that later question.

## 1. Implement the audit and seal the design

Add reusable modules under `src/jepa_action_effects/` and a staged
`python -m jepa_action_effects.concept_erasure` command with `audit`, `fit`,
`validate` and `report`. Artifacts go under ignored `outputs/erasure001/`.
Preserve historical implementations and artifacts.

Use the existing 240 READOUT fit families (160 challenge/80 ordinary). Divide the
60 READOUT tuning families into calibration and assessment sets of 30 each
(20 challenge/10 ordinary), by deterministic cohort-wise shuffle. Master seed
`20261002` provides independent partition, control, nonlinear-training, verification
and bootstrap streams. Record exact IDs before inspecting any new probe results.
The 60 formerly reserved families are excluded entirely from this experiment.

All selected families have already contributed to historical development analyses.
Assessment is held out from fitting in this run; it is **not fresh confirmation**.
All related recipes stay together. Verify upstream manifests, checkpoint/cache
fingerprints, split disjointness, frame timestamps and hashes.

Use only the three observed-history frames per family and their aligned current
agent/block coordinates. Do not use future frames or future labels as training
examples. The current READOUT cache stores history and future encodings together:
verify its existing content-addressed identity, then explicitly extract history
rows. Do not call the recovery helper that computes both projected and raw CLS.
Missing history encodings may be written only to the new artifact directory.

Deduplicate RGB-identical history frames within families. Report conflicting labels
and cross-family duplicates. A conflicting frame-to-label association blocks the
affected concept analysis pending a documented correction; do not choose the first
label. Cross-partition identical frames also block the purported held-out check
until the partition defect is resolved before fitting. Each family receives equal
total weight, divided among its unique frames.

Seal code, configuration, dependency, upstream input, checkpoint and membership
hashes before fitting. Changes require a new version, retaining previous evidence.
Preselect twelve alignment-verification families, six per cohort, across partitions.
Recheck history encodings, state-label order and frozen/evaluation-mode invariants.

## 2. Fit exactly two continuous-concept erasers

Concepts are agent `(x,y)` and block `(x,y)`, separately. Apply weighted LEACE to
the centered projected embeddings, minimizing change in the original embedding's
Euclidean metric. Fit means and covariances exclusively on fit families. Do not
standardize the representation before defining this primary erasure metric.

For continuous labels this is a **linear cross-covariance erasure**, not a claim
that all position information has been removed. Record the affine map, removed
rank, covariance spectrum, pseudoinverse cutoff, idempotence residual, fit
cross-covariance residual and per-family perturbation norms. Fix the numerical
cutoff using synthetic numerical tests before real-data fitting, never by probe
or predictor performance; include it in the execution seal.

Verify the implementation against the published LEACE definition and a pinned
author implementation or equivalent reference fixture. Avoid changing the locked
environment unless required; record any dependency choice and license explicitly.

Fit the following controls for each concept:

- Identity transformation.
- Ten seeded random orthogonal removals with the same removed rank. Record their
  distortion as well as rank; rank matching alone does not match perturbation size.
- Ten seeded shuffled-concept erasers. Shuffle entire three-frame label histories
  between fit families within cohorts before applying each family's deduplication
  map. Never shuffle individual correlated frames across partitions.

Do not select favorable control seeds. These controls characterize specificity
and collateral changes; no downstream causal claim is made at this stage.

## 3. Test accessibility, removal and preservation

For identity, each eraser and every control, refit separate ridge probes for both
position concepts using transformed real fit embeddings. Reusing only the old
probe would test damage to that decoder, not whether information remains accessible.
Feature and target statistics are fit-only and family-weighted. Use the existing
six ridge penalties `{0.01, 0.1, 1, 10, 100, 1000}`; select each probe's penalty by
family-averaged calibration error, with stronger regularization winning ties.
Assessment labels cannot affect fitting, penalty selection or erasure construction.

For each concept define predictive gain on a scored set as
`G = 1 - E_probe / E_constant`, where both errors are family-weighted mean squared
two-coordinate errors and the constant predicts the **fit-set mean**. Report raw
workspace-unit mean/p95 errors alongside G. A zero baseline error or inadequate
variation makes the gain unassessable; do not stabilize it with an arbitrary
epsilon. Report both coordinates separately to expose cancellation or one-axis
failure. Primary gates use the joint two-coordinate gain.

Report these fixed feasibility gates for each eraser, overall and in each cohort:

| Check | Proposed criterion |
| --- | --- |
| Pre-erasure accessibility | Identity ridge gain `G >= 0.50` for both concepts |
| Target removal | Refit target ridge gain `G <= 0.10` after erasure |
| Other-concept preservation | Refit other-concept gain retains at least 90% of its identity gain |
| Support | At least ten assessment families in each reported cohort |
| Numerical validity | Finite outputs and reference-tested covariance/idempotence checks pass |

These are new engineering feasibility criteria, not literature-established
universal thresholds and not relaxed physical-accuracy gates. They must be sealed
before execution. Original mean <=1.28/p95 <=2.56 physical gates remain unchanged.

Use 2,000 paired family bootstrap resamples, preserving cohort counts when pooled.
Keep fitted maps/probes fixed in the bootstrap and disclose that intervals omit
training-set variability. For each numerical gate, report its point estimate and
95% interval: a fully satisfying interval passes; a fully violating interval fails;
an interval crossing the boundary is unassessable. Missing/unfinished checks are
unassessable, never failed evidence. For retention, bootstrap the paired difference
`G_other_after - 0.9 * G_other_identity`, rather than an unstable ratio.

As a bounded nonlinear-leakage diagnostic, refit the existing 128-unit ReLU probe
on identity and the two erasers only, predicting all four coordinates. Use three
derived fixed seeds, standardized inputs/targets, equal family weights, full-batch
Adam at 1e-3, weight decay 1e-4, at most 1,000 epochs, calibration early stopping
after 100 epochs without a 1e-4 workspace-unit mean-error improvement. Log every
ten epochs and restore the best calibration checkpoint. Report every seed and
both concepts separately; do not use assessment results to extend training.

If nonlinear probes recover the target after linear erasure, explicitly describe
the intervention as removal of linear accessibility only. It cannot support the
claim that the predictor received no agent/block information. This residual does
not justify another architecture search. Calibration and assessment contact/motion
metadata may describe coverage only; they do not enter concept fitting or select
cases. No contact-mechanism conclusion is available yet.

## 4. Budget, resumability and integrity

Cap cumulative experimental elapsed execution time at 3,600 seconds, with no
parallel worker processes. Stop computation at 3,480 seconds and reserve 120 for
reporting. Include verification, cache reads, missing encodings, fitting and
bootstrap statistics. Time implementation/tests separately. Check the ledger
between families, probe fits, nonlinear epochs and bootstrap batches; report any
unavoidable overrun. Budget exhaustion cannot be relabeled as a failed probe.

Save completed cells atomically, including optimizer and early-stopping state for
nonlinear runs. Matching resumptions reuse validated cells and share the cumulative
ledger. Reject changed signatures or corrupted artifacts. Assessment opens only
after all intended fitted configurations and calibration selections are hashed;
write an opening record and never silently retune after that access.

## 5. Outcomes and the following decision

- **Both erasers pass with bounded collateral damage:** propose a separately
  scoped exploratory predictor intervention. Candidate design: transform only the
  original three observed-history embeddings, keep actions and all real targets
  unchanged, propagate predicted states without repeatedly erasing them, and
  compare unchanged LATENT/GALLERY scores against distortion-aware controls.
  Do not automatically execute this follow-up.
- **Only one eraser passes:** narrow any subsequent question to that representation
  intervention. Do not infer an agent-versus-block asymmetry from unequal eraser
  quality, rank or perturbation size.
- **Removal or preservation fails, nonlinear leakage undermines the intended
  interpretation, or precision is inadequate:** report the exact obstacle and stop
  this route. No automatic move to circuit probing, DAS, new bins or more data.
- **Integrity fails:** block affected analysis and report the defect. Incomplete
  computation remains explicitly incomplete.

A later predictor study would need its own behavior metrics, action/history
isolation tests, intervention-distribution checks, matched controls and contact
coverage. Strong confirmation of a revised mechanism claim would then require
newly reserved families under a separately approved plan. A successful feasibility
gate here is neither new causal evidence nor proof of novelty.

## Validation and deliverables

Before real fitting, test known independent and correlated synthetic concepts,
rank-deficient covariance, exact erasure/idempotence, translation conventions,
collateral damage detection, refitted-probe recovery, identical-frame conflicts,
family weighting, fit-only statistics, calibration-only selection, partition
isolation, projected-cache identity, corruption rejection, atomic resume, budget
exhaustion, sparse/zero-variance scores and pass/fail/unassessable reporting.

Run `make check`, focused tests, `make test`, locked-environment verification and
`git diff --check`; record actual outcomes. Required deliverables are a sealed
manifest, complete fit/control configurations, erasure maps, probe curves,
per-family assessment scores, bootstrap summaries, timing/storage ledger, and a
concise tracked result report with the decision above. Update TASKS, append LOGBOOK
evidence and record the sealed protocol and observed outcome separately in
DECISIONS. Preserve all earlier physical-readout failures and historical
training-simulator uncertainty.

## Method and contribution references

This plan adapts [LEACE](https://arxiv.org/abs/2306.03819) and the selectivity
concerns in [Improving Causal Interventions in Amnesic Probing with Mean Projection
or LEACE](https://arxiv.org/abs/2506.11673). The exact application proposed here
has not been established as novel. `PROBING-001_REVIEW.md` records close JEPA
precedents and the distinction between a new empirical mechanism finding and
applying an established method. No new literature claim supersedes those limits.
