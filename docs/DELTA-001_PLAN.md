# DELTA-001 — Bounded displacement measurement pilot

Status: planned, not executed. This is an exploratory revision to measurement,
not a frozen main-study protocol. Preserve PILOT-001 and READOUT-001 evidence.

## Objective and limits

Test whether block displacement and differences between candidate effects can be
decoded accurately even though absolute block position could not. Shared position
bias may cancel; this is a hypothesis, not an established remedy.

Reuse the frozen checkpoint, Python 3.10.21 locked environment, CPU float32 model
inference, two-native-step horizon, six candidates, and `delta=5.12`. Use only
existing families and projected CLS embeddings. No collection, new world-model
training, accelerator migration, threshold relaxation, or main-test inference.

Cap experiment execution at two cumulative hours on the local CPU across stages
and resumes. Persist consumed runtime; restarting must not reset the allowance.
Check the budget before each stage/cell and at nonlinear checkpoints. Report
unfinished work as incomplete, not a failed scientific gate. Implementation and
ordinary test-suite time are recorded separately from experiment execution.

## 1. Validate and seal the inputs

- Add reusable modules under `src/jepa_action_effects/`, exposed through
  `python -m jepa_action_effects.delta_readout` with `audit`, `fit`, `confirm`,
  `evaluate`, and `report` stages. Use ignored `outputs/delta001/` artifacts.
- Verify upstream manifests, checkpoint/cache identities, candidate ordering,
  family isolation, horizon, and native timestamps. Bind every downstream stage
  to input/configuration/code/model hashes; reject mismatches on resume.
- Use the same 240 fit families (160 challenge / 80 ordinary) and fresh 60-family
  tuning partition (40/20) as READOUT-001. Do not repeat the nested-size search.
- Retain its 60 reserved confirmation families (40/20). Verify their outcomes
  have not informed previous fitting, selection, or diagnostic metrics. Distinguish
  metadata/hash inspection from evaluation; do not claim raw files were never
  accessed merely because no confirmation report exists. If independence cannot
  be supported, mark confirmation unavailable and stop before claiming recovery.
- Restrict fit/tuning loaders to explicit allowlists. Confirmation labels must
  remain outside fitting and model-selection data structures.
- Freeze this protocol and a seed configuration using master seed `20260928`,
  with independent streams for fit diagnostics, action permutations, and bootstrap
  resampling, before displacement scores are computed.

## 2. Define the measured quantity

For each family, let `z0` encode the final history frame and let `p0` be its
simulator block position. For each candidate, let `zT` encode the real frame at
the selected two-step endpoint, with simulator position `pT`. The target is
`d = pT - p0`; the readout receives embeddings only. Simulator positions, contact
labels, candidate actions, and candidate IDs are never readout input features.

Train on endpoint transition pairs only. Deduplicate identical `(z0-frame,
zT-frame)` input pairs within each fit family; detect and report conflicting
displacement labels before deduplication. Preserve transition-to-candidate maps.
Give each family equal total weight in scaling, training, and aggregate errors.
Do not add intermediate-time targets to this endpoint-specific experiment.

For candidates `a,b`, paired effect is `d_a-d_b`, which equals `pT_a-pT_b`.
Keep endpoint displacement and paired-effect errors separate. No-contact and
no-motion remain distinct: a block can move without new candidate-phase contact.

## 3. Try only the following ordered readouts

Stop searching as soon as the next candidate in this fixed order passes tuning.
All selection uses real-frame tuning outcomes; predictor errors cannot select a
readout, family, seed, or regularization value.

1. **Existing ridge difference:** Apply the saved 240-family projected ridge
   unchanged: `r(zT)-r(z0)`. This tests whether its absolute position bias cancels
   without fitting another probe. Also report zero displacement as a baseline.
2. **Direct displacement ridge:** Fit `W(zT-z0)` with no intercept. Use fit-only,
   family-weighted feature scales without centering the difference, preserving
   an exact zero output when the two embeddings are equal. Select alpha from
   `{0.01, 0.1, 1, 10, 100, 1000}` by family-averaged tuning displacement error;
   exact ties favor stronger regularization. Do not choose another alpha by
   inspecting whether its secondary gate happened to pass.
3. **One bounded nonlinear diagnostic:** One 128-unit ReLU hidden layer, two
   displacement outputs, and the three existing seeds `20260931–20260933`.
   Input consists of `z0` and `zT-z0` (384 features). Define the output as
   `F(z0,zT-z0)-F(z0,0)` using the same standardized input transform in both
   calls. This enforces zero displacement for an identical endpoint embedding.
   Use fit-only, family-weighted statistics and scale-only displacement targets
   so zero is preserved. Train with full-batch family-weighted squared loss,
   Adam `1e-3`, weight decay `1e-4`, at most 3,000 epochs; early-stop after 200
   epochs without a `1e-4` workspace-unit improvement in tuning mean error.
   Restore the best checkpoint, log every ten epochs, and distinguish early
   stopping, epoch cap, and runtime exhaustion. Require all three seeds to pass;
   do not select or ensemble the best seed.

Before fitting the full new probes, verify exact-zero behavior and recovery of
known synthetic displacement targets. A deterministic 32-pair fit-only diagnostic
may check memorization; it cannot establish generalization. Do not add readouts,
raw-CLS alternatives, data sizes, or seeds if these candidates fail.

## 4. Freeze a displacement-specific measurement gate

Keep numerical accuracy limits at mean Euclidean error `<=1.28` and weighted p95
`<=2.56`. These are proposed displacement/effect gates, not a retroactive pass
of the failed absolute-position gate.

Require both limits, with at least ten contributing families, for:

- Endpoint displacement overall, separately by ordinary/challenge cohort, and
  in pooled `neither`, `exactly_one`, and `both` contact-pair regimes. Within each
  family/regime count each participating candidate endpoint once.
- Paired-effect error overall, in each pooled contact regime, and among meaningful
  pairs (`norm(d_a-d_b)>5.12`). Count each unordered candidate pair once. The
  meaningful-pair gate prevents mostly stationary examples from validating a
  near-zero decoder.

Average within families before aggregation; construct p95 using weights giving
each contributing family equal total mass. For pair regimes, overlapping families
are permitted and reported. Report family/endpoint/pair counts and cohort-specific
regime breakdowns. Fewer than ten families is **unassessable**, not pass. Sparse
cohort-by-regime cells cannot support claims even if pooled gates pass.

Report zero-displacement baseline error on exactly the same subsets, including
its meaningful-effect misses. Do not change subset definitions after seeing
readout scores. Accuracy gates establish measurement validity; they do not by
themselves establish frozen-predictor competence.

## 5. Select once and confirm once

- If no candidate passes every required tuning gate, save a failed-measurement
  report and leave confirmation outcomes unused. Skip physical predictor analysis.
- If one passes, hash its specification, fit/tuning membership, feature statistics,
  weights, gates, and code before opening confirmation. For nonlinear selection,
  seal all three seed models. Do not refit on tuning families.
- Evaluate only that selected configuration on all 60 confirmation families,
  using the identical required gates. Report every nonlinear seed separately.
- Make confirmation idempotent: a repeat invocation reads the sealed result and
  rejects a changed configuration. Failed/insufficient confirmation ends this
  recovery attempt; do not return to tuning and reuse confirmation for selection.

## 6. Conditional exploratory predictor evaluation

Only after confirmation passes, evaluate on the existing 30 pilot development
families (20 challenge / 10 ordinary). These have already been examined; all
comparisons remain exploratory. Confirmation serves readout validation only.

Apply the same selected readout unchanged to `(z0,zT)` and `(z0,predicted_zT)`.
Predictor inputs stay observed history and candidate actions only. Include frozen
JEPA, real-future-frame readout control, zero displacement/persistence, simulator
oracle, and a deterministic within-family action-permutation control. Use one
seeded derangement of the six candidate sequences per family, leaving history
fixed and assigning outputs to the intended candidate IDs; preserve magnitudes
and the candidate set. Cached prediction reindexing is allowed only after an
order-equivariance check. Treat this as action-channel correspondence evidence,
not proof of physically correct reasoning.

Check the real-frame control on this development distribution too. Failed accuracy
or insufficient counts make the corresponding physical claims unassessable, even
after confirmation passed. Do not tune again. If nonlinear readouts were selected,
report predictor metrics for all three seeds and qualify claims that change with
the readout seed; never select a seed using predictor performance.

Use the existing evaluator for displacement and paired effects, preserving D008
contact, tie, eligibility, negligible-effect, direction, and goal conventions.
Use 2,000 family bootstrap samples for exploratory 95% intervals and paired
contrasts on common families. Require ten contributing development families per
central comparison; otherwise mark the comparison unavailable.

For existing fixed-goal regret, score candidates with `p0 + predicted_d`. The
true starting position is supplied only to this downstream scoring reference,
not the predictor or readout. Label this **oracle-initialized regret**: it does
not establish visually localized endpoint prediction or autonomous visual action
selection. Report uniform-random expected regret, informative-goal counts, and
all-goal versus informative-goal results. Unavailable agent/orientation/trajectory
outputs remain unavailable. Never subtract real-frame decoder error from model
error or rehabilitate the historical absolute endpoint results.

## 7. Deliverables, checks, and stopping decisions

Add behavioral tests for timestamp/label pairing, nonzero moving no-contact cases,
shared-bias cancellation, exact-zero anchoring, meaningful-effect detection,
family weights and deduplication, fit-only transforms, tuning-only selection,
confirmation isolation/one-time opening, action permutation, oracle-initialized
regret, artifact mismatch, resume, and cumulative runtime accounting. Retain
existing evaluator fixtures. Run `make check`, focused tests, `make test`, locked
environment verification, and `git diff --check`; record actual commands/results.

Save recipes by reference, paired-frame identities, embeddings, scaling, fitted
probes, curves, raw family metrics, selection/confirmation manifests, runtime by
stage, storage, and a concise report under `outputs/delta001/`. Add tracked
`docs/manifests/delta001-runtime.json` and update TASKS/LOGBOOK/DECISIONS. Preserve
prior evidence and distinguish planned, executed, failed, and unassessable work.

- **Measurement passes:** Complete the conditional exploratory comparison. A
  null action contrast is a valid result; report its interval. Recommend any
  larger study only as a separate proposal, with adequate ordinary competence,
  coverage, and precision established first.
- **Readout fails or confirmation fails/is insufficient:** Close this bounded
  physical-measurement recovery route. Report the limitation without asserting
  that block information is absent or that JEPA physics has failed. No automatic
  expansion or additional readout search.
- **Execution blocked/incomplete:** Identify the implementation/data/budget issue;
  do not count missing work as negative scientific evidence.

LIT-001 can proceed independently. A later latent-space study needs its own
question, controls, and claim boundaries; these physical gates do not authorize
it automatically. Historical training-simulator identity remains unproven.
