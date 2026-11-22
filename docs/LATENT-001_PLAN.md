# LATENT-001 — Exploratory action-to-future correspondence

Status: proposed protocol, not executed or frozen. Created 2026-09-27 following
LIT-001. This is a scope revision; RESEARCH_PLAN.md and historical experiment
results remain intact. Main-study physical evaluation remains paused.

## Question and claim boundary

For the frozen LeWM checkpoint, do correct candidate actions predict their
corresponding real future embeddings better than alternative candidate actions
and latent persistence, under a common observed history?

Report separately whether predictions respond to actions, correspond to their
real encoded futures, and improve absolute prediction over persistence. These
are distinct outcomes. A positive result establishes usefulness in the model's
own representation on this sampled distribution. Global CLS scores can be
dominated by agent motion and cannot establish block accuracy, contact
reasoning, physical regret, or general physical competence. Negative results
also cannot localize representation versus predictor versus distribution issues.

The contribution is a controlled case study, grounded in LIT-001_REVIEW.md,
not a claim to invent action permutation or latent-response evaluation.

## Bounded data and inference

Use exactly the existing 60 READOUT-001 tuning families (40 challenge/20
ordinary), all six candidates, and the existing two-native-step horizon.
These families already informed readout diagnostics; this is explicitly
exploratory reuse, not an untouched test. No fit, probe selection, extra
collection, pixel decoder, altered thresholds, or checkpoint training.
Do not access the 60 readout-confirmation families or main-test data.

Seal a configuration and exact family IDs before computing scores. Validate
upstream manifest, checkpoint, preprocessing, projected-space identity, action
normalization, candidate IDs, and frame alignment. Reuse verified real-frame
caches; generate only missing frozen predictions using observed history and
candidate actions. Encode actual future frames only for scoring. CPU float32,
Python 3.10.21, locked environment, original checkpoint/simulator identity.

Proposed execution cap: 30 minutes cumulative CPU-run wall time, checkpointed
between stages; record incomplete work without widening the cap. Artifact root
`outputs/latent001/`. Master seed 20260929, separate derived streams for
bootstrap and preselected visual examples. No seed is selected by performance.

## Primary measurements

For family f and K=6 candidates, let z_j be the real terminal projected CLS,
p_i the predicted terminal CLS for action i, and z_0 the latest observed CLS.
Dimension d=192. Define `C_ij = ||p_i-z_j||²/d` in the unchanged model space.
No whitening, PCA, feature selection, or learned distance enters scoring.

1. Matched error: `L_f = mean_i C_ii`.
2. Wrong-action error: `W_f = sum_(i!=j) C_ij / (K*(K-1))`.
3. Correspondence advantage: `A_f = W_f-L_f`; positive is better matching.
4. Persistence error: `P_f = mean_j ||z_0-z_j||²/d`.
5. Persistence improvement: `B_f = P_f-L_f`; positive is better prediction.

W equals the expected error under uniform within-family derangements of whole
candidate sequences. Compute it exactly from off-diagonal entries; optional
enumeration of all 265 derangements is a verification, not another source of
independent samples. Hold the executed history fixed. Never permute individual
controls, future frames across families, or model/target IDs independently.

For pair (i,j), use
`A_fij = (C_ij + C_ji - C_ii - C_jj)/2`.
Its average over all unordered pairs equals A_f, so strata use the same
correspondence estimand. This contrast cancels additive row/column costs and
is zero for action-independent predictions, even when targets differ.
It measures alignment in the encoder's metric, not accuracy in workspace units.

Report predicted pairwise spread, real pairwise spread, and real-target
collapse/duplicate counts as secondary diagnostics. Keep all valid families
in primary scores, including identical futures; no model-error-based filtering.
Do not divide by small true effects or tiny latent spread. Floating-point
degeneracy flags use `1e-8 * max(1, mean_j ||z_j||²/d)` as a reported numerical
tolerance, never an exclusion or physical threshold.

## Strata, uncertainty, and interpretation

Report A and B overall and separately for challenge/ordinary. For contact
regimes neither/exactly_one/both, average A_fij within family before combining
families. Additionally split pairs using unchanged simulator block endpoint
difference `>5.12` versus `<=5.12`. Report contact and motion separately.
For regime-specific endpoint L/P/B, deduplicate participating candidate IDs
within each family. Also record true agent displacement descriptively: block
effect strata do not remove the agent-motion confound.

Use 2,000 family bootstrap resamples and percentile 95% exploratory intervals.
Overall bootstrap preserves cohort counts; subset contrasts use common families
and resample their paired differences. Report counts, raw effects, and intervals.
With fewer than ten contributing families, mark comparison unassessable.
Ordinary/challenge differences are descriptive distribution comparisons, not a
causal contact effect. No significance claims from treating branches as samples;
no multiple-comparison search for a favorable subset.

Positive correspondence evidence requires the A interval to lie above zero;
positive improvement over persistence separately requires B above zero. Intervals
crossing zero are inconclusive, not evidence of equality. Negative intervals
support worse performance on this distribution only. These rules do not restore
the failed physical gate. Even positive A and B may reflect agent features.

## Implementation and stopping criteria

Add a reusable metric module and staged audit/evaluate/report command. Before
real scoring, test exact oracle matches, action-independent outputs, swapped
labels, identical targets, additive row/column invariance, pair/all-family
equivalence, whole-sequence permutation equivalence, family weighting, cache
mismatch rejection, future-input isolation, and reserved-family exclusion.
Check reversed candidate execution order leaves ID-aligned predictions unchanged.
Reuse existing manual/autoregressive alignment tests. An oracle embedding is
an evaluator fixture/reference, not a model that may see future inputs.

Run `make check`, relevant tests, `make test`, locked-environment verification,
and `git diff --check`. Record actual commands, source/configuration/input hashes,
family membership, exclusions, timing, and storage in a tracked runtime manifest.
Save raw family scores and all matrices in ignored artifacts. Show a seeded
preselected example set alongside aggregate results; examples do not set claims.

End with a concise result, including an inconclusive or negative one. Do not
automatically add datasets, action controls, alternate distance metrics, readouts,
or checkpoints in response to results. The final course report should retain
the physical feasibility failures alongside this narrower audit. Any independent
confirmation or change of model requires a separate proposal and new provenance.
