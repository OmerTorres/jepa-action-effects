# MEAS-001 engineering review — 2026-10-01

Reviewed commit **8ecd208**, with a clean working tree at the start. The failed
engineering outcome is credible and the stop was appropriate. Accept this as a
documented failed engineering pass, **not a freeze-ready instrument**. Fix the guards
below before progressing. One bounded engineering version 2 is reasonable; no new
experiment or protocol amendment was executed by this review.

## Verified evidence

- `make check && make test`: **299 passed**, no skips, 24 Gymnasium warnings;
  pytest 14.88 s. Python 3.10.21.
- Reloaded all **72** saved traces and checked their hashes, configurations and
  classifications against the sealed engineering spec and saved results. All match.
- Recomputed `engineering_diagnostics` from those traces: exact equality with the
  tracked engineering manifest. S2 median best error is **1.0519233 units**, with
  engineering τ **0.1282357**. All settings accept **0/18 real queries** (including
  starts) and **0/6 synthetic queries**.
- All 30 engineering and 48 MEAS family-file hashes match their recorded hashes;
  proposal source/checkpoint hashes match current files. Existing evidence has no
  detected lineage mismatch.
- Traces concern only the six phase-A GALLERY families. No phase-B traces, frozen
  configuration or calibration result exist. This is consistent with the reported
  stop; saved artifacts cannot prove that a file was never accessed by any process.
- The ledger contains one completed session of **1,191.423332 s**. It does not include
  the separately disclosed implementation profiling.

Tests include simulator fixtures. Review diagnostics otherwise used saved engineering
traces, source inspection and synthetic/mocked data. No MEAS family inversion,
checkpoint inference, threshold calibration or predictor-output analysis was run.

## Findings to fix before freeze

### F1 — P2: recorded provenance is not enforced at input and result reuse

Locations: `meas001.py:300–311`, `900–925` and `933–964`.

`_run_target` returns an existing result after checking only its NPZ hash. It does not
compare the saved search configuration, query identity, seed, target/context or source
version with the requested run. A synthetic reproduction saved a four-start result for
`old-query`, then requested two starts for `different-query` at the same path. The old
result was returned without error or computation. The proposed version-2 work makes
this particularly relevant: an intact file is not necessarily evidence for the current
instrument.

Also, `require_frozen` checks the membership JSON's hash but does not check the family
files named inside it. `query_inputs` verifies the latent-cache hash but omits the family
NPZ hash, and `_labels` loads that family unchecked. In a temporary synthetic fixture,
changing the family seed from 1 to 999 after freezing passed both `require_frozen` and
`query_inputs`, despite a recorded hash mismatch.

**Correction:** verify family/collection/cache lineage before use and before reporting.
Bind each trace to the exact query, target and observed context, search config, seed,
checkpoint and source version. Refuse mismatches; use distinct version directories.
Record source/input fingerprints when engineering runs are created, not only when a
later proposal is written. Add mutation tests for each dependency and ensure failures
leave existing reports unchanged. The current files match; these are guard defects,
not evidence that the saved run actually mixed versions.

### F2 — P2: the optional prediction gate ignores failed assessment gates

Locations: `meas001_metrics.py:169–172`; `meas001.py:1315–1345`.

`predictions_allowed` checks only endpoint/start coverage and position accuracy.
`score` combines this with integrity, but ignores completion, displacement accuracy,
relative floors and known-failure controls. Two synthetic reproductions show:

- all position cells pass but every displacement floor fails: only start cells are
  eligible, yet `predictions_allowed=True`;
- known-failure controls fail: the decision is `not_feasible`, yet
  `predictions_allowed=True`.

The prediction stage trusts that boolean and would open its saved predictor targets.
D032 says failed cells stay ineligible and requires all named cells to pass before
this optional check.

**Correction:** derive eligibility from the complete assessment decision. Require
completion, integrity, known-failure controls, both start cells, and the four specified
endpoint cells including their displacement and relative-floor gates. Preserve the
all-or-none rule. Test each failing condition with a sentinel proving prediction
latents are never loaded.

### F3 — P2: restart agreement need not support the returned block center

Location: `pose_search.py:443–473`.

`Inversion.best` includes nuisance-search candidates, but restart agreement uses only
the original two search results. A better nuisance candidate can exclude both original
results from the near-optimal set, leaving apparent agreement and zero diameter for a
returned pose that neither independent search recovered. Nuisance convergence flags
are also ignored by classification.

Reproduction: both independent searches converge at block `(250, 260)` with residual
0.1; one nuisance candidate reaches `(260, 260)` with residual 0 and has
`converged=False`. At τ=0.01 the classifier returns **accepted**, block `(260, 260)`,
restart distance 0 and near-optimal diameter 0. This is 10 units from both independently
supported solutions.

**Correction:** require independent support for the final reported solution. A newly
discovered better basin should trigger bounded, separately initialized verification or
an unresolved abstention. Explicitly define the outcome of unfinished applicable
nuisance challenges; do not treat an exhausted challenge as evidence that no alternative
exists. Add fixtures for both final-basin disagreement and nuisance-budget exhaustion.
All saved engineering nuisance challenges finished, and every saved query abstained;
this defect does not turn the existing failure into a success or invalidate its counts.

### F4 — P2: the diagnosis overstates what finite search traces establish

Locations: `MEAS-001_ENGINEERING.md:49–70` and the current TASKS handoff.

The distance table is a median of per-target **best visited** residuals in distance
bins. Different bins can contain different targets; poses also differ in agent position
and angle. This is not a controlled residual-versus-distance sweep, a smooth landscape
measurement, or evidence that all alternative minima were excluded. In particular,
failure to find a wrong pose that beats truth cannot establish identifiability.

**Correction:** say that near-truth poses can fit well, sampled distant poses generally
fit worse, and the optimizer often fails to reach the known low-residual solution.
Search error and the broad τ-based band are supported concerns; joint block
identifiability remains unvalidated. Preserve the small positive evidence without
calling the inverse problem solved. The diagnostic counts `better_than_truth_but_far`
only for best poses more than 5.12 units from truth; it does not itself certify the
broader sentence “No wrong pose fitted better than the true pose.”

## Recommended review decisions

1. **Support the agent-geometry and reset-identity corrections.** The pinned simulator
   defines the agent as kinematic; excluding realized wall/off-canvas poses would be
   inappropriate. Retain block checks and the declared contact tolerance. Treat an
   invisible agent's coordinates as nuisance representatives, not identified positions.
   Verify the proposed finite bounds cover the renderer's clipped/invisible cases with
   fixtures. The constant archive seed is not an independent-family identifier; explicit
   reset states and source episodes are the appropriate aliases, with seeds for seeded
   resets. Keep this rule and the actual disjointness evidence explicit in the next spec.

2. **Recommend one bounded version 2 after the guard fixes.** Preserve version 1 and
   seal a new spec/output directory before another encoder run. Develop on phase A;
   run the selected design once on phase B with settings and engineering thresholds
   fixed. Phase B has no search traces, but the earlier all-30-family renderer audit
   included these families: describe it as unused for search validation, not wholly
   uninspected. Keep the original cumulative 60-minute engineering and four-hour
   overall limits. Correct their accounting first. Stop if version 2 fails.

   **Do not pool the two searches' bests into a shared final initialization and then
   count the resulting agreement as independent evidence.** Refine each search from
   its own best basin under a fixed budget. A pooled candidate can be a proposal, but
   still needs independent support. Reallocate the declared evaluation budget or
   explicitly propose a changed cap and new full-workload projection before running.

3. **Revise the threshold design before calibration.** Separate an absolute fit
   acceptance threshold from the tolerance used to define near-optimal alternatives.
   D032's shared τ lets optimization error inflate both. Do not merely tighten the
   band until coverage passes: require converged searches, numerical/rendering-floor
   evidence, ambiguity controls and phase-B recovery with fixed rules.

   The suggested 1.28-unit offset experiment could characterize sensitivity, but is
   not yet a justified tolerance formula. Rasterization, direction, scene and agent/
   angle compensation can change the residual response; a single offset quantile
   does not validate identification. Develop the complete rule on engineering data,
   retaining the physical accuracy/coverage gates. If calibration later exposes a
   problem requiring changed rules, follow D033's retirement/version rule.

These are review recommendations, not an executed protocol amendment. No value for a
replacement threshold has been chosen, and no further computation has been authorized
or started by this review.

## Remaining completion and accounting work

- The “complete runner” description is premature. `predictions` saves inversion
  records but does not compute the planned MLP/inversion disagreement on identical
  targets. `report` writes a JSON manifest; it produces neither the promised draft/
  tables nor preselected visual examples. Finish and test this against mock artifacts
  before the source freeze, without opening real predictor outputs early.
- Score the six genuine rendered controls explicitly for recovery. They currently
  retain truth and inversion results, but no recovery summary; the final control boolean
  is only the analytic known-failure result. Preserve the distinction from mixtures,
  which have no required physical-mean inverse or blanket rejection criterion.
- The runbook's claim that every stage charges the ledger is incorrect: `score`
  performs replay/renderer checks and bootstraps outside `_session`, and implementation
  profiling was explicitly uncharged. Add a transparent accounting correction for
  prior profiling/integrity work, using measured times or a disclosed conservative
  estimate, and charge future simulator/encoder validation. Do not reset either cap.
  The existing 141-minute workload estimate is encouraging, not a validated projection
  for a more expensive search version.

## Reproduction and handoff

Review command: `make check && make test`. Additional temporary script
`/tmp/meas001_review_checks.py` reloaded saved traces, recomputed diagnostics, checked
hashes and ran the synthetic counterexamples above. It changed no study artifacts.
The counterexamples are specified here so regression tests need not depend on that
temporary file.

Next implementation handoff: address F1–F4, finish mock-tested output/accounting paths,
and prepare the revised engineering spec with independent refinement and separate
threshold roles. Keep the 48 MEAS families closed. Further engineering execution, if
pursued, ends at a new freeze-ready or not-ready proposal; it does not automatically
proceed to calibration, assessment or saved predictions.
