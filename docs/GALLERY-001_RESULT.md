# GALLERY-001 — Rendered block-position measurement feasibility

Executed 2026-09-28 under GALLERY-001_PLAN.md. **Decision: proceed only to a
separately scoped, exploratory predictor-to-gallery audit.** Both proposed
simulator/encoder gates passed on the selected tuning families. The related-work
review does not support a general method-novelty claim; the defensible next
question is checkpoint-specific and empirical. No predictor outputs or
confirmation labels were read for this experiment.

## Evidence and coverage

The sealed configuration selected 30 existing READOUT tuning families (20
challenge, 10 ordinary; six candidates each) at the unchanged two-native-step
horizon. Twelve were preselected LATENT verification families; the remaining
18 were drawn with the recorded seed before rendering. All 180 terminal
branches replayed and restored their saved RGB exactly. The four preselected
contact sheets in `outputs/gallery001/examples/` were inspected: the red
outline follows the rendered T-block geometry, the blue outline marks the
held agent, and the goal and background remain fixed. Changed pixels outside
the old/new block footprints were disallowed by the runner. There were no
integrity failures or budget interruptions.

Each gallery has the realized block pose and up to four admissible translations
at fixed orientation and agent pose. The query is a separate, nonidentical
1.28-unit translation; it is never one of the templates. Exact-frame retrieval
is reported solely as an identity control. Geometry and RGB rejection reasons
are saved for every attempted direction. One ordinary endpoint could supply
only three valid 10.24-unit negatives because five of its eight directions
penetrated the wall; it is excluded only from that scale's score. All other
endpoints met the four-negative/two-query requirement. The 360 selected
queries are nonidentical to their true templates. Across attempts, 53
otherwise proposed query directions were rejected for agent/wall penetration.
Nineteen actual endpoints had slight existing contact penetration (minimum
clearance about -0.99 workspace units); alternatives were checked against the
strict geometry rule rather than silently inheriting those overlaps.

| Negative displacement | Cohort | Eligible endpoints | Families with >=4 eligible candidates | Nonidentical-query top-1 | Family-weighted fifth percentile margin ratio (95% family bootstrap interval) |
| --- | --- | ---: | ---: | ---: | ---: |
| 10.24 | Challenge | 120/120 | 20/20 | 100% | 0.847 [0.792, 0.903] |
| 10.24 | Ordinary | 59/60 | 10/10 | 100% | 0.831 [0.815, 0.903] |
| 20.48 | Challenge | 120/120 | 20/20 | 100% | 0.965 [0.962, 0.974] |
| 20.48 | Ordinary | 60/60 | 10/10 | 100% | 0.946 [0.936, 0.966] |

The top-1 family bootstrap intervals are [100%, 100%] in all four cells.
These intervals describe the selected-family resampling distribution, not
independent generalization. All four cells exceed the predeclared 80%
endpoint/ten-family coverage, 95% retrieval and 0.5 margin-ratio gates.
Uniform five-way guessing has 20% expected accuracy. The constant-embedding
fixture fails as intended; oracle and tie fixtures pass. Exact-frame identity
retrieval was 179/179 eligible endpoints at 10.24 and 180/180 at 20.48,
reported separately from nonidentical-query scores. Duplicate and reversed
encoding of every gallery image in the twelve verification families had zero
maximum absolute discrepancy; the measured numerical floor was `1e-12`.

Contact-pair regimes had 30 `neither`, 21 `exactly_one`, and 20 `both`
contributing families at each scale. Motion-pair regimes had the same family
counts. All meet the ten-family assessability convention; their deduplicated
endpoint metrics and intervals are in `outputs/gallery001/report.json`. Saved
real branch labels determined regimes. No contact label was inferred for a
synthetic image.

## Interpretation and limits

The frozen projected CLS reliably separates these controlled rendered block
positions at the two specified distances. It is an encoder measurement with a
known simulator pose, not evidence that the action-conditioned predictor
locates the block. Holding the image of the agent fixed in each bank removes
that visible nuisance but does not algebraically isolate block features inside
the nonlinear embedding. The queries and negatives are synthetic single-frame
poses; some may be outside the training distribution or unreachable by the
recorded action. The larger translations may be visually easy. Success here
does not satisfy the prior 1.28/2.56-unit physical readout gates, estimate a
continuous error, establish contact dynamics, or measure regret.

The [related-work comparison](GALLERY-001_RELATED_WORK.md) finds prior
analysis-by-synthesis, feature-space pose retrieval, object interventions and
action-discrimination evaluations. General method novelty is unsupported.
The potential contribution is a reproducible empirical diagnosis of this one
frozen checkpoint: whether global action correspondence persists when outcome
assessment is conditioned on block pose at a fixed rendered agent pose. That
claim still requires a later sealed predictor comparison with persistence and
action-permutation controls, plus fresh reserved evidence if protocol choices
are informed by these tuning families. Historical training-simulator identity
remains unproven.

## Execution and provenance

The locked Python 3.10.21 CPU float32 run took **46.79 seconds cumulative**:
0.10 audit, 8.46 replay/render, 35.93 encoding and 2.30 reporting. The cap was
1,740 seconds for experimental work plus 60 seconds for reporting. Rendered
NPZ files occupy 6.31 MB, projected cache 1.73 MB, encoded records 0.26 MB,
and contact sheets 0.07 MB. The tracked
`docs/manifests/gallery001-runtime.json` records exact artifact hashes, source
and checkpoint identity, selected IDs, seed streams, commands and exclusions.
The first successful run was preserved in ignored `outputs/gallery001-preliminary/`
before adding the required detailed strata, distances and geometry overlays;
the final sealed run was executed again from an empty GALLERY artifact root.

The final source passed `make check`, the nine focused gallery tests, all 75
tests in `make test`, locked-environment verification, and `git diff --check` as
recorded in LOGBOOK. Further model measurement is outside this run's frozen
scope.
