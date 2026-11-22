# MEAS-001 — Bounded feasibility of joint pose inversion

**Prepared:** 2026-10-01, after CALIB-REVIEW-002 and D031.
**Status:** proposal only. No implementation, experiment or new confirmation has run.
**Purpose:** decide whether inverse rendering can provide a useful, abstaining
block-position diagnostic for the existing LeWM/PushT pair. A passing result would
justify a later measurement protocol, not certify physical prediction or a mechanism.

## 1. Question and scope

Given a projected latent, can we recover a sufficiently precise block position
without fixing the agent or block angle to its true future value? Can we recognize
poor fits, competing poses and search failures instead of always returning a pose?

Keep the same pinned simulator, checkpoint, preprocessing, 192-dimensional projected
representation and two-native-step horizon. Freeze the encoder and every existing
readout. No training, second model, new simulator cohort, planning experiment or
fresh confirmation is included. Existing scientific seals remain unchanged.

GALLERY-001 recovered real poses at coarse translation scales with the future
agent and block angle fixed. PREDICTOR-GALLERY-001 failed its primary challenge
criterion. Neither validates joint continuous inversion. MEAS-001 must address
that gap explicitly; it must not repeat exact-template retrieval and call it
physical measurement.

Inverse rendering is another decoder using the same encoder. In particular,
inverting a conditional-mean latent need not return a conditional-mean physical
state. Agreement with the MLP would be corroboration, not proof of either decoder's
validity on predictions.

## 2. Bounded data and partitions

Use **48 existing SHIFT-002 development families**, selected from the saved
collection by metadata alone: 24 challenge and 24 archive. Seed `2026100101`;
sort canonical IDs, then permute within each scene. Assign the first 12 per scene
to calibration and the next 12 to an instrument assessment partition. Keep source
episode/reset families together; aliases or repeated episodes must not cross
partitions. Abort selection if the required independent families are unavailable.

All source data have previously been opened. “Assessment” means withheld from
this instrument's calibration, not fresh scientific confirmation. No reserved
cohort is opened by this task. Record membership and input hashes before encoding.

For every selected family, invert the observed start and these terminal real
latents:

| Scene | Three endpoint queries per family |
| --- | --- |
| Challenge | constant commands 0.30, 0.45, 0.60 |
| Archive | varying commands at ×1.0, ×1.5, ×2.0 |

Also retain the challenge 0.15 endpoint as a low-motion control. Resolve labels
from stored candidate metadata, never array position assumptions. This gives
**216 primary real-latent queries**: 108 calibration and 108 assessment.
Store all exclusions and count them against coverage; do not replace difficult
families. Synthetic controls below use only calibration families.

## 3. Instrument and information boundary

Minimize raw projected-latent MSE between the supplied target and
`encode(render(agent_x, agent_y, block_x, block_y, block_angle))`.

- Jointly search all five pose coordinates over the visible workspace and angle
  modulo 2π. Keep goal, background, shape and camera equal to the known scene
  context. They must be obtained from the observed scene, not a future endpoint.
- Use render-only mutation and restoration. Set block angle before position and
  reindex shapes, following the existing renderer audit. Do not step physics
  inside the search. Extend geometry checks to the moving agent as well as block.
- Use a uniform **1.0-workspace-unit maximum contact penetration tolerance** for
  all poses, including controls. This is a proposed engineering allowance for
  the small contact overlaps seen in GALLERY-001, not a query-dependent allowance.
  Reject larger overlaps and out-of-workspace geometry. Report unsupported real
  frames; do not widen the tolerance after inspecting them.
- Search cannot receive the true future pose, a bank containing its exact latent,
  the MLP's predicted pose, or ground-truth endpoint-centered search bounds.
  Validator code may access labels after inversion for scoring. Separate that API
  from the optimizer. The observed start pose may be a declared privileged seed;
  it cannot define a narrow search box. This is a diagnostic, not a deployable
  state estimator or planner.

Proposed search: two independently seeded space-filling coarse searches followed
by bounded coordinate-pattern refinement. Per target, cap at **2,048 render/encode
evaluations**: each search has 256 coarse proposals and four 160-evaluation local
starts; reserve 256 evaluations for nuisance/ambiguity challenges. Use independent
seed streams derived from the master seed and query ID. Cache exact duplicate
poses without hiding their attempted-query counts.

The implementation must spell out proposal generation, invalid-pose attempt caps,
local step sizes, tie-breaking and termination in its config before calibration.
Use existing NumPy/SciPy facilities; no new optimizer dependency is assumed.
Unit fixtures may guide implementation. No optimizer/configuration sweep on
assessment or model predictions is allowed. Failure to converge within the fixed
budget is an abstention, not evidence of a bad world model.

## 4. Rejection and ambiguity rules

Every query returns diagnostics and one of `accepted`, `residual_reject`,
`ambiguous`, `optimizer_unresolved`, or `unsupported_geometry`. It never silently
falls back to an MLP estimate or to the observed start.

Calibration fixes a residual threshold using real queries only:

1. Measure the duplicate/reordered encoder MSE floor ε, bounded below by 1e-12.
2. For each calibration family, take the maximum best-fit residual across its
   real queries. Set τ to the larger of the two scene-specific 95th percentiles
   of these family maxima, with lower bound ε. Freeze its numerical value.
3. A target's best residual must be ≤ τ. Retain every evaluated candidate with
   residual ≤ best residual + τ as a near-optimal candidate.

If any calibration query has no valid finite candidate, threshold calibration
fails and assessment stays closed; do not drop that query from the quantile.

Accept a **block center**, not a unique full physical state, only if all hold:

- Both independent searches terminate successfully; their best block centers
  differ by at most **1.28 units**.
- The retained near-optimal block centers have diameter at most **2.56 units**.
- Nuisance challenges do not find an admissible near-optimal pose outside that
  diameter. Challenge fitted agent position and block angle, then refine block
  position within the reserved evaluation budget. Freeze the offsets and schedule
  before calibration; proposed offsets are ±5.12 units on agent axes and ±10°
  in block angle.
- Renderer identity, geometry and provenance checks pass.

These checks establish empirical search stability only; a finite search cannot
prove there is no missed alternative. Preserve the best pose and residual for
rejected targets for diagnostics, but exclude them from accepted-measurement claims.
No tuning τ or the ambiguity cutoffs after assessment begins.

## 5. Controls before instrument assessment

Run the following with an explicit maximum of **24 additional synthetic targets**.
Use six metadata-selected calibration families, three per scene, with deterministic
pose/mixture seeds. Numerical unit tests are additional and require no model run.
Allocate the 24 targets as six genuine rendered-pose queries, twelve mixtures
(four fixed pairs × three weights), and six perturbed latents. Freeze exact pair
selection and perturbation scales in the implementation config before encoding.

| Control | Required behavior / interpretation |
| --- | --- |
| Renderer round trip | Replay saved real frames bit-for-bit; pose mutation and restoration reproduce the original RGB and leave physics state untouched |
| Genuine rendered poses | Recover nonidentical, valid pose queries not inserted as exact templates or warm starts; include block translations, rotations and agent changes |
| Known ambiguity / constant encoder unit fixture | Return ambiguity when two injected near-optimal poses differ by more than 2.56 units; a flat encoder must not return a confident pose |
| Search failure fixture | Exhausted iterations, inconsistent restarts and invalid geometry produce the declared abstention, with no successful-report overwrite |
| Latent mixtures | Use weights 0.25/0.50/0.75 between known rendered latents. Record pose sets and residuals; do not assign the physical mean as ground truth |
| Perturbed latent targets | Add fixed-seed perturbations scaled to real latent separations. Record acceptance and residuals; a perturbation is not automatically certified off-manifold |
| Analytic curved-manifold fixture | With f(s)=(s,s²), the mean of f(-1) and f(1) can invert to ±sqrt(1/2). Preserve ambiguity; do not label this latent-mean example predictor bias |

An actual latent mixture may lie close to a unique rendered pose. That does not
make the pose the true physical outcome or prove a faulty predictor. Therefore
there is **no blanket “reject all mixtures” gate**. If controls show plausible
false certainty, document the limitation and do not promote inversion into a
physical-truth instrument. The negative fixtures test what the procedure can
reject, not a claim that arbitrary off-manifold targets are detectable.

## 6. Assessment gates and decision

Run the locked instrument once on the 24 assessment families. Thresholds below
are proposed engineering tolerances inherited from the study's 1.28/2.56-unit
measurement scale, not established universal standards. Report uncertainty and
small sample limitations; passing these gates is feasibility only.

| Gate | Required result |
| --- | --- |
| Integrity | All applicable replay/renderer/source/hash/isolation checks pass; any integrity failure stops the task |
| Coverage | At least 10 of 12 families yield accepted measurements in each scene × query cell; report start/end joint coverage separately |
| Block position | Among accepted measurements, mean endpoint error ≤1.28 and family-weighted p95 ≤2.56 units in each scene × query cell |
| Displacement | On jointly accepted start/end measurements, mean vector error ≤2.56 units in each scene × dose cell; report low-motion control separately |
| Relative floor | For moving branches in each non-null scene × dose cell, at least 10 families and mean displacement error / median true displacement <0.25; if insufficient motion, that cell is unassessable |
| Known failure controls | No confident output for the deliberately ambiguous, flat-encoder or failed-search unit fixtures |

The absolute/relative error checks are measurement controls on **real** latents.
They do not validate behavior on predicted latents. Motion selection is used only
for this disclosed relative-floor check, not for choosing families or fitting the
instrument. Report errors of the best searched poses on rejected real queries
separately, so selective coverage cannot hide poor performance.

Aggregate equally over families and bootstrap families (2,000 draws); branches,
starts and synthetic variants are not independent replicates. Keep scene/dose
cells visible. Do not pool away a failed cell. Report position and displacement
errors separately; no subtraction of decoder error from predictor error.

Possible outcomes:

- **Feasible for a named subset:** all gates pass for the stated cells; proceed
  only to the bounded descriptive check below. Failed cells stay ineligible.
- **Not feasible under this budget:** accuracy, coverage or search stability fails;
  save the failure analysis and stop. No automatic larger search, threshold
  relaxation, changed readout or additional training.
- **Unresolved:** resource cap or an integrity problem prevents assessment. State
  the reason; do not count incomplete work as a pass.

## 7. Optional final check on already opened predictions

Only if all four endpoint cells below and both scene-specific start cells pass,
invert **16 saved frozen-predictor
terminal latents**: the first four assessment families per scene, at challenge
0.30/0.60 or archive ×1.0/×2.0. Membership is fixed before calibration. Use the
existing SHIFT-002 NPZ predictions; do not regenerate world-model inference.
Otherwise skip this entire optional check; do not substitute passing cells.

Report acceptance, residual, pose-set diameter and MLP/inversion disagreement on
the exact same targets. Show a few preselected examples including abstentions.
These outputs have no known inverse pose label. Comparing to simulator endpoints
measures a composed predictor-plus-decoder error; it cannot isolate either source.
Do not require agreement with the MLP as a pass condition, select cases by their
agreement, or estimate a new mechanism from this small diagnostic sample.

Even universal agreement would not eliminate uncertainty/nonlinear-encoding
explanations. Fresh-family confirmation, fine-tuned-model comparison and any
mechanism test require a later proposal with a distinct scientific question.

## 8. Resource, implementation and evidence contract

- **Hard cap:** four cumulative CPU machine-hours, four threads, float32 encoding
  and float64 metric reductions. Count smoke tests, calibration, assessment,
  retries, synthetic controls and optional prediction inversion in one ledger.
  No model training or cloud/GPU spend is included.
- Benchmark on the first two calibration families. Estimate the remaining
  frozen workload with **1.5× runtime headroom**. If it exceeds the remaining
  budget, stop and report infeasibility; do not quietly shrink assessment.
  The maximum target count is 256 (216 real + 24 synthetic + 16 optional), or
  524,288 search render/encode evaluations, plus explicitly ledgered integrity
  and duplicate-encoding checks. This is a cap, not a claim about measured runtime.
- Freeze code/config/input hashes, membership, seed streams, exact search rules,
  checkpoint/preprocessing identity and budget before calibration. Seal τ and
  calibrated numerical-floor values before instrument assessment. Check this
  lineage on every resume and before writing a successful report.
- Reuse audited replay/render/encoder components where appropriate; keep new
  reusable logic in `src/jepa_action_effects/`. Implement positive, negative,
  no-label-leak and no-successful-write-on-failure tests. Run `make check` and
  relevant tests. Do not modify the historical experiment source snapshots.
- Proposed outputs: ignored `outputs/meas001/` for per-query search traces, state
  candidates, caches, configuration, seals and ledger; tracked
  `docs/manifests/meas001-runtime.json`, `MEAS-001_RESULT.md`, a runbook, and compact
  preselected examples. Record all attempted queries and rejection reasons.

The final result must answer whether another measurement study is justified,
where the instrument fails, and what it still cannot establish. It must not
declare a world-model failure mechanism or change TRAIN-001's physical verdict.

## Handoff

This document completes the requested proposal. MEAS-001 implementation/execution
remains a separate TODO. An implementing agent should first make this proposal
concrete in code and a reviewable frozen configuration, preserving the existing
uncommitted CALIB work. The experimental stop rules are binding once that run is
frozen; do not repair a failed assessment by reopening its protocol.
