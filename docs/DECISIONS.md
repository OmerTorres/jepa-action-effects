# Decisions and protocol status

## D001 — 2026-09-25 — Development foundation

Use a repository-local uv environment, Python 3.11, src-layout package, Ruff, and
pytest. Lock dependencies and verify native numerical imports and headless plots.
Rationale: a small reproducible base for the two laptops and later Colab work.
Python 3.11 is provisional; revise with evidence if the exact upstream requires it.

## D002 — 2026-09-25 — Defer model and simulator dependencies

Do not select gym-pusht, stable-worldmodel, PyTorch, or a JEPA implementation based
only on names. ENV-001 establishes the exact pair, revision, preprocessing, action
semantics, and rollout API before these enter the environment. No checkpoint or
simulator is currently validated. No model training starts as part of setup.

## D003 — 2026-09-25 — Shared project memory

Keep concise always-on guidance in AGENTS.md and a focused repository-local
`jepa-study` skill for research operations. TASKS.md tracks current state;
LOGBOOK.md preserves evidence and handoffs. RESEARCH_PLAN.md is the supplied
baseline, retained verbatim; later decisions supersede it explicitly.

## D004 — 2026-09-25 — Match LeWM's documented Python series

Supersede D001's provisional Python 3.11 choice with Python 3.10.21. The upstream
LeWM installation uses Python 3.10, and `stable-worldmodel` requires Python 3.10 or
newer; neither establishes that this checkpoint stack works on Python 3.12–3.14.
Use the latest Python 3.10 security release available at setup time and treat any
newer interpreter as a separate compatibility experiment after ENV-001 succeeds.

## D005 — 2026-09-26 — Pin a checkpoint-era simulator candidate

Use `stable-worldmodel` commit
`44c45bd1f774995aac0cc2b41444ebea24c1d5ad` (package 0.0.5) as the audited
PushT candidate for `quentinll/lewm-pusht` revision
`22b330c28c27ead4bfd1888615af1340e3fe9052`. The simulator commit predates the
checkpoint publication and contains the environment/model support referenced by
the checkpoint-era LeWM source. This is a provenance-based candidate choice: the
published checkpoint lacks a training run manifest, simulator commit, dependency
lock, and numeric action-normalization statistics, so it is not proof of the exact
training-time pair.

Keep the dependency source revision locked. Defer the 13.1 GB training archive to
MODEL-001, where it is required to recover action statistics and inspect collection
provenance. Do not run or interpret checkpoint inference using identity scaling or
statistics fitted on this study's trajectories. ENV-001 may close with this gate
explicitly unresolved because its acceptance criterion is the interface audit;
MODEL-001 remains responsible for a strict load and leakage-free inference smoke.

## D006 — 2026-09-26 — Use published evaluation action scaling

Use the checkpoint archive's full action column with the published evaluation
convention (`StandardScaler`, population standard deviation) for frozen inference.
The recovered mean is `[-0.0078125644, 0.0068606872]` and scale is
`[0.2084674428, 0.2067486264]`. Training used PyTorch's sample standard deviation;
the recovered scale difference is below `4.1e-8` per coordinate. The choice was
made before checkpoint predictions were inspected. Normalize each 2-D control
before flattening five consecutive controls into one width-10 model action.

MODEL-001 uses float32 CPU inference through the explicitly mapped pinned March
`JEPA` / `ARPredictor` implementation and Hugging Face ViT construction. MPS,
CUDA, and mixed precision are separate compatibility experiments.

## D007 — 2026-09-26 — SIM-001 replay diagnostic threshold

For the locked Python 3.10.21 / Pymunk 7.3.0 runtime, require duplicate replay,
branch-order replay, and observational instrumentation comparisons to use
`atol=1e-10, rtol=0` for finite numeric state and exact equality for contact
identities/timestamps, flags, and RGB bytes. SIM-001 measured zero maximum numeric
and positional discrepancy across its 16-seed diagnostic, so this threshold passed
without relaxation.

This is a deterministic replay diagnostic for the recorded runtime. It is separate
from the provisional physical-effect tolerance `delta`, does not establish
cross-platform bitwise determinism, and does not freeze the study horizon or effect
threshold. Branch by seeded reset plus complete warm-up replay; state cloning remains
unimplemented because solver caches are not exposed or validated.

## D008 — 2026-09-26 — Provisional evaluator conventions

EVAL-001 evaluates every unordered candidate pair in canonical candidate-ID order.
It labels effects with magnitude at most `delta` as negligible and larger effects
as meaningful. A no-contact branch is stationary when its maximum supplied true
trajectory displacement is at most `delta`. Direction error is unavailable for a
negligible true effect or an exactly zero predicted vector. Exact cost ties resolve
by canonical candidate ID. An informative goal has true candidate-cost spread
strictly greater than `delta`.

These conventions make metric behavior deterministic and keep contact separate
from motion. They are provisional: PILOT-001 must assess coverage and physical
relevance, and FREEZE-001 must confirm or revise them before locked test inference.
The evaluator retains raw records so later aggregation changes do not require
checkpoint inference to be rerun.

## Scientific protocol — NOT FROZEN

The plan's 120 pilot / 900 main families, six candidates, four goals, two native
prediction steps, practical tolerance, readout gate, and 2,000 bootstrap samples
are design defaults, not achieved counts or finalized runtime settings. Choose
the horizon using simulator-only pilot evidence as specified in the plan. Freeze
thresholds, splits, exclusions, sample size, models, and analysis before opening
test results. Record any later changes as exploratory.

## D009 — 2026-09-27 — PILOT-001 bounded protocol and stop decision

The simulator-only pilot used master seed 20260926, 80 geometry-guided contact-boundary
families, 40 outcome-unselected ordinary families, six candidates, four goals from
the fixed `{128,256,384}²` grid, and family-intact 60/30/30 splits. The generator
accepted the first valid families within 800/400 attempt caps. Its challenge
geometry was derived from the configured Pymunk block polygons; acceptance required
a contact/no-contact pair separated by at most 0.2 action units. These rules were
fixed before checkpoint inference.

Keep the two-native-step horizon and provisional `delta=5.12` for this exploratory
pilot: 88/120 simulator families (73.3%) exceeded delta, above the predeclared 20%
four-step trigger. This does not freeze the main-study protocol.

Stop expansion after PILOT-001. The selected ridge readout (alpha 0.1) had 18.42
workspace-unit mean and 42.12 p95 validation block-position error, versus gates of
1.28 and 2.56. A bounded 128-unit nonlinear diagnostic with three fixed seeds also
failed. The decoded checkpoint endpoint results therefore cannot support a physical
reasoning claim, even though replay, contact coverage, runtime, and the earlier
ordinary-transition latent diagnostic passed. Do not start FREEZE-001 or inspect
locked test data without a new, explicitly scoped measurement strategy.

## D010 — 2026-09-27 — READOUT-001 measurement correction and bounded recovery

READOUT-001 retained the pilot's two-step horizon, `delta=5.12`, and real-frame
block-position gates (mean at most 1.28, p95 at most 2.56, with ten families per
required contact regime). It used a new simulator-only seed, 20260927, for 180
additional fit, 60 tuning, and 60 reserved readout-confirmation families. The
original 60 pilot fit families formed the first nested fit tier; all recovery
partitions were disjoint by reset seed and family recipe. No model error influenced
collection.

The audit reproduced the saved pilot ridge validation error and identified no
block-label conflict among identical frames. Family-equal loss weights had been
applied, but feature and nonlinear-target statistics gave repeated frames extra
weight. Correct those statistics in recovery fits; retain the original pilot
artifacts and the before-correction recovery results as separate evidence. Pilot
physical competence is **unassessable** when its ridge readout fails, regardless
of any nonlinear diagnostic: the reported endpoints were decoded with ridge.

No projected ridge or three-seed nonlinear readout passed tuning at 60, 120, or
240 fit families. Keep the 60 confirmation labels reserved, recommend narrowing
or revising measurement, and keep main-study expansion paused. The unprojected
CLS ridge diagnostic was better but also failed, and cannot decode projected
predictor outputs. Probe failure alone is insufficient evidence that block state
is absent from the representation. Historical training-simulator identity remains
unproven.

## D011 — 2026-09-27 — Proposed displacement-measurement follow-up

The user requested a next-step plan after READOUT-001. `DELTA-001_PLAN.md`
specifies an exploratory displacement measurement test using existing 240 fit,
60 tuning, and 60 reserved confirmation families. Start with differences of the
saved projected ridge outputs, then a bounded direct-displacement ridge and,
only if needed, one zero-anchored 128-unit nonlinear diagnostic with three seeds.
Proposed accuracy gates keep the same numerical limits and explicitly cover
meaningful paired effects. Any regret calculation initialized with true current
position must be labeled oracle-initialized.

This records a planning choice only: no displacement result exists, no new
protocol is frozen, and main-study expansion remains paused. Implementation must
seal the concrete protocol before scores and preserve one-time confirmation.

## D012 — 2026-09-27 — DELTA-001 measurement result and stop decision

DELTA-001 executed the ordered, existing-data displacement readouts at the
two-native-step horizon with `delta=5.12`, retaining the numerical mean/p95
limits of 1.28/2.56. The fixed 240 fit and 60 tuning families were used;
no new data were collected. The saved absolute ridge, differenced, reached
9.39 mean endpoint displacement error; direct displacement ridge reached
4.83; the three zero-anchored nonlinear seeds reached 4.29, 4.39, and 4.01.
Meaningful paired-effect mean errors were 13.33, 8.54, 8.79, 8.75, and 8.25,
respectively. All exceeded the predeclared gate, including their tail errors.

No model was selected. The reserved 60 confirmation families were not used for
readout evaluation, and the conditional frozen-predictor comparison did not
run. Close this bounded physical-measurement recovery route and keep
FREEZE-001 paused. The readouts improve over zero displacement but cannot
support fine-grained physical claims. A failed probe does not establish absent
block information or frozen-predictor physics failure. Historical
training-simulator identity remains unproven.

## D013 — 2026-09-27 — Literature review and proposed question revision

LIT-001_REVIEW.md verifies the closest-work scope and records exact definitions
where recovered, with explicit omissions otherwise. Action sensitivity,
counterfactual controls, paired contexts, and regret are established precedents;
claim no general methodological novelty. Published LeWM probe and planning
results do not certify our physical precision gate on challenge families.

The user authorized the literature/checkpoint review and revised question.
LATENT-001_PLAN.md proposes measuring matched-versus-swapped real-future latent
error and improvement over persistence on the existing 60 READOUT tuning
families. This is exploratory reuse with explicit agent-motion confounding,
family-level uncertainty, and a proposed 30-minute execution cap. It is not a
frozen or executed protocol. Keep readout-confirmation families reserved and
main physical-study expansion paused; preserve D009–D012 and all original results.

## D014 — 2026-09-27 — LATENT-001 protocol and outcome

Before scoring, seal the 60 READOUT tuning-family IDs and hashes, frozen
checkpoint identity, two-step horizon, raw projected-CLS metric, bootstrap and
verification/example seeds, and 30-minute cumulative cap under
`outputs/latent001/config.json`. Use real encoded futures only as evaluation
targets. Compare correct-action matching to the exact mean over wrong actions
and separately to latent persistence; preserve family units in intervals.

The completed run found positive correspondence advantage A=0.43975
[0.40560, 0.47353] and persistence improvement B=0.52431
[0.49227, 0.55754] overall, with positive intervals in both cohorts.
Preselected duplicate/reversed-order checks passed. These exploratory
representation-space findings cannot repair the failed physical readout gate.
Keep the 60 confirmation families reserved, main-study expansion paused, and
close experimentation for this course-scale report. Historical
training-simulator identity remains unproven.

## D015 — 2026-09-28 — Reopen measurement design, provisionally

The user raised a substantive concern that LATENT-001's global latent result
does not adequately validate novelty or block-specific contribution. The
targeted NOVELTY-002 review found AD-WM, a close September 2026 precedent
using LeWM and PushT for action-discriminative world-model analysis. Do not
claim novelty from action correspondence, candidate ranking, or regret alone.

The existing 60 tuning families have no pair with block separation >5.12 and
agent endpoint separation <=20.48, so simple agent matching cannot resolve
the confound. Consider a fixed-agent, renderer-based block-pose gallery as a
new measurement hypothesis. This is a design decision only: renderer identity,
valid pose coverage, real-frame margin, and literature distinction remain
unverified. Keep FREEZE-001 paused and confirmation closed. D014's completed
measurements and evidence remain unchanged.

## D016 — 2026-09-28 — Renderer manipulation order and gallery limit

For a render-only counterfactual, replay the exact branch to its terminal
state. When changing the T-block pose, set angle before position and reindex
its shapes; restore in the same order. The block's offset center of mass makes
position-then-angle assignment shift its physical location. This procedure
reproduced all 72 saved endpoint frames after mutation and restoration in the
preselected LATENT-001 verification set. Borrowing the next branch's block pose
changed 32/36 challenge images but 0/36 ordinary images. Treat renderer
identity as preliminarily passed for replayed endpoints; geometry-valid
alternatives, projected-CLS discriminability, and model contribution remain
open. No predictor or reserved confirmation data were used.

## D017 — 2026-09-28 — Proposed nonidentical-query gallery feasibility

GALLERY-001_PLAN.md defines the next planning artifact at the user's request.
Exact true-frame retrieval from a bank containing that same frame is an
identity/collapse check, not evidence of a useful block representation. The
proposed feasibility gate therefore uses nonidentical 1.28-unit block-position
queries against fixed-agent galleries with 10.24- and 20.48-unit negatives.
It limits execution to 30 existing tuning families, a 30-minute cumulative
runtime, and simulator/encoder work. Geometry coverage, retrieval stability,
and contribution/precedent review precede any predictor study. Numerical gates
are proposed engineering criteria and do not replace the original physical
readout thresholds. No configuration is frozen and no new experiment was run
in this planning session; confirmation remains closed.

## D018 — 2026-09-28 — GALLERY-001 measured outcome and claim boundary

The sealed 30-family simulator/encoder audit completed in 46.79 cumulative
seconds. Exact replay/restoration and duplicate/order checks passed. Both
predeclared translation scales met geometry coverage, nonidentical-query
retrieval and margin gates in challenge and ordinary cohorts; 179/180 and
180/180 candidate galleries were eligible at 10.24 and 20.48 units.
GALLERY-001_RELATED_WORK.md finds direct precedents for rendered feature-space
pose retrieval, so general method novelty is unsupported. The resulting
empirical opportunity is a checkpoint-specific comparison of frozen predicted
latents against these controlled block-pose galleries. No predictor score was
measured in GALLERY-001. A subsequent comparison needs a separate sealed
protocol, persistence and action-permutation controls, and genuinely reserved
evidence after tuning-family design choices. Keep original physical readout
failures, main-study pause, confirmation closure, and historical simulator
identity limitation intact.

## D019 — 2026-09-28 — PREDICTOR-GALLERY-001 frozen protocol and outcome

The separate predictor-to-gallery protocol used the existing 30 GALLERY-001
tuning families and one fixed evaluation on all 60 READOUT reserved families,
irrespective of the development predictor result. The code/config/input hashes,
primary 10.24-unit gallery, controls, gates, seed streams and central comparison
were sealed before opening confirmation. The experiment retained the frozen
two-step checkpoint and simulator, family-level bootstrap, and the validated
nonidentical-query measurement gate. This protocol was exploratory with a
genuinely reserved within-project confirmation set, not the main study's
locked test.

Confirmation completed all 60 families. The primary encoder/geometry gate
passed. On 40 challenge families, matched predictions exceeded wrong-action
predictions in gallery margin by 0.04272 (95% family bootstrap interval
[0.03287, 0.05159]), but exceeded persistence by only 0.00131
[-0.00807, 0.00959]; mean matched margin was -0.00923
[-0.01927, -0.00005]. Thus the prespecified conjunction for positive evidence
failed. This supports a narrow checkpoint-specific finding of action-sensitive
gallery correspondence without demonstrated superiority to persistence at the
primary block-position scale. It does not prove information loss or continuous
physical incompetence. The 60 confirmation families are now opened and cannot
serve as fresh reserved evidence for a revised question. Keep main-study
expansion paused and the earlier failed physical readout gates explicit.

## D020 — 2026-09-28 — Proposed selective-erasure feasibility boundary

At the user's request for next steps after PROBING-001, ERASURE-001_PLAN.md
specifies an existing-data test of agent/block linear concept erasure using LEACE.
Fit-only covariance estimates, refitted probes, calibration-only selection and
separate assessment are required before interpreting an intervention. The new
gain/preservation criteria are proposed engineering gates, not substitutes for
historical physical-readout thresholds. The one-hour study excludes predictor
behavior experiments, new collection and the already-open confirmation cohort.
Nonlinear recoverability must remain explicit; removing linear accessibility
does not mean removing the concept. This is a planning decision, not a frozen
execution protocol or observed outcome. No automatic method/data expansion follows
a failed or inconclusive feasibility check. Historical evidence is unchanged.

## D021 — 2026-09-28 — Proposed floor-relative physical criterion; defer erasure

This follows REVIEW-EXT-001. NEXT-001_PLAN.md proposes to re-measure H1/H2 with a
readout trained on 60k frames drawn from seeded simulator resets. The old absolute
real-frame gate (mean ≤ 1.28, p95 ≤ 2.56) is replaced by a floor-relative
criterion. Real and predicted futures of the same branches are decoded with the
same frozen readout, and displacements are measured relative to `readout(z_t)`.

- A predictor physical failure is claimed only if both hold:
  - the lower bound of the 95% family-bootstrap CI of
    `err_pred − err_real` is above 0;
  - the real-frame mean displacement error is below 25% of the median meaningful
    effect.
- Shrinkage is claimed if the upper bound of the displacement-slope CI is below 1.
- Otherwise the result is inconclusive.

Confirmation uses 150 fresh families collected under a new master seed. The 60
opened families are never used again. ERASURE-001 is deferred until PHYS-002
establishes a physical effect worth explaining. This is a proposal: the criteria
must be sealed in code before the reserved families are opened. D009–D020 and
their evidence are unchanged.

### D021 amendment — 2026-09-28 — implementation details before any execution

This amendment was written before any DIST/READOUT-002/PHYS-002 execution or seal.
The implemented rules are in `next_metrics.apply_criteria`:

- **Shrinkage** also requires the upper CI of `slope_pred − slope_real` to be below
  0. A ridge readout can attenuate displacements by itself, so predictor shrinkage
  is judged relative to the same readout on real futures.
- **The floor ratio** is the family-mean real-frame effect error on meaningful
  designed boundary pairs, divided by the median true effect on those pairs.
- **Boundary pairs** have a per-control action distance of at most 0.2 and exclude
  `hold`.
- **Reserved families:** 150 families (100 challenge, 50 ordinary) under master seed
  20261010, disjoint by reset seed from all pilot/READOUT families.
- **Optional variant:** 60 in-support challenge families, reported separately. It
  is used only if DIST-001's archive-only rule in `NEXT-001_RUNBOOK.md` triggers.

PHYS-002 `seal` freezes all of this with the source hashes.

## D022 — 2026-09-28 — PHYS-002 outcome, audit reframing, next experiments

**Outcome under the sealed D021 rules.** On the 100 reserved challenge families:

- the floor passed (ratio 0.120);
- physical failure is supported (D_boundary 17.0 [15.8, 18.1]);
- shrinkage is supported (slope_pred 0.561 vs slope_real 1.006; difference −0.445
  [−0.490, −0.401]);
- H1 is not established (h1_contrast −12.4 [−14.3, −10.5]).

The ordinary cohort failed the floor (0.35) and is inconclusive. These verdicts are
final for PHYS-002.

**Reframing from the exploratory audit.** No data, alignment or probe-size error was
found (`phys002_audit.py`, `phys002-audit-runtime.json`). On in-distribution archive
windows the same predictor and probe show no shrinkage (slope 0.96, and 1.02 when the
block starts at rest). PHYS-002 is therefore reported as a property of the designed
challenge distribution, not of the checkpoint in general. At matched effect size
(development, exploratory), the error is about 60% of the effect in every contact
regime, so the H1 reversal reflects effect size.

**Protocol gap.** DIST-001's in-support rule tested action magnitude only. The study
controls are constant within each 5-control block, and no archive block is; the
sealed variant did not cover this temporal-structure shift.

**Next.**

- ERASURE-001 stays deferred.
- SHIFT-001 (states × action structure) and H3-PLAN-001 (regret under the model's
  latent planning cost) are proposed. Their rules will be sealed in code before any
  fresh family is opened and recorded as D023.

D009–D021 and their evidence are unchanged.

## D023 — 2026-09-28 — SHIFT-001 stopped at development; SHIFT-002 dose-response protocol

**SHIFT-001.**

- Development failed two of its sealed-in-code manipulation checks:
  - the push-shrink bound (upper 0.807 vs 0.8);
  - the effect-floor ratio (0.36 challenge, 0.54–0.56 archive).
- `seal` refused, and its 200 reserved families are left unopened and will not be
  reused for confirmation.
- Its development result (no within-block-structure effect; a large push-magnitude
  effect) is exploratory.

**SHIFT-002 (new version, fresh families; `shift002.py`, `shift002_metrics.py`).**

- **Candidates.**
  - Challenge geometry: constant pushes of 0.15, 0.30, 0.45 and 0.60, plus varying
    (archive-residual) versions at 0.30 and 0.60.
  - Archive rest windows: own-future controls scaled 1.0, 1.5 and 2.0, each constant
    and varying. Own-future controls must be ≤ 0.5 so the scaled versions stay in
    bounds.
- **Seeds and exclusions.** Master seed 20261040. Seeds and archive episodes are
  disjoint from DIST-001, the PHYS-002 audits and SHIFT-001.
- **Sizes.** 60 + 60 development and 120 + 120 reserved families.
- **Primary quantities** (R = pred/real displacement slope on moved candidates):
  - dM_challenge = R(0.15, 0.30 constant) − R(0.60 constant);
  - dM_archive = R(×1.0) − R(×2.0);
  - structure contrasts dS at matched magnitude, judged by ±0.1 equivalence.
- **Readout gate.** The effect-floor ratio is replaced by slope_real's 95% CI lying
  within (0.85, 1.15) in every state × structure cell. Rationale: R compares predicted
  and real-frame decodes against the same truth, and zero-mean readout noise does not
  bias either inner product. What would bias R is readout attenuation, which this gate
  tests.
- **Other manipulation checks.** R(0.60 constant) CI upper bound < 1; R(archive ×1.0
  varying) faithful (CI includes 1 or lower bound > 0.9); ≥ 30 moved families per gate
  cell.

The gate choices were informed by SHIFT-001 development. They are sealed before any
SHIFT-002 family is scored and apply only to SHIFT-002's fresh families.

## D024 — 2026-09-28 — H3-PLAN-001 protocol (regret under the model's latent planning cost)

**Code.** `planning_regret.py` and `planning_metrics.py`.

**Setup.**

- For each family and each of its 4 grid goals, render a goal image with the
  render-only pose setter: block at the goal, start block angle, agent at its start
  position. Goals whose pose overlaps the agent, a wall or the image border are
  excluded, by geometry only.
- Candidates are ranked by the L2 distance between their final projected latent and
  the goal image's latent.
- Simulator regret is ‖q_end(â) − g‖ − min_a ‖q_end(a) − g‖.
- Choosers: predicted latents (`jepa`), encoded real future frames (`real_latent`,
  i.e. a perfect predictor under the same cost), and uniform random.
- A goal is informative when simulator costs differ by more than 5.12.

**Sealed rule (pooled reserved families).**

- Checks: real-latent regret must beat random (upper CI bound of
  real_latent − random < 0), and there must be ≥ 50 families.
- H3 is supported if the lower CI bound of jepa − real_latent is > 0, i.e.
  prediction error adds regret under the model's own objective.

**Data.** Development uses the 300 PHYS-002 development families. Confirmation uses
100 challenge + 50 ordinary fresh families under master seed 20261030, with every
earlier reset seed forbidden.

## D025 — 2026-09-29 — H3-PLAN-001 stopped; H3-PLAN-002 agent-matched goal images

**H3-PLAN-001.**

- Development failed its sealed check: real-latent regret was 20.9 vs random 17.7.
- Cause: the start-pose agent in the goal image made the latent cost reward `hold`
  (59% of challenge goals).
- `seal` refused; the 150 H3-PLAN-001 reserved families stay unopened and unused.

**H3-PLAN-002** (Omer's choice, option (a); `planning002.py`, `planning002_metrics.py`).

- **Goal images.** Each candidate gets its own goal image: the agent at that
  candidate's realized final position, the block at the grid goal with that
  candidate's realized final angle. A goal counts only if all six images are
  geometry-valid.
- **Cost and choosers.** Candidates are ranked by ‖z_final − e(own goal image)‖;
  choosers, regret and informative-goal definition are as in D024.
- **Sealed rule.** Unchanged from D024 (`planning_metrics.apply_planning_criteria`):
  - the check requires real-latent regret below random, and ≥ 50 families;
  - H3 is supported if the lower CI bound of jepa − real_latent is > 0 on pooled
    reserved families.
- **Data.** Development on the 300 PHYS-002 development families. Confirmation on
  100 challenge + 50 ordinary fresh families under master seed 20261050, with every
  earlier reset seed forbidden.

**Limitation.** The goal images use the realized final agent pose and block angle,
which a real planner would not have. H3-PLAN-002 is a controlled measurement of
whether block-prediction error costs reward under a latent cost, not an evaluation of
LeWM's deployable planner.

## D026 — 2026-09-29 — Dispositions of REVIEW-BRANCH-002 findings

All eight findings in `BRANCH_REVIEW_2026-09-29.md` are accepted. No sealed verdict,
seal-hashed source file or saved output is changed. Corrections are appended as
dated sections in each result document.

1. **Confirm does not enforce frozen inputs.** Accepted.
   - The completed runs were checked by the review (hashes matched), so this is an
     enforcement defect, not evidence of changed inputs.
   - Fixed for future runs by `provenance.py` (REVIEW-FIX-002). Existing sealed
     runners are historical and not retrofitted, because editing them would break
     their source seals.
2. **The SHIFT-002 readout gate pools doses.** Accepted. The claim is qualified in
   the SHIFT-002 corrections. A per-dose × structure gate is added for future
   protocols in `protocol_gates.py`. The D023 text stays as historical.
3. **The H3 narrative overclaims.** Accepted. "Clearly better than random" and "at
   most a few units" are withdrawn. The reading is: not established, and a cost above
   tolerance is not excluded.
4. **"Endpoint" metrics are displacement errors.** Accepted.
   - `endpoint_err_*` in `next_metrics.py` and the PHYS-002 manifests means
     displacement error.
   - The absolute endpoint error is reported separately (review: 17.895 / 3.613
     reserved challenge; recomputed in REVIEW-FIX-001).
   - `next_metrics.py` is seal-hashed and is not renamed.
5. **Agent-matched goals do not isolate block error.** Accepted. H3-PLAN-002 measures
   total latent prediction error under a privileged, candidate-specific goal cost. The
   `planning002_metrics.py` docstring ("compares block placement") is superseded by
   this record, because that file is seal-hashed.
6. **The archive audit bootstraps dependent windows.** Accepted. An episode-clustered
   interval is recomputed in REVIEW-FIX-001; the audit module and its manifest stay
   unchanged.
7. **The faithfulness rule has no upper bound.** Accepted. A two-sided rule with a
   precision requirement is added in `protocol_gates.py` for future protocols; the
   D023/SHIFT-001 rules stay as historical.
8. **The runbook prefix fails in zsh.** Accepted and fixed: `run_stage` shell
   function, verified in `zsh -f`.

Also corrected by record:

- the PHYS-002 "56% as far" and "not a probe or data artifact" wording;
- the SHIFT-002 "noise, not attenuation" wording;
- the H3-PLAN-002 "decision cost is small" statement in STUDY_SYNTHESIS.

## D027 — 2026-09-29 — Dispositions of REVIEW-BRANCH-003 findings

All three findings in `REVIEW_FOLLOWUP_AUDIT.md` are accepted. No sealed verdict or
seal-hashed source is changed.

1. **Reversed readout-bias explanation.** Accepted. A correction section is prepended
   to REVIEW_FOLLOWUP_RESULT.md:
   - the challenge readout drift opposes the decline in R;
   - the archive readout drift contributes to it;
   - raw predictor slopes decline in both.

   The length, direction and slope summaries are not an exact multiplicative
   decomposition.
2. **Incomplete provenance safeguard.** Accepted. `provenance.validate_run` now:
   - hashes the selected probe artifact against `path_sha256`;
   - checks each cached record's fingerprint against the development fingerprint;
   - has a strict mode (default; new runs must record `input_sha256` and
     `fingerprint` per record) and an explicit legacy mode for completed runs.

   REVIEW-FIX-002 is a validator foundation: it is not yet wired into a runner, and
   that integration remains until a new confirmation is scheduled.
3. **Spurious direction for zero predictions.** Accepted. `decompose` excludes zero
   predicted vectors from the angle statistics, returns no angle when none is
   defined, and reports `undefined_angle_count`. The saved data contain none.

## D028 — 2026-09-30 — TRAIN-001 protocol and pre-data amendments

**Protocol.** `docs/TRAIN-001_PLAN.md` (the supplied plan, saved verbatim) is the
frozen protocol for TRAIN-001. It compares the frozen LeWM checkpoint with two
fine-tuned copies trained on identical transitions: archive-weighted and
magnitude-balanced. Implementation: `train001*.py`. The rules below were fixed on
2026-09-30, before any TRAIN-001 family was collected. They adopt Omer's decisions
and all six corrections in `TRAIN-001_IMPLEMENTATION_PLAN_REVIEW.md`. The sealed
`config.json` is generated from `train001_protocol.Protocol`.

**Measurement-gate policy (replaces the plan's literal stop rule for measurement).**

| Development outcome | Action |
| --- | --- |
| Implementation, optimization, data-integrity or budget failure | Stop dependent stages; report it |
| All four primary measurement cells pass | Train; run fresh confirmation of the pooled physical comparison |
| Any primary measurement cell fails | Continue training and fresh latent evaluation. The pooled physical hypothesis is **not assessable under the measurement protocol** |

> Development measurement-gate failure does not stop TRAIN-001 training or fresh latent
> evaluation. It makes the pooled physical hypothesis ineligible for confirmation; retain
> latent outcomes as secondary and physical subgroup comparisons as exploratory. Preserve
> thresholds, cohorts and readouts. Stop dependent execution only for implementation,
> optimization, data-integrity or budget failures. Report all gate outcomes explicitly,
> and do not promote a secondary result to replace the primary hypothesis.

**Safeguards.**
- The 25% floor, magnitudes, cohorts, readout and sample sizes are never adjusted to
  make a gate pass.
- Development gates use real-frame controls only. They are sealed in `eligibility.json`
  before training.
- A development failure cannot be reversed by confirmation. A development pass
  followed by a confirmation failure makes the conclusion inconclusive.
- Passing-cell physical results are exploratory subgroups.
- Readout repair would need a new protocol and freeze.
- A measurement failure is not evidence that fine-tuning has no effect.

**Gate definitions.**
- Cells: scene × {0.45, 0.60}.
- Moved: true two-step block displacement > 5.12.
- A cell passes only if all three hold:
  - at least 50 **observed** moved families in the cohort (60 per scene at
    development, 120 at confirmation);
  - the real-frame projection-slope CI on moved families lies strictly inside
    (0.85, 1.15), using 2,000 draws with seed 20261121;
  - floor < 0.25, where floor = mean real-frame displacement error over **all accepted
    families** in the cell ÷ median true displacement over **moved** families.
- Missing or degenerate statistics fail.
- Eligibility uses the selected READOUT-002 probe, `mlp-20261005`.

**Verdicts.** The pooled primary has five outcomes:
- development failure: `not_assessable_under_measurement_protocol`;
- confirmation failure: `inconclusive_measurement_invalid_at_confirmation`;
- both balanced − archive and balanced − frozen CIs below 0: `balanced_benefit_supported`;
- a required CI above 0: `balanced_error_increase_supported` (the contrast is named);
- otherwise: `physical_benefit_not_established`.

Ordinary windows use the **new 1.28-unit tolerance** on balanced − frozen endpoint error.
It is not a retroactive change to earlier gates.
- upper CI < 1.28: non-inferiority established;
- lower CI > 1.28: regression supported;
- otherwise: not established, which is not proof of regression.

Terminal step-2 latent MSE on the 0.45/0.60 branches is the prespecified secondary
outcome. It carries the same contrasts and is labelled representation prediction.

**Admission vs integrity.**
- A proposal that fails a rule is **rejected and counted**. Rules: forbidden or used
  seed/episode, duplicate recipe/content, out of bounds, not at rest, near-zero
  direction, identity failure, incomplete branch, history mismatch, movement.
- These are **fatal**:
  - an attempt cap is reached (challenge 20,000; archive 60,000; ordinary 5,000);
  - a weight bin has fewer than 100 windows;
  - an admitted artifact violates identity, split or bytes;
  - the provenance validator fails.
- Identities are source-specific:
  - challenge families: reset seed + recipe hash;
  - archive families and windows: episode + window + recipe hash, because every archive
    replay uses simulator seed 0.
- Content hashes are also recorded.
- Exclusions come from every earlier study: 1,210 reset seeds, 2,014 archive episodes
  (including MODEL-001 episodes 0–31) and 1,549 history-frame hashes. The
  TRAIN-001 smoke run's families are excluded as well.

**Archive-derived families.**
- SHIFT-002's rest filter is kept (block moves ≤ 1.0 over the recorded history), so
  the population is "archive rest windows".
- The direction is the normalized mean of the recorded controls `t..t+9`.
- Means with norm < 0.05 are rejected. This is ¼ of the per-coordinate control SD and ⅓
  of the smallest study magnitude; below it the recorded future has no consistent
  direction. Its rejection count is reported.
- Identity is checked with a verification-only recorded-future branch, which is never
  trained on or evaluated. The SHIFT-002 tolerances apply in their original units:
  - history positions ≤ 0.5;
  - **raw uint8** frame MAE ≤ 1.0;
  - recorded-future endpoint ≤ 1.0.

**Weights.**
- 1,000 episodes, disjoint from every other pool.
- Non-overlapping 10-control windows from the episode start. The statistic is the mean
  per-control norm, binned `<0.225`, `[0.225,0.375)`, `[0.375,0.525)`, `≥0.525`, in
  magnitude order.
- The balanced condition uses 0.25 each.

**Training details.**
- CPU only, 4 threads, float32. No accelerator is available, so no fallback is promised.
- Whole model in `eval()`, with only `predictor.train()`. Dropout is active, and both
  BatchNorm layers keep frozen statistics. `pred_proj` weights and BN affine parameters
  train.
- The two-step rollout feeds back the first prediction.
- Every trial and run reloads the original checkpoint (hash-checked by state digest)
  with a fresh AdamW optimizer.
- Each scene has its own seeded per-epoch permutation sampler, shared by both arms of a
  seed. At an epoch wrap, families already in the partial batch move to the end of the
  new permutation.
- Resume states every 50 updates hold the optimizer, RNG and sampler. Checkpoints are
  written at 500/1000/1500/2000; the final one is primary.
- Overfit gate: eval-mode loss on 16 families at LR 1e-4, measured every 50 updates;
  `1 − min/L0 ≥ 0.5` within 500 updates.
- LR tie rule: within 1e-12 relative, the lower rate wins.
- Rollout equivalence tolerance: 1e-5, frozen model, eval mode. Measured 1.3e-6 on a
  synthetic check.

**Calibration comparator.** One nonnegative scalar per readout, fitted on the
600 training families by closed-form least squares. Weights: equal per scene, family
and branch. It is labelled a state-supervised output correction.

**Provenance.** `train001_provenance.validate` has two modes:
- `entry`, used at the first `confirm` and every resume: all sealed inputs and the
  complete registry are valid, and records form a valid subset;
- `complete`, used for the summary and report: exact coverage.

In both modes:
- Records are bound to their model by a fingerprint that includes the checkpoint SHA,
  condition, seed, inference source, device and thread count.
- Foreign, stale, corrupt, swapped or stray records fail.
- The sealed config must equal the code-generated protocol.

**Budget.**
- `outputs/train001-budget/ledger.json` is cumulative across run identities.
- Heartbeats let crashed sessions be charged up to their last heartbeat plus 60 s.
- An exclusive lock prevents concurrent stages.
- Budget checks run per unit (proposal, encoding, update, evaluation), with a 5-minute
  shutdown allowance. A watchdog interrupts at the hard deadline.
- Implementation, tests and the smoke run are not charged.

**Other seeds.** Collection master seed 20261100. Bootstrap seeds 20261121–20261123;
derangement seed 20261124.

**Scope boundary.** `train001_report.py` only formats sealed results and is outside the
sealed sources. D009–D027, their evidence and completed runners are unchanged.

### D028 outcome — 2026-09-30 — TRAIN-001 verdicts under the sealed rules

All stop gates passed: implementation, equivalence (3.1e-6), overfit (99.6%), data
integrity and budget (1.06 h used). LR 1e-5 was selected.

**Development eligibility (sealed before training):**
- `not_assessable_under_measurement_protocol`;
- the archive floors were 0.278 and 0.304; the challenge cells passed.

**Confirmation (240 fresh families, 120 windows):**
- All four measurement cells passed (archive floors 0.156 and 0.149). This does not
  reverse the development failure.
- Pooled verdict: `not_assessable_under_measurement_protocol`.
- Measured balanced − archive-weighted: −0.09 [−0.53, 0.38]. It spans zero, so no
  benefit would have been established regardless.
- Balanced − frozen: −11.66 [−14.29, −9.11]. Reported, not claimed.
- Secondary terminal latent MSE, balanced − archive: −0.0070 [−0.0127, −0.0014].
- Ordinary balanced − frozen: +1.05 [0.11, 1.99], so non-inferiority is not established.
- The verdict does not change with either alternative readout.

TRAIN-001's sealed verdicts are final. Any readout repair or generalization test needs
a new protocol and fresh families.

## D029 — 2026-09-30 — Dispositions of TRAIN-REVIEW-002 findings

All four findings in `TRAIN-001_REVIEW.md` are accepted. No sealed TRAIN-001 artifact,
verdict, seal-hashed source or number is changed; the six models are not retrained.

1. **Physical conclusion too strong.** Accepted.
   - "Magnitude weighting did not matter physically" and the 2%-weight sufficiency
     remark are withdrawn.
   - Replacement: *no pooled balanced-over-archive physical benefit was established*;
     the descriptive decoded contrast was near zero. No equivalence margin was
     specified.
   - Both fine-tuned conditions reduced decoded errors on the sampled constant-push
     distribution. The cause (exposure to high-magnitude pushes) is a hypothesis,
     because there was no dose-omission or archive-only control.
   - Physical figures and tables are now labelled descriptive.
2. **Sensitivity flag uninformative.** Accepted. The sealed `measurement_dependent` flag
   compares verdict strings. Eligibility uses the primary readout, so after a
   development failure the flag is false by construction.
   - It is superseded by `train001_followup.contrast_sensitivity`. That function compares
     each contrast's interval sign, the ordinary outcome and the gates across readouts.
   - Production result: all agree except ordinary balanced − frozen. That interval is
     above zero for `mlp-20261005` and `mlp-20261006`, but spans zero for
     `mlp-20261004` [−0.149, 1.822].
   - The flag in `train001-runtime.json` and `confirm/summary.json` is historical.
3. **Training-lineage gap in the validator.** Accepted.
   - `train001_followup.validate_lineage` checks each trained model's recorded
     collection/cache hashes against the seal and files, its loss weights against its
     condition, and its optimization specification against the protocol.
   - `validate_with_lineage` combines it with the sealed validator. Future TRAIN runners
     must use it before seal and confirm.
   - Applied read-only to production: 58/58 checks passed, including all six lineage
     checks. Fixtures show the sealed validator alone admitting stale lineage.
   - `train001_provenance.py` stays unchanged because it is seal-hashed.
4. **Latent interval mis-transcribed.** Accepted.
   - Sealed value: −0.0068376941 [−0.0127569325, −0.0011002090], reported as −0.00684
     [−0.01276, −0.00110]. The D028 outcome's "−0.0070 [−0.0127, −0.0014]" is superseded
     by this record.
   - Cited numbers now come from `train001-followup.json` (`exact_numbers`).

**Also disclosed.** D028 left the 105 s real-data smoke run outside the production
ledger. It is reported separately; the 1.06 h figure covers production stages only.

## D030 — 2026-10-01 — CALIB-001 calibration-side reanalysis (definitions before values)

Exploratory. Every family below is already opened, so nothing here can confirm a claim.
No sealed artifact, verdict or number changes, and no model is re-run. These definitions
were fixed before any calibration value was computed.

**Question.** Is the frozen predictor's decoded displacement shrinkage consistent with a
calibrated conditional-mean predictor under outcome uncertainty (LIT-002 §3.1)? Or do its
predictions fall short even of its own information?

**Data.**
- PHYS-002 development and reserved (challenge and ordinary cohorts).
- SHIFT-002 development and reserved.
- TRAIN-001 confirmation families and ordinary windows (frozen, archive-weighted and
  balanced models).

**Vectors.** As in the sealed analyses, at the two-step horizon:
- p = decoded predicted end − decoded start;
- r = decoded real end − decoded start;
- t = simulator true end − true start.

**Cells.** Design cells only: dataset, phase, scene or state, dose, structure, condition
and readout. All candidates in a cell are used, because selecting moved candidates
conditions on the outcome and biases the calibration slope. Moved-only statistics are
computed solely to reproduce sealed values.

**Estimands.**
- Projection slope b_rev = Σp·t / Σt·t.
- Calibration slope b_cal,k = Σp·t / (Σp·p − k·Σ|r − t|²), for k ∈ {0, 0.5, 1, 2}. k = 1 is
  the reference: it assumes the readout is as noisy on predicted latents as on real ones.
- R²_k = (Σp·t)² / ((Σp·p − k·Σ|r − t|²)·Σt·t), so that b_rev = R²_k / b_cal,k.
- Readout-free latent slopes (PHYS-002 and SHIFT-002 only), in the 192-d projected space
  at steps 1 and 2: Δẑ = ẑ − z_t and Δz = z − z_t.
- Static excess: (Σ|p|² − Σ|r|²)/n over candidates whose block did not move.

**Readouts.** Primary `mlp-20261005`. Sensitivity readouts are `mlp-20261004` and
`mlp-20261006`, plus the READOUT-002 ridge readout for PHYS-002 and SHIFT-002, where
latents are stored.

**Uncertainty.**
- Family bootstrap with 2,000 draws and percentile 95% intervals. Ordinary windows are
  resampled individually.
- There is one joint resampling per group, so dose and condition contrasts are paired.
- Fine-tuned seeds are pooled within families.

**Classification (k = 1).**
- Interval above 1: *under-prediction beyond calibrated hedging*.
- Interval below 1: *over-prediction*.
- Interval containing 1: *consistent with calibration*.
- *Undefined* if the corrected denominator is not positive.
- An above-1 call is *robust to the noise assumption* if the k = 0 interval is also above
  1, because under classical noise k = 0 is a lower bound.

**Dose-trend contrasts.** b_cal,1 at the highest dose minus b_cal,1 at the lowest dose that
moves blocks:
- TRAIN-001: 0.60 − 0.30, per scene and condition;
- SHIFT-002 challenge: m0.60 − m0.30 (constant);
- SHIFT-002 archive: ×2.0 − ×1.0 (structures pooled).

**Readings** (these are hypotheses, not claims):
- Calibrated hedging: b_cal,1 ≈ 1 at every dose while b_rev falls.
- Pure attenuation: b_cal,1 > 1 wherever b_rev < 1, and never below 1.
- Reversion toward a typical transition (LIT-002 §2): b_cal,1 below 1 at low doses and
  above 1 at high doses, a positive trend.

**Gate before interpretation.** The run must reproduce, to 1e-9, the sealed:
- PHYS-002 reserved projection slopes;
- SHIFT-002 reserved pred/real ratios;
- TRAIN-001 frozen pred/real ratios (primary readout).

**Limits.**
- Decoded calibration assumes the readout is valid on predicted latents.
- Latent calibration is readout-free but not block-specific.
- The noise correction assumes equal readout noise on predicted and real latents.
- This analysis cannot establish a mechanism.

### D030 amendment — 2026-10-01 — degenerate cells (after the first run)

The first run showed that challenge 0.15 pushes never move the block, so t ≡ 0 and
Σp·t = 0 exactly. The calibration slope was therefore 0 by construction, and the table
labelled those cells "below 1".
- **Change:** a calibration slope now also requires Σt·t > 0. Cells without block motion
  are *undefined*, and their spurious predicted motion is reported through the static
  excess, which D030 already defined.
- **Scope:** no other cell, estimand or interval changes. This amendment was made after
  seeing results and is recorded as such.
- **Not added:** an agent-displacement calibration was considered and deliberately not
  run. The READOUT-002 agent head is weak (study-check mean error 20.4 units, R² 0.86),
  and the analysis was not prespecified. It is listed as a next step.

## D031 — 2026-10-01 — Dispositions of CALIB-REVIEW-001 findings

All four findings in `CALIB-001_REVIEW.md` are accepted. No historical CALIB-001 number
changes; a rerun reproduced all 13,768 earlier output values exactly and added estimands
only. D030 stays as written; this entry qualifies how its outputs are read.

1. **Latent versus physical conditional means.** Accepted. b_cal = 1 is necessary, not
   sufficient, for p = E[t | inputs] in physical units. Because LeWM predicts latents and
   the readout is nonlinear, b_cal ≠ 1 does not distinguish predictor bias from latent
   uncertainty through decoder geometry, or from readout error on predicted latents.
   "Honest hedging does not explain the shrinkage" is withdrawn. Replacement: *decoded
   displacements depart from the scalar-calibration condition*. "Fine-tuned models are
   calibrated" becomes "their scalar gains are close to 1". The review's three
   counterexamples are now fixtures.
2. **Scalar slopes do not isolate magnitude or information.** Accepted. With
   P = Σ|p|², T = Σ|t|², C = Σp·t: b_rev = ρ·g and b_cal = ρ/g, where g = sqrt(P/T) and
   ρ = C/sqrt(PT). "R²" is renamed *squared uncentered alignment*. The "half of the
   shortfall" reading, the "over-dispersion" label and the claim that two mechanisms were
   separated are withdrawn. Magnitude (`energy_ratio`) and alignment (`alignment`) are
   now reported separately, uncorrected.
3. **The k adjustment needs explicit assumptions.** Accepted. k > 0 is a sensitivity
   analysis under p = p* + e with E[p*·e] = 0, E[e·t] = 0 and error energy k times the
   real-frame displacement error, which itself includes systematic and shared-start
   error. k = 0 is primary, and the primary classification is now `class_k0`.
   "Above 1 for every k" is withdrawn: monotonicity in k holds only while the adjusted
   denominator stays positive. Adjusted alignment can exceed 1 and is not a resolution
   fraction.
4. **Inversion needs rejection.** Accepted. Simulator inversion is a candidate decoder
   using the same encoder, not the deciding instrument by construction. Any protocol
   needs residual-based abstention, near-optimal ambiguity checks, optimizer-recovery
   controls, controlled off-manifold and latent-mixture fixtures, nuisance stability,
   and permission to report "not identifiable". Gates are set before scoring reserved
   outcomes.

**Also corrected (LIT-002).** The ordinary-case question is one-sided non-inferiority,
not equivalence: use a one-sided test at the chosen α and margin, not TOST. Power must
use the actual independent unit. Agent-matched goal images are privileged and
candidate-specific, not a deployable goal specification. Reversion remains a hypothesis.
TRAIN-FIX-002 remains open; the CALIB runner's guard does not fix it.


## D032 — 2026-10-01 — Proposed MEAS-001 feasibility scope (not an execution freeze)

At Omer's request, MEAS-001_PLAN.md specifies a bounded next measurement study after
D031 and CALIB-REVIEW-002. This records proposed protocol choices; it neither changes
a historical seal nor reports that implementation or experimental execution has begun.

- Use 48 existing, previously opened SHIFT-002 development families, split by family
  into 24 calibration and 24 instrument-assessment families, balanced across scenes.
- Jointly invert agent position, block position and block angle with the frozen encoder;
  no true future pose enters initialization, search bounds or nuisance conditioning.
- Calibrate residual rejection on real calibration latents, then freeze it before
  assessment. Require restart agreement, a narrow near-optimal block-position set,
  nuisance challenges and explicit abstention. Renderer/search failures are distinct
  from predictor failures.
- Use the proposed 1.28/2.56-unit measurement tolerances and per-cell coverage/floor
  gates as engineering feasibility criteria. Real-latent accuracy does not establish
  that a predicted latent has a valid physical interpretation.
- Include nonlinear latent-mean and ambiguity controls. Do not label every latent
  mixture off-manifold or require its inverse to equal the physical mean.
- Four cumulative CPU machine-hours, including calibration, assessment, smoke work,
  failed attempts and the optional 16-target saved-prediction diagnostic. Benchmark
  first; stop if the projected workload with 1.5x headroom does not fit.
- Implementation must freeze exact search/config/source/input lineage before
  calibration. No new training, model replication or fresh scientific confirmation
  is included. The implementation and execution task remains TODO.

These choices are a proposal, not validated instrument specifications. Any revision
before execution must be recorded; after instrument assessment opens, a change
requires a new version and cannot relabel that assessment as untouched confirmation.

## D033 — 2026-10-01 — MEAS-001 engineering set, version rule and this pass's stop point

Recorded before any MEAS-001 engineering encode. It refines D032 at Omer's direction;
no historical seal changes.

- **Engineering set.** Search settings (step sizes, proposal counts, restart and
  stopping rules, evaluation cap) are developed only on:
  - analytic and mocked fixtures;
  - the previously opened GALLERY-001 families (`outputs/gallery001/config.json` `ids`).

  Their engineering IDs are declared and sealed first. A metadata check must show no
  shared reset seed, reset state or archive episode with the 48 MEAS-001 families.
  Ground-truth labels may score and debug engineering runs but never enter the
  optimizer's input interface. World-model predictions are never used to choose settings.
- **Budget.** Engineering runs are charged to the four-hour MEAS-001 ledger, with a 60-min
  engineering sub-cap. The later calibration throughput check may stop the study; it may
  not trigger tuning.
- **Sequence.**
  1. Develop and test settings on the engineering set.
  2. Freeze the implementation, settings and all 48 family IDs.
  3. Use the 24 calibration families only to compute ε and τ by the predeclared rules.
  4. Freeze those values, then run the 24 assessment families once.
- **Version rule.** If calibration exposes a problem that needs changed settings, that
  version stops. Its calibration families become development data, and a revised
  instrument needs a replacement calibration partition. If assessment has opened, its
  families must also be replaced for a genuinely held-out assessment.
- **This pass.** Implement the complete runner and tests, run engineering validation, and
  produce a freeze-ready configuration or an explicit not-ready report. The 48 families'
  metadata may be read for membership and disjointness; they are not encoded or
  inverted. τ is not computed, assessment does not run, and saved predictor targets are
  not inspected.

### D033 amendment — 2026-10-01 — agent geometry and reset identity (proposed; needs review)

Two rule corrections came out of engineering-set and metadata checks before any
engineering encode, as D033 permits. Both need review sign-off before freeze.

1. **Agent geometry (changes a D032 rule).** D032 asked to extend wall and workspace
   checks to the agent. In the pinned simulator the agent is a kinematic body that walls
   do not stop. Engineering ordinary families show realized states with the agent inside a
   wall (14 of 60 endpoints) or fully off-canvas (9 of 60); challenge families show none.
   - Proposed: check the agent only for agent–block penetration (1.0-unit tolerance).
   - Search agent centres over [−16, 528], which reaches every distinct rendering; further
     positions render identically.
   - Keep all block checks.

   The literal D032 reading stays available as `agent_walls=True`. Under it, such real
   frames would be `unsupported_geometry` and count against coverage.
2. **Reset-family identity (clarifies D032 "aliases").** All archive-derived families
   reset from archive states with a constant seed (0), so a seed identifies a family only
   for seeded resets. Identity is therefore the exact reset state (when recorded), the
   archive episode, and, for seeded resets, the seed.
   - All 120 SHIFT-002 development families have distinct reset states and episodes.
   - Engineering and MEAS-001 families share no seed, state or episode.

**Also observed (renderer).** Replayed frames reproduce bit-for-bit, and restoring after
pose mutation reproduces the observed RGB. Rendering a saved endpoint pose by direct
assignment differs at 2–3 edge pixels in 7 of 210 engineering frames, because
physics-stepped and assigned transforms differ at floating-point level. The latent residual
at the true pose is therefore an integrity diagnostic, reported in engineering.

### D033 amendment disposition — 2026-10-01 — accepted after MEAS-REVIEW-001

Omer accepts both proposed corrections, as MEAS-001_REVIEW.md recommends. Neither
changes a historical seal, and no MEAS family has been encoded or inverted.

1. **Agent geometry.** The pinned agent is kinematic, so realized wall-overlapping or
   off-canvas agent poses are physical states and are not rejected. Block checks and the
   1.0-unit agent–block contact tolerance stay. An invisible agent's coordinates are
   nuisance representatives, not identified positions. Fixtures proving that the
   [−16, 528] bounds cover the renderer's clipped and invisible cases belong to
   MEAS-FIX-001 and must pass before any freeze.
2. **Reset-family identity.** Identity is the exact reset state (when recorded) and the
   source episode, plus the seed only for seeded resets. The next spec states this rule
   and the disjointness evidence explicitly.

MEAS-001 stays blocked on MEAS-FIX-001. Per the 2026-10-01 project plan it is off the
course-report critical path; any engineering version 2 follows MEAS-001_REVIEW.md
(independent refinement per search, separate fit-acceptance and ambiguity thresholds,
corrected cumulative accounting, stop on failure).

## D034 — 2026-10-01 — ROBUST-001 multi-instrument re-decoding (definitions before values)

Exploratory reanalysis of already-opened records; recorded before any kNN fit or
off-manifold value was computed. It asks whether the frozen model's dose-dependent
compression of decoded block displacement (SHIFT-002; CALIB-001 energy ratio) could be
an artifact of one readout's extrapolation. No model is run and no sealed file is written.

- **Inputs.** PHYS-002 development and reserved, and SHIFT-002 development and reserved,
  saved latents and family files. TRAIN-001 is excluded because its predicted latents were
  not saved; re-running fine-tuned inference is out of scope here.
- **Instruments (5).** READOUT-002 `mlp-20261004/5/6` and `ridge`, plus a new kNN
  readout: mean block xy of the k nearest READOUT-002 training-shard latents (50k real
  frames, Euclidean in projected space). k ∈ {1, 2, 4, 8, 16, 32, 64} is chosen by
  validation-shard (10k) mean block error before any study latent is decoded. The
  instruments differ in extrapolation: ridge is linear, kNN clamps to the bank, MLPs can
  saturate.
- **Estimands.** As CALIB-001 (`calib001_metrics`): step-2 decoded block displacement;
  b_rev, energy ratio, alignment and b_cal_k0 per instrument and cell; paired
  family-bootstrap dose contrasts (SHIFT challenge m0.60 − m0.30 constant; archive
  x2.0 − x1.0 pooled), 2,000 draws, CALIB-001's seeds.
- **Off-manifold flag.** d₁ is the distance from a latent to its nearest training-bank
  latent. A predicted step-2 latent is flagged if d₁ exceeds the 95th percentile of d₁ over
  the encoded real step-2 future latents of the same group (dataset × split × scene).
  Flag rates are reported by dose. An "on-manifold" stratum recomputes cells and contrasts
  from unflagged candidates only. This stratifies by a model-output property, so it is a
  sensitivity analysis, not case selection.
- **Disagreement.** Per candidate, the largest pairwise distance between the primary MLP,
  ridge and kNN decoded endpoints, on predicted and on real step-2 latents; its Spearman
  correlation with d₁. Descriptive only.
- **Reproduction gate.** Before anything is reported, every CALIB-001 cell for the four
  shared instruments (estimates and intervals of b_rev, energy ratio, alignment, b_cal_k0)
  must be reproduced within 1e-9 from the same saved latents. Family files are verified
  against their collection hashes, and all consumed latent, shard and model files are
  hashed into the manifest. A failed gate writes no report.
- **Interpretation rules.**
  - *Instrument-robust trend:* the energy-ratio dose contrast is negative with a CI
    excluding 0 for all five instruments in both scene types.
  - A trend shown only by kNN does not count; kNN can manufacture reversion.
  - If the on-manifold stratum keeps a negative CI-excluded contrast, off-manifold
    extrapolation is not needed to explain the trend. If it vanishes, the result is
    inconclusive, since the stratum is selected on model output.
  - Agreement is corroboration, not validation of any readout on predicted latents (D031).
    No mechanism follows.

## D035 — 2026-10-01 — TRAIN-002 support-manipulated fine-tuning (protocol before data)

Recorded before any TRAIN-002 collection, training or scoring. It adds the controls
TRAIN-001 lacked, to ask whether removing the frozen model's compression at the 0.60
command requires 0.60 in the fine-tuning data.

- **Arms.** All start from the released checkpoint and train the predictor, action
  encoder and prediction projector (11,740,334 parameters): AdamW, LR 1e-5 (TRAIN-001
  pilot), weight decay 0.01, clip 1.0, 2,000 updates, 64 rollouts per update, seeds
  20261111–13. The encoder, projector, BatchNorm statistics and readouts stay frozen.
  - `frozen`: no training.
  - `balanced` and `archive_weighted`: the six TRAIN-001 runs, reused unchanged. Their
    checkpoint hashes must match the TRAIN-001 seal.
  - `dose_omitted` (new): the TRAIN-001 development training families with magnitude
    weights (1/3, 1/3, 1/3, 0) over (0.15, 0.30, 0.45, 0.60). The batches and RNG streams
    are those of `balanced` for the same seed, so the two arms are paired. Effective
    examples per update are 48, not 64; this is disclosed, not matched.
  - `archive_only` (new): 2,400 fresh natural archive windows (recorded controls), one
    branch each, 64 windows per update. A generic-fine-tuning control on the
    pretraining distribution.
- **Confirmation.** Fresh 120 challenge + 120 archive-rest families (TRAIN-001 family
  process, constant pushes 0.15–0.60) and 120 ordinary windows. They are disjoint from
  every earlier study and from the archive-only training windows.
- **Measurement gate.** The TRAIN-001 `cell_gate` (≥ 50 moved, real-frame slope CI
  inside (0.85, 1.15), floor < 0.25) is applied per scene × {0.45, 0.60} on
  *confirmation real frames only*. It is computed and sealed before any predictor scores
  a confirmation unit. A failed 0.60 cell makes the pooled primary verdict
  `not_assessable`, and that scene is reported descriptively.
- **Primary.** Absolute terminal block-position error on the 0.60 branch (readout
  `mlp-20261005`), averaged over seeds. Contrast P1 = dose_omitted − balanced, with scene
  means weighted equally and 2,000 scene-stratified paired family bootstrap draws
  (seed 20261322). Verdicts:
  - CI lower > 0 → `held_out_dose_deficit_supported`;
  - CI upper < 0 → `omission_better_supported`;
  - otherwise → `held_out_dose_deficit_not_established` (no equivalence claim).
- **Secondary (prespecified, not substitutes for P1).**
  - Transfer fraction T = mean(frozen − dose_omitted) / mean(frozen − balanced) at 0.60,
    on the same bootstrap draws.
  - Manipulation check: dose_omitted − balanced at 0.45, which is in support.
  - Generic fine-tuning G = frozen − archive_only at 0.60. CI lower > 0 →
    `generic_finetuning_gain_supported`; otherwise `not_established`.
  - Pred/real projection-slope ratio R per arm, scene and magnitude.
  - Terminal latent MSE at 0.60.
  - Ordinary-window errors relative to frozen (tolerance 1.28 reported as in TRAIN-001).
  - Sign agreement of P1 and G across the three readouts.
- **Stop gates** (implementation and optimization): trainable count; frozen tensors
  unchanged; finite loss; mean training loss over the last 100 updates below that over
  the first 100; reused checkpoints match. The budget is 6 CPU hours on its own ledger.
- **Exploratory, not part of this verdict:** later scoring of all arms on SUPPORT-001's
  over-range doses.

## D036 — 2026-10-01 — SUPPORT-001 support map and account test (preregistration)

Recorded before the support map, account fit or any SUPPORT-001 family exists. It tests
which account predicts how the frozen predictor distorts block displacement *outside the
dose range used to fit the account*.

- **Support map** (archive only, no model). Over all archive episodes, windows at stride
  5 run from state t to t+10 under the recorded controls [t, t+10). All vectors are in
  the block frame (rotation by −block angle). State features s are the agent position
  relative to the block and the agent displacement over the previous 5 steps, each
  standardized by the archive SD. The command a is the mean of the 10 controls. Effect
  d is the 10-step block displacement. For a query (s, a):
  - N_K(s): the K = 512 nearest archive windows in s;
  - m(s): their mean d, rotated back to the world frame;
  - log support ℓ = log((#{i ∈ N_K(s): ‖a_i − a‖ ≤ 0.1} + 0.5) / (K + 1)).
- **Accounts** for the decoded predicted displacement p, given the true displacement t:
  - N: p = κ·t;
  - A (attenuation): p = clip(α + β‖a‖, 0, 2)·t;
  - B (reversion): p = w·t + (1 − w)·m(s), with w = sigmoid(θ₀ + θ₁ℓ);
  - C (clamp): p = min(1, c/‖a‖)·t.
- **Fit (development).** Parameters are fitted by least squares on the 2-D residual,
  using the opened TRAIN-001 confirmation records of the frozen model (240 families ×
  doses 0.15–0.60, primary readout). They are frozen and hashed before any SUPPORT-001
  family is collected.
- **Fresh conditions.** 150 challenge and 150 archive-rest families with constant pushes
  at 0.15, 0.30, 0.45, 0.60, 0.80 and 1.00. These use the TRAIN-001 family process and
  admission rule (movement at 0.30 and 0.60), disjoint from every earlier study
  including TRAIN-002.
- **Gate.** The TRAIN-001 `cell_gate` thresholds per scene × {0.60, 0.80, 1.00}, on
  real frames only, computed and sealed before the predictor runs on any fresh family.
  Over-range cells that fail are excluded from the primary comparison. If both scenes
  fail at both over-range doses, the verdict is `not_assessable`.
- **Primary decision.** The squared error ‖p − f_X‖² per branch, averaged per family
  over the gated over-range branches (0.80, 1.00), with scene means weighted equally and
  2,000 scene-stratified paired family bootstrap draws (seed 20261422). Account X is
  `established` if X − Y has a CI entirely below 0 for every other account Y. Otherwise
  the verdict is `no_account_established`. This follows PhysWeep's held-out selection;
  a null is reportable.
- **Secondary (descriptive).**
  - The same errors on the fresh in-range doses.
  - Dose-response of the projection slope, pred/real ratio R, energy ratio and alignment
    across all six doses.
  - **Agent probe:** pred/true projection slope of the decoded *agent* displacement per
    dose (outputs 4–5 of the same MLPs). The kinematic agent moves as commanded, so
    compressed agent motion would implicate the action pathway rather than interaction.
  - Sign agreement across the three MLP readouts.
  - Later, exploratory scoring of the TRAIN-002 arms.

## D037 — 2026-10-01 — MEAS-001 engineering version 2 (after MEAS-REVIEW-001)

Recorded before any version-2 encode. It follows MEAS-001_REVIEW.md's recommendations and
does not alter version-1 artifacts, which still reproduce exactly.

- **Guard fixes first (MEAS-FIX-001).**
  - Lineage binding: each stored inversion is bound to its query, config, seed, target
    latent hash, input-file hashes, search sources and checkpoint; reuse needs an exact
    match.
  - Family-file hash checks in `require_frozen`, `query_inputs` and `_labels`.
  - `predictions_allowed` is derived from the complete assessment decision.
  - A versioned classifier (below).
  - F4 wording corrections.
  - A disclosed 120 s accounting charge, and a charged `score` stage.
- **Search (version 2).** Base S2: 2 searches × (256 Sobol + 2 × 320 compass). Each search
  then gets an independent Nelder–Mead refinement from its *own* best pose (300
  evaluations, simplex 2 units / 0.05 rad, xatol 0.1). Nothing pooled is used as restart
  evidence.
- **Classifier (version 2).**
  - The reported pose is the best independent-search result.
  - A better candidate farther than 1.28 units, or any unfinished applicable nuisance
    challenge, gives `optimizer_unresolved`.
- **Thresholds, now separate.**
  - Acceptance τ_fit uses the D032 percentile rule, applied to reported residuals.
  - Ambiguity τ_amb is per query and label-free: the least residual increase among eight
    1.28-unit block offsets of the reported pose. Probes stay outside the trace.
  - The near-optimal diameter limit stays at 2.56.
- **Procedure.**
  - New spec and output directory (`outputs/meas001/engineering-v2/`), on the same sealed
    engineering families.
  - Phase A develops the single declared setting; phase B validates it once with phase-A
    τ_fit.
  - Engineering criteria (D033) unchanged.
  - The cumulative 60-min engineering sub-cap includes v1 and the correction (about
    38 min remain).
- **Stop if either phase fails.** v2 ends in a freeze-ready or not-ready proposal only.
  The 48 MEAS families stay closed. Calibration, assessment and predictions need review
  and a freeze.

## D038 — 2026-10-02 — Derived SUPPORT-001 plotting summary (REPORT-FIX-001)

Not a protocol change. CPU_STAGE_EVALUATION F1 found that the report figure plotted
account B's mixture weight as its projection slope. `support001_derived.py` recomputes,
from the saved predicted latents of the 300 sealed SUPPORT-001 families (primary readout
mlp-20261005, sealed fits), each account's implied projection slope Σp̂·t/Σ‖t‖² per scene
and dose, plus the archive mean-command tail of the support map. It writes
`docs/manifests/support001-account-slopes.json` with seal, summary, fit and map hashes.
It lives outside `support001.py` so the sealed source hashes and the seal verification
are unchanged; it runs no model and changes no sealed file or verdict. B's slopes match
the review's recomputation (archive 0.45/0.60/1.00: 0.687899/0.654153/0.642711; designed
0.60: 0.635454); designed 0.15 has no true block motion and is left undefined.

## D039 — 2026-10-13 — Report examples and reader vocabulary (REPORT-REDESIGN-001)

Not a protocol change. The redesigned course report shows real test examples. These
selection rules were written before any example was exported or plotted:

- **SUPPORT-001 example.** Among the 150 designed (`challenge`) families of the sealed
  SUPPORT-001 cohort, take the family whose true block displacement at strength 1.00
  (‖s₂ − s₀‖ from the saved simulator states) is closest to that cohort's median at 1.00;
  break ties by the lowest family id. It is shown at strengths 0.30, 0.60 and 1.00.
- **TRAIN-002 example.** Among the 120 designed families of the TRAIN-002 confirmation
  cohort, take the family whose true block displacement at strength 0.60 is closest to that
  cohort's median at 0.60; break ties the same way. Its saved frames are shown at all four
  strengths, and its decoded endpoints for every model at 0.60.

Selection uses simulator outcomes only, never model predictions or errors. The examples
are for display: no estimate, test or verdict depends on them, and every aggregate in the
report still comes from the sealed summaries. TRAIN-002 decoded endpoints come from the
saved confirmation records. SUPPORT-001 endpoints apply the sealed primary readout
(mlp-20261005) to the saved predicted latents, as D038 does. No world model is run.
`report_examples.py` writes `docs/manifests/report-examples.json` (with source hashes) and
the example frames as PNGs under `docs/figures/report-examples/`.

The saturation of true block displacement from strength 0.60 to 1.00 (the agent sweeps
past the block, which has no momentum in this simulator) is reported as a descriptive
observation from the existing sealed gate cells (median moved true displacement per cell).
The export also records, from simulator states only, the share of SUPPORT-001 families whose
agent ends inside the rendered table at each strength. It explains why agent motion cannot
be read from frames at 1.00 (the kinematic agent passes the walls), the reason those agent
cells fail their real-frame band.

Report vocabulary (report text and figure labels only; code, manifests and macro names
keep their internal names):

| Internal term | Report term |
| --- | --- |
| archive (LeWM's training data) | training demos |
| archive scenes / archive controls ×1, ×2 | demo scenes / demo commands ×1, ×2 |
| ordinary windows | ordinary demo clips |
| frozen model | released LeWM ("frozen" is kept for modules held fixed in fine-tuning) |
| dose, command magnitude | push strength |
| challenge scenes | designed scenes |
| balanced / archive-weighted / dose-omitted / archive-only arm | all-strength / demo-weighted / without-0.60 / ordinary-clip fine-tuning |

