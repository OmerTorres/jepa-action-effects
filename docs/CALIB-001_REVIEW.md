# CALIB-001 / LIT-002 review — CALIB-REVIEW-001

**Date:** 2026-10-01. **Reviewed revisions:** `c0777c8` (literature) and
`619a151` (calibration reanalysis). Working tree clean at the start of review.

The numerical reanalysis reproduces exactly, and the literature review identifies
relevant precedents. The main problems are interpretation and the proposed
measurement acceptance criteria. These results do not yet identify two physical
failure mechanisms or eliminate uncertainty in the latent prediction objective.
No experiment source or saved result was changed in this review.

## Findings

### [P2] Distinguish physical scalar calibration from conditional-mean latent prediction

**Locations:** `CALIB-001_RESULT.md:14–25,70–73`; matching claims in TASKS,
STUDY_SYNTHESIS and the LIT-002 addendum.

The identity E[p·t] = E[p·p] is necessary when p is the conditional mean of the
**physical displacement t**. LeWM predicts a latent target and the physical readout
is nonlinear. Even an exact decoder need not commute with conditional expectation:

`d(E[z | inputs]) != E[d(z) | inputs]`.

A concrete counterexample, evaluated with `calib001_metrics`: let future latent z
be equally likely 1 or 3 and let physical displacement be t = (z², 0). The optimal
latent mean predictor outputs 2, and an exact decoder outputs p = (4, 0). The true
physical mean is (5, 0). The measured b_cal is **1.25**, with zero real-frame readout
error. Thus a slope above 1 can coexist with perfect conditional-mean latent
prediction and exact decoding. It does not uniquely identify predictor bias rather
than the interaction between uncertainty and nonlinear representation geometry.

Conversely, b_cal = 1 is not sufficient for conditional calibration. For predictions
(1,0), (2,0) and corresponding deterministic outcomes (3,0), (1,0), it is exactly 1,
although the outcomes conditional on either prediction are wrong. An interval
containing 1 is also not an equivalence result.

**Correction:** describe departures from the scalar-calibration condition for the
decoded displacements. Replace “honest hedging does not explain” with that narrower
statement, and “fine-tuned models are calibrated” with “their estimated scalar gains
are close to 1.” Keep conditional-mean latent prediction, nonlinear decoding, and
physical calibration as separate hypotheses. Add both counterexamples to the
interpretation fixtures before using this diagnostic to decide a mechanism.

### [P2] A scalar slope does not isolate excess magnitude or loss of information

**Locations:** `CALIB-001_RESULT.md:29–31,48–63,74–76,130–137`;
`calib001_metrics.py:7–12`; generated draft definition of “resolution.”

Write P = Σ|p|², T = Σ|t|², C = Σp·t. Without the noise adjustment, let
g = sqrt(P/T) and rho = C/sqrt(P*T). Then:

`b_rev = rho*g`, `b_cal = rho/g`, and the reported `R² = rho²`.

These combine magnitude and directional alignment. A 60-degree rotation of exact
true displacement vectors has unchanged magnitudes and retains all information,
but produces b_rev = b_cal = **0.5** and this R² = **0.25**. Calling that excess
motion magnitude or lost forecast resolution would be incorrect. The product
identity is an algebraic factorization, not a decomposition into identified causes.

The issue affects interpretation of actual records: TRAIN-001 frozen challenge
0.60 has b_rev = 0.663405 and uncorrected b_cal = 0.836538, yet its aggregate
predicted/true displacement energy ratio is **0.793036**, below 1. This is not a
centered variance estimate, but it directly shows why a below-1 slope cannot alone
support the statement that predictions vary more than outcomes.

**Correction:** call b_cal,k=0 the optimal through-origin scalar gain, and call the
uncorrected product squared uncentered alignment (or explicitly scoped scalar-fit
R²). Report magnitude and direction separately when discussing compression or
over-dispersion. Withdraw the “half of the shortfall” causal reading and the claim
that two mechanisms have been separated. A centered calibration analysis with an
intercept or a vector-valued calibration map would be an additional exploratory
diagnostic, not something the current two scalar slopes already establish.

### [P2] The k adjustment needs stronger assumptions than equal error size

**Locations:** `CALIB-001_RESULT.md:113–115,143–145`; D030 estimand/limits;
`calib001_metrics.py:14–17`.

If decoded p = p* + e, subtracting E|e|² from E|p|² recovers E|p*|² only when
E[p*·e] = 0. Leaving the numerator unchanged additionally requires E[e·t] = 0.
Matching the real-frame displacement MSE to prediction-side error energy does not
establish either orthogonality condition. That real-frame MSE includes systematic
decoder errors and the shared decoded-start error; it is not automatically a
classical independent noise variance.

The module mentions classical noise, but the result and decision text abbreviate
the assumption to equal noise and use k=1 as the default physical interpretation.
Also, monotonicity in k holds only while the adjusted denominator remains positive;
it does not justify “above 1 for every k.” The code correctly makes nonpositive
denominators undefined.

**Correction:** retain k=0 as the directly measured scalar-fit statistic; present
k>0 as sensitivity under an explicit additive, orthogonal-error model. Do not treat
the adjustment as recovered physical truth. The adjusted alignment need not remain
in [0,1]: the saved SHIFT-002 development archive x1.0/varying primary-readout value
is **1.0010585**. Do not label that quantity a demonstrated fraction of resolution.
The PHYS-002 above-1 result at k=0 remains a valid observation about decoded outputs.

### [P2] Simulator inversion needs an off-manifold rejection option

**Locations:** `LIT-002_REVIEW.md:163–170,191–204`;
`CALIB-001_RESULT.md:154–155`.

An inverse-rendering optimizer always returning the nearest valid state does not
establish that the predicted latent represents that state. For a latent outside
the rendered-state manifold, nearest-state projection can create displacement
bias. Encoder geometry, nuisance variables, search boundaries, and local minima
also affect the answer. This is another decoder using the same encoder; agreement
with the MLP is corroboration, not fully independent physical ground truth.

Passing retrieval on real latents and running multiple optimizer restarts are
necessary checks but do not test this issue. For example, with encoder
f(s) = (s,s²), the exact latent mean of equally likely states -1 and +1 is (0,1).
Its nearest rendered states are +/-sqrt(1/2), neither the realized endpoints nor
their physical mean. A correct latent mean can therefore invert to an intermediate
physical state even with a globally solved optimizer and an exact renderer.

**Correction before implementing a new measurement protocol:** include residual
acceptance/abstention, near-optimal-pose ambiguity, optimizer recovery controls,
and controlled off-manifold/latent-mixture fixtures. Validate that results are
stable to plausible nuisance choices. Revisit the prior gallery limitations.
If these checks fail, report non-identifiability rather than forcing a pose.
Treat inversion as a bounded feasibility study, not the deciding instrument by
construction. Develop gates before scoring newly reserved model outcomes.

## Literature assessment and smaller follow-ups

The targeted source checks support the broad overlap warning. The
[Intervention Gap](https://arxiv.org/html/2608.29998) studies native-model
intervention fidelity and reports rotation with excess gain in its LeWM setting;
this is a close precedent, not evidence that the same cause explains PushT.
[IMPLY](https://arxiv.org/html/2609.12441) distinguishes physically anchored
evaluation from self-consistency.
[PhysWeep](https://arxiv.org/html/2609.06207) explicitly compares reversion and
clamping and rejects both in its adjudicated examples. These support testing
competing accounts, including “neither,” rather than promoting reversion to a
finding. [Kang et al.](https://arxiv.org/abs/2310.00873) motivate a hypothesis about
extrapolation; they do not establish it for this state-conditioned action pathway.

The source check also covered the abstracts of
[Sensitivity Shaping](https://arxiv.org/abs/2606.14585),
[Do-JEPA](https://arxiv.org/abs/2609.37378), and
[ARC-Bench](https://arxiv.org/abs/2609.05461). This is not an exhaustive verification
of all cited papers, the reported search counts, checkpoint compatibility, or cost
estimates. Literature overlap alone does not establish that the proposed remaining
niche is novel.

Before freezing any extension, also correct the LIT-002 suggestion to use TOST for
the one-sided ordinary-case non-inferiority question: equivalence and non-inferiority
have different hypotheses. Any new power calculation should use the chosen alpha,
margin and actual independent sampling unit. The proposed planner evaluation must
also distinguish privileged candidate-specific agent-matched goals from a deployable
goal specification. TRAIN-FIX-002 remains TODO; the new CALIB runner's guard does not
fix the earlier TRAIN follow-up command.

## Verification

- `make check && make test`: **256 passed**, no skips, 19 existing Gymnasium
  warnings; pytest 13.26 s.
- Ran `calib001.phys`, `.shift`, and `.train` into a temporary directory. All three
  complete output objects and file hashes match the saved analysis outputs exactly.
  No world-model inference or simulator collection was run; PHYS/SHIFT saved
  latents were decoded through the existing frozen readouts.
- `reproduction(results)`: **27 checks passed**, maximum difference
  **1.1102230246251565e-15**. The compact manifest results and generated draft also
  match exactly.
- All source, input and readout hashes listed by the CALIB manifest match current
  files. Manifest SHA-256:
  `9e00e501280328cd8a98489f1c649d158937b18e9d5b99bf5d0aaf7f6bfeb678`.
- The three scalar-calibration counterexamples above were evaluated in memory with
  `row_statistics` and `estimates`; they are synthetic logic checks, not study data.
- Review writes are limited to this review, TASKS and LOGBOOK. Existing sources,
  generated tables, manifests and sealed artifacts were not rewritten.

## Recommended disposition

Keep CALIB-001 as a reproducible exploratory description of scalar gains, alignment
and their dependence on dose. Correct its interpretation before using it in
REPORT-001 or a publication proposal. The reversion mechanism remains unestablished.
The next research task should be bounded measurement feasibility with explicit
failure cases and abstention, before another training campaign or broad replication.
The course report can proceed using the already qualified evidence.

## Follow-up verification — CALIB-REVIEW-002, 2026-10-01

Reviewed the uncommitted CALIB-FIX-001 changes on top of `e952b13`. **All four
requested corrections are implemented.** The result document withdraws the
conditional-mean, over-dispersion and causal-decomposition claims; k=0 is primary;
the additive-error assumptions are explicit; D031 treats inversion as a candidate
decoder with rejection and ambiguity controls. Historical statements are preserved
with correction addenda. The three diagnostic counterexamples and algebraic
factorization are now tested. No new blocking implementation issue was found.

**Verification:**

- `make check && make test`: **260 tests passed**, no skips; 19 existing Gymnasium
  warnings, pytest 13.57 s.
- Ran the current PHYS/SHIFT/TRAIN reanalysis into a temporary directory: each
  saved output hash matches. The compact manifest results and generated draft match
  exactly; all **27 reproduction checks pass**.
- Loaded the original metrics and runner directly from `git show 619a151:...` into
  temporary Python modules and reran the original analysis separately. Comparing
  every original leaf with the current outputs gives **13,768 compared, 0 changed**.
  The new energy/alignment statistics and paired contrasts are additions.
- All six frozen energy-ratio dose contrasts reproduce with negative intervals;
  the three MLP readouts agree on k=0 classifications in all 12 key cells.
- Current metric/runner/lockfile hashes and readout hashes match the updated
  manifest. The TRAIN-001 seal and confirmation summary still match their original
  runtime-manifest hashes.

**Reporting refinements:** keep the exact names *energy ratio* and *uncentered
alignment*. P/T is squared RMS magnitude ratio, not a mean length ratio: 0.46
corresponds to an RMS ratio of approximately 0.68, not 46% of the true length.
Also, rho is the cosine between stacked vectors and depends on variation in
per-candidate gains as well as angular error. For true vectors (1,0), (1,0) and
predictions (0.1,0), (1.9,0), every angle is correct but rho is approximately 0.743.
This refines the earlier review's shorthand about separating magnitude and
direction. Use the existing angle diagnostic to make claims specifically about
direction. “Alignment changes modestly” is supported descriptively; “direction is
unchanged” or an equivalence claim is not. The TRAIN challenge alignment contrast
is -0.0576 [-0.0998, -0.0154], so it should not be described as no change.

The opening handoff sentence saying b_cal reflects energy rather than direction
is imprecise: **b_rev/b_cal = P/T**, while b_cal itself still mixes alignment and
magnitude. The corrected module's equations are right.

**Disposition:** accept CALIB-FIX-001 with those reporting qualifications. Its
descriptive dose trend is supported for these decoded outputs; no mechanism or
general physical failure has been established. A short measurement-feasibility
proposal under D031 is the next research step. No further training or replication
is needed to close this review. TRAIN-FIX-002 remains a separate open task.
The implementing changes remain uncommitted; this review did not modify their
code, outputs or manifest.
