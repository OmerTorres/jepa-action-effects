# Study synthesis — evidence and remaining limits

The bounded study produced a clear split result. Replay and frozen inference
worked; the projected-latent predictor responded to candidate actions and
matched its real encoded futures better than swapped actions or persistence.
The available readouts did not measure block position precisely enough to
attribute fine-grained physical effects or action-choice regret to that
predictor. Stop experiments and report that boundary.

| Claim | Evidence | Status |
| --- | --- | --- |
| Simulator branches and contacts are reproducible | SIM-001 order/replay audit; PILOT-001 12-family duplicate/reversal checks | Supported for the validated checkpoint/simulator runtime |
| Pilot has meaningful block-effect coverage | PILOT-001: 88/120 families exceed delta 5.12 | Supported for the sampled pilot distribution |
| Physical block endpoint errors can be measured within the gate | PILOT-001, READOUT-001, DELTA-001 real-frame readout gates all failed; best bounded DELTA mean/p95 4.01/12.97 versus 1.28/2.56 | Failed measurement gate |
| Correct actions better match real future projected CLS than mismatched actions | LATENT-001: overall A 0.43975, 95% interval [0.40560, 0.47353], 60 families | Supported exploratorily in this representation and distribution |
| Frozen predictor beats latent persistence | LATENT-001: overall B 0.52431 [0.49227, 0.55754] | Supported exploratorily in this representation and distribution |
| Frozen predictor accurately predicts block motion or gives low physical regret | Requires a passing physical readout or independent physical measurement | Unassessable |
| Result generalizes to other seeds, checkpoints, simulators, or natural task prevalence | One frozen checkpoint and deliberately enriched challenge sampling; historical training-simulator identity unproven | Not established |

The latent signal may be substantially driven by agent motion. Contact-regime
differences are descriptive associations, not evidence of a causal contact
mechanism. LATENT-001 reused tuning families and has no untouched confirmation;
its intervals characterize family sampling within this exploratory set, not
generalization to new tasks. The physical gate was never relaxed after failure.

The literature review in LIT-001_REVIEW.md identifies precedents for paired
counterfactuals, action sensitivity, latent response, and regret. The defensible
contribution is a reproducible, checkpoint-specific feasibility and
correspondence case study with explicit measurement limits. A course report can
present the positive latent result and the negative physical-measurement result
together. Further expansion needs a new question and measurement strategy.

## 2026-09-28 addendum

The user reopened the contribution question. NOVELTY-002_MEMO.md identifies
AD-WM as a closer LeWM/PushT precedent and proposes a fixed-agent rendered
block-pose gallery as an unvalidated measurement candidate. A simulator-only
check reproduced 72/72 preselected real endpoint frames after render-only
block-pose mutation and restoration, but branch-derived alternatives changed
0/36 ordinary images. The existing LATENT-001 findings above remain unchanged;
no object-specific predictor result has been obtained from the proposed method.

## 2026-09-28 GALLERY-001 measurement addendum

GALLERY-001_RESULT.md records a completed 30-family simulator/encoder audit.
Nonidentical nearby queries retrieved their true block-pose template in every
eligible gallery at both predeclared scales, with 179/180 and 180/180 endpoint
coverage. This establishes measurement feasibility for a coarse controlled
position comparison in this projected CLS, not action-conditioned block
prediction or physical competence. GALLERY-001_RELATED_WORK.md finds clear
rendered-pose and action-evaluation precedents; the defensible remaining
question is an empirical one about this frozen checkpoint.

## 2026-09-28 NEXT-001 / PHYS-002 addendum

The physical question is now measurable. READOUT-002's 50k-frame readout has a
real-frame floor about 8× smaller than the median challenge effect.

PHYS-002 confirmed on fresh families that, on designed challenge pushes, the predictor
under-predicts block displacement (slope 0.56 vs 1.01; effect error 17 units above
the floor). H1 was not established: at matched effect size the error is proportional
in every contact regime.

An exploratory audit found no data or probe error. It also found no shrinkage on
in-distribution archive windows (slope 0.96). The physical row above therefore
changes from "Unassessable" to **"Measured: shrinkage on the challenge
distribution, absent on archive windows."**

SHIFT-001 will test whether action structure or scene layout causes it, and
H3-PLAN-001 will test decision relevance under the model's own planning cost. See
PHYS-002_RESULT.md and D022.

## 2026-09-28 SHIFT-002 addendum

SHIFT-002 confirmed on 240 fresh families (D023) that the predictor's displacement
shrinkage grows with push-command magnitude, in both designed and archive scenes.

- Challenge pushes: R 0.91 → 0.81 → 0.74 for 0.30/0.45/0.60.
- Archive controls: R 1.04 → 0.96 → 0.88 for ×1/×1.5/×2.
- Within-block constancy has no effect.

The PHYS-002 finding is therefore not a scene artifact. The model under-predicts block
motion under large push commands, which is where its training action distribution
thins out. See SHIFT-002_RESULT.md.

## 2026-09-29 H3-PLAN-002 addendum

Under a latent planning cost with agent-matched goal images (D025), the predictor's
choices were confirmed to be no worse than a perfect predictor's at a CI-separated
level. Regret was JEPA 15.4 vs real latent 13.3, a difference of +2.1 [−1.3, 5.6], on
150 fresh families. H3 is not established.

The shrinkage found in PHYS-002 and SHIFT-002 is a measurable physical error whose
short-horizon decision cost is small under this measurement. That is a
well-measured null for H3 within its stated limits: realized agent pose in the goal
images, a two-block horizon, and one checkpoint.

## 2026-09-29 correction addendum (REVIEW-BRANCH-002, D026)

Corrections to the NEXT-001 / PHYS-002, SHIFT-002 and H3-PLAN-002 addenda above:

- **PHYS-002.**
  - The quantity is the readout-decoded projection slope (0.56), not a literal "moves
    56% as far". It pools length and direction errors.
  - "Endpoint" errors are displacement errors; the absolute endpoint errors are
    17.9 predicted / 3.6 real frame.
  - The controls rule out alignment offset, probe-size dependence and real-frame
    attenuation. They do not fully validate the readout on predicted latents.
- **SHIFT-002.** The projection-slope ratio decreases with push-command magnitude.
  The readout gate pooled doses. Per-dose real-frame slopes are near 1, but not all
  CIs include 1.
- **H3-PLAN-002.** "Its short-horizon decision cost is small" is withdrawn. H3 is not
  established, and a cost above the 5.12-unit tolerance is not excluded (pooled upper
  bound 5.59). The measurement concerns total latent prediction error under a
  privileged goal cost.

H1 and H3 remain not established. The magnitude trend is checkpoint-, distribution-,
horizon- and measurement-specific.

## 2026-09-29 REVIEW-FIX-001 addendum (exploratory)

**Reproducibility.** An independent re-scoring of all 150 PHYS-002 reserved families
reproduces the sealed records and summary exactly (difference 0.0). The provenance
validator passes for all three completed experiments.

**What the PHYS-002 slope is made of.**
- predicted displacements about 63% of the true length;
- a mean direction error of 19°, against 5° for the readout on real frames.

**SHIFT-002 dose trend.** It is mainly compression of the predicted displacement
length, which is over-predicted for small pushes and about matched for large ones.
In designed scenes a direction error of about 27° is present at every dose.
Displacement size and command magnitude are not separated.

See REVIEW_FOLLOWUP_RESULT.md.

## 2026-10-01 CALIB-001 addendum (exploratory)

A projection slope below 1 is expected even from an honestly uncertain predictor. CALIB-001
therefore added the calibration slope Σ(p·t)/Σ(p·p) on the opened records (D030;
CALIB-001_RESULT.md). It reproduced 27 sealed values exactly first.

- **In distribution.** The frozen model is calibrated on ordinary archive windows (0.99
  [0.88, 1.08]).
- **PHYS-002 challenge.** The projection slope 0.561 equals resolution 0.746 ÷ calibration
  1.329 [1.236, 1.435]. The calibration interval is above 1 without the noise correction
  too, so the shrinkage is not honest hedging, provided the readout is valid on predicted
  latents.
- **Compression grows with dose.** In archive-type scenes the calibration slope rises from
  ≈ 1 to 1.39. The trend is also visible in the readout-free latent change, and
  fine-tuning removes it. *(2026-10-02 correction: substantially reduces it; R 0.87–0.93 after fine-tuning.)*
- **Over-dispersion in designed challenge states.** At fixed dose the calibration slope is
  below 1 (R² ≈ 0.6). This component is sensitive to the noise assumption.

Readout validity on predicted latents remains the open question. Nothing here is
confirmatory.

## 2026-10-01 CALIB-001 correction addendum (CALIB-REVIEW-001, D031)

The CALIB-001 addendum above overstated its interpretation. Corrections:

- **"Not honest hedging" is withdrawn.** A scalar gain above 1 shows that decoded
  predictions depart from scalar calibration. It cannot separate predictor bias from
  latent uncertainty passing through the nonlinear readout, or from readout error on
  predicted latents.
- **"Over-dispersion" is withdrawn.** Below-1 slopes in challenge cells reflect lower
  alignment (about 0.75–0.80 against 0.91–0.93 in archive scenes), not excess variability.
- **The k = 1 adjustment** is a sensitivity analysis under an orthogonal additive-error
  model.

What stands (exploratory, readout-mediated): decoded displacement **magnitude** falls with
dose relative to truth in every frozen cell family. For example, the TRAIN-001 archive
energy ratio falls 0.86 → 0.46 (paired −0.40 [−0.50, −0.30]), while alignment barely
changes. The readout-free latent change shows a comparable decline. Fine-tuned models
stay near 1. No mechanism is established. See CALIB-001_RESULT.md.
