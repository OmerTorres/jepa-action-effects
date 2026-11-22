# TRAIN-001 result: magnitude-balanced vs archive-weighted fine-tuning

**Status:** complete under the sealed protocol (TRAIN-001_PLAN.md + D028). The nine
production stages were charged 1.06 of 24 cumulative machine-hours on a Mac CPU (4
threads). The real-data implementation smoke run (105 s) ran outside that ledger under
D028 and is reported separately.

> **Corrections (2026-09-30, TRAIN-REVIEW-002; D029).** This document was revised after
> review. No sealed artifact, verdict or number was changed. The original wording is in
> Git history (`68d9fbc`).
> 1. The secondary latent interval was mis-transcribed. The sealed value is
>    **−0.00684 [−0.01276, −0.00110]**, not −0.0070 [−0.0127, −0.0014].
> 2. "Magnitude weighting did not matter physically" overclaimed. The pooled
>    hypothesis was not assessable, and no equivalence test was specified. The
>    supported statement is: **no pooled balanced-over-archive physical benefit was
>    established**; the descriptive contrast was near zero.
> 3. The statement "not measurement-dependent" relied on a verdict-string flag.
>    That flag cannot detect disagreement once development eligibility fails. It is
>    replaced by the contrast-level comparison below (`train001-followup.json`).
> 4. The cause of the fine-tuning gain is a hypothesis, not a finding. Claims that a 2%
>    high-dose weight was sufficient are withdrawn: no dose-omission control was run.

**Pooled physical verdict:** `not_assessable_under_measurement_protocol`. The
development measurement gate failed in the two archive cells before training. D028
makes that failure irreversible, even though every confirmation gate later passed.

**Prespecified secondary (latent) outcome:** balanced fine-tuning has a small latent
advantage over archive-weighted fine-tuning: −0.00684 [−0.01276, −0.00110], about 4%
relative. Both fine-tuned conditions reduce high-magnitude terminal latent MSE by about
60% relative to the frozen model.

**Ordinary-case non-inferiority:** not established.

Numbers come from `docs/manifests/train001-runtime.json`, and unrounded cited values from
`docs/manifests/train001-followup.json` (`exact_numbers`). The auto-generated tables
are in `TRAIN-001_RESULT_DRAFT.md`. All physical numbers are **descriptive**: the pooled
physical hypothesis is not assessable.

## Question and design

Does emphasizing larger commands during fine-tuning improve the frozen LeWM
checkpoint's high-magnitude physical predictions beyond ordinary magnitude
weighting, without degrading ordinary cases?

**Three conditions.**
- Frozen checkpoint.
- Archive-weighted fine-tuning. Magnitude weights 0.512 / 0.336 / 0.132 / 0.020 for
  0.15 / 0.30 / 0.45 / 0.60, taken from 11,783 windows in 1,000 separate archive episodes.
- Magnitude-balanced fine-tuning, with 0.25 each.

**What was trained.**
- The predictor, action encoder and prediction projector: 11,740,334 parameters.
- The visual encoder, projector, action normalization, readouts and all BatchNorm
  statistics stayed frozen.
- Loss: two-step autoregressive latent MSE on cached real embeddings, with the first
  prediction fed back.
- Optimizer: AdamW (weight decay 0.01), gradient clip 1.0, LR 1e-5 (pilot-selected),
  2,000 updates. Batches of 16 families × 4 magnitudes.
- Seeds 20261111–13 are paired, so both arms saw identical batches and dropout draws.

**Data.**
- Four constant-push branches per family (0.15–0.60). History is three frames; the
  horizon is two native steps.
- Training: 600 families; validation: 120 families + 60 ordinary windows.
- Fresh confirmation: 240 families + 120 ordinary windows.
- Challenge and archive-rest scenes are equal in every split. Episodes, seeds and
  content are disjoint from each other and from all earlier studies.

## Gate outcomes

| Gate | Outcome |
| --- | --- |
| Implementation (boundary, frozen/BN tensors, gradients, finiteness) | Passed |
| Cached vs pixel rollout equivalence | Passed: 3.1e-6 (tolerance 1e-5) |
| Overfit diagnostic (≥50% in 500 updates) | Passed: 99.6% reduction |
| Budget projection (1.5× contingency) | Passed: 71 min projected vs 23.7 h remaining |
| Data integrity (caps, weight bins ≥100, disjointness) | Passed. Smallest bin 240; 1,140 records disjoint |
| **Development measurement gate** (sealed before training) | **Failed in the archive cells** (floors below) |
| Confirmation measurement gate | Passed in all four cells; cannot reverse the development failure (D028) |
| Strict provenance, entry and completion | Passed: 2,880 records, 8 model identities |

Measurement gates, primary readout `mlp-20261005` (floor limit < 0.25):

| Cell | Development moved / slope_real CI / floor | Confirmation moved / slope_real CI / floor |
| --- | --- | --- |
| challenge 0.45 | 60 / [0.948, 1.049] / 0.234 | 120 / [1.006, 1.067] / 0.206 |
| challenge 0.60 | 60 / [0.931, 1.025] / 0.222 | 120 / [1.011, 1.071] / 0.203 |
| archive 0.45 | 60 / [0.957, 1.014] / **0.278** | 120 / [0.953, 1.005] / 0.156 |
| archive 0.60 | 60 / [0.953, 1.014] / **0.304** | 120 / [0.963, 0.996] / 0.149 |

**Why the development archive cells failed.** Mean real-frame displacement error was
10.8–11.8 units against a median moved displacement of 38.7. At confirmation the archive
pushes moved further (median about 74), so the same readout noise fell below the limit.

This does not make the development failure reversible. A failure is a measurement
limitation, **not evidence that fine-tuning has no effect**.

## Measured physical contrasts (reported; not a confirmatory claim)

Absolute terminal block-position error on the 0.45/0.60 branches, in workspace units.
Values are seed-averaged, scene-equal-weighted, with 2,000 paired scene-stratified
family-bootstrap draws.

| Quantity | Estimate [95% CI] | Challenge | Archive |
| --- | --- | --- | --- |
| Frozen | 26.07 [23.56, 28.69] | 14.56 | 37.57 |
| Archive-weighted | 14.49 [12.89, 16.29] | 8.38 | 20.61 |
| Balanced | 14.40 [12.73, 16.30] | 8.05 | 20.76 |
| **Balanced − archive-weighted** | **−0.09 [−0.53, 0.38]** | −0.33 [−0.62, −0.04] | +0.15 [−0.69, 1.02] |
| Balanced − frozen | −11.66 [−14.29, −9.11] | −6.52 [−7.86, −5.18] | −16.81 [−22.12, −11.84] |
| Archive-weighted − frozen | −11.57 [−14.04, −9.17] | −6.19 [−7.53, −4.86] | −16.96 [−21.82, −12.37] |

**Training seeds.**
- Balanced − archive-weighted per seed: −0.04, −0.22, −0.01 (SD 0.11).
- Balanced − frozen per seed: −11.66, −11.85, −11.48 (SD 0.18).

**Scene subgroups** are exploratory. The challenge-only interval below zero cannot stand
in for the pooled cross-scene claim.

**What the numbers show.**
- The primary comparison between the two weightings is near zero. Its interval spans
  zero, so even without the gate failure no balanced-over-archive benefit would be
  established.
- Descriptively, both fine-tuned conditions reduced decoded errors on the sampled
  constant-push distribution. They roughly halve the frozen model's high-magnitude endpoint
  error.

## Secondary outcome: terminal latent MSE (prespecified)

This measures representation prediction, not physical prediction.

| Contrast | Estimate [95% CI] | Challenge | Archive |
| --- | --- | --- | --- |
| Balanced − archive-weighted | −0.00684 [−0.01276, −0.00110] | −0.00661 [−0.01073, −0.00294] | −0.00706 [−0.01839, 0.00453] |
| Balanced − frozen | −0.249 [−0.295, −0.203] | −0.157 | −0.341 |
| Archive-weighted − frozen | −0.242 [−0.287, −0.198] | −0.150 | −0.334 |

- Means: frozen 0.406; archive-weighted 0.162–0.165; balanced 0.156–0.158.
- Per-seed balanced − archive: −0.006, −0.008, −0.006.
- The balanced advantage in latent MSE is about 4% relative and consistent in sign
  across seeds. The archive-scene subgroup interval spans zero.
- No corresponding pooled physical benefit was established.

## Ordinary archive windows (1.28-unit tolerance)

- **Balanced − frozen endpoint error: +1.05 [0.11, 1.99].** Outcome:
  **non-inferiority not established**.
  - The upper bound exceeds 1.28 and the lower bound is below it, so regression beyond
    tolerance is not supported either.
  - With the primary readout the interval excludes zero, so balanced fine-tuning is
    slightly worse than frozen on these 120 windows. This is readout-dependent:
    `mlp-20261004` gives [−0.149, 1.822].
- Archive-weighted − frozen: +0.67 [−0.21, 1.62]. Balanced − archive-weighted:
  +0.38 [0.05, 0.69].
- Absolute errors: frozen 13.54; real-frame readout floor 11.94.

**Tradeoff.** Balanced weighting shows a small ordinary-case cost relative to
archive-weighted training: above zero with all three readouts. No pooled high-magnitude
physical gain was established to offset it.

## Controls and diagnostics (primary readout)

**Magnitude response.** Pred/real projection-slope ratio, seed-averaged:

| Cell | Frozen | Archive-weighted | Balanced |
| --- | ---: | ---: | ---: |
| challenge 0.30 | 0.917 | 0.906 | 0.898 |
| challenge 0.45 | 0.726 | 0.960 | 0.944 |
| challenge 0.60 | 0.637 | 0.926 | 0.936 |
| archive 0.30 | 0.891 | 0.992 | 0.993 |
| archive 0.45 | 0.737 | 0.964 | 0.980 |
| archive 0.60 | 0.628 | 0.922 | 0.948 |

- The frozen model reproduces the SHIFT-002 shrinkage pattern on fresh families.
- Fine-tuning brings the slope close to one. On its own, that does not establish better
  physical prediction.
- At 0.60 the frozen displacement direction error is about 30° (mean); fine-tuned models
  are at about 11–20°.

**Calibration comparator** (state-supervised output correction of the frozen model).
- The fitted scale is 1.03 for all readouts.
- It changes error by only −0.31 [−0.45, −0.16].
- Balanced − calibrated frozen is −11.36 [−13.93, −8.87].
- A global rescaling cannot reproduce the fine-tuning gain, which is direction- and
  magnitude-specific.

**Action permutation.** Deranged magnitudes within each family (permuted − correct
endpoint error, 0.45/0.60 branches):
- frozen: +8.8 [6.5, 11.0];
- every fine-tuned model: +16.7 to +17.2.

Fine-tuned predictions depend more strongly on the supplied action.

**Other controls.**
- Persistence: 55.4 [49.9, 60.8]. Every predictor beats it by a wide margin.
- **Readout sensitivity, by contrast** (`train001-followup.json`).
  - The pooled verdict string is the same for every readout by construction, because
    eligibility uses the primary readout only. It therefore says nothing about
    robustness.
  - Contrast by contrast, the three readouts agree on:
    - the interval sign of balanced − archive (spans zero: −0.09 / −0.19 / −0.15);
    - balanced − frozen and archive − frozen (below zero);
    - the calibration change (below zero);
    - both exploratory scene subgroups;
    - ordinary balanced − archive (above zero) and archive − frozen (spans zero);
    - the ordinary outcome (non-inferiority not established);
    - all confirmation gates.
  - **They disagree on ordinary balanced − frozen.** The interval lies above zero for
    `mlp-20261005` [0.111, 1.993] and `mlp-20261006` [0.124, 1.958], but spans zero for
    `mlp-20261004` [−0.149, 1.822]. The observed ordinary degradation against zero is
    therefore measurement-dependent.
  - Agreement among readouts does not validate them on predicted latents.

**Development evaluation** (exploratory, `development-eval.json`) shows the same pattern:
- balanced − archive −0.17 [−0.73, 0.32];
- balanced − frozen −7.33;
- latent balanced − archive −0.008 [−0.016, −0.001].

## Interpretation

1. **The research question's pooled physical answer is not assessable** under the
   frozen measurement protocol. This follows from the development archive floor
   failure, not from the training results.
2. **No pooled balanced-over-archive physical benefit was established.**
   - The descriptive decoded contrast was near zero, both overall and per seed.
   - No equivalence margin was specified, so a negligible physical difference is not
     established either.
   - Balanced weighting shows a small secondary latent-MSE advantage.
   - On ordinary windows, balanced − archive is +0.38 [0.05, 0.69], above zero for
     all three readouts.
3. **Both fine-tuned conditions reduced decoded errors on the sampled constant-push
   distribution.** This is descriptive.
   - A working hypothesis is that exposure to the designed high-magnitude constant
     pushes drives the adaptation. This was not isolated: there was no archive-only
     or dose-omission training control, so other effects of extra fine-tuning are
     not excluded.
   - The training and confirmation families come from the same generating process.
     Any gain is in-distribution adaptation, not evidence of general physical
     competence.
4. No planning benefit follows from this study.

## Limitations

- **Scope.** One checkpoint, one two-step horizon, a CPU runtime, and one simulator/archive
  pair.
- **Readout.** Physical quantities depend on the frozen READOUT-002 readout. Its validity
  on predicted (especially fine-tuned) latents is not established.
- **Measurement noise.** Archive-scene readout noise is large relative to motion at
  development scale, which caused the gate failure.
- **Training distribution.** Both arms were fine-tuned on exactly the transition family
  used for evaluation. The protocol did not test generalization to other action
  structures or to the archive's own controls.
- **Archive overlap.** Archive-derived observations may overlap the checkpoint's
  pretraining distribution; they are not described as unseen.
- **Ordinary windows.** 120 windows give a wide interval. Non-inferiority at 1.28 units
  could not be established. Whether balanced − frozen lies above zero depends on the
  readout.
- **Budget accounting.** The 1.06-hour ledger covers the production stages only. The
  105 s smoke run is disclosed separately.
- **Validator lineage gap.** The sealed validator did not check training-data lineage.
  A read-only check (`train001_followup.validate_lineage`) confirms that all six runs'
  collection/cache hashes, weights and specifications match the seal and protocol.
- **Uncertainty.** Seed variation is reported separately. Intervals are conditional on
  the three trained runs.

## Artifacts and reproduction

- **Ignored, local:** `outputs/train001/`, containing collections, caches, `pilot/`, six
  runs with checkpoints at 500/1000/1500/2000 and resume states, `seal.json`,
  `confirm/records`, `confirm/summary.json`. Also `outputs/train001-budget/ledger.json`
  and `outputs/train001-logs/`.
- **Tracked:**
  - `docs/manifests/train001-runtime.json`, `TRAIN-001_RESULT_DRAFT.md`;
  - `docs/manifests/train001-followup.json`: review follow-up with lineage validation,
    contrast-level sensitivity and exact cited numbers;
  - figures `docs/figures/train001/{learning_curves,magnitude_response,paired_errors}.png`.
    In the learning curves, the weighted training losses of the two conditions are not
    comparable in scale, because the weights differ.
- **Reproduction:** TRAIN-001_RUNBOOK.md.
