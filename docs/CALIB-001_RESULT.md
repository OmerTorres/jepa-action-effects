# CALIB-001 result — scalar gains, magnitude and alignment of decoded displacements

**Status:** exploratory reanalysis of opened records (D030 and amendment), with its
interpretation corrected after review (CALIB-REVIEW-001, D031).

> **Corrections (2026-10-01, CALIB-REVIEW-001; D031).** No earlier number changed. A rerun
> reproduced all 13,768 earlier output values exactly and added magnitude and alignment
> estimands. The original wording is in Git history (`619a151`).
> 1. "Honest hedging does not explain the shrinkage" is withdrawn. A scalar gain b_cal ≠ 1
>    shows that decoded predictions depart from the scalar-calibration condition. It does
>    not separate predictor bias from latent uncertainty acting through a nonlinear
>    decoder: an exact latent mean through an exact decoder can give 1.25.
> 2. The "over-dispersion" component, the "half of the shortfall" decomposition and
>    "two mechanisms separated" are withdrawn. b_rev and b_cal both mix magnitude and
>    direction: a 60° rotation of exact vectors gives 0.5 for both. Magnitude and alignment
>    are now reported separately.
> 3. The k = 1 "noise correction" is a sensitivity analysis under an additive error model
>    orthogonal to prediction and outcome, not recovered truth. k = 0 is primary.
> 4. Simulator inversion is a candidate decoder needing abstention and ambiguity gates,
>    not the deciding instrument (LIT-002 addendum).

The reproduction gate passed: 27 sealed PHYS-002, SHIFT-002 and TRAIN-001 values were
reproduced with a maximum difference of 1.1e-15. No model was re-run and no sealed artifact
changed. The primary readout is `mlp-20261005`. All quantities are uncorrected (k = 0)
unless marked. Intervals are 95% family-bootstrap intervals (2,000 draws, paired within
groups). Full tables: CALIB-001_RESULT_DRAFT.md; numbers: `docs/manifests/calib001-runtime.json`.

## Quantities

With C = Σp·t, P = Σ|p|² and T = Σ|t|² over all candidates in a design cell:

| Quantity | Definition | Reads as |
| --- | --- | --- |
| Projection slope b_rev | C/T | Earlier headline; mixes magnitude and direction |
| Scalar gain b_cal | C/P | Optimal through-origin factor from prediction to outcome; necessary, not sufficient, for physical calibration |
| Energy ratio | P/T | Magnitude of decoded predictions relative to truth; readout error inflates it |
| Alignment ρ | C/sqrt(PT) | Uncentered directional agreement |

b_rev = ρ·sqrt(P/T) and b_cal = ρ/sqrt(P/T). This is algebra, not a decomposition into
causes.

## Observations (frozen model, decoded block displacement)

1. **Decoded magnitude falls with dose in every cell family.** Paired energy-ratio
   contrasts:

   | Cells | Energy ratio by dose | Top − lowest dose |
   | --- | --- | --- |
   | TRAIN-001 archive scenes (0.30/0.45/0.60) | 0.86, 0.61, 0.46 | −0.40 [−0.50, −0.30] |
   | TRAIN-001 challenge scenes | 1.45, 0.93, 0.79 | −0.66 [−0.85, −0.48] |
   | SHIFT-002 reserved challenge | 1.61, 1.15, 0.98 | −0.64 [−0.89, −0.41] |
   | SHIFT-002 reserved archive (×1.0/×1.5/×2.0) | 1.19, 1.01, 0.84 | −0.35 [−0.48, −0.23] |
   | SHIFT-002 development challenge / archive | — | −0.61 [−1.01, −0.26] / −0.35 [−0.55, −0.18] |

   Alignment changes little over dose (−0.07 to +0.04). On PHYS-002 reserved challenge
   families the energy ratio is 0.435 [0.378, 0.497], with alignment 0.851.

2. **Readout error does not appear to produce this trend, if it behaves on predicted
   latents as on real frames.** Real-frame decodes give energy ratios of 0.98–1.00 at every
   dose in TRAIN-001 archive scenes, and 1.27 → 1.14 in challenge scenes. The frozen decline
   is much larger. Whether the readout behaves the same on predicted latents is untested.

3. **Alignment depends on scene type.** It is about 0.91–0.93 in archive scenes and
   0.75–0.80 in challenge scenes at every dose. Lower challenge-scene alignment matches
   the 26–28° direction errors (REVIEW_FOLLOWUP_RESULT §4). Both slopes below 1 in challenge
   cells reflect alignment, plus energy ratios above 1 at low dose. They are not evidence
   of excess variability.

4. **Scalar gains.**
   - PHYS-002 reserved challenge: b_cal 1.291 [1.204, 1.389] (above 1).
   - TRAIN-001 archive: 1.000, 1.184, 1.335 at 0.30/0.45/0.60.
   - Challenge cells: 0.59–0.84 (below 1).
   - Ordinary windows: 0.714 [0.547, 0.835], where small motions make readout error large
     relative to signal.

   These describe departures from the scalar-calibration condition. They do not identify
   predictor bias, uncertainty propagated through a nonlinear decoder, or readout error on
   predicted latents.

5. **Readout-free latent magnitude.** The step-1 energy ratio of the projected latent
   change needs no readout, but it is dominated by agent motion. It also falls with dose:
   - SHIFT-002 challenge: 1.23, 1.03, 0.85;
   - SHIFT-002 archive: 1.05, 0.93, 0.83;
   - PHYS-002 challenge: 0.85.

   This is the strongest evidence that magnitude shrinks in the model's own space, not only
   after decoding. It is not block-specific.

6. **Null-dose motion.** Challenge pushes of 0.15 never moved the block. Decoded frozen
   predictions there exceed the real-frame decodes in squared length by 72.8 [54.6, 92.5]
   (TRAIN-001) and 104.6 [73.7, 144.0] (SHIFT-002) units². For fine-tuned models the excess
   is 8–15 units². This could be predicted motion or readout error specific to predicted
   latents.

7. **Fine-tuned models.**
   - Their energy ratios are near 1 at every dose: archive 1.05/1.00/0.94, challenge
     1.10/1.10/1.11 (balanced).
   - The dose decline is small in archive scenes (−0.11 to −0.16) and absent in challenge
     scenes.
   - Challenge-scene alignment is higher, at 0.90–0.93.
   - Scalar gains are close to 1 (0.86–1.03).

## Sensitivity analyses

- **Error model (k > 0).** These subtract k times the real-frame displacement error energy
  from P, assuming that error is additive and orthogonal to prediction and outcome. That
  error includes systematic and shared-start error. Values are in the draft. They do not
  change the direction of the magnitude trend, which uses only k = 0.
- **Readouts.** The three MLPs give the same k = 0 class in all 12 key cells. The ridge readout is nearly unbiased on real frames but
  very noisy (19–45 units rms) and cannot confirm anything independently.

## What this supports

- A descriptive, readout-mediated finding: for this frozen checkpoint, decoded block
  displacement magnitude falls relative to true magnitude as the push dose grows, while
  alignment stays roughly constant within scene type. A comparable trend appears in the
  readout-free latent change.
- The finding does **not** establish:
  - that the predictor is biased rather than uncertain in latent space;
  - two separate mechanisms;
  - the reversion hypothesis;
  - readout validity on predicted latents.

## Limitations

- Exploratory: all families were already opened, and no multiplicity correction was
  applied.
- One checkpoint. Fine-tuned models were trained on the evaluation distribution.
- The latent quantities are not block-specific. The decoded quantities depend on an
  unvalidated readout of predicted latents.
- The 0.15 challenge cells have no block motion; their slopes are undefined.

## Next

1. Draft a bounded measurement-feasibility proposal with abstention and ambiguity gates:
   - residual thresholds;
   - near-optimal pose ambiguity;
   - optimizer recovery;
   - controlled off-manifold and latent-mixture fixtures;
   - nuisance stability.

   It should be gated on development data before any scoring of reserved outcomes (D031).
2. Use magnitude and alignment, not either scalar slope alone, in any future prespecified
   analysis.
3. No new training or replication until the measurement question is settled.

## Reproduction

```bash
.venv/bin/python -m jepa_action_effects.calib001 all
```

It reads `outputs/phys002`, `outputs/shift002`, `outputs/train001` and
`outputs/readout002`, and writes:
- `outputs/calib001/` (ignored);
- `docs/manifests/calib001-runtime.json`;
- CALIB-001_RESULT_DRAFT.md.

The run takes about 10 s on CPU. Code: `calib001.py` and `calib001_metrics.py`; tests:
`tests/test_calib001*.py`, including the review's counterexamples.
