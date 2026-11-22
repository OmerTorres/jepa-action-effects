# SHIFT-002 result — predictor shrinkage grows with push magnitude

**Status:** confirmed under sealed rules (D023) on 120 + 120 fresh reserved families.
All manipulation checks passed.


## Corrections (2026-09-29, REVIEW-BRANCH-002)

These corrections follow `BRANCH_REVIEW_2026-09-29.md` and D026. The original text
below is kept as written.

- **"This is noise, not attenuation" is not established.** The D023 readout gate
  pools slope_real across doses within each state × structure cell, so dose-dependent
  readout bias could cancel in it.

  Per-dose real-frame slopes (review, exploratory):

  | | Doses | Real-frame slope |
  | --- | --- | --- |
  | Challenge | 0.30 / 0.45 / 0.60 | 1.045 / 0.995 / 1.004 |
  | Archive | ×1 / ×1.5 / ×2 | 0.936 / 0.959 / 0.965 |

  All are within the 0.85–1.15 band, but several archive CIs exclude 1. This does not
  reproduce a dose-dependent bias large enough to create the trend, and it does not
  prove there is none.
- **The archive faithfulness rule has no upper bound.** It accepts any CI whose lower
  bound is above 0.9 (a CI of [2, 3] would pass). The observed 1.042 [1.003, 1.083] is
  reassuring, but the rule is defective. Corrected rules for future protocols are in
  `protocol_gates.py`.
- **Wording.** "Magnitude drives the shrinkage" should read: *the readout-decoded
  projection-slope ratio decreases with push-command magnitude*. The claim is
  checkpoint-, distribution-, horizon- and measurement-specific.
- **Unchanged:** the sealed D023 verdicts and all numbers below.

## Question

PHYS-002 found that the frozen predictor under-predicts block displacement on designed
challenge pushes, but not on archive windows. Is the cause the push magnitude, the
within-block action structure, or the scene?

## Design

- **Challenge scenes:** constant pushes of 0.15, 0.30, 0.45 and 0.60, plus varying
  versions at 0.30 and 0.60. The varying versions keep each block's mean command and
  add an archive residual.
- **Archive rest windows:** the recorded future controls scaled by ×1.0, ×1.5 and
  ×2.0, each as recorded (varying) and as block means (constant).
- **Ratio:** R = predicted / real-frame displacement slope on moved candidates, with
  the frozen READOUT-002 probe and a two-block horizon.
- **Bootstrap:** by family within state type.

## Confirmed result (reserved families)

| Cell | R (pred/real slope) | 95% CI |
| --- | ---: | --- |
| Challenge push 0.30, constant | 0.913 | [0.837, 0.994] |
| Challenge push 0.45, constant | 0.807 | [0.724, 0.895] |
| Challenge push 0.60, constant | 0.744 | [0.673, 0.820] |
| Archive ×1.0 | 1.035 | [0.996, 1.077] |
| Archive ×1.5 | 0.961 | [0.926, 0.999] |
| Archive ×2.0 | 0.881 | [0.817, 0.939] |

| Contrast | Estimate | 95% CI | Verdict |
| --- | ---: | --- | --- |
| dM challenge (0.30 − 0.60, constant) | +0.168 | [0.107, 0.232] | magnitude **supported** |
| dM archive (×1.0 − ×2.0) | +0.154 | [0.095, 0.212] | magnitude **supported** |
| dS archive (varying − constant) | +0.001 | [−0.014, 0.017] | no effect within ±0.1 |
| dS challenge (varying − constant, matched 0.30/0.60) | −0.062 | [−0.102, −0.024] | varying slightly **more** shrunk |

**Readout checks.** slope_real is 0.96–1.03 in every state × structure cell. Real-frame
displacement error is about 5 in challenge frames and about 13–15 in archive-scene
frames; this is noise, not attenuation.

**Other checks.** Replay verification error is 0.0. Moved families per gate cell: 120.
The 0.15 push never reached the block within two blocks, so the low cell is
effectively the 0.30 push.

## Interpretation

- **Magnitude drives the shrinkage in both scene types.** Scaling the archive's own
  controls by 2× induces shrinkage (R 1.04 → 0.88) in scenes where the unscaled
  controls are predicted faithfully.
- **Within-block constancy does not cause it.** Constant and varying controls with
  the same block means give the same R in archive scenes. In challenge scenes the
  varying version is slightly more shrunk, which is the opposite of the constancy
  hypothesis.
- **Exploratory:** challenge pushes 0.45 and 0.60 move the block about the same
  median distance (21 units), yet R falls from 0.81 to 0.74. That suggests the
  command magnitude matters beyond the displacement it produces.
- Together with PHYS-002, the defensible statement is: *this frozen LeWM predictor
  increasingly under-predicts block motion as push commands move toward and beyond the
  upper tail of its training action distribution.*

## Limitations

- One checkpoint, two-block horizon, CPU inference.
- Archive-scene readout is noisier (about 13 units). The gate checks bias, not
  precision.
- Magnitude is manipulated on command scale. Displacement and contact duration
  co-vary with it.
- The SHIFT-001 development result that motivated this design is exploratory. Its 200
  reserved families were never opened.

## Auto-generated tables

## Development (exploratory)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| R1_challenge_high_constant | 0.7142 | [0.5895, 0.8610] |
| R_archive_x1.0 | 1.0412 | [1.0125, 1.0739] |
| R_archive_x1.0_constant | 1.0379 | [0.9968, 1.0778] |
| R_archive_x1.0_varying | 1.0448 | [1.0205, 1.0761] |
| R_archive_x1.5 | 0.9670 | [0.9271, 1.0166] |
| R_archive_x1.5_constant | 0.9691 | [0.9194, 1.0275] |
| R_archive_x1.5_varying | 0.9648 | [0.9252, 1.0148] |
| R_archive_x2.0 | 0.8212 | [0.7642, 0.8783] |
| R_archive_x2.0_constant | 0.8368 | [0.7777, 0.9013] |
| R_archive_x2.0_varying | 0.8041 | [0.7344, 0.8688] |
| R_challenge_high_constant | 0.6843 | [0.5736, 0.8122] |
| R_challenge_low_constant | 0.9042 | [0.7796, 1.0444] |
| R_challenge_m0.15_constant | n/a | n/a |
| R_challenge_m0.30_constant | 0.9042 | [0.7796, 1.0444] |
| R_challenge_m0.30_varying | 0.8731 | [0.7424, 1.0210] |
| R_challenge_m0.45_constant | 0.8176 | [0.6972, 0.9551] |
| R_challenge_m0.60_constant | 0.6843 | [0.5736, 0.8122] |
| R_challenge_m0.60_varying | 0.6377 | [0.5094, 0.7845] |
| dM_archive | 0.2201 | [0.1624, 0.2808] |
| dM_challenge | 0.2199 | [0.1048, 0.3464] |
| dS_archive | -0.0175 | [-0.0654, 0.0302] |
| dS_challenge | -0.0499 | [-0.1283, 0.0147] |
| slope_real_archive_constant | 0.9572 | [0.9152, 0.9901] |
| slope_real_archive_varying | 0.9571 | [0.9143, 0.9924] |
| slope_real_challenge_constant | 1.0511 | [1.0051, 1.0969] |
| slope_real_challenge_varying | 1.0426 | [1.0020, 1.0809] |

Moved families per gate cell: `{'archive_x1.0': 60, 'archive_x2.0': 60, 'challenge_high_constant': 60, 'challenge_low_constant': 60}`

Criteria: `{'magnitude_archive': 'supported', 'magnitude_challenge': 'supported', 'manipulation_checks': {'archive_unit_faithful': True, 'challenge_high_shrinks': True, 'enough_moved_families': True, 'readout_unbiased': True}, 'passed': True, 'structure_archive': 'no_effect_within_0.1', 'structure_challenge': 'inconclusive'}`

## Reserved confirmation

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| R1_challenge_high_constant | 0.7219 | [0.6422, 0.8060] |
| R_archive_x1.0 | 1.0347 | [0.9963, 1.0772] |
| R_archive_x1.0_constant | 1.0275 | [0.9826, 1.0738] |
| R_archive_x1.0_varying | 1.0420 | [1.0030, 1.0828] |
| R_archive_x1.5 | 0.9612 | [0.9263, 0.9993] |
| R_archive_x1.5_constant | 0.9589 | [0.9212, 0.9995] |
| R_archive_x1.5_varying | 0.9636 | [0.9300, 1.0019] |
| R_archive_x2.0 | 0.8807 | [0.8165, 0.9387] |
| R_archive_x2.0_constant | 0.8843 | [0.8192, 0.9437] |
| R_archive_x2.0_varying | 0.8767 | [0.8122, 0.9368] |
| R_challenge_high_constant | 0.7443 | [0.6726, 0.8204] |
| R_challenge_low_constant | 0.9127 | [0.8369, 0.9941] |
| R_challenge_m0.15_constant | n/a | n/a |
| R_challenge_m0.30_constant | 0.9127 | [0.8369, 0.9941] |
| R_challenge_m0.30_varying | 0.8872 | [0.8042, 0.9756] |
| R_challenge_m0.45_constant | 0.8069 | [0.7245, 0.8949] |
| R_challenge_m0.60_constant | 0.7443 | [0.6726, 0.8204] |
| R_challenge_m0.60_varying | 0.6675 | [0.5987, 0.7420] |
| dM_archive | 0.1540 | [0.0949, 0.2119] |
| dM_challenge | 0.1684 | [0.1075, 0.2321] |
| dS_archive | 0.0008 | [-0.0141, 0.0169] |
| dS_challenge | -0.0620 | [-0.1021, -0.0236] |
| slope_real_archive_constant | 0.9620 | [0.9324, 0.9907] |
| slope_real_archive_varying | 0.9564 | [0.9237, 0.9885] |
| slope_real_challenge_constant | 1.0071 | [0.9777, 1.0359] |
| slope_real_challenge_varying | 1.0249 | [0.9998, 1.0503] |

Moved families per gate cell: `{'archive_x1.0': 120, 'archive_x2.0': 120, 'challenge_high_constant': 120, 'challenge_low_constant': 120}`

Criteria: `{'magnitude_archive': 'supported', 'magnitude_challenge': 'supported', 'manipulation_checks': {'archive_unit_faithful': True, 'challenge_high_shrinks': True, 'enough_moved_families': True, 'readout_unbiased': True}, 'passed': True, 'structure_archive': 'no_effect_within_0.1', 'structure_challenge': 'varying_more_shrunk'}`

