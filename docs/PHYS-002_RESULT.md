# PHYS-002 result — predictor against the real-frame readout floor

**Status:** confirmed under sealed criteria (D021) on 150 fresh reserved families.
The exploratory audit shows the effect is specific to the designed challenge
distribution (D022).


## Corrections (2026-09-29, REVIEW-BRANCH-002)

These corrections follow `BRANCH_REVIEW_2026-09-29.md` and D026. The original text
below is kept as written.

- **"Moves the block about 56% as far" is an overstatement.** The measured quantity
  is the pooled, readout-decoded projection slope of predicted on true displacement,
  Σ(pred·true)/Σ(true·true) = 0.56. It weights large displacements more, and a low
  value can reflect direction errors as well as shorter predicted vectors. The split
  into length and direction is reported in REVIEW_FOLLOWUP_RESULT.md.
- **"This is not a probe or data artifact" is too categorical.** The controls ruled
  out:
  - a one-step alignment offset;
  - dependence on probe training size (2k–50k frames);
  - real-frame attenuation (slope_real ≈ 1).

  They do not fully validate a nonlinear readout applied to predicted latents that
  may lie off the real-frame manifold, and global latent contraction does not
  uniquely identify block motion.
- **"Endpoint error" rows are displacement errors.**
  `endpoint_err_pred/real` is ‖(decoded_end − decoded_start) − (true_end −
  true_start)‖, which cancels shared position bias. The true absolute endpoint error
  ‖decoded_end − true_end‖ on the reserved challenge cohort is 17.895 (predicted) /
  3.613 (real frame), per the review; it is recomputed in REVIEW_FOLLOWUP_RESULT.md.
  The substantive result is unchanged.
- **Unchanged:** the sealed D021 verdicts and all numbers below.

## Question and protocol

Does the frozen LeWM predictor get block displacement right, measured against the
error of the same frozen readout on real future frames?

- **Readout:** READOUT-002 `mlp-20261005`, trained on 50k random-pose frames.
  Validation block error 3.48 mean / 7.15 p95; 4.41 / 9.80 on real study frames.
- **Setting:** two-block horizon. Displacements are decoded relative to
  `readout(z_t)`.
- **Families:** development on 300 families (exploratory). Confirmation on 100
  challenge + 50 ordinary families under master seed 20261010, disjoint by reset seed
  from all earlier families, opened once.

## Confirmed result (challenge cohort, 100 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| Floor ratio (real-frame effect error / median effect) | 0.120 | [0.113, 0.128] |
| D_boundary (predicted − real-frame effect error) | 17.05 | [15.85, 18.15] |
| slope_pred / slope_real | 0.561 / 1.006 | difference [−0.490, −0.401] |
| h1_contrast (exactly one − both contact) | −12.37 | [−14.25, −10.49] |
| Endpoint error predicted / real frame | 17.86 / 3.93 | |
| Regret JEPA / real frame / random | 3.96 / 0.19 / 23.9 | |

Verdicts:

- floor passed;
- physical failure **supported**;
- shrinkage **supported**;
- H1 **not established**.

The ordinary cohort failed the floor (0.35) and is inconclusive. Development gave the
same direction (slope_pred 0.592, D_boundary 16.7).

## Exploratory audit (after opening; `phys002_audit.py`)

| Check | Result |
| --- | --- |
| Time alignment | Real frames: slope 1.06 (step 1) / 1.01 (step 2). Predicted: 0.88 / 0.56. No one-step offset; the shortfall appears at the second autoregressive step. |
| Probe size | pred/real slope ratio 0.557, 0.550, 0.563, 0.581, 0.575 for 2k, 5k, 10k, 20k, 50k training frames. |
| Latent space (readout-free) | Step-2 slope 0.78–0.79; predictor/persistence MSE ratio 0.24. |
| Archive control (600 windows) | slope_pred 0.96 [0.93, 1.00] vs slope_real 0.95; block starting at rest: 1.02 [0.96, 1.09]. |
| Matched effect size (development) | Effect error ≈ 0.55–0.68 × effect in both contact regimes. |

## Interpretation and limitations

- On the designed challenge families, the predictor moves the block about 56% as far
  as it really moves. This is not a probe or data artifact.
- The same code on training-archive windows shows no shrinkage. The claim is
  therefore limited to this distribution. The candidate causes are the constant,
  high-magnitude 5-control pushes and the challenge scene layouts (SHIFT-001).
- H1's reversal is explained by effect size.
- Every moved block was contacted, so contact and motion cannot be separated.
- One checkpoint and one horizon.
- Regret here uses the readout, not the model's own planning cost (H3-PLAN-001).

## Auto-generated tables

## Development (exploratory): challenge cohort

| Quantity | Estimate | 95% family bootstrap CI | Families |
| --- | ---: | --- | ---: |
| floor_ratio | 0.1212 | [0.1155, 0.1268] | 200 |
| D_boundary | 16.6763 | [15.8753, 17.4670] | 200 |
| D_exactly_one | 11.7966 | [10.9250, 12.7214] | 200 |
| D_both | 23.0501 | [21.8942, 24.2744] | 200 |
| h1_contrast | -11.2535 | [-12.5341, -10.0174] | 200 |
| slope_pred | 0.5916 | [0.5573, 0.6268] | 200 |
| slope_real | 1.0142 | [1.0059, 1.0227] | 200 |
| slope_pred_minus_real | -0.4226 | [-0.4539, -0.3887] | 200 |
| slope_pred_contact | 0.5916 | [0.5578, 0.6233] | 200 |
| slope_pred_no_contact | n/a | n/a | 200 |
| endpoint_err_pred | 17.8967 | [17.1563, 18.6772] | 200 |
| endpoint_err_real | 4.1662 | [3.9177, 4.4621] | 200 |
| regret_jepa | 2.9643 | [2.3642, 3.5849] | 200 |
| regret_real_frame | 0.2130 | [0.1512, 0.2797] | 200 |
| regret_random | 24.5023 | [23.4531, 25.6810] | 200 |

Criteria: `{'floor': 'passed', 'h1': 'not_established', 'physical_failure': 'supported', 'shrinkage': 'supported'}`

## Development (exploratory): pooled slope and endpoint rows

| Quantity | Estimate | 95% family bootstrap CI | Families |
| --- | ---: | --- | ---: |
| slope_pred | 0.5956 | [0.5629, 0.6295] | 300 |
| slope_real | 1.0134 | [1.0054, 1.0218] | 300 |
| endpoint_err_pred | 14.1494 | [13.5896, 14.7261] | 300 |

## Reserved confirmation: challenge cohort

| Quantity | Estimate | 95% family bootstrap CI | Families |
| --- | ---: | --- | ---: |
| floor_ratio | 0.1200 | [0.1125, 0.1284] | 100 |
| D_boundary | 17.0460 | [15.8492, 18.1497] | 100 |
| D_exactly_one | 11.6206 | [10.3602, 12.8352] | 100 |
| D_both | 23.9904 | [22.2223, 25.8639] | 100 |
| h1_contrast | -12.3698 | [-14.2533, -10.4880] | 100 |
| slope_pred | 0.5610 | [0.5167, 0.6048] | 100 |
| slope_real | 1.0063 | [0.9951, 1.0182] | 100 |
| slope_pred_minus_real | -0.4453 | [-0.4900, -0.4009] | 100 |
| slope_pred_contact | 0.5610 | [0.5184, 0.6062] | 100 |
| slope_pred_no_contact | n/a | n/a | 100 |
| endpoint_err_pred | 17.8566 | [16.7972, 18.9728] | 100 |
| endpoint_err_real | 3.9348 | [3.6245, 4.2361] | 100 |
| regret_jepa | 3.9634 | [2.7767, 5.3413] | 100 |
| regret_real_frame | 0.1936 | [0.1170, 0.2807] | 100 |
| regret_random | 23.9086 | [22.6424, 25.2573] | 100 |

Criteria: `{'floor': 'passed', 'h1': 'not_established', 'physical_failure': 'supported', 'shrinkage': 'supported'}`

## Reserved confirmation: pooled slope and endpoint rows

| Quantity | Estimate | 95% family bootstrap CI | Families |
| --- | ---: | --- | ---: |
| slope_pred | 0.5754 | [0.5314, 0.6168] | 150 |
| slope_real | 1.0062 | [0.9955, 1.0178] | 150 |
| endpoint_err_pred | 13.8858 | [13.1650, 14.6557] | 150 |
