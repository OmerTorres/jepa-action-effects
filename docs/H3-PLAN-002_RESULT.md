# H3-PLAN-002 result — does prediction error cost reward under a latent planning cost?

**Status:** confirmed under the sealed rule (D024/D025) on 150 fresh reserved families.
The latent-cost check passed. **H3 was not established.**


## Corrections (2026-09-29, REVIEW-BRANCH-002)

These corrections follow `BRANCH_REVIEW_2026-09-29.md` and D026. The original text
below is kept as written.

- **"Clearly better than random in challenge families" is withdrawn.** JEPA − random
  is −5.67 [−11.86, +0.24] in challenge families and −3.97 [−8.26, +0.09] pooled.
  Both CIs cross 0.
- **"The predictor's errors cost at most a few units" is withdrawn.**
  - JEPA − real latent is +2.07 [−1.32, +5.59] pooled and +3.41 [−1.43, +8.55] in
    challenge families.
  - The pooled upper bound exceeds the 5.12-unit tolerance, and no equivalence
    criterion was set.
  - Correct reading: *H3 not established; a cost above the tolerance is not
    excluded.* Failing to establish a penalty is not evidence that it is negligible.
- **What is measured.** This is the effect of *total* latent prediction error
  (including predicted agent position and block angle) under a privileged,
  candidate-specific goal cost. Matching goal images to the realized agent pose
  cancels agent position only in the real-future comparison, so the result cannot be
  attributed to block-prediction error or displacement shrinkage specifically. The
  docstring in `planning002_metrics.py` ("compares block placement") is seal-hashed
  and is corrected by this record, not by editing the file.
- **Unchanged:** the sealed verdict "H3 not established" and all numbers below.

## Question and design

Candidates are ranked by the distance between their final latent and a goal-image
latent. Each candidate gets its own goal image: the agent and block angle from that
candidate's realized final pose, and the block at the grid goal. This removes the
agent confound that made H3-PLAN-001's cost pick `hold`.

Choosers:

- predicted latents (JEPA);
- encoded real future frames (a perfect predictor under the same cost);
- random.

Regret is simulator block-to-goal distance above the best candidate. A goal counts
only if all six of its images are geometry-valid.

## Confirmed result (pooled reserved families)

148 families; 361 valid goals, 240 of them informative.

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| Regret: JEPA | 15.35 | [12.24, 18.81] |
| Regret: real latent | 13.29 | [10.23, 16.99] |
| Regret: random | 19.32 | [17.24, 21.48] |
| real latent − random (check) | −6.03 | [−9.28, −2.44] |
| **JEPA − real latent (H3)** | **+2.07** | **[−1.32, 5.59]** |
| JEPA − random | −3.97 | [−8.26, 0.09] |
| JEPA − real latent, informative goals | +2.96 | [−1.89, 7.67] |

- The check passed: the latent cost works.
- H3 was **not established**: the lower CI bound is below 0.
- Challenge cohort (98 families): JEPA − real = +3.41 [−1.43, 8.55].
- The ordinary cohort failed its own check: it has few informative goals, and the
  latent cost does not beat random there.

## Development (exploratory, 295 families)

- JEPA − real = +4.30 [2.23, 6.49].
- Real latent − random = −9.08 [−11.26, −7.00].
- The positive development effect did not replicate at a CI-separated level on fresh
  families.

## Interpretation

- Under a latent cost that works, the predictor's errors cost at most a few units of
  block-to-goal distance relative to perfect prediction. The confirmation cannot
  distinguish that from zero.
- The predictor still chooses clearly better than random in challenge families
  (−5.7, CI touching 0 pooled).
- **Choice pattern:** in challenge families the JEPA chooser concentrates on candidate
  1 (`lateral_0`, 134/224 choices), while the real-latent chooser spreads its choices
  more. This shows decision-level bias without a CI-separated regret cost.
- Together with PHYS-002/SHIFT-002: the displacement shrinkage is real and
  magnitude-dependent. Its decision consequence under this short-horizon latent cost
  is small and not confirmed.

## Limitations

- The goal images use the realized final agent pose and block angle, so this is a
  controlled measurement, not LeWM's deployable planner.
- The two-block horizon, six candidates and four grid goals keep regret coarse.
- Requiring all six images to be geometry-valid removed about 40% of goals; power
  was lower at confirmation than in development.
- One checkpoint.

## Auto-generated tables

## Development (exploratory)

### pooled (295 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 14.680 | [12.616, 17.034] |
| regret_real_latent_all | 10.380 | [8.577, 12.433] |
| regret_random_all | 19.459 | [18.123, 20.880] |
| jepa_minus_real_all | 4.300 | [2.233, 6.494] |
| real_minus_random_all | -9.079 | [-11.263, -7.001] |
| jepa_minus_random_all | -4.779 | [-7.237, -2.250] |
| regret_jepa_informative | 20.020 | [17.020, 23.099] |
| regret_real_latent_informative | 14.177 | [11.694, 17.038] |
| regret_random_informative | 26.547 | [24.780, 28.369] |
| jepa_minus_real_informative | 5.843 | [2.986, 8.686] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': True}, 'h3': 'supported', 'passed': True}`

### challenge (197 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 20.175 | [17.158, 23.425] |
| regret_real_latent_all | 14.362 | [11.710, 17.085] |
| regret_random_all | 27.898 | [26.003, 29.868] |
| jepa_minus_real_all | 5.813 | [2.778, 8.977] |
| real_minus_random_all | -13.536 | [-16.659, -10.304] |
| jepa_minus_random_all | -7.723 | [-11.227, -4.044] |
| regret_jepa_informative | 20.221 | [17.116, 23.627] |
| regret_real_latent_informative | 14.393 | [11.800, 17.381] |
| regret_random_informative | 27.969 | [26.124, 29.950] |
| jepa_minus_real_informative | 5.828 | [2.715, 8.868] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': True}, 'h3': 'supported', 'passed': True}`

### ordinary (98 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 3.634 | [1.458, 6.394] |
| regret_real_latent_all | 2.376 | [0.972, 4.317] |
| regret_random_all | 2.495 | [1.286, 4.090] |
| jepa_minus_real_all | 1.259 | [0.288, 2.443] |
| real_minus_random_all | -0.119 | [-1.094, 0.936] |
| jepa_minus_random_all | 1.139 | [-0.278, 2.949] |
| regret_jepa_informative | 18.038 | [9.116, 29.565] |
| regret_real_latent_informative | 12.045 | [5.973, 19.140] |
| regret_random_informative | 12.545 | [8.033, 17.738] |
| jepa_minus_real_informative | 5.993 | [0.923, 11.454] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': False}, 'h3': 'inconclusive_check_failed', 'passed': False}`

Goals: `{'families_without_valid_goal': 5, 'informative': 497, 'valid': 737}`

Choice histograms (candidate index): `{'challenge': {'jepa': [67, 256, 39, 15, 36, 33], 'real_latent': [129, 143, 56, 26, 54, 38]}, 'ordinary': {'jepa': [67, 49, 37, 51, 41, 46], 'real_latent': [66, 56, 45, 47, 39, 38]}}`

## Reserved confirmation

### pooled (148 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 15.354 | [12.241, 18.813] |
| regret_real_latent_all | 13.286 | [10.231, 16.986] |
| regret_random_all | 19.320 | [17.241, 21.484] |
| jepa_minus_real_all | 2.067 | [-1.324, 5.588] |
| real_minus_random_all | -6.034 | [-9.285, -2.444] |
| jepa_minus_random_all | -3.966 | [-8.259, 0.087] |
| regret_jepa_informative | 21.670 | [16.799, 26.326] |
| regret_real_latent_informative | 18.712 | [14.409, 23.346] |
| regret_random_informative | 27.229 | [24.171, 30.440] |
| jepa_minus_real_informative | 2.957 | [-1.893, 7.668] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': True}, 'h3': 'not_established', 'passed': True}`

### challenge (98 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 22.321 | [17.705, 27.288] |
| regret_real_latent_all | 18.907 | [14.577, 23.878] |
| regret_random_all | 27.991 | [25.075, 31.080] |
| jepa_minus_real_all | 3.414 | [-1.430, 8.547] |
| real_minus_random_all | -9.084 | [-14.429, -3.762] |
| jepa_minus_random_all | -5.670 | [-11.859, 0.242] |
| regret_jepa_informative | 22.321 | [17.740, 27.095] |
| regret_real_latent_informative | 18.907 | [14.390, 23.715] |
| regret_random_informative | 27.991 | [24.999, 31.147] |
| jepa_minus_real_informative | 3.414 | [-1.306, 8.149] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': True}, 'h3': 'not_established', 'passed': True}`

### ordinary (50 families)

| Quantity | Estimate | 95% family bootstrap CI |
| --- | ---: | --- |
| regret_jepa_all | 1.698 | [0.048, 4.814] |
| regret_real_latent_all | 2.270 | [0.045, 6.634] |
| regret_random_all | 2.324 | [0.357, 4.941] |
| jepa_minus_real_all | -0.572 | [-1.767, 0.334] |
| real_minus_random_all | -0.054 | [-2.352, 2.398] |
| jepa_minus_random_all | -0.626 | [-2.387, 0.978] |
| regret_jepa_informative | 12.560 | [0.254, 35.991] |
| regret_real_latent_informative | 15.996 | [0.153, 47.581] |
| regret_random_informative | 16.566 | [3.884, 30.611] |
| jepa_minus_real_informative | -3.435 | [-11.950, 4.300] |

Criteria: `{'checks': {'enough_families': True, 'latent_cost_beats_random': False}, 'h3': 'inconclusive_check_failed', 'passed': False}`

Goals: `{'families_without_valid_goal': 2, 'informative': 240, 'valid': 361}`

Choice histograms (candidate index): `{'challenge': {'jepa': [31, 134, 20, 10, 12, 17], 'real_latent': [63, 79, 25, 16, 21, 20]}, 'ordinary': {'jepa': [25, 11, 22, 23, 22, 34], 'real_latent': [26, 16, 22, 19, 24, 30]}}`

