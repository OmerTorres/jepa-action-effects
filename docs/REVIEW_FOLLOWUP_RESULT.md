# REVIEW-FIX-001 — exploratory follow-up to REVIEW-BRANCH-002

**Status:** exploratory. All families involved were already opened, so nothing here
changes a sealed verdict (D026).

**Code:** `review_followup.py` and `followup_metrics.py` (pure functions and tests),
plus `provenance.py`. Numbers come from `docs/manifests/review-followup-runtime.json`
and `docs/manifests/provenance-check.json`. Family-cluster bootstrap, 2,000 draws.


## Corrections (2026-09-29, REVIEW-BRANCH-003)

These follow `REVIEW_FOLLOWUP_AUDIT.md` and D027. The original text below is kept.

- **The readout-bias explanation in §4 is reversed.** R = slope_pred / slope_real, so
  a falling real-frame slope *raises* R and a rising one *lowers* it.
  - Challenge slope_real falls from 1.045 to 0.993 (0.30 → 0.60, constant). This
    **opposes** the observed decline in R and cannot produce it.
  - Archive slope_real rises from 0.939 to 0.967 (×1 → ×2, constant). This
    **contributes** a little to the decline in R, rather than working against it.
  - The raw predictor slopes fall in both cases (point estimates, not new confirmed
    contrasts):

    | Constant controls | slope_pred low → high | R low → high |
    | --- | --- | --- |
    | Challenge 0.30 → 0.60 | 0.954 → 0.739 | 0.913 → 0.744 |
    | Archive ×1 → ×2 | 0.964 → 0.855 | 1.028 → 0.884 |

    So the trend is not produced by the readout denominator.
- **Length ratio, direction error and projection slope use different weightings.**
  They describe the error side by side and are not an exact multiplicative
  decomposition of the 0.56 slope.
- **Zero predicted vectors now have an undefined direction.** They are excluded from
  the angle statistics and counted. The regenerated manifest has 0 such vectors, and
  every reported number is unchanged.

## 1. Independent rerun — the sealed results reproduce exactly

| Rerun | Families | Max difference vs sealed |
| --- | ---: | --- |
| PHYS-002 reserved, re-scored from scratch (encode, predict, decode) | 150 | records 0.0; embeddings 0.0; re-aggregated summary 0.0 |
| SHIFT-002 reserved subset (15 challenge + 15 archive) | 30 | records 0.0 |

The re-aggregated PHYS-002 criteria are unchanged: floor passed, physical failure
supported, shrinkage supported, H1 not established.

The provenance validator, run read-only, passes every check for PHYS-002, SHIFT-002
and H3-PLAN-002. SHIFT-002 records never stored an input hash, and the validator
reports that as "not recorded".

## 2. Absolute endpoint vs displacement error (PHYS-002, D026 finding 4)

| Reserved challenge (100 families) | Predicted | Real frame |
| --- | ---: | ---: |
| Absolute endpoint ‖decoded_end − true_end‖ | 17.895 [16.884, 18.969] | 3.613 [3.411, 3.819] |
| Displacement error (published as `endpoint_err`) | 17.857 [16.806, 18.945] | 3.935 [3.643, 4.218] |

Both match the review and the sealed values. Development gives the same picture:
absolute 18.13 / 3.66 and displacement 17.90 / 4.17. Shared position bias is small, so
the substantive result is unchanged.

## 3. What the 0.56 projection slope is made of (review: angle vs magnitude)

Moved candidates on the PHYS-002 reserved challenge cohort:

| Quantity | Predicted | Real frame |
| --- | ---: | ---: |
| Projection slope Σ(p·t)/Σ(t·t) | 0.561 | 1.006 |
| Pooled length ratio Σ\|p\|/Σ\|t\| | 0.629 [0.583, 0.673] | 1.011 |
| Median length ratio | 0.646 | — |
| Mean \|direction error\| | 19.0° [16.1, 22.4] | 4.9° |
| Median \|direction error\| | 11.8° | 2.8° |

The slope combines two effects:
- predicted displacements that are shorter (about 63% of true length);
- a direction error roughly four times the readout's own.

"Moves the block 56% as far" (withdrawn in D026) conflated the two.

## 4. SHIFT-002 per dose: readout, slope and decomposition (reserved)

| Cell | Real-frame slope | Pred/real ratio | Pred length ratio | Mean \|angle\| |
| --- | --- | --- | ---: | ---: |
| Challenge 0.30 constant | 1.045 [1.008, 1.084] | 0.913 [0.836, 0.990] | 1.27 | 26.6° |
| Challenge 0.45 constant | 0.995 [0.965, 1.026] | 0.807 [0.727, 0.893] | 1.03 | 25.8° |
| Challenge 0.60 constant | 0.993 [0.964, 1.022] | 0.744 [0.672, 0.823] | 0.96 | 27.5° |
| Challenge 0.60 varying | 1.013 [0.989, 1.038] | 0.667 [0.598, 0.736] | 0.88 | 27.7° |
| Archive ×1.0 constant | 0.939 [0.883, 0.985] | 1.028 [0.981, 1.071] | 1.10 | 31.5° |
| Archive ×1.0 varying | 0.933 [0.876, 0.981] | 1.042 [1.005, 1.084] | 1.12 | 34.4° |
| Archive ×1.5 constant | 0.963 [0.921, 1.002] | 0.959 [0.923, 1.000] | 1.01 | 23.0° |
| Archive ×2.0 constant | 0.967 [0.939, 0.993] | 0.884 [0.819, 0.942] | 0.95 | 17.5° |
| Archive ×2.0 varying | 0.963 [0.931, 0.993] | 0.877 [0.811, 0.939] | 0.94 | 19.9° |

**Readout by dose (finding 2).**
- Real-frame slopes stay within the 0.85–1.15 band in every dose × structure cell.
- The challenge readout slope falls from 1.045 to 0.99 as the dose rises. Some of the
  ratio trend could come from that, but it is far smaller than the ratio change
  (0.91 → 0.74).
- Archive readout slopes rise slightly with dose (0.94 → 0.97), which works
  *against* the observed trend.
- Several archive CIs exclude 1, so the readout is near-unbiased but not exactly
  unbiased.

**Decomposition (new, exploratory).**
- In challenge scenes, the predicted length ratio falls with dose, from 1.27 (small
  pushes over-predicted) to about 0.96. The direction error stays about 26–28°.
- In archive scenes both fall with scale: length 1.11 → 0.95 and angle 33° → 18°.
- So the dose trend is mainly **compression of predicted displacement length**
  (lengths grow less than proportionally with the push) together with a large,
  roughly dose-independent direction error in challenge scenes. It is not uniform
  "shortening".
- SHIFT-002 manipulated command magnitude, and true displacement co-varies with it.
  These data cannot separate a command-magnitude effect from a
  displacement-size effect.

## 5. Archive control with episode-clustered intervals (finding 6)

Replaying the audit's seeded sampling reproduces it exactly: 600 windows from 584
episodes, and 335 moved windows from 331 episodes.

| Moved windows | Window bootstrap | Episode bootstrap |
| --- | --- | --- |
| slope_pred 0.963 | [0.933, 0.995] | [0.932, 0.993] |
| slope_real 0.950 | [0.921, 0.975] | [0.922, 0.976] |
| ratio 1.014 | [0.996, 1.032] | [0.996, 1.033] |

Clustering by episode leaves the intervals practically unchanged, because nearly
every episode contributes one window. Predicted length ratio: 1.05; mean direction
error 25.5° (median 11.1°).

## Limitations

- **Exploratory.** Everything here uses opened families.
- **One readout.** The decomposition uses the same frozen READOUT-002 probe, so the
  direction errors include readout noise; the real-frame rows show its size.
- **No mechanism established.** The length/direction split describes the error, not
  its cause.
