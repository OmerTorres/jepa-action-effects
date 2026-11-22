# SUPPORT-001 result — what predicts the distortion beyond the training range?

> **Wording correction (2026-10-02, REPORT-FIX-001; CPU_STAGE_EVALUATION F1, F3).**
> "Beyond the training range" below should read **above the 0.60 account-fit range and the
> archive's 99.9th percentile of mean-command magnitude**. Of 399,758 sampled windows, 413
> exceed 0.60 and the largest is 0.801 (`support001-account-slopes.json`), so 0.80 is the
> extreme archive tail; absence from pretraining is not shown. Account B's figure curve is
> now its implied projection slope (derived manifest), not its mixture weight; its
> near-flat high-dose slope is an empirical property of these fits, not an algebraic
> consequence of the weight saturating. The sealed verdict is unchanged.


**Status:** complete under the D036 preregistration, on 300 fresh families (150 designed
+ 150 archive-rest). The accounts were fitted and sealed on opened TRAIN-001 records
before collection. All six real-frame measurement cells passed, including both
over-range doses. Numbers come from `docs/manifests/support001-runtime.json`; the
generated tables are in `SUPPORT-001_RESULT_DRAFT.md`.

## Question

The frozen LeWM predictor compresses block displacement more as the push command grows
(SHIFT-002). Is the distortion better predicted by:
- **A**, attenuation with command magnitude;
- **B**, reversion toward the typical archive transition, weighted by command support;
- **C**, clamping at a support-edge command?

The test is the commands beyond the training range (0.80 and 1.00; the archive's 99.9th
percentile mean command is 0.60). Every account was fitted on doses up to 0.60 only.

## Preregistered verdict: **no account established**

The three readouts agree. Squared error over the over-range branches:

| Account | Over-range error | In-range error |
| --- | ---: | ---: |
| N (global scale) | 1577 | 514 |
| A (attenuation) | 1183 | 493 |
| B (reversion) | 1361 | 492 |
| C (clamp) | **1171** | 506 |

- C beats A: C − A = −11.8 [−21.9, −3.8].
- A, B and C each beat N.
- C − B = −191 [−506, 107] and A − B = −179 [−499, 126] span zero, so the decision rule
  (beat every alternative) is not met.

## Descriptive findings (prespecified as secondary)

**1. Compression keeps growing past the training range, with no plateau.** Block
projection slope (pred vs simulator), with family-bootstrap CIs:

| Dose | 0.30 | 0.45 | 0.60 | 0.80 | 1.00 |
| --- | --- | --- | --- | --- | --- |
| Designed | 0.93 [0.84, 1.02] | 0.76 [0.69, 0.82] | 0.71 [0.65, 0.77] | 0.67 [0.60, 0.73] | 0.57 [0.50, 0.64] |
| Archive | 0.88 [0.83, 0.92] | 0.77 [0.73, 0.82] | 0.66 [0.61, 0.72] | 0.52 [0.44, 0.60] | 0.41 [0.35, 0.49] |

- The archive-scene energy ratio falls from 0.92 (0.30) to 0.28 (1.00).
- In designed scenes, alignment is 0.79–0.80 up to 0.60, 0.76 at 0.80 and 0.66 at 1.00.
  The trend is mainly in magnitude until the highest dose.
- B's fitted support covariate is already at its floor by 0.45–0.60, so B predicted a
  plateau. The continued decline contradicts that prediction. This is why B ranks last
  among A, B and C, although its interval is wide.
- This replicates the SHIFT-002 dose response on fresh families. At 0.30, 0.45 and 0.60
  in designed scenes, the pred/real ratio is 0.88, 0.74 and 0.69, against SHIFT-002's
  0.91, 0.81 and 0.74.

**2. The agent moves as commanded while the block is under-predicted.** This is the
agent probe on decoded agent displacement.
- The kinematic agent's true displacement is a fixed function of the command.
- Agent cells are reported only where the real-frame agent slope lies in (0.85, 1.15):
  designed 0.15–0.80 and archive 0.30–0.60. Agent cells outside that band are excluded,
  because the real-frame agent readout itself compresses there (slope 0.66–0.69 at 1.00),
  probably as the agent leaves the readout's training range. This gating is an extension
  of the D036 gate rule to agent cells, applied after the values were seen.

| Cell | Agent pred slope | Agent real-frame slope | Block pred slope |
| --- | ---: | ---: | ---: |
| Designed 0.60 | 1.01 | 1.01 | 0.71 |
| Designed 0.80 | 0.96 | 0.96 | 0.67 |
| Archive 0.60 | 0.96 | 0.98 | 0.66 |

Within the measurable range, the predictor reproduces the agent's commanded motion. The
compression is in the predicted *effect on the block*, not in how the command is read.

## Interpretation

- **Supported (confirmatory):** the dose response, which is now confirmed three times on
  fresh families (SHIFT-002, the TRAIN-001 frozen arm, SUPPORT-001), extends beyond the
  training range without a plateau.
- **Not established:** which account predicts it. The point estimates favour
  magnitude-based attenuation or clamping (A, C) over support-count reversion (B), but B
  is not excluded.
- **Descriptive:** agent motion is preserved where it can be measured. That locates the
  distortion in the agent–block interaction rather than the action pathway. It is not a
  causal mechanism claim.

## Limitations

- **The support covariate saturates.** With K = 512 neighbours, a zero count
  cannot distinguish "rare" from "far beyond support". A distance-to-support covariate
  could let B express a continued decline. This is a design limit of this B, not a
  refutation of reversion in general.
- One checkpoint, a two-step horizon, constant pushes, and a readout-mediated measurement
  (see ROBUST-001 for instrument agreement on the earlier cohorts).
- Account errors are dominated by large-displacement branches, so the CIs are wide.
- The agent gate was applied after the values were seen.
