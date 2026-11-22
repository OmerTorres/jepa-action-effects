# TRAIN-002 result — does removing the compression need the high dose in training?

> **Wording correction (2026-10-02, REPORT-FIX-001; CPU_STAGE_EVALUATION F2).** Numbers
> below are unchanged. Read "Generic fine-tuning does nothing" as **no generic fine-tuning
> gain was established** (−0.47 [−1.54, 0.63] is not an equivalence test), and "exactly as
> in the frozen model" as R within 0.01 of frozen. Fine-tuning **substantially reduces**
> rather than removes the compression (R 0.87–0.93). "Beyond the training range" means
> beyond the doses used in fine-tuning. The omission arm had 48 rather than 64 effective
> examples per update (D035), so the small deficit does not isolate exact-dose coverage,
> and these interventions motivate, but do not establish, a pretraining-support cause.


**Status:** complete under D035, on fresh families (120 designed, 120 archive-rest, 120
ordinary windows).
- All six new runs passed the optimization gate, and the six reused TRAIN-001 runs matched
  their seal.
- The real-frame measurement gate passed in all four cells before any predictor was
  scored. Floors were 0.17–0.23.

Numbers come from `docs/manifests/train002-runtime.json`; the generated tables are in
`TRAIN-002_RESULT_DRAFT.md`.

## Design (recap)

All arms fine-tune the predictor, action encoder and prediction projector (11.7M
parameters) from the released checkpoint: 2,000 updates, LR 1e-5, seeds 20261111–13.

| Arm | Training data | Role |
| --- | --- | --- |
| balanced (TRAIN-001) | designed + archive-rest constant pushes 0.15–0.60 | in-support reference |
| archive_weighted (TRAIN-001) | same, archive magnitude weights | descriptive |
| **dose_omitted** | same batches, weight 0 on 0.60 | held-out-dose test (paired with balanced) |
| **archive_only** | 2,400 fresh natural archive windows | generic fine-tuning control |

## Primary (D035): **held-out-dose deficit supported, but small**

Mean terminal block error at 0.60, in world units:

| Arm | Error |
| --- | ---: |
| frozen | 26.84 |
| balanced | 16.21 |
| dose_omitted | 17.03 |

The paired contrast dose_omitted − balanced at 0.60 is **+0.82 [0.11, 1.63]**.
- Per seed: 0.74, 0.93 and 0.77.
- All three readouts give the same sign: 1.26 [0.63, 1.99], 0.82 [0.11, 1.63] and
  0.78 [0.05, 1.59].
- At the in-support 0.45 dose, the same contrast is 0.04 [−0.28, 0.36]. This manipulation
  check passes: omitting 0.60 does not change in-support accuracy.

**Transfer fraction at 0.60: 0.92 [0.85, 0.99].** Training without the 0.60 dose recovers
92% of the in-support gain at 0.60. The deficit is real but small, about 8% of the gain.

## Secondary

**Generic fine-tuning does nothing** (`generic_finetuning_gain_not_established`):
- frozen − archive_only at 0.60 is −0.47 [−1.54, 0.63];
- archive_only − balanced is +11.10 [8.37, 14.09].

| Arm | R, designed 0.45 | R, designed 0.60 | R, archive 0.45 | R, archive 0.60 |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.79 | 0.71 | 0.73 | 0.64 |
| archive_only | 0.82 | 0.71 | 0.72 | 0.65 |
| dose_omitted | 0.92 | 0.89 | 0.92 | 0.87 |
| balanced | 0.92 | 0.92 | 0.93 | 0.90 |

Here R is the pred/real projection-slope ratio. Archive-only fine-tuning leaves the
compression exactly as in the frozen model. Fine-tuning on the designed pushes removes
most of it, including at the held-out dose.

**Latent.** Terminal latent MSE at 0.60, dose_omitted − balanced: +0.032 [0.019, 0.046].
That is 15% relative, a larger relative deficit than the physical one.

**Ordinary windows.** Error relative to frozen:

| Arm | Change |
| --- | --- |
| balanced | +1.08 [0.28, 1.91] |
| dose_omitted | +0.93 [0.08, 1.80] |
| archive_weighted | +0.83 [0.02, 1.67] |
| archive_only | −0.39 [−1.00, 0.25] |

The ordinary-case cost of designed-push fine-tuning replicates TRAIN-001's +1.05
[0.11, 1.99] on fresh windows.

## Interpretation

- **Supported (confirmatory):** omitting the 0.60 dose leaves a small, CI-separated
  held-out deficit.
- **Descriptive, but strong:** about 92% of the gain transfers one dose step beyond the
  training range. So the fine-tuning fix does not require the exact high dose.
- **Supported (secondary):** generic fine-tuning on the pretraining distribution does not
  reduce the compression. Together with the above, what removes it is exposure to the
  designed strong-push interactions, which the expert archive rarely contains, rather than
  extra training as such.
  - This fits a coverage-of-strong-interactions account better than one tied to the exact
    command magnitude. That account is a hypothesis for PRETRAIN-001 and MECH-002, not a
    result.
- Trade-off: designed-push fine-tuning costs about 1 unit on ordinary windows, replicated.

## Limitations

- One checkpoint, a two-step horizon, constant pushes, readout-mediated measurement.
- Dose_omitted sees 48 effective examples per update against balanced's 64 (disclosed in
  D035).
- The archive-only arm uses different data, not a paired batch sequence.
- The held-out dose sits one step above the trained range; transfer to 0.80–1.00 is
  untested here. SUPPORT-001's over-range families could be scored exploratorily.
