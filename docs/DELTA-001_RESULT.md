# DELTA-001 result — displacement readout did not pass

Executed 2026-09-27 as an exploratory measurement follow-up. The frozen
two-step checkpoint/simulator pair, all 240 existing fit families, and the same
60 tuning families were used. No new families were collected. None of the fixed
readout candidates passed the real-frame measurement gate, so the 60 reserved
confirmation families were not evaluated and frozen-predictor physical evaluation
was unavailable.

| Readout | Tuning endpoint mean / p95 | Meaningful paired-effect mean / p95 | Gate |
| --- | ---: | ---: | --- |
| Zero displacement | 19.04 / 97.23 | 51.84 / 101.55 | Fail |
| Existing absolute ridge, differenced | 9.39 / 26.78 | 13.33 / 27.86 | Fail |
| Direct displacement ridge | 4.83 / 13.03 | 8.54 / 17.55 | Fail |
| Anchored nonlinear seed 20260931 | 4.29 / 14.64 | 8.79 / 21.56 | Fail |
| Anchored nonlinear seed 20260932 | 4.39 / 14.38 | 8.75 / 20.59 | Fail |
| Anchored nonlinear seed 20260933 | 4.01 / 12.97 | 8.25 / 20.59 | Fail |

The predeclared limits were mean `<=1.28` and family-weighted p95 `<=2.56`,
with at least ten contributing families in each required subset. Forty-three of
the 60 tuning families contributed meaningful candidate pairs. The best seed
also failed the ordinary cohort and all pooled contact-regime endpoint gates.
Full required-subset counts and scores are in `outputs/delta001/fit-report.json`.
The reporting addendum `outputs/delta001/cohort-regime-breakdown.json` gives
endpoint and paired-effect errors within each cohort and contact regime;
sparse cells are marked unassessable.

The audit verified upstream hashes, the 240/60/60 partition, two-step horizon,
and absence of confirmation-family embeddings in READOUT-001's development cache.
Metadata and artifact hashes were inspected; confirmation outcomes did not enter
fitting, selection, or evaluation. Fit deduplication retained 1,424 real
transition pairs, removed 16 identical-frame pairs, and found no conflicting
block-displacement labels. A repeated fit-stage invocation reused the sealed
result. `confirm` returned `unavailable` without opening confirmation labels;
`evaluate` returned `unavailable` without predictor inference.

The probe improvements over zero displacement show that these measurements
contain some useful signal. Their errors remain too large for the intended
fine-grained physical claims. This does not prove that the representation lacks
block information or that the frozen predictor fails at physics. Per the
predeclared stopping rule, close this bounded physical-measurement recovery
route and keep main-study expansion paused. A latent-space question would need
its own separately defined controls and claim limits.

Generated artifacts, including curves, model files, stage manifests, and the
full report, are ignored under `outputs/delta001/`. Hashes, runtime, test
results, and limitations are tracked in `docs/manifests/delta001-runtime.json`.
