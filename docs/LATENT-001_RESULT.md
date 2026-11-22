# LATENT-001 — Exploratory projected-latent correspondence result

Completed 2026-09-27 on the frozen CPU float32 checkpoint, two native steps,
and all six candidates in each of 60 READOUT tuning families (40 challenge,
20 ordinary). These families were already used in readout diagnostics. The
reserved 60 confirmation families were not loaded or scored.

`A` is the wrong-action minus matched-action projected-CLS mean squared error;
`B` is persistence minus matched-prediction error. Positive values favor the
correct action and the predictor, respectively. Units are squared projected
embedding coordinates, not workspace distance. Intervals are exploratory
2,000-resample family bootstrap percentile intervals.

| Cohort | Families | Matched L | Wrong W | Persistence P | A [95% interval] | B [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Challenge | 40 | 0.26079 | 0.86413 | 1.02008 | 0.60334 [0.55894, 0.64851] | 0.75929 [0.71914, 0.80618] |
| Ordinary | 20 | 0.03914 | 0.15172 | 0.09350 | 0.11257 [0.06351, 0.17107] | 0.05435 [0.02469, 0.09058] |
| Overall, fixed 40:20 mix | 60 | 0.18691 | 0.62666 | 0.71122 | 0.43975 [0.40560, 0.47353] | 0.52431 [0.49227, 0.55754] |

Both prespecified primary intervals exclude zero in each cohort. The tested
checkpoint therefore uses actions in a way that better matches real encoded
futures than mismatched actions do, and its terminal predictions improve on
latent persistence in this exploratory sample. These are separate claims:
predicted candidate spread alone would not establish matching.

Among 900 unordered candidate pairs, pooled contact-regime correspondence
advantages were 0.41627 for neither (59 families), 0.71416 for exactly one
(42), and 0.36520 for both (41); all family bootstrap intervals were above
zero. The exactly-one minus both contrast was 0.35945 [0.32729, 0.39224]
within 40 common families. Meaningful block-effect pairs contributed from
43 families and had A=0.61903 [0.56594, 0.67411]. Motion-regime and full
contact/effect summaries, all three paired contrasts, and endpoint subsets
are in the ignored `outputs/latent001/report.json`. These strata are
descriptive, because contact and agent motion are not independently assigned.

Coverage was 137 contacting and 127 moving candidate branches, 17 identical
real terminal frame pairs, and 17 pairs flagged as numerically collapsed.
Mean true agent displacement was 158.97 workspace units. That substantial
agent movement can dominate global CLS similarity. The correct-action signal
does **not** establish accurate block effects, contact reasoning, physical
regret, or successful planning.

Mean pairwise projected-CLS spread, averaged within families, was 0.72147
for real futures and 0.46242 for predictions overall. The challenge values
were 1.00530/0.63079 and ordinary values 0.15382/0.12567 (real/predicted).
These are diagnostics of embedding variation, not calibrated block motion.

The twelve preselected verification families passed duplicate and reversed
candidate-order inference with maximum absolute discrepancy at most 1e-6.
All 60 projected real embeddings were cache hits; 60 new two-step prediction
entries were written. Raw CLS was not loaded. The initial complete run used
7.859 seconds, including audit (0.193), scoring (7.658), and report (0.008).
A repeated evaluation reused all 60 unchanged family records and raised
cumulative run time to 9.731 seconds, within the 1,800-second cap. Final
artifacts occupied 1,115,889 bytes. The first local run was archived as
`outputs/latent001-preliminary/` after formatting and reporting fixes; its
code hash is not the final sealed run. The final result above comes exclusively
from `outputs/latent001/`.

Preselected illustrative families, chosen before model scores, were
`new:challenge-131` (A 0.57448, B 0.66741), `new:challenge-199` (0.48639,
0.66506), `new:ordinary-000` (0.06906, 0.02567), and
`new:ordinary-070` (0.06813, 0.04244). They illustrate the cohort scale;
they were not selected as evidence of a particular effect.

The original physical readout gate remains failed: the best DELTA-001
real-frame displacement model had 4.01 mean and 12.97 p95 workspace error,
versus required 1.28 and 2.56. Thus physical block competence remains
unassessable. Historical checkpoint training-simulator identity is unproven.
This case study extends established action-ablation ideas to one audited
checkpoint and distribution; it does not claim a novel diagnostic.
