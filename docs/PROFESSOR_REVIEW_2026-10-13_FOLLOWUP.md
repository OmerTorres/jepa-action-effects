# Course report follow-up review — 2026-10-13

Reviewed merged main **ba25aed** (PR #2), whose tree equals the clean checked-out
**b3123a5**. Scope: REPORT-FIX-002 relative to the previous course review, the revised
five-page report, new reporting/recomputation utilities, and submission readiness.
The professor's explicit instruction to discuss contributions verbally remains
accepted; no written-contributions penalty or request is reinstated.

## Assessment

**Provisional grade: 91/100 for the written package and supporting work.** The
revision substantially improves accessibility: the data and arm tables, loss and
metric definitions, absolute-error comparison, training curves, and runnable
saved-record recomputation make the project much easier to evaluate. The earlier
90/100 already credited the underlying experimental depth; this revision does
not change the scientific scope or establish a new finding.

Under the same reviewer-selected rubric: technical quality 34/35 (the reproducibility
handoff is stronger), originality 12/15, significance 12/15, insights 19/20,
written communication/compliance 14/15. These are judgment calls, not official
weights. The oral presentation and individual mastery remain ungraded.

The project has ample substance for the course. Defer further GPU work. Before
submission, correct the reporting issues below; none needs new experiments or
changes to sealed estimates. A strong final report/defense could reach the low
90s, but scope limitations remain: one checkpoint, one environment, a short
horizon, shared measurement infrastructure, no established mechanism or planning
penalty, and the omission-arm effective-sample-count confound.

## Verified improvements

- The report is **five pages including references**. All five rendered pages were
  inspected: no clipping or overlapping text. Contributions are handled verbally
  as instructed by the professor.
- Training data, four fine-tuning arms, model updates, frozen modules and the
  autoregressive training loss are now explained. The loss matches the implemented
  scene/family/dose/step averaging for the family-based arms; the archive-only
  exception is stated in Table 2.
- TRAIN-002 now leads the fine-tuning discussion, with absolute errors and the
  ordinary-window tradeoff. TRAIN-001's failed development gate stays explicit.
- The structure-null and aggregate-agent wording flagged previously are corrected.
- `train002_recompute` provides an actual read-only command that checks input/model/
  readout identities and regenerates the entire result from saved records, rather
  than returning a previously sealed summary.
- Training curves are derived from twelve hash-checked saved runs. The figure
  correctly warns that training-loss levels are not comparable across arms.
- The new code does not change the sealed training/evaluation source files inspected
  in this review, and the numerical results remain identical.

## Corrections before submission

### F1 — P2: the measurement-gate description misstates SHIFT-002's protocol

`reports/report001/main.tex:223–228` says the floor must be below 0.25 and that,
from the dose-response experiment onward, each scene × dose cell additionally
passes the slope band. That is not the original SHIFT-002 gate.

In `shift002_metrics.py:212–231`, the real-frame slope bands pool doses within
scene × structure; the minimum-count checks use selected low/high groups.
SHIFT-002 does **not** require the 0.25 displacement-error floor. D023 explicitly
replaced that floor, and D026 records that the original readout gate pools doses.
Later per-dose checks are separate evidence, not the original sealed decision rule.
The higher-dose cells gated in TRAIN/SUPPORT also should not be described as every
plotted cell.

**Fix:** name the gates by study: the strong-push floor gate; SHIFT-002's pooled
slope-band/count/manipulation checks; then the later designated scene × dose
floor/band/count checks. Preserve the TRAIN-001 development failure. This is a
report correction, not a request to change the original protocol or numbers.

### F2 — P2: the new metric definitions overgeneralize sample selection and bias correction

`main.tex:207–220` introduces all listed metrics as computed over moved families.
That matches the SHIFT/TRAIN slope ratios but not all other reported estimands:
`calib001_metrics.estimates` uses unmasked statistics for `b_rev` and `energy_ratio`,
with separate `_moved` fields, while TRAIN-002 endpoint errors use all accepted
families. SUPPORT's archive 0.15 cell, for example, contains 150 families but only
98 moved ones; the reported all-family slope is 0.915688 versus 0.912915 for the
moved-only slope. This distinction matters even though high-dose headline cells
have all families moving.

The phrase “divides out readout bias” at line 213 is also too strong. Dividing by
the real-frame slope normalizes against that control; it does not establish that
a nonlinear readout has the same bias on predicted latents. A slope of one alone
also does not prove all motion is faithful (orthogonal errors can remain).

**Fix:** remove the blanket moved-family preamble; give the subset for each metric
or study and say “normalizes by the real-frame slope; predicted-latent readout
bias may remain.” Label one as faithful *projection gain* where appropriate.

### F3 — P2: lower flag rates do not establish that flagged latents do not drive the result

`main.tex:290–292` now correctly avoids claiming that predictions lie on the true
latent manifold, but adds that high-dose readings “are not driven by” flagged
latents. A smaller flagged fraction says nothing by itself about those points'
influence on a magnitude/energy statistic. A small number of large errors can
dominate it.

The saved ROBUST-001 sensitivity analysis is explicitly inconclusive under its
all-instruments rule: after restricting to unflagged predictions, the challenge
ridge contrast is −0.36 [−0.88, 0.09]. Four other instruments retain negative
CI-separated contrasts, which is useful partial corroboration.

**Fix:** keep the falling flag rates, delete the “not driven by” inference, and
briefly state that the unflagged-only robustness check was inconclusive across
all instruments. No need to rerun this analysis.

### F4 — P3: an inconclusive contrast is described as no deficit

`main.tex:325–327` says the omission arm has a deficit at 0.60 “but none” at 0.45.
The latter interval, 0.04 [−0.28, 0.36], establishes neither equality nor absence.
Use “no deficit was established at 0.45.” This is the same distinction now
correctly made for the action-structure and archive-only comparisons.

## Presentation polish and handoff

- Figure 1's labels are roughly 5.5–6 points at final size, noticeably smaller
  than body text. Increase/simplify them. The true-future path would be clearer
  with an explicit route through the shared readout. The figure is a schematic,
  not an observed individual trajectory; retain that label and do not interpret
  aggregate R × median displacement as a measured representative prediction.
- The broad Table 1 statement that all splits are disjoint from every earlier
  study should explicitly allow the deliberately reused TRAIN-001 development
  data and trained models in TRAIN-002. Fresh confirmation families are disjoint;
  the training data are intentionally shared.
- The half-empty last page is acceptable and leaves room to improve diagram
  readability. No need to force the report back to three or four pages.
- A teammate clean-clone run remains a useful readiness check. The handout asks
  for reproducibility and clear instructions; it does not explicitly mandate
  that Inbal personally perform this particular six-step checklist. The existing
  dry run reused the local environment, so fresh-machine setup remains unverified.

## Checks performed

- `git fetch origin` → main advanced to ba25aed; `git diff --stat HEAD origin/main`
  empty. Initial working tree clean.
- `make check && make test` → passed; **354 tests passed**, 25 existing warnings,
  pytest 17.32 s, Python 3.10.21.
- `.venv/bin/python -m jepa_action_effects.train002_recompute` → **match**, 5,040
  records, 14 model/control entries, full summary equal to sealed and tracked
  results, no differences.
- Inline Python: all **75** `reporting.numbers()` values reproduce saved
  `numbers.tex` and the claims ledger exactly. `training_curves.collect()` verifies
  all twelve registered run hashes and equals the tracked curve manifest.
- `reporting.build(Path('/private/tmp/jepa-review-round2/assets'))` → 75 numbers
  and five figures. It creates the PDF output directory successfully. Local
  font-cache warnings were nonfatal; asset generation completed.
- `/opt/homebrew/bin/pdfinfo outputs/report001/pdf/main.pdf` → five pages;
  `/opt/homebrew/bin/pdftoppm -scale-to 1800 -png outputs/report001/pdf/main.pdf
  /private/tmp/jepa-review-round2/report` → all five pages visually reviewed.
- Targeted comparison of current TeX with `train001_model`, `train001_metrics`,
  `train002_metrics`, `shift002_metrics`, `calib001_metrics`, D023/D026 and the
  saved ROBUST-001 result identifies F1–F4. No new inference, training or collection.

Only review/task/logbook documentation is changed by this review. Report fixes
are recommendations, not silently applied edits. Earlier literature verification
was not repeated because the cited related-work passage is materially unchanged.

## PR #3 verification — 2026-10-13

Reviewed [PR #3](https://github.com/OmerTorres/jepa-action-effects/pull/3) at
**efff5d337789267b8afbf604610ab9afd70af239**, against main **ba25aed**.
The local checkout matches the remote PR head. GitHub reports mergeable, with no
status checks listed; this review's test results are local checks, not CI.

**Disposition:** F1's SHIFT-002 gate correction, F2, F3 and F4 are implemented,
and the Figure 1/Table 1 polish is verified. One residual method-definition issue
remains in the same gate paragraph; correct it before submission. This is not a
numerical failure or a request for new experiments. Provisional written-package
assessment is now approximately **92/100**; presentation remains ungraded.

### F5 — P2: distinguish the PHYS-002 paired-effect floor from later displacement floors

`reports/report001/main.tex:231–233` defines the floor as mean real-frame displacement
error divided by median moved true displacement, then applies that definition to the
strong-push study. That definition describes the later TRAIN/SUPPORT gate, but not
the PHYS-002 gate that supported its headline result.

`next_metrics.floor_ratio` (lines 358–365) uses the `boundary_meaningful` pairs:
the family-mean real-frame **paired action-effect** error divided by the median
true **paired effect** magnitude. Pair construction at lines 207–235 compares
branch endpoint differences, not each endpoint's displacement from its initial
state. `PHYS-002_RESULT.md` reports this paired-effect floor as 0.120. In contrast,
`train001_metrics.cell_gate` uses mean displacement error over accepted families
divided by median displacement among moved families. The two quantities need
not agree: an endpoint offset shared by both branches cancels in the pair effect.

Suggested wording: “The strong-push study uses a paired-effect floor: family-mean
real-frame error in meaningful designated branch-pair effects, divided by their
median true effect magnitude. Later fine-tuning and beyond-range studies use mean
real-frame displacement error over accepted families divided by the median true
displacement among moved families. Both floor thresholds are 0.25.” Keep the
corrected SHIFT-002 description, which has no floor gate.

This distinction was missed in the previous review; it is a remaining description
error, not evidence that the original gate was implemented incorrectly.

### Verification and limits

- `make check && make test`: **354 passed**, 25 existing warnings, 16.18 s pytest.
- All **76 macros** and the claims ledger reproduce exactly. Every previous ledger
  line is preserved; the new unflagged-ridge interval is −0.36 [−0.88, 0.09].
- `reporting.build(Path('/private/tmp/jepa-pr3-review/assets'))`: 76 numbers and
  five figures generated successfully with local plotting caches.
- All five pages of the saved report PDF inspected after `pdftoppm` rendering.
  Figure 1 is larger and clearer, its real-future path is explicit, and its example
  is labelled schematic. No clipping/overlap; five pages including references.
- PR file list confirms no sealed manifests, experiment runners, or estimators
  changed. The previously verified 5,040-record computation was not repeated for
  this reporting-only patch. `git diff --check origin/main...HEAD` passes.
- No report edits, GitHub review/comments, merge, new inference or training were
  performed. Only local review/task/logbook documentation was updated.
