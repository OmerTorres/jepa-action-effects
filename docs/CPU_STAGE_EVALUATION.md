# Evaluation of the completed CPU study cycle — 2026-10-02

Reviewed **9060a58**, initially with a clean working tree. Scope: evaluate the current
evidence, recent implementations and remaining work; no new experiments, training or
protocol changes. This is a targeted review of the latest study cycle, not a claim to
have independently rerun every historical experiment.

## Assessment

The project now has a substantial, reproducible **single-checkpoint empirical study**.
Its strongest contribution is a controlled dose response plus useful fine-tuning
comparisons, with explicit measurement controls and negative results. It has ample
research content for the course. The course deliverables are unfinished, and the
conference extension is not waiting on GPUs alone.

Engineering and numerical analysis are stronger than the current report's wording.
Several claims become too categorical in the abstract, section titles and handoff even
though the underlying result documents or decision records contain the right caveats.
Fix that mismatch before release. More experiments are not needed to resolve these
reporting issues.

## What the work establishes

| Component | Assessment |
| --- | --- |
| Dose response | Strongest result: decoded block-displacement compression grows with command magnitude on fresh families, in both sampled scene types, and extends beyond the doses used to fit the explanatory accounts. This remains a result for one encoder/predictor checkpoint and the tested short-horizon distributions. |
| ROBUST-001 | Useful corroboration: the energy-ratio dose trend survives five readouts. Different readout forms make a peculiarity of one MLP less persuasive, but shared encoder/data and the absence of physical labels for predicted latents remain limitations. |
| SUPPORT-001 | A valuable preregistered null: no account beats every alternative. The growing dose response is more convincing than a claim that one mechanism explains it. The saturated support covariate limits the tested reversion account. |
| TRAIN-002 | The most informative new intervention: most fine-tuning benefit transfers to a dose omitted from fine-tuning. The primary deficit is +0.82 [0.11, 1.63] units; transfer fraction is 0.92 [0.85, 0.99]. The ordinary-window cost is visible, rather than hidden. |
| Agent/block probe | A useful descriptive dissociation: agent displacement is decoded relatively faithfully in cells where block displacement is compressed. Its gating rule was extended after values were seen; it is not independent confirmation of causal localization. |
| MEAS-001 | Correctly stopped after two failed engineering passes. Accurate accepted poses at very low coverage do not validate the instrument. Preserve the failure and the closed 48-family set; do not reopen it merely because budget remains. |
| Planning | A decision cost remains unestablished. A physical prediction discrepancy cannot yet be sold as a demonstrated control failure. |

TRAIN-002's omission intervention also reduces effective examples per update from 64 to
48. This was disclosed in D035, so it is not an undisclosed execution error, but the
small deficit does not isolate the effect of exact-dose coverage from the change in
effective sample count. The archive-only arm changes several aspects of the training
distribution. These interventions motivate a training-support hypothesis; they do not
establish that pretraining support caused the original distortion. Bootstrap intervals
condition on the released checkpoint and the fitted fine-tuning runs, not on a population
of independently pretrained models.

## Verification performed

- `make check && make test`: **343 passed**, no skips, 25 warnings; pytest 18.30 s,
  Python 3.10.21. The warnings are Gymnasium array casts and joblib's physical-core
  detection fallback.
- TRAIN-002: the seal verifier passes. Checked hashes and recorded unit/model/input/
  readout/fingerprint identities for **5,040** saved records. Recomputed the **entire
  summary exactly**, including all readouts, latent contrasts, transfer intervals,
  per-seed contrasts and ordinary-window results.
- SUPPORT-001: the seal verifier passes. Verified all **300 family files and real
  caches**, decoded saved predicted latents, recomputed support covariates and the
  primary readout's account contrasts. All primary estimates and interval endpoints
  match exactly; verdict remains `no_account_established`.
- ROBUST-001: reran the saved-output reproduction gate, **1,888 checks**, maximum
  difference **8.881784197001252e-16**, no failures. This did not refit readouts or
  independently regenerate the entire reanalysis from source latents.
- MEAS-001: re-derived **72 v1 and 24 v2** classifications; both diagnostic manifests
  reproduce exactly. Both versions remain `failed_selection`. Inspected the fixes
  for input/result binding, full-decision prediction gating, independent final-pose
  support and unfinished nuisance challenges; their regression tests pass.
- Reporting: all **43 macros**, the saved `numbers.tex` and the claims ledger regenerate
  identically in memory. This verifies transcription, not the scientific wording or
  every plotted quantity.

Review computation used `/tmp/non_gpu_review.py`, with results at
`/tmp/non_gpu_review_results.json`. No study output or sealed source was overwritten.
This is saved-evidence reproduction, not a new simulator collection or independent
checkpoint-inference replication. PDF typography, slide quality and current literature
positioning were not independently audited in this pass.

## Findings before report release

### F1 — P2: the plotted reversion curve is not the fitted account's projection slope

`src/jepa_action_effects/reporting.py:510–519` plots
`sigmoid(theta0 + theta1 * median(log_support))` as account B on an axis labelled
“Slope, pred vs true.” The actual fitted account is

\[
\hat p_i=w_i t_i+(1-w_i)m_i,
\qquad
b_B=\frac{\sum_i \hat p_i\cdot t_i}{\sum_i\|t_i\|^2}.
\]

Its slope includes the typical-transition contribution and displacement weighting.
Neither is captured by the sigmoid of median support. Even a constant weight does not
generally imply a constant slope: with w=0.5, m=(5,0), and t=(10,0) versus (20,0),
the slopes are 0.75 and 0.625.

Recomputing from the actual saved SUPPORT families gives:

| Cell | Plotted B value | Actual account-B projection slope |
| --- | ---: | ---: |
| Archive 0.45 | 0.611026 | 0.687899 |
| Archive 0.60 | 0.611026 | 0.654153 |
| Archive 1.00 | 0.611026 | 0.642711 |
| Designed 0.60 | 0.611026 | 0.635454 |

Designed 0.15 has zero total true block-displacement energy, so its projection slope is
undefined, although the plot supplies a finite B curve there.

**Fix:** plot the account's actual predicted projection slope on the same family set,
or move its mixture weight to a separately labelled diagnostic. If Route A must remain
manifest-only, save a traceable derived plotting summary without altering the sealed
account test. Add a nonzero-typical-transition fixture. Qualify the plateau explanation:
the actual high-dose fitted curve is nearly flat here, but saturation of w alone does
not establish that algebraically. The primary account comparison is correct and unchanged.

### F2 — P2: headline claims exceed the study's measurement and inferential scope

Examples in `reports/report001/main.tex` include “confirming every claim” (33–34),
“Not a readout artifact” (157), “The error lies in the interaction” (174), and generic
fine-tuning leaving compression “unchanged” (194). `TRAIN-002_RESULT.md:39–40,47–60`
likewise says the in-support manipulation changes nothing and generic fine-tuning does
nothing. These are stronger than the registered verdicts.

Required corrections:

- Distinguish fresh confirmatory tests, exploratory ROBUST/CALIB analyses, descriptive
  transfer, and the post hoc agent-gate extension. Fresh families alone do not make an
  analysis prespecified.
- Replace “not a readout artifact” with robustness across readouts. The shared encoder/
  data and failed independent inversion remain live measurement limitations.
- Describe an agent/block dissociation; do not infer that the action-conditioning
  pathway is correct or that the internal source of error has been localized. An
  approximately correct aggregate agent slope is also weaker than accurate trajectories.
- Replace “generic fine-tuning does nothing” with “no generic fine-tuning gain was
  established under this protocol.” The interval [-1.54, 0.63] is not an equivalence
  test. The same rule applies to “no effect” for in-support accuracy or action structure.
- Say fine-tuning **substantially reduces** decoded compression. Ratios around 0.87–0.93
  do not demonstrate its complete removal.
- Carry TRAIN-001's development measurement failure explicitly into the report:
  its pooled physical comparisons remain descriptive. TRAIN-002's fresh gated result
  is additional evidence; it does not retroactively change TRAIN-001's verdict.
- Include the 48-versus-64 effective-example limitation in the report's description
  of the dose-omission intervention, rather than only in the long result document.

These corrections preserve a substantial result. They remove claims that the data did
not test. A matching number in the claims ledger does not validate the surrounding prose.

### F3 — P2: a high percentile is being presented as the boundary of pretraining support

`reports/report001/main.tex:97–98` infers that both 0.80 and 1.00 are beyond the
training range because p99.9 is 0.60. The figure and several result summaries repeat
this. The map contains **399,758** stride-5, ten-control windows; its maximum mean-command
magnitude is **0.801336968**, with **413** windows above 0.60 and **one** above 0.80.

**Fix:** distinguish three statements: beyond the 0.60 range used to fit the explanatory
accounts; above the archive's 99.9th percentile for this particular window statistic;
and absent from pretraining. The first two are supported. Absence from pretraining
cannot be inferred from a percentile or from this sampled-window map. Describe 0.80 as
an extreme archive tail and the experiment as extending the tested/account-fit dose
range. Use a precise statistic when discussing support; mean command is not contact
speed, effect size, or the complete conditional action-sequence distribution.

### F4 — P2: Route B's published execution order cannot run from scratch

`reports/report001/REPRODUCE.md` places SUPPORT-001 before TRAIN-002 in its dependency
table. But `support001._registry` (`support001.py:303–326`) unconditionally loads both
TRAIN-002 `train_windows/collection.json` and `confirmation/collection.json`. SUPPORT
collection therefore fails if those earlier TRAIN-002 collection stages have not run.
The original run did collect TRAIN-002 first.

**Fix:** publish stage-level ordering: TRAIN-002 audit and both collections precede
SUPPORT collection. Also distinguish rebuilding report assets from reproducing the
experiment. Returning an existing sealed summary is a provenance/reuse check, not
recomputing its scientific estimates. The teammate handoff should include a saved-record
headline recomputation as well as Route A on a clean clone. Do not advertise the
2–3-hour full pipeline estimate as measured end-to-end: the listed times mainly cover
recent studies and exclude substantial historical setup/readout work.

## What remains without a GPU

For the **course**, finish F1–F4, the report's missing contributions and methods detail,
the 15-minute slides, a teammate reproduction and final PDF/claim inspection. The
research content is ample; instructor acceptance is distinct from this scientific
assessment. The task board's old “deliverables not implemented” description is stale:
the report draft and reproduction guide now exist, while slides/contributions remain open.

For the **paper**, PAPER_PLAN explicitly assigns additional CPU work: MECH-001/002,
Cube/TwoRooms interface and replay audits, and planning evaluation. Those runners are
not present. The GPU handoff is also unimplemented: `pretrain001.py`, the launcher,
resume/provenance tests and Colab runbook remain to be built.

Before that handoff, resolve the roadmap's conflicting tail-removal definitions. It
alternates between p90 of the ten-control mean-command norm (~0.328) and p90 of the
maximum five-control block-mean norm within a training clip. Those are different
selection rules and need separately computed thresholds. Freeze one precise rule,
matched-removal/step controls and a timing budget before launching expensive runs.

The scientifically useful next decision is which narrow claim the paper will test.
More GPUs can provide independent training seeds and controlled pretraining ablations;
they do not automatically establish measurement validity, identify a mechanism or show
a planning consequence. Those remain questions, not promised outcomes.

**Disposition:** close the current course experiment cycle, preserve the measured nulls,
and finish its release-quality reporting. Treat the conference roadmap as a separate,
unfinished research programme with CPU and GPU work, rather than a final GPU-only stage.
