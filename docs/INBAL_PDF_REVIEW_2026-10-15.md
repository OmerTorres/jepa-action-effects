# Review of Inbal's PDF — 2026-10-15

**Recommendation: selectively incorporate the useful edits into the tracked report;
do not replace it wholesale with this PDF.** The changes improve some explanations,
but introduce an ambiguous mathematical claim and unsupported causal explanations.
Equation presentation still needs work. No new experiment is needed for these fixes.

## Scope and evidence

- Supplied file: `~/Downloads/final_project_deep_learning_JEPA.pdf`.
  Five A4 pages; SHA-256
  `df1bb44dc76566dc1188ed659a944c7ef9b88438866913dd5d9e06fd7388bdfa`.
- Compared with `reports/report001/main.tex` at
  `0ee9d10e7b5ce3f827df5363d78e1e44df984683` and the existing local report PDF.
  The supplied PDF's source is unavailable; attribution of its differences to Inbal
  follows the user's description, not a commit history.
- Read all pages and visually inspected all five 1600-pixel renders. Checked relevant
  readout, gain, training-loss and planning code, and the correction banner in
  `docs/H3-PLAN-002_RESULT.md`.
- Main reported results are retained: 26.84 → 16.21 endpoint error, omitted-minus-all
  0.82 [0.11, 1.63], transfer 0.92 [0.85, 0.99], ordinary-data cost
  1.08 [0.28, 1.91], and planning difference 2.07 [−1.32, 5.59]. A numeric-token
  comparison with the current PDF found only small additions/deletions associated
  with the edited prose/captions. This is not a full verification of plotted curves
  or a source rebuild.

## Edits worth keeping

- Page 4 explicitly states omitted-arm weights `(1/3, 1/3, 1/3, 0)` and explains
  that the loss is not divided by four again. This agrees with
  `train002_metrics.DOSE_OMITTED_WEIGHTS` and `train001_model.weighted_loss`.
- The explanation of 48 nonzero-weight branches out of 64 is more precise than
  calling this simply a smaller batch. Keep the coverage/effective-example-count
  confound and the existing disclosure that 0.60 participates in development.
- Page 3 tries to explain why pooling inner products is preferable to averaging
  unstable per-branch ratios. Keep that motivation after correcting F1 below.
- Explicit Linear/ReLU labels communicate the architecture better, but the actual
  drawing needs to be enlarged or replaced by readable equations.
- Page 4 identifies squared table units for the explanation-comparison errors and
  briefly explains the failed readout-free method. These are useful clarifications.
- The real-outcome selection explanation in Figure 1 remains appropriate. Mixing
  ordinary clips into targeted training is a reasonable future experiment, provided
  it is presented as a proposal rather than a demonstrated remedy.

## Findings to resolve before adoption

### F1 — P2: distinguish the pooled gain from the ratio of gains (page 3)

After defining `R = b / b_real`, the new sentence says “this aggregate ratio”
“equals a least-squares slope through the origin.” The slope interpretation belongs
to `b`, not generally to `R`:

\[
b=\frac{\sum_i p_i^\top t_i}{\sum_i\|t_i\|^2},\qquad
b_{\mathrm{real}}=\frac{\sum_i r_i^\top t_i}{\sum_i\|t_i\|^2},\qquad
R=\frac{b}{b_{\mathrm{real}}}
=\frac{\sum_i p_i^\top t_i}{\sum_i r_i^\top t_i}.
\]

`b` minimises the pooled squared residual between `p_i` and `b t_i`; `R` is a
normalised ratio of two such gains. For example, `p=2t` and `r=0.5t` gives `b=2`
but `R=4`. Pooling avoids separate divisions for tiny individual movements; it does
not guarantee that the aggregate denominator is nonzero. The code explicitly checks
the denominator (`shift002_metrics.py:66–78`).

Suggested wording: “We fit a pooled gain through the origin instead of averaging
per-branch motion ratios. We then divide the predicted gain by the real-frame gain.”

### F2 — P2: keep the ordinary-data tradeoff separate from its cause (page 5)

The new statement that the ordinary-data degradation “likely reflects
state-distribution shift” because half the training families are designed scenes is
not identified by this experiment. Scene mix, action mix and task-specific adaptation
change together. There is no comparison isolating the designed-scene fraction.

Use: “The ordinary-data degradation may reflect specialisation to the targeted
push distribution; we did not isolate its cause. Mixing ordinary clips into
fine-tuning is a candidate mitigation.” The measured tradeoff remains unchanged.

### F3 — P2: qualify the new explanation for the planning null (page 5)

The warning that E7 does not isolate block error is correct. The added explanation
that the model could choose from agent motion while ignoring block-push errors is
untested, and omits the crucial candidate-specific goal construction.

Each candidate has a different goal image preserving that candidate's **real final
agent position and block angle**, while moving the block to the goal. Matching cancels
those pose differences in the real-frame comparison; errors in predicted agent pose
and angle can still affect predicted-latent rankings. This is not a fixed goal agent
location that simply rewards a particular agent motion. See
`planning002.py:140–154` and the correction banner in `H3-PLAN-002_RESULT.md:21–27`;
the older implementation docstring is explicitly superseded by that correction.

Retain the measured null and describe the privileged construction accurately. State
that total latent prediction error is tested and the result does not establish that
block-motion errors are harmless. Agent masking or a block-only objective can remain
proposed alternatives; neither has been validated here. Near-one aggregate agent
gain alone does not establish accurate candidate-specific agent endpoints in E7.

### F4 — interpretation: the new component-level mechanism is untested (page 5)

The claim that fine-tuning repaired contact physics specifically in predictor
transformer blocks while the action encoder already transmitted magnitude is clearly
labelled a hypothesis and acknowledges absent ablations. It is therefore not a false
report of a measurement, but it adds specificity unsupported by the evidence and is
placed under “What we can conclude.” Predictor, action encoder and prediction projector
were all updated. Accurate aggregate agent gain does not localise the repair.

Prefer the current report's statement that the experiment does not locate the
mechanism. If retaining this hypothesis, move it to future work and describe a
component-freezing ablation that would test it.

### F5 — author decision: equal contributions are not established by the repository

Page 5 adds equal contributions to every activity: design, implementation,
experiments, analysis and writing. That could be true, but the evidence reviewed
does not establish it; the task board explicitly says proposed owners are not a
record of acceptance. Do not infer actual contributions from that board or commit
counts. The recorded professor instruction permits contributions to be discussed
verbally, so this paragraph is unnecessary unless the authors want it and confirm
that the wording accurately describes their work.

## Mathematical presentation and layout

The feedback about equation presentation remains applicable to both versions.
Inbal's PDF contains no numbered mathematical equations. Its readout diagram is
particularly small on page 3; rollout equations on page 4 are unnumbered, and the
loss and gain definitions are crowded into running prose. The five-page layout is
otherwise coherent, with no obvious clipping or overlapping body text in the renders.

For a subsequent editing pass, use numbered display equations with labels and text
references for the central definitions, grouping related expressions:

1. Readout mapping, with input standardisation, two 256-unit ReLU layers, a linear
   six-output layer, and output de-standardisation. Either retain a **readable,
   captioned figure** for the architecture or give its mapping as an equation;
   a diagram need not be labelled as an equation.
2. Block-only predicted, real-frame and true displacements.
3. Pooled gain, real-frame gain and their ratio; explicitly define the summed sample
   set and its moved/all-branch rules. Family-level bootstrap remains separate from
   the pooled inner-product estimator.
4. Two-step autoregressive rollout, grouped under one number or subequations.
5. Latent loss with magnitude weights and family/scene averaging. Define normalised
   weights explicitly, preserving the ordinary-clip arm's separate sampling rule.
6. Endpoint error and finite-candidate regret if expressed mathematically; keep
   their averaging conventions explicit.

Numbering alone is not sufficient. Both reports define `g` as six-output but then
write `p=g(z_hat)-g(z_0)` for a two-dimensional block displacement. Define
`g_block` as the block-coordinate projection. This is an inherited notation gap,
not a newly introduced implementation bug. The implementation's actual output
order is `(block_x, block_y, sin(angle), cos(angle), agent_x, agent_y)`
(`readout_scale.py:50–63`), and it standardises both inputs and targets
(`readout_scale.py:170–182, 227–236`). An explicit vector equation must match that
order or declare a reordering.

Retain the current report's clearer Table 1 definition of `R` as predicted gain
divided by **real-image readout** gain. Inbal's shortened “predicted/real motion gain”
can be mistaken for predicted/true motion. Also retain the four E2 count groups and
the Figure 4 validation breakdown (60 designed + 60 demo), which were removed for
brevity. They explain details the authors previously asked to clarify.

## Validation and handoff

Commands run:

```sh
pdfinfo ~/Downloads/final_project_deep_learning_JEPA.pdf
pdftotext -layout ~/Downloads/final_project_deep_learning_JEPA.pdf /private/tmp/inbal-review-20261015/inbal.txt
pdftoppm -scale-to 1600 -png ~/Downloads/final_project_deep_learning_JEPA.pdf /private/tmp/inbal-review-20261015/page
pdftotext -layout outputs/report001/pdf/main.pdf /private/tmp/inbal-review-20261015/current.txt
shasum -a 256 ~/Downloads/final_project_deep_learning_JEPA.pdf
UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync pytest tests/test_train001_model.py tests/test_train002.py tests/test_shift002.py tests/test_planning002.py
git diff --check
```

The focused suite passed: **29 tests, two Gymnasium casting warnings, 7.95 seconds**.
No Python/environment change, training, model inference, bootstrap rerun or source
compilation was performed. Tests validate existing implementation, not unavailable
PDF source. Report/PDF, scientific artifacts and protocol are unchanged; only this
review and project task/log memory are updated. No commit, push or external message.

Next: authors decide whether to incorporate the identified edits into the tracked
LaTeX source. Then implement equation/notation improvements and inspect a rebuilt
PDF. The present review does not perform that editing pass.

## Implementation verification — REPORT-MATH-001, 2026-10-15

Omer subsequently authorised incorporation, corrections, validation, commit, push and
a PR, and selected a layer-block architecture diagram. The following records the
disposition of the supplied PDF's substantive additions; the original review above
is retained as historical evidence.

| Addition or change | Evidence checked | Final disposition |
| --- | --- | --- |
| Explicit omitted-arm weights and no second division by four | `train002_metrics.DOSE_OMITTED_WEIGHTS`; `train001_model.branch_losses` and `weighted_loss` | Incorporated, with a numbered branch/family/scene loss equation |
| 48 nonzero-weight branches per 64-branch update | 16-family × four-magnitude batches; the fourth weight is zero | Incorporated; described as a coverage/nonzero-example-count confound, without asserting a smaller physical batch |
| Motivation for pooled motion statistics | `train001_metrics.projection_slope`; `shift002_metrics.ratio` | Incorporated after distinguishing the pooled slope from the ratio of slopes; positive-denominator qualification retained |
| Linear/ReLU architecture labels | `readout_scale.fit_mlp`, `mlp_predictor`, `targets`, selected probe artifact | Replaced with a readable vector layer-block figure and numbered mapping; included input/output standardisation and exact output order |
| Squared units for explanation-comparison errors | `support001_metrics` squared endpoint residuals | Incorporated |
| Brief explanation of failed readout-free method | MEAS-001 plan/engineering evidence | Incorporated as render-and-encode pose inversion; did not imply its development/validation never used labels |
| Median-outcome example-selection explanation | D039 and tracked example metadata | Incorporated; retained the unusually large example-error caveat |
| More compact glossary and captions | Compared definitions with existing report and implementation | Incorporated selectively; retained Table 1's real-image reference, E2 count groups and the 60/60 validation split |
| Ordinary-data degradation caused by designed-scene distribution shift | No isolated scene-mixture intervention in the training design | Causal explanation not adopted; cause remains unidentified |
| Mixing ordinary clips into targeted fine-tuning | A prospective mitigation, not an experiment performed here | Incorporated as a candidate mitigation only |
| Planning null explained by choosing from agent motion alone | Candidate-specific goal renderer and D026 correction in H3-PLAN-002_RESULT | Not adopted; replaced by the actual matched-goal construction and total-error limitation |
| Repair localised to predictor transformer blocks | Predictor, action encoder and prediction projector all updated; no component ablation | Not adopted; retained explicit inability to locate the mechanism |
| Equal contribution to every activity | Not verifiable from PDF/task board/commit history; professor permits verbal contributions | Not adopted |
| Explicit reporting command | Existing Route A includes separate asset-generation and compilation commands | Retained reproduction-file pointer; final text accurately covers the full PDF build |

Inherited notation was also corrected: `g_B` selects the first two of the readout's
six outputs, and the measurement figure now explicitly subtracts the start reading
before calling the result displacement. Seven numbered equation groups cover readout,
displacements, gains/energy ratio, endpoint error, rollout, loss and regret. Shortened
repetition preserves the five-page limit with the original body font and margins.

### Validation evidence

- `UV_CACHE_DIR=/private/tmp/jepa-uv-cache make check`: passed (Ruff, formatting,
  doctor). `UV_CACHE_DIR=/private/tmp/jepa-uv-cache make test`: **362 passed**, 25
  existing Gymnasium/joblib warnings, 17.04 seconds. The existing tracked-manifest
  figure-build test now includes the readout architecture.
- Generated all five figures and 111 numbers with
  `UV_CACHE_DIR=/private/tmp/jepa-uv-cache MPLCONFIGDIR="$PWD/.cache/matplotlib" uv run --locked --no-sync python -m jepa_action_effects.reporting`.
  `numbers.tex`, `inputs.json` and `docs/REPORT-001_CLAIMS.md` match the pre-edit
  snapshots byte-for-byte. No scientific manifest changed.
- Directly evaluated the readout equation with PyTorch float64 linear/ReLU operations
  using the hash-verified selected probe; compared seven standardised-input offsets
  against the saved NumPy implementation. Maximum block-coordinate difference:
  **5.684341886080802e-14** units. Checked matrix shapes 256×192, 256×256 and 6×256,
  plus exact block/orientation/agent target ordering.
- Checked the pooled-gain counterexample: predicted `2t`, real-frame `0.5t` gives
  `b=2`, `b_real=0.5`, `R=4`. Evaluated the written two-step/192-coordinate loss and
  magnitude/family/scene averaging on four synthetic families: equation
  `1.9675759276421378`, implementation `1.9675759276421383`. These are algebra/code
  checks, not new model experiments. Results: ignored
  `outputs/report001/equation-review/equation-audit.json`.
- Compiled with
  `TECTONIC_CACHE_DIR="$PWD/.cache/tectonic" tectonic -X compile reports/report001/main.tex --outdir outputs/report001/pdf`.
  Final PDF: five A4 pages; no overfull boxes or undefined references. Non-blocking
  font-request and underfull-box warnings remain. Inspected all five 1800-pixel
  renders and the architecture asset; no clipping or overlap found.
- Copied only `git ls-files` sources, including the current edits, into
  `/private/tmp/jepa-report-equations-clean-20261015` (no experimental `outputs/`,
  checkpoints or archive). Built there using the locked interpreter and exported
  `PYTHONPATH="$PWD/src"`, then compiled with the existing TeX cache and rendered
  with `pdftoppm -scale-to 1800 -png`. All five figure PNGs, numerical/input files,
  claims ledger and five final PDF page renders match the workspace build
  byte-for-byte. This validates source reproducibility with the current runtime,
  not a fresh-machine dependency installation or experiment replication.
- `git diff --check`: passed. Only report/presentation code, its existing build test,
  reproduction instructions and review/task/log notes changed. No scientific
  protocol, model weights, simulator, dependencies or sealed evidence changed.
