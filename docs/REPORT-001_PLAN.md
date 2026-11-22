# REPORT-001 — course report, presentation and reproduction plan

**Prepared:** 2026-09-29, following REVIEW-BRANCH-004 at `de678d9`.
**Status:** plan ready; deliverables not yet implemented. Course acceptance of the
training component remains unresolved. No new experiment is authorized by this plan.

**Update (2026-09-30, after TRAIN-001).** The training component now exists.
TRAIN-001 fine-tuned the LeWM predictor, action encoder and prediction projector
(11.7M parameters) in two paired conditions × three seeds, with sealed fresh
confirmation (TRAIN-001_RESULT.md, D028). It replaces the separate-MLP suggestion
below. The report and talk are rebalanced around its actual results, within the same
five-page and 15-minute limits. REPORT-001 remains open.

## Scope and deliverables

Produce an English report in LaTeX with a compiled PDF, an English presentation
with editable slides, PDF and speaker notes, reproducible scientific figures, and
a concise reproduction guide. Report the completed bounded study, with corrections
through D027. The older FREEZE-001 → FIT-001 → TEST-001 expansion remains paused;
it is not a prerequisite for reporting the completed cohorts.

Working title: **Measuring and Fine-Tuning Action-Dependent Prediction Errors in a
JEPA World Model** (formerly *Measuring Action-Dependent Prediction Errors in a Frozen
JEPA World Model*). The contribution is a controlled, checkpoint-specific measurement
study with trained readouts, physical-effect controls and an inconclusive regret
result. Do not present it as a new general benchmark or established contact-failure
mechanism.

## 1. Resolve course fit while preparing the evidence

The source is the four-page course handout supplied by Omer,
`הנחיות לפרויקט בקורס למידה עמוקה 2026ב.pdf`. These are submission requirements,
not instructions to execute new experiments or submit anything automatically.

| Requirement | Source | Plan and acceptance condition |
| --- | --- | --- |
| Substantial deep-learning model training | p. 1 | Present TRAIN-001: autoregressive fine-tuning of an 11.7M-parameter PyTorch predictor stack, 6 runs × 2,000 updates, paired conditions and seeds. READOUT-002 stays as the supporting readout training. Still confirm with the instructor that this meets the requirement |
| PyTorch implementation; clear baseline, model comparisons, metrics and deep analysis | p. 2 | Describe the implemented PyTorch readout, frozen model, applicable controls and family-level evaluation |
| Report no longer than five pages | p. 3 | Conservatively count references, contributions and links within five pages; no separate cover or submitted appendix |
| Motivation, related work, data, models/hyperparameters, results, analysis, limitations/future work and references | p. 3 | Include all in the page budget below; put implementation detail in the repository |
| Actual teammate contributions and repository link; clear reproduction instructions | pp. 1, 3 | Confirm names and actual work with both teammates; provide a short report pointer to the complete run guide |
| 15-minute presentation; all teammates participate and answer questions | pp. 3–4 | Rehearse a 14-minute main talk with one minute of buffer and optional backup slides |

The handout also requires a 300–500-word proposal (p. 2). Check its submission
status; creating or revising that earlier deliverable is outside REPORT-001.
No deadline, document template, reference-page exemption, or Q&A allocation is
specified in the supplied handout. Confirm these along with full author names and
contributions during authoring. English is the user's chosen language for both
deliverables.

### Existing training: present it accurately

READOUT-002 trained a PyTorch MLP on 50,000 rendered real frames, with 10,000
validation frames. Its architecture is 192 → 256 → 256 → 6 with ReLU hidden layers;
the targets are block position, sine/cosine of block angle and agent position.
It used three initialization seeds, Adam at 0.001, batch size 512, a 300-epoch cap,
and validation-based early stopping with patience 20. The selected run stopped at
170 epochs, with its best checkpoint at epoch 150. The JEPA encoder and predictor
were frozen. This is real neural-network training, but the handout does not say
whether readout-only training qualifies as a **substantial** component.

Draft question for Omer to send to the instructor (do not send automatically):

> Our project evaluates a frozen JEPA world model through controlled simulator
> experiments. We also train a two-hidden-layer PyTorch state readout on 50,000
> rendered frames, validate on 10,000 frames, and compare three initialization
> seeds and linear/nonlinear readouts. Does this satisfy the substantial-training
> requirement? If additional dynamics training is needed, would a trained
> action-conditioned predictor on frozen visual representations, compared with
> the pretrained predictor and an action-blind control, meet the intended scope?

### Completed training component: TRAIN-001 (2026-09-30)

What to report, from `docs/manifests/train001-runtime.json` and TRAIN-001_RESULT.md:

- **Design.** Frozen vs archive-weighted vs magnitude-balanced fine-tuning, on
  identical transitions.
  - Trainable: predictor + action encoder + prediction projector.
  - Frozen: encoder, projector, BatchNorm statistics and readouts.
  - Loss: two-step autoregressive latent MSE with the first prediction fed back.
  - AdamW, LR 1e-5 (pilot-selected), 2,000 updates, seeds 20261111–13.
  - Data: 600 training, 120 validation and 240 fresh confirmation families, plus 180
    ordinary windows. 1.06 of 24 production machine-hours on CPU; the 105 s smoke run
    is disclosed separately.
- **Gates.**
  - All implementation, optimization, data-integrity and budget gates passed.
  - The development measurement gate failed in the archive cells (floor 0.278 / 0.304 >
    0.25). The pooled physical hypothesis is therefore **not assessable**, even though
    every confirmation gate passed (D028).
- **Headline** (after TRAIN-REVIEW-002, D029): a small secondary latent-MSE advantage
  for balanced weighting; ordinary-case non-inferiority not established; physical
  comparisons descriptive only.
- **Numbers.** Physical results are reported, not claimed. Take every value from
  `train001-runtime.json` or `train001-followup.json` (`exact_numbers`); never
  retype them.
  - Balanced − archive-weighted endpoint error: −0.09 [−0.53, 0.38].
  - Both fine-tuned conditions minus frozen: about −11.6 [−14.3, −9.1]. Error falls from
    26.1 to about 14.4 units, and the pred/real slope ratio at 0.60 rises from 0.63 to
    about 0.93.
  - Secondary latent: balanced − archive **−0.00684 [−0.01276, −0.00110]**; both minus
    frozen about −0.25.
  - Ordinary windows, balanced − frozen: +1.05 [0.11, 1.99]. Non-inferiority not
    established. The interval lies above zero for two of three readouts
    (`mlp-20261004`: [−0.15, 1.82]).
- **Controls.**
  - The calibration scalar (1.03) recovers only −0.31.
  - Fine-tuned predictions are more action-dependent: permuted − correct about +17 vs
    +8.8 for frozen.
  - Readout sensitivity is reported contrast by contrast, not by verdict string (the
    verdict is readout-independent once eligibility fails). All contrasts agree in sign
    across the three readouts except ordinary balanced − frozen.
- **Required qualifications.**
  - This is in-distribution adaptation to the designed constant-push family process.
  - No pooled balanced-over-archive physical benefit was established. The descriptive
    contrast is near zero, but no equivalence test was specified, so "no difference" is
    not claimed.
  - The cause of the fine-tuning gain is a hypothesis: there was no dose-omission or
    archive-only training control.
  - A failed measurement gate is not evidence of no effect.
  - Readout validity on fine-tuned latents is unproven.
  - One checkpoint and one horizon.
  - No planning claim.

Updated draft question for Omer to send to the instructor (do not send automatically):

> Beyond the frozen-model measurements, we fine-tuned the JEPA predictor stack
> (11.7M parameters) in PyTorch under two loss-weighting schemes with three paired
> seeds, and evaluated it on freshly collected, sealed test families against the frozen
> model, calibration and persistence baselines. Does this satisfy the
> substantial-training requirement?

### Superseded: extension proposal written before TRAIN-001

Propose a **small action-conditioned latent dynamics predictor**, trained from
scratch while the visual encoder remains frozen. Unlike a state readout, this
learns how candidate actions change future representations. It fits the study's
question and avoids making full world-model training the default next step.

- Input: the verified three-frame observed history, corresponding available past
  controls, and the proposed future controls. No future observations or future
  proprioception enter the predictor.
- Output: the two future projected latent representations at the existing native
  prediction stride. Decode with the same frozen readout for physical evaluation.
- Initial architecture proposal: two 256-unit ReLU hidden layers, latent MSE loss,
  three training seeds. Final optimization settings, data volume and compute cap
  belong in a separate frozen protocol after a development-only timing pilot.
- Comparisons: frozen LeWM; the trained action-conditioned model; an otherwise
  matched action-blind model with its action inputs masked; physical persistence;
  and the real-future readout control. Use the same training transitions and update
  budget for the two newly trained models.
- Split by source episode or initial-state family before making windows. Keep
  related branches together. Train and select checkpoints on development data,
  tune only on validation, and reserve fresh families for a new confirmation.
  Existing opened cohorts may be used only for explicitly exploratory evaluation.
- Evaluate endpoint/displacement error and action effects at the same primary
  horizon. Separate variation across training seeds from uncertainty across
  families. A failed improvement is still a valid training result.
- Before a new sealed run, integrate strict provenance checks into that runner,
  freeze gates and exclusions, and document the protocol in DECISIONS. Do not
  silently revise the completed study's gates or relabel its results.

If the instructor specifically requires adapting the pretrained world model,
consider predictor-only fine-tuning with the encoder frozen instead. Select one
extension, not both. The original research plan's privileged state-dynamics MLP
was proposed but is not a completed baseline; do not describe it as implemented.

The proposal above is kept for history. TRAIN-001 implemented the "predictor
fine-tuning with the encoder frozen" alternative instead of a from-scratch MLP.

**Decision:** if existing training is accepted, complete REPORT-001 from current
evidence. If additional training is required, scope it as a separate task and
integrate its actual results before final submission. Report drafting can proceed
in either case; neither option guarantees instructor acceptance in advance.

## 2. Freeze the reporting evidence and claim wording

Create `docs/REPORT-001_CLAIMS.md`, linking each numerical claim to a tracked
manifest, exact JSON field, cohort, family count, units, interval definition,
evidence status and relevant correction. Use result documents to explain the
protocol; use manifests for numbers. Record the source revision and dirty state.

| Report message | Evidence | Required qualification |
| --- | --- | --- |
| The nonlinear readout enables the bounded physical measurement | READOUT-002; matched real-frame controls in PHYS/SHIFT | Real-frame accuracy does not establish readout validity on every predicted latent |
| Challenge predictions have a reduced projection slope | PHYS-002, 150 reserved families including 100 challenge families; slope 0.561 vs real-frame 1.006 | This is not a claim that displacement length is 56% of truth; H1 was not established |
| Projection-slope ratio declines with command magnitude in both scene types | SHIFT-002, 240 fresh reserved families | Command magnitude and realized displacement co-vary; this does not establish a universal shortening effect or a unique mechanism |
| Length and direction both help characterize PHYS errors | Exploratory follow-up: pooled length ratio 0.629 [0.583, 0.673], mean direction error 19.0° vs 4.9° real-frame | Different weighting prevents an exact multiplicative decomposition of the projection slope |
| Decision harm remains unresolved | H3-PLAN-002: 150 reserved families, 148 with valid goals; pooled JEPA − real regret +2.07 [−1.32, 5.59] | H3 not established; no equivalence-to-zero or negligible-harm claim; privileged candidate-specific goals are not a deployable planner |
| Key computations reproduce | REVIEW-FIX-001: 150 PHYS and 30 SHIFT families matched saved results exactly | A computational rerun using shared implementation, not independent reimplementation or proof of portability |
| Both fine-tuned conditions reduced decoded errors on the sampled constant-push distribution | TRAIN-001 confirmation, 240 fresh families: endpoint error 26.1 → about 14.4; slope ratio at 0.60 0.63 → about 0.93 | **Descriptive only.** The pooled physical hypothesis is not assessable (development archive floor failure). Cause is a hypothesis (no dose-omission control). In-distribution adaptation only |
| No pooled balanced-over-archive physical benefit was established | TRAIN-001: −0.09 [−0.53, 0.38]; per-seed SD 0.11 | Not an equivalence result. The challenge-only subgroup −0.33 [−0.62, −0.04] is exploratory and cannot replace the pooled result |
| Balanced weighting slightly improves latent prediction (secondary headline) | TRAIN-001 secondary: −0.00684 [−0.01276, −0.00110] | Representation prediction, not physical prediction; about 4% relative |
| Ordinary-case non-inferiority of balanced fine-tuning is not established | TRAIN-001 ordinary windows: balanced − frozen +1.05 [0.11, 1.99] | Regression beyond 1.28 not shown. Above zero with two of three readouts (readout-dependent) |

Keep absolute endpoint error distinct from displacement error: on the PHYS
challenge cohort the exploratory endpoint values are 17.895 predicted / 3.613
real-frame, while the historically misnamed `endpoint_err` displacement values
are 17.857 / 3.935. Preserve the corrected ratio explanation: declining challenge
readout slope opposes the observed ratio decline; rising archive readout slope
contributes to it. Do not revive the superseded explanation below the correction
banner in REVIEW_FOLLOWUP_RESULT.

Also disclose the H3 valid-goal filter, uncertainty crossing zero, checkpoint and
horizon specificity, unavailable historical fingerprints/input hashes, original
failed measurement gates, and the fresh cohorts used after protocol redesigns.
Do not elevate H2 or an exploratory follow-up to a newly confirmed hypothesis.

Add a concise current summary to STUDY_SYNTHESIS, preserving historical evidence
and marking superseded interpretations. Verify the selected primary papers before
constructing bibliography entries; begin with LIT-001_REVIEW and NOVELTY-002_MEMO,
particularly LeWM and the closest AD-WM comparison. Do not invent citation details
or claim implementation of an unavailable published metric.

**Done when:** every headline and displayed number has a traceable source and the
current narrative agrees with D026/D027 and REVIEW-BRANCH-004.

## 3. Generate the figures and tables reproducibly

Implement a small reporting module in `src/jepa_action_effects/reporting.py`.
Its default path reads tracked manifests and writes to ignored
`outputs/report001/`; it must not run checkpoint inference or overwrite evidence.
Export vector PDF/SVG and high-resolution PNG figures, compact tables, and LaTeX
number macros from the same extracted values. Record input hashes with the exports
and fail clearly on missing fields rather than inserting placeholder results.

Primary sources: `readout002-runtime.json`, `phys002-runtime.json`,
`shift002-runtime.json`, `planning002-runtime.json`,
`review-followup-runtime.json` and `train001-runtime.json` under `docs/manifests/`.

Plan three main figures:

1. **Physical effects:** PHYS projection slope and correctly named physical error
   with real-future readout controls; distinguish primary and exploratory panels.
2. **Magnitude response:** separate challenge/archive panels showing SHIFT dose
   response and matching real-frame controls. Preserve the source intervals and
   identify constant versus varying controls.
3. **Decision consequence:** H3 regret contrasts and family-bootstrap intervals,
   with a visible zero line and the prespecified interpretation.
4. **Training intervention (TRAIN-001).** Start from the tracked
   `docs/figures/train001/magnitude_response.png` (frozen vs both fine-tuned conditions,
   by magnitude and scene) and `paired_errors.png` (primary, secondary latent and ordinary
   contrasts, per seed).
   - Mark the development measurement failure on the figure or in its caption.
   - Consider merging figure 2 with the TRAIN-001 magnitude response to save space.
   - Put the learning curves (`learning_curves.png`) in backup or supplement.
   - Label the pooled physical result "not assessable".

Use a compact table for model/readout configuration and core counts. Move extended
readout learning curves, per-dose decomposition and historical failed-gate detail
to repository supplements and backup slides if they crowd the five-page report.
Any trajectory illustration requires the ignored run artifacts; use a seed/order
or simulator-outcome selection rule, never choose examples by model errors.

Test source-field mapping, metric labels, cohort counts and consistent exported
values. Do not average CI endpoints or derive new contrasts from marginal CIs.
Any genuinely new analysis needs its own stated procedure and evidence label.

## 4. Write the five-page report

Keep LaTeX and bibliography sources in `reports/report001/`. Start with a readable
A4, two-column, 10-point academic layout unless a required template is supplied.
The following is a content budget, not permission to shrink text to force a fit:

| Content | Approximate pages |
| --- | ---: |
| Title, short abstract, motivation/questions and focused related work | 0.60 |
| Data/splits, exact simulator/checkpoint pair, trained readout, TRAIN-001 training setup, controls and metrics | 1.25 |
| Results, figures and interpretation (measurement study ≈ 1.2; TRAIN-001 ≈ 0.9) | 2.10 |
| Limitations, insights and future work | 0.55 |
| References, actual contributions, repository/reproduction links | 0.50 |

Describe the current scientific account rather than recounting every development
iteration. Include critical gate changes and failed controls where they affect
interpretation. Report unperformed baselines as limitations, never as results.
If a training extension is commissioned, rebalance this space around its actual
findings instead of exceeding the page cap.

Use a pinned report-only TeX toolchain; no TeX engine was found in the inspected
local environment. Verify and install a suitable compiler during implementation
and record its version without changing the scientific Python lockfile merely to
compile documents. A source-only deliverable does not satisfy the PDF requirement.

## 5. Build and rehearse the presentation

Target 12 main slides, plus about three backup slides. After TRAIN-001, merge the
original slides 8–9 (SHIFT and the exploratory length/direction analysis) and shorten
slide 11. That frees two slides: "TRAIN-001 design and gates" and "TRAIN-001
results". Put the learning curves and the pilot/LR sweep in backup. Suggested main
sequence (original numbering):

1. Question and bounded contribution.
2. Why action-dependent physical prediction matters.
3. Closest work and the specific measurement gap.
4. Simulator, history/action interface, replay and family splits.
5. Frozen JEPA and what was actually trained.
6. Readout controls, physical metrics and uncertainty.
7. PHYS result and the unestablished contact hypothesis.
8. SHIFT magnitude response.
9. Exploratory length/direction analysis.
10. H3 result, interval and privileged-goal limitation.
11. Main limitations and reproduction evidence.
12. Conclusions and next research question.

Backups: training hyperparameters/readout learning curve; TRAIN-001 learning curves,
pilot and LR sweep; gates, exclusions and splits; reproduction/provenance details. If a training extension is completed,
replace lower-priority material so its method and results fit the same duration.

Use the presentation skill and bundled slide tooling during creation, retaining
editable source and PPTX plus PDF. Use the same numerical exports as the report,
readable labels and speaker notes that explain the limitations. Confirm both
speakers' allocation from their actual work; rehearse a 14-minute talk together.
The handout does not specify whether Q&A is inside the 15 minutes, so verify that
before the final rehearsal.

## 6. Reproduction, final checks and handoff

Write `reports/report001/REPRODUCE.md` with two distinct routes:

- **Regenerate report assets:** use the tracked manifests, locked Python environment
  and pinned report toolchain. A new checkout must generate the same numerical
  tables without downloading the checkpoint or simulator archive.
- **Reproduce TRAIN-001:** follow `docs/TRAIN-001_RUNBOOK.md` (nine staged
  commands with a shared budget ledger). Checkpoints and caches stay ignored.
- **Reproduce experiment evidence:** list the pinned model/archive prerequisites,
  hashes, local artifact paths and exact existing rerun commands. Explain the
  historical provenance gaps and avoid describing legacy validation as strict
  certification. Keep data and model weights outside Git and out of source bundles.

Have the other teammate execute a fixed reproduction subset and record who ran
it, revision, commands, dependencies, differences and limitations. The existing
exact rerun is evidence already available; it does not establish that this teammate
handoff or a clean-machine rerun has occurred. Cache-dependent paths must say so.

Add documented build commands for figure generation and PDF compilation. Keep
report sources and small evidence mappings tracked, with generated deliverables
in `outputs/report001/`. Include a source bundle or precise clean-checkout build
instructions that resolve generated-asset paths. Keep frozen experiment sources
and sealed artifacts unchanged.

Final acceptance:

- Instructor clarification resolves the substantial-training requirement; names
  and actual contributions are confirmed, and any required extension is complete.
- All required report sections fit within five total pages, with working citations
  and links, no draft placeholders and no unsupported headline claims.
- Every PDF page and slide is rendered and visually inspected for clipped content,
  legibility, units, legends and agreement with the source tables.
- Both teammates participate in a timed rehearsal within the presentation limit.
- Figure/table regeneration and the documented reproduction subset succeed, with
  limitations recorded honestly.
- After Python changes, run `make check` and relevant reporting tests; run broader
  tests if changes touch scientific logic. The prior 151-test result is a baseline,
  not validation of future reporting code.
- Update TASKS and append commands/results to LOGBOOK. Mark REPORT-001 DONE only
  when the complete package and handoff exist, not when this plan is written.

**Immediate next work:** prepare the instructor clarification (updated question
above) and the claim ledger, including the TRAIN-001 rows. Then generate the core
figures (three measurement figures plus the TRAIN-001 figure). TRAIN-001 is complete;
any further training (readout repair or generalization tests) would need a new
protocol and fresh families. Report drafting can proceed while course
scope is clarified. New training remains a separate, explicitly scoped task.
