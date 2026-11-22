# Professor-style course assessment — 2026-10-13

## Assessment and scope

**Provisional grade: 90/100 for the current written submission and supporting work.**
Updated after Omer clarified on 2026-10-13 that the professor explicitly instructed
the team to discuss individual contributions verbally at the presentation. This
supersedes the handout's written-contributions requirement for this submission;
the original 88/100 assessment incorrectly deducted for that omission. Two points
are restored; the scientific assessment is unchanged.
This is a reviewer estimate, not the instructor's grade. The handout names five
criteria but assigns no weights; the weights below are my proposed rubric. The
presentation and individual mastery have not been assessed. They are significant
parts of the actual project grade. The project constitutes 80% of the course grade;
90 here is not a prediction of the final course grade.

Reviewed the three-page PDF at `outputs/report001/pdf/main.pdf`, its LaTeX source,
the supplied four-page Hebrew course handout, current study records, reporting
code, selected training/evaluation code and previous audits. Fetched `origin/main`
at merge commit **2e3af92**: `git diff HEAD origin/main` is empty. The checked-out
branch remains `claude/youthful-clarke-akm28u` at **b6802a9**; the reviewed trees
are identical. No report, model, protocol or experimental evidence was changed.

The handout is evidence about the grading requirements, not authorization to
execute new experiments, contact the professor, or submit the work.

| Criterion from handout | Proposed mark | Reason |
| --- | ---: | --- |
| Technical quality | 33/35 | Substantial PyTorch fine-tuning, controlled simulator branching, measurement controls, family-level inference and strong evidence checks. Remaining limitations include shared readout infrastructure and the omission-arm sample-count confound. |
| Originality | 12/15 | A useful checkpoint-specific dose-response study and training-data intervention. The general intervention-fidelity problem already has close precedents; no new architecture or identified mechanism. |
| Significance | 12/15 | The problem matters for action-conditioned world models; the measured distortion and adaptation are substantial. One checkpoint/environment/horizon and unestablished planning cost limit demonstrated impact. |
| Insight and analysis | 19/20 | Strong distinction between measurement failure, descriptive findings and confirmation; informative nulls and ordinary-distribution tradeoffs. Some headline wording still outruns the evidence. |
| Written communication and compliance | 14/15 | Professional figures and logical structure, with room to improve compressed exposition and metric/training definitions. Contributions will be discussed verbally as instructed by the professor. Oral presentation ungraded. |
| **Total** | **90/100** | **Strong course project, with opportunities to improve exposition.** |

With the report fixes below and a convincing oral defense, low-to-mid 90s would
be a plausible assessment. This is conditional, not a promised grade or an
official deduction schedule.

## Course requirements

- **Substantial deep-learning training (handout p. 1): satisfied in my assessment.**
  This is more than frozen-model inference or readout training: the project
  fine-tuned 11,740,334 predictor/action-encoder/prediction-projector parameters
  in PyTorch, four training arms with three seeds, 2,000 updates per run. It also
  trained three MLP readouts. The handout does not require GPU use, training from
  scratch, a minimum compute budget, or a newly invented architecture. Instructor
  acceptance remains their decision, but the earlier readout-only concern is no
  longer a good description of the completed project.
- **Systematic experiments (p. 2): satisfied.** Frozen and real-frame controls,
  multiple fine-tuning conditions, alternative readouts and random/oracle planning
  references provide substantive comparisons. Metrics and uncertainty receive
  considerable attention.
- **At most five pages and required report sections (p. 3): mostly satisfied.**
  The PDF is three pages including references. Motivation, literature, data,
  models, results, analysis and limitations are present. Methods need more detail.
- **Individual contributions: written omission accepted under the professor's
  explicit guidance.** Omer clarified on 2026-10-13 that the professor instructed
  the team to discuss contributions verbally when presenting the report. No written
  contribution paragraph is required for this submission. Actual contributions
  and individual mastery remain matters for the presentation, not assessed here.
- **Repository and reproduction (p. 3): substantially present, incompletely proven
  on another machine.** The repository link and run guide exist. Asset rebuilding
  is correctly distinguished from experiment reproduction in the run guide.
  Teammate reproduction has not been recorded. A saved-record recomputation is
  described as optional and points to an audit narrative whose temporary script
  is not committed, so the guide is less turnkey than the report suggests.
- **Presentation (pp. 3–4): pending, not counted as a missed current deadline.**
  The handout specifies 15 minutes, participation by every member and command of
  the material. The latest logbook says the talk is later. Proposal submission
  status is not established by this review.

## What deserves high marks

The strongest scientific contribution is a controlled finding that decoded block
motion is increasingly compressed as command magnitude rises. Fresh cohorts,
real-frame measurement checks and the extension beyond the account-fitting range
make this substantially more persuasive than an isolated probe result.

TRAIN-002 is the strongest adaptation evidence. Its full summary reproduces
exactly from saved records. At command 0.60, mean terminal block error is:

| Model | Error in workspace units |
| --- | ---: |
| Frozen | 26.84 |
| Archive-only fine-tuning | 27.31 |
| Archive-weighted fine-tuning on designed pushes | 16.81 |
| Balanced fine-tuning on designed pushes | 16.21 |
| Fine-tuning with 0.60 omitted | 17.03 |

The balanced gain is 10.6 [7.9, 13.7] units; the omitted-dose deficit is
0.82 [0.11, 1.63]; the descriptive transfer fraction is 0.92 [0.85, 0.99]. The
replicated ordinary-window cost, +1.08 [0.28, 1.91], usefully prevents a universal
improvement claim. These results should be easier to find in the report.

The planning and mechanism nulls should receive credit for honest analysis.
They restrict the conclusion; they do not invalidate a good course project.
Likewise, stopping failed readout-free inversion preserves scientific integrity.
Five-readout agreement is useful corroboration while leaving the shared encoder
and extrapolation problem unresolved.

The literature positioning is appropriately narrower than a general claim to
discover action-fidelity failures. Primary-source spot checks confirmed the
relevant papers exist, including [LeWM](https://arxiv.org/abs/2603.19312),
[The Intervention Gap](https://arxiv.org/abs/2608.29998),
[IMPLY](https://arxiv.org/abs/2609.12441),
[PhysWeep](https://arxiv.org/abs/2609.06207),
[Do-JEPA](https://arxiv.org/abs/2609.37378),
[AD-WM](https://arxiv.org/abs/2609.30264),
[ARC-Bench](https://arxiv.org/abs/2609.05461), and
[Sensitivity Shaping](https://arxiv.org/abs/2606.14585).
This was a targeted bibliographic/abstract check, not a new exhaustive literature
review or independent verification of every claim about those papers.

## Fixes ranked for the remaining two days

1. **Use roughly four to five pages to explain the experiment.** Three pages is
   allowed; the problem is missing explanation, not page count itself. Add one
   diagram of observed history → frozen encoder → action-conditioned predictor →
   readout → simulator comparison. A concrete PushT example would make the
   physical meaning of compression accessible. Replace some internal study-ID
   emphasis with names that communicate the experiment's purpose. Current figures
   are clean; Table 1 and prose are dense, and floats split paragraphs across pages.
2. **Make the training and measurement self-contained.** Add a compact table of
   training/validation/test sizes and four fine-tuning arms. Distinguish
   archive-weighted losses on designed pushes from training on ordinary archive
   windows. Include 600 training/120 validation designed families, the readout's
   50k/10k frame split, early-stopping settings, and the loss equation with the
   autoregressive feedback and frozen modules. Define energy ratio used in Figure 2
   (squared displacement-energy ratio), terminal endpoint error, finite-candidate
   regret and transfer fraction. Explain the movement/sample-count eligibility
   rule accompanying the later real-frame gates and their scope across studies.
3. **Lead adaptation results with the valid TRAIN-002 comparison.** The current
   paragraph leads with TRAIN-001's descriptive improvement after its failed
   development measurement gate. Give the fresh TRAIN-002 frozen-versus-balanced
   result and absolute means first. Keep TRAIN-001's failed eligibility and the
   dose-omission arm's 48-versus-64 effective examples explicit. Add saved training
   or validation curves if they clarify optimization; no new training is needed.
4. **Tighten the remaining inferential wording.** At `main.tex:149`, replace
   “has no effect” with “no structure effect was established,” retaining
   0.001 [-0.014, 0.017]. At lines 170–171, report the lower nearest-neighbor
   outlier-flag rate rather than asserting distance to the true latent manifold.
   At lines 226–227, describe aggregate agent displacement slopes near one in
   gated cells rather than implying accurate agent motion generally. A useful
   leading qualifier is “decoded block displacement” wherever physical fidelity
   is summarized without the readout caveat nearby.
5. **Complete a small reproduction handoff.** Have Inbal rebuild Route A on a clean
   clone and verify one saved-record result with an explicit runnable procedure.
   Record commands, revision, outcomes and limits. This is a useful contribution
   within the deadline; it is not equivalent to independent model training or
   full fresh-data replication. Ensure repository/artifact access works for the
   evaluator. Check the final exported PDF after any revisions.

## GPU decision and likely defense questions

**Defer the proposed GPU extension for the course submission.** It could improve
generality and test pretraining-support hypotheses, but it is not needed to supply
a substantial training component now. New pretraining, new environments or new
measurement protocols two days before submission create interpretation and
integration work. The existing evidence supports a strong bounded project.
This recommendation does not cancel the separate publication roadmap.

Both authors should be able to explain:

- What was trained, what stayed frozen, and why the future frames enter only the loss.
- Why real-frame readout accuracy does not prove validity on predicted latents.
- Why families, rather than branches, are the bootstrap units, and what seed averaging
  does not tell us about independently pretrained models.
- Why a below-one projection slope alone is not pure magnitude loss, and which
  magnitude/alignment checks support the interpretation here.
- Why the omission experiment does not isolate exact-dose coverage from batch size.
- Why an inconclusive planning contrast does not show that the distortion is harmless.
- What each author actually contributed and can reproduce.

## Verification performed in this review

- `git fetch origin`: remote main updated to 2e3af92; `git diff --stat HEAD origin/main`
  empty. Review began with a clean working tree.
- `make check && make test`: checks passed, **347 passed**, 25 warnings, 16.57 s
  pytest runtime. Warnings concern Gymnasium input casting and joblib core detection.
- `/opt/homebrew/bin/pdftotext -layout <handout> -`: read all four pages;
  `pdftoppm` rendered the handout and report. Visually inspected the relevant
  handout pages (1, 3, 4) and all three report pages. No clipping/overlap found.
- Inline `.venv/bin/python` verification: `reporting.numbers()`, `macros_tex()`
  and `claims_md()` match the saved 46 macros and ledger exactly.
- Inline `.venv/bin/python` verification: `train002.verify_seal(protocol())` passed;
  loaded all 5,040 confirmation records in sorted collection-unit/model order,
  checked record SHA-256 sidecars, unit/model identifiers, model fingerprints and
  readout hashes; called `train002_metrics.summarize` with the sealed gates and
  protocol bootstrap settings. The entire result equals both sealed
  `outputs/train002/confirm/summary.json` and the tracked runtime manifest summary.
- Inspected the actual differentiable training rollout and trainable boundary.
  No future target enters predictor context in that implementation.

Limits: no new simulator collection, model inference or training; no full clean-clone
reproduction; no independent re-audit of every historical result. Matching saved
estimates establishes computational consistency, not universal measurement validity.
The current report and all scientific artifacts remain unchanged.
