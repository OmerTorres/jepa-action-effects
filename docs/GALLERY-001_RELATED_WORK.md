# GALLERY-001 — Closest work and contribution boundary

Reviewed 2026-09-28 from primary papers and an author repository. This is a
bounded source comparison, not an exhaustive search or an independent
reproduction. The method and model results must be described separately.

## Search record

The starting set was LIT-001_REVIEW.md and NOVELTY-002_MEMO.md. Web searches
on 2026-09-28 used these concepts: `renderer-based latent representation
evaluation object pose templates frozen world model counterfactual`,
`analysis by synthesis latent world model object state retrieval rendered
candidate poses`, `object centric counterfactual interventions latent world
models evaluation agent nuisance control`, `world model evaluation rendered
counterfactual object position latent embedding nearest template benchmark`,
and `category level pose retrieval rendered embedded reference set`. The
closest JEPA/action work led to LeWM, AD-WM and Delta-JEPA; the render/compare
trail led to the pose papers below. We inspected ten primary sources in detail.
The AD-WM author project page linked from its paper did not load in this
review, so its comparison uses the paper text; no AD-WM code claim is made.
The LeWM source comparison in LIT-001 uses a pinned author repository revision,
while the WAMProbe comparison below uses its mutable author repository.
Driving scene generation and language-model counterfactual results were
excluded because they do not test the present image-to-projected-CLS interface.
This bounded search cannot prove the absence of another matching method.

## Direct comparison

| Primary source and verified version | Relevant mechanism | Boundary for this work |
| --- | --- | --- |
| [AD-WM, Qiu et al., arXiv v1, 24 Sep 2026](https://arxiv.org/html/2609.30264), §III-E and §IV-D | Matched LeWM baseline, PushT in five-task simulation, factual prediction versus action comparison, whole-bank ranking and elite regret; controlled diagnostic emphasis on Cube | The general action-discrimination and planning-metric question is established. Its action-recovery training and elite-selection result differ from a frozen LeWM block-pose representation audit. |
| [LeWorldModel, Maes et al., arXiv v1, 13 Mar 2026](https://arxiv.org/html/2603.19312v1), §5 and App. F | Published physical probes of latent state plus image-goal planning, including PushT; violation-of-expectation interventions | Existing probes make “the first object-state test of LeWM” false. They do not certify our physical precision gate on these contact families or supply a nuisance-fixed rendered gallery for this checkpoint. |
| [WorldSimProbe, Co et al., arXiv v1, 10 Aug 2026](https://arxiv.org/html/2608.09298v1), Tasks 1, 4, 5 | Simulator-referenced action calibration, interaction grounding and dynamics for generated video | Contact-aware simulator comparison is established. Its video-based evaluator is not a direct metric for a latent-only predictor. |
| [Delta-JEPA, Zhang et al., arXiv v1, 30 Jun 2026](https://arxiv.org/html/2606.31232v1), §Experiments | Action-sensitive latent dynamics, PushT, state/delta probes, action recovery as training supervision | Latent action sensitivity and PushT are precedents. A frozen-checkpoint, rendered true-versus-alternative block-position comparison asks a narrower empirical question. |
| [Causal-JEPA, Nam et al., arXiv v2, 28 May 2026](https://arxiv.org/abs/2602.11389), abstract and method | Object-level latent masking during world-model training, with counterfactual-like queries | Object intervention is established as a training idea. It does not make rendered pose retrieval a new general method. |
| [CoCo, Shi et al., arXiv v1, 5 Aug 2026](https://arxiv.org/html/2608.04653v1), §Method and §Experiment | Counterfactual action consistency and a same-state multiple-action dataset | Same-state comparisons are established. Its control metrics require a different output and assumptions; we do not borrow their names. |
| [WAMProbe, author repository, accessed 28 Sep 2026](https://github.com/myheart521/WAMProbe), README | Reset-matched branches, action response, state error, and ranking/regret profiles | Paired action evaluation is established. The repository is mutable and its README is not an independent demonstration on this checkpoint. |
| [Krull et al., arXiv v1, 19 Aug 2015](https://arxiv.org/abs/1508.04546), abstract | Compares observations to rendered object poses for 6D pose estimation with a learned comparison model | Analysis by synthesis predates this project; rendering candidate poses is not method novelty. |
| [Periyasamy et al., arXiv v1, 8 Oct 2019](https://arxiv.org/abs/1910.03412), abstract | Refines object pose by comparing rasterized scenes in a learned abstract feature space | Feature-space render-and-compare also predates this project. Our frozen projected CLS and action-conditioned predictor question differ, but the general measurement family is established. |
| [Kouros et al., arXiv v3, 12 Oct 2022](https://arxiv.org/abs/2208.06195), abstract | Pre-renders and embeds a pose reference set, then retrieves poses by embedding comparison | The closest procedural precedent for a rendered embedding gallery. Its learned category-level pose estimator differs from using a frozen world-model representation as an evaluation instrument. |

One citation hop from the closest action paper reaches LeWM/Delta-JEPA; the
render-and-compare trail reaches earlier pose-estimation work. The 2020
[neural analysis-by-synthesis pose paper](https://arxiv.org/abs/2008.08145)
was also screened, but its continuous optimization of a synthesis network adds
less to the specific finite-gallery comparison than the three pose references
above. This is an exclusion from the detailed matrix, not an absence claim.

## Contribution decision and required evidence

**Method novelty is unsupported.** Analysis by synthesis, rendered feature
comparison, pose retrieval, object interventions, and action discrimination
all have precedents. A new name for their combination would overclaim.

The defensible empirical question is: **for one frozen LeWM PushT checkpoint,
does its projected CLS distinguish realized from nearby admissible block poses
when agent/goal imagery is fixed, and can its action-conditioned predictions
then select the realized block pose?** GALLERY-001 addresses only the first
clause. A subsequent registered predictor audit would be needed for the
second; it must compare to persistence and action permutations and report
contact/motion coverage. A positive gallery encoder result cannot itself show
object-effect prediction, and a negative one would make the proposed predictor
comparison difficult to interpret.

| Claim | Evidence needed |
| --- | --- |
| Rendered position gallery measures block sensitivity in this representation | Exact replay/restoration, valid geometry, nonidentical-query retrieval and margins in both cohorts |
| Frozen predictor uses actions to predict block outcomes | Later sealed predictor-to-gallery comparison, persistence/permutation controls, family uncertainty and appropriate regime coverage |
| Physical competence or low decision regret | A separate physical measurement meeting its error gate; the gallery alone cannot establish this |
| General methodological novelty | Broader independent source review and a substantive difference from feature-space pose retrieval; not established here |

This review supports a checkpoint-specific measurement and feasibility case
study. It does not establish a new benchmark, a universal JEPA claim, or the
historical identity of the checkpoint's training simulator.
