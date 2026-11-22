# LIT-002 — Literature for a main-conference extension

Source review, searched 2026-09-30 and written 2026-10-01. It follows an external review
that asked what a main-conference (ICLR/ICML/NeurIPS) paper would need. No experiment was
run, no data were opened and no protocol changed. Paper claims are author-reported and
come from abstracts unless marked; nothing was reproduced. Every idea below is a
hypothesis or proposal, not a result.

## Bottom line

- **The niche has narrowed sharply.** August–September 2026 preprints already audit LeWM
  intervention effects, sweep push speeds, and name "reversion" as a failure mechanism.
  A remedy lane (auxiliary losses on LeWM) is crowded.
- **The best available question** sharpens the review's candidate: *is action-effect
  distortion in latent world models a predictable reversion toward the state-conditional
  typical training transition?* That account makes checkable predictions in new
  conditions, including the sign of the gain error (§2).
- **Measurement comes first.** The current projection slope can fall below 1 for an
  unbiased, honestly uncertain predictor. Add a calibration-side slope and a readout-free
  simulator-inversion instrument before any mechanism claim (§3).
- **Replication is now cheap to start.** Official LeWM checkpoints exist for three more
  environments. JEPA-WM, DINO-WM and a diffusion world model exist for Push-T (§5).

## The review's requirements (summary)

1. **Measurement.** Confirm the physical finding with an independent instrument whose
   failure modes differ from a readout fitted on real latents.
2. **Mechanism.** Separate low exposure to large commands, large-displacement difficulty,
   contact configuration, generic adaptation, and readout distortion. The explanation
   must predict a new condition.
3. **Scope.** A second model or objective, a second environment, and replication that
   separates fine-tuning seeds from independently pretrained checkpoints.
4. **Consequence.** Harm to control under a usable planner, or a failure of common
   evaluation metrics.
5. **Remedy (optional).** Held-out magnitudes, structures or environments; matched data
   and compute; simple calibration; a prespecified ordinary-case margin.

It recommends staging: measurement feasibility, then mechanism, then broader replication.

## Search record

- **Tools.** 14 OpenAlex queries (`title_and_abstract.search`) and 50 arXiv API queries,
  plus 55 full abstracts fetched by arXiv ID and 4 Hugging Face Hub lookups on
  2026-09-30. The paper-search plugin's helper scripts were missing, so both APIs were
  queried directly.
- **Yield.** 225 unique records screened; about 45 are cited below.
- **Themes.** Action-conditioned JEPA and latent world models, OOD extrapolation and data
  support, contact dynamics, forecast verification and measurement error, replication
  and robust fine-tuning, planning consequences.
- **Limits.** Targeted, not systematic. Most 2026 sources are unreviewed preprints read at
  abstract level. The pace of LeWM follow-ups warrants a re-check before any submission.

## 1. The novelty boundary moved in August–September 2026

| Source | Author-reported content | Consequence here |
| --- | --- | --- |
| [The Intervention Gap, Vakalis, 2608.29998](https://arxiv.org/abs/2608.29998) | Audits whether a model's own transitions move task variables as matched environment interventions do. On Cheetah, three LeWM checkpoints' imagined 5-step effects are worse than predicting no effect: "task-direction rotation with excess gain". Ensemble disagreement ranks error only near training support. | Closest precedent: LeWM effect fidelity is already audited. Its gain error has the opposite sign to our Push-T shrinkage. |
| [IMPLY, Mehta & Baviskar, 2609.12441](https://arxiv.org/abs/2609.12441) | Pushes at several speeds; inverts a simulator to read the physics each rollout implies. Self-consistency cannot detect a model that "always predicts a typical push", and V-JEPA 2-AC shows this. | Speed sweeps and simulator inversion have precedent; "typical push" is our hypothesis in another model. |
| [PhysWeep, Khanbayov & Kurban, 2609.06207](https://arxiv.org/abs/2609.06207) | Parameter sweeps for video generators. Recovers realized values with a tracker, uses a deterministic-simulator positive control, and adjudicates "prior reversion" against "case-based clamping" with a leave-one-out rule. Finds neither: seed-locked wrong constants. | A method template for §2. Reversion is already a named hypothesis for video generators. |
| [Sensitivity Shaping, Yu et al., 2606.14585](https://arxiv.org/abs/2606.14585) | Unsupported actions can produce latent predictions that resemble demonstrated transitions. Proposes support-conditioned control-sensitivity regularization. | Same mechanism family; a remedy baseline. |
| [Do-JEPA, Resani & Shi, 2609.37378](https://arxiv.org/abs/2609.37378) | Branches from one saved simulator state and trains predicted latent effect differences. On LeWM: latent effect error −28.4%, physical effect error −13.5%. | Branch-based effect supervision of LeWM is taken. |
| [ARC-Bench, Zhang & Li, 2609.05461](https://arxiv.org/abs/2609.05461) | Released JEPA-WM checkpoints rank fixed candidates badly; closed-loop replanning masks it. | Supports the evaluation-practice route; predicts that replanning hides our distortion. |
| [WorldEcho, 2608.24885](https://arxiv.org/abs/2608.24885); [DUET-DINO, 2609.10506](https://arxiv.org/abs/2609.10506); [AV world-action model, 2606.12987](https://arxiv.org/abs/2606.12987) | Video world models ignore off-expert actions. V-JEPA 2 underestimates dynamics from fine-grained actions. Single-pass regression models capture under half of true motion magnitude, and distortion metrics reward the blurry mean. | Under-predicted action effects recur across model families. |
| [DA-LeWM, 2608.18746](https://arxiv.org/abs/2608.18746); [SCALE, 2608.16287](https://arxiv.org/abs/2608.16287); [JEPA-x, 2608.24044](https://arxiv.org/abs/2608.24044) | Decodability does not track planning: probe scores stay similar while success rises, and state regression improves decoding but not forecasting or control. | "Probes don't predict planning" is established. |

The remedy lane is crowded. SCALE, [AnisoWM](https://arxiv.org/abs/2609.37441),
[SALT](https://arxiv.org/abs/2609.33595), [Bilinear WM](https://arxiv.org/abs/2609.36305),
[FIRM-WM](https://arxiv.org/abs/2609.22816), JEPA-x,
[PSG-JEPA](https://arxiv.org/abs/2608.06799), [Flow-JEPA](https://arxiv.org/abs/2608.29029),
[D-JEPA](https://arxiv.org/abs/2609.24749) and Do-JEPA each add a loss or structure to a
LeWM-style model and report planning gains.

**Remaining niche (an inference, not a novelty proof).** A quantitative account that
predicts when and in which direction action effects are distorted. It would need
measurement independent of a readout fitted to real latents, and tests across models.
IMPLY, PhysWeep and the Intervention Gap each cover part of this; none claims all of it.

## 2. Best idea: test reversion toward the typical training transition

**Sources.**
- [Kang et al. 2023](https://arxiv.org/abs/2310.00873): as high-dimensional inputs
  become more OOD, predictions tend toward the *optimal constant solution*. This holds
  across datasets, losses (including MSE) and architectures. Whether it transfers to a
  low-dimensional conditioning input is exactly what §4.4 tests.
- [PhyWorld, Kang et al. 2024](https://arxiv.org/abs/2411.02385): video models generalize
  "case-based", mimicking the closest training example, and fail OOD.
- [Xu et al. 2021](https://arxiv.org/abs/2009.11848): ReLU MLPs extrapolate linearly
  along rays. This contrasting prediction applies to structured action pathways.

**Hypothesis B.** For history *s* and command *a*, the decoded predicted displacement
approaches `w·t + (1 − w)·m(s)`.
- *t* is the true displacement.
- *m(s)* is the typical displacement from similar states under the training action
  distribution.
- *w* falls as the training density of *a* given *s* falls.

It must work through *conditional* density. DIST-001 found no study block above the
archive's marginal p99 magnitude (0.683), and the challenge median sat at the 98th
percentile. The zero joint-tail mass reflects within-block constancy, which SHIFT-002
showed does not change R.

**What B would explain, if it holds:**
- the magnitude trend (SHIFT-002 R 0.91 → 0.74 challenge, 1.04 → 0.88 archive);
- faithful archive windows;
- the dose-independent 26–28° direction error in designed challenge states, if those
  states are rare in the archive (not yet measured);
- TRAIN-001's gains from adding high-magnitude examples;
- the Intervention Gap's excess gain, where the typical transition may exceed the true
  effect. The account would then predict the *sign* of the gain error.

**Existing data do not discriminate yet.** The 1.27 predicted length ratio for 0.30 pushes
(REVIEW_FOLLOWUP_RESULT §4) is not evidence of over-prediction. The norm of a vector with
zero-mean readout noise is biased upward, most for small displacements. The unbiased
inner-product ratio at 0.30 is 0.913 [0.836, 0.990].

**Competing accounts.** Adjudicate them with a prespecified held-out selection rule, as
PhysWeep does, rather than by best fit. D is motivated by
[Parmar et al. 2021](https://arxiv.org/abs/2103.15406) and
[Bianchini et al. 2021](https://arxiv.org/abs/2112.06881): neural models smooth nearly
discontinuous contact dynamics. Readout distortion is handled in §3.

| Condition | A: attenuation g(\|a\|) | B: reversion to m(s) | C: case-based clamp | D: contact smoothing |
| --- | --- | --- | --- | --- |
| Doses beyond 0.60 (0.8, 1.0; within the declared [−1, 1]) | ratio keeps falling | effect falls back toward m(s) | effect plateaus at the support edge | error follows contact events, not dose |
| Small push where m(s) is large | under-predicted | over-predicted | about faithful | error near contact onset |
| Large command in free space, if common in that state | under-predicted | faithful | faithful | faithful |
| Equal commands, unequal block displacement (contact geometry) | follows command | follows p(a \| s) | nearest example | follows contact onset |
| Pretraining tail removed | unchanged if architectural | onset moves with density | clamp level moves | unchanged |

## 3. Make the physical finding survive independent measurement (review item 1)

### 3.1 Report the calibration side, not only the projection slope

`projection_slope = Σ(p·t)/Σ(t·t)` regresses the prediction on the outcome. That is the
likelihood–base-rate side of
[Murphy & Winkler's (1987)](https://doi.org/10.1175/1520-0493%281987%29115%3C1330%3AAGFFFV%3E2.0.CO%3B2)
forecast-verification framework. For a conditional-mean predictor, E[p·t] = E[|p|²] ≤
E[|t|²]. **The slope therefore falls below 1 whenever outcomes are uncertain, even
without bias.**

Add the calibration slope `Σ(p·t)/Σ(p·p)`
([Van Calster et al. 2016](https://doi.org/10.1016/j.jclinepi.2015.12.005),
[2019](https://doi.org/10.1186/s12916-019-1466-7)). It equals 1 for a conditional-mean
predictor. Values above 1 mean predictions are too small even given the model's own
information.

Design notes:
- Readout noise inflates Σ(p·p) and attenuates this slope (regression dilution:
  [Hutcheon et al. 2010](https://doi.org/10.1136/bmj.c2289);
  [Carroll et al. 2006](https://doi.org/10.1201/9781420010138)). Correct it with the
  real-frame noise variance, and report sensitivity, because the noise on predicted
  latents is unknown.
- Moved-candidate selection conditions on the outcome. It leaves the current slope
  unbiased but biases the calibration slope; select by command or state instead.
- Use inner products or noise-corrected squared norms instead of length ratios.

A first pass is an exploratory CPU reanalysis of opened records. If the frozen model's
high-dose calibration slope is near 1, the shrinkage is honest uncertainty and the paper
changes. If it is well above 1, the bias claim is much harder to dismiss.

### 3.2 A readout-free instrument: simulator-in-the-loop inversion

Estimate the state a predicted latent encodes as `argmin_s ‖enc(render(s)) − ẑ‖²` over
agent and block pose, using CMA-ES or a grid plus local search. This extends GALLERY-001's
finite gallery, which already passed renderer identity and encoder retrieval.

- **Different failure modes.** It returns only valid physical states, and the only learned
  part is the frozen encoder. It fails through non-identifiability, not by regressing
  toward a readout's training distribution.
- **Gates.** Measure the floor on real latents. Use restarts to detect flat directions.
- **Precedent.** IMPLY and PhysWeep use simulator inversion or independent trackers with a
  deterministic-simulator positive control; render-and-compare is established
  (GALLERY-001_RELATED_WORK).
- **Power.** The archive-scene readout floor (about 12 units) dominates the ordinary-case
  contrasts. A lower-noise instrument could buy power more cheaply than more windows (§7).

### 3.3 Readouts with opposite extrapolation behaviour

- **Linear.** It extrapolates linearly.
  [The Observer Effect (2602.12218)](https://arxiv.org/abs/2602.12218) argues low-capacity
  probes are more faithful OOD, and
  [Klindt, LeCun & Balestriero (2605.26379)](https://arxiv.org/abs/2605.26379) prove linear
  identifiability for LeJEPA-style Gaussian-regularized latents. The ridge readout's
  imprecision (15.7 units) is tolerable for slopes if its noise is zero-mean.
- **MLP.** The current `mlp-20261005`.
- **kNN over a bank of real latents.** It clamps to the manifold and can itself
  manufacture reversion.
- **Off-manifold score.** Distance from each predicted latent to its nearest real latent,
  relative to real–real distances. Report readout disagreement as a function of it.

If the linear readout and the inversion agree with the MLP, readout extrapolation is an
unlikely explanation. If only the kNN and MLP readouts show reversion, suspect the
instrument.

### 3.4 Optional pixel route

Decode predicted latents with an RCDM-style decoder
([Bordes et al. 2022](https://arxiv.org/abs/2112.09164)) and estimate T-block pose with a
non-learned tracker. `facebook/jepa-wms` ships DINOv2 decoder heads, so this is nearly
free for the DINO-WM and JEPA-WM route.

**Gate.** Make a physical-mechanism claim only if at least two instruments with different
failure modes agree in sign and rough size on development data. Then freeze the protocol
and use fresh families. TRAIN-001's failed gate stays in the record.

## 4. Separate the causes (review item 2)

Ordered by cost. Items 1–6 need only CPU and existing infrastructure.

1. **Support map.** From the 2.3M-row archive, estimate the density of commands given the
   relative agent–block pose, and *m(s)* by nearest neighbours over archive states. This
   turns "low exposure" into a measured covariate.
2. **Agent-only probe.** Issue large commands with no block contact and decode agent
   displacement. Faithful agent motion with shrunken block motion implicates interaction;
   shrunken agent motion implicates the action pathway. The official LeWM Reacher and
   TwoRooms checkpoints offer contact-free environments.
3. **Contact-geometry dissociation.** Vary the initial gap and contact offset so equal
   commands give unequal block displacement, and the reverse. SHIFT-002 hints at this
   exploratorily: 0.45 and 0.60 moved the block a similar median distance (21 units),
   yet R fell from 0.81 to 0.74.
4. **Activation patching**
   ([Heimersheim & Nanda 2024](https://arxiv.org/abs/2404.15255);
   [Zhang & Nanda 2023](https://arxiv.org/abs/2309.16042)).
   - In the pinned `le-wm` source, actions enter only through `Embedder` (a 1×1 Conv1d and
     a SiLU MLP) and each block's adaLN shift, scale and gate. The predictor output then
     passes a final LayerNorm and a projector.
   - Replace an OOD command's embedding with a linear extrapolation of in-support
     embeddings. If predictions become faithful, the bottleneck is the action pathway.
   - Add action-Jacobian sensitivity against support
     ([Walker et al. 2026](https://arxiv.org/abs/2609.33940)).
5. **State-space baseline.** Train a small dynamics model on simulator poses from the same
   archive, with matched information: three historical poses, not velocities. If it
   reverts at the same density, the effect comes from data support and function
   approximation, not the latent space. If not, the latent adds distortion.
6. **Dose-omission fine-tuning.** This is the control TRAIN-001 lacked.
   - Arms, with matched updates and examples: (i) doses ≤ 0.45 only; (ii) all doses;
     (iii) archive only.
   - Test at 0.60 and beyond.
   - [Byrd & Lipton (2019)](https://arxiv.org/abs/1812.03372) found importance weighting's
     effect fades over training in over-parameterized networks. That is a candidate
     explanation for TRAIN-001's null weighting contrast. Manipulate support, not weights.
7. **Support-manipulated pretraining** (GPU; the strongest causal test).
   - LeWM has about 15M parameters and trains "on a single GPU in a few hours"
     ([Maes et al.](https://arxiv.org/abs/2603.19312)).
   - Pretrain at least three LeWMs per arm, with and without the high-magnitude tail, at
     equal sample count and compute.
   - B predicts that reversion onset follows the density change.

## 5. Beyond one checkpoint (review item 3)

**Units of replication.** [MultiBERTs (Sellam et al. 2022)](https://arxiv.org/abs/2106.16163)
argues that claims about a training *procedure* need independently pretrained checkpoints.
Its Multi-Bootstrap resamples checkpoints and examples jointly. TRAIN-001's seeds are
fine-tuning seeds.

**Released checkpoints** (Hub, checked 2026-09-30):

| Checkpoint | Why it helps | Caveat |
| --- | --- | --- |
| `quentinll/lewm-{cube,tworooms,reacher}` | Same code path; different dynamics: 3-D contact manipulation, navigation, a contact-free arm | Each needs its own interface audit |
| `facebook/jepa-wms`: `jepa_wm_pusht`, `dino_wm_pusht` (plus PointMaze, Wall, Metaworld) and DINOv2 decoder heads | A frozen DINOv2 patch encoder and a different objective on Push-T | CC-BY-NC 4.0. DINO-WM's Push-T data and simulator differ from swm: a shared name is not compatibility |
| `knightnemo/nanowm-b2-dino-wm-pusht-100k` | A diffusion-forcing (sampling) world model; separates conditional-mean averaging from support effects | Community checkpoint of 158.6M parameters; needs an audit |

**Comparisons that challenge the account:**
- Action-structured dynamics (Bilinear WM, SALT): if reversion disappears, the
  unstructured action pathway is implicated.
- Stochastic predictors (Flow-JEPA, diffusion) isolate regression to the mean.
- Action-sensitive losses (AD-WM, Do-JEPA, Sensitivity Shaping) train on the same support.
  B predicts they help within support but not at held-out magnitudes.

**Platforms.** [stable-worldmodel](https://arxiv.org/abs/2605.21800) offers baselines and
environments with controllable physical factors. [EB-JEPA](https://arxiv.org/abs/2602.03604)
targets single-GPU training in hours.

## 6. Why it matters (review item 4)

**Control route.** B predicts two signatures under LeWM's own CEM protocol (goal offset 25,
budget 50, horizon 5; LIT-001):
- Under-predicted large effects push CEM toward larger commands at the action bound, with
  open-loop overshoot.
- If small effects are over-predicted, near-goal corrections are too timid and progress
  stalls.

ARC-Bench suggests replanning masks both. Measure first-plan outcomes and vary replanning
frequency, with agent-matched goals as in H3-PLAN-002. Isolate the distortion by comparing
the frozen model with a model corrected for it alone (for example, a density-aware affine
latent correction), all else matched.

**Evaluation-practice route.** Show that standard metrics miss the distortion *and change a
decision*. Candidates are LeWM-style real-latent probes (r 0.986–0.999 reported; LIT-001),
latent MSE, and closed-loop success (ARC-Bench).
- An exploratory seed from one checkpoint: in TRAIN-001, latent MSE prefers balanced
  weighting (−0.00684), while ordinary-case physical error prefers archive weighting
  (+0.38 [0.05, 0.69]).
- Framing references: [objective mismatch (Lambert et al. 2020)](https://arxiv.org/abs/2002.04523)
  and [World-in-World (2025)](https://arxiv.org/abs/2510.18135), which found controllability
  matters more than visual quality.

## 7. If a remedy is included (review item 5)

- Test on held-out magnitudes, action structures and environments.
- Baselines: ordinary fine-tuning, matched data and compute, and calibration of the right
  functional form. TRAIN-001's global scale (1.03) cannot fix an affine reversion.
- Cheap robust fine-tuning:
  - [WiSE-FT](https://arxiv.org/abs/2109.01903) interpolates frozen and fine-tuned
    predictor weights, giving an ordinary-versus-high-magnitude trade-off curve at no
    training cost.
  - [LP-FT](https://arxiv.org/abs/2202.10054),
    [Sandwich-Residuals](https://arxiv.org/abs/2609.21740) and
    [AdaJEPA](https://arxiv.org/abs/2606.32026).
  - Do-JEPA and Sensitivity Shaping as method baselines.
- **Ordinary-case power.** This is a normal approximation from TRAIN-001: SE ≈ 0.48 units
  at 120 windows, assuming independent windows and unchanged variance. For 80% power to
  show non-inferiority at 1.28 units:
  - about 360 windows if the true cost is 0.5 units;
  - about 2,800–4,100 windows if it is 1.0–1.05 units, as observed.

  Use TOST ([Lakens 2017](https://doi.org/10.1177/1948550617697177)).

## 8. Recommended sequence

| Stage | Work | Compute | Stop if |
| --- | --- | --- | --- |
| 1. Measurement | Calibration-side reanalysis of opened records (exploratory); inversion instrument with a real-latent floor; linear and kNN readouts; off-manifold score | CPU | independent instruments disagree on the shrinkage |
| 2. Mechanism | Support map; agent-only and contact-geometry probes; action-pathway patching; state-space baseline; dose-omission fine-tuning; then preregister B against A, C and D for new conditions and confirm on fresh families | CPU | B does not beat the alternatives on held-out conditions |
| 3. Replication and consequence | LeWM Cube, Reacher and TwoRooms; JEPA-WM, DINO-WM and diffusion on Push-T; support-manipulated pretraining (≥3 seeds per arm); planning signatures | GPU | effects fail to replicate beyond this checkpoint |

As the review says, if the finding disappears under independent measurement, or the
explanation reduces to ordinary distribution adaptation with no predictive content,
reconsider the main-conference direction.

**Read in full before scoping.** Only abstracts were read here: the Intervention Gap,
IMPLY, PhysWeep, Sensitivity Shaping, Do-JEPA, ARC-Bench, and Kang et al. 2023.

## Verification

Documentation only. No simulator, checkpoint, readout or confirmation data were accessed.
- Architecture facts in §4 come from the cached pinned source
  (`.cache/env001/le-wm-ca231f9…/module.py`: `Embedder`, `ConditionalBlock`,
  `Transformer`).
- The projection-slope definition comes from `train001_metrics.projection_slope` and
  `followup_metrics`.
- Archive percentiles come from `docs/manifests/dist001-runtime.json`.
- Project numbers are quoted from SHIFT-002_RESULT, REVIEW_FOLLOWUP_RESULT,
  TRAIN-001_RESULT and LIT-001_REVIEW.
- The search scripts were scratch tools and are not committed.

## Addendum — 2026-10-01

§3.1 was run as CALIB-001 (D030; CALIB-001_RESULT.md).
- The frozen model's high-dose calibration slope is well above 1 in archive-type scenes,
  and 1.33 on PHYS-002 challenge families. The shrinkage is therefore not honest
  uncertainty, conditional on readout validity.
- The reversion account in §2 fits the compression and the null-dose over-prediction. It
  does not explain the over-dispersion in challenge states.
- The inversion instrument in §3.2 is now the deciding measurement.

## Correction addendum — 2026-10-01 (CALIB-REVIEW-001, D031)

- **Reversion (§2)** remains a hypothesis to test against alternatives, including
  "neither". PhysWeep's adjudicated examples rejected both reversion and clamping. Kang et
  al. motivate the hypothesis but do not establish it for a state-conditioned action
  pathway. The addendum above overstated the CALIB-001 support; see the D031 corrections.
- **§3.1:** b_cal ≠ 1 does not identify predictor bias, because the readout is nonlinear
  in a latent-predicting model. Report magnitude (energy ratio) and alignment separately.
  The noise adjustment needs an orthogonal-error model.
- **§3.2:** inversion is a candidate decoder using the same encoder, not the deciding
  measurement. It needs:
  - residual-based abstention ("not identifiable");
  - near-optimal ambiguity checks;
  - optimizer-recovery controls;
  - controlled off-manifold and latent-mixture fixtures;
  - nuisance stability.

  Its gates must be fixed before scoring reserved outcomes. Agreement with the MLP readout
  would be corroboration, not validation.
- **§7:** ordinary-case non-inferiority is a one-sided test at a chosen α and margin, not
  TOST (an equivalence test). Any power calculation must use the actual independent
  sampling unit; the figures given are rough normal approximations only.
- **§6:** agent-matched goal images are privileged and candidate-specific. A planning
  study must distinguish them from a deployable goal specification.
