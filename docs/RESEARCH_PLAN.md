
# Candidate 1: Contact-dependent action effects in latent JEPA world models

## 1. Objective, contribution, and scope

**Deliver a rigorous, course-scale study of whether a latent JEPA world model predicts object-specific action effects correctly across contact and no-contact conditions, and whether its errors affect action selection.**

The project should begin with evaluation of an existing model. Training a new method is an optional extension, not a requirement for success.

**Confirmed constraints:** two researchers, approximately 8–10 hours each per week, 6–8 weeks, M1/M2 laptops, and intermittent Colab Pro access. This provides approximately 96–160 person-hours. GPU type, memory, runtime allowance, and storage remain unmeasured.

**Current state:** the inspected local project contains instructions but no implementation or dataset. No simulator, checkpoint, or training run has been validated.

### Necessary revision to the novelty claim

Further inspection found closer precedents than the original audit included:

| Established precedent | Consequence |
|---|---|
| [WorldSimProbe](https://arxiv.org/html/2608.09298v1) evaluates action-response calibration and simulator-verified contact hallucination. | Neither contact hallucination nor contact-aware evaluation should be presented as newly discovered. |
| [WAMProbe](https://github.com/myheart521/WAMProbe) documents contact-aware toy environments, paired interventions, and action-selection regret. Its results have not been independently reproduced here. | The general evaluation pipeline and regret metric are not the contribution. |
| [Delta-JEPA](https://arxiv.org/html/2606.31232v1) evaluates PushT, action supervision, and latent action responses. | Moving the experiment to PushT or adding inverse-action supervision is insufficient novelty. |
| [CoCo](https://arxiv.org/html/2608.04653v1) evaluates counterfactual action consistency and explicitly acknowledges limitations involving irreversible contacts. | Do not claim discovery of the limitations of inverse-action assumptions. |

**Intended contribution:** a reproducible empirical extension to latent JEPA models that measures contact-conditioned physical effects, checks whether conclusions survive changes in state readout, and connects measured errors to short-horizon action choice.

A stronger contribution would identify a specific failure pattern that persists after controlling for readout quality, input ambiguity, and unfamiliar actions. No such finding is assumed.

### Research question and hypotheses

**Question:** Under matched initial histories and admissible alternative actions, how accurately does a JEPA world model predict block motion across contact regimes, and what consequences do its errors have for choosing an action?

| Hypothesis | Supporting evidence | Challenging evidence |
|---|---|---|
| **H1: Physical-effect errors are concentrated in particular contact regimes.** | Reproducible excess errors around contact/no-contact distinctions, surviving readout checks and comparisons at similar action magnitudes and true effect sizes. | Errors are uniformly small, or the apparent pattern disappears after controlling for observation quality or effect size. |
| **H2: Action responsiveness is an incomplete diagnostic of object-effect accuracy.** | A trained model distinguishes actions while producing materially inaccurate block effects on the same cases. | Responsiveness and physical correctness agree throughout the tested distribution. |
| **H3: These errors have decision consequences.** | The model selects actions with meaningful simulator-measured regret; correcting block endpoint predictions reduces that regret. | Candidate choices remain near-optimal despite prediction errors. |

H2 is a conditional secondary hypothesis. A mock predictor alone cannot establish it for trained JEPA models. A claim about a published metric requires its exact implementation and applicable inputs.

**Essential scope:** one simulator, one frozen JEPA checkpoint, one primary horizon, paired physical-effect evaluation, readout controls, and finite-candidate action selection.

**Optional:** a second checkpoint or action-supervised configuration, additional training seeds, one additional horizon, or a targeted remedy. Exclude full robotics deployment, new model architectures, extensive sweeps, and a new general-purpose benchmark.

---

## 2. Data, simulator, and feasibility

### Default model and environment

Use the listed [LeWorldModel PushT checkpoint](https://huggingface.co/quentinll/lewm-pusht), subject to successful download, configuration inspection, and inference validation. Its public listing is verified; its weights have not been tested here.

Use the **exact simulator version and preprocessing required by that checkpoint**. Do not substitute a different PushT implementation because its name matches. For example, the documented [stable-worldmodel PushT](https://galilai-group.github.io/stable-worldmodel/envs/pusht/) uses relative velocity control, whereas [gym-pusht](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py) uses target-position commands.

Before collecting data, record:

- Repository commit, package versions, checkpoint hash, and license.
- Action meaning, normalization, control frequency, and action repetition.
- Image size, preprocessing, context length, and proprioceptive inputs.
- Predictor rollout interface, including how future proprioception is handled.

**Never supply ground-truth future observations or future proprioception during prediction.**

### Branch generation

Start from a seeded simulator reset and execute a recorded warm-up sequence. Recreate that reset and replay the identical warm-up separately before each candidate action sequence. This is the default branching method; it avoids relying on incomplete pose-only restoration.

Verify that replay reproduces positions, velocities, orientation, and contact history. Full-state cloning may replace replay only after equivalence tests pass.

Each initial-state family contains:

- Reset seed and warm-up actions.
- Observed history and current simulator state.
- Six candidate future action sequences.
- Simulator trajectories, block and agent poses, velocities, and pair-specific contact records.
- Model predictions, readout predictions, and exclusion reasons.

Generate candidate actions using two fixed, documented components: bounded action perturbations and geometry-guided push/near-miss proposals. Selection may use simulator outcomes, never model errors. Randomize candidate ordering.

### Contact and effect labels

Record agent–block contact at physics-substep resolution. Aggregate environment contact counts are insufficient if they also include walls or other collision pairs.

Label **contact and movement separately**:

- Neither branch contacts the block.
- Exactly one branch contacts it.
- Both branches contact it.

Separately label whether paired block endpoint effects are negligible or meaningful. No contact does not imply no movement: initial velocity may move the block in both branches.

The primary analysis emphasizes families containing a contact and a nearby no-contact candidate. Include ordinary rollouts as a reference distribution so the selected challenge set is not mistaken for natural failure prevalence.

### Planned data sizes

These are collection targets, not available data or guaranteed statistical power.

| Stage | Initial-state families | Use |
|---|---:|---|
| Feasibility pilot | 120 | 60 readout fit, 30 validation, 30 development evaluation |
| Main study | 900 | 400 readout fit, 200 validation, 300 locked test |
| Test expansion, if justified before evaluation | Up to 600 test families | Improve precision using pilot variance estimates |

Each family supplies six branches. Initial-state families, including related warm-ups and perturbations, remain within one split.

At least 50 test families should contribute to each central reported contact comparison. Otherwise report the affected comparison as underpowered rather than filling it through model-error-driven selection.

### Quality checks and access limitations

- Repeat identical reset/warm-up/branch sequences in different execution orders.
- Confirm contact labels against rendered examples and collision identities.
- Check action clipping, termination handling, timestep alignment, and coordinate conventions.
- Verify that rendering contains no future-action overlays or privileged state annotations.
- Preserve all valid outcomes, including near misses and failed pushes.
- Store replay recipes and simulator states; cache model embeddings to reduce repeated inference.
- Check overlap with accessible checkpoint training data. If training records are unavailable, disclose that limitation rather than asserting complete training-set independence.

All data are simulated. No human participants or real robot interactions are planned. Check software/checkpoint redistribution licenses and course requirements; do not assume additional ethics approval is necessary for this setup.

---

## 3. Methodology and essential comparisons

### Model and state readout

Freeze the JEPA encoder and predictor. Fit a ridge-regression readout from encoded real observations to block position. Fit separate outputs for agent position and, secondarily, block orientation represented by sine and cosine.

Select ridge regularization using validation data. Fit a small one-hidden-layer nonlinear readout as a diagnostic, with three initialization seeds. Fit readouts only on real encoded observations—not predicted latents or test labels.

Evaluate the same readout on:

1. Encoded real future frames.
2. Predicted future representations.

The difference helps diagnose where errors arise, but does not perfectly separate encoder, predictor, and decoder causes.

Use one primary horizon: **two native prediction steps**, respecting the checkpoint’s actual temporal stride. During simulator-only development, move to four steps only if fewer than 20% of valid pilot families exhibit a meaningful action-dependent block effect. Freeze this choice before examining model performance.

### Essential comparisons

| Comparison | Purpose and interpretation |
|---|---|
| **Frozen JEPA versus block persistence** | Establish whether learned predictions improve over assuming unchanged block position. Persistence should perform well in genuinely stationary cases. |
| **Simulator oracle** | Validate metrics and establish the best action within the finite candidate set. It is not a learned baseline. |
| **Oracle future frames → encoder → readout** | Quantify state-readout error on exactly the evaluation distribution. |
| **Linear versus nonlinear readout** | Test whether an apparent failure is merely poor linear accessibility of state. |
| **Small supervised state-dynamics model** | Predict block endpoints from full initial simulator state and candidate actions. Diagnose task/data difficulty; label its privileged information explicitly. |
| **Correct versus action-permuted predictor inputs** | Check whether the model uses the action channel. This is an intervention control, not a competing model. |
| **Ordinary versus contact-boundary cases** | Determine whether a failure is localized or reflects general model incompetence. |
| **Model action choice versus random choice and oracle choice** | Establish the practical consequence of endpoint errors. |

The supervised reference uses a small MLP with two hidden layers of 128 units, trained on the same development-family branches. It receives no test labels. Its performance is a diagnostic, not an upper bound.

### Published action-response diagnostics

Within the first week, extract exact definitions and input requirements from the closest papers.

CoCo’s inspected main paper refers to metric definitions in an appendix that was not recovered in the inspected document. Do not reconstruct ARC or DE from prose and present the result as a faithful reproduction.

If an exact applicable published metric cannot be implemented within the first-week review budget:

- Keep its comparison in related work.
- Report action permutation and latent-response visualizations as study-specific diagnostics.
- Remove claims that a published metric was empirically disproved.

Do not add a pixel decoder solely to make a video metric applicable; that would introduce another source of error.

### Optional extensions

Admit at most one after the essential evaluation works:

- A verified second checkpoint, preferably action-supervised.
- Three independent training seeds for a compact, explicitly identified JEPA recipe.
- A paired-outcome remedy, only after a reproducible failure is established.

For a remedy, compare against ordinary training on the **same extra transitions**, matching updates and reporting compute. Counterfactual data alone must not be credited to the paired objective.

---

## 4. Evaluation, uncertainty, and acceptance criteria

For block endpoint \(q_h(s,a)\) and prediction \(\hat q_h(s,a)\), define:

\[
e_{\mathrm{effect}}
=
\left\|
[\hat q_h(s,a)-\hat q_h(s,b)]
-
[q_h(s,a)-q_h(s,b)]
\right\|_2.
\]

**Primary metric:** family-averaged paired block-effect error, reported by contact regime.

**Required secondary metrics:**

- Absolute block endpoint error, to expose biases that cancel in differences.
- Predicted effect magnitude in true zero-effect pairs.
- Missed meaningful effects.
- Agent endpoint error, separately from block error.
- Effect magnitude and direction errors for non-negligible true effects.
- Maximum trajectory displacement for stationary no-contact cases, to catch transient movement that returns to the starting point.

Do not divide by near-zero true effects. Report zero predicted vectors as missed effects rather than assigning an arbitrary direction.

### Action-selection test

For each family, draw four target block positions from a fixed workspace grid, independently of model predictions. Give every model the same six candidate sequences.

Choose the candidate minimizing predicted endpoint distance to the target. Measure:

\[
\mathrm{regret}
=
\|q_h(s,\hat a)-g\|_2
-
\min_{a\in A}\|q_h(s,a)-g\|_2.
\]

This measures short-horizon selection among supplied candidates, not full PushT task success.

Evaluate all goals; additionally report informative cases where candidate costs differ materially. Define that subset using simulator outcomes before inference.

### Practical thresholds

Let \(L\) be workspace width and \(\epsilon_{\mathrm{sim}}\) the maximum positional discrepancy in replay checks. Use the provisional physical tolerance

\[
\delta=\max(0.01L,\;10\epsilon_{\mathrm{sim}}).
\]

This is a study-design default, not an established benchmark standard. Confirm its physical relevance using simulator-only plots in week 1 and freeze it before model evaluation.

For a readout to support fine-grained failure claims, require on validation real frames:

- Mean block-position error at most \(\delta/4\).
- 95th-percentile error at most \(\delta/2\), separately in central contact regimes.

If it fails, improve readout training or narrow the claim. Do not simply subtract decoder error from predictor error and treat the remainder as ground truth.

### Statistical strategy

- Average correlated branch-pair and goal measurements within each family.
- Use 2,000 family-level bootstrap resamples for 95% confidence intervals.
- Report paired contrasts on identical families.
- Report nonlinear-readout seed variation separately.
- A single pretrained checkpoint supports claims about that checkpoint, not variation across JEPA training runs.
- If independently trained checkpoints become available, report each seed and their dispersion; branch count does not replace training replication.

Use pilot estimates to choose test size before opening final results. Aim for confidence-interval half-width at most \(\delta/2\) for the central error contrast and regret estimate. If the collection cap is insufficient, report the uncertainty.

Contact strata are not randomized treatments. Compare within-family action perturbations and report effect size, action magnitude, starting distance, and timing. A contact-associated error pattern alone does not prove contact is its causal source.

### Scientific success

A positive failure claim requires:

1. Simulator and readout validity.
2. A practically meaningful, reproducible physical error.
3. Evidence that it is not explained entirely by unfamiliar inputs, hidden dynamics, or general poor performance.
4. Action-selection consequences if decision relevance is claimed.

A precise null result is acceptable. Wide intervals, failed readout checks, or an incompetent baseline yield an inconclusive study, not evidence for or against the hypothesis.

---

## 5. Phased execution plan

Ownership is a proposed allocation: **Omer** leads literature, model integration, and reporting; **Inbal** leads simulator generation, evaluator tests, and statistical analysis. Each reviews the other’s core measurements.

Durations are elapsed working periods with parallel work where possible.

| Phase | Task | Rationale | Dependencies | Deliverable | Estimated duration | Completion criterion |
|---|---|---|---|---|---|---|
| **1 — Essential** | Update closest-work matrix; inspect checkpoint and simulator interfaces | Avoid rediscovery and incompatible model/environment combinations | None | Two-page contribution memo; pinned dependency list | Week 1, 2–3 days | Exact extension stated; metric limitations recorded; no unsupported novelty claim |
| **2 — Essential** | Implement reset/replay branching, collision logging, and fixtures | Establish trustworthy physical interventions | Simulator selected | Tested generator and replay report | Weeks 1–2, 3–4 days | Branch order does not alter outcomes beyond tolerance; pair-specific contacts verified |
| **3 — Essential** | Run the minimum viable study | Expose inference, readout, and coverage problems cheaply | Phases 1–2 | Pilot report, timing measurements, example trajectories | Week 2, 2–3 days | End-to-end data → prediction → physical errors → regret works |
| **4 — Essential** | Freeze hypotheses, tolerances, horizon, splits, and analysis; collect main data | Prevent model-driven test construction | Valid pilot | Versioned protocol and hashed dataset manifest | Week 3 | Coverage and exclusion rules satisfied; final test data locked |
| **5 — Essential** | Fit readouts and supervised reference; run frozen-model inference | Obtain interpretable predictions and controls | Phase 4 | Prediction cache, validation report, baseline results | Weeks 3–4 | Readout gate passed; ordinary-case model sanity checks passed |
| **6 — Essential** | Open test results; compute family-level uncertainty and contact comparisons | Test hypotheses without selective reporting | Frozen models and analysis | Main tables, effect plots, regret analysis | Weeks 5–6 | Every hypothesis classified as supported, challenged, or inconclusive |
| **7 — Essential** | Independent rerun, limitations review, report and presentation | Make the contribution reproducible and defensible | Phase 6 | Course report, reproducibility bundle, slides | Weeks 6–8 | Teammate reproduces a subset; every claim maps to evidence |
| **8 — Optional** | Add one comparator or targeted intervention | Strengthen explanation or scope | Core study complete; time and compute remain | Separate extension analysis | Up to 1–2 weeks within remaining capacity | Does not delay or weaken the essential report |

**Compute rule:** benchmark inference and probe training in week 2. Estimate total runtime from measured throughput, including at least 50% contingency for interruptions. If it does not fit available Colab sessions, reduce optional comparisons first. No full world-model training is on the essential path.

---

## 6. Minimum viable study and decision gates

### Smallest end-to-end experiment

**Inputs:** one functioning checkpoint, its matched simulator, 120 development families, six candidates per family, the model’s native observation history, and simulator ground truth.

**Implementation sequence:**

1. Generate and replay families with contact and near-miss candidates.
2. Validate the evaluator using oracle, persistence, doubled-effect, reversed-effect, and spurious-motion mock predictions.
3. Run frozen-model inference and cache predicted and real-frame latents.
4. Fit the linear readout; add the nonlinear readout only if needed for diagnosis.
5. Compute endpoint error, paired-effect error, contact strata, and candidate-selection regret.
6. Produce a compact report with automatic plots and a fixed random sample of videos, including successes.

**Outputs:** a data-validity report, decoder-validity report, runtime estimate, contact coverage table, and preliminary hypothesis assessment. Pilot results remain exploratory.

### Go/no-go criteria

**Proceed to the main study when:**

- Replay and contact labels pass.
- The checkpoint receives the correct inputs and performs nontrivially on ordinary transitions.
- At least ten pilot families support each central contact comparison.
- Readout error is below the stated gate, or a justified revised measurement passes before the test protocol is frozen.
- Collection and inference fit the available schedule.

**A visible failure is not required to proceed.** A well-measured null result remains scientifically useful.

**Stop expansion when:**

- Reliable branching cannot be established.
- Decoder errors remain comparable to the physical effects.
- The checkpoint is broadly incompetent in the selected environment.
- Closest work already answers the exact question and no substantive extension remains.

In the last case, complete a clearly labeled replication and measurement-validation report rather than manufacturing a novelty claim.

---

## 7. Risks and practical alternatives

| Risk / early warning | Response |
|---|---|
| **Substantial novelty overlap** | Narrow the claim to a measured JEPA-specific result; retain replication as the course deliverable. |
| **Simulator/checkpoint mismatch** | Correct version, rendering, action semantics, and temporal stride before interpreting any failure. |
| **Incomplete state restoration** | Use seeded reset plus complete warm-up replay; do not use pose-only restoration for moving states. |
| **Unobservable initial dynamics** | Check the checkpoint’s available history; restrict the main study to identifiable, low-velocity starts if necessary and state that restriction. |
| **Poor real-frame readout** | Add training coverage and the nonlinear diagnostic; if still inadequate, report representation/readout limitations rather than predictor failure. |
| **Counterfactual actions are far outside training support** | Include familiar-action references and bounded perturbations; label unfamiliar-action results separately. |
| **No measurable contact-specific effect** | Report bounds and null results; do not increase difficulty after viewing test predictions. |
| **Only a tiny or undertrained model fails** | Restrict conclusions to that recipe; do not generalize to JEPA as an architecture family. |
| **GPU access is unreliable** | Cache embeddings, prioritize frozen-checkpoint inference, and omit optional world-model training. |
| **Checkpoint cannot be loaded by end of week 2** | Spend at most one additional week on a compact from-scratch feasibility pilot; otherwise deliver the validated evaluation study without claims about trained JEPA behavior. |

Any post-test change to thresholds, model settings, or case generation creates an exploratory follow-up. Confirm it only on newly reserved families.

---

## 8. First two weeks: prioritized checklist

### Week 1 — establish the scientific and technical foundation

- [ ] **Omer:** compare Candidate 1 explicitly with WorldSimProbe, WAMProbe, Delta-JEPA, CoCo, and UWM-JEPA; write the precise remaining distinction.
- [ ] **Omer:** obtain and inspect checkpoint configuration, preprocessing, action normalization, history, and rollout inputs; record hashes and licenses.
- [ ] **Inbal:** inspect the matching simulator’s reset, controller, collision callbacks, and state fields.
- [ ] **Inbal:** implement deterministic reset/warm-up replay and pair-specific contact logging.
- [ ] **Both:** inspect stationary, moving-without-contact, genuine push, grazing-contact, and near-miss fixtures.
- [ ] **Both:** test mock predictions against endpoint, effect, and regret calculations.
- [ ] **Both:** record which published diagnostics are reproducible and applicable; abandon unsupported metric reproductions early.

**Week 1 deliverable:** contribution memo, interface contract, simulator fixtures, and a list of measured versus unresolved feasibility facts.

### Week 2 — complete the pilot and freeze the main-study rules

- [ ] Generate the 120-family development dataset with fixed split membership.
- [ ] Validate duplicate replay, branch-order independence, contact labels, and timestamp alignment.
- [ ] Run a checkpoint inference smoke test without future-state leakage.
- [ ] Fit state readouts and evaluate them on real future frames.
- [ ] Run frozen-model predictions and all inexpensive controls.
- [ ] Generate contact-conditioned errors and action-selection regret.
- [ ] Measure runtime, memory, storage, and the achievable test-set precision.
- [ ] Record a joint proceed/narrow/stop decision using the gates above.
- [ ] Freeze the primary horizon, practical tolerance, exclusions, analysis, and main-data collection plan.

**Week 2 deliverable:** an end-to-end pilot report that establishes whether the project can answer its question. The final course contribution is the validated empirical answer—including a well-supported negative answer—not a promised failure case or new algorithm.
