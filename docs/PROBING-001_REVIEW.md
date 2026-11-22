# PROBING-001 — Alternative probes and a mechanism question

Literature review, 2026-09-28. No experiment executed; no new protocol frozen.

## Recommendation

One additional direction merits a feasibility plan: determine whether the frozen
predictor uses agent and block information differently, and whether that dependence
changes with contact. Start with **amnesic probing using selective concept erasure**.
The first deliverable should establish whether a selective intervention is possible,
before collecting evidence about predictor mechanisms.

This changes the question from accurate coordinate decoding to the influence of
identified information on the existing predictor. It does not repair failed physical
readouts, establish physical competence, or promise a novel method. A new empirical
contribution would require a specific, controlled mechanism finding and fresh
confirmation after exploratory development.

## Search scope and evidence status

Searched arXiv for amnesic probing, LEACE, circuit probing, distributed alignment
search, minimum-description-length probing, and demixed PCA; cross-searched each
with JEPA, V-JEPA, I-JEPA and LeWorldModel. Followed close primary-source leads to
arXiv full text and an official mechanistic-interpretability workshop listing.
This is a targeted review, not a systematic census or proof of absence.

The exact applications of LEACE, circuit probing, MDL probing and demixed PCA to
JEPAs were not established by this search. Nearby JEPA intervention work **was**
found. Therefore “first probing/causal analysis of JEPA” is not a defensible claim.
Paper results below are author-reported literature evidence, not reproductions.

## Methods worth considering

| Method and primary source | What it contributes | Proposed application here | Main limitation / priority |
| --- | --- | --- | --- |
| [Amnesic Probing, Elazar et al., 2020](https://arxiv.org/abs/2006.00995) and [LEACE, Belrose et al., 2023](https://arxiv.org/abs/2306.03819) | Test behavioral dependence after removing linearly accessible concept information; LEACE computes a minimally disruptive linear eraser in closed form. | Separately erase validated agent-position and block-position information in observed-history representations, then measure changes in frozen prediction and unchanged gallery scores. | Recommended first feasibility study. Selectivity must be measured; correlated concepts and off-distribution activations can confound the intervention. |
| [Circuit Probing, Lepori et al., 2023](https://arxiv.org/abs/2311.04354) | Search for subnetworks associated with hypothesized intermediate variables and validate them through targeted ablations. | Investigate whether predictor components compute relative geometry or contact-related distinctions beyond action direction and agent motion. | Higher implementation and optimization cost. Predicting contact labels can reflect shortcuts; masks require held-out behavioral validation. |
| [Distributed Alignment Search, Geiger et al., 2023](https://arxiv.org/abs/2303.02536) | Learn a rotated representation in which interchange interventions correspond to hypothesized causal variables. | Swap a candidate block-state component between matched histories and test a simulator-defined counterfactual prediction. | Stronger but more demanding: valid counterfactual histories must account for velocity, angle and agent state, not just final rendered position. |
| [Minimum Description Length probing, Voita and Titov, 2020](https://arxiv.org/abs/2003.12298) | Measure both predictive quality and the amount of model/data needed to transmit labels, rather than accuracy alone. | Compare accessibility of agent, block, motion and contact-related labels using online coding over whole families. | Useful auxiliary analysis with label-shuffle and simple baselines; does not show that the predictor uses the information. |
| [Demixed PCA, Kobak et al., 2014 preprint](https://arxiv.org/abs/1410.6031) | Separate population variation associated with experimental factors and their interactions. | In controlled factorial renders, separate agent-pose, block-pose and interaction variation across representation stages. | Useful design diagnostic; needs crossed factors. Existing correlated trajectories or block-only galleries are not a balanced agent-by-block factorial experiment. |

These applications are proposals inferred from the methods, not reported results
on our checkpoint. A coarse-position probe could support an erasure diagnostic
without satisfying the original continuous-position gate. It must never be
reported as recovering fine physical measurement.

## Close prior work changes the novelty boundary

- **[MotionJEPA, Karmann et al., 20 September 2026](https://arxiv.org/abs/2609.23881)**
  already investigates selective feature retention, static/dynamic features,
  nonlinear and linear probes, and latent trajectories. Its appendix includes
  agent position, block position and orientation probes on PushT with static
  wood-texture distractors. This is close precedent for an agent/object feature
  imbalance question. Its trained models and distractor setting do not establish
  the explanation for our checkpoint; unsuccessful probes here still do not prove
  information loss. Full text and the state-probing appendix were inspected.
- **[World Model Control by Trajectory Reachability Metrics, Li et al., 2026,
  v2](https://arxiv.org/abs/2605.22164)** studies frozen LeWM, latent metric failure,
  state-probe subspaces and controls. Its TwoRoom subspace audit separates a
  position-probe rowspace from residual directions when scoring predicted
  terminal latents; it also includes PushT. Merely projecting latent distances
  onto a physical subspace is therefore close to existing work. Intervening in
  history/predictor computation would ask a different question, but is not
  automatically novel. The full-text diagnostic protocol was inspected.
- **[Causal State Variables in V-JEPA 2 Latents: Discovery, Intervention, and
  Portability, Bouwens](https://mechinterpworkshop.com/posters/virtual/)** is listed
  by the workshop with probing, PCA, motion subspaces and causal interventions on
  a frozen V-JEPA 2 encoder. The official abstract establishes overlap with a
  generic JEPA causal-subspace proposal. The linked OpenReview PDF returned a
  browser-verification barrier; this review does not claim a full methodological
  assessment of that paper or independently endorse its causal claims.

These are additions to the prior project reviews. Historical reports remain
unchanged; earlier novelty assessments should be read with this correction.

## What a selective-erasure feasibility study would need

The proposed question is: **does removing independently validated block-state
information change action-conditioned predictions more around contact, while
removing agent-state information explains more of global latent correspondence?**
Both parts are hypotheses. A difference in ablation magnitude alone is insufficient
because concepts can differ in recoverability, removed rank and intervention size.

1. Fit concept directions only on real development representations. Validate
   recoverability and selective removal on disjoint families, including preservation
   of the other concept. Start from observed state labels; future contact labels
   may stratify outcomes but must never enter predictor inputs.
2. Specify categorical spatial concepts or continuous covariance removal explicitly.
   LEACE's linear-erasure guarantee does not imply removal of all nonlinear
   information, and an empirical fit does not guarantee held-out erasure. The
   previous fine-position readout failures make this a substantive feasibility gate.
3. Apply the intervention before prediction, at a prespecified observed-history
   representation or internal site. Keep real targets, candidate actions, and
   scoring geometry unchanged. Erasing output embeddings and changing the metric
   would answer a different question.
4. Include identity, shuffled-concept and random-direction controls; assess rank,
   perturbation magnitude and collateral concept damage. Keep family weighting and
   contact/motion distinctions. Do not search layers, concepts or erasure strengths
   until a favorable effect appears.
5. Report global correspondence and the existing conditional gallery diagnostic.
   A selective effect could indicate dependence on the intervened representation;
   it cannot alone establish causal contact reasoning or continuous physical
   accuracy. Failure to construct a selective intervention blocks that inference.
6. Develop on already-open evidence only. The 60 READOUT confirmation families are
   now open and cannot confirm a revised hypothesis. New reserved families would
   require a separately scoped collection and execution plan.

These controls follow the concerns raised in [Improving Causal Interventions in
Amnesic Probing with Mean Projection or LEACE, Dobrzeniecka et al.,
2025](https://arxiv.org/abs/2506.11673): behavioral damage must exceed appropriate
random-removal damage, and removal itself must be checked. An exact add-back of
everything removed is only an implementation check, not a concept-specific rescue.

For learned subspace swaps, [Makelov et al., 2023](https://arxiv.org/abs/2311.17030)
demonstrate that a successful intervention can exploit a dormant pathway rather
than identify the mechanism normally used. Simulator-backed counterfactuals and
convergent controls would therefore be necessary for the more ambitious DAS path.

## Local feasibility and stopping boundary

The current inference module exposes real projected encodings and prediction from
observed history/actions. The projected space is 192-dimensional. A linear eraser
is a small matrix operation and does not require world-model training. This is an
engineering feasibility inference, not a runtime benchmark.

Circuit probing/DAS would require a separate gradient-enabled diagnostic path:
the present `encode_real`, `predict_future` and manual rollout use inference mode.
All original model parameters must remain frozen; fitting masks or rotations is
additional diagnostic training whose cost has not been measured. Raw CLS results
would not substitute for an intervention in the predictor's projected space.

The sensible next artifact is a bounded selective-erasure feasibility plan with a
clear stop if selective removal fails. Circuit discovery should follow only if a
specific mechanism question survives this check. MDL and demixed PCA remain
supporting options, not an automatic experiment queue. The historical
training-simulator identity uncertainty and failed physical-readout gates remain.

## Verification of this review

Documentation-only changes. Primary sources were searched/opened using the web
tool; the local task board, recent logbook, research-plan contribution/readout
sections, study skill and inference API were inspected. No new simulator runs,
encodings, predictor evaluations, readout fits or confirmation accesses occurred.
`git diff --check` passed. Python tests were not rerun for this literature memo.
