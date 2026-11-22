# LIT-001 — Source review and contribution boundary

Reviewed 2026-09-27. This is a primary-source review, not an independent
reproduction of any external result. The original RESEARCH_PLAN remains a
historical proposal; this memo supersedes its unverified literature status.

## Closest-work matrix

| Source and inspected version | Verified scope | Consequence for this project |
| --- | --- | --- |
| [WorldSimProbe, v1](https://arxiv.org/html/2608.09298v1), §4 | Simulator-referenced action calibration, flow correspondence, no-contact grounding, and interaction primitives | Contact-aware evaluation and action sensitivity versus fidelity are established motivations. Its generated-video measurements are unavailable for our latent-only predictor. |
| [WAMProbe repository](https://github.com/myheart521/WAMProbe), README accessed 2026-09-27 | Paired contexts, contact-aware toys, permutation diagnostics, direction, endpoint errors, ranking and regret; analytic counterexamples distinguish action dependence from correctness | Our evaluator structure and permutation idea are not novelty claims. Repository evidence is distinct from a peer-reviewed benchmark or independent real-model reproduction. |
| [Delta-JEPA, v1](https://arxiv.org/html/2606.31232v1), action-sensitive dynamics section, Fig. 6 | Fixed-history action interventions and latent response visualization; action reconstruction from latent differences is a training method | PushT, latent differences, and action responsiveness already have direct precedents. Our proposed correspondence score adds real branch references to our checkpoint audit; originality is not established. |
| [CoCo, v1](https://arxiv.org/html/2608.04653v1), action-controllability section and conclusion | Reference/inverse/zero-action diagnostics and counterfactual training; acknowledges difficulties with irreversible contacts | Do not assume opposite controls undo contact, or claim that this limitation is new. |
| [UWM-JEPA author repository](https://github.com/santoshkumarradha/uwm-jepa/blob/main/README.md), accessed 2026-09-27 | Hidden-velocity prediction compared under correct, absent, shuffled, and wrong actions | Action ablations in latent JEPA models are established. This task and its state-label head differ from our proposed matching of encoded futures. Repository results were not rerun. |

The GitHub pages above are mutable access-date references, not pinned replication
dependencies. No external implementation was installed or executed. This review
is sufficient to reject broad novelty claims; it is not an exhaustive novelty
search or evidence of publication acceptance.

## Exact definitions and explicit omissions

**WorldSimProbe T1:** with temporally aligned full-video MSE D, form
`r = D(pred0,predSmall)/D(pred0,predLarge)` and the analogous simulator ratio
`r*`. Simulator-only selection requires `0 < D(real0,realSmall) <
D(real0,realLarge)` and successful valid trajectories. Its score is
`100 * mean(clip(r/r* if r <= r* else (1-r)/(1-r*), 0, 1))`;
degenerate scaling scores zero. This requires rendered model videos and a
particular triplet construction. Substituting latent distances would create
a different diagnostic. T4 additionally assumes the evaluated object should
remain stationary; no-contact alone is insufficient for our moving branches.
We omit these published scores. [Definition](https://arxiv.org/html/2608.09298v1#S4)

**Delta-JEPA Fig. 6:** for 512 fixed Two-Room histories, vary the final action
and visualize `predicted_z(a) - predicted_z(0)` with PCA and action-wise means.
This measures predicted response, without a simulator-future error in that
definition. We can use response magnitude as a study-specific secondary
diagnostic, but do not label a six-branch PushT analysis a reproduction of
Fig. 6. [Definition](https://arxiv.org/html/2606.31232v1)

**CoCo ARC/DE:** both the HTML and the nine-page v1 PDF refer to Appendix A
for definitions; the retrieved documents did not supply that appendix.
Exact implementation, normalization, and edge cases remain unverified.
Omit ARC/DE rather than reconstructing them from prose. This is an explicit
review outcome, not a completed metric reproduction.
[PDF](https://arxiv.org/pdf/2608.04653v1)

**WAMProbe:** README-level metric scope is verified. The linked core metric
card failed to load in this review. Exact permutation statistic and tie rules
were therefore not verified; do not borrow the published metric name for our
own formula. **UWM-JEPA:** the documented action conditions establish precedent;
exact corruption distributions and training protocol were not audited.

## Checkpoint-native evaluation review

The [LeWM paper v1](https://arxiv.org/html/2603.19312v1), §5 and Appendix F,
reports linear/MLP physical probes using MSE and Pearson correlation. PushT
block-location results are linear MSE 0.029, r 0.986, and MLP MSE 0.001,
r 0.999. These numbers alone do not establish workspace-unit mean/p95 error
under our challenge distribution; the inspected probing description does not
fully specify the conversion and fitting recipe needed for that comparison.
Do not convert those MSEs into our gate or claim a failed replication.
Appendix F.1 uses PushT goals 25 controls ahead and a 50-control evaluation
budget. Its violation-of-expectation experiment compares perturbations
including teleportation and color changes; that is a different question from
local candidate-action correspondence.

The locally cached, pinned source commit
`ca231f9f9d9ab041034b6d05e90b6e04bd6cff82` was inspected directly:

- [eval.py](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/eval.py)
  samples starts/goals from a dataset, fits action scaling from archive rows,
  applies ImageNet normalization, and sets the loaded model to evaluation mode
  with gradients disabled. Its model path uses CUDA; our validated runtime is CPU.
- [PushT configuration](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/config/eval/pusht.yaml)
  sets seed 42, 50 episodes, goal offset 25, budget 50, planning horizon 5,
  action block 5, and receding horizon 5. The default policy is random until
  overridden. Its world history size is 1; our audited three-frame branch
  history is not a literal reproduction of that evaluation configuration.
- [jepa.py](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/jepa.py)
  projects encoder CLS tokens, predicts in that space, and scores plans by
  the sum of squared terminal embedding differences from the encoded goal.
  Our proposed per-coordinate MSE differs only by the fixed dimension factor.

The [pinned checkpoint card](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/README.md)
does not replace the local identity and compatibility audit in
INTERFACE_CONTRACT.md. Historical training-simulator identity remains unproven.
We have not rerun published closed-loop planning or published probes.

## Contribution recommendation

Retain the completed work as a reproducible measurement-feasibility result:
three bounded readout routes did not meet the predeclared physical precision.
That supports a limitation of this measurement approach, not missing object
information or predictor physics failure.

A defensible course-scale extension is a **checkpoint-specific exploratory
audit of action-to-future correspondence**, with explicit persistence and
action-permutation controls and simulator-defined strata. It is an empirical
application of established ideas, not a new benchmark, general JEPA claim, or
replacement proof of physical competence. See LATENT-001_PLAN.md. If this scope
is insufficient for the course's contribution requirement, use the feasibility
report and explicitly revise the project direction; further probe search is
not justified by this review.
