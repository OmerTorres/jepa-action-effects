# NOVELTY-002 — Contribution reassessment and object-focused measurement candidate

Reviewed 2026-09-28 after LATENT-001. This is a source review and renderer
feasibility assessment, **not** a newly executed model evaluation or validated
novelty claim.
LIT-001_REVIEW.md remains the record of the earlier literature review.

## What the closest work already covers

The newly identified [AD-WM v1, dated 2026-09-24](https://arxiv.org/html/2609.30264)
is especially close: it uses LeWM as a matched baseline, includes PushT among
five simulated tasks, distinguishes factual prediction from action comparison,
and evaluates whole-bank ranking plus realized elite regret. Its equations 6–8
define Spearman cost ranking, normalized best-in-elite regret, and elite-mean
regret. Its controlled diagnostic emphasis is Cube, and its main contribution
is action-recovery training. LATENT-001's positive action correspondence and
improvement over persistence should therefore be reported as a checkpoint
observation, not a newly discovered evaluation principle or evidence that
its predictor is suitable for physical planning.

[WorldSimProbe v1](https://arxiv.org/html/2608.09298v1) already evaluates
interaction grounding and dynamics from generated videos;
[WAMProbe](https://github.com/myheart521/WAMProbe) includes paired interventions,
contact-aware toys and permutation/regret diagnostics;
[Delta-JEPA v1](https://arxiv.org/html/2606.31232v1) studies latent action
responses and PushT; and [C-JEPA](https://arxiv.org/abs/2602.11389) studies
object-level latent interventions through training. Their existence rules out
claims that action sensitivity, paired counterfactuals, object interventions,
or contact-aware evaluation are new. This targeted primary-source search is
not an exhaustive novelty review or an independent reproduction.

## Why the current latent result is insufficient

LATENT-001's global projected CLS can reflect the agent, which moved 158.97
workspace units on average. A simulator-only census of all 60 tuning families,
six candidates each, found **zero** candidate pairs with block endpoint
separation above the existing `delta=5.12` and agent endpoint separation at
most 5.12, 10.24, or 20.48. The existing branches cannot cleanly identify a
block-specific latent correspondence result by simple agent-endpoint matching.
This census used saved state labels only; it did not select by model errors or
perform new model inference. It is a feasibility count, not a pilot result.

## Candidate: fixed-agent rendered block-pose gallery

For each real candidate endpoint, keep its **actual rendered agent pose, goal,
background, and shape** fixed. Use the simulator renderer to create a small
gallery of endpoint images that differ only in block pose: the realized block
pose and admissible alternative block poses from other branches or a
predeclared geometry sampler. Encode each gallery image with the frozen real
encoder, then compare the candidate's predicted terminal projected CLS with
the true-pose template and alternatives. Future agent state and block poses
enter the *evaluation gallery only*; the predictor still sees observed history
and proposed controls. This is a privileged diagnostic, not a deployable
planner or a state readout.

The primary score would be the within-candidate margin between the closest
admissible wrong-block template and the true-block template, with top-1
true-pose retrieval as a secondary metric. Family-level aggregation and
intervals would retain candidate correlation. Compare frozen prediction with
real-future embedding as a measurement control, persistence, and action
permutation. A block-insensitive prediction should have little reliable
preference among templates sharing the same agent pose. Report representation
margin on real frames separately from predictor margin; the latter is
interpretable only if the former is adequate.

The first gate is **renderer identity**: with the true endpoint state and exact
recorded reset/goal recipe, an unstepped render must reproduce the saved RGB
frame bit-for-bit (or meet a predeclared measured pixel tolerance). The pinned
simulator's `_set_state` advances physics by one substep, so that method cannot
be assumed to preserve the endpoint exactly; inspect/use a render-only pose
setter and test it before building a gallery. Validate the transformed T-block
geometry and reject alternative poses that collide with the held agent or
leave the workspace. Do not choose alternatives using model scores. If true
RGB identity, valid alternatives, or real-frame discriminability fail, stop
without scoring the predictor.

An initial renderer check used the twelve LATENT-001 verification families
(six challenge and six ordinary), all six candidates each. Directly assigning
saved body poses after reset did **not** reproduce all saved frames. Fully
replaying each branch reproduced all 72 terminal RGB frames bit-for-bit.
From each replayed endpoint, changing the block pose and then restoring it
also reproduced all 72 frames exactly, provided the block **angle is assigned
before position**, followed by `space.reindex_shapes_for_body(block)`. The
block's offset center of mass makes the reverse assignment order change its
reported position. This establishes a workable render-only manipulation for
these replayed endpoints, not gallery validity or latent discriminability.
Using the next candidate's block pose as a preliminary alternative changed
the image in 32/36 challenge endpoints and 0/36 ordinary endpoints. The
ordinary branches usually share a visually identical block pose, so a usable
ordinary gallery will need prespecified, geometry-valid alternatives rather
than simply borrowing another branch's endpoint. Neither collision validity
nor representation margin has yet been measured.

This measurement is promising because the *rendered comparison* controls
agent pose exactly and avoids fitting a physical readout. It has real limits:
synthetic endpoint combinations may be outside the checkpoint's training
distribution; negative templates may be perceptually easier than naturally
reachable alternatives; and choosing among a finite set of rendered poses is
not continuous block-error estimation or contact-dynamics proof. These limits
must be measured and reported. An independent reservation would be needed
after protocol selection; the 60 READOUT confirmation families must stay
closed until a new measurement protocol and its gates are sealed.

## Decision gate before a full study

The next bounded protocol should run a small
**simulator-and-encoder-only feasibility audit** on prespecified development
families. It must establish renderer identity, collision-valid pose galleries,
adequate real-frame block-pose margins, coverage across cohorts/contact
regimes, and runtime before any new predictor scoring or confirmation opening.
Set numerical real-frame margin and coverage gates before inspecting predictor
results. If this feasibility audit fails, report the measured limitation and
reconsider the research question rather than cycling through additional
readouts or thresholds. If it passes, register the exact comparison and use
newly reserved families for a distinct exploratory confirmation.

The potential contribution is a carefully controlled **object-specific audit
of one frozen latent world model**, with a method note about readout-free
rendered comparisons. Novelty remains provisional until the gallery is
validated and its distinction from existing evaluations survives a broader
source review. Do not advertise it as a new general benchmark or as proof of
physics competence.
