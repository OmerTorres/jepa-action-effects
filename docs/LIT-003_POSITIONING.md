# LIT-003 — Full-text reading of the seven flagged papers and positioning

Read on 2026-10-01 from arXiv HTML full text. These are author claims; nothing was
reproduced. It completes the "read in full before scoping" item of LIT-002 and corrects
two abstract-level readings there.

## What each paper actually does, and what it means here

| Paper | What it does (full text) | Overlap with this study | Consequence |
| --- | --- | --- | --- |
| **The Intervention Gap** (Vakalis, 2608.29998) | Three-gate audit: capture → real-effect resolvability → propagation, scored against zero effect, persistence and an environment-endpoint oracle. Readouts and thresholds are frozen on calibration families and bootstrapped by family. On DMC Cheetah, three LeWM checkpoints (independently seeded runs the paper trains on cheetah-run; not the released PushT model) show five-step effects worse than no effect: "task-direction rotation with excess gain". Five PreJEPA (DINOv2) predictors instead *attenuate toward zero*. Severity depends on candidate, task, seed and support. In PreJEPA, predicted branches contract and move toward offline-training support, but a joint "attractor" signature was not established. | Closest precedent. Its gate logic matches our floor-relative design (READOUT-002 gates ↔ capture and resolvability; real-frame control ↔ oracle). | We cannot claim the audit design or "LeWM effects are distorted". Our measured direction is **compression**, not excess gain, on a different task and checkpoint. The sign difference across settings is exactly what a support-conditional account must explain. |
| **IMPLY** (Mehta & Baviskar, 2609.12441) | Reads the implied mass and friction of each rollout by simulator inversion, and scores a set of rollouts by anchored disagreement. In V-JEPA 2-AC fine-tuned on the scene, the model is *object-blind* without calibration context. Cosmos-Predict2 predicts the same displacement at every commanded speed. | Speed sweeps and simulator inversion as measurement. | **Correction to LIT-002:** "typical push" refers to ignoring object *identity*, not to action magnitude. Overlap with our magnitude question is small. |
| **PhysWeep** (2609.06207) | Black-box audit of video generators over swept physical parameters, with a deterministic-simulator positive control. Preregistered indices for prior reversion and case-based clamping, selected by leave-one-out, with "neither" allowed. Finds seed-locked wrong constants: neither mechanism. | Method template for adjudicating mechanisms. | SUPPORT-001's decision rule follows this pattern: preregistered account indices, held-out selection, and an explicit "no account established" outcome. |
| **Sensitivity Shaping** (Yu et al., 2606.14585) | Control insensitivity: unsupported actions yield latent predictions that *resemble demonstrated transitions*, which hides OOD signals. Remedy: a control-Jacobian regularizer applied in high-support regions. Tested on obstacle avoidance, manipulation and a real robot. | The same qualitative mechanism as our reversion hypothesis B. | B has a stated precedent, so cite it. The open part is quantitative: dose-response, sign prediction, and a test on a released JEPA checkpoint. |
| **Do-JEPA** (Resani & Shi, 2609.37378) | A paired same-state effect loss. Fine-tuning released LeWM on PushT lowers latent effect error. The physical gain holds on the training intervention distribution but **not** on an outcome-unfiltered holdout. Trained on natural sequences, it improves natural test error. The authors conclude that the gain follows the training intervention distribution. | Branch-based effect supervision of LeWM PushT is taken. Its finding parallels TRAIN-001's "in-distribution adaptation". | Supports TRAIN-002's held-out-dose design and predicts that dose-omission gains will not transfer. Fine-tuning is not our novelty; the support manipulation as a *causal test* is. |
| **ARC-Bench** (Zhang & Li, 2609.05461) | Fixed-candidate audit of JEPA-WM (DINO-WM-style) latent planning costs. The top candidate is wrong on 96.8% of PushT anchors. Reducing replanning frequency collapses success (PushT 36.1% → 12.5%). | Decision-level consequence; a different model family (JEPA-WM, not LeWM). | Frames H3. With privileged agent-matched goals, LeWM's latent cost beats random (−6.0), and JEPA − real-latent regret is not established (+2.1 [−1.3, 5.6]). Say plainly that our goals are privileged and the candidate set small. Open-loop or first-plan evaluation is the consequence route (paper track). |
| **Kang et al. 2023** (2310.00873, ICLR 2024) | For high-dimensional OOD inputs, predictions revert toward the optimal constant solution. The proposed mechanism is smaller OOD feature norms, so outputs are dominated by input-independent terms. | Motivates reversion. | **Correction to LIT-002 framing:** the evidence concerns high-dimensional *input* OOD. Our action is low-dimensional, and ROBUST-001 finds high-dose predicted latents *not* more off-manifold. Transfer to a state-conditioned action pathway is untested (LIT-002 §4.4 patching). |

## Positioning paragraph (draft for the report)

> Recent audits establish that latent world models can capture a task variable yet
> mispropagate action effects. The closest is the Intervention Gap, whose capture-gated
> audit of LeWorldModel on DMC Cheetah finds rotated, *amplified* effects, while a
> DINOv2-based predictor *attenuates* them. Sensitivity Shaping identifies control
> insensitivity, in which unsupported actions produce predictions resembling demonstrated
> transitions. Do-JEPA and related remedies show that effect supervision on LeWM improves
> effects mainly within the training intervention distribution. ARC-Bench shows that
> frequent replanning hides broken candidate ranking in JEPA-WM planners. We contribute a
> dose-controlled audit of the released LeWM PushT checkpoint that these works do not
> provide:
> - exact replay branching with substep contact labels;
> - command magnitude manipulated across designed and archive scenes, confirmed on fresh
>   families;
> - calibration-side measures (magnitude and alignment);
> - five-instrument robustness;
> - fine-tuning interventions that manipulate training support.
>
> The compression we measure has the opposite sign to the Intervention Gap's excess gain.
> Explaining both signs is the test for a support-conditional account.

## Implications for the plan

- SUPPORT-001 should state its decision rule in PhysWeep's form and cite Sensitivity
  Shaping and the Intervention Gap as the qualitative precedents for B.
- TRAIN-002's held-out-dose design is well motivated by Do-JEPA's distribution-following
  gain.
- The paper must cite the Intervention Gap as the closest work. The cross-environment sign
  test (paper track 2) is the strongest available differentiator.
- Re-check arXiv before any submission: four of these seven appeared in the last six weeks.
