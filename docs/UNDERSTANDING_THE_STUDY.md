# Understanding and defending this study

This guide separates implementation facts, documented design choices and things we did
not establish. It accompanies the report; it is not an extra submission section.

## The question in plain language

We have a pretrained model that predicts what a simulated scene will look like after
the robot acts. The scene contains a disk-shaped robot and a T-shaped block. We ask:

> When we give the robot larger movement commands, does the model still predict the
> block's actual movement? If it increasingly under-predicts that movement, can targeted
> training reduce the error?

We are not testing the law "twice the command should produce twice the block motion."
For every command, the simulator supplies the actual outcome. That outcome is the
reference against which we measure the model's prediction.

## What the robot is actually told to do

A simulator control is a vector `a = (x, y)`. It has both a direction and a magnitude
`m = sqrt(x*x + y*y)`. It is not a force applied to the block and it does not directly
specify the distance travelled in one second.

At the beginning of each 0.1-second control interval the simulator sets:

```
target position = current robot position + 100 * a
```

A proportional-derivative controller moves the robot toward that target during the
interval. For example, `a = (0.3, 0)` places a target 30 simulator units to the right of
the robot's position at that moment. It does not teleport the robot there. At the next
control interval, the target is recomputed relative to the robot's new position.

In the constant-command tests we send the same vector ten times. The robot therefore
keeps being asked to move in the same direction, toward a repeatedly updated target.
"Push strength" in the figures is shorthand for this **command magnitude**, not force.
The observed approximately `400*m` robot displacement over a second is a consequence of
this controller/configuration and the illustrated trajectory, not the action definition
or a universal physical law.

Source: installed pinned simulator `stable_worldmodel/envs/pusht/env.py`, initialization
and `step`; recorded in [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md).

## Your contact-point and contact-time objection is correct

Block displacement depends on the start configuration, approach, contact geometry,
contact duration, rotation and simulator dynamics. Increasing command magnitude can
change the contact trajectory. A faster robot can pass the block sooner, rather than
continue pushing it farther. This simulator strongly damps the block; it does not
represent an object that coasts freely after a strike.

We replay the identical start state and keep the commanded direction fixed within each
constant-command family. We vary command magnitude and let the simulator determine the
contact and motion that follow. We do not hold realized contact time or realized contact
point fixed. They are consequences of the intervention, not independently controlled
variables in this experiment. Therefore we cannot claim to isolate a direct effect of
contact force or explain the error solely by contact duration.

In the designed scenes, larger commands eventually give nearly the same block endpoint:
the robot passes the block. The model can nevertheless predict progressively less motion.
Demo scenes are more varied: similar median displacement does not imply every family's
outcome is unchanged. The report now distinguishes those cases.

## Five steps, ten steps and one second are different quantities

| Quantity | Duration | Why it is used |
| --- | --- | --- |
| One physics integration step | 0.01 s | Pinned simulator setting |
| One control command | 0.1 s, containing 10 physics steps | Simulator control frequency, 10 Hz |
| One LeWM prediction step | 5 control commands = 0.5 s | Inherited model action grouping and image sampling |
| Our future prediction horizon | 2 LeWM steps = 10 controls = 1 s | Study choice, retained after a simulator-only pilot |

The model gets three past images at times -1.0, -0.5 and 0 seconds, plus the commands
connecting them. Given the proposed future commands, it predicts a latent at +0.5 s,
then feeds that prediction back to predict +1.0 s. It does not get the real images from
those future times as inputs.

There is another potentially confusing "10": five two-dimensional controls flatten
to **10 input numbers per model action block**. That input width is not a ten-control
window. The two future blocks together contain ten controls.

Changing five-control grouping to an arbitrary number while keeping this checkpoint
would change the input representation/time interpretation it was trained to use.
Rolling out more whole model steps is possible, but would be a different evaluation
horizon. The authors of LeWM chose their training interface; this study did not prove
that their grouping is optimal.

Evidence: checkpoint `config.json` (`num_frames=3`, action encoder `input_dim=10`),
pinned upstream evaluation config (`action_block: 5`), `model_inference._model_actions`,
and [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md).

## Why did we choose one second? Was it best?

**We did not show it was best.** The original research plan chose a bounded two-step
diagnostic and specified a feasibility rule before model-performance inspection:
extend to four model steps if fewer than 20% of pilot families show a meaningful
action-dependent block effect. "Meaningful" here meant that at least two candidate
actions from the same start produced block endpoints more than 5.12 units apart.

The simulator-only pilot found this in **88/120 families (73.3%)**, so the extension
rule was not triggered. This justified keeping the horizon as usable for the diagnostic;
it did not optimize prediction accuracy or prove that one second is ideal for planning.
The 20% trigger and 5.12-unit threshold were study design choices too, not laws of physics.

Two prediction steps also exercise one autoregressive feedback step. Keeping the horizon
short makes interpretation more local than a long rollout, but this is an advantage of
the design, not experimental proof of its superiority. We did not run a horizon ablation.
Our results do not establish what happens over a full task or over longer rollouts.

Sources: [RESEARCH_PLAN.md](RESEARCH_PLAN.md), section 3;
[DECISIONS.md](DECISIONS.md), D009; [pilot001-runtime.json](manifests/pilot001-runtime.json),
`collection`; `pilot_collection._meaningful_family` and horizon selection.
The original pilot's physical readout failed; its simulator-only horizon check does not
turn those early model measurements into valid physical evidence.

## What is gain? Gain of what?

Here **motion gain** compares predicted displacement with true displacement along the
true movement direction. It is unrelated to the improvement obtained by fine-tuning.

For a single illustrative example:

- True block displacement: 10 units right; predicted displacement: 6 units right.
  Motion gain is 0.6.
- Predicting 10 units right gives 1; predicting no movement gives 0.
- Predicting 5 units left gives -0.5: motion in the opposite direction.
- Predicting 6 units right and 8 units up still gives 0.6. Sideways error is not captured
  by this score, so we also examine alignment and endpoint error.

Across a cohort the formula is `b = sum(p · t) / sum(t · t)`, where `p` and `t` are the
predicted and true displacement vectors. This is a through-origin projection slope,
equivalently a true-displacement-squared-weighted average of individual gains where
true displacement is nonzero. It is **not** an unweighted average percentage accuracy,
not a derivative of block movement with respect to command magnitude, and not a force
transmission coefficient. A value of one does not mean all predictions are correct.
For example, equal-size true motions with gains 0.5 and 1.5 aggregate to one.

The cohort/eligibility rules differ between studies and are disclosed in the methods.
Study uncertainty is resampled by start-state family, not by treating each alternative
action as independent.

## How can we measure position if LeWM does not predict images?

LeWM predicts latent feature vectors. We train a small readout network on **real encoded
images with known simulator positions** and then freeze it. We apply it to:

1. The predicted future latent: this produces the position we evaluate.
2. The latent of the real future image: this checks how much error the measuring method
   introduces on the actual scene distribution.

We also use alternative readouts. Their agreement supports the trend but does not prove
that readouts behave correctly on predicted latents. The readouts share an encoder and
training frame bank. A common bias specific to predicted latents remains possible.

`R = predicted motion gain / real-frame motion gain` normalizes by the real-frame
reference. It does not mathematically remove every measurement error.

## What did we actually train and establish?

The released visual encoder stays fixed. We trained the position readout and fine-tuned
11,740,334 parameters in the predictor, action encoder and prediction projector across
four training conditions with three seeds each. The future images supply targets for
training, never inputs for predicting those same future states.

The strongest adaptation result is lower block endpoint error on fresh push families:
26.84 to 16.21 units at command magnitude 0.60. Training equally long on ordinary demo
clips did not establish the same improvement. Giving that magnitude zero weight in the
training loss retained most of the improvement, but its smaller effective batch size prevents
attributing the remaining deficit solely to missing that magnitude. Ordinary-clip
performance gets slightly worse after targeted training.

The targeted dataset is specialized: 600 simulator push families, rather than ordinary
demonstrations with their recorded actions. A short run builds on the expensive released
pretraining; it does not replace it. At 0.45 and 0.60 the actual block endpoints are almost
identical in the designed scenes, so learning from 0.45 can plausibly help at 0.60.
That explanation is plausible, not a demonstrated learning mechanism.

Do not call 0.60 completely unseen: it informed family admission and the earlier
learning-rate selection, and appears in logged validation. It is excluded from the
production training loss. The [leakage audit](TRAIN002_LEAKAGE_AUDIT_2026-10-14.md)
found no direct contamination in the checks performed, including an actual-model
test showing that changing the zero-weight branch leaves every gradient unchanged.

This supports a bounded claim about a particular checkpoint and evaluation setup. It
does not prove why the pretrained model acquired the distortion, that all JEPAs do this,
or that a practical planner necessarily suffers. The planning comparison did not
establish a penalty.

## Preparing for questions without memorizing a script

Before calling the presentation ready, each presenter should be able to work through
one example in their own words:

1. Describe what command `(0.3, 0)` does during a control interval and why it is not force.
2. Draw the three observed frames and two predicted frames on the time axis.
3. Explain why changing command magnitude can change contact time without increasing
   the block's final displacement.
4. Calculate gain for one correct-direction prediction and one sideways-error example.
5. Explain what the real-frame readout control checks and what it cannot establish.
6. State what was trained, the main improvement, the ordinary-data tradeoff, and one
   conclusion the experiment cannot support.

The repository's tests and saved evidence address implementation consistency. Being able
to answer these questions addresses a separate requirement: understanding and defending
the study. The earlier submission-readiness assessment did not establish that requirement.
