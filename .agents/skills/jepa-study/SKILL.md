---
name: jepa-study
description: Implement, run, or interpret experiments in the JEPA contact-dependent action-effects study, preserving replay validity, readout controls, family splits, and evidence-backed claims.
---

# JEPA study workflow

Use this skill for scientific work in `jepa-action-effects`. Resolve the project
root from its AGENTS.md and pyproject.toml. Read the current task board and latest
logbook entry; consult only the relevant research-plan sections. Paths below are
relative to the project root.

## Before simulator or checkpoint integration

Complete the relevant fields in `docs/INTERFACE_CONTRACT.md` from immutable source
revisions and configuration. A PushT name does not establish matching control,
rendering, or prediction stride. Keep unknowns explicit. Check that predictor
rollouts never receive ground-truth future observations or proprioception.

For branches, use seeded reset and identical full warm-up replay. Validate state,
velocities, orientation, contact history, and branch-order independence before
trusting labels. Log agent–block collisions at physics-substep resolution, not
aggregate contacts that include walls. Contact and motion are distinct labels.

## Before experiments or analysis

Consult `docs/DECISIONS.md` for freeze status. Six branches and four goals are
correlated observations within each initial-state family. Keep related warm-ups
and perturbations together in one split; average and bootstrap by family.
Generate/select cases using simulator outcomes, never model errors.

Fit state readouts on real encoded development frames only. Evaluate real-future
frame readout quality on the matching distribution before attributing predicted
latent errors to physical reasoning. Apply the research plan's mean and tail
readout gates; do not subtract decoder error and call the remainder ground truth.

Validate effect and regret calculations with oracle, persistence, doubled-effect,
reversed-effect, and spurious-motion fixtures. Preserve absolute endpoint errors
alongside paired differences. Do not normalize by nearly zero effects or assign
an arbitrary direction to zero predicted vectors.

For a real run, record revision/dirty state, dependency and checkpoint hashes,
configurations, seeds, split manifest, exclusions, timing, and artifact paths.
Use `docs/DEVELOPMENT.md` for provenance fields. Cache embeddings where useful.

## Reporting and handoff

Distinguish simulator checks, mock predictor results, trained-checkpoint evidence,
and unverified literature claims. Reproduce a published diagnostic only from its
exact definition and applicable inputs. A contact association is not by itself
causal evidence; one checkpoint does not establish training-seed variability.

Report null or inconclusive findings when supported. Post-test changes are
exploratory and require new reserved families for confirmation. Do not expand
into new world-model training unless it is part of the current authorized scope.

Update `docs/TASKS.md`, append commands/results and the next action to
`docs/LOGBOOK.md`, and record material protocol changes in `docs/DECISIONS.md`.
