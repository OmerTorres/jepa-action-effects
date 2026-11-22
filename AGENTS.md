# Working in this project

Read `docs/TASKS.md` and the latest `docs/LOGBOOK.md` entry at the start of substantive
work. Consult relevant sections of `docs/RESEARCH_PLAN.md`; do not reload the full
plan for routine changes. User instructions govern scope and priorities.

## Workflow and handoff

- Use `uv` and Python 3.10.21 in `.venv`; `make setup` restores the locked environment.
- Keep reusable code in `src/jepa_action_effects/`. Notebooks are for exploration;
  move reusable logic into modules.
- Run `make check` after Python/environment changes and relevant pytest tests when
  scientific logic exists. Do not report empty test collection as a passing suite.
- Update `docs/TASKS.md` and append to `docs/LOGBOOK.md` after substantive work:
  task IDs, changes, exact commands/results, evidence, limitations, next action.
  Preserve historical entries; append corrections instead of rewriting evidence.
- Record dependency/protocol choices in `docs/DECISIONS.md`. Do not silently
  change a frozen protocol. Keep generated data, weights, and secrets out of Git.
- Use `.agents/skills/jepa-study/SKILL.md` for simulator/model integration,
  experiment execution, measurement, or interpretation. Ordinary edits need only
  these instructions and the relevant document.

## Scientific invariants

- Essential scope: one exact simulator/checkpoint pair, a frozen model, one primary
  horizon, physical-effect evaluation, readout controls, finite-candidate regret.
- Simulator/action semantics, preprocessing, history, stride, and model interface
  must be verified together. A shared PushT name is not evidence of compatibility.
- Never give the predictor future observations or future proprioception.
- Branch via seeded reset plus identical warm-up replay until full-state cloning
  passes equivalence checks. Record agent–block contacts at physics-substep level.
- Keep contact and motion labels separate; no-contact blocks can already be moving.
- Keep related initial-state families in one split. Fit readouts on real encoded
  development observations only; tune on validation, never locked test labels.
- Select cases from simulator outcomes, never model errors. Freeze rules before
  test inference; later changes are exploratory and need newly reserved families.
- Aggregate and bootstrap at family level. Branches are not independent samples.
- Distinguish measured facts, hypotheses, mock tests, and unverified source claims.
  Readout failure or a poor baseline makes physical failure claims inconclusive.

No full world-model training is required. A well-measured null result is success.
