# EVAL-001 implementation plan

Status: implemented and verified 2026-09-26. EVAL-001 is DONE. The conventions
remain provisional until PILOT-001 and FREEZE-001.

## Outcome and scope

Deliver a tested, checkpoint-independent evaluator for physical endpoints, paired
effects, contact strata, and finite-candidate action-selection regret. Validate
with analytic mock predictions and a small SIM-001 integration fixture. This
establishes metric correctness, not checkpoint competence or readout validity.

Use RESEARCH_PLAN.md sections 2 (labels), 4 (metrics), and 6 (fixtures), the audited
INTERFACE_CONTRACT.md, and D007. No dependency change is expected. Candidate
generation, readout fitting, pilot collection, final confidence intervals, and
protocol freezing remain in their existing downstream tasks.

## 1. Define inputs and validity

- Add `src/jepa_action_effects/evaluation.py` with typed, NumPy-based inputs and
  structured per-branch, per-pair, per-goal, and per-family outputs. Keep simulator
  adaptation separate so synthetic tests need neither a checkpoint nor a simulator.
- Key all data by family and stable candidate IDs; require matching prediction and
  truth sets, finite 2-D positions, unique IDs, and a shared horizon. Preserve method,
  split/group identity, units, configuration, and exclusion reasons in output.
- Accept current block/agent positions, true endpoints, candidate-phase contact
  flags, and optional timestamped trajectories. Predictions supply decoded block
  endpoints and, when available, agent endpoints and native-step trajectories.
  Missing optional predictions produce explicit unavailable metrics, never zeros.
- Add `src/jepa_action_effects/evaluation_replay.py` to adapt `FamilyResult`.
  Recover the branch point from the final warm-up physics step and endpoints from
  the final candidate step; verify agreement with recorded observations. Use only
  candidate-phase agent–block contact events, excluding warm-up and wall contacts.
- Fail on malformed inputs. Mark incomplete simulator families ineligible for the
  primary evaluation with explicit reasons; never silently shrink the candidate
  set or use a shorter horizon. Prediction failures are reported separately and
  cannot change truth-based case selection or eligibility.

## 2. Implement endpoint and paired-effect metrics

- Proposed pair policy: all unordered pairs of the supplied candidates (15 for
  six candidates), canonically ordered by candidate ID. Keep raw pair records.
- For each pair define true effect `d = q_a - q_b`, predicted effect
  `d_hat = q_hat_a - q_hat_b`, and primary error `norm(d_hat - d)`.
- Report absolute block and agent endpoint errors separately, true/predicted effect
  magnitudes, absolute magnitude error, and direction error in radians. Direction
  is unavailable for negligible true effects or zero predicted vectors; clip cosine
  roundoff before arccos. Do not normalize effect errors by true effect magnitude.
- Contact strata are neither/exactly-one/both branches contacting. Independently
  label true effects negligible (`norm(d) <= delta`) or meaningful (`> delta`).
  Report predicted magnitude for negligible pairs and missed-effect rate among
  meaningful pairs (`norm(d_hat) <= delta`), with denominators. Preserve exact-zero
  fixtures as a distinct diagnostic within the negligible category.
- For no-contact branches, measure maximum true displacement from the branch point
  over physics substeps. Proposed stationary rule: maximum displacement `<= delta`.
  Measure predicted maximum displacement only over available prediction timestamps;
  label its temporal resolution and never claim substep coverage from sparse frames.
  A return-to-start trajectory must remain detectable at its supplied resolution.
- Average pairs within each family and contact stratum, then give contributing
  families equal weight. Report family/pair counts and unavailable strata explicitly.
  Aggregate endpoints over branches within family, not over repeated pair endpoints.

## 3. Implement finite-candidate regret

- Accept explicit goal coordinates and goal IDs shared across methods. PILOT-001
  will own the fixed workspace grid and seeded four-goal sampling; record goals and
  their provenance before inference. EVAL-001 uses hand-checkable fixture goals.
- Select the candidate minimizing predicted Euclidean block endpoint distance;
  compute realized true cost minus the minimum true cost across the identical set.
  Proposed exact-tie rule: canonical candidate ID, independent of execution order.
- Return selected/oracle IDs, costs, and regret for every goal. Include oracle
  regret and the exact expected uniform-random regret (mean candidate cost minus
  minimum cost), avoiding unnecessary Monte Carlo noise.
- Proposed informative-goal rule: true maximum-minus-minimum candidate cost
  `> delta`. Determine it from simulator endpoints only. Report all-goal results
  plus this subset, averaging goals within families before averaging families.
- Check nonnegative regret within numerical roundoff; identical-cost candidates
  yield zero regret. Do not conflate this metric with full PushT success.

## 4. Validate against analytic fixtures

Add `tests/test_evaluation.py` and `tests/test_evaluation_replay.py`:

| Fixture | Required result |
| --- | --- |
| Oracle endpoints | Zero endpoint/effect error and regret |
| Persistence at branch point | Zero predicted effects; effect error equals true magnitude; meaningful effects missed |
| Doubled displacement from branch point | Predicted paired effect `2d`; error `norm(d)` |
| Reversed displacement from branch point | Predicted paired effect `-d`; error `2 norm(d)` and direction pi when defined |
| Candidate-dependent spurious motion | Nonzero predicted effect on identical true endpoints |
| Shared endpoint bias | Zero effect error with nonzero absolute endpoint errors |
| Moving no-contact block | Motion retained independently of contact and paired-effect labels |
| Return-to-start transient | Zero endpoint displacement but nonzero maximum trajectory displacement |

Also test all three contact strata, exactly-at-threshold boundaries, zero vectors,
undefined-metric serialization, mismatched IDs/shapes, NaN/Inf, incomplete families,
stable ties, candidate/goal permutations, unequal family pair/goal counts, empty
subsets, hand-calculated positive regret, and exact random-choice expectation.
The replay adapter test must verify time alignment and wall/warm-up exclusion using
the existing SIM-001 fixtures or a small deterministic replay. Model weights are
not needed. Avoid interpreting synthetic cases as empirical results.

## 5. Audit, document, and hand off

- Add `src/jepa_action_effects/evaluation_audit.py` to run fixed analytic fixtures,
  write `outputs/eval001/report.json`, and record configuration, expected/observed
  values, provenance, and limitations in `docs/manifests/eval001-runtime.json`.
- Use workspace units (`L = 512`). With measured SIM-001 `epsilon_sim = 0`, the
  provisional formula gives `delta = 5.12`; accept these as explicit configuration,
  not hidden constants. The current primary horizon is two native steps. This task
  neither validates physical relevance of delta nor extends replay to four steps.
- During implementation record the pair, tie, boundary, stationary, and informative
  subset conventions in DECISIONS.md as provisional evaluator conventions. They
  must be reviewed in PILOT-001 and frozen in FREEZE-001 before test inference.
- Run `make check`, `make test`, the audit via locked uv/Python 3.10.21, and
  `git diff --check`. Record exact commands and outcomes. Update TASKS.md and append
  logbook evidence; mark DONE only after analytic and replay integration checks pass.

Acceptance: formulas and edge cases pass independently calculated tests; results
are invariant to execution order; family weighting is correct; incomplete and
unavailable data are explicit; audit artifacts are reproducible. PILOT-001 can then
connect simulator truth and decoded predictions without changing metric semantics.
