# Review of claude/youthful-clarke-akm28u

Reviewed by Codex at Omer's request, 2026-09-29. Base: `59cfa746272519aa7ab2e73d91b5b16861085a4f`;
branch tip: `181ca5586c9bceb25146ada24e7333cb9167b30e`. Both tips were verified
against GitHub with `git ls-remote`. Scope: 13 commits, 49 changed files, including
the experimental code, tests, protocols, result narratives and saved local outputs.

## Assessment

This is useful, substantial research work. It recovers a workable readout, tests
plausible explanations on fresh families, and records failed development checks
instead of silently treating them as successful experiments. The primary numerical
results reproduce. I found no evidence of future observations entering the predictor,
and no mismatch in the completed runs' audited input hashes.

I would revise the reporting and strengthen the experiment runner before relying on
it for another confirmation. Passing tests do not establish that the seals protect
the inputs, and some interpretation goes beyond what the measurements isolate.
The issues below do not, by themselves, overturn the observed magnitude trend.

## Verification performed

- `make check`: passed Ruff, formatting and doctor under Python 3.10.21.
- `make test`: **134 passed**, no skips, 19 Gymnasium input-casting warnings,
  13.98 seconds. Artifacts were available locally.
- Recomputed PHYS-002, SHIFT-002 and H3-PLAN-002 confirmation summaries from their
  saved per-family JSON records using the original bootstrap seeds and 2,000 draws.
  Maximum numerical difference from the saved summaries: **0.0 for all three**.
- Compared current source hashes and the config/audit/development/collection/probe
  hashes recorded in each completed experiment's seal: all checked values matched.
  Development and confirmation model fingerprints also matched.
- Verified **660 family artifact hashes**: PHYS-002 reserved 150/150,
  SHIFT-002 development plus reserved 360/360, H3-PLAN-002 reserved 150/150.
- Ran targeted counterexamples in temporary directories. These used mock predictions
  and mock artifacts; they are software/measurement checks, not checkpoint evidence.
- Recomputed true absolute PHYS-002 endpoint errors from saved latents, replay states
  and the frozen probe; checked additional SHIFT-002 readout slopes by dose.

The review did not retrain the 60k-frame readout or repeat every collection and
checkpoint inference stage. Recomputing a summary verifies aggregation, not every
upstream measurement. Existing result artifacts were read without modification.

## Actionable findings

### 1. Confirmation does not enforce the complete frozen input set

**Priority: fix before another experimental run.** Locations:
`planning002.py:277–300`, `phys002.py:337–351`, `shift002.py:445–461`, and their
`freeze`/cache helpers; the earlier planning and SHIFT runners share the pattern.

The seal records hashes of config, collection manifest and probe selection, but
confirmation largely verifies source hashes and the opening marker. Checking a
JSON file against its own sidecar is insufficient: a producing stage can rewrite
both the JSON and its sidecar. Confirmation must compare it against the hash in
the original experiment seal. PHYS/SHIFT also load the current probe without
checking its selection against the sealed selection. The checkpoint loader does
independently pin weight/config bytes, which is good; the action-scaler fingerprint
and the full runtime fingerprint still need comparison with the frozen run.

Reproduction: in an isolated mock H3 run, changed and re-sealed the config and
reserved goal from `[100,100]` to `[400,400]` after freezing. `confirm()` accepted
the changed goal and returned complete. Cached scoring also accepted an old record
whose family file did not even exist. `_score_all()` returns the cache before
checking its recorded `input_sha256`; SHIFT's scorer similarly returns early.

Additionally, `freeze()` records the current sources without checking that they
match the audit/development sources. Editing code after development but before
the first seal can therefore bless development results from older code.

**Fix:** one shared validator that checks the audit-to-development-to-seal chain,
all frozen input hashes, family bytes, probe selection and inference fingerprint,
before opening or resuming confirmation. Bind cached rows to the same fingerprint
and input hashes. Add mutation/resume tests. The existing completed artifacts
passed my explicit checks, so this is an enforcement defect, not evidence that
these completed runs changed inputs.

### 2. SHIFT-002's readout gate can miss dose-dependent measurement bias

**Priority: strengthen the next protocol; qualify the present claim.**
`shift002_metrics.py:126–130, 218–227` pools real-frame slopes across magnitudes
within each state × structure cell, while the central contrast compares magnitudes.
Opposite readout biases at different doses can cancel in the pooled check.

Counterexample: perfect predicted physical displacements, but real-frame decoded
displacements with gains 0.8 at low dose and 1.1 at high dose. With intermediate
gains, every current manipulation check passes and both magnitude effects are
reported supported, with dM = 0.341. There is no physical prediction error in this
fixture. The implementation follows D023; the problem is the gate's design.

**Actual-data follow-up, exploratory:** real-frame slopes pooled over structure
were 1.045/0.995/1.004 across challenge doses 0.30/0.45/0.60 and
0.936/0.959/0.965 across archive scales ×1/×1.5/×2. All six CIs lie inside the
existing 0.85–1.15 band. This is reassuring and does not reproduce the severe
synthetic failure on the saved data. It does not establish zero bias: several
archive intervals exclude 1.

**Fix:** validate the readout at each dose × structure used in a contrast, and
report predicted/true and real-frame/true slopes separately alongside their ratio.
Keep new checks on already opened families exploratory; do not rewrite D023's
historical gate or describe its near-unity pooled slope as proof that all error is
unbiased noise.

### 3. H3's narrative claims more than its confidence intervals support

**Priority: correct before the course report.**
`docs/H3-PLAN-002_RESULT.md:51–61` says the predictor chooses "clearly better than
random in challenge families" and that its extra cost is "at most a few units".
The challenge JEPA-minus-random CI is **[-11.859, +0.242]**, so it also crosses
zero; this is not merely a pooled-cohort issue. JEPA-minus-real has CI
**[-1.324, +5.588]** pooled and **[-1.430, +8.547]** in challenge families.
No equivalence/small-effect criterion was established. The pooled upper bound is
also above the study's 5.12-unit tolerance.

**Fix:** report favorable point estimates with inconclusive contrasts. Preserve
the correct sealed verdict, "H3 not established." Do not turn failure to establish
a penalty into evidence that the penalty is negligible.

### 4. The reported endpoint error is displacement error

**Priority: correct metric naming and retain the missing control.**
`next_metrics.py:190–194, 253–257` computes
`||(decoded_end - decoded_start) - (true_end - true_start)||` and publishes it as
`endpoint_err_pred/real`. Absolute endpoint error is `||decoded_end - true_end||`.
The former cancels shared position bias, which is exactly what the research plan's
separate absolute-endpoint control is intended to expose.

Counterexample: add `[100,100]` to both decoded start and end. The reported
"endpoint" error is zero, while true absolute endpoint error is 141.42 units.

**Actual-data impact:** PHYS-002 reserved challenge absolute endpoint means are
**17.895 predicted / 3.613 real-frame**, compared with the published displacement
errors **17.857 / 3.935**. The substantive result survives this correction.

**Fix:** retain the existing displacement metric under that name and add genuine
absolute endpoint metrics. Publish a correction instead of overwriting sealed
historical measurements.

### 5. Agent-matched goals do not isolate block prediction error

**Priority: narrow the scientific interpretation.**
`planning002_metrics.py:7–10, 41–43` and D025 describe a block-placement or
block-prediction-error measurement. Matching a goal image to the *realized* agent
pose cancels agent position in the real-future comparison, but predicted latents
can still contain agent prediction errors. Block-angle errors can also affect
the predicted-latent cost. A nonlinear encoder need not separate these factors.

A simple separable-latent fixture with perfectly predicted block coordinates and
agent-only prediction error changes the chosen action and adds regret. The
current tests vary block coordinates while keeping agent predictions exact, so
they do not challenge this interpretation.

**Fix:** call it the effect of total latent prediction error under a privileged,
candidate-specific goal cost. It remains a useful controlled measurement, and
the documentation correctly discloses its use of future poses. It cannot attribute
a regret difference specifically to displacement shrinkage or demonstrate deployed
LeWM planning performance without further controls.

### 6. Archive audit confidence intervals resample dependent windows

**Priority: repair the exploratory uncertainty estimate.**
`phys002_audit.py:261–263, 303–308` samples archive episodes with replacement, then
bootstraps each selected window as independent. Windows from one episode can share
scene and dynamics. The code retains episode IDs but does not cluster on them.

Replaying only the seeded sampling/acceptance logic found **600 windows from 584
episodes**, and **335 moved windows from 331 episodes**. Duplication is limited, so
the numerical impact may be small; the reported bootstrap unit is nevertheless
incorrect for related windows. SHIFT-002's one-accepted-family-per-episode
collection avoids this particular problem.

**Fix:** bootstrap archive episodes with all of their windows grouped, or sample
one accepted window per episode. Report an exploratory corrected interval. Do not
infer an interval's revised width without recomputing it.

### 7. The archive "faithfulness" check has no upper bound

**Priority: fix before reusing the rule.**
`shift002_metrics.py:223–224` and `shift001_metrics.apply_shift_criteria` accept
any slope CI whose lower bound exceeds 0.9. A CI of **[2,3] passes**. An arbitrarily
wide interval containing 1 passes as well. This checks neither proximity nor
equivalence to faithful prediction.

**Fix:** prespecify a two-sided tolerance and adequate precision for faithfulness,
or rename the existing check to the narrower property it measures. The actual
SHIFT-002 archive ×1 varying result, 1.042 [1.003,1.083], is reassuring; the
counterexample demonstrates a rule defect, not a bad observed control.

### 8. The Mac runbook's command prefix fails in default zsh

**Priority: small reproducibility fix.** `docs/NEXT-001_RUNBOOK.md:29–41` stores an
entire command in a scalar `RUN` and executes `$RUN.distribution_audit ...`.
Default zsh does not split an expanded scalar into command words. A `zsh -f`
reproduction exits 127 with "command not found" for the entire expanded string.

**Fix:** show a shell function, e.g. `run_stage() { env ... uv run --locked
--no-sync python -m "jepa_action_effects.$1" "${@:2}"; }`, or write explicit
commands. Merely changing the executable to `python` inside the scalar will not
fix the shell parsing issue.

## Quality and recommended next steps

The strongest parts are the separated pure metrics, deterministic collection,
simulator-only case selection, family bootstrap in the main confirmations,
validation-only probe selection, disjoint follow-up collections, and explicit
stopping of SHIFT-001/H3-PLAN-001. The broader readout and archive controls were
worth doing. Tests exercise meaningful analytic cases, rather than only successful
CLI execution.

The weak point is orchestration: experiment modules repeat audit/collect/develop/
seal/confirm/report logic and import private helpers from previous experiment
versions. The same provenance omissions recur across runners. For a future version,
extract a small common runner and test rejection of changed sources, goals, scaler,
probe, family bytes and stale caches. Do not casually refactor already sealed
source files and then claim they reproduce under the old seal.

For interpretation, distinguish the measured **readout-decoded projection slope**
from a literal claim that every block moves 56% as far. The statistic is a pooled
dot-product regression and weights larger true displacements more. A low slope
can reflect direction errors as well as shorter predicted vectors. Report angle
and magnitude diagnostics before identifying a mechanism.

The statement in PHYS-002 that the effect is "not a probe or data artifact" is too
categorical. Probe-size stability and near-unity real-frame slopes are useful
evidence. They do not fully validate a nonlinear readout away from the distribution
of real encoded frames, nor does global latent contraction uniquely identify block
motion. State what the controls ruled out, and keep this residual limitation.

The next useful work is a corrected, compact synthesis and an independent rerun
subset, not another automatic expansion of experiments. The task board's long
handoff and doctor readiness text still describe earlier pilot stages; reconcile
them with the completed studies. Preserve H1/H3 as not established and the
magnitude trend as checkpoint-, distribution-, horizon- and measurement-specific.

## Reproduction notes

The review's local scripts are under the ignored directory
`outputs/branch_review_2026-09-29/`:

```sh
.venv/bin/python outputs/branch_review_2026-09-29/jepa_branch_review_checks.py
.venv/bin/python outputs/branch_review_2026-09-29/jepa_branch_review_recompute.py
.venv/bin/python outputs/branch_review_2026-09-29/jepa_branch_review_followup.py
```

They were executed initially from `/tmp/` with the same contents. They are review
artifacts, not new tests incorporated into the scientific protocol. The first
uses temporary mocks; the second reaggregates saved records and computes exploratory
per-dose checks; the third checks episode duplication and absolute endpoints.
No production source or frozen protocol was changed by this review.
