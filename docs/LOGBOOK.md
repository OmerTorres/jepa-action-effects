# Project logbook

Append one entry per substantive work session. Dates use Asia/Jerusalem; distinguish
performed checks from plans. Record failed attempts when they affect reproducibility.

## 2026-09-29 — REVIEW-FIX-001 Phase B / REVIEW-FIX-002 — Follow-up reanalysis, rerun, provenance

**Actor:** Claude (Claude Code desktop, Omer's Mac), continuing the approved
review-response plan.

**Code (new modules only; no seal-hashed file edited):**

- `followup_metrics.py`, `review_followup.py` (stages `phys`, `shift`, `archive`,
  `rerun`, `report`);
- `provenance.py`, `protocol_gates.py`;
- tests in `test_followup_metrics.py` and `test_provenance.py`.

**Commands and results:**

- `python -m jepa_action_effects.review_followup phys` (17 s):
  - PHYS-002 reserved challenge absolute endpoint error 17.895 / 3.613 (matches
    the review); displacement error 17.857 / 3.935 (matches the sealed values);
  - predicted length ratio 0.629 [0.583, 0.673]; mean \|angle\| 19.0°, against
    real frame 1.011 and 4.9°.
- `... shift` (17 s): per-dose readout slopes 0.93–1.05, all inside the band.
  Challenge length ratio 1.27 → 1.03 → 0.96 with angle about 26–28°; archive length
  1.11 → 0.95 with angle 33° → 18°.
- `... archive` (1 min 7 s): reproduces 600/584 and 335/331. Episode-clustered CIs
  are practically unchanged (slope_pred [0.932, 0.993]).
- `... rerun` (43 s): 150 PHYS-002 reserved families re-scored, with record,
  embedding and summary differences all 0.0. 30 SHIFT-002 families, record
  difference 0.0.
- `... report` wrote `review-followup-runtime.json`.
- `python -m jepa_action_effects.provenance`: PHYS-002, SHIFT-002 and H3-PLAN-002
  all valid. SHIFT records lack `input_sha256` (reported as not recorded).
  Output: `provenance-check.json`.
- `make check` and `make test`: see the commit.
- `_sources()` for PHYS-002, SHIFT-002 and planning002 still equal their seals.

**Limitations:** exploratory, on opened families. The decomposition uses the same
probe. Command magnitude and displacement size are not separated.

**Next action:** the course report (REPORT-001), using the corrected wording in D026
and REVIEW_FOLLOWUP_RESULT.md.

## 2026-09-29 — REVIEW-FIX-001 Phase A — Corrections after REVIEW-BRANCH-002

**Actor:** Claude (Claude Code desktop, Omer's Mac), at Omer's request to fix and
correct before improving.

**Changes:**

- D026 records the dispositions of all eight review findings.
- Correction sections were prepended to PHYS-002_RESULT, SHIFT-002_RESULT and
  H3-PLAN-002_RESULT; the original text is kept.
- STUDY_SYNTHESIS correction addendum.
- NEXT-001_RUNBOOK: `$RUN` scalar replaced by a `run_stage` function, with a dated
  correction note.
- `doctor.py` readiness text updated (unsealed file).
- TASKS: a current-state handoff paragraph, and rows REVIEW-FIX-001 (IN_PROGRESS)
  and REVIEW-FIX-002 (TODO).

No seal-hashed source, sealed output or verdict was changed.

**Commands:**

- `zsh -f` test of `run_stage doctor`: exit 0 and prints the new readiness text.
- `make check` and `make test` (results below in the commit's checks).

**Limitation:** the absolute-endpoint and per-dose numbers cited in the corrections
come from the review's scripts. They are recomputed in-repo in Phase B.

**Next action:** Phase B, `review_followup.py` (absolute endpoints, length/direction
split, per-dose readout, episode-clustered archive CI, independent re-scoring).

## 2026-09-29 — H3-PLAN-002 — Agent-matched latent-planning regret; H3 not established

**Actor:** Claude (Claude Code desktop, Omer's Mac). Omer chose option (a).

**Code and protocol:** `planning002.py`, `planning002_metrics.py` and
`tests/test_planning002.py`, recorded as D025 (commit 10fecef). `make check` passed;
`make test` gave 134 passed.

**Commands:** `python -m jepa_action_effects.planning002 audit|collect|develop|seal|confirm|report`.

- **Audit:** 1,060 forbidden seeds.
- **Collect:** 150 fresh families.
- **Develop** (2 min 27 s): 295 of 300 families with a valid goal; 737 valid goals,
  497 informative. The check passed (real − random −9.08); JEPA − real +4.30
  [2.23, 6.49].
- **Seal**; **confirm** (1 min 15 s; `confirmation-opened.json`, seal 501f9eb0…);
  **report**.

**Confirmed (148 families; 361 valid goals, 240 informative):**

- regret JEPA 15.35, real latent 13.29, random 19.32;
- check passed: real − random −6.03 [−9.28, −2.44];
- **H3 not established:** JEPA − real +2.07 [−1.32, 5.59];
- challenge cohort +3.41 [−1.43, 8.55];
- the ordinary cohort failed its own check.

**Choice histograms (challenge, reserved):**

- JEPA: [31, 134, 20, 10, 12, 17];
- real latent: [63, 79, 25, 16, 21, 20].

**Limitations:** realized agent pose in the goal images; about 40% of goals dropped by
the all-six-valid rule; the development effect did not replicate at CI level.

**Next action:** course report. The essential-scope experiments (PHYS-002, SHIFT-002,
H3-PLAN-002) are complete.

## 2026-09-28 — H3-PLAN-001 — Development fails the latent-cost check; seal refused

**Actor:** Claude (Claude Code desktop, Omer's Mac), NEXT-002 Phase 2.

**Code and protocol:** `planning_regret.py`, `planning_metrics.py` and
`tests/test_planning_regret.py`, recorded as D024 (commit fe5cd0b). `make check`
passed; `make test` gave 130 passed.

**Commands:** `python -m jepa_action_effects.planning_regret audit|collect|develop|report`.

- Collect: 150 fresh families in 29 s, not scored.
- Develop: 300 PHYS-002 development families in 79 s; 1,018 valid goals, 741
  informative, no family without a valid goal.

**Development (pooled, 300 families):**

- Regret: JEPA 21.4, real-latent 20.9, random 17.7.
- real_latent − random = +3.26 [1.84, 4.73]: the latent cost is **worse than
  random**, so the `latent_cost_beats_random` check failed.
- jepa − real = +0.50 [−1.41, 2.47].
- **`seal` refuses; the reserved families remain unopened.**

**Diagnosis.** In challenge families the real-latent chooser picks `hold` for 398 of
678 goals. The goal image keeps the agent at its start position, and latent distance
is dominated by agent position, so the cost rewards not moving. This is the known
agent confound (REVIEW-EXT-001), a goal-image design error rather than a model
property.

**Next action:** Omer chooses the goal-image redesign, e.g. placing the agent at each
candidate's realized final position, or at a fixed reachable pose. Seal it as a new
version on fresh families.

## 2026-09-28 — SHIFT-002 — Push-magnitude dose-response, confirmed

**Actor:** Claude (Claude Code desktop, Omer's Mac). Omer said "continue" after
SHIFT-001 stopped; the recommended SHIFT-002 redesign was executed.

**Code and protocol:** `shift002.py`, `shift002_metrics.py`, `tests/test_shift002.py`,
recorded as D023 (commit 8cfe445). `make check` passed; `make test` gave 126 passed.

**Commands:** `python -m jepa_action_effects.shift002 audit|collect|develop|seal|confirm|report`.

- **Audit:** 730 forbidden seeds, 1,803 excluded archive episodes.
- **Collect** (4 min 40 s): 180 challenge + 180 archive families. Archive
  rejections over 12,627 attempts: block not at rest 7,277; own future static 3,149;
  identity failed 1,665; too large to scale 126. Identity max: position 0.497, frame
  0.55/255, own future 0.98.
- **Develop** (31 s): all checks passed; challenge R 0.90/0.82/0.68, archive R
  1.04/0.97/0.82.
- **Seal**; **confirm** (60 s; `confirmation-opened.json`, seal 0349327d…);
  **report**. Verification error 0.0.

**Confirmed (240 reserved families):**

- dM challenge +0.168 [0.107, 0.232] and dM archive +0.154 [0.095, 0.212]: both
  supported.
- dS archive +0.001 [−0.014, 0.017]: no effect within ±0.1.
- dS challenge −0.062 [−0.102, −0.024]: varying slightly more shrunk.
- slope_real 0.96–1.03 in every cell.

Full tables are in SHIFT-002_RESULT.md.

**Limitations:** one checkpoint and horizon; noisy archive-scene readout (about 13
units, unbiased); the 0.15 push never reached the block.

**Next action:** H3-PLAN-001 (regret under the model's own latent planning cost).

## 2026-09-28 — SHIFT-001 — Implementation, collection, development; seal refused

**Actor:** Claude (Claude Code desktop, Omer's Mac), executing the approved NEXT-002
plan, Phase 1.

**Code:**

- New `shift001.py` (stages `audit`, `collect`, `develop`, `seal`, `confirm`,
  `report`) and `shift001_metrics.py` (pure metrics and `apply_shift_criteria`).
- New tests in `tests/test_shift001.py`. `make check` passed; `make test` gave 122
  passed.
- No PHYS-002-sealed source was edited.

**Simulator finding.** Resetting with `goal_state = state` terminates on the first
step: PushT success needs agent and block within 20 units of the goal.
`distribution_audit._replay` ignored termination; `run_family` refuses it. Archive
families now use `safe_goal`: goal agent and block at the farthest interior corners,
with no wall overlap and no kick. A simulator test checks exact start reproduction.

**Collection (simulator/archive only, no model).**

- 160 challenge + 160 archive families (60 development / 100 reserved each) in
  3 min 14 s.
- Archive rejections over 8,131 attempts: block not at rest 4,662; own future static
  2,110; identity failed 1,068; out of bounds 128.
- Accepted archive replays: position error ≤ 0.50 (median 0.23), frame mean-abs ≤
  0.56/255, own-future error ≤ 0.90.

**Pre-scoring revision.** The challenge `lateral` pair (+0.2 sideways) moved the block
in 0 constant / 9 varying families: the offset steers the agent past the block. The
challenge cell statistics were restricted to the balanced `push` and `small` pairs
(`PRIMARY_LABELS`). Nothing had been scored. The first audit and collection were kept
in `outputs/shift001-prerevision/`. Re-audit and recollection reproduced byte-identical
families.

**Development (120 families, exploratory;
`python -m jepa_action_effects.shift001 develop`, 31 s, verification error 0.0):**

| Quantity | Estimate [95% CI] |
| --- | --- |
| R challenge constant push 0.6 | 0.677 [0.555, 0.807] |
| R challenge constant small 0.21 | 1.221 [0.995, 1.457] |
| dM (small − push, constant) | +0.544 [0.341, 0.754] |
| dS challenge (varying − constant) | −0.046 [−0.130, 0.042] |
| dS archive | +0.029 [−0.020, 0.065] |
| dT (archive varying − challenge varying small) | −0.316 [−0.564, −0.046] |
| R archive varying own future | 0.969 [0.893, 1.065] |
| slope_real per cell | 0.95–1.01 |
| floor challenge / archive | 0.36 / 0.54–0.56 |

**Manipulation checks.**

- Failed: `challenge_constant_push_shrinks` (upper bound 0.807 vs 0.8) and
  `floors_below_limit`.
- Passed: archive own-future faithful and ≥ 30 moved families per cell.
- **`seal` refuses by design; the 200 reserved families remain unopened.**

The floor fails for two reasons. Small pushes move the block about 12 units (readout
error about 5). Archive-state frames read out with about 12.5 units of error versus
about 5 on challenge frames, although slope_real stays about 0.95.

**Exploratory reading (development only, not confirmatory):** within-block constancy
shows no effect. Push magnitude shows a large one: the 0.21 push is over-predicted
(R ≈ 1.2) and the 0.6 push under-predicted (R ≈ 0.68). This is consistent with
predictions pulled toward a typical displacement. Magnitude-matched archive states do
not remove it.

**Next action:** Omer chooses the redesign. The candidate is a magnitude dose-response
version (SHIFT-002) with a readout-validity gate on slope_real instead of the
effect-floor ratio, sealed on fresh families.

## 2026-09-28 — NEXT-002 Phase 0 — Record PHYS-002 and plan SHIFT-001 / H3-PLAN-001

**Actor:** Claude (Claude Code desktop), at Omer's request for a next-steps plan
(2–3 weeks; scope: SHIFT-001 and H3-PLAN-001).

**Changes (docs only):**

- DECISIONS D022: PHYS-002 verdicts, the audit reframing, and the magnitude-only gap
  in DIST-001's in-support rule.
- New `PHYS-002_RESULT.md`: final write-up plus audit table; the auto-draft is kept.
- STUDY_SYNTHESIS addendum.
- TASKS: H3-PLAN-001 row; the PHYS-002 row now links the final result.

No code, experiment or seal changed.

**Plan:**

- SHIFT-001: a 2×2 of challenge vs archive rest-window states × constant vs
  archive-residual action blocks, with matched block means within family. Includes a
  magnitude check and manipulation checks that must reproduce PHYS-002 and the audit
  on development. Sealed, then confirmed on 100 + 100 fresh families.
- H3-PLAN-001: regret when choosing by latent distance to a rendered goal image,
  compared with real-latent, random and oracle choice. Sealed and confirmed on 150
  fresh families.

**Next action:** implement `shift001.py` / `shift001_metrics.py` with model-free
tests, then run development on the Mac.

## 2026-09-28 — PHYS-002 — Exploratory post-confirmation audit (data, probe size, archive control)

**Actor:** Claude (Claude Code desktop, Omer's Mac), at Omer's request to check whether
the PHYS-002 data or probes were flawed. Exploratory: run after the reserved families
were opened, read-only on sealed outputs. First run from session scratch scripts,
then folded into `src/jepa_action_effects/phys002_audit.py` (tests in
`tests/test_phys002_audit.py`) and rerun with `env PYTHONWARNINGS=ignore
UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked --no-sync
python -m jepa_action_effects.phys002_audit run` (2 min 11 s). Numbers below are
from that run (`docs/manifests/phys002-audit-runtime.json`). No sealed file changed;
`phys002._sources()` still equals the PHYS-002 seal. `make check` passed and
`make test` gave 115 passed.

**Checks and results (challenge cohort unless stated):**

- **Time alignment.** Decoded real displacement vs true: slope 1.06 at step 1 and
  1.01 at step 2. Predicted: 0.88–0.90 at step 1, 0.56–0.59 at step 2 (development and
  reserved). No one-step offset (that would give ≈0 at step 1). The shortfall is
  concentrated in the second autoregressive step.
- **Family construction.** `simulator_replay` stores history frames at warm-up
  controls end−10/end−5/end with `warmup_controls[-10:]`, future frames every 5
  controls: the same convention as the archive windows used below.
- **Probe size.** Retraining the MLP (seed 20261005) on 2k/5k/10k/20k/50k frames gives
  validation block error 10.9/7.8/6.3/4.9/3.5 and pred/real slope ratio
  0.557/0.550/0.563/0.581/0.575. The shrinkage estimate does not depend on probe size.
- **Latent space (readout-free).** Step-2 slope of (ẑ−z_t) on (z−z_t) is 0.79
  [0.78, 0.80] (includes agent motion); predictor/persistence MSE ratio 0.24.
- **Archive control (in-distribution states and actions, same predictor/probe code).**
  600 windows (9 skipped for out-of-bound controls), 335 with block motion > 5.12
  (median 29 units): slope_pred 0.96 [0.93, 1.00], slope_real 0.95 [0.92, 0.98];
  endpoint error pred 13.3 vs real 11.6 (window bootstrap). Matched by displacement
  size, archive slope_pred is 0.89–0.99 while study development slope_pred is
  0.55–0.66 for 20–400-unit displacements. Archive windows where the block was at rest
  in the history (n=97) also show no shrinkage (slope_pred 1.02 [0.96, 1.09]); every
  study challenge family starts with the block at rest (history motion median 0).
- Real-frame readout error is higher on archive frames (11.6 vs about 4 on study
  frames); the pred/real comparison uses the same probe, so the contrast holds.

**Interpretation (revised):** No data or pipeline error was found, and the probe is
not the limiting factor. The shrinkage is not a general property of this checkpoint:
it appears on the designed challenge families but not on archive windows. The
remaining explanations are the study's constant, high-magnitude (~98th-percentile)
5-control pushes and/or the challenge scene layouts. DIST-001's in-support rule only
tested magnitude, so this temporal-structure shift was not covered by the sealed
variant. PHYS-002's sealed verdicts stand as measured on the study distribution but
must be reported as distribution-specific.

**Next action:** A 2×2 follow-up (challenge vs archive-like states × constant vs
archive-like action blocks) to identify which factor causes the shrinkage, sealed and
confirmed on fresh families.

## 2026-09-28 — NEXT-001 — Mac execution of DIST-001, REANALYSIS-001, READOUT-002, PHYS-002

**Actor:** Claude (Claude Code desktop, on Omer's Mac), at Omer's request to continue
the local work on branch `claude/youthful-clarke-akm28u` (commit 92fe9d4).

**Commands:** `NEXT-001_RUNBOOK.md` steps 0–5 in order, each stage via
`env PYTHONWARNINGS=ignore UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python
uv run --locked --no-sync python -m jepa_action_effects.<module> <stage>`.
`make setup` was skipped (the locked `.venv` was present); `make check` passed and
`make test` gave 111 passed, 0 skipped, before and after the run. No file under
`src/` was edited. Runtime: macOS 15.7.3 arm64, CPU, float32, Python 3.10.21,
torch 2.14.0. The shell sandbox printed a harmless OMP `/tmp` warning.

**DIST-001 (measured):**

- Archive keys: `state`, `pixels`, `proprio`, `step_idx`, …; no goal key.
- Identity: 20 archive episodes × 40 steps, kick-free reset. Final position error
  0.0046; frame mean-abs 0.53/255 at reset and 0.42/255 at step 40;
  `rendering_mismatch_flag` false. No pause.
- Gap: 198 windows (2 skipped for out-of-bound controls). Predictor/persistence
  latent-MSE ratio 0.0318 on archive frames vs 0.0398 on simulator re-renders.
  Simulator/rendering does not explain the earlier 20–30x ratio gap.
- Actions: 0% of study blocks (challenge or ordinary) above the archive p99
  magnitude (0.683); challenge median magnitude rank 0.979. In-support variant rule
  not triggered, `PUSH` unset. `study_joint_tail_mass` is 0 everywhere because study
  controls are exactly constant within a block (spread 0) and no archive block is;
  this is a temporal-structure shift, not a magnitude one.

**REANALYSIS-001 (exploratory, opened families):** On gallery confirmation families
with 20.48 negatives, challenge static endpoints have matched accuracy 0.93 and
persistence advantage +0.011 [−0.006, 0.027]; moved endpoints have matched accuracy
0.47 and matched margin −0.036 [−0.066, −0.006]. LATENT slopes: designed 0.38
[0.32, 0.45], designed-neither 0.82 [0.68, 0.92]; boundary A exactly-one minus both
−0.16 [−0.20, −0.13].

**READOUT-002 (measured):** 50k train / 10k validation frames (rejected poses 17,554
/ 3,457), collect 21 min. Ridge learning curve block mean 17.7 → 15.7 (1k → 50k),
R² 0.984: linear readout saturates. MLPs: validation block mean/p95 3.48–3.64 /
7.1–7.4. Sealed selection: `mlp-20261005` (3.48 / 7.15, angle 1.8°). Study-check
(60 READOUT tuning families, real frames): 4.41 / 9.80, R² 0.996. Sensitivity of
the selected MLP: gain 0.91/0.95/0.93/0.99 and SNR 0.68/1.04/1.53/2.66 for
0.5/1/2/5-unit shifts. Agent error roughly doubles on study frames (10.2 → 20.0);
agent is not used by PHYS-002.

**PHYS-002 (measured):** audit passed (300 development ids, 60 excluded, 420
forbidden seeds, no in-support variant); collect 150 reserved families in 29 s;
develop 68 s; seal; confirm 45 s (`confirmation-opened.json` written, seal sha256
ef26d54c…); report. Replay verification max error 0.0 in both phases.

Reserved confirmation, challenge cohort (100 families, 95% family bootstrap):

- floor ratio 0.120 [0.113, 0.128]: floor passed;
- D_boundary 17.05 [15.85, 18.15]: physical failure **supported**;
- slope_pred 0.561 [0.517, 0.605], slope_real 1.006, difference −0.445
  [−0.490, −0.401]: shrinkage **supported**;
- h1_contrast (exactly-one minus both) −12.37 [−14.25, −10.49]: H1 **not
  established**; the error is larger when both branches contact the block;
- endpoint error pred 17.86 vs real 3.93; regret JEPA 3.96, real-frame 0.19,
  random 23.9.

The ordinary cohort fails the floor (ratio 0.35), so its claims are inconclusive.
Development (300 families, exploratory) gave the same direction: challenge floor
0.121, D_boundary 16.68, slope_pred 0.592, h1 −11.25.

**Limitations:** One checkpoint, one horizon (2 native steps). All moved blocks were
contacted, so `slope_pred_no_contact` is undefined. The ordinary cohort is
floor-limited. Study controls are constant within a block, unlike the archive.
Interpretation is not yet written.

**Next action:** A follow-up session interprets PHYS-002 against D021, updates
STUDY_SYNTHESIS/DECISIONS, and decides whether ERASURE-001 is unblocked.

## 2026-09-28 — NEXT-001 — Phase 0–3 implementation (cloud, no model)

**Actor:** Claude (Claude Code), at Omer's request to proceed with NEXT-001.

**Changes:**

- **Phase 0.** The two artifact-dependent tests now skip when `outputs/` is absent.
- **New modules:**
  - `next_common.py`: paths, sealing re-exports, frozen-model helpers.
  - `next_metrics.py`: pure action-support, boundary-pair, floor-relative
    displacement, slope, regret, bootstrap and D021 decision rules.
  - `distribution_audit.py`: DIST-001 `keys`, `actions`, `identity`, `gap`,
    `report`.
  - `reanalysis.py`: REANALYSIS-001, reporting only.
  - `readout_scale.py`: READOUT-002 `collect`, `fit`, `report`.
  - `phys002.py`: PHYS-002 `audit`, `collect`, `develop`, `seal`, `confirm`,
    `report`.
- **New tests:** `tests/test_next_metrics.py` and `tests/test_next_stages.py`.
- **Docs:** added `NEXT-001_RUNBOOK.md`, a D021 amendment, an interface-contract
  note, and task statuses set to IN_PROGRESS (awaiting the Mac run).

**Commands and results:** Linux container, system Python 3.10.20 (uv 0.8.17 cannot
fetch the pinned 3.10.21); `uv sync --locked --python /usr/bin/python3.10`.

- `uv run --no-sync ruff check .`: passed.
- `ruff format --check .`: 75 files formatted.
- `uv run --no-sync pytest -q`: **108 passed, 3 skipped, 0 failed**. The skips are
  the checkpoint test and the two artifact tests.

Synthetic fixtures show the decision rules working as intended:

- an oracle predictor gives `not_established`;
- a 0.5-shrunk predictor gives shrinkage and physical failure `supported`;
- a readout noisy relative to the effect gives `inconclusive_readout_floor`.

LATENT pair terms give the latent slope exactly (A = mean Δẑ·Δz). An end-to-end
identity fixture on a simulator-generated HDF5 reproduced every frame.

**Finding during implementation.** A reset with `goal_state` = FAR_GOAL moves the
requested start block by about 2 units, with zero residual velocity. The cause is
stale collision shapes from the wall-overlapping goal pose. Measured case:
(300, 250, 0.4) → (299.64, 251.71, 0.40007); using the start state as the goal
removes the shift. Earlier results use realized states and are unaffected; the
designed clearances were approximate. An intermediate hypothesis that agent overlap
caused it was tested and rejected.

**Limitations:** No checkpoint inference, archive access or collection was run
here, so real-data behaviour is untested. The HDF5 state/pixel key names are
discovered at runtime; if they are missing, `identity` records itself as
unavailable. Runtime estimates in the runbook are projections.

**Next action:** Omer runs `NEXT-001_RUNBOOK.md` on the Mac and pushes the four
manifests and the PHYS-002 draft. A follow-up session interprets them.

## 2026-09-28 — NEXT-001 — Post-review plan

**Actor:** Claude (Claude Code), at Omer's request to plan the next steps.

**Changes:** Added NEXT-001_PLAN.md, the DIST-001/REANALYSIS-001/READOUT-002/PHYS-002
TODO rows, ERASURE-001 marked BLOCKED (deferred), and the proposed D021. This was
documentation only: no code, experiments or protocol seals.

**Evidence:** The Hugging Face connector lists `quentinll/lewm-pusht/weights.pt`
(72,290,721 B, LFS) and the dataset `pusht_expert_train.h5.zst` (13.1 GB), but it
returns only text and images. A direct `curl` to huggingface.co from the container
still gets proxy 403. Omer chose to run every model/archive stage on the Mac and
to go through Phase 3 before reporting back.

**Next action:** Implement Phase 0 plus the code and mock tests for Phases 1–3 in
the cloud, write NEXT-001_RUNBOOK.md, then run the stages on the Mac.

## 2026-09-28 — REVIEW-EXT-001 — External methodological review

**Actor:** Claude (Claude Code), at Omer's request for a research review.

**Changes:** Added `EXTERNAL_REVIEW_2026-09-28.md` and a DONE task row. No source,
protocol, decision or historical entry changed.

**Commands and results:** Read all tracked docs, manifests and core modules. In a
Linux container with system Python 3.10.20 (uv could not fetch the pinned 3.10.21),
`uv sync --locked --python /usr/bin/python3.10` succeeded; `uv run --no-sync pytest -q`
gave 88 passed, 1 skipped, 2 failed. Both failures are `FileNotFoundError` on ignored
`outputs/` manifests (`test_gallery_audit.py::test_real_replay_pose_order_geometry_and_restoration`,
`test_predictor_gallery.py::test_audit_excludes_confirmation_payloads_and_rejects_changed_source`).
Ruff check and format passed. A scratch render test (200 random poses, simulator
only) found identical 224-px frames for 39.5%/9%/1%/0% of 0.1/0.25/0.5/>=1.0-unit
block shifts. Checkpoint download from Hugging Face was blocked by the container's
network policy, so no inference was run.

**Main conclusions:** The absolute 1.28-unit gate is 0.56 px at 224 px. Readouts
were trained on about 240 block poses with an unsaturated learning curve, and the
predictor's physical error was never compared against the real-frame readout floor.
LATENT-001's contact contrast is confounded by the `hold` candidate. Gallery
persistence contrasts pool stationary endpoints. Action and simulator distribution
shift were never measured, although the archive and fresh-simulator predictor
error ratios differ by 20-30x.

**Limitations and next action:** Review only. The inferences about ignored
artifacts are design-based. Next is N1/N2/N4 in the review; see its section 5.

## 2026-09-28 — ERASURE-001 — Selective-erasure feasibility planning

**Actor:** Codex, at Omer's request to plan next steps.

**Changes:** Added ERASURE-001_PLAN.md, a TODO task and provisional D020. The plan
uses observed-history embeddings from 240 existing fit families plus a fixed
30/30 calibration/assessment subdivision of the 60 tuning families. It defines
two continuous linear-covariance erasers, refitted ridge/control probes, bounded
nonlinear leakage diagnostics, explicit accessibility/removal/preservation gates,
one-hour cumulative execution and a stop boundary before predictor intervention.

**Commands and results:** Read TASKS, latest LOGBOOK, study skill, PROBING review,
relevant RESEARCH_PLAN/DECISIONS sections and model/recovery APIs. Refreshed the
two primary LEACE/amnesic-probing arXiv sources. File searches for nonexistent
`readout_data.py` and `gallery.py` returned errors; the applicable inspected
implementations are `readout_recovery.py` and `model_inference.py`.
`git diff --check` passed after documentation edits. No Python/environment change,
scientific test, model inference, probe fitting or new confirmation access.

**Limitations and next action:** Numerical gates are proposed feasibility choices;
execution must seal them before fitting. Existing assessment families are not
fresh evidence. Implement and validate ERASURE-001 if the plan is taken forward;
report failure/incompletion without adding methods or data. Earlier readout and
historical simulator-identity limitations remain. PROBING-001 documentation edits
were already present and were preserved.

## 2026-09-28 — PROBING-001 — Alternative probing literature review

**Actor:** Codex, at Omer's request to search arXiv before ending the study.

**Changes and evidence:** Added `PROBING-001_REVIEW.md` and updated TASKS. Reviewed
amnesic probing/LEACE, circuit probing, distributed alignment search, MDL probing,
and demixed PCA. Cross-checked JEPA applications and found closer precedents in
MotionJEPA, World Model Control by Trajectory Reachability Metrics, and a V-JEPA 2
causal-state workshop paper. The latter's official abstract was accessible, but
its OpenReview PDF was blocked by browser verification. The memo distinguishes
primary-paper claims, local facts, and proposed adaptations; failure to locate an
exact method/JEPA pairing is not treated as evidence of first use.

**Commands and results:** Web search/open/find calls inspected primary sources;
`rg` inspected research-plan contribution/readout sections and inference functions.
`rg --files docs` resolved an attempted read of nonexistent `docs/RELATED_WORK.md`:
the actual gallery review is `docs/GALLERY-001_RELATED_WORK.md`. `git status --short`
was clean before edits. `git diff --check` passed after the documentation changes.
No Python/environment changes, scientific tests or experimental computation.

**Interpretation and next action:** Selective erasure in observed-history or
predictor representations is a promising feasibility question, conditional on
held-out removal/selectivity checks and matched intervention controls. This is a
recommendation, not a frozen protocol or authorization to collect fresh data.
Prior failed readout gates remain; historical training-simulator identity remains
unproven. All 60 formerly reserved families are open, so a revised claim needs
fresh confirmation. No historical experiment or decision was rewritten.

## 2026-09-28 — PREDICTOR-GALLERY-001 — Sealed reserved comparison completed

**Actor:** Codex, implementing Omer's approved plan.

**Changes:** Added the staged `predictor_gallery` command, pure gallery-comparison
metrics, and eight behavioral fixtures. The scorer reuses GALLERY-001 banks,
READOUT projected real-frame cache, frozen predictor loader and action scaler.
It scores matched predictions, persistence and all five other-action
predictions against each endpoint's same bank. Added provenance, result report,
task update and D019. Earlier experiment code and artifacts were unchanged.

**Execution:** `PYTHONWARNINGS=ignore UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked --no-sync python -m
jepa_action_effects.predictor_gallery {audit,develop,seal,confirm,report}`
completed in that order (each brace term was a separate invocation). `audit`
selected exactly 30 development and 60 reserved families using metadata only.
`develop` completed 30 before `seal`; `confirm` wrote a persistent opening
record before reading reserved payloads, passed the primary measurement gate,
then completed all 60 family predictions. No result-dependent family selection
occurred. An earlier development-only run was saved as ignored
`outputs/predictor_gallery001-preliminary/` after adding control and interval
reporting; the final run began under a new source hash with an empty artifact
root. Confirmation was opened only in the final run.

**Measurements:** All 360 reserved endpoints replayed/restored saved RGB.
Primary gallery eligibility was 240/240 challenge and 116/120 ordinary;
nonidentical-query top-1 was 100%/98.75%. Twelve preselected families per
phase passed duplicate and reversed-candidate inference with zero maximum
absolute discrepancy; 12 reserved families passed duplicate/reversed gallery
encoding at zero. Four preselected prediction/gallery overlays per phase were
inspected. The 40-family primary challenge result: matched top-1 53.75%,
matched margin -0.00923 [-0.01927,-0.00005], matched-minus-persistence margin
0.00131 [-0.00807,0.00959], and matched-minus-wrong-action margin 0.04272
[0.03287,0.05159]. The central positive-evidence conjunction failed.
Global projected-CLS correspondence remained positive on the same reserved
families. Detailed cohort/scale and sparse-regime results are in
`PREDICTOR-GALLERY-001_RESULT.md` and ignored JSON records.

**Commands and results:** `make check` passed (Ruff, format, doctor on Python
3.10.21); `UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run
--locked --no-sync pytest -q tests/test_predictor_gallery.py` passed 8 tests;
`make test` passed 83 tests with 12 Gymnasium warnings; `UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python uv sync --locked --check` would make no
changes; `git diff --check` passed. The final runtime ledger recorded
112.273 seconds (13.139 development, 99.048 confirmation, 0.003 report plus
audit), below the 1,800-second cap. Final artifacts occupy 25.11 MB. The
tracked `manifests/predictor-gallery001-runtime.json` records hashes, family
IDs, stage times and checkpoint identity. Git HEAD at execution was
`0a6717f6080ec6c2adb9e856f245cd1754c8934a` with the new files untracked.

**Decision and limitations:** D019 records the sealed protocol and negative
central outcome. The results support action-sensitive gallery correspondence,
but not superiority to persistence at the primary challenge scale. The
gallery is conditional on realized nuisance state, projected CLS is not
object-isolated, physical readout gates still fail, and historical
training-simulator identity remains unproven. The reserved families are open.

**Next action:** Keep main-study expansion paused. Prepare a bounded claim
around the confirmed dissociation or separately scope a new measurement/model
question with fresh reserved families.

---

## 2026-09-28 — GALLERY-001 — Bounded gallery feasibility completed

**Actor:** Codex, implementing Omer's requested plan.

**Changes:** Added `gallery_render.py`, `gallery_metrics.py`, the staged
`gallery_audit.py` command and behavioral fixtures. Added the primary-source
related-work matrix, result report and tracked runtime manifest. Marked
GALLERY-001 done and appended D018. No checkpoint weights, upstream pilot
artifacts, predictor outputs or reserved confirmation labels changed.

**Execution:** `UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv
run --locked --no-sync python -m jepa_action_effects.gallery_audit audit`
passed and sealed 30 tuning IDs (20 challenge, 10 ordinary), family and
source hashes, and five seed streams. `render`, `encode`, and `report` stages
used the same command with `PYTHONWARNINGS=ignore` and completed. Total
cumulative experimental runtime was 46.792 seconds (0.099 audit, 8.460
render, 35.930 encode, 2.303 report), below the 1,800-second ceiling. The
first completed execution was preserved as ignored
`outputs/gallery001-preliminary/` before adding more detailed reporting;
the final source hashes were sealed and all stages reran from an empty root.

**Measurements:** All 180 branches replayed/restored exact RGB. At 10.24
units, 120/120 challenge and 59/60 ordinary endpoints had four geometry-valid
negatives and two nonidentical queries; at 20.48 units, 120/120 and 60/60.
One ordinary endpoint had only three 10.24-unit negatives because wall
penetration removed other proposed directions. Both cohorts had 100%
nonidentical-query retrieval at each scale. Family-weighted fifth-percentile
margin ratios were challenge/ordinary 0.847/0.831 at 10.24 and 0.965/0.946 at
20.48, all above the prespecified 0.5 gate. Duplicate/reversed encoding on
the twelve verification families had zero maximum absolute discrepancy.
Four seeded geometry overlays were inspected. Contact and motion regime
counts, bootstrap intervals, per-template distances and all rejection records
are in ignored `outputs/gallery001/`; artifact and source hashes are in
`docs/manifests/gallery001-runtime.json`.

**Verification:** `make check` passed (Ruff and doctor, Python 3.10.21).
Focused gallery pytest passed with nine tests. `make test` passed with 75
tests after the ninth focused resume test was added. `UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python uv sync --locked --check` passed and
would make no changes. `git diff --check` passed after documentation updates.

**Interpretation and limits:** The encoder can distinguish the controlled
rendered block positions; this is not a predictor result. Synthetic poses may
be outside the checkpoint's training distribution, and the 30 families are
exploratory tuning reuse. The bounded literature review finds feature-space
pose retrieval and action-discrimination precedents, so general method
novelty is unsupported. Physical readout gates remain failed and historical
training-simulator identity remains unproven.

**Next action:** Scope and freeze a predictor-to-gallery comparison with
persistence/action-permutation controls and independent evidence. Keep main
study expansion paused and confirmation labels closed until then.

---

## 2026-09-28 — GALLERY-001 — Next-step execution plan

**Actor:** Codex, at Omer's request to plan the next step.

**Changes:** Added GALLERY-001_PLAN.md, updated TASKS handoff, and appended D017.
The plan combines a primary-source contribution review with a bounded
30-family simulator/encoder feasibility audit. It adds nonidentical positive
queries because exact-frame retrieval from a gallery containing that frame
would be tautological. Prespecified translation scales, geometry rules,
coverage/robustness gates, artifact isolation, runtime recovery, tests and
proceed/narrow/stop/incomplete outcomes make execution reviewable.

**Commands and results:** Read the jepa-study skill, TASKS, recent LOGBOOK
entries, NOVELTY-002_MEMO, D013–D016, relevant RESEARCH_PLAN sections,
DEVELOPMENT, LATENT-001_PLAN and Makefile. Checked `uv sync --help` for the
planned `--locked --check` invocation; both flags are available. Ran
`git diff --check` successfully. Documentation-only changes: no scientific
test suite, simulator run, model encoding, or literature search executed in
this session. Existing dirty source files and historical artifacts preserved.

**Limitations and next action:** This is a proposed protocol, not a completed
measurement or frozen configuration. Implement and execute its literature,
geometry and encoder gates next. Passing would justify planning a subsequent
predictor comparison; it would not establish physical competence or authorize
confirmation data use.

---

## 2026-09-28 — NOVELTY-002 — Contribution reassessment

**Actor:** Codex, responding to Omer's concern about novelty and measurement.

**Changes:** Added NOVELTY-002_MEMO.md, a new task-board handoff, and D015.
Identified a closer September 2026 paper, AD-WM, that studies LeWM and PushT
action discrimination and planning diagnostics. Proposed a conditional
fixed-agent rendered block-pose gallery, with renderer and real-frame gates
before predictor scoring. No model result, collection, or confirmation use.

**Commands and results:** Read TASKS, latest LOGBOOK, skill, RESEARCH_PLAN,
LIT-001_REVIEW, LATENT-001_RESULT, simulator renderer/state-setter source, and
interface contract. Searched/opened primary papers for AD-WM, WorldSimProbe,
Delta-JEPA, C-JEPA, and adjacent object-centric evaluation. A read-only Python
script loaded the 60 saved tuning-family state artifacts and counted candidate
pairs with block endpoint distance >5.12 and agent endpoint distance <=5.12,
10.24, or 20.48: **zero pairs and zero families at all three cutoffs**.
This is a simulator-label feasibility census, not model-based selection.
`git diff --check` passed. Documentation-only changes; Python tests and model
inference were not run.

**Interpretation:** Current global CLS advantage may reflect agent motion.
AD-WM and other work substantially narrow any action-discrimination novelty
claim. A renderer-controlled block-pose comparison could test a more distinct
object-specific question, but its real-frame identity, valid alternatives,
representation sensitivity, and novelty are unproven.

**Next action:** Define and run a bounded renderer/encoder feasibility protocol
if pursuing this path. Keep physical main-study expansion and readout
confirmation paused until that protocol passes its gates.

---

## 2026-09-28 — GALLERY-001 — Preliminary renderer identity check

**Actor:** Codex, continuing Omer's measurement-strategy request.

**Changes:** Extended NOVELTY-002_MEMO.md with a simulator-only rendering
check; recorded D016 and marked GALLERY-001 in progress. No source, model,
checkpoint, collection, or reserved-confirmation artifacts changed.

**Commands and results:** Ran `.venv/bin/python /tmp/gallery_render_probe.py`
on the twelve preselected LATENT-001 verification families (six per cohort,
72 candidate endpoints): exact baseline replay RGB in 72/72, but restoration
after position-then-angle assignment in only 40/72. Traced the discrepancy
with `.venv/bin/python /tmp/gallery_render_trace.py`: setting the angle shifts
the T-block's reported position because its center of mass is offset. Ran
`.venv/bin/python /tmp/gallery_render_probe_fixed.py` with angle-before-position
assignment and shape reindexing: exact baseline and restored RGB in 72/72.
The cohort-count extension, `.venv/bin/python /tmp/gallery_render_probe_cohort.py`,
found a visible image change when borrowing the next candidate's block pose in
32/36 challenge and 0/36 ordinary endpoints. These temporary scripts were
diagnostic commands, not a reusable protocol. `git diff --check` passed after
documentation changes. No Python source changed, so project test suites were
not rerun.

**Interpretation and limit:** Render-only block manipulation is feasible at
these replayed endpoints when setter order is correct. Borrowed alternatives
do not furnish ordinary-family coverage; geometry validity and frozen-encoder
discriminability are still untested. No predictor scoring or novelty claim.

**Next action:** Specify a bounded, predeclared simulator-and-encoder gallery
protocol with geometry-valid alternatives and real-frame sensitivity gates.

---

## 2026-09-27 — LATENT-001 — Bounded latent audit completed

**Actor:** Codex, implementing Omer's authorized plan.

**Changes:** Added pure projected-CLS metrics, a staged audit/evaluate/report
command, sealed family records and source hashes, cumulative budget ledger,
preselected inference-order checks, and a study synthesis. Marked LATENT-001
done and recorded D014. Readout-confirmation families were excluded by split
metadata and never loaded for scoring. All 60 tuning families were scored.

**Commands and results:** `make repair-macos-pth` passed.
`UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked
--no-sync python -m jepa_action_effects.latent_audit` passed for `audit`,
`evaluate`, and `report`. Initial audit failed because READOUT used the older
`.sha256` sidecar naming; the new reader now validates either spelling. An
initial complete local run was moved to ignored `outputs/latent001-preliminary/`
after lint/reporting fixes changed the source hash. The final audit was sealed
before scores and all stages reran successfully. The final cumulative ledger
was 7.859 seconds, including 0.193 audit, 7.658 scoring, and 0.008 report.
A repeated `evaluate` and `report` reused all 60 sealed family records unchanged;
the aggregate record hash stayed
`4546fb56690397b9afe74eba994524afddc1f48d0581d773dc0d3c91cbae9b7e`.
The final cumulative ledger was 9.731 seconds, well within 1,800 seconds.
Twelve preselected duplicate/reversed-order checks
had maximum absolute discrepancy at most 1e-6.

**Measurements:** Overall projected CLS matched/wrong/persistence errors were
0.18691/0.62666/0.71122. Correspondence A was 0.43975 [0.40560, 0.47353]
and persistence improvement B was 0.52431 [0.49227, 0.55754], with 2,000
family bootstrap resamples. Challenge and ordinary intervals were positive.
Raw matrices, contact/motion/effect strata, paired contrasts, four preselected
examples, and coverage are in ignored `outputs/latent001/`. No readout or
checkpoint was trained. Sixty predictions were newly cached; all real projected
embeddings were reused; no raw CLS embeddings were accessed by this run.

**Validation:** Focused LATENT and inference tests: 21 passed. `make check`
passed Ruff, formatting of 46 files, and doctor. `make test` passed 66 tests
with 11 nonfailing Gymnasium warnings. `UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python uv lock --check` passed with 118 packages.
`git diff --check` passed. `docs/manifests/latent001-runtime.json` records
artifact/input hashes, source provenance, storage, and run conditions.

**Interpretation and limits:** Action conditioning improves prediction in the
checkpoint's projected representation on this exploratory sample. Agent motion
is substantial and may dominate global CLS differences. The earlier physical
readout gate remains failed, so block physics and physical regret are
unassessable. This is one checkpoint with unproven historical simulator
identity, and the tuning families are not independent confirmation data.

**Next action:** Stop experiments and prepare the course report from
LATENT-001_RESULT.md and STUDY_SYNTHESIS.md. Further scope requires a new
measurement strategy and proposal.

---

## 2026-09-27 — LIT-001 and LATENT-001 — Review and revised question

**Actor:** Codex, at Omer's request to carry out the literature/checkpoint review
and formulate the narrower question.

**Changes:** Added LIT-001_REVIEW.md and LATENT-001_PLAN.md; marked literature
review done and latent audit TODO. Recorded D013. Reviewed primary sources for
WorldSimProbe, WAMProbe, Delta-JEPA, CoCo, UWM-JEPA, and LeWM. The memo distinguishes
published/repository claims from independently measured results, extracts the
WorldSimProbe T1 and Delta-JEPA response definitions, and omits inapplicable video
metrics and unavailable exact definitions. The proposal defines exact matched,
wrong-action, persistence, and paired scores with family-level uncertainty.

**Commands and results:** Read TASKS, latest LOGBOOK, jepa-study skill, relevant
RESEARCH_PLAN/DECISIONS/INTERFACE_CONTRACT sections, and DELTA-001_RESULT. Inspected
the cached LeWM commit ca231f9f9d9ab041034b6d05e90b6e04bd6cff82 using `cat` on
`eval.py`, `jepa.py`, and `config/eval/pusht.yaml`, and `rg --files` for evaluation
and probing sources. An initial read of `config/eval.yaml` failed because that
file does not exist; the actual PushT configuration was then read successfully.
Web opens/searches inspected the versioned paper pages and linked author
repositories recorded in the memo. WAMProbe's metric-card link failed to load;
CoCo's retrieved v1 HTML/PDF referred to unrecovered appendix definitions. Those
limitations are explicit omissions, not inferred reproductions.

**Validation:** `git diff --check` passed. Documentation-only work; no Python or
environment change, so no tests or model inference were run. Existing dirty
implementation files and all historical experiment artifacts were preserved.

**Evidence and limits:** LIT-001_REVIEW.md includes source links, versions/access
date, checkpoint source settings, and the contribution boundary. No new model
result, reserved-family evaluation, or source replication was performed. The
proposed latent audit can assess representation-space correspondence; it cannot
establish object-specific physical correctness. Historical simulator identity
remains unproven. No claim of exhaustive novelty verification is made.

**Next action:** Implement and seal LATENT-001 before scoring, if proceeding
with the proposed exploratory scope. Physical expansion remains paused.

---

## 2026-09-27 — DELTA-001 — Bounded displacement measurement completed

**Actor:** Codex, implementing Omer's authorized DELTA-001 plan.

**Intent and changes:** Added a reusable `delta_probe` and staged
`python -m jepa_action_effects.delta_readout` workflow with input/code hashes,
family-weighted displacement and paired-effect gates, exact-zero readouts,
training curves, one-time confirmation isolation, conditional evaluator wiring,
and a persistent two-hour experiment ledger. Reused the 240 fit and 60 tuning
families from READOUT-001 without new collection. Preserved the prior pilot and
readout artifacts. The frozen plan remains in `docs/DELTA-001_PLAN.md`; its
outcome is in `docs/DELTA-001_RESULT.md`, D012, and the runtime manifest.

**Commands and results:** Ran `make repair-macos-pth` and
`UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked --no-sync python -m jepa_action_effects.delta_readout`
with each of `audit`, `fit`, `confirm`, `evaluate`, and `report`; repeated `fit`
and `report` to verify sealed-stage reuse and capture runtime. Audit passed:
the horizon was two native steps, fit/tuning/confirmation counts were 240/60/60,
and no confirmation identifier was present in READOUT-001's 600-entry real-frame
cache. Fit retained 1,424 transition pairs after 16 duplicate-frame pairs;
zero block-label conflicts were found. Its 32-pair diagnostic used distinct real
projected differences. Tuning had 43 families with meaningful paired effects.

**Measurement:** Existing-ridge difference endpoint mean/p95 was 9.39/26.78;
direct-displacement ridge was 4.83/13.03; the three anchored nonlinear seeds
were 4.29/14.64, 4.39/14.38, and 4.01/12.97. Their meaningful paired-effect
mean/p95 errors were 13.33/27.86, 8.54/17.55, 8.79/21.56, 8.75/20.59, and
8.25/20.59, respectively. All failed the prescribed mean/p95 limits of
1.28/2.56. Zero-displacement reference endpoint mean was 19.04. Two nonlinear
seeds reached the 3,000-epoch cap; one stopped after its 200-epoch tuning
patience. Fit and tuning labels only determined selection; no model passed.

**Conditional stages:** `confirm` returned `unavailable` with
`confirmation_opened=false`; `evaluate` returned `unavailable`. No reserved
confirmation outcome was scored and no new frozen-predictor inference was run.
This is a failed measurement gate, not a predictor-physics finding. The local
experiment ledger recorded 26.52 seconds for the complete fit stage, below the
two-hour cap. A reporting-only `python -m jepa_action_effects.delta_breakdown`
run added cohort-by-regime endpoint and effect metrics for the same 60 tuning
families, without changing selection or opening confirmation.

**Validation and evidence:** The focused DELTA fixtures passed 10/10. Final
`make check` passed Ruff, formatting, and doctor; `make test` passed 60/60 with
11 non-failing Gymnasium warnings. `UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python uv lock --check` resolved 118 packages;
`git diff --check` passed. An initial focused fixture expected less ridge
shrinkage than alpha 0.01 permits; correcting its numerical tolerance made the
fixture reflect the prescribed regularization. Detailed ignored artifacts are
under `outputs/delta001/`; `docs/manifests/delta001-runtime.json` records hashes,
timing, storage, and checks.

**Decision and limitations:** Close this bounded physical-measurement recovery
route and keep main expansion paused. Better performance than zero displacement
shows usable signal but does not satisfy the physical-error gate. Probe failure
does not prove missing block information or physics failure. Historical
training-simulator identity remains unproven; no main locked-test data was used.

**Next action:** Finish LIT-001 independently. Any future latent-space question
needs a new protocol with explicit action and baseline controls before analysis.

---

## 2026-09-27 — DELTA-001 — Next-step plan recorded

**Actor:** Codex, responding to Omer's request to plan the next steps.

**Changes:** Added `docs/DELTA-001_PLAN.md`; marked DELTA-001 TODO in TASKS and
recorded proposed scope in D011. The ordered measurement test starts with the
saved ridge's output differences, then a direct displacement ridge, and only if
needed a bounded zero-anchored nonlinear readout. It reuses the existing
240/60/60 fit/tuning/reserved-confirmation partitions. Gates cover displacement,
paired effects, and meaningful-effect examples; one-time confirmation precedes
any conditional exploratory frozen-predictor comparison. True-start-position
regret is explicitly labeled oracle-initialized. Main expansion remains paused.

**Evidence and checks:** Read TASKS, the latest READOUT-001 LOGBOOK entry, D009–D010,
the saved READOUT-001 report, relevant RESEARCH_PLAN measurement/evaluation sections,
and existing readout/evaluator interfaces. Ran `git diff --check` successfully.
Documentation-only planning: no Python/environment change, experiment, new
collection, training, or confirmation evaluation was performed; tests were not
rerun.

**Next action:** Implement the staged DELTA-001 plan, seal the concrete configuration
before displacement scoring, and enforce its cumulative two-hour experiment cap.
This entry records planned work, not new scientific evidence.

---

## 2026-09-27 — READOUT-001 — Measurement audit and bounded recovery

**Actor:** Codex, implementing Omer's approved READOUT-001 plan.

**Intent and changes:** Added staged audit, collection, fitting, and reporting under
`python -m jepa_action_effects.readout_diagnostic`. Preserved all PILOT-001 files.
Replayed a predetermined 12-family subset balanced across cohorts and splits,
checked exact RGB replay, candidate order, native-step labels and circular block
angles, and made actual-polygon block/agent overlays. Reproduced the original
ridge gate from its saved model and latents. Identical-frame deduplication showed
14 agent-label conflicts but no block-position conflicts. Manual rollout and
projected CLS encoding matched the frozen inference path exactly. A 32-frame fit
subset was recovered by least squares and memorized by the nonlinear probe.

**Measurement correction:** The original ridge fit used family-equal regression
weights but unweighted feature statistics; nonlinear target statistics similarly
counted repeated frames unequally. Recovery fits use fit-only family-weighted
feature and target statistics. The original and before-correction artifacts remain
separate under `outputs/readout001/`. Corrected pilot reporting marks physical
competence **unassessable** when its ridge readout fails. A passing nonlinear
diagnostic cannot validate endpoints decoded by ridge. No pilot results were
rewritten.

**Commands and results:** After `make repair-macos-pth`, ran
`UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked --no-sync python -m jepa_action_effects.readout_diagnostic {audit,collect,fit,report}`
as four separate stage commands. `audit` passed in 4.40 seconds;
`collect` accepted 200 challenge and 100 ordinary families in exactly 200/100
attempts, with no reset-seed overlap against the pilot. The collection run took
47.78 seconds and assigned 180/60/60 new fit/tuning/reserved-confirmation
families. The nested fit sizes were 60, 120, and 240, all using the same fresh
60-family tuning set. Corrected projected ridge tuning mean/p95 errors were
19.96/47.70, 15.68/38.97, and 11.68/29.02, respectively, versus gates of
1.28/2.56. Three 3,000-epoch-capped nonlinear seeds at each size also failed;
at 240 families their mean errors were approximately 10–11. Unprojected CLS
ridge improved to 8.60 mean at 240 but still failed and cannot decode projected
predictor outputs. No projected readout was selected; the 60 confirmation labels
were not opened. Fit progress and curves are saved at cell boundaries. The full
experiment remained well under the two-hour CPU cap.

**Verification:** `make check` passed Ruff, formatting, and doctor;
`make test` passed 50/50 tests with 11 non-failing Gymnasium warnings;
`UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv lock --check`
resolved all 118 locked packages; `git diff --check` passed. One interim `make test`
found two collector fixtures with old hard-coded split defaults; those fixtures
were corrected for configurable counts, and the full suite subsequently passed.
One interim `make check` exposed formatting/line-length issues; they were fixed.
The signed `fit` stage was repeated after the family-weighted correction and
resume checks; no confirmation labels were opened on any run. See
`docs/manifests/readout001-runtime.json` for hashes and measured run times.

**Decision and limitations:** Narrow or revise the measurement approach; do not
expand the main study. The audit found no block alignment defect sufficient to
explain the error, and additional data and training did not meet the physical
gate. This does not prove block information is absent from the representation.
Historical training-simulator identity remains unproven. The reserved
readout-confirmation families and all main-study locked-test labels remain closed.

**Next action / handoff:** Scope a revised block-state measurement strategy before
any further physical interpretation or main collection. LIT-001 remains separate.

---

## 2026-09-27 — PILOT-001 — End-to-end feasibility pilot completed

**Actor:** Codex, implementing Omer's approved bounded pilot plan.

**Intent and changes:** Implemented deterministic, resumable collection; actual-shape
geometry proposals; ordinary reference rollouts; family-intact splits and fixed goals;
the simulator-only horizon gate; manifest-checked latent caches; family-weighted ridge
and bounded nonlinear readouts; oracle, persistence, real-frame, and frozen-JEPA
evaluation; 2,000-resample family bootstrap intervals; paired contrasts; diagnostics;
and an automatic proceed/narrow/stop report. Extended replay to four native steps and
added regression coverage. D009 records the exploratory protocol and stop decision.

**Commands and results:** A 12-family smoke accepted all proposals, found 58.3%
meaningful effects, and passed reversed-order replay. The full `pilot collect` run
accepted 80/80 challenge and 40/40 ordinary attempts, assigned 60/30/30 families,
found 88/120 (73.3%) meaningful at two steps, retained the two-step horizon, and
passed duplicate/reversed replay on a fixed balanced 12-family subset. Cold frozen
evaluation encoded all 120 families in 17.39 seconds, predicted them in 5.41 seconds,
and completed in 26.76 seconds. `make check`, `make test`, `uv lock --check`, and
`git diff --check` passed; pytest passed 39/39 with ten non-failing Gymnasium casting
warnings. Tests cover deterministic proposals, bounds, four-step rollout alignment,
attempt caps/rejections, interrupted-run recovery, family weighting, contact-regime
gates, and manifest/cache mismatch rejection.

**Evidence:** `docs/manifests/pilot001-runtime.json` records hashes and measured
results. Detailed ignored artifacts are in `outputs/pilot001/`, including 120 family
recipes/traces, sealed split and goal membership, latent caches, fitted readouts,
four diagnostic GIFs, six aggregate contact overlays, two summary plots, raw family
evaluations, and the final report. Visual inspection confirmed that the overlays and
plots render correctly. Total storage was 9,702,358 bytes. The 900-family projection
with 50% contingency was 572.64 seconds.

**Decision or deviation:** PILOT-001 is DONE with a **STOP expansion** recommendation.
Replay, collection, central coverage, horizon, latent ordinary-transition, and runtime
gates passed. The alpha-0.1 ridge block readout failed: validation mean error was
18.42 and p95 was 42.12, versus gates of 1.28 and 2.56. All three fixed nonlinear
seeds also failed. Decoded JEPA ordinary endpoint error was 24.94 versus persistence
at 2.43, but this physical comparison is inconclusive because even real-future-frame
readout error was 20.68. No locked test data was inspected.

**Limitations:** This run supports a representation/readout limitation for the chosen
measurement setup, not a checkpoint physical-reasoning failure. Challenge families
were simulator-selected and do not estimate natural prevalence. Historical simulator
identity remains unproven, and one checkpoint gives no training-seed variability.

**Next action / handoff:** Do not begin FREEZE-001 or main collection. Any attempt to
continue needs a separately scoped measurement strategy that resolves the real-frame
readout failure without using locked test labels. LIT-001 can proceed independently.

---

## 2026-09-26 — EVAL-001 — Physical-effect evaluator completed

**Actor:** Codex, implementing Omer's approved EVAL-001 plan.

**Intent and changes:** Implemented typed, checkpoint-independent family inputs;
endpoint, paired-effect, contact-stratum, motion, and finite-candidate regret
metrics; family-first aggregation; explicit failure/exclusion handling; and a
SIM-001 replay adapter. Added an audit covering oracle, persistence, doubled,
reversed, spurious-motion, and positive-regret fixtures plus a real two-candidate
replay integration. D008 records the provisional metric conventions. Corrected
the interface contract's state order from source and measured integration evidence:
`[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]`.

**Commands and results:** Focused evaluator/replay tests passed 8/8. The audit
passed in 0.048 seconds: effect error was 0 for oracle, 10 for persistence and
doubled effect, 20 with direction error pi for reversed effect, and 10 for a
spurious effect on identical true endpoints; the hand-calculated regret was 10.
Its replay family was eligible, with contact on `push` and none on `hold`.
`make check` passed Ruff, formatting, doctor, and numerical checks. `make test`
passed 33/33 in 5.00 seconds; nine existing Gymnasium array-casting warnings were
non-failing. `uv lock --check` and `git diff --check` passed. No dependency changed.
The initial direct uv commands failed because they targeted the sandbox-inaccessible
global cache; the repository-local locked-cache workflow passed. Integration first
failed because the previous interface prose gave the wrong state indices, which the
pinned `_get_obs` source and measured values resolved.

**Evidence:** `docs/manifests/eval001-runtime.json` records source and lock hashes,
runtime, conventions, and limitations. The detailed audit report remains ignored at
`outputs/eval001/report.json`. Regression coverage includes all contact strata,
threshold boundaries, zero vectors, transient return-to-start motion, wall-only
collisions, mismatched horizons, stable ties/permutations, prediction failures,
unequal family sizes, and empty subsets.

**Decision or deviation:** EVAL-001 is DONE. D008 conventions and the plan's
`delta=5.12` and two-step horizon remain provisional; this task validates arithmetic
and integration, not their physical relevance. Prediction failures never redefine
truth-based eligibility or candidate selection.

**Limitations:** Mock predictors are not checkpoint evidence. Candidate/goal
generation, readout fitting and validation, natural contact coverage, confidence
intervals, and physical-threshold/horizon selection remain downstream.

**Next action / handoff:** PILOT-001 should generate 120 simulator-selected families,
connect frozen predictions through development-only readouts, and use these metric
semantics unchanged while assessing coverage and the provisional protocol choices.

---

## 2026-09-26 — SIM-001 — Replay branching and substep contacts completed

**Actor:** Codex, implementing Omer's approved SIM-001 plan.

**Intent and changes:** Added a typed simulator-only replay API that validates and
copies reset/warm-up/candidate recipes, performs a complete seeded reset and identical
warm-up before every candidate, returns observed history separately from future
frames, and explicitly records incomplete terminated branches. Added a pinned PushT
subclass that preserves the upstream collision callback while recording reset,
warm-up, and candidate physics steps; observable agent/block dynamics; stable shape
identities; contact geometry; wall events; and agent–block labels across both T
polygons. Added a reproducible audit, five deterministic collision/dynamics fixtures,
annotated examples kept separate from model-compatible frames, and nine regression
tests. D007 records the runtime-specific replay threshold.

**Commands and results:**

- `python -m jepa_action_effects.simulator_audit`: passed in 23.53 seconds. Seeds
  0–15, six fixed candidates, duplicate execution, reverse order, and a permutation
  from ordering seed 20260926 all passed. All candidates completed. Maximum numeric
  replay discrepancy, positional `epsilon_sim`, and instrumented-versus-base
  discrepancy were exactly 0 under `atol=1e-10, rtol=0`.
- Direct-push, sustained-contact, near-miss, block-wall-only, and moving/rotating
  fixtures passed. Direct push had 99 agent–block contact substeps; sustained contact
  had 100 and covered both T polygons; near miss had zero; block-wall-only had 100
  wall-contact and zero agent–block-contact substeps.
- Inspected all four generated annotated examples. Contact overlays appear on the
  direct and sustained fixtures; near-miss and wall-only model/annotation hashes are
  identical because pair-specific contact is absent.
- `make check`: passed Ruff lint/format, doctor numerical checks, and the updated
  study-readiness status. `make test`: passed 25/25 tests in 4.18 seconds. Seven
  existing Gymnasium array-casting warnings remain non-failing. `uv lock --check`
  and `git diff --check`: passed. No dependency changed.
- The initial early-termination regression used a goal equal to the reset state and
  correctly ended during warm-up. It was replaced with candidate-phase termination
  injection so the test addresses the intended stop/mark behavior.

**Evidence:** `docs/manifests/sim001-runtime.json` records code/runtime provenance,
configuration, gates, artifact hashes, and limitations. The detailed report, traces,
raw model-compatible frames, and annotated diagnostics remain ignored under
`outputs/sim001/`. Replay/contact semantics are documented in
`docs/INTERFACE_CONTRACT.md`.

**Decision or deviation:** SIM-001 is DONE. D007 uses the planned strict numerical
diagnostic without relaxation and keeps it separate from the provisional physical
effect tolerance. Full-state cloning was not implemented because replay passed and
Pymunk solver caches are not exposed. Candidate proposal generation remains scoped
to PILOT-001.

**Limitations:** Results establish deterministic replay and label behavior only on
the measured locked macOS runtime. They do not prove cross-platform bitwise
determinism, historical training-simulator identity, natural contact prevalence,
checkpoint physical competence, or a frozen study horizon. Observable body dynamics
are validated; hidden solver caches are neither captured nor claimed equivalent.

**Next action / handoff:** Implement EVAL-001 synthetic physical-effect and regret
fixtures. Then combine its metrics with SIM-001 families and MODEL-001 inference in
PILOT-001, selecting contact/near-miss cases from simulator outcomes only.

---

## 2026-09-26 — MODEL-001 — Frozen CPU inference and latent cache completed

**Actor:** Codex, at Omer's request.

**Intent and changes:** Implemented the source-audited LeWM architecture mapping,
strict checkpoint load, evaluation action normalization, leakage-free one/two-step
prediction API, real-frame encoding, content-addressed latent cache, archive
preparation/statistics command, and real smoke/benchmark command. Extended the
ordinary simulator transition to two future blocks. Added Transformers to the lock
and repaired Make targets so each package-importing command handles the known macOS
hidden `.pth` behavior. D006 freezes evaluation scaling for inference; the study
horizon remains provisional.

**Commands and results:**

- Resumable `curl --continue-at -` download completed for the pinned 13.1 GB archive.
  `python -m jepa_action_effects.action_stats` verified published SHA-256
  `7cfbd6d...212f318`, atomically extracted the 46,300,921,856-byte HDF5 file, and
  recorded HDF5 SHA-256 `b6ebd9ac...f8d4fb`.
- Archive inspection found 2,336,736 rows in 18,685 contiguous episodes, pixel shape
  `(2336736,224,224,3)`, action shape `(2336736,2)`, and zero NaN action rows. No
  simulator provenance attributes exist. Evaluation mean is
  `[-0.0078125644, 0.0068606872]`; population SD is
  `[0.2084674428, 0.2067486264]`. Its difference from training sample SD is below
  `4.1e-8` per coordinate.
- `python -m jepa_action_effects.model_smoke`: strict-loaded all 303 state entries
  with no missing/unexpected keys; one/two-step manual rollout agreement had maximum
  absolute difference 0; future-target isolation, candidate ordering, finite output,
  frozen parameters, and unchanged BatchNorm checks passed. Content-addressed real
  and predicted caches round-tripped with manifest validation.
- Seed-42 simulator ordinary transition: two-step latent MSE 0.008529 versus latent
  persistence 0.018635. The first 32 eligible archive episodes, selected at start 0
  before inference, gave MSE 0.004975 versus persistence 0.354868. The latter is a
  training-data diagnostic, not generalization evidence.
- Six-candidate K=2 CPU rollout took 0.0851 seconds; encoding two targets took 0.0221
  seconds. A 120-family rollout-only estimate with 50% contingency is 15.3 seconds.
- `make setup && make check && make test && uv lock --check && git diff --check`:
  passed. Pytest passed 16/16 in 3.70 seconds; two existing non-failing Gymnasium
  array-casting warnings remain.
- Initial combined Make verification exposed that Python restores the hidden flag on
  the editable `.pth` file after a process exits. `--no-sync` alone was insufficient;
  adding the narrow repair immediately before doctor/test commands made the complete
  workflow pass reproducibly.

**Evidence:** `docs/manifests/model001-runtime.json`, `outputs/model001/report.json`,
and `outputs/model001/action-stats.json`. The archive, extracted data, checkpoint,
cache, and detailed output stay ignored. The manifest records hashes, runtime,
scalers, diagnostics, throughput, and limitations.

**Decision or deviation:** MODEL-001 is DONE. Evaluation uses published population-
SD scaling chosen before predictions (D006). The explicit March class mapping was
preferred over changing the pinned simulator package. CPU float32 is the validated
runtime; no accelerator comparison was needed after measured local throughput.

**Limitations:** The dataset has no exact simulator commit or collection manifest.
The 32 archive windows are checkpoint training data. Latent errors do not establish
block-state readability, contact-conditioned physical reasoning, or generalization.
No readout, contact label, MPS/CUDA run, or main-study family was produced.

**Next action / handoff:** Complete SIM-001 replay and pair-specific substep contact
validation, then combine its families with MODEL-001 inference and EVAL-001 metrics
in PILOT-001. Fit readouts only on real encoded development observations.

---

## 2026-09-26 — ENV-001 — Checkpoint and simulator interface audit completed

**Actor:** Codex, continuing Omer's ENV-001 request.

**Intent and changes:** Audited `quentinll/lewm-pusht` and checkpoint-era LeWM /
stable-worldmodel sources before model inference. Pinned stable-worldmodel commit
`44c45bd1f774995aac0cc2b41444ebea24c1d5ad` in `pyproject.toml` / `uv.lock`, added
the minimal PushT runtime, downloaded the ignored 72 MB weights-only checkpoint,
and verified its published SHA-256. Replaced the pending interface template with an
evidence-linked contract covering licenses, actions, physics/control timing, image
preprocessing, history/action grouping, latent shapes, rollout alignment, reset
state, termination, and contact limitations. Added source/runtime manifests, a
repeatable audit command, and three behavioral tests. D005 records why this is a
provenance-based simulator candidate rather than a proven training-time match.

**Commands and results:**

- Hub/API and immutable archive inspection: checkpoint revision
  `22b330c28c27ead4bfd1888615af1340e3fe9052`; training-data revision
  `655cd446b9929369d7d406001da85c15d1457850`; model source
  `ca231f9f9d9ab041034b6d05e90b6e04bd6cff82`; simulator source
  `44c45bd1f774995aac0cc2b41444ebea24c1d5ad`.
- `shasum -a 256 checkpoints/lewm-pusht/*`: config
  `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`;
  weights `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`,
  matching Hub LFS metadata. The audit loaded tensors with `weights_only=True`:
  303 entries, 18,042,672 tensor/buffer elements, all finite.
- Initial `make setup` final verification: failed inside the restricted sandbox
  because DNS could not resolve PyPI while rebuilding the local package. Authorized
  retry passed: 112 locked packages resolved and the editable package rebuilt.
- `make check`: passed Ruff lint, format check, numerical ridge fit, and headless
  plotting. Doctor now reports the simulator audit passed while inference remains
  unvalidated.
- `make test`: passed 3/3 tests in 7.10 seconds. Two non-failing Gymnasium warnings
  reported array conversion during reset/step.
- `python -m jepa_action_effects.interface_audit`: passed and wrote the report plus
  a four-frame seed-42 transition image. History frames at controls 0/5/10 have
  shape `(1,3,3,224,224)`; the separately held future target at control 15 has
  shape `(1,1,3,224,224)`. Each control uses 10 physics substeps; five controls form
  one 0.5-second native model step. No checkpoint inference was performed.
- `uv lock --check` and `git diff --check`: passed. Final `uv.lock` SHA-256 is
  `dbf2f3de2876e3f575d0abd7341825a349ce1b2e2d86d6f3e11bda618ac89691`.
- OpenCV 4.14 initially emitted duplicate SDL class warnings with pygame on macOS.
  Pinning `opencv-python-headless==4.11.0.86` removed them.

**Evidence:** `docs/INTERFACE_CONTRACT.md`, D005 in `docs/DECISIONS.md`,
`docs/manifests/env001-sources.json`, and `docs/manifests/env001-runtime.json`.
Generated checkpoint bytes and `outputs/env001/` remain ignored. The runtime
manifest records base Git HEAD `706365b33c0ca208dd1362a58017833c7c9eec61`, dirty
state, exact package versions, seed, states, action blocks, and frame hashes.

**Decision or deviation:** ENV-001 is DONE as an interface audit. It does not claim
scientific compatibility. The 13.1 GB training archive was deliberately deferred to
MODEL-001: its action column is required to recover the checkpoint's unpublished
normalization statistics. Identity scaling and statistics from new study trajectories
are prohibited substitutes. The primary two-step horizon still remains a protocol
default rather than a frozen or validated choice.

**Limitations:** The checkpoint has no training run manifest, simulator commit,
dependency lock, or numeric normalization statistics. Its Hub config references
newer class paths than the checkpoint-era runtime. No strict model construction,
prediction, competence result, GPU/MPS check, deterministic replay, branch-order
test, pair-specific substep contact label, or training-overlap claim was completed.
The smoke transition moves the agent without block contact and is interface evidence
only.

**Next action / handoff:** MODEL-001 should obtain and hash the pinned training
archive, recover both training/evaluation action scalers, resolve the config/class
mapping, strictly load every weight, and perform leakage-free ordinary-transition
inference. SIM-001 can proceed independently with seeded warm-up replay, full
dynamic-state comparisons, branch-order checks, and agent–block-only collision
instrumentation at each physics substep.

---

## 2026-09-25 — SETUP-001 — Shared GitHub remote

**Actor:** Codex, at Omer's request.

**Intent and changes:** Configured `origin` as
`https://github.com/OmerTorres/jepa-action-effects.git`. The GitHub repository
already contained an independent initial commit with an MIT `LICENSE`; fetched and
merged that history without overwriting either side, then prepared local `main` for
an upstream-tracking push.

**Commands and results:** The first `git push -u origin main` was safely rejected
as non-fast-forward. `git fetch origin` showed the remote commit contained only
`LICENSE`. `git merge origin/main --allow-unrelated-histories --no-edit` completed
without conflicts. No generated data, environments, secrets, or weights were added.

**Limitations:** Repository access for Inbal must be configured separately if the
repository is private; her GitHub username was not available during this step.

**Next action / handoff:** Push the merged `main`, verify upstream tracking, then
invite Inbal when her GitHub username is known.

---

## 2026-09-25 — SETUP-001 — Development foundation completed

**Actor:** Codex, at Omer's request.

**Intent and changes:** Replaced the provisional Python 3.11 environment with
Python 3.10.21, matching LeWM's documented Python series. Regenerated `uv.lock`,
recreated `.venv` from the repository-local Python/cache directories, added a
targeted macOS `.pth` hidden-flag repair, validated the research skill, and marked
SETUP-001 done. No simulator, checkpoint, PyTorch, or model dependency was added.

**Commands and results:**

- `uv lock`: passed; resolved 54 packages for Python 3.10.
- `make setup`: the sandboxed attempt failed because PyPI DNS was unavailable;
  the authorized network retry passed and installed the locked environment.
- `UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv lock --check`:
  passed; lockfile SHA-256
  `b794e5c42b0d9324046d5d51e554db2b4ca9acdc73530396e48070aeae28161a`.
- Clean import: passed; `jepa_action_effects` loaded from `src/` through the
  editable installation without `PYTHONPATH`. Both `.pth` files had no hidden flag.
- `make check`: passed Ruff lint, Ruff format check, ridge-fit smoke check, and
  headless PNG generation under Python 3.10.21.
- `quick_validate.py .agents/skills/jepa-study`: passed (`Skill is valid!`).
- `make test`: pytest exit 5 with zero collected tests. This is expected because
  scientific code and behavioral tests do not exist yet; it is not a passing suite.
- `git check-ignore -v`: confirmed exclusions for `.venv`, `.cache`, secrets,
  datasets, checkpoints, outputs, WandB artifacts, and common model-weight formats.

**Evidence:** `docs/environment-local.json` records the measured local runtime.
The supplied plan remains unchanged in `docs/RESEARCH_PLAN.md`. Git was still on
unborn local `main` immediately before the initial foundation commit.

**Decision or deviation:** D004 supersedes the provisional Python choice in D001.
The scientific protocol remains unfrozen and untested.

**Limitations:** This validates only the base developer environment. It does not
establish checkpoint loading, PushT compatibility, deterministic branching,
contact labels, model quality, or any research claim.

**Next action / handoff:** Begin ENV-001 by inspecting immutable source and
checkpoint revisions before downloading weights or selecting model/simulator
dependencies. Run LIT-001 alongside it; EVAL-001 may begin with synthetic fixtures.

---

## 2026-09-25 — SETUP-001 — Development foundation

**Actor:** Codex, at Omer's request.

**Starting state:** Empty project folder; no Git repository, implementation, data,
or checkpoints. Host: Apple Silicon arm64. Existing tools: uv 0.12.5; system Python
3.14.3 (also 3.13.0 and macOS 3.9.6). No ancestor AGENTS.md found in the inspected path.

**Work:** Preserved supplied research plan; scaffolded package, locked-environment
workflow, project documentation, task board, interface audit, and research skill.
Kept simulator/model dependencies pending compatibility evidence.

**Verification:** In progress; append final setup measurements below.

**Limitation:** Scientific claims and cited precedents in the supplied plan are
user-provided context, not independently verified by this setup session.

**Next:** ENV-001 interface audit and LIT-001 contribution/metric verification.

---

## 2026-09-26 — EVAL-001 — Implementation planning

**Actor:** Codex, at Omer's request to plan EVAL-001.

**Intent and changes:** Added `docs/EVAL-001_PLAN.md` and linked it from the task
handoff. The plan defines evaluator inputs, replay adaptation, effect/endpoint
metrics, independent contact/motion labels, family aggregation, candidate regret,
analytic fixtures, and completion evidence. EVAL-001 remains TODO.

**Commands and results:** Read TASKS.md, the latest SIM-001 logbook entry, the
jepa-study skill, DECISIONS.md, relevant RESEARCH_PLAN.md sections, the interface
contract, replay API, and Makefile. `git status --short` showed a clean starting
tree. `git diff --check` passed after the documentation changes. No Python or
environment changes; scientific tests and audit runs are planned, not performed.

**Evidence and limitations:** `docs/EVAL-001_PLAN.md` is a design artifact only.
Pair selection, tie rules, and threshold boundaries are explicitly proposed
conventions; no protocol was frozen and no model-performance evidence was produced.

**Next action / handoff:** Implement the pure evaluator and analytic fixtures,
then its SIM-001 adapter and reproducible audit; connect to PILOT-001 afterward.

---

## 2026-09-29 — REVIEW-BRANCH-002 — Review of claude/youthful-clarke-akm28u

**Actor:** Codex, at Omer's request for a branch-quality and scientific review.

**Scope and changes:** Reviewed `59cfa74...181ca55` (13 commits, 49 files).
Added `BRANCH_REVIEW_2026-09-29.md` and the task-board review row. No production
source, historical result, or frozen protocol was edited. Review scripts are kept
under ignored `outputs/branch_review_2026-09-29/`.

**Commands and results:**

- Initial sandboxed `git ls-remote origin refs/heads/main
  refs/heads/claude/youthful-clarke-akm28u` failed DNS resolution. Authorized retry
  passed and verified GitHub tips `59cfa746272519aa7ab2e73d91b5b16861085a4f`
  and `181ca5586c9bceb25146ada24e7333cb9167b30e` respectively.
- `make check && make test`: passed; Ruff and format passed, doctor passed on
  Python 3.10.21, **134 tests passed, no skips**, 19 Gymnasium casting warnings,
  pytest duration 13.98 seconds.
- `.venv/bin/python /tmp/jepa_branch_review_checks.py`: reproduced acceptance of
  changed post-seal config/goals and stale cached rows using isolated mocks;
  endpoint-bias cancellation; acceptance of archive-faithfulness CI [2,3]; and
  a synthetic false magnitude finding with perfect physical predictions and
  dose-dependent real-frame readout bias. These are mock counterexamples.
- `.venv/bin/python /tmp/jepa_branch_review_recompute.py`: PHYS-002, SHIFT-002
  and H3-PLAN-002 confirmation summaries reproduced from saved rows with maximum
  numerical difference **0.0**. Exploratory per-dose real-frame slopes remained
  within the existing readout tolerance band; this does not overturn the trend.
- `.venv/bin/python /tmp/jepa_branch_review_followup.py`: the archive audit's
  600 windows cover 584 episodes; 335 moved windows cover 331 episodes. True
  absolute PHYS-002 reserved challenge endpoint means are 17.895 predicted /
  3.613 real-frame, rather than the displacement metrics labeled as endpoints.
- Separate read-only hash comparison: current source/config/audit/development/
  collection/probe hashes matched the completed seals; development/confirmation
  inference fingerprints matched; 150 PHYS, 360 SHIFT and 150 planning family
  files matched their manifests (660/660).
- Agent-only prediction-error fixture with exact block predictions changed the
  matched-goal choice and added regret, showing the H3 cost does not isolate block
  prediction error.
- `zsh -f -c 'RUN="printf example"; $RUN.distribution_audit keys'`: expected exit
  127, reproducing the Mac runbook's scalar-command splitting defect.

**Evidence and limitations:** The report contains source locations, observed
impact and proposed fixes. All three main confirmation aggregations reproduce;
no evidence of changed inputs was found in the audited completed artifacts. The
review did not retrain probes or rerun every collection/inference stage. Additional
per-dose checks used already opened families and are exploratory only.

**Decision or deviation:** No new scientific decision or protocol amendment.
The strongest observed conclusions survive these checks, but the H3 narrative
overstates its confidence intervals and the experiment runner does not enforce
all the hashes it records. Corrected metrics/uncertainty must be recorded as
corrections, preserving historical evidence.

**Next action / handoff:** Address review findings before further confirmation;
correct the course-report narrative, add the missing absolute endpoint control,
repair provenance/cache validation in a new protocol version, and complete an
independent rerun subset. Review recommendations are not implemented fixes.

---

## 2026-09-29 — REVIEW-FIX-003 — Address REVIEW-BRANCH-003

**Actor:** Claude (Claude Code desktop, Omer's Mac), after Omer's audit
(REVIEW_FOLLOWUP_AUDIT.md, committed as 7e7cd78).

**Changes (unsealed files only):**

- `followup_metrics.decompose`: zero predicted vectors are excluded from the angle
  statistics; `undefined_angle_count` is added; the angle is `None` when undefined.
- `provenance.validate_run`: checks the probe artifact bytes, checks cached-record
  fingerprints, and gains strict (default) and legacy modes;
  `check_completed_runs` uses legacy mode.
- Tests:
  - a zero-vector fixture;
  - modified probe bytes;
  - a foreign record fingerprint;
  - strict vs legacy handling of missing metadata.
- Correction section in REVIEW_FOLLOWUP_RESULT.md (ratio direction); D027; TASKS
  (REVIEW-FIX-002 described as a foundation; REVIEW-FIX-003 row); DEVELOPMENT
  (strict mode).

**Commands and results:**

- `python -m jepa_action_effects.review_followup phys|shift|archive|report`:
  regenerated manifest identical to the previous one on every numeric field (max
  difference 0.0); 0 undefined angles.
- `python -m jepa_action_effects.provenance` (legacy): PHYS-002, SHIFT-002 and
  H3-PLAN-002 valid; the probe artifact check passed. Missing record fingerprints
  (all runs) and input hashes (SHIFT) are reported as not recorded.
- `make check` and `make test`: see the commit.
- Completed-run `_sources()` still equal their seals.

**Next action:** the course report (REPORT-001).

## 2026-09-29 — REVIEW-BRANCH-003 — Audit of REVIEW-FIX-001/002

**Actor:** Codex, following Omer's update that the review follow-up was implemented.

**Scope and changes:** Reviewed local commit `e241f54`, D026, the new follow-up
report, metrics, provenance/gate helpers and tests. Added `REVIEW_FOLLOWUP_AUDIT.md`
and a task-board review row. No production source or sealed artifact changed.

**Commands/results:**

- `make check && make test`: passed; **148 tests, no skips**, 19 Gymnasium
  input-casting warnings; 12.21 seconds.
- `UV_CACHE_DIR=.cache/uv uv run --locked --no-sync python -` with temporary mock
  runs from `tests/test_provenance.py`: changed selected probe bytes and a changed
  cached-row fingerprint were each accepted by `validate_run`. No real study
  artifact was mutated.
- Same interpreter, `check_completed_runs()`: actual PHYS-002, SHIFT-002 and
  H3-PLAN-002 return valid under the validator's implemented checks.
- Same interpreter, `decompose([[0,0]], [[10,0]])`: reports 0-degree mean and median
  angle despite undefined predicted direction.
- Applied `readout_gate_per_cell` to all reserved dose × structure CIs in
  `docs/manifests/review-followup-runtime.json`: passed, no failing cells.
- Inspected saved predictor/readout slopes: the follow-up's explanation reverses
  the denominator's effect. Falling challenge readout slope opposes the decrease
  in pred/real; rising archive readout slope contributes to it. Predictor slopes
  themselves still decline (challenge constant 0.954 → 0.739; archive constant
  0.964 → 0.855).

**Evidence/limitations:** See REVIEW_FOLLOWUP_AUDIT.md for source locations and
impact. Most original concerns are addressed. The rerun implementation and saved
zero differences were inspected; the full checkpoint rerun was not repeated here.
The zero-vector counterexample does not establish an effect on the saved angles.
The existing probe loader independently verifies probe bytes; the new standalone
validator does not yet do so. No new protocol or confirmed claim was introduced.

**Next:** correct the ratio explanation before reporting; complete strict probe/
cache provenance checks and zero-vector handling before future protocol reuse.

---

## 2026-09-29 — REVIEW-BRANCH-004 — Verification of D027 fixes

**Actor:** Codex, at Omer's request to review the latest fixes again.

**Scope:** Reviewed `7e7cd78..de678d9`, including code, tests, D027, result
corrections and regenerated manifests. Added a verification section to
REVIEW_FOLLOWUP_AUDIT.md and a task-board row; no production source or result changed.

**Commands/results:**

- `make check && make test`: passed Ruff, format and doctor; **151 tests, no
  skips**, 19 existing Gymnasium warnings, pytest duration 11.52 seconds.
- `UV_CACHE_DIR=.cache/uv uv run --locked --no-sync python -`: compared the current
  follow-up manifest with `git show 7e7cd78:docs/manifests/review-followup-runtime.json`.
  No existing numerical field changed. The metrics-source hash changed and 31
  undefined-angle counts were added, all zero.
- Same interpreter, read-only `check_completed_runs()`: PHYS-002, SHIFT-002 and
  H3-PLAN-002 passed implemented legacy checks. Missing historical fingerprints
  (150/240/150 records) and SHIFT input hashes (240) remain explicitly reported.
- `git diff 7e7cd78..HEAD --check`: passed.

**Outcome/limitations:** All three REVIEW-BRANCH-003 findings are addressed. No new
actionable issue was found in the patch. Probe-byte, foreign-fingerprint and
strict/legacy mutation fixtures pass, as do zero/mixed-vector direction fixtures.
The full experimental pipeline was not rerun. Strict validator integration into
the actual confirmation/resume path is still a prerequisite for a future runner,
as already documented; this review does not certify that deferred integration.

**Next:** proceed with REPORT-001 using corrected claims and the existing evidence.
No new scientific decision or protocol amendment.

---

## 2026-09-29 — REPORT-001 — Course-aligned delivery plan

**Actor:** Codex, continuing Omer's request to plan REPORT-001 with the supplied
course requirements and chosen English LaTeX/PDF report and presentation package.

**Changes:** Added REPORT-001_PLAN.md with requirements mapped to the handout's
pages, training-scope assessment, a conditional dynamics-training proposal,
claim-to-evidence work, figure/report/slide deliverables, reproduction handoff and
acceptance checks. Updated REPORT-001's task dependencies to the completed bounded
cohorts and latest review, leaving the original main-study expansion paused.
REPORT-001 remains TODO: planning is not implementation.

**Commands/results:**

- `git status --short`: existing modifications to LOGBOOK, TASKS and
  REVIEW_FOLLOWUP_AUDIT from REVIEW-BRANCH-004 preserved.
- Bundled Python `pypdf.PdfReader(...).pages` and `page.extract_text()` on Omer's
  `הנחיות לפרויקט בקורס למידה עמוקה 2026ב.pdf`: all four pages read. Confirmed
  substantial training, PyTorch, five-page report, 15-minute presentation,
  individual contributions and reproduction requirements.
- Read TASKS, latest LOGBOOK entries, jepa-study instructions, relevant
  RESEARCH_PLAN sections, corrected follow-up/H3 results and Makefile. No checkpoint
  inference, experimental analysis, training or artifact regeneration performed.
- `git diff --check`: passed after the planning changes. No Python/environment
  changes; no tests rerun for this documentation-only plan.

**Evidence/limitations:** Anchored to reviewed HEAD `de678d9` with existing local
review documentation changes. The report must use corrected D026/D027 claims and
label opened-cohort follow-up analyses exploratory. The supplied handout requires
substantial model training but does not resolve whether the existing trained
readout is sufficient. Instructor clarification, author contributions and deadline
remain pending; no acceptance, teammate reproduction or new training is claimed.

**Next action:** prepare the instructor clarification for Omer to send and build
the claim ledger and report figures. Any additional training requires its own
scope, development budget and frozen protocol before a fresh confirmation.
No scientific decision or protocol amendment was made here.

---

## 2026-09-30 — TRAIN-REVIEW-001 — Implementation-plan review before handoff

**Actor/scope:** Codex, after Omer explicitly selected review rather than execution
of the supplied TRAIN-001 implementation plan.

**Changes:** Added TRAIN-001_IMPLEMENTATION_PLAN_REVIEW.md and a completed review
task. The review preserves the agreed measurement-failure policy and identifies
pre-execution ambiguities in candidate rejection, image units, partial confirmation
validation, trained-model fingerprints, runtime enforcement and gate/claim rules.
Suggested clarifications are not a frozen D028 protocol amendment.

**Commands/results:**

- `git status --short` and `git rev-parse --short HEAD`: clean starting tree at
  `03c7d39`.
- Read the supplied Pasted text.txt with line numbers, TASKS and latest LOGBOOK;
  inspected SHIFT-002 collection, original inference fingerprints, provenance
  validation and per-cell gates. SHIFT-002 uses raw-scale frame MAE <=1.0 and
  rejects unsuitable identity-check proposals without aborting collection.
- `.venv/bin/python` read-only availability checks: MPS false, CUDA false;
  `docs/TRAIN-001_PLAN.md` absent.
- `git diff --check`: passed after review documentation edits. No production
  source changed; no tests, collection, training or confirmation executed.

**Next:** pass the review to the implementing agent and resolve the listed rules
before its audit/freeze. TRAIN-001 execution and final REPORT-001 authoring remain
outside this review. Existing completed results and protocols are unchanged.

---

## 2026-09-30 — TRAIN-001 — Protocol freeze, implementation and smoke check

**Actor:** Claude, at Omer's request to implement and execute TRAIN-001.

**Scope and changes:**

- Saved the supplied plan verbatim as `TRAIN-001_PLAN.md`.
- Recorded D028 before any TRAIN-001 data was collected. It holds Omer's
  measurement-gate policy and all six corrections from
  TRAIN-001_IMPLEMENTATION_PLAN_REVIEW.md.
- **Design choices:**
  - the archive rest filter is retained;
  - identity uses raw-uint8 MAE ≤ 1.0;
  - the direction cutoff is 0.05;
  - development gates count observed moved families;
  - verdict and ordinary-outcome truth tables are fixed;
  - fingerprints bind records to their model;
  - the budget ledger is cumulative.
- **New modules:** `train001.py` (stages and CLI), `train001_protocol.py`,
  `train001_budget.py`, `train001_collect.py`, `train001_model.py`,
  `train001_metrics.py`, `train001_provenance.py`, `train001_report.py`.
- **New tests:** `tests/test_train001_{model,collect,metrics,budget,provenance}.py`.
- Added TRAIN-001_RUNBOOK.md and a TASKS row.
- No completed runner, sealed artifact or `uv.lock` was changed.

**Commands and results:**

- **Real-checkpoint sanity check**, via `.venv/bin/python` (`load_model`,
  `count_trainable`, `predict_future` vs cached `rollout`):
  - trainable parameters: **11,740,334**;
  - cached vs pixel-path max abs difference: 1.25e-6 (tolerance 1e-5);
  - 64-sequence update: about 0.20 s on 4 CPU threads.
- **`pytest tests/test_train001_*.py`:** all pass.
  - Each provenance mutation fails on its intended check: checkpoint → registry, readout,
    cache, family bytes → integrity, record content, foreign, swapped model, stale
    input, corrupt, config resealed, run resealed, eligibility resealed, stray file.
- **Smoke run.** `python -m jepa_action_effects.train001 <stage> --smoke` for all nine
  stages completed end to end:
  - separate root, seed 20261199 and ledger; about 105 s in total;
  - 96 confirmation records, strict entry and completion validation, and report
    figures all worked;
  - smoke numbers are implementation checks only, not evidence.
- **Suite.** `make check` passed (Ruff, format, doctor). `make test`: **233 passed**, 19
  existing Gymnasium warnings, 12.9 s.

**Evidence:** HEAD `6e091a6` plus the uncommitted TRAIN-001 files listed above.

**Limitations:**
- CPU only: no MPS or CUDA is available.
- The smoke challenge floor of about 0.74 comes from 4 families with an outlying
  decoded start. It is not informative about production.

**Next:** production `audit → collect-development → cache → pilot`.

---

## 2026-09-30 — TRAIN-001 — Production audit, development collection, eligibility and pilot

**Actor:** Claude (Mac CPU, 4 threads). All stages below are charged to
`outputs/train001-budget/ledger.json`.

**Commands:** `.venv/bin/python -m jepa_action_effects.train001 <stage>` for `audit`,
`collect-development`, `cache` and `pilot`. Logs are in `outputs/train001-logs/`.
Environment: `UV_CACHE_DIR`, `MPLCONFIGDIR` and `XDG_CACHE_HOME` under `.cache/`.

**Audit:**
- Passed. Archive HDF5: 46,300,921,856 bytes, SHA-256 `b6ebd9ac…` (matches the
  extraction).
- Exclusions: 1,226 reset seeds, 2,098 archive episodes and 1,581 history-frame
  hashes. These include the smoke run's 16 seeds and 84 episodes.
- Git: `6e091a6`, dirty (the uncommitted TRAIN-001 files).

**collect-development (523 s):**
- 300 + 60 families per scene and 60 ordinary windows. The integrity check passed.
- Challenge: 420 proposals; 59 rejected on movement, 1 unsafe.
- Archive: 11,162 attempts. Rejections by reason:
  - not at rest: 6,397;
  - identity failed: 3,562;
  - movement: 535;
  - out of bounds: 189;
  - near-zero direction: 119.
- Archive weights: 11,783 windows from 1,000 episodes; bin counts 6,030 / 3,956 /
  1,557 / 240. Weights are 0.512 / 0.336 / 0.132 / 0.020 for magnitudes
  0.15 / 0.30 / 0.45 / 0.60.

**cache:** passed. About 0.10 s per family.

**Eligibility.** Sealed first, from real-frame controls on 120 development-validation
families, before any training. Status: **not_assessable_under_measurement_protocol**.

| Cell | Moved | slope_real [CI] | Floor |
| --- | ---: | --- | ---: |
| challenge 0.45 | 60 | 0.999 [0.948, 1.049] | 0.234 |
| challenge 0.60 | 60 | 0.977 [0.931, 1.025] | 0.222 |
| archive 0.45 | 60 | 0.985 [0.957, 1.014] | **0.278** |
| archive 0.60 | 60 | 0.985 [0.953, 1.014] | **0.304** |

- The archive cells fail only the 25% floor. Their mean real-frame displacement errors
  are 10.8 and 11.8 against a median moved displacement of 38.7.
- Under D028, training and fresh latent evaluation continue. The pooled physical
  hypothesis cannot be confirmed; challenge-only physical comparisons are exploratory.

**pilot (426 s): passed.**
- Implementation checks all passed (11,740,334 trainable parameters; frozen and BN
  tensors unchanged).
- Cached vs pixel rollout: 3.1e-6 (tolerance 1e-5).
- Overfit: 16 families, LR 1e-4; loss 0.237 → 0.00098 (99.6% reduction).
- LR sweep (validation latent MSE, mean of the two conditions):

  | LR | MSE |
  | --- | ---: |
  | 1e-5 | 0.1213 |
  | 3e-5 | 0.1226 |
  | 1e-4 | 0.1386 |

  **Selected: 1e-5.**
- Budget: 0.21 s/update. Projected remaining 4,266 s including the 1.5× contingency,
  against 85,351 s remaining (1,049 s used).

**Next:** `train` (six runs), then `collect-confirmation`, `seal`, `confirm` and `report`.

---

## 2026-09-30 — TRAIN-001 — Training, fresh confirmation and report

**Actor:** Claude (Mac CPU, 4 threads). The stages were charged to the ledger.

**Commands:** `.venv/bin/python -m jepa_action_effects.train001 <stage>` for `train`,
`collect-confirmation`, `seal`, `confirm` and `report`. Before `seal`, ran
`make check` (passed) and `make test` (**233 passed**, 19 existing warnings).

**Ledger** (seconds per session, all `completed`):

| Stage | Seconds |
| --- | ---: |
| audit | 27.6 |
| collect-development | 523.2 |
| cache | 76.4 |
| pilot | 422.4 |
| train | 2,530.1 |
| collect-confirmation | 210.3 |
| seal | 1.5 |
| confirm | 20.2 |
| report | 3.3 |

Total **1.06 h of 24 h**.

**Results:**

- **train.** Six runs (archive-weighted and balanced × seeds 20261111/12/13, LR 1e-5,
  2,000 updates). Validation latent MSE went from about 0.12 at update 250 to
  0.092–0.096 at update 2,000. Calibration scales: 1.030 / 1.030 / 1.038. Exploratory
  development evaluation: balanced − archive endpoint error −0.17 [−0.73, 0.32].
- **collect-confirmation.** 120 + 120 families and 120 windows. Archive: 3,922 attempts;
  1,248 identity failures, 37 near-zero-direction rejections. Integrity check: 1,140
  records, 1,660 disjoint episodes, 480 unique seeds.
- **seal and confirm.** Strict entry validation passed on an empty confirmation
  directory. 2,880 records were written; completion validation passed.
- **Confirmation gates.** All four cells passed (archive floors 0.156 and 0.149;
  challenge 0.206 and 0.203). Development eligibility still makes the pooled verdict
  **not_assessable_under_measurement_protocol**.
- **Measured contrasts** (primary readout, reported not claimed):
  - balanced − archive-weighted −0.09 [−0.53, 0.38];
  - balanced − frozen −11.66 [−14.29, −9.11];
  - archive − frozen −11.57 [−14.04, −9.17].
  - Seed SD for balanced − archive: 0.11.
- **Secondary latent MSE:** balanced − archive −0.0070 [−0.0127, −0.0014]; both minus
  frozen about −0.25.
- **Ordinary windows:** balanced − frozen +1.05 [0.11, 1.99], so non-inferiority is not
  established. Balanced − archive +0.38 [0.05, 0.69].
- **Controls:**
  - calibrated frozen improves error by only −0.31;
  - permuted − correct error: frozen +8.8, fine-tuned about +17;
  - persistence 55.4;
  - both alternative readouts give the same verdict.
- **report.** Wrote `docs/manifests/train001-runtime.json`, `TRAIN-001_RESULT_DRAFT.md` and
  three figures in `docs/figures/train001/`. Hand-written interpretation:
  TRAIN-001_RESULT.md. D028 outcome appended.

**Evidence and hashes:**
- `outputs/train001/seal.json` sha256 `e2cceaca9d7c451c…`;
- `confirm/summary.json` `01c1ad9e371c7844…`;
- manifest `d36485ff1adb195f…`.
- Git HEAD `6e091a6` with uncommitted TRAIN-001 files. Checkpoints and caches stay
  ignored.

**Limitations:**
- In-distribution adaptation: training and confirmation use the same constant-push
  family process.
- Readout validity on predicted latents is unproven.
- One checkpoint and one horizon.

**Next:** REPORT-001 authoring from the updated plan. Commit when Omer asks.

---

## 2026-09-30 — TRAIN-REVIEW-002 — Review of completed TRAIN-001

**Actor:** Codex, continuing the review after Omer supplied the implementing
agent's execution handoff.

**Scope/changes:** Reviewed the uncommitted TRAIN-001 implementation, D028, saved
records/checkpoints and reports at HEAD `6e091a6`. Added TRAIN-001_REVIEW.md and the
review task/handoff. Preserved all existing implementation changes and sealed
artifacts. No scientific code fixes or retraining were performed.

**Commands/results:**

- `make check && make test`: passed; **233 tests, no skips**, 19 existing Gymnasium
  warnings; pytest duration 13.73 seconds.
- `.venv/bin/python -` with production `train001_provenance.validate` in COMPLETE
  mode: all **51 checks passed** for the 2,880 saved confirmation records.
- Same interpreter, read-only `train001_metrics.summarize` from saved records:
  full confirmation summary equals the sealed saved summary exactly. Actual
  secondary latent contrast: -0.0068376941 [-0.0127569325, -0.0011002090]; the
  hand-written -0.0070 [-0.0127, -0.0014] in the preceding outcome entry needs this
  correction. Qualitative sign is unchanged.
- Same interpreter, reloaded all seven predictor checkpoints and rescored fixed
  first/last challenge, archive and ordinary units (six units, 42 model/unit
  combinations): maximum decoded-coordinate difference 6.062e-5; latent-MSE
  difference 5.216e-8. No result artifacts written.
- All six saved run collection/cache hashes match the production seal; training
  weights match their conditions.
- Temporary existing provenance fixture with foreign training collection/cache
  hashes, resealed before confirmation: entry validation incorrectly passes.
- In-memory alternative-readout perturbation: an interval strongly above zero
  still produces `measurement_dependent=False` when development eligibility fails.
  Twenty bootstrap draws used only for this synthetic logic check.
- Visually inspected all three scientific figures. `git diff --check`: passed
  after review-document changes. No tests were added to frozen experiment sources.

**Outcome/limitations:** Numerical evidence reproduces, but physical interpretation
overclaims, a secondary interval transcription error, the sensitivity flag and
training-data lineage validation need correction. Actual lineage hashes match;
the validator counterexample does not establish contamination. This is not a full
training rerun or independent implementation. See TRAIN-001_REVIEW.md for concrete
locations, corrections and preservation of sealed evidence.

**Next:** correct reporting before REPORT-001; address code findings through a
versioned follow-up preserving the original source/seal. No new experiment or
protocol decision was introduced by this review.

---

## 2026-09-30 — TRAIN-FIX-001 — Address TRAIN-REVIEW-002

**Actor:** Claude, at Omer's request.

**Scope and changes:** D029 accepts all four findings. No sealed artifact, seal-hashed
source, verdict or number changed. The six models were not retrained.

- Added `src/jepa_action_effects/train001_followup.py`:
  - `validate_lineage` and `validate_with_lineage`;
  - `contrast_sensitivity`;
  - `exact_numbers`;
  - smoke-time disclosure.
- Added `tests/test_train001_followup.py`:
  - six stale-lineage fixtures that the sealed validator admits and the lineage check
    rejects;
  - the review's sensitivity counterexample;
  - agreement and outcome-disagreement cases.
- `train001_report.py` (formatting only, not seal-hashed): physical figures, draft
  tables and titles are labelled descriptive; the draft has a contrast-level
  sensitivity table; latent values are printed to 5 decimals. `regenerate_presentation`
  rewrites figures and the draft without touching `train001-runtime.json`.
- Corrected TRAIN-001_RESULT.md (dated banner) and REPORT-001_PLAN.md: wording, latent
  interval, sensitivity and smoke disclosure. Updated TASKS. The earlier LOGBOOK and
  D028-outcome values stand as history, superseded by D029.

**Commands and results:**

- `.venv/bin/python -m jepa_action_effects.train001_followup`:
  - `valid_with_lineage: true`: 58/58 checks (51 sealed + 7 lineage);
  - sign disagreements: `ordinary:balanced_minus_frozen` only;
  - `seal.json` and `confirm/summary.json` hashes unchanged before and after
    (`e2cceaca9d7c451c…`, `01c1ad9e371c7844…`);
  - smoke run: 105.2 s outside the production ledger.
- Exact latent balanced − archive: −0.0068376941 [−0.0127569325, −0.0011002090].
- `make check`: passed (Ruff, format, doctor). `make test`: **242 passed**, no skips,
  19 existing Gymnasium warnings. `git diff --check`: passed.

**Evidence:** `docs/manifests/train001-followup.json`; regenerated
`docs/figures/train001/{magnitude_response,paired_errors}.png` and
`TRAIN-001_RESULT_DRAFT.md`. Originals are at `68d9fbc`.

**Limitations:** The lineage validator is new code. It is applied read-only to
TRAIN-001 and is not part of that run's seal. The follow-up reuses the sealed summary
and is not an independent reimplementation.

**Next:** REPORT-001 authoring. Use the corrected headline: a small secondary latent
advantage for balanced weighting, with ordinary-case non-inferiority unestablished;
physical comparisons are descriptive.

---

## 2026-09-30 — TRAIN-REVIEW-003 — Verify D029 fixes at 3932604

**Actor:** Codex, reviewing the implementing agent's handoff for Omer.

**Changes:** Appended verification and one new runner finding to TRAIN-001_REVIEW.md;
updated TASKS with closure of the original findings and TODO TRAIN-FIX-002. No source,
experiment, manifest, figure or sealed artifact was edited; no retraining occurred.

**Commands/results:** `make check && make test` passed: 242 tests, no skips, 19 existing
Gymnasium warnings (13.09 s). Read-only `.venv/bin/python -` checks invoked
`validate_with_lineage(..., COMPLETE, ...)`: 58/58 passed. Recomputed
`contrast_sensitivity` and `exact_numbers` exactly match train001-followup.json.
SHA-256 checks of seal/summary match the original runtime manifest from
`git show 68d9fbc:docs/manifests/train001-runtime.json` and the follow-up manifest;
the original runtime manifest is byte-identical. Follow-up/report source hashes
match the follow-up manifest. Both modified physical figures are legible and
explicitly descriptive.

**Finding/evidence:** An in-memory `unittest.mock.patch` fixture supplied a failed
lineage validation, intercepted regeneration and JSON writes, and called `main([])`.
The function returned normally and called both writers once despite `valid=False`.
No production failure is alleged; real production validation passed. Require failure
before output writes and nonzero CLI exit. Details and full evidence hashes are in
the appended review. The fixture wrote no artifacts.

**Limitations/next:** Original four findings are addressed. The remaining failure-path
bug affects future command reuse, not the current validated results. Fix TRAIN-FIX-002
in unsealed code, with a no-write-on-failure test. REPORT-001 drafting can proceed;
no new confirmation is needed. Prior full summary/inference checks were not repeated.

---

## 2026-10-01 — LIT-002 — Literature search for a main-conference extension

**Actor:** Claude, for Omer.

**Changes:** Added LIT-002_REVIEW.md and updated TASKS (row and handoff). No source code,
experiment, manifest, figure or sealed artifact was edited. No data were opened.

**Commands/results:**
- Queried the OpenAlex and arXiv APIs directly with a scratch client, because the
  paper-search plugin's scripts were missing. Searches ran on 2026-09-30.
  - 64 successful queries (14 OpenAlex, 50 arXiv) returned 225 unique records.
  - 55 abstracts were fetched by arXiv ID.
- Four Hugging Face Hub lookups confirmed released checkpoints:
  - `quentinll/lewm-{cube,tworooms,reacher}`;
  - `facebook/jepa-wms` (`jepa_wm_pusht`, `dino_wm_pusht`, DINOv2 decoder heads);
  - `knightnemo/nanowm-b2-dino-wm-pusht-100k`.
- Read-only local checks:
  - `projection_slope = Σ(p·t)/Σ(t·t)` (`train001_metrics`, `followup_metrics`);
  - the action pathway in the pinned `le-wm@ca231f9` `module.py`;
  - DIST-001 archive percentiles (p99 magnitude 0.683; challenge median rank 0.979).

**Findings (source review, not results):**
- Preprints from August–September 2026 overlap substantially with this study:
  - the Intervention Gap (2608.29998) audits LeWM intervention effects;
  - IMPLY (2609.12441) sweeps push speeds and names "typical push" failures;
  - PhysWeep (2609.06207) adjudicates "prior reversion" against "case-based clamping";
  - Sensitivity Shaping (2606.14585), Do-JEPA (2609.37378) and ARC-Bench (2609.05461).
- The projection slope is the reverse regression. It falls below 1 for an unbiased
  conditional-mean predictor under uncertainty, so a calibration-side slope is needed.
- The 1.27 small-push length ratio may be norm bias from readout noise. The inner-product
  ratio at 0.30 is 0.913.

**Limitations:**
- The search was targeted, not systematic, and mostly at abstract level.
- The 2026 sources are largely unreviewed preprints.
- The "reversion" account and every experiment in the review are proposals.
- The power figures are normal approximations from the TRAIN-001 interval.

**Next:** Read the seven flagged papers in full. Then decide whether to open Stage 1
(measurement) as a new, separately frozen task. REPORT-001 is unaffected.

---

## 2026-10-01 — CALIB-001 — Calibration-side reanalysis of opened records

**Actor:** Claude, for Omer (run on Omer's Mac).

**Changes:**
- D030 fixed definitions and interpretation rules before any value was computed. Its
  amendment, made after the first run, makes cells without block motion undefined and
  records that the agent analysis was deliberately not run.
- New `calib001_metrics.py` (pure estimands) and `calib001.py` (runner). The runner reuses
  the sealed loaders, reads saved latents or decodes, and never runs a model or writes a
  sealed file. Its report stage writes nothing if the reproduction gate fails.
- New `tests/test_calib001_metrics.py` (11 fixtures: hedging, shrinkage, reversion, noise
  correction, outcome selection, static excess, latent slopes, undefined cells, pairing)
  and `tests/test_calib001.py` (3 guards, including no writes on a failed gate).
- New docs: CALIB-001_RESULT.md, the generated CALIB-001_RESULT_DRAFT.md, and
  `docs/manifests/calib001-runtime.json`.
- Addenda to STUDY_SYNTHESIS and LIT-002; TASKS updated.

**Commands/results:**
- `.venv/bin/python -m jepa_action_effects.calib001 all` ran twice (before and after the
  amendment). Stage times were phys 1.4 s, shift 1.5 s, train 0.9 s and report 0.1 s.
  - Units: PHYS-002 develop 200 + 100, reserved 100 + 50; SHIFT-002 development 60 + 60,
    reserved 120 + 120; TRAIN-001 120 + 120 families and 120 ordinary windows.
  - Reproduction gate passed: 27 sealed values, maximum difference 1.1102e-15.
- `make check` passed. `make test`: 256 passed (242 before), with the 19 existing
  Gymnasium warnings.

**Findings (exploratory, primary readout, k = 1):**
- PHYS-002 reserved challenge: projection slope 0.561 = R² 0.746 ÷ calibration slope 1.329
  [1.236, 1.435]. At k = 0 it is 1.291 [1.204, 1.389].
- TRAIN-001 frozen: archive-scene calibration slope 1.05, 1.08, 1.24, 1.39 (0.15–0.60);
  challenge 0.75, 0.86, 0.90 (0.30–0.60).
- Dose trends are positive in all four reserved/confirmation families of cells.
- Fine-tuned models are calibrated in archive scenes (0.97–1.05).
- Ordinary windows: 0.99 [0.88, 1.08].
- Null-dose spurious motion: 72.8 and 104.6 units² beyond the real-frame readout (frozen),
  against 8–15 units² (fine-tuned).
- Readout-free latent step-1 slopes rise with dose: challenge 0.76 → 0.99, archive
  0.95 → 1.05.

**Evidence:** manifest SHA-256 `9e00e501280328cd…`; outputs `phys.json`
`0123ebcdde32dc85…`, `shift.json` `b30b23526a124724…`, `train.json`
`bba5833ebf0606d8…` (ignored). Run from `21f8b68` plus the uncommitted CALIB-001 changes,
which are committed with this entry.

**Limitations:**
- Exploratory: all families were already opened.
- Decoded calibration assumes readout validity on predicted latents.
- The noise correction assumes equal noise on predicted and real latents; the
  over-dispersion component depends on it.
- Latent slopes are not block-specific.
- One checkpoint; no multiplicity correction.

**Next:** Scope the simulator-in-the-loop inversion (LIT-002 §3.2) as a separately frozen
measurement task. Consider an agent-displacement calibration only with a better agent
readout.

---

## 2026-10-01 — CALIB-REVIEW-001 — Review literature and calibration handoffs

**Actor:** Codex, reviewing Omer's two pasted handoffs at `619a151` (CALIB-001) and
`c0777c8` (LIT-002). Working tree was clean at review start.

**Changes:** Added CALIB-001_REVIEW.md and review/fix rows in TASKS. No source, manifest,
figure, generated table or sealed artifact was edited. No new experiment was authorized.

**Commands and evidence:**
- `make check && make test`: passed; 256 tests, no skips, 19 existing Gymnasium warnings;
  pytest 13.26 s.
- `.venv/bin/python -` called `calib001.phys`, `.shift`, and `.train` into a temporary
  directory. All complete output objects and hashes match the saved outputs exactly.
  `reproduction(results)` passed all 27 checks with maximum difference
  1.1102230246251565e-15. `_compact(results)` and `draft(results, checks)` match the
  tracked manifest results and generated draft exactly.
- All CALIB manifest source/input/readout hashes match current files. Manifest SHA-256:
  `9e00e501280328cd8a98489f1c649d158937b18e9d5b99bf5d0aaf7f6bfeb678`.
- In-memory synthetic fixtures through `row_statistics`/`estimates`: exact latent
  conditional mean plus exact nonlinear decoder yields b_cal=1.25; pure 60-degree
  rotation yields b_cal=0.5 and squared alignment=0.25 with unchanged lengths; a
  miscalibrated deterministic predictor can have b_cal=1. These are logic checks,
  not observations about the actual predictor.
- Targeted primary-source checks: Intervention Gap, IMPLY, PhysWeep, Kang et al.,
  Sensitivity Shaping, Do-JEPA, ARC-Bench. Links and source-review limits are in the
  review. The full literature search and checkpoint inventory were not reproduced.

**Findings:** Four P2s concerning latent versus physical conditional means; interpreting
scalar gains/alignment as excess magnitude/lost resolution; insufficiently stated noise
assumptions; and inversion lacking off-manifold rejection. Numerical findings stand as
exploratory descriptions; the stronger mechanism claims do not follow. The new CALIB
report guard passes its tests, but the existing TRAIN-FIX-002 remains open.

**Limits and next:** This rerun uses the author's metric implementation and saved
world-model outputs, not an independent end-to-end replication. Correct reporting via
CALIB-FIX-001 before publication planning. A bounded inversion/measurement feasibility
proposal is a reasonable next research step with rejection and ambiguity controls; no
additional training was run or scheduled. REPORT-001 can proceed with qualified claims.

---

## 2026-10-01 — CALIB-FIX-001 — Qualify CALIB-001 after review

**Actor:** Claude, for Omer.

**Changes:**
- D031 records dispositions of CALIB-REVIEW-001. All four findings are accepted, and the
  LIT-002 TOST and planner-goal corrections are included.
- `calib001_metrics.py`: corrected docstring (scalar gain, orthogonal-error model, what
  b_cal cannot show); new `energy_ratio`, `alignment` and latent `z{step}_energy_ratio`
  estimands.
- `calib001.py`: tables relabelled, with k = 0 gain and class primary, magnitude and
  alignment columns, and k = 1 marked as a sensitivity analysis. Dose contrasts were added
  for the k = 0 gain, energy ratio and alignment.
- Tests: the review's three counterexamples (nonlinear decoder 1.25; 60° rotation 0.5/0.5;
  unit gain with wrong predictions) and the magnitude-alignment factorization.
- CALIB-001_RESULT.md rewritten under a corrections banner (original at `619a151`);
  correction addenda appended to STUDY_SYNTHESIS and LIT-002; TASKS updated. The
  reviewer's CALIB-001_REVIEW.md and their TASKS/LOGBOOK edits are kept as written.

**Commands/results:**
- `.venv/bin/python -m jepa_action_effects.calib001 all` (rerun):
  - the reproduction gate passed: 27 values, maximum difference 1.1102e-15;
  - a leaf-by-leaf comparison with the pre-fix outputs found 0 of 13,768 earlier values
    changed; only the new estimands were added.
- New frozen-model paired energy-ratio dose contrasts. All exclude zero:
  - TRAIN-001: archive −0.40 [−0.50, −0.30]; challenge −0.66 [−0.85, −0.48];
  - SHIFT-002 reserved: challenge −0.64 [−0.89, −0.41]; archive −0.35 [−0.48, −0.23];
  - SHIFT-002 development: challenge −0.61 [−1.01, −0.26]; archive −0.35 [−0.55, −0.18].
- Alignment contrasts range from −0.07 to +0.04. Real-frame energy ratios are
  0.98–1.00 (TRAIN-001 archive) and 1.27 → 1.14 (challenge). All 12 key cells agree in
  k = 0 class across the three MLP readouts.
- `make check` passed. `make test`: 260 passed (256 + 4 new), 19 existing Gymnasium warnings.

**Limitations:** Everything remains exploratory and readout-mediated. The latent
magnitude trend is agent-dominated. Whether the readout behaves on predicted latents as
on real frames is untested.

**Next:** Draft the bounded measurement-feasibility proposal under the D031 gates. Do not
start training or replication.

---

## 2026-10-01 — CALIB-REVIEW-002 — Verify CALIB-FIX-001

**Actor:** Codex, reviewing Omer's implementing-agent handoff. Base `e952b13`, with
CALIB-FIX-001 changes already uncommitted; those changes were preserved.

**Changes:** Appended follow-up verification to CALIB-001_REVIEW.md, added the review
row/current handoff to TASKS and this log entry. No implementation or result edits.

**Commands/results:**
- `make check && make test`: 260 passed, no skips, 19 existing Gymnasium warnings;
  pytest 13.57 s. `git diff --check` passed.
- `.venv/bin/python -`: reran current `calib001.phys/shift/train` into a temporary
  directory, matching each saved output hash, compact manifest results and generated
  draft exactly. Reproduction passes all 27 checks.
- Loaded original runner/metrics through `git show 619a151:...` into temporary Python
  modules and reran them on the same saved inputs. Recursive original-versus-current
  comparison: 13,768 original leaf values, zero changed.
- Verified all six frozen energy-ratio contrasts are negative with intervals excluding
  zero and all 12 key cells agree in k=0 class across the three MLP readouts.
- Source/lockfile/readout hashes match the updated manifest. TRAIN-001 seal and summary
  still match the original runtime manifest. Updated CALIB manifest SHA-256:
  `ef803755bce8045df1983e011828ea5a1c5e2483a67f575246a9fe5b862cc724`.

**Disposition:** All four requested corrections are implemented; no new blocking issue
found. Carry two reporting qualifications into the paper: energy ratio is squared RMS
magnitude ratio, and pooled alignment includes per-candidate gain heterogeneity as well
as angular errors. A zero-angle synthetic example gives alignment 0.743. The actual
TRAIN challenge alignment contrast is -0.0576 [-0.0998, -0.0154], not no change.

**Limits/next:** Exploratory reanalysis using the same stored model outputs, not new
confirmation or independent end-to-end replication. Accept CALIB-FIX-001 with the review's
qualifications. Next research step: bounded measurement-feasibility proposal under D031;
no training or replication started. TRAIN-FIX-002 remains open. All changes remain
uncommitted, including the implementing agent's pre-existing work.

---

## 2026-10-01 — MEAS-001-PLAN — Bounded measurement-feasibility proposal

**Actor:** Codex, at Omer's request to draft the next proposal; continued after Omer's
follow-up. Existing uncommitted CALIB-FIX-001 and review changes were preserved.

**Changes:** Created MEAS-001_PLAN.md; recorded proposal choices in D032; added a completed
planning row and TODO implementation/execution row to TASKS. No experiment source,
checkpoint, generated result or sealed artifact was changed.

**Basis:** Read GALLERY-001 and PREDICTOR-GALLERY-001 results, current TASKS/LOGBOOK,
D031 and CALIB review. Consulted gallery renderer/search scaffolding, SHIFT-002
candidate metadata definitions, RESEARCH_PLAN measurement thresholds and DEVELOPMENT
provenance requirements. Prior gallery success fixed true future agent/angle, and the
predictor gallery failed its primary challenge criterion; neither is joint inversion
validation.

**Proposed design:** 48 existing SHIFT-002 development families; 24 calibration and
24 instrument-assessment, scene-balanced. Joint agent/block pose inversion with
residual rejection, near-optimal ambiguity, restart/optimizer and nuisance controls;
216 real queries, up to 24 synthetic controls and an optional 16 already-opened
prediction diagnostic. Four-hour cumulative CPU cap with throughput stop rule. All
interpretation is feasibility/descriptive, with no new training or fresh confirmation.
Exact implementation config must be sealed before calibration and learned residual
thresholds before assessment. These are proposed engineering gates, not measured success.

**Verification:** `git diff --check` passed. Checked partition/query/budget arithmetic
and candidate labels against the existing SHIFT-002 collector. Documentation only;
no tests, simulator replay, encoding, model inference or literature search were run.
The previous review's 260-test result is not a test of this unimplemented proposal.

**Next:** Reviewable implementation and frozen configuration for MEAS-001, followed by
its bounded execution if pursued. TRAIN-FIX-002 remains independently open. Proposal
and all prior work remain uncommitted; no commit or push was requested in this turn.

---

## 2026-10-01 — MEAS-001-ENG — Runner, tests and engineering pass

**Actor:** Claude, for Omer (run on Omer's Mac).

**Changes:**
- D033, recorded before any encode: the engineering-set rule, version rule and stop
  point.
- D033 amendment (proposed; needs review): agent geometry and reset-family identity.
- New modules `pose_search.py`, `pose_render.py`, `meas001_metrics.py` and `meas001.py`,
  with tests `test_pose_search.py`, `test_pose_render.py`, `test_meas001_metrics.py` and
  `test_meas001.py`.
- New docs: MEAS-001_RUNBOOK.md, MEAS-001_ENGINEERING.md and
  `docs/manifests/meas001-engineering.json`; TASKS updated.

**Commands/results:**
- Implementation profiling on a non-study scene (default reset, seed 0; uncharged, about
  1 min): render 1.1 ms and encode 9.6 ms per pose. Batch-order encoding differences are
  at most 5e-6 (MSE 8e-13).
- `meas001 select` sealed membership: 48 families, 216 real queries, 108 for calibration.
- The first `select` refused a seed alias, because archive resets share seed 0. The
  identity rule was corrected (amendment) and the rerun passed.
- `meas001 engineer-select`: no shared seed, reset state or episode with the 48. Spec
  sealed.
- Renderer identity on all 30 GALLERY families: replay is exact, and direct assignment
  differs at 2–3 px in 7 of 210 frames.
- `meas001 engineer`: 1,191 s charged, 72 targets, status `failed_selection`. Every
  setting accepted 0 of 18 real queries and recovered 0 of 6 synthetic poses.
  - Seconds per target: S1 18.3, S2 12.9, S3 18.3.
  - Phase-A τ: 0.030, 0.128 and 0.041.
- `meas001 propose`: `not_ready`, proposal SHA-256 `6c2754091787…`.
- `make check` passed; `make test`: 299 passed (260 + 39 new).

**Diagnosis:** see MEAS-001_ENGINEERING.md.
- For S2, the median best block error is 1.05 units; 14 of 24 targets are within 1.28.
- Search pairs end a median 4.6–5.1 units apart, and no search reached the true pose's
  near-zero residual.
- The residual rises smoothly with block distance: 2e-4 below 0.64 units, 0.14 above
  20 units.
- The D032 τ rule sets τ from worst-case achieved residuals.

**Evidence:** `outputs/meas001/` (ignored): membership, engineering spec, 72 traces,
result and proposal; `outputs/meas001-budget/ledger.json` (one complete session).
No SHIFT-002 family trace exists.

**Limitations:** small engineering sample; GALLERY scenes only; nothing about predicted
latents.

**Next:** review the agent-geometry amendment, then decide between engineering version 2
(new sealed spec; phase-B families unused) and stopping MEAS-001; also consider the τ rule.

---

## 2026-10-01 — MEAS-REVIEW-001 — Review failed engineering pass at 8ecd208

**Actor:** Codex, reviewing the implementing agent's committed handoff for Omer.
The working tree was clean at the start.

**Changes:** Added MEAS-001_REVIEW.md, review/fix rows and a qualified current handoff
to TASKS. No experiment source, sealed configuration, generated study result, budget
ledger or protocol was changed. Recommendations are not an executed D033 amendment.

**Commands/results:**
- `make check && make test`: all checks pass, 299 tests passed, no skips, 24 existing
  Gymnasium warnings; pytest 14.88 s; Python 3.10.21.
- `.venv/bin/python /tmp/meas001_review_checks.py`: reloaded all 72 engineering traces;
  verified trace hashes, settings and classifications; recomputed diagnostics exactly
  matching `docs/manifests/meas001-engineering.json`. Current proposal source/checkpoint
  hashes and all 30 engineering plus 48 MEAS family-file hashes match.
- S2 median best block error 1.0519232868, engineering tau 0.1282356828; all settings
  accepted 0/18 real queries and 0/6 synthetic queries. All saved nuisance challenges
  finished. Only six phase-A GALLERY families appear in traces; no phase-B traces,
  frozen config or calibration result. Ledger: 1,191.423332 s, one completed session.
- Synthetic/mocked counterexamples: changed family seed accepted despite recorded
  hash mismatch; existing four-start result reused for a different query requesting
  two starts; prediction eligibility true after failed relative floors or known-failure
  controls; classifier accepts a nuisance-only best 10 units from both independent
  search solutions, even when that nuisance fit did not converge.
- Source inspection confirms the pinned simulator's agent is kinematic. Tests include
  simulator fixtures; no MEAS family search, checkpoint inference, calibration or saved
  predictor analysis was performed by the review.
- `git diff --check` passed. Final source-hash verification still matches the proposed
  configuration; it remains `not_ready`, with no frozen config or calibration result.

**Findings:** Four P2 findings: incomplete input/result lineage enforcement, incomplete
optional-prediction eligibility, missing independent support for the final returned
pose, and overstated identifiability/landscape diagnosis. Current artifacts match, so
these defects do not alter the verified failure counts. Additional work remains on
MLP-disagreement/report output, genuine-control recovery summaries and ledger coverage.

**Correction to preceding diagnosis:** Saved distance bins summarize visited candidates
across differing targets and nuisance poses; they do not establish a smooth residual
landscape or that position is identifiable. Near-truth fit and search shortfall are
supported. The earlier statement that no search reached the true basin should retain
the actual diagnostic's 0–1 of 24 qualification.

**Next:** MEAS-FIX-001 before freeze or revised engineering. Recommend geometry/reset
corrections and one bounded version 2, with independent refinement, separate threshold
roles and corrected cumulative accounting. Preserve v1 and keep the 48 MEAS families
closed. No new run, commit or push was requested or performed during this review.

---

## 2026-10-01 — TRAIN-FIX-002, D033 disposition and project plan

**Actor:** Claude, for Omer.

**Changes:**
- `train001_followup.py`: `run()` now raises `ValidationFailure` (with the failed checks)
  before computing sensitivity, regenerating figures/tables or writing the manifest;
  `main()` prints the failure and exits 1. TRAIN-REVIEW-003's P2 is addressed. The
  sealed TRAIN-001 sources are untouched (this module is the unsealed follow-up).
- `tests/test_train001_followup.py`: a regression test patches validation to fail and
  makes every writer raise; it checks `run()` raises, `main([])` exits 1 and an existing
  manifest stays byte-identical.
- DECISIONS: D033 amendment accepted (agent geometry; reset-family identity), with the
  MEAS-REVIEW-001 qualifications; bounds-coverage fixtures assigned to MEAS-FIX-001.
- TASKS: TRAIN-FIX-002 done; ROBUST-001, SUPPORT-001 and TRAIN-002 rows; plan handoff.
- Committed the reviewer's MEAS-001_REVIEW.md and TASKS/LOGBOOK edits as written
  (`1ed14d0`).

**Commands/results:**
- `.venv/bin/python -m pytest tests/test_train001_followup.py -q`: 10 passed.
- Read-only `validate_with_lineage(Protocol.production().run_root, …, COMPLETE, …)` on
  the real outputs: valid, 58 checks, 0 failed, so the success path is unaffected.
- `make check`: passed. `make test`: 300 passed, 24 existing Gymnasium warnings.

**Plan:** reviewed the full record and the course handout with Omer. The course report
(due within two weeks) takes priority, and the work should also seed a main-conference
paper. Gaps: no report yet; mechanism descriptive only; a single readout family; one
checkpoint; novelty pressure from Aug–Sep 2026 preprints. Sprint: ROBUST-001, then
SUPPORT-001 (preregistered account test) and TRAIN-002 (dose-omission / archive-only
fine-tuning), an evidence freeze on 2026-10-07, then REPORT-001. MEAS-001 moves to the
paper track, after MEAS-FIX-001.

**Next:** ROBUST-001 on saved predicted latents; read the seven flagged papers in full.

---

## 2026-10-01 — ROBUST-001 — Multi-instrument re-decoding (exploratory)

**Actor:** Claude, for Omer (run on Omer's Mac).

**Changes:**
- D034, recorded before any kNN fit or off-manifold value: instruments, estimands,
  off-manifold flag, reproduction gate and interpretation rules.
- New `robust001_metrics.py` (kNN readout, k selection, off-manifold threshold, spread,
  Spearman) and `robust001.py` (stages knn/shift/phys/report). The runner reuses the
  CALIB-001 loaders and estimands without editing `calib001.py`.
  - Lineage: family files are hash-verified (including PHYS-002 reserved files, which
    `review_followup._phys_items` did not check); latent, shard, model and kNN-selection
    hashes are recorded. Outputs made with other shards, readouts or kNN selections are
    refused. A failed reproduction writes no report.
- New `tests/test_robust001.py` (9 tests): bank recovery, kNN clamping, k selection under
  label noise, threshold/spread/Spearman, shard-change refusal, the reproduction gate,
  verdict rules (kNN-only trends discounted), and no writes on a failed gate.
- New docs: ROBUST-001_RESULT.md, the generated ROBUST-001_RESULT_DRAFT.md and
  `docs/manifests/robust001-runtime.json` (SHA-256 `0ce80ed96f55bd58…`).

**Commands/results:**
- `LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m jepa_action_effects.robust001 knn|shift|phys|report`
  (1.6 s, 13.6 s, 16.2 s, 0.3 s).
  - kNN: k = 8, validation block error 8.99 units (MLP 3.48, ridge 15.68).
  - Reproduction gate: 1,888 CALIB-001 values, maximum difference 8.88e-16, passed.
  - Reserved SHIFT-002 energy-ratio dose contrasts, all five instruments with CI < 0:
    challenge −0.58 to −0.71, archive −0.35 to −0.58.
  - PHYS-002 reserved challenge b_rev 0.543–0.574 across instruments.
  - Off-manifold flag rates 0.04–0.12, falling with dose. On-manifold stratum
    inconclusive under D034: ridge challenge −0.36 [−0.88, 0.09].
- `make check`: passed. `make test`: 309 passed (300 + 9 new), 25 existing warnings.

**Limitations:** Exploratory, opened records, frozen model only. All instruments share
one encoder and one 50k-frame bank, so agreement is corroboration, not validation.

**Next:** SUPPORT-001 support map and TRAIN-002 protocol.

---

## 2026-10-01 — LIT-003 — Full-text reading of the flagged papers

**Actor:** Claude, for Omer.

**Changes:** Added LIT-003_POSITIONING.md (per-paper findings, overlap, consequences, a
draft related-work paragraph) and a TASKS handoff. No code or evidence changed.

**Commands/results:** `curl https://arxiv.org/html/<id>` for 2608.29998, 2609.12441,
2609.06207, 2606.14585, 2609.37378, 2609.05461 and 2310.00873 (all HTTP 200), converted
to text in the session scratchpad (not committed); about 68k words read.

**Findings:** The Intervention Gap is the closest precedent; its gate design matches ours,
and LeWM shows excess gain on Cheetah where we measure compression. Corrections to
LIT-002: IMPLY's "typical push" means object-blind, not magnitude reversion; Kang et al.'s
reversion is for high-dimensional input OOD. Do-JEPA supports TRAIN-002's held-out-dose
design. Sensitivity Shaping is the qualitative precedent for hypothesis B.

**Limitations:** Author claims from preprints; not reproduced. Appendices skimmed.

**Next:** SUPPORT-001 stage a and the TRAIN-002 protocol.

---

## 2026-10-01 — TRAIN-002 and SUPPORT-001 — Protocols, runners and pre-collection stages

**Actor:** Claude, for Omer (run on Omer's Mac).

**Changes:**
- D035 (TRAIN-002) and D036 (SUPPORT-001), both recorded before any of their data.
- `train002.py`, `train002_metrics.py`, `train002_report.py`, `tests/test_train002.py` (8).
  These reuse the sealed TRAIN-001 modules without editing them. Reused TRAIN-001
  checkpoints are verified against its seal. Gates are computed on confirmation real
  frames before the seal. Scoring refuses changed sources, inputs or checkpoints.
- `support001.py`, `support001_metrics.py`, `support001_report.py`,
  `tests/test_support001.py` (9). Includes the block-frame support map, the four accounts
  (each recovered and selected on its own synthetic data in tests), a six-dose collector
  mirroring the TRAIN-001 family process, a real-frame gate before the seal, and the agent
  probe.

**Commands/results:**
- `train002 audit`: TRAIN-001 seal `e2cceaca…`, development data and all six reused
  checkpoints verified.
- `collect-windows`: 2,400 archive training windows.
- `collect-confirmation` (3 min): 120 + 120 families, 120 windows.
- `cache` (2.5 min). `train` started 18:11; running at this entry.
- `support001 map`: 399,758 archive windows (18,685 episodes). Mean-command magnitude
  quantiles: p50 0.172, p90 0.328, p99 0.478, p99.9 0.601.
- `support001 fit` (development: opened TRAIN-001 frozen records, 960 units). In-sample
  MSE: N 468.2, A 436.6, B 443.1, C 451.2. Parameters:
  - A: g = 1.078 − 0.767‖a‖;
  - B: w = σ(2.059 + 0.232ℓ);
  - C: c = 0.347.

  The over-range predictions differ: A keeps falling (0.46 at 0.80, 0.31 at 1.00); B
  plateaus (w ≈ 0.61, since ℓ is already at its floor by 0.60); C ∝ 1/‖a‖.
  `fit.json` SHA-256 `9dc94898c5ca…`, `map.json` `85f77113e485…`.
- `make check`: passed. `make test`: 326 passed.

**Next:** `support001 collect` (fresh six-dose families, disjoint from TRAIN-002), then
cache, gate and seal. Finish TRAIN-002 training, then its gate, seal and confirmation.

---

## 2026-10-01 — SUPPORT-001 — Fresh collection, gate, confirmation

**Actor:** Claude, for Omer (run on Omer's Mac).

**Commands/results:**
- `support001 collect` (6.4 min): 150 designed and 150 archive families with six doses.
  - Rejections: archive identity 1,507, not at rest 2,637, movement 233.
  - The registry excluded every earlier study, including the TRAIN-002 collections.
- `cache` (1.1 min).
- `gate` (real frames only): all six cells passed, with floors 0.147–0.216.
- `seal` was followed by `confirm` (38 s) and `report`. The verdict was
  `no_account_established`, the same for all three readouts.
- Over-range squared errors: N 1577, A 1183, B 1361, C 1171. C − A = −11.8
  [−21.9, −3.8], while C − B and A − B span zero.
- Block projection slope at 0.30 / 0.60 / 0.80 / 1.00:
  - designed: 0.93, 0.71, 0.67, 0.57;
  - archive: 0.88, 0.66, 0.52, 0.41.
- Agent probe: the predicted agent slope is 1.01, 0.96 and 0.96 (designed 0.60 and
  0.80, archive 0.60), against block slopes of 0.71, 0.67 and 0.66.
  - At 1.00 the real-frame agent readout compresses (0.66–0.69), so those cells are
    excluded by the real-frame band.
  - That band rule was extended to agent cells after the values were seen.
- Manifest `docs/manifests/support001-runtime.json`; draft `SUPPORT-001_RESULT_DRAFT.md`.

**Also:**
- Tectonic 0.17.0 was installed via Homebrew at Omer's request. It ran outside the sandbox,
  because the sandbox blocks writes to `/opt/homebrew` and its proxy breaks Tectonic's TLS.
  TeX packages are cached in the ignored `.cache/tectonic`.
- The draft `reports/report001/main.tex` compiles.
- `reporting.py` and its tests are added (26 macros, the instrument figure, and the
  claims ledger generator).

**Limitations:**
- The support covariate saturates beyond the support edge, which limits account B.
- One checkpoint, constant pushes, readout-mediated measurement.

**Next:** TRAIN-002 gate, seal, confirm and report, then the figures and results text.

---

## 2026-10-01 — TRAIN-002 — Training, gate, seal, confirmation

**Actor:** Claude, for Omer (run on Omer's Mac).

**Commands/results:**
- `train002 train`, 18:11–18:58. Six runs of 440–503 s each; all passed the optimization
  gate.
  - dose_omitted loss 0.093–0.108 → 0.024–0.026.
  - archive_only loss 0.011–0.012 → 0.005–0.006.
- `train002 gate` (real frames only): all four cells passed, floors 0.168–0.226, slopes
  within band.
- `seal` (14 models), then `confirm` (32 s), then `report`.
  - Primary `held_out_dose_deficit_supported`: dose_omitted − balanced at 0.60 is
    +0.82 [0.11, 1.63], the same sign for all readouts.
  - Transfer fraction 0.92 [0.85, 0.99].
  - In-support check at 0.45: +0.04 [−0.28, 0.36].
  - Generic `not_established`: frozen − archive_only −0.47 [−1.54, 0.63].
  - Ordinary balanced − frozen +1.08 [0.28, 1.91].
- Also added PAPER_PLAN.md (the approved conference roadmap; ICML 2027 with NeurIPS as
  fallback; GPU pretraining owned by Inbal on Colab).

**Limitations:** One checkpoint; constant pushes; the held-out dose is one step above the
trained range; dose_omitted has 48 rather than 64 effective examples per update.

**Next:** MEAS-FIX-001, then the MEAS-001 v2 engineering pass (D037), then REPORT-001.

---

## 2026-10-01 — MEAS-FIX-001 and MEAS-001 v2 engineering (D037)

**Actor:** Claude, for Omer (run on Omer's Mac).

**Changes:**
- MEAS-FIX-001 (commit `c5ce0fc`), with regression tests for each fix:
  - F1: inversions are bound to their query, config, seed, target, inputs, sources and
    checkpoint; family-hash checks added to `require_frozen`, `query_inputs` and `_labels`.
  - F2: `predictions_allowed` is derived from the full decision.
  - F3: classifier version 2, with independent final-pose support and defined nuisance
    outcomes.
  - F4: wording corrections.
  - A 120 s accounting correction, and a charged `score` stage.
- v2 search: independent per-search Nelder–Mead refinement, and a label-free ambiguity band
  (the least increase among 1.28-unit block offsets).
- Version 1 is preserved: tracked diagnostics are identical, and 72/72 classifications
  re-derive.
- D037 was recorded before the run.

**Commands/results:**
- `meas001 account-correction`: charged 120 s.
- `engineer-select-v2`: spec sealed.
- `engineer-v2` (23:29–23:37): `failed_selection`.
  - Phase A: 3/18 real queries accepted, 2/6 synthetic recovered.
  - Coverage 0.22 / 0.11; restart agreement 0.33 / 0.11.
  - 18 of 24 queries unresolved for restart disagreement; both searches converged in 23/24.
  - Median search separation 6.0 units; 0/24 reached the true-pose basin; accepted poses
    within 0.47–0.55 units.
- `propose-v2`: `not_ready`, SHA-256 `e5d10f867fe8…`. Engineering charge 1,748 s of the
  3,600 s sub-cap.
- `make test`: 343 passed.

**Decision:** stop MEAS-001 per D037; the 48 families stay closed. The paper's WP1 uses
its fallback (internal controls).

**Also:** TRAIN-002 is recorded (`2f94ab6`). The course report draft is complete apart
from the authors' contributions (3 pages; `8cae87b`). REPRODUCE.md is added (`daff191`).

**Next:** course slides and the polish pass. Then the GPU handoff for Inbal
(`pretrain001.py` and runbook, by Oct 20), MECH-001/002 and ENV-002.

---

## 2026-10-02 — REVIEW-CPU-001 — Evaluate the completed CPU study cycle

**Actor:** Codex, at Omer's request to evaluate whether the non-GPU work is complete.
Reviewed commit 9060a58; working tree initially clean.

**Changes:** Added CPU_STAGE_EVALUATION.md, REVIEW-CPU-001/REPORT-FIX-001 rows and a
current handoff to TASKS; corrected REPORT-001 status to reflect the existing draft and
remaining deliverables. No study source, protocol, checkpoint, sealed evidence or generated
report was changed. This review does not launch a new research work package.

**Commands/results:**
- `make check && make test`: 343 passed, no skips, 25 warnings (Gymnasium casts and
  joblib physical-core fallback), pytest 18.30 s; Python 3.10.21.
- `.venv/bin/python /tmp/non_gpu_review.py`: TRAIN-002 seal passes; 5,040 record hashes
  and unit/model/input/readout/fingerprint identities verified; full summary recomputes
  exactly. SUPPORT-001 seal, all 300 family files and real caches verified; primary
  account estimates and CI endpoints recomputed from saved latents with zero difference.
  Its verdict remains no_account_established.
- Same script: ROBUST-001 saved-output reproduction passes all 1,888 checks (maximum
  difference 8.881784197001252e-16); MEAS v1/v2 72+24 classifications and both diagnostic
  manifests reproduce exactly; 43 report macros, numbers.tex and claims ledger match.
- Read-only archive-map inspection: 399,758 windows, maximum mean-command magnitude
  0.8013369676; 413 above 0.60 and one above 0.80. A percentile is not a support boundary.
- Analytic account-B example: w=0.5 and m=(5,0), with true motion 10 versus 20, gives
  slopes 0.75 versus 0.625. Actual archive-0.60 B slope is 0.654153, whereas the report
  plots its weight 0.611026. Primary account tests use the correct full model.

**Assessment/findings:** Strong single-checkpoint empirical study and ample course
research content. Four P2 release findings: incorrect B figure, overclaimed confirmation/
readout independence/mechanism/null results, percentile-as-training-range language, and
Route B placing SUPPORT collection before its required TRAIN-002 collections. The 48/64
effective-example difference limits interpretation of the dose-omission contrast.
Current course reporting is unfinished; the paper roadmap has substantial remaining
CPU work as well as an unimplemented GPU handoff.

**Limits/next:** Saved-evidence reproduction, not independent model-inference or collection
replication. No new predictor inference/training, MEAS calibration, literature search or
PDF visual audit. Fix/report the verified evidence, finish slides/contributions and teammate
reproduction; retain MEAS's stop. Freeze a precise pretraining-selection rule and build
the GPU handoff before GPU execution. Review changes remain uncommitted.

---

## 2026-10-02 — REPORT-FIX-001 — Report corrections F1–F4

- **Date / task IDs / actor:** 2026-10-02 / REPORT-FIX-001, REPORT-001 / Claude, for Omer.
- **Intent and changes:** Fix the four release findings in CPU_STAGE_EVALUATION.md; no new
  experiment, sealed file or verdict changed.
  - F1: new `support001_derived.py` (D038) writes account-implied projection slopes and the
    archive command tail to `docs/manifests/support001-account-slopes.json`;
    `reporting.support_curves` plots those instead of B's mixture weight (A/C gains), and
    skips the undefined designed-0.15 cell. Annotation now "above account-fit range".
  - F2: `main.tex` abstract, contributions, Table 1, result headings and discussion
    rescoped: confirmatory vs exploratory/descriptive; "robust across readouts" (inversion
    failure noted); agent/block dissociation, not localization (post hoc gate noted);
    "substantially reduces"; "no generic fine-tuning gain established (not equivalence)";
    TRAIN-001 physical comparisons descriptive; 48-vs-64 effective-example caveat; added
    single-checkpoint and shared-infrastructure limitations.
  - F3: support paragraph now gives p99.9 = 0.60, 413 windows above, max 0.80 (new macros
    supMapPninetynine/supMapAbove/supMapMax) and says absence from pretraining is not shown.
  - F4: REPRODUCE.md uses stage-level order (TRAIN-002 audit + both collections before
    SUPPORT collect), separates asset rebuild from reproduction, drops the unmeasured
    2–3 h total, labels re-running a sealed stage as a provenance check, and adds
    saved-record and derived-manifest checks to the teammate subset.
  - Dated wording-correction notes added to TRAIN-002_RESULT, SUPPORT-001_RESULT,
    HANDOFF_2026-10-01 and STUDY_SYNTHESIS (evidence text preserved).
- **Commands and results:**
  - `.venv/bin/python -m jepa_action_effects.support001_derived` → completed in 8 s; seal
    verification passes; rerun output byte-identical.
  - `.venv/bin/python -m jepa_action_effects.reporting` → 46 numbers, 3 figures;
    `numbers.tex` and `REPORT-001_CLAIMS.md` byte-identical on rerun.
  - Tectonic compile → 3-page PDF; all pages rendered and inspected. Only red marker left
    is the Contributions placeholder (deliberately left for the authors).
  - `make check` passes; `make test` → 347 passed, 25 warnings.
- **Evidence:** support001-account-slopes.json records seal d381476c…, summary cc2ee512…,
  fit 9dc94898…, map c26c5bea….
- **Decision or deviation:** D038 (derived plotting summary; not a protocol change).
- **Blockers / uncertainty:** Contributions text pending from both authors. Teammate
  reproduction not yet run. Slides out of scope for this pass.
- **Next action / handoff:** Authors write Contributions; Inbal runs the REPRODUCE.md
  teammate subset and records it here; slides; final PDF check; instructor acceptance.

---

## 2026-10-13 — REPORT-001 — Final written-report pass

- **Date / task IDs / actor:** 2026-10-13 / REPORT-001 / Claude, for Omer.
- **Intent and changes:** Close the written report for submission (deadline 2026-10-15;
  only the report is due now, slides later). `reports/report001/main.tex` only:
  - Removed the Contributions placeholder and the `\pending` draft macro. Authors' decision:
    individual contributions are covered verbally in the presentation, not in the report
    or slides. The handout (pp. 1, 3) lists teammate contributions as a report item; this
    is the authors' call.
  - Added `\raggedbottom` (twocolumn implies `\flushbottom`, which stretched paragraph gaps
    on pages 1–2); ragged-right Table 1 cells; reworded the readout spec and Reproduction
    paragraph to remove stretched lines (`array.sty` was not in the Tectonic cache and the
    bundle host failed TLS through the sandbox proxy, so the table uses plain `\raggedright`).
  - Wording: abstract "into and beyond the archive's extreme command tail (archive-scene
    block slope … at command 1.00)"; "Across 399,758 sampled windows" (was "Over",
    ambiguous); discussion now says the model predicts the agent's own motion closely
    rather than "reads strong commands correctly" (the agent result is descriptive).
  - No number, figure, manifest, sealed file or claim status changed.
- **Commands and results:**
  - `git fetch --all`: branch `claude/youthful-clarke-akm28u` at `2635813` equals its
    remote; `main` (`59cfa74`) is an ancestor. Nothing to pull.
  - `.venv/bin/python -m jepa_action_effects.reporting` → 46 numbers, 3 figures; no tracked
    file changed.
  - Tectonic 0.17.0 compile → 3 pages; no errors or overfull boxes; all pages rendered and
    inspected; `pdftotext` finds no placeholder text.
  - `make check` passes; `make test` → 347 passed, 25 warnings.
- **Evidence:** PDF at `outputs/report001/pdf/main.pdf` (ignored; rebuild per REPRODUCE.md
  Route A).
- **Decision or deviation:** none on protocol. Contributions omission is an authoring choice.
- **Blockers / uncertainty:** instructor acceptance of the training scope remains unconfirmed;
  teammate reproduction not run.
- **Next action / handoff:** submit the PDF; build the .pptx slides for the talk; Inbal
  runs the REPRODUCE.md teammate subset and records it here.

---

## 2026-10-13 — REVIEW-COURSE-001 — Professor-style course assessment

- **Date / task IDs / actor:** 2026-10-13 / REVIEW-COURSE-001, REPORT-001 / Codex, for Omer.
- **Intent and changes:** Review completed work and report against the supplied Hebrew
  course handout with two days remaining; assess deferring Inbal's proposed GPU work.
  Added `docs/PROFESSOR_REVIEW_2026-10-13.md` and task-board review status. Provisional
  grade 88/100 for the written package under explicitly reviewer-selected weights;
  presentation/individual mastery not graded. Report, code and scientific outputs unchanged.
- **Commands and results:**
  - `git fetch origin` (approved escalation after sandbox denied FETCH_HEAD) → main
    advanced to merge 2e3af92. `git log -1 --oneline origin/main` and
    `git diff --stat HEAD origin/main` → merged tree identical to checked-out b6802a9.
  - `make check && make test` → checks passed; 347 passed, 25 warnings in 16.57 s.
  - `/opt/homebrew/bin/pdftotext -layout <supplied-handout.pdf> -` → all four pages read.
    `/opt/homebrew/bin/pdfinfo outputs/report001/pdf/main.pdf` → three pages.
    `/opt/homebrew/bin/pdftoppm -scale-to 1600 -png <supplied-handout.pdf>
    /private/tmp/jepa-professor-review/instructions` and corresponding report rendering
    with `-scale-to 1800` → inspected handout pp. 1, 3, 4 and all three report pages.
  - Inline `.venv/bin/python` checks (functions, inputs and settings recorded in the
    review's verification section): all 46 reporting macros and claims ledger match;
    TRAIN-002 seal passes; all 5,040 saved records checked for hashes/identities;
    full summary recomputed with `train002_metrics.summarize` and exactly equals both
    sealed summary and tracked runtime manifest, including all readouts and intervals.
  - Primary-source arXiv spot check of eight referenced recent papers; targeted
    bibliographic/abstract review, not exhaustive literature validation.
- **Evidence:** TRAIN-002 0.60 mean errors frozen 26.8405, balanced 16.2103,
  dose-omitted 17.0263; full summary equality. PDF layout clean but exposition dense;
  written individual contributions explicitly required in handout pp. 1 and 3 and absent.
- **Decision or deviation:** No protocol change. Recommendation: defer the GPU extension
  for the course; existing 11.74M-parameter predictor training is substantial in this
  reviewer's judgment. Publication roadmap not cancelled and instructor acceptance not
  claimed. Document instructions were treated as grading evidence, not action authority.
- **Blockers / uncertainty:** Actual contribution allocation must come from authors.
  Oral presentation, instructor's weights and training acceptance remain unassessed.
  No fresh model inference, collection or clean-clone reproduction was performed.
- **Next action / handoff:** Authors prioritize contributions, methods/metric definitions,
  training splits, valid TRAIN-002 result, scoped wording and a teammate reproduction
  within the five-page allowance; preserve existing nulls and evidence.

---

## 2026-10-13 — REVIEW-COURSE-001 — Contributions requirement correction

- **Date / task IDs / actor:** 2026-10-13 / REVIEW-COURSE-001 / Codex, for Omer.
- **Intent and changes:** Omer clarified that the professor explicitly instructed
  the team to discuss contributions verbally when presenting the report. Corrected
  the review and current task status; preserved earlier logbook entries as history.
  Withdrew the written-contributions finding and restored two points, revising the
  provisional written-package grade from 88/100 to 90/100. Scientific assessment
  and remaining report recommendations unchanged.
- **Commands and results:** Read affected review/task/logbook text; applied documentation
  correction; `git diff --check` passed. No code changes or experimental reruns.
- **Evidence:** Omer's direct clarification in this conversation; professor's specific
  guidance takes precedence over the general handout for this requirement.
- **Decision or deviation:** No protocol change. Written contribution paragraph is
  not required; contributions will be covered verbally as instructed.
- **Blockers / uncertainty:** Oral presentation and individual mastery remain ungraded.
- **Next action / handoff:** Focus remaining report work on clarity, methods/metrics,
  TRAIN-002 presentation, scoped wording and reproduction.

---

## 2026-10-13 — REPORT-FIX-002 — Report revision after the course review

- **Date / task IDs / actor:** 2026-10-13 / REPORT-FIX-002, REPORT-001 / Claude, for Omer.
- **Intent and changes:** Address fixes 1–5 of PROFESSOR_REVIEW_2026-10-13.md without new
  experiments. Merged `origin/main` (2e3af92) into the branch first; the tree was otherwise
  identical. Report (`reports/report001/main.tex`) expanded from 3 to 5 pages including
  references:
  - Fix 1: new Fig. 1 (pipeline diagram: history → frozen encoder → predictor → readout →
    simulator comparison, plus a PushT schematic using the measured 23.6-unit median and
    0.71 ratio); a concrete PushT paragraph in the motivation; experiments renamed by
    purpose (strong push, dose–response, beyond range, readouts, fine-tuning I/II,
    planning) with repository IDs in small type.
  - Fix 2: Table 1 (data: readout 50k/10k frames; 600 training / 120 validation designed
    families + 60 windows; 2,400 archive-only windows; test cohorts); Table 2 (four arms
    and dose weights; 48 vs 64 effective rollouts); readout training details (Adam 1e-3,
    batch 512, ≤300 epochs, patience 20, best-epoch restore, selection rule); the
    two-step autoregressive rollout and loss equation with frozen modules; definitions of
    projection slope, R, energy ratio (squared), endpoint error, transfer fraction,
    regret, floor; gate scope by study (floor-only for the strong-push test; band + ≥30
    or ≥50 moved families from the dose–response test onward).
  - Fix 3: the fine-tuning paragraph now leads with TRAIN-002 on fresh data and a new
    Table 4 of absolute 0.60 errors (frozen 26.84, archive-only 27.31, archive-weighted
    16.81, all-dose 16.21, ≤0.45 17.03), R per scene, ordinary-window errors (frozen 10.58,
    all-dose 11.67, real-frame decode 9.70), the in-support manipulation check
    0.04 [−0.28, 0.36] and archive-only − all-dose 11.1 [8.4, 14.1]. TRAIN-001 follows as
    descriptive with its failed development gate stated. New Fig. 5: training loss and
    validation latent MSE for the twelve saved runs (no new training).
  - Fix 4: "has no effect" → "No effect of making the controls constant within a block was
    established"; off-manifold sentence now reports flag rates (designed 0.12 → 0.04,
    archive 0.07 → 0.04) and says this does not show on-manifold; discussion now says
    "decoded agent motion … has aggregate slopes near one in the gated cells"; abstract
    says "decoded block displacement". Strong-push cohort described accurately as constant
    0.60 pushes at lateral offsets (the earlier draft of this revision said "up to 0.60").
  - Fix 5: new `train002_recompute` module (REPRODUCE.md step 4 is now one command);
    new `training_curves` module exporting hash-checked curves to
    `docs/manifests/training-curves.json`; `reporting` gains 29 macros (75 total) and two
    figures; tests added. Sealed sources (`train001*`, `train002*`, `model_inference`,
    `next_common`) untouched.
- **Commands and results:**
  - `.venv/bin/python -m jepa_action_effects.train002_recompute` → `"status": "match"`,
    5,040 records, 14 models, full summary equal to both `outputs/train002/confirm/summary.json`
    and `docs/manifests/train002-runtime.json` (exact, ~8 s).
  - `.venv/bin/python -m jepa_action_effects.training_curves` → 12 runs, every `run.json`
    SHA-256 equal to the TRAIN-002 registry `run_sha256`.
  - `.venv/bin/python -m jepa_action_effects.reporting` → 75 numbers, 5 figures.
  - `make check` passed; `make test` → 354 passed, 25 warnings.
  - Tectonic compile → 5 pages, no overfull boxes, all pages rendered and inspected.
    Compilation needed network access to `relay.fullyjustified.net` for missing TeX
    fonts; the session sandbox's proxy breaks Tectonic's TLS, so the compile ran outside
    the sandbox. A TikZ version of Fig. 1 was replaced by a matplotlib figure so Route A
    needs no extra TeX packages.
  - Clean-clone Route A dry run (local `git clone` at 36e70e2, then dd5a6c9; repository
    `.venv` with the clone's `src` on `PYTHONPATH`): `numbers.tex` byte-identical, claims
    ledger git diff empty, PDF 5 pages with text identical to the working copy. It found
    that Tectonic refuses a missing `outputs/report001/pdf`; fixed in `reporting.build`
    and REPRODUCE.md (dd5a6c9).
- **Evidence:** commits 36e70e2, dd5a6c9; `outputs/report001/pdf/main.pdf` (ignored);
  `docs/manifests/training-curves.json`; `docs/REPORT-001_CLAIMS.md`.
- **Decision or deviation:** No protocol, sealed artifact or estimate changed. New
  macros only expose existing manifest fields; validation-curve numbers appear only in
  the figure. The dry run is Claude's on Omer's machine, not the teammate reproduction.
- **Blockers / uncertainty:** Inbal's clean-clone run (REPRODUCE.md steps 1–6) is still
  needed for the handout's reproduction item; `make setup` on a truly fresh machine was
  not exercised here. Whether the five-page limit includes references is not stated in
  the handout; the PDF is five pages including them. Fig. 1 text is small (~5.5 pt) at
  print size.
- **Next action / handoff:** Inbal runs REPRODUCE.md steps 1–6 and records results; Omer
  reads the PDF end to end and exports it; slides for the later talk.

---

## 2026-10-13 — REVIEW-COURSE-002 — Review revised report on merged main

- **Date / task IDs / actor:** 2026-10-13 / REVIEW-COURSE-002 / Codex, for Omer.
- **Intent and changes:** Reassess REPORT-FIX-002 after merge. Added
  `docs/PROFESSOR_REVIEW_2026-10-13_FOLLOWUP.md` and task-board status; provisional
  written-package grade 91/100. Contributions remain verbal per professor's instruction.
  Report, production code and scientific evidence unchanged.
- **Commands and results:**
  - `git fetch origin` → ba25aed (merge PR #2); `git diff --stat HEAD origin/main`
    empty against checked-out b3123a5; initial tree clean.
  - `make check && make test` → checks pass; 354 passed, 25 warnings in 17.32 s.
  - `.venv/bin/python -m jepa_action_effects.train002_recompute` → match, 5,040
    records, 14 model/control entries; sealed and tracked full summaries equal.
  - Inline `.venv/bin/python`: `reporting.numbers/macros_tex/claims_md` match all 75
    saved macros and ledger; `training_curves.collect` verifies all 12 run hashes and
    equals the tracked manifest. `reporting.build(Path('/private/tmp/jepa-review-round2/assets'))`
    → 75 numbers, five figures; PDF output directory created. Nonfatal font-cache warnings.
  - `pdfinfo outputs/report001/pdf/main.pdf` → five pages; `/opt/homebrew/bin/pdftoppm
    -scale-to 1800 -png outputs/report001/pdf/main.pdf /private/tmp/jepa-review-round2/report`
    → all five pages inspected, no clipping/overlap; Figure 1 text small.
  - Targeted code/protocol checks against D023/D026 and saved ROBUST-001 evidence
    support F1–F4. `git diff --check` passes after review-document edits.
- **Evidence:** Current main.tex misstates SHIFT-002 pooled gates as per-dose + floor;
  blanket moved-family metric scope conflicts with unmasked CALIB/SUPPORT estimates;
  real-slope normalization does not establish removal of readout bias; lower flag rates
  do not prove flagged points cannot drive the trend (unflagged challenge ridge CI
  −0.36 [−0.88, 0.09]); 0.45 deficit interval spanning zero is not an absence claim.
- **Decision or deviation:** No protocol change, inference, training or collection.
  Recommend small reporting corrections, not more GPU work. Clarified that the
  handout requires reproducibility but not this particular teammate-run procedure.
- **Blockers / uncertainty:** Oral defense and fresh-machine setup unassessed;
  existing shared readout/single-checkpoint limitations remain.
- **Next action / handoff:** Address F1–F4, improve Figure 1 readability, perform a
  practical reproduction handoff, and inspect the final exported PDF within five pages.

---

## 2026-10-13 — REPORT-FIX-003 — Follow-up review corrections

- **Date / task IDs / actor:** 2026-10-13 / REPORT-FIX-003, REPORT-001 / Claude, for Omer.
- **Intent and changes:** Apply PROFESSOR_REVIEW_2026-10-13_FOLLOWUP.md on branch
  `review_v2` (69c005c). Report text only, plus Figure 1 and one new macro.
  - F1: measurement gates described per study. Strong-push: floor < 0.25 gates every claim.
    SHIFT-002: no floor; real-frame slope CI in (0.85, 1.15) per scene × structure with
    doses pooled, ≥ 30 moved families in four designated groups, two manipulation checks
    (verified in `shift002_metrics.apply_dose_criteria`). TRAIN-001/002 and SUPPORT-001:
    designated cells (0.45/0.60; 0.60/0.80/1.00) on floor, band and ≥ 50 moved. TRAIN-001
    development failure kept; undesignated doses stated as ungated.
  - F2: "moved" defined once; slopes/ratios for strong-push, dose–response and
    fine-tuning are moved-only; SUPPORT slopes and the energy ratio use all branches;
    endpoint error uses all accepted families (checked in `next_metrics`,
    `calib001_metrics.estimates`, `train001_metrics`). R "normalizes by the real-frame
    slope … readout bias specific to predicted latents may remain"; slope described as a
    projection gain that does not rule out orthogonal errors; "1 = faithful" replaced in
    the abstract, Fig. 2 and Table 4.
  - F3: removed "not driven by"; added the unflagged-only check as inconclusive, citing
    new macro `robustOnChallengeRidge` = −0.36 [−0.88, 0.09] from
    `robust001-runtime.json` → `verdicts/contrasts/challenge:energy_ratio:on/ridge`.
  - F4: "no deficit was established at the in-support 0.45 dose".
  - Polish: Figure 1 redrawn in a Times-style font at ~7 pt final size, with the real
    future frame routed explicitly through the same encoder and readout; panel (b)
    labelled "schematic, not to scale" and the caption says R is a family-level
    aggregate, not a representative prediction (the results text likewise). Table 1
    caption now allows the deliberate reuse of TRAIN-001 training data and models in
    TRAIN-002 while stating that test cohorts are disjoint.
- **Commands and results:** `.venv/bin/python -m jepa_action_effects.reporting` → 76
  numbers, five figures; `make check` passed; `make test` → 354 passed, 25 warnings;
  Tectonic compile (outside the sandbox, which breaks its TLS) → 5 pages, no overfull
  boxes; all five pages rendered and inspected.
- **Evidence:** `reports/report001/main.tex`, `src/jepa_action_effects/reporting.py`,
  `docs/REPORT-001_CLAIMS.md`; `outputs/report001/pdf/main.pdf` (ignored).
- **Decision or deviation:** No protocol, sealed artifact or estimate changed; the new
  macro exposes an existing manifest field.
- **Blockers / uncertainty:** Fresh-machine `make setup` and a teammate run remain
  unverified; oral defense ungraded.
- **Next action / handoff:** Optional fresh-machine reproduction; read and export the final
  PDF; slides.

---

## 2026-10-13 — REVIEW-COURSE-003 — PR #3 verification

- **Date / task IDs / actor:** 2026-10-13 / REVIEW-COURSE-003 / Codex, for Omer.
- **Intent and changes:** Review open PR #3 at efff5d3 against main ba25aed. Attached
  PR to the chat; appended verification to the follow-up review and task handoff.
  Earlier corrections verified; one remaining P2 method-definition issue (F5).
- **Commands and results:**
  - `gh pr list --state open --json number,title,url,headRefName,baseRefName` and
    `gh pr view 3 --json number,title,url,headRefOid,baseRefOid,mergeable,reviewDecision,statusCheckRollup,body`
    → PR #3, head efff5d3, base ba25aed, mergeable, no checks listed.
  - `git fetch origin`; `git rev-parse HEAD origin/review_v2` → identical efff5d3.
  - `make check && make test` → 354 passed, 25 warnings, 16.18 s pytest.
  - Inline Python: 76 `reporting.numbers()` macros and ledger exactly match saved
    files; every prior ledger line remains; added unflagged-ridge result verified.
    `reporting.build(Path('/private/tmp/jepa-pr3-review/assets'))` → five figures.
  - `pdfinfo outputs/report001/pdf/main.pdf` → five pages; `pdftoppm -scale-to 1800
    -png outputs/report001/pdf/main.pdf /private/tmp/jepa-pr3-review/report` with
    /opt/homebrew/bin tools → all pages inspected, no clipping/overlap.
  - `git diff --check origin/main...HEAD` and final local `git diff --check` pass.
- **Evidence:** `next_metrics.floor_ratio` uses family-mean real-frame paired-effect
  error / median meaningful designated-pair effect (PHYS-002 result 0.120), whereas
  `train001_metrics.cell_gate` uses displacement error / median moved displacement.
  main.tex:231–233 still conflates them. Earlier review missed this distinction.
- **Decision or deviation:** No protocol/numerical change. Approximate written-package
  grade 92/100; oral defense ungraded. No posted GitHub review, merge, report edit,
  new model inference or training. Saved-record recomputation not repeated because
  this patch changes neither estimates nor estimators.
- **Blockers / uncertainty:** One report-definition correction; fresh-machine setup
  and presentation remain unverified.
- **Next action / handoff:** Correct F5, inspect final PDF, then finish submission
  and presentation preparation; no additional GPU experiment needed.

---

## 2026-10-13 — REPORT-FIX-003 — F5 floor definition (PR #3 review comment)

- **Date / task IDs / actor:** 2026-10-13 / REPORT-FIX-003 / Claude, for Omer.
- **Intent and changes:** PR #3 review comment 6139076279 (F5): the report applied the
  later displacement-floor definition to the strong-push study. Verified in code:
  `next_metrics.floor_ratio` uses `boundary_meaningful` pairs (designated branch pairs
  with true effect > 5.12), family-mean real-frame paired-effect error over the median
  true effect magnitude (PHYS-002 value 0.121); `train001_metrics.cell_gate` uses mean
  real-frame displacement error over all accepted families over the median moved true
  displacement. `main.tex` now names a *paired-effect floor* (strong-push) and a
  *displacement floor* (fine-tuning, beyond-range), both below 0.25, notes that a shared
  offset cancels in paired effects, and calls TRAIN-001's failed value a displacement
  floor. SHIFT-002 (no floor), designated cells and TRAIN-001 failure unchanged.
- **Commands and results:** `make check` passed; `make test` → 354 passed; Tectonic
  compile (outside sandbox) → 5 pages, no overfull boxes; pages 3 and 5 inspected after
  the change, earlier pages unchanged in layout.
- **Evidence:** `reports/report001/main.tex`; no macro, manifest or sealed file changed.
- **Decision or deviation:** Documentation correction only.
- **Blockers / uncertainty:** None new.
- **Next action / handoff:** Merge PR #3 after review; final exported-PDF read; slides.

---

## 2026-10-13 — REPORT-REDESIGN-001 — Rewrite and redesign the course report

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001, REPORT-001 / Claude, for Omer.
- **Intent and changes:** Omer found the report hard to follow (unintuitive figure placement,
  language errors, numbers without a story, unmotivated experiments, undefined data terms,
  a sense that nothing was achieved). Approved plan: same evidence, told as a question →
  answer story, two-column layout with a page-1 teaser, real test examples.
  - D039 (before any export): example rule (designed family whose true block displacement is
    closest to the cohort median; SUPPORT-001 at 1.00, TRAIN-002 at 0.60; ties to the lowest
    id; never model error) and the report vocabulary (archive → training demos, frozen model →
    released LeWM, dose → push strength, windows → ordinary demo clips).
  - New read-only `report_examples.py`: verifies both seals and all input hashes, reads saved
    states, frames and decoded ends (SUPPORT-001 via the sealed readout on saved latents, as
    D038), writes `docs/manifests/report-examples.json`, 11 frame PNGs in
    `docs/figures/report-examples/`, a demo command histogram and the share of agents that
    stay inside the table. Six tests.
  - `reporting.py`: one figure system (exact print sizes, STIX/Times font, Okabe–Ito semantic
    palette, direct labels) and four figures replacing five: teaser (real example, gain vs
    strength over the demo-command histogram, error by training arm), method (real frames,
    pipeline, metric geometry), robustness, fine-tuning (R by strength, example, validation
    curves). 35 new manifest-backed numbers (`count`/`percent` kinds, list indexing); every
    macro is set in math mode (true minus signs) with a point-estimate twin for intervals.
  - `main.tex` rewritten: findings up front, questions Q1–Q3, experiments E1–E7 as a chain
    of questions (Table 1 with status tags), Key terms box, physics of the data explained
    (block has no momentum; true displacement is constant from 0.60 to 1.00 while the
    imagined one shrinks), nulls reported as bounds, Insights paragraph. All F1–F5 review
    corrections kept; claims re-checked against the claims ledger.
- **Commands and results:**
  - `.venv/bin/python -m jepa_action_effects.report_examples` → s1-challenge-077 and
    challenge-conf-059, 11 frames; rerun byte-identical (manifest and PNG SHA-256 equal).
  - `.venv/bin/python -m jepa_action_effects.reporting` → 111 numbers, 4 figures; all 76
    earlier macro values unchanged (82 macros added).
  - Tectonic 0.17 (inside the sandbox; cached packages sufficed) → 5 pages including
    references, no overfull boxes or undefined references; every page rendered and inspected
    in colour and greyscale; `pdftotext` proofread and jargon scan.
  - `make check` passed; `make test` → 362 passed, 25 existing warnings.
- **Evidence:** `reports/report001/main.tex`, `src/jepa_action_effects/{reporting,report_examples}.py`,
  `docs/manifests/report-examples.json`, `docs/figures/report-examples/`, D039;
  `outputs/report001/pdf/main.pdf` (ignored).
- **Decision or deviation:** No protocol, sealed file or estimate changed. The example family
  of SUPPORT-001 is more extreme than average; the caption says so and points to the averages.
  The saturation of true displacement and the agent-in-view shares are descriptive.
- **Blockers / uncertainty:** Omer's own read of the PDF; teammate clean-clone run still open.
- **Next action / handoff:** Omer reads and exports the PDF; slides can reuse the four figures.

---

## 2026-10-13 — REPORT-REDESIGN-001 — Figure 1(a) redesign after Omer's review

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001 / Claude, for Omer.
- **Intent and changes:** Omer questioned Fig. 1(a). The three tight crops showed only
  fragments of the (upside-down) T, never the agent, and LeWM's imagined motion as a stub of a
  few pixels. Replaced with an overview plus magnifier from the same D039 example and data:
  left, the whole scene to scale (block, agent, its stops at 0.30 and 0.60, leaving the table at
  1.00, end pose outline, zoom box); right, the block's end points magnified (true, imagined,
  real-frame reading per strength, with a 10-unit scale bar). Caption rewritten and shortened.
  No numbers, data or other figures changed.
- **Commands and results:** `reporting` → 111 numbers, 4 figures; Tectonic → 5 pages, no
  overfull boxes; page 1 inspected; `make check` passed; `make test` → 362 passed.
- **Next action / handoff:** Omer rereads page 1.

---

## 2026-10-13 — REPORT-REDESIGN-001 — Figure 1(a) simplified (supersedes the magnifier)

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001 / Claude, for Omer.
- **Intent and changes:** Omer found the overview-plus-magnifier version of Fig. 1(a) worse:
  overlapping block outlines and zoom box, nine arrows and labels in one small panel. Replaced
  with two simple parts from the same D039 example: the real start frame (agent, block, push
  direction) and a bar chart of how far the block moves along its true direction at 0.30, 0.60
  and 1.00, true (black) vs imagined by LeWM (vermilion), with the real-frame reading as
  circles. Unused scene/magnifier helpers removed; caption rewritten. No data changed.
- **Commands and results:** `reporting` → 111 numbers, 4 figures; Tectonic → 5 pages, no
  overfull boxes; page 1 inspected; `make check` passed; `make test` → 362 passed.
- **Next action / handoff:** Omer approves; then push and open a follow-up PR (PR #4 merged).

---

## 2026-10-13 — REPORT-REDESIGN-001 — Push-strength definition corrected

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001 / Claude, for Omer.
- **Intent and changes:** Omer asked how push strength is defined. The report wrongly called
  1 "the maximum" (intro and Key terms box). Each command part is limited to [−1, 1], so
  diagonal commands reach √2; 1 is the largest length allowed in every direction. Corrected
  both places, and §3.1 now states that the simulator only receives (x, y) commands and that
  push strength (the length of one command held for a second) is our own definition. Small
  wording trims in §6–§7 keep the report at five pages; the Reproduction paragraph now points
  to `REPRODUCE.md` only. No data, numbers or figures changed.
- **Commands and results:** Tectonic → 5 pages, no overfull boxes; page 5 inspected.
- **Next action / handoff:** merge PR #5.

---

## 2026-10-13 — REPORT-REDESIGN-001 — Related work tightened

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001 / Claude, for Omer.
- **Intent and changes:** Omer asked to tighten related work; we have not reproduced any of the
  cited recent results. The paragraph now cites only the works the report builds on or
  positions against (Intervention Gap, AD-WM, Do-JEPA, Kang et al., PhysWeep, whose held-out
  account comparison we adopted), describes them as proposals or reports rather than stating
  their findings, and says that most are recent preprints whose results we have not reproduced.
  Sensitivity Shaping, IMPLY and ARC-Bench were removed (11 references, was 14). Earlier
  primary-source checks confirmed only that the papers exist, not their results.
- **Commands and results:** Tectonic → 5 pages, no overfull boxes or undefined references.
- **Next action / handoff:** merge PR #6.

---

## 2026-10-13 — REPORT-REDESIGN-001 — Clarity pass for a reader new to the material

- **Date / task IDs / actor:** 2026-10-13 / REPORT-REDESIGN-001 / Claude, for Omer.
- **Intent and changes:** Read the report as an unfamiliar grader and against the handout
  criteria. Fixes: baselines named (released model = baseline, real-frame reading = reference,
  ordinary-data fine-tuning = control); the introduction states our training work (readout +
  twelve fine-tuning runs of the 11.7M-parameter predictor); undefined terms removed or
  defined ("gain" in the abstract, "push families", "arm" and the weights w_m, privileged goal
  images, interval notation, table units with table/block sizes); Table 1's check column
  rewritten in plain words; "accounts" → explanations, with the squared-error contrasts
  explained; E2's constant-command test motivated; PD controller and related-work jargon
  explained. Background condensed and limitations merged with next steps to stay at five
  pages. No data, numbers or figures changed.
- **Commands and results:** Tectonic → 5 pages, no overfull boxes or undefined references;
  page 3 inspected.
- **Next action / handoff:** merge PR #6.

---

## 2026-10-14 — REVIEW-COURSE-004 — Final report redesign review

- **Date / task IDs / actor:** 2026-10-14 / REVIEW-COURSE-004, REPORT-001 / Codex, for Omer.
- **Intent and changes:** Reviewed the final submission at 6aec9fc against the handout,
  earlier corrections and saved evidence. Wrote PROFESSOR_REVIEW_2026-10-14.md and updated
  TASKS; did not change the report, experiment code, manifests, models or sealed evidence.
  Approximately 92/100 for the written package; oral presentation ungraded and
  contributions verbal per the professor. Two P2 wording findings: unchanged medians
  do not establish identical demo-family effects, and shared probes/agent gains cannot
  exclude measurement or action-path explanations. One P3 caption: 120 validation
  families means 60 designed plus 60 demo.
- **Commands and results:** `git fetch origin`; `git diff --stat HEAD origin/main` empty
  (main 87f1e9e). `git diff --check 10770ff..HEAD` passed. `make check && make test` passed:
  362 tests, 25 existing warnings, 18.38 s. `git archive HEAD` to a temporary clean source
  export, then `PYTHONPATH=<export>/src .../.venv/bin/python -m jepa_action_effects.reporting`
  there produced 111 numbers/four figures; `cmp` of numbers.tex, claims ledger and inputs.json
  passed. Tectonic compile using the existing cache succeeded; `pdfinfo` reports five pages.
  `pdftotext` output and PIL pixel comparisons of all five 1800-pixel Poppler renders match
  the current PDF exactly; all original pages visually inspected.
- **Evidence:** Review doc contains the exact read-only F1 reproduction command, which
  verifies all 300 SUPPORT family file hashes. Demo mean true displacement rises
  84.7732 to 96.4449 at 0.60 to 1.00; 34/150 demo endpoints change by more than 5.12 units.
  Designed means 24.0879 to 24.0008, all endpoint changes below 1.63. These are descriptive
  audit facts, not new confirmatory evidence. D039 example selections reproduce, all eleven
  citations checked against primary sources, and validation count checked in tracked manifests.
- **Decision or deviation:** Review only; no protocol change. Defer additional GPU research
  for the course submission. Previous F5 is addressed; these findings concern the later rewrite.
- **Blockers / uncertainty:** Fresh-source Route A reused the locked local environment and
  TeX cache; it is not a teammate/fresh-machine reproduction or an experiment rerun. The
  remaining limitations are shared probes, one checkpoint/environment/horizon and the
  omitted-strength arm's effective batch-size confound.
- **Next action / handoff:** Implement F1–F3, rebuild/inspect the five-page PDF, and perform
  the authors' final read before submission. Findings are suitable for the implementing agent.
  Posted with `gh pr comment 6 --body-file /private/tmp/jepa-final-review-20261014/pr-review.md`:
  https://github.com/OmerTorres/jepa-action-effects/pull/6#issuecomment-6160331296.
  Local review notes remain uncommitted; the report is unchanged.

---

## 2026-10-14 — REPORT-FIX-004 — Apply REVIEW-COURSE-004 corrections

- **Date / task IDs / actor:** 2026-10-14 / REPORT-FIX-004, REPORT-001 / Claude, for Omer.
- **Intent and changes:** Applied PROFESSOR_REVIEW_2026-10-14.md after verifying each claim on
  the saved SUPPORT-001 families (demo mean true displacement 84.77 → 96.44 from 0.60 to 1.00,
  34/150 demo endpoints change by > 5.12; designed means 24.09 → 24.00, none change) and the
  TRAIN-001 manifest (validation 60 designed + 60 demo).
  - F1: the unchanged-effect claim is scoped to designed scenes in §3.1, E3 and the discussion;
    E3 now says the median barely changes and that demo families can move further.
  - F2: findings headline "The trend is robust across readouts and differs between agent and
    block"; E4 heading "the trend survives alternative readouts"; Table 1 E4 answer "All five
    readouts agree; shared bias remains possible". E5 made consistent (heading "agent and block
    behave differently"; an aggregate agent gain cannot rule out command-processing problems).
  - F3: Fig. 4(c) caption and the `reporting.py` comment now say 60 designed + 60 demo.
  - Optional: REPRODUCE.md no longer claims every export input hash is checked (the SUPPORT-001
    prediction file's digest is recorded, not compared).
  - Caption and sentence trims keep five pages. No numbers, data or figures changed.
- **Commands and results:** Tectonic → 5 pages, no overfull boxes or undefined references;
  `make check` passed; `make test` → 362 passed.
- **Next action / handoff:** merge the follow-up PR; authors' final read and submission.

---

## 2026-10-14 — REPORT-CLARITY-001 — Explain the study before reporting its results

- **Date / task IDs / actor:** 2026-10-14 / REPORT-CLARITY-001, REPORT-001 / Codex, for Omer.
- **Intent and changes:** Omer could not explain gain, command magnitude, contact effects
  or why five/ten controls and a one-second horizon were used. Rewrote the abstract and
  introduction; defined gain in the first figure caption with a simple example; made the
  controller/magnitude distinction and lack of assumed monotonic block motion explicit.
  Added a timing/rationale paragraph: inherited five controls per native prediction step
  vs chosen two-step horizon, retained after the simulator-only pilot (88/120 meaningful
  families, above the predeclared 20% extension trigger). No horizon optimality claimed.
  Reduced duplicate findings/definitions to preserve five pages. Added
  UNDERSTANDING_THE_STUDY.md with source-backed explanations, worked gain examples,
  design limits and questions the authors should be able to answer.
- **Evidence checked:** Pinned simulator env.py `step` and dt/control settings;
  upstream pinned `config/eval/pusht.yaml` (`action_block: 5`); checkpoint config
  (`num_frames=3`, action width 10); `model_inference._model_actions`;
  `pilot_collection._meaningful_family` and horizon rule; D009 and
  `docs/manifests/pilot001-runtime.json`. Pilot meaningful effects are pairwise endpoint
  separations over 5.12, not the later moved-branch definition. Readout failure in that
  pilot remains disclosed in the understanding guide.
- **Commands and results:** Tectonic `-X compile main.tex --outdir ../../outputs/report001/pdf`
  with existing local cache; first expanded draft was six pages, then duplicate exposition
  was shortened without shrinking type. Final five-page PDF inspected using `pdftoppm`
  at 1600 pixels; no clipping/overlap, overfull boxes or undefined references.
  `reporting.numbers`, `macros_tex`, `claims_md` verified all 111 existing generated values
  and ledger unchanged. `git diff --check` passed. No executable code changed; suite not
  rerun (previous independent check: 362 passed).
- **Decision or deviation:** Explanation only, not a new protocol or experiment. D009
  supports feasibility/retention of one second; it does not establish an optimum. The
  prior submission-readiness statement assessed artifact/evidence consistency, not the
  authors' ability to defend the work; that remains pending.
- **Next action / handoff:** Authors read the revised opening and understanding guide,
  explain one family and the timing/metric choices in their own words, then decide on
  submission. Changes remain local and uncommitted; no external comment or submission made.

---

## 2026-10-14 — REVIEW-LEAKAGE-001 — Audit the omitted-magnitude result

- **Intent:** Omer questioned whether improvement at 0.60 without training on that
  magnitude indicates leakage or an implausible claim of beating LeWM. Applied the
  jepa-study interpretation workflow; no training or frozen-protocol changes.
- **Evidence:** TRAIN002_LEAKAGE_AUDIT_2026-10-14.md records scope, findings and limits.
  Independent pairwise manifest identity checks found no overlaps. Source-file hashes
  and earlier-study exclusions passed for 1,140 TRAIN-001 and 2,760 TRAIN-002 records.
  Actual-model gradient checks on two training families and three dropout seeds showed
  bitwise identical gradients in all 93 tensors after replacing omitted-branch actions
  and targets. All three omitted runs have the original base digest, zero 0.60 weight,
  matching development provenance and fixed final checkpoints. Six fresh model/family
  inference comparisons matched saved endpoints within 0.00003891 units.
- **Interpretation correction:** 0.60 informed family admission and the inherited
  learning-rate choice and appears in validation. It is omitted from the production
  training loss, not the whole development process. Corrected abstract, captions,
  findings and discussion and added this qualification to methods and the understanding
  guide. Preserved existing uncommitted clarity changes. No scientific estimate changed.
- **Exploratory observation:** In all 300 designed training and 120 designed confirmation
  families, 0.45 and 0.60 endpoints differ by less than 5.12; means 0.281 and 0.298 units.
  Demo-derived fractions within that distance are 66.7% and 58.3%. This makes transfer
  plausible without establishing its learning mechanism; not a new confirmatory claim.
- **Commands/results:** Exact audit commands and ignored scripts/results are listed in
  the audit document (`outputs/leakage_audit_20261014/`). `UV_CACHE_DIR=/private/tmp/jepa-uv-cache
  uv run --locked --no-sync python -m jepa_action_effects.train002_recompute` reproduced
  all 5,040 records' summaries and intervals exactly. Targeted pytest command in the
  audit document: 59 passed in 3.76 s. `make check` passed. Tectonic with the existing
  `.cache/tectonic` compiled the report; five pages. Pages 4–5 visually inspected for
  methods/results/discussion placement. `git diff --check` passed.
- **Limits/next action:** No direct leakage found within the audited paths, not a proof
  against every near-duplicate or upstream pretraining overlap. Training history was
  not independently rerun. Authors should describe narrow adaptation with ordinary-data
  tradeoff, not general superiority or an entirely unseen magnitude. Changes remain
  local and uncommitted; audit scripts/results are ignored local artifacts.

---

## 2026-10-14 — REPORT-COMMENTS-001 — Address twelve annotated PDF comments

- **Intent and changes:** Applied Omer's twelve annotations to the report source and
  Figure 2 generator, preserving earlier uncommitted clarity and leakage-audit changes.
  Removed the contact-geometry aside from the abstract (1); expanded the common crop
  to include every final block outline and full agent disk (2); replaced "corroborate"
  and trimmed the repeated findings paragraph (3); kept the expanded PD name without
  the damped-spring analogy (4); explained the pilot-based horizon choice without a
  repository decision reference (5); replaced the 400m formula with the measured
  0.45-to-0.60 example (6); removed the unestablished contact-duration explanation (7).
  Defined seed, warm-up, branch and physics substep at first use (8–9); separated constant
  commands from scaled recorded sequences and defined endpoint-based moved labels (10);
  added the readout architecture diagram and six output meanings (11); moved the key-terms
  box to page 1 before the introduction, adding latent/readout and branch (12).
- **Factual correction:** Demo scene collection checks block positions (`states[:11,2:4]`)
  with `is_rest_window`, not agent rest. Corrected the report to "block is nearly
  stationary". Inspected TRAIN-001, SHIFT-002 and SUPPORT-001 collection paths. This
  corrects exposition; it does not change the protocol or any saved outcome.
- **Figure evidence:** In the chosen TRAIN-002 example, the m0.60 agent centre is
  (142.339, 224.523), within the table but outside the old block-centred crop's left
  edge (~159.184). The new crop is shared across all four panels and includes the
  radius-15 disk. Frames and example-selection rules are unchanged.
- **Commands/results:** `UV_CACHE_DIR=/private/tmp/jepa-uv-cache MPLCONFIGDIR="$PWD/.cache/matplotlib"
  uv run --locked --no-sync python -m jepa_action_effects.reporting` regenerated four
  figures and 111 numbers; the tracked claims ledger is unchanged. `make check` passed.
  `UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync pytest
  tests/test_reporting.py tests/test_report_examples.py`: 16 passed in 14.83 s.
  Tectonic initially failed because underbrace notation requested an uncached font;
  replaced it with ordinary boxed labels. The expanded draft was six pages; shortened
  repeated findings/background/discussion prose without reducing body font size.
  Final report is five pages, with all pages rendered using `pdftoppm` and visually
  checked. Compile and render logs are under `outputs/report001/comment-review/`.
  Final `git diff --check` passed.
- **Limits/next action:** No model, simulator, protocol, data or scientific result
  changed. Only report text and the display crop changed. Authors review the updated
  PDF; changes remain local and uncommitted.

---

## 2026-10-14 — REVIEW-E7-001 — Retention recommendation for the planning result

- **Intent:** Omer asked whether E7 is unfinished exploration or a completed experiment
  whose inconclusive result merits reporting. Reviewed D024–D026, relevant research-plan
  scope, H3-PLAN-002_RESULT (including its corrections), planning002 implementation and
  the report's E7 paragraph.
- **Evidence/results:** Read `outputs/planning002/confirm.json` through `next_common.sealed`
  using `UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync python` and
  compared it to `docs/manifests/planning002-runtime.json['confirm']`: exact equality;
  status complete. 150 reserved families yielded 148 with valid goals (361 goals).
  Both pooled checks passed. Predicted-minus-real-latent regret was
  2.0674 [-1.3240, 5.5878]; real-latent-minus-random was -6.0338 [-9.2848, -2.4445].
  This verifies the saved result and its provenance sidecar, not a fresh inference or
  bootstrap rerun. No new tests were needed for this interpretation-only review.
- **Recommendation:** Keep E7 briefly as a completed, limited action-selection test.
  The exploratory development effect did not establish a positive penalty on fresh
  confirmation, which bounds the report's conclusions. It is not a full planner
  benchmark, a block-error-specific causal test, or evidence that prediction errors
  are harmless. Candidate-specific goal images use actual future agent pose and block
  angle; the test has six choices and a short horizon. Recommend a precise title and
  plain-language explanation rather than deleting an inconclusive result.
- **Next action:** Explain this distinction to the authors. Report/PDF and scientific
  artifacts remain unchanged in this turn; no commit or external action.

---

## 2026-10-14 — REPORT-TABLE-CLARITY-001 — Explain E2/E3 checks and R

- **Intent and changes:** Addressed three new annotations on Table 1. Replaced the
  unexplained E2 R statement with the physical finding: larger commands worsen motion
  underestimation in both scene types. Described real-image motion recovery in percent,
  replaced "moved" shorthand with moving-family counts, and made E3 checks explicitly
  per scene and strength. Caption defines R, moving (>5.12 units), and the requirement
  that the whole 95% gain interval fit inside 0.85–1.15. Named E2's two baseline checks
  and its four count groups; kept strengths pooled for E2's real-image gain checks.
- **Evidence:** Inspected `shift002_metrics.apply_dose_criteria`, its statistics/count
  groups and SUPPORT gate source references. No threshold, result or protocol changed.
  E7 source text/row is unchanged per Omer's instruction. Condensed existing Figure 1/2
  captions and status definitions to accommodate the explanation without reducing type.
- **Commands/results:** PDF skill authoring marker ran successfully. Tectonic with the
  existing `.cache/tectonic` compiled the report; expanded draft initially overflowed
  to six pages, final is five. `pdfinfo`, `pdftoppm` and visual inspection checked the
  rebuilt pages. Compile log: `outputs/report001/comment-review/table-compile.log`.
  No overfull boxes or undefined references; `git diff --check` passed. Text-only changes,
  no new tests needed; previous reporting check remains 16 passed.
- **Next action:** Authors review the clearer table. No experiment, commit, push or
  external comment was made; prior local changes are preserved.

---

## 2026-10-15 — REVIEW-INBAL-PDF-001 — Review supplied PDF before adoption

- **Intent:** Omer supplied Inbal's `final_project_deep_learning_JEPA.pdf` after no
  newer contribution was identifiable in the current GitHub branches/PRs. Review
  first, then decide whether to build on it; equation numbering is part of the review.
- **Evidence:** Compared all five PDF pages with report source at `0ee9d10` and the
  existing local report PDF. Supplied SHA-256:
  `df1bb44dc76566dc1188ed659a944c7ef9b88438866913dd5d9e06fd7388bdfa`.
  INBAL_PDF_REVIEW_2026-10-15.md records findings, page locations, implementation
  evidence, proposed wording and a future equation-presentation checklist.
- **Findings:** Keep omitted-loss weighting/effective-example explanations and
  selected clarifications. Distinguish pooled slope from ratio of slopes; qualify
  causal explanations for ordinary-data degradation and the planning null; avoid
  locating the repair in a particular module without ablations. Equal-contribution
  wording needs author knowledge and is optional under the professor's instruction.
  Equations remain unnumbered; readout diagram is too small; six-output versus
  block-only readout notation is an inherited gap in both versions.
- **Commands/results:** `pdfinfo`, `pdftotext -layout`, `pdftoppm -scale-to 1600 -png`
  and `shasum -a 256` on the supplied file; complete commands in the review.
  All five page renders inspected. Numeric-token comparison with the current PDF
  found only additions/deletions associated with edited prose/captions; this is not
  a plot-data audit. `UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync
  pytest tests/test_train001_model.py tests/test_train002.py tests/test_shift002.py
  tests/test_planning002.py`: 29 passed, two Gymnasium warnings, 7.95 s.
  `git diff --check` passed. No Python/environment changes; no `make check` required.
- **Limits/next action:** Supplied LaTeX source unavailable; no compilation or
  experiment rerun. Recommend selectively applying edits to tracked source after
  the authors' decision. Only review/task/log notes changed; report, PDF, protocol,
  code and scientific evidence untouched. No commit, push or external message.

---

## 2026-10-15 — REPORT-MATH-001 — Verified additions and formal mathematical presentation

- **Intent/authorisation:** Omer approved incorporation and correction of Inbal's PDF,
  a layer-block architecture diagram, self-validation, commit, push and a PR. Used
  jepa-study and PDF workflows. Based branch `codex/report-equations-readout` on
  `origin/main` at `e850e11`, preserving the earlier review notes.
- **Changes:** Added seven numbered equation groups for readout, displacements,
  gains/energy ratio, endpoint error, rollout, loss and regret. Corrected six-output
  versus block-only readout notation, input/output standardisation and output order.
  Generated a readable vector layer-block diagram through `reporting.py`; the method
  figure now shows start subtraction. Adopted verified omitted-arm weighting and
  nonzero-example explanations, squared units, method/selection clarifications and
  selected glossary edits. Corrected candidate-specific planning-goal explanation;
  retained unlocalised mechanism and unidentified tradeoff cause. Unverified causal
  explanations/equal-contribution statement were not imported. Full disposition table
  and validation detail are in INBAL_PDF_REVIEW_2026-10-15.md's implementation addendum.
- **Commands/results:** `UV_CACHE_DIR=/private/tmp/jepa-uv-cache make check` passed;
  `UV_CACHE_DIR=/private/tmp/jepa-uv-cache make test`: 362 passed, 25 existing warnings,
  17.04 s. `UV_CACHE_DIR=/private/tmp/jepa-uv-cache MPLCONFIGDIR="$PWD/.cache/matplotlib"
  uv run --locked --no-sync python -m jepa_action_effects.reporting`: five figures,
  111 numbers. `cmp` against pre-edit snapshots: numbers, input hash registry and
  claims ledger byte-identical. Direct float64 readout-equation check against selected
  probe: maximum difference 5.684341886080802e-14; loss equation and implementation
  differ by 4.44e-16; target ordering, layer shapes and slope/ratio distinction verified.
- **PDF validation:** `TECTONIC_CACHE_DIR="$PWD/.cache/tectonic" tectonic -X compile
  reports/report001/main.tex --outdir outputs/report001/pdf`; initial drafts overflowed
  to six pages, final five pages after removing repetition, without changing font or
  margins. No overfull boxes or undefined references; non-blocking font/underfull
  warnings. All five `pdftoppm -scale-to 1800 -png` renders inspected. Logs and audit
  JSON are under ignored `outputs/report001/equation-review/`.
- **Clean-source check:** Copied current `git ls-files` sources into
  `/private/tmp/jepa-report-equations-clean-20261015`, built with the locked original
  interpreter and exported-source `PYTHONPATH`, compiled with existing TeX cache.
  All five figure PNGs, numbers, claims/input files and all five rendered PDF pages
  match the workspace build byte-for-byte. `git diff --check` passed.
- **Limits/next action:** No experiment rerun or fresh-machine environment installation.
  Numerical results, protocols, dependencies and sealed artifacts unchanged. Commit,
  push the validated report branch and open/attach a PR; authors review the final PDF.

---

## 2026-10-15 — REPORT-COLUMNS-001 — Fit key terms and abstract together

- **Intent/change:** Omer requested swapping key terms and abstract to use the empty
  first-page space. Moved the glossary before the abstract and set only its local
  leading to 10 pt, keeping its 9 pt type size. Both fit fully in the left column;
  the introduction begins at the top of the right column. Initial direct swap left
  one abstract line in the right column; the final local leading fixes that spill.
- **Commands/results:** `TECTONIC_CACHE_DIR="$PWD/.cache/tectonic" tectonic -X compile
  reports/report001/main.tex --outdir outputs/report001/pdf` passed; five pages.
  `pdftoppm -f 1 -l 1 -scale-to 1800 -png` and `pdftoppm -f 2 -l 5 -scale-to 1500 -png`
  rendered all pages for visual inspection: no clipping/overlap; no overfull boxes or
  undefined references. Logs/renders: `outputs/report001/column-review/` (ignored).
  Exact source-block comparisons against `37b39ee` verify abstract text unchanged,
  glossary unchanged apart from local leading, and all remaining source identical.
  `git diff --check` passed.
- **Limits/next action:** Layout only; no Python/environment/scientific changes, so
  tests were not rerun (previous full suite: 362 passed). Update the existing open
  report PR #9 with this follow-up; authors inspect the revised first page.

---

## 2026-10-22 — DATA-OVERLAP-001 — Which evaluation data LeWM was trained on

- **Date / task IDs / actor:** 2026-10-22; DATA-OVERLAP-001; Claude Code / Omer.
- **Intent and changes:** While preparing the talk, Omer asked whether evaluating on
  expert-demonstration data is leakage. Checked LeWM's training source, the released
  dataset and our own data loaders. No code, protocol, sealed artifact, estimate or
  report text changed; this entry only records the finding.
- **Commands and results:**
  - Fetched pinned LeWM source `ca231f9` (`config/train/lewm.yaml`,
    `config/train/data/pusht.yaml`, `train.py`, `config/eval/pusht.yaml`, `eval.py`)
    and stable-worldmodel `44c45bd` `stable_worldmodel/data/dataset.py` from
    raw.githubusercontent.com. Training loads dataset name `pusht_expert_train`
    (`frameskip: 5`, `num_steps = history 3 + 1 = 4`) and calls
    `spt.data.random_split(dataset, [0.9, 0.1], generator=seed 3072)`. The dataset's
    items are `clip_indices = [(ep, start) for start in range(length - span + 1)]`,
    span 20: one overlapping clip per start frame. The 10% hold-out is therefore of
    random clips, not episodes.
  - Hugging Face API: dataset `quentinll/lewm-pusht` at `655cd44` contains only
    `.gitattributes`, `README.md` and `pusht_expert_train.h5.zst` (SHA-256
    `7cfbd6d9…`, matching INTERFACE_CONTRACT.md). Model `quentinll/lewm-pusht` at
    `22b330c` declares `datasets: [quentinll/lewm-pusht]`.
  - `h5py` on the local extracted file: 18,685 episodes, 2,336,736 frames, mean length
    125.1 (min 49, median 123, max 246); 1,981,721 training clips of span 20; no
    episode shorter than the span. Each frame lies in four clips, so the probability it
    appears only in held-out clips is about 0.1^4 = 0.01%.
  - LeWM paper (arXiv 2603.19312v1 HTML, App. E) describes PushT as "20,000 expert
    episodes with an average length of 196 steps", which does not match the released
    file. LeWM's PushT evaluation config also samples start/goal states from
    `pusht_expert_train` ("evaluation from dataset (replay expert trajectories)").
  - Our code: `train001_collect.py` ordinary windows read recorded pixels
    `h5["pixels"]` at t−10…t+10 directly from `pusht_expert_train.h5`;
    `shift002.py` builds the ×1 "varying" archive branch from the recorded controls and
    requires replay identity (history position ≤ 0.5, last-frame mean |Δ| ≤ 1.0 uint8,
    own-future block ≤ 1.0); demo-scene histories replay archive episodes;
    `pilot_collection.propose_challenge` samples designed starts uniformly in the
    simulator, with no archive input.
- **Evidence:** Clean `main` at `373925d`. Downloaded paper HTML kept only in the
  session scratchpad; no data or weights added to Git.
- **Decision or deviation:** No protocol change. Classification:
  - Ordinary windows (TRAIN-001/002 ordinary metrics, e.g. released 10.58, real-frame
    9.70, FT-all 11.67) are scored on frames LeWM almost certainly trained on. The
    released model's ordinary score may be optimistic, and the ordinary-case cost of
    push fine-tuning may partly reflect forgetting memorised clips.
  - The SHIFT-002 archive ×1-varying branch reproduces a training clip, so the pooled
    ×1 archive R (1.03) may be optimistic.
  - Archive-scene histories are training-distribution inputs; their constant or scaled
    futures are new.
  - Designed (challenge) scenes contain no archive data. E1, the designed arm of E2, E3
    and the fine-tuning result each hold on designed scenes alone. Seeing the data can
    only flatter the model, so the overlap cannot produce the under-prediction finding.
  - Splits inside this study remain disjoint, as previously audited
    (REVIEW-LEAKAGE-001).
- **Blockers / uncertainty:** The exact clip-level split was not reproduced; the
  released checkpoint's config names newer classes than the pinned commit, so an exact
  replay of `random_split` is not guaranteed. Overlap of frames is near-certain
  regardless. Whether an unreleased validation file exists is unknown. The written
  report does not yet state this limitation explicitly.
- **Next action / handoff:** Talk notes now explain the overlap and a prepared defence
  (ignored `outputs/presentation001/`). Authors decide whether to add one limitation
  sentence to `reports/report001/main.tex`. A clean fix would regenerate demo-like
  episodes in the simulator with a scripted pushing policy and repeat the archive-scene
  and ordinary-window tests on them.

### 2026-11-22 — Pre-publication privacy redaction

- **Date / task IDs / actor:** 2026-11-22 / repository hygiene / Claude Code.
- **Intent and changes:** Before making the repository public, replaced absolute
  local paths containing the macOS username with `~/` or `<repo>/` placeholders in
  `docs/HANDOFF_2026-10-01.md`, `docs/INBAL_PDF_REVIEW_2026-10-15.md` and
  `docs/environment-local.json`. No measurements, hashes or conclusions changed.
- **Commands and results:** `git grep -nE '/Users/|/home/[a-z]'` returns no matches;
  the full-history scan found no API keys, tokens, passwords, private keys or
  student ID numbers.
- **Blockers / uncertainty:** Earlier commits still contain the original paths, and
  commit metadata lists author email addresses; removing either requires a history
  rewrite.
- **Next action / handoff:** Authors decide whether to rewrite history before
  publishing.

---

## Entry template

Copy this template into a new dated entry above this section; leave earlier entries intact.

- **Date / task IDs / actor:**
- **Intent and changes:**
- **Commands and results:** exact command, pass/fail, relevant measurements.
- **Evidence:** code revision/dirty state, report/artifact paths, hashes when applicable.
- **Decision or deviation:** link DECISIONS.md if needed; exploratory vs frozen.
- **Blockers / uncertainty:**
- **Next action / handoff:**
