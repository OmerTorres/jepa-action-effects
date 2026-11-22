# MEAS-001 engineering pass — handoff (D033)

**Outcome: not ready.** Engineering validation failed at setting selection. All three
predeclared search settings produced zero accepted measurements on real engineering
endpoints. Per D033, the configuration is **not** presented as freeze-ready, and the run
stopped before calibration.
- `config-proposed.json` has status `not_ready` and SHA-256 `6c275409…`.
- None of the 48 MEAS-001 families was encoded or inverted.
- τ was not computed for MEAS-001, and no saved predictor latent was read.

Evidence: `docs/manifests/meas001-engineering.json`, which holds the proposal, per-setting
criteria and diagnostics. Traces are in `outputs/meas001/engineering/` (ignored).

## What was built

| Module | Role | Tests |
| --- | --- | --- |
| `pose_search.py` | Model-free joint search: two Sobol + coordinate searches, nuisance challenges, five-way abstaining classification, τ rule | 14 analytic fixtures, including flat, competing-minima, curved-mean, invalid-geometry and budget cases |
| `pose_render.py` | Label-free `SceneContext`; render-only joint poses; restore and nuisance checks; geometry | 5 (restore, overlap tolerance, agent/wall rule, label guard, replay identity) |
| `meas001_metrics.py` | D032 §6 gates, floor, decision, §7 prediction rule, engineering criteria | 6 |
| `meas001.py` | All stages, write-once seals, lineage checks, four-hour ledger, failure diagnostics | 14 (membership, aliases, disjointness, label isolation, lineage/freeze refusal, stage guards, no report after integrity failure, budget resume, trace reuse/verification) |

`make check` passes; `make test` passes with 299 tests (260 existing + 39 new). Commands are in
MEAS-001_RUNBOOK.md.

## What ran (charged: 19.9 min of 240)

1. `select` (metadata only): 48 families, 216 real queries, 108 for calibration. Roles:
   2 benchmark families, 6 control families, 16 prediction targets.
2. `engineer-select`: 30 GALLERY-001 families declared.
   - No shared seed (30 against 24 seeded), reset state (20 against 48) or archive
     episode (0 against 24).
   - Phase A: 3 challenge + 3 ordinary families. Phase B: 6 more families.
3. `engineer`, phase A: 24 targets × 3 settings (start, 2 endpoints and 1 synthetic pose
   per family).

   | Setting | Seconds/target | Evaluations/target | Phase-A τ | Both searches converged | Accepted (real) | Synthetic recovered |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | S1 4×160 | 18.3 | 1,698 | 0.0296 | 12/24 | 0/18 | 0/6 |
   | S2 2×320 | 12.9 | 1,194 | 0.1282 | 24/24 | 0/18 | 0/6 |
   | S3 512 + 2×192 | 18.3 | 1,677 | 0.0413 | 24/24 | 0/18 | 0/6 |

   Almost every query was `optimizer_unresolved`; a few were `ambiguous`. Phase B did not
   run because no setting passed phase A.
4. `propose` returned `not_ready`; the reason is engineering `failed_selection`.

## Diagnosis (engineering labels; read-only, from saved traces)

- **Identifiability looks adequate; search does not.** The best pose usually lands near
  the truth. For S2:
  - median block error 1.05 units;
  - 14 of 24 targets within 1.28 units, and 21 within 5.12.

  No wrong pose fitted better than the true pose.
- **The searches stop short of the true basin.** Rendering a labelled pose reproduces
  the target almost exactly (median true-pose residual 0), yet no search reached that
  basin (0–1 of 24 per setting). The two independent searches end a median 4.6–5.1 units
  apart, against a 1.28-unit restart tolerance.
- **The residual resolves block position.** For S2, the per-target best residual rises
  smoothly with block distance from the truth:

  | Block distance (units) | Best residual |
  | --- | ---: |
  | < 0.64 | 2.2e-4 |
  | 1.28–2.56 | 1.2e-3 |
  | 5.12–10.24 | 1.1e-2 |
  | > 20.48 | 0.14 |
- **The D032 τ rule amplifies search failure.** τ is the 95th percentile of per-family
  maximum *achieved* residuals, so search failures set it. S2's τ (0.128) corresponds to
  block distances above about 20 units. τ also defines the near-optimal band, so near-
  optimal sets become far wider than 2.56 units even when the searches agree.

## Decisions needed at review (nothing changed yet)

1. **Proposed D033 amendment: agent geometry.** The kinematic agent passes through
   walls. Engineering ordinary families have the agent inside a wall in 14 of 60 endpoints
   and fully off-canvas in 9 of 60. The implemented default checks the agent only against
   the block and searches agent centres over [−16, 528]. The literal D032 rule is
   available as `agent_walls=True`.
2. **Whether to attempt engineering version 2** on the engineering set, or stop MEAS-001.
   Version 2 needs a newly sealed spec. Phase-A families are now development data;
   phase-B families remain unused and could validate. About 40 min of the engineering
   sub-cap remain. Candidate changes:
   - a final pooled refinement from the better of the two searches' bests, with finer
     block steps;
   - more local budget for the best basin;
   - restart agreement judged after that refinement.
3. **Whether to revise the D032 τ rule.** This is a protocol change. One option keeps the
   acceptance threshold but derives the near-optimal band from a label-free resolution
   scale: the residual increase for controlled 1.28-unit block offsets of rendered
   calibration poses.

## Budget and runtime

- 1,191 s used; 12,909 s remain after the 300 s shutdown allowance.
- Measured cost is 12.9–18.3 s per target; the cap of 2,048 evaluations implies about
  22 s. The full frozen workload (256 targets × 22 s × 1.5 headroom ≈ 141 min) would fit
  the remaining budget. Runtime is not the blocker.

## Limitations

- Engineering scenes are GALLERY challenge and ordinary scenes, not SHIFT archive-rest
  scenes.
- 24 targets per setting is a small sample. The diagnosis is about search behaviour on
  real latents, not about predicted latents.
- Renderer identity:
  - Replayed frames reproduce bit-for-bit.
  - Direct assignment of saved poses differs at 2–3 edge pixels in 7 of 210 frames.
  - The resulting true-pose residual is reported per target.

## Correction (2026-10-01, MEAS-REVIEW-001 F4; MEAS-FIX-001)

The diagnosis above overstates what finite search traces establish:
- The distance table summarizes the best *visited* residual per bin, across targets that
  differ in agent position and angle. It is not a controlled residual-versus-distance sweep
  or a landscape measurement.
- Failing to find a wrong pose that beats the true pose does not establish
  identifiability. The `better_than_truth_but_far` count covers only best poses more than
  5.12 units from truth.

The supported reading:
- Sampled near-truth poses can fit well, and sampled distant poses generally fit worse.
- The searches often fail to reach the known low-residual solution: 0–1 of 24 targets per
  setting.
- The τ rule turns achieved fit errors into wide near-optimal sets.

**Joint block identifiability remains unvalidated.** The other three review findings (F1
lineage, F2 prediction gate, F3 final-pose support) are fixed in code with regression
tests. Version 1 artifacts still reproduce exactly: diagnostics are identical to the
tracked manifest, and the 72 classifications re-derive with 0 mismatches.

## Version 2 engineering pass (2026-10-01, D037) — **not ready; MEAS-001 stops**

The guards were fixed first (MEAS-FIX-001). The pass used one setting, V2-S2r300 (S2 base
plus an independent 300-evaluation Nelder–Mead refinement per search, with a label-free
ambiguity band), on the same six phase-A engineering families.

| | v1 S2 | v2 |
| --- | ---: | ---: |
| Real queries accepted | 0 / 18 | 3 / 18 (challenge 2, ordinary 1) |
| Coverage (challenge / ordinary) | 0 / 0 | 0.22 / 0.11 (need ≥ 0.8) |
| Restart agreement | — | 0.33 / 0.11 (need ≥ 0.8) |
| Synthetic recovered | 0 / 6 | 2 / 6 |
| Accepted block error (mean) | — | 0.47 / 0.55 units |
| Median reported block error | 1.05 | 1.04 |
| Searches reaching the true-pose basin (≤ 1.5 × true-pose residual) | 0–1 / 24 | 0 / 24 |

- **Outcome.** Status `failed_selection`, so phase B was not run. The proposal
  `outputs/meas001/engineering-v2/proposal.json` is `not_ready` (SHA-256
  `e5d10f867fe8…`); the manifest is `docs/manifests/meas001-engineering-v2.json`. Charged
  engineering time is 1,748 s of the 3,600 s sub-cap (v1 1,191 s + correction 120 s + v2).
  The 48 MEAS-001 families remain closed.
- **Why it failed** (engineering diagnosis from saved traces).
  - 18 of 24 queries were `optimizer_unresolved` for restart disagreement.
  - Both independent searches *converged* (23 of 24 pairs) but to different minima: median
    separation 6.0 units, interquartile range 1.3–19.7.
  - Refinement improved accepted accuracy but not where the searches end. No search reached
    the true-pose basin.
  - The render-and-encode residual surface over joint agent and block pose has many local
    minima at the scale that matters. Local refinement from Sobol starts does not find the
    narrow true basin within this budget.
- **Consequence for the paper plan (WP1 fallback).** The readout-free inversion instrument
  is not available. Measurement independence rests on two internal controls:
  - the agent/block dissociation, which uses the same readout and encoder;
  - real-frame floors at every dose, plus the five-instrument agreement.

  A different instrument (for example a decoder route on a model that ships one) would need
  its own protocol. This does not show that block pose is unidentifiable; it shows only that
  this search does not recover it reliably.
