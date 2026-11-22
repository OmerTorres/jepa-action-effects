# Task board

Status values: TODO, IN_PROGRESS, BLOCKED, DONE. Owners below follow the plan's
proposed allocation and are not a record of teammate acceptance. Completion needs
linked evidence in the logbook; a scaffold is not an implemented experiment.

| ID | Task | Proposed owner | Status | Depends on | Completion evidence |
| --- | --- | --- | --- | --- | --- |
| DATA-OVERLAP-001 | Check whether our evaluation data was used to train LeWM | Claude Code / Omer | DONE | ENV-001, TRAIN-002, SHIFT-002 | 2026-10-22 logbook: LeWM trains on the same `pusht_expert_train` file with a random 90/10 split of overlapping clips (≈all frames seen); ordinary windows and archive histories are training data, archive ×1-varying replays a training clip; designed scenes contain no archive data; no protocol, estimate or report change; report limitation sentence pending author decision |
| REPORT-COLUMNS-001 | Swap key terms and abstract to use the first-page column space | Codex / Omer | DONE | REPORT-MATH-001 | Key terms followed by complete abstract in left column; introduction begins right; local glossary leading 10 pt with unchanged type size; five pages rebuilt/visually inspected; exact block/rest comparison proves prose/equations unchanged |
| REPORT-MATH-001 | Incorporate verified Inbal edits, number equations and draw the readout architecture | Codex / Omer | DONE | REVIEW-INBAL-PDF-001 | INBAL_PDF_REVIEW_2026-10-15.md implementation addendum: per-addition evidence/disposition; seven equation groups, vector layer diagram, corrected block projection and planning explanation; 362 tests and make check pass; 111 numbers unchanged; clean-source figures and all five PDF page renders identical |
| REVIEW-INBAL-PDF-001 | Review Inbal's supplied report PDF and equation presentation before adopting edits | Codex / Omer | DONE | REPORT-TABLE-CLARITY-001 | INBAL_PDF_REVIEW_2026-10-15.md: all five pages inspected; 29 focused tests pass; recommend selective incorporation, gain/ratio clarification, qualified causal/planning explanations and numbered equations; report unchanged, author decision pending |
| REPORT-TABLE-CLARITY-001 | Explain E2/E3 validation checks and R in Table 1 | Codex / Omer | DONE | REPORT-COMMENTS-001 | Reworded motion recovery/counts, defined R and CI thresholds, identified E2 baseline checks and four count groups; five-page PDF rebuilt; E7 unchanged |
| REVIEW-E7-001 | Assess whether E7 is complete and worth retaining | Codex / Omer | DONE | H3-PLAN-002 | Checked D024–D026, implementation and sealed confirmation against tracked manifest: completed controlled test, 148 valid families, H3 not established; recommend brief secondary result with explicit limits, not removal |
| REPORT-COMMENTS-001 | Address Omer's twelve PDF clarity annotations | Codex / Omer | DONE | REPORT-001 | Abstract and methods simplified; glossary moved to page 1; seeds/warm-up/substeps explained; demo-rest wording corrected; readout diagram added; common figure crop includes agent at every magnitude; 16 reporting tests and make check pass |
| REVIEW-LEAKAGE-001 | Audit omitted-magnitude fine-tuning for leakage and overstatement | Codex / Omer | DONE | TRAIN-002 | TRAIN002_LEAKAGE_AUDIT_2026-10-14.md: split/hash checks, actual-model gradient invariance, fresh inference sample, 5,040-record recomputation, 59 tests; report distinguishes zero loss weight from unseen magnitude |
| SETUP-001 | Dev environment, project memory, research skill | Codex / Omer | DONE | — | Locked install, doctor, Ruff, skill validation; see 2026-09-25 logbook |
| ENV-001 | Audit frozen checkpoint and exact simulator | Omer | DONE | SETUP-001 | Pinned source/checkpoint audit, hashes, licenses, timing, and transition smoke; see 2026-09-26 logbook and INTERFACE_CONTRACT.md |
| LIT-001 | Verify closest-work matrix and applicable metrics | Omer | DONE | — | LIT-001_REVIEW.md: primary-source matrix, metric applicability/omissions, pinned checkpoint evaluation review |
| SIM-001 | Reset/warm-up replay and pair-specific collisions | Inbal | DONE | ENV-001 | Zero-discrepancy 16-seed order/replay report and verified substep contacts; see 2026-09-26 logbook |
| EVAL-001 | Physical effects, endpoints, strata, finite-candidate regret | Inbal | DONE | SETUP-001 | Analytic fixtures, replay adapter, 33-test suite, and runtime manifest; see 2026-09-26 logbook |
| MODEL-001 | Frozen inference and latent cache | Omer | DONE | ENV-001 | Strict load, archive statistics, leakage-free smoke, cache/throughput report; see 2026-09-26 logbook |
| PILOT-001 | 120 families, six branches, splits 60/30/30 | Both | DONE | SIM-001, EVAL-001, MODEL-001 | Complete run and STOP-expansion decision; see 2026-09-27 logbook and pilot001-runtime.json |
| READOUT-001 | Audit and bounded readout recovery | Codex / Omer | DONE | PILOT-001 | Reproduced pilot metrics, corrected family-weighted scaling and competence reporting, collected 300 disjoint families, tested nested probes; see 2026-09-27 logbook and readout001-runtime.json |
| DELTA-001 | Bounded displacement measurement pilot | Codex / Omer | DONE | READOUT-001 | Existing-data run failed all real-frame gates; confirmation remained closed; see DELTA-001_RESULT.md and delta001-runtime.json |
| LATENT-001 | Exploratory action-to-future correspondence audit | Codex / Omer | DONE | LIT-001, DELTA-001 | All 60 tuning families and required checks completed; see LATENT-001_RESULT.md and latent001-runtime.json |
| NOVELTY-002 | Reassess nearest work and identify object-specific measurement path | Codex / Omer | DONE | LIT-001, LATENT-001 | NOVELTY-002_MEMO.md: AD-WM comparison, saved-state feasibility census, provisional rendered-gallery diagnostic |
| GALLERY-001 | Validate renderer-identity and block-pose gallery feasibility | Codex / Omer | DONE | NOVELTY-002 | 30-family simulator/encoder audit, both gates passed; GALLERY-001_RESULT.md, RELATED_WORK.md and gallery001-runtime.json |
| PREDICTOR-GALLERY-001 | Test frozen predictor against the validated block-pose galleries with controls | Codex / Omer | DONE | GALLERY-001 | 30 development and 60 reserved families completed; primary challenge rule not met; see PREDICTOR-GALLERY-001_RESULT.md and predictor-gallery001-runtime.json |
| PROBING-001 | Review alternative probing methods and JEPA overlap | Codex / Omer | DONE | PREDICTOR-GALLERY-001 | PROBING-001_REVIEW.md: arXiv shortlist, closer JEPA precedents, selective-erasure feasibility recommendation; no experiments |
| REVIEW-EXT-001 | External methodological review of completed work | Claude / Omer | DONE | PREDICTOR-GALLERY-001 | EXTERNAL_REVIEW_2026-09-28.md: findings, confounds, bugs, ranked next steps; no experiments on the checkpoint |
| DIST-001 | Action-support, simulator-identity and archive/simulator error-gap checks | Claude / Omer (Mac run) | DONE | REVIEW-EXT-001 | No rendering mismatch; archive/simulator predictor ratio 0.032/0.040; no study block above archive p99 (in-support variant not triggered); dist001-runtime.json |
| REANALYSIS-001 | Exploratory reanalysis of saved gallery/latent outputs (moved/static, boundary pairs, latent slope) | Claude / Omer (Mac run) | DONE | REVIEW-EXT-001 | Exploratory; gallery accuracy driven by static endpoints; latent designed slope 0.38; reanalysis001-runtime.json |
| READOUT-002 | 60k-frame readout with R², learning curve and near-pose sensitivity | Claude / Omer (Mac run) | DONE | DIST-001 | Selected mlp-20261005: validation block mean 3.48, study-check 4.41; ridge plateaus at 15.7; readout002-runtime.json |
| PHYS-002 | Predictor vs real-frame readout floor, shrinkage slope, 150 fresh reserved families | Claude / Omer (Mac run) | DONE (interpretation pending) | READOUT-002 | Reserved challenge: floor passed (0.12), physical failure and shrinkage supported, H1 not established; PHYS-002_RESULT.md and phys002-runtime.json. Exploratory audit (phys002-audit-runtime.json): no data/probe error, but archive windows show no shrinkage (slope 0.96), so the result is distribution-specific |
| SHIFT-001 | Identify the cause of challenge-family shrinkage: 2×2 of challenge vs archive-like states × constant vs archive-like action blocks | Claude / Omer | DONE (stopped) | PHYS-002 | Implemented and collected; development manipulation checks failed (push-shrink bound, readout floor), so seal refused; reserved unopened. Exploratory: magnitude effect, no structure effect. Awaiting redesign decision; superseded by SHIFT-002 (D023) |
| SHIFT-002 | Push-magnitude dose-response across challenge and archive scenes, with structure contrast | Claude / Omer | DONE | SHIFT-001 | Confirmed on 240 fresh families: magnitude supported in both scene types, no constancy effect; SHIFT-002_RESULT.md, shift002-runtime.json |
| H3-PLAN-001 | Regret when choosing among candidates by latent distance to a rendered goal image (LeWM planning cost) vs real-latent, random and oracle choice | Claude / Omer | DONE (stopped) | PHYS-002 | Sealed rule: regret_jepa − regret_real_latent CI > 0 and real-latent beats random; fresh families. Development failed the latent-cost-beats-random check (agent confound in goal image; `hold` chosen 59%); seal refused, reserved unopened; awaiting goal-image redesign; superseded by H3-PLAN-002 (D025) |
| H3-PLAN-002 | Regret under latent planning cost with agent-matched goal images | Claude / Omer | DONE | H3-PLAN-001 | Confirmed on 150 fresh families: latent cost works (real − random −6.0); H3 not established (JEPA − real +2.1 [−1.3, 5.6]); H3-PLAN-002_RESULT.md, planning002-runtime.json |
| REVIEW-BRANCH-002 | Review claude/youthful-clarke-akm28u implementation, evidence and claims | Codex / Omer | DONE | H3-PLAN-002 | BRANCH_REVIEW_2026-09-29.md: 134 tests passed, three confirmation summaries reproduced exactly, 660 family hashes verified; provenance, metric, uncertainty and reporting issues documented. Fixes remain proposed; no protocol changed |
| REVIEW-FIX-001 | Corrections plus exploratory reanalysis and independent rerun after REVIEW-BRANCH-002 | Claude / Omer | DONE | REVIEW-BRANCH-002 | D026 corrections; REVIEW_FOLLOWUP_RESULT.md: rerun of 150 PHYS + 30 SHIFT families matches sealed records exactly; absolute endpoints 17.895/3.613; slope = length 0.63 + direction 19°; per-dose readout in band; clustered archive CI unchanged |
| REVIEW-FIX-002 | Provenance validator foundation and corrected protocol gates for future sealed runs | Claude / Omer | DONE (foundation) | REVIEW-FIX-001 | provenance.py (strict/legacy modes, probe-artifact and record-fingerprint checks after D027) + protocol_gates.py, mutation tests; legacy check passes for PHYS-002, SHIFT-002, H3-PLAN-002. Runner integration remains until a new confirmation is scheduled |
| REVIEW-BRANCH-003 | Verify review follow-up and remaining limitations | Codex / Omer | DONE | REVIEW-FIX-002 | REVIEW_FOLLOWUP_AUDIT.md: 148 tests pass; saved per-dose gates pass; remaining ratio-sign explanation, validator artifact/cache checks and zero-vector direction issues documented; fixes not applied |
| REVIEW-FIX-003 | Address REVIEW-BRANCH-003 findings | Claude / Omer | DONE | REVIEW-BRANCH-003 | D027: ratio-direction correction in REVIEW_FOLLOWUP_RESULT.md; validator probe/fingerprint checks and strict/legacy modes; zero-vector angle handling; regenerated manifests unchanged (0 undefined angles) |
| REVIEW-BRANCH-004 | Verify D027 fixes and check regressions | Codex / Omer | DONE | REVIEW-FIX-003 | REVIEW_FOLLOWUP_AUDIT.md verification at de678d9: all three findings addressed; 151 tests passed; legacy provenance passed; existing numerical results unchanged; no new actionable findings in the patch |
| ERASURE-001 | Validate selective agent/block linear concept erasure | Codex / Omer | BLOCKED | PHYS-002 | Deferred by D021. `concept_erasure.py` and its tests are committed but have never been run |
| FREEZE-001 | Freeze protocol and collect main dataset | Both | TODO | PILOT-001, LIT-001 | Versioned protocol, tolerance/horizon rationale, hashed split manifest |
| FIT-001 | Readouts, privileged reference, inference controls | Both | TODO | FREEZE-001 | Validation readout gate; prediction cache; sanity controls |
| TEST-001 | Locked test analysis and family bootstrap | Inbal | TODO | FIT-001 | CIs, contact contrasts, regret; hypothesis assessment |
| TRAIN-REVIEW-001 | Review TRAIN-001 implementation plan before handoff | Codex / Omer | DONE | REVIEW-BRANCH-004 | TRAIN-001_IMPLEMENTATION_PLAN_REVIEW.md: measurement-failure policy preserved; collection, validator lifecycle, model fingerprints, budget and gate definitions need clarification before execution; no experiments run |
| TRAIN-001 | Magnitude-balanced vs archive-weighted fine-tuning of the LeWM predictor (predictor, action encoder, prediction projector), with sealed fresh confirmation | Claude / Omer | DONE | TRAIN-REVIEW-001 | TRAIN-001_RESULT.md (corrected per D029), D028 + outcome, train001-runtime.json: all stop gates passed; development archive floor failed, so the pooled physical verdict is not assessable. Descriptive balanced − archive −0.09 [−0.53, 0.38]; secondary latent balanced − archive −0.00684 [−0.01276, −0.00110]; ordinary non-inferiority not established; 1.06 of 24 production h |
| TRAIN-REVIEW-002 | Review completed TRAIN-001 implementation and evidence | Codex / Omer | DONE | TRAIN-001 | TRAIN-001_REVIEW.md: 233 tests passed; strict validation passed; complete summary reproduced exactly; 42 model/unit inference checks agree within 6.062e-5 units; original findings addressed by TRAIN-FIX-001 and verified in TRAIN-REVIEW-003 |
| TRAIN-FIX-001 | Address TRAIN-REVIEW-002 findings without changing sealed evidence | Claude / Omer | DONE | TRAIN-REVIEW-002 | D029; corrected RESULT/REPORT-001 wording and latent interval; `train001_followup.py` (lineage validator + contrast-level sensitivity) and 9 tests; train001-followup.json: 58/58 checks incl. lineage for all six runs; one readout disagreement (ordinary balanced − frozen); figures relabelled descriptive |
| TRAIN-REVIEW-003 | Verify TRAIN-FIX-001 and sealed-evidence preservation | Codex / Omer | DONE | TRAIN-FIX-001 | TRAIN-001_REVIEW.md follow-up at 3932604: all four original findings addressed; 242 tests and 58 validation checks pass; seal/summary/runtime manifest unchanged; one new P2: follow-up regeneration continues after validation failure |
| TRAIN-FIX-002 | Stop follow-up report generation on failed validation | Claude / Omer | DONE | TRAIN-REVIEW-003 | `train001_followup.run` raises `ValidationFailure` before any presentation/manifest write; CLI exits 1; regression test proves the existing manifest is unchanged and no writer is called; real outputs still pass 58/58 checks; 300 tests pass. Sealed sources and evidence unchanged |
| LIT-002 | OpenAlex/arXiv search for a main-conference extension after the external review | Claude / Omer | DONE | TRAIN-001 | LIT-002_REVIEW.md: 64 queries, 225 records screened; Aug–Sep 2026 precedents (Intervention Gap, IMPLY, PhysWeep, Sensitivity Shaping, Do-JEPA, ARC-Bench); reversion hypothesis, calibration-slope and inversion measurement, staged plan; no experiments |
| CALIB-001 | Calibration-side reanalysis of opened records (calibration slope, noise correction, latent check) | Claude / Omer | DONE | LIT-002 | CALIB-001_RESULT.md, D030 + amendment, calib001-runtime.json. Exploratory; the reproduction gate reproduced 27 sealed values (1.1e-15). PHYS-002 challenge calibration slope 1.33 [1.24, 1.44], above 1 at k = 0 too; compression grows with dose (archive 1.05 → 1.39) and fine-tuning removes it; challenge over-dispersion (0.64–0.90) is noise-sensitive; ordinary windows calibrated (0.99); 256 tests pass |
| CALIB-REVIEW-001 | Review LIT-002 and CALIB-001 evidence and interpretation | Codex / Omer | DONE | LIT-002, CALIB-001 | CALIB-001_REVIEW.md at 619a151: 256 tests pass; full reanalysis reproduces exactly, 27 reference checks pass; four P2 interpretation/measurement findings: latent versus physical means, scalar gain versus dispersion, noise assumptions, inversion acceptance |
| CALIB-FIX-001 | Qualify calibration claims and prospective inversion gates | Claude / Omer | DONE | CALIB-REVIEW-001 | D031 accepts all four findings; CALIB-001_RESULT.md corrected with banner; STUDY_SYNTHESIS and LIT-002 correction addenda; energy-ratio/alignment estimands and dose contrasts added (13,768 earlier values unchanged; gate 27/27); k=0 primary, k>0 labelled orthogonal-error sensitivity; review counterexamples are fixtures; 260 tests pass. Inversion gates specified in D031, protocol not yet drafted |
| CALIB-REVIEW-002 | Verify CALIB-FIX-001 and preservation of earlier values | Codex / Omer | DONE | CALIB-FIX-001 | CALIB-001_REVIEW.md follow-up: all four corrections implemented; 260 tests pass; current outputs reproduce exactly; original runner rerun confirms all 13,768 earlier leaves unchanged; 27 reference checks pass. Keep energy ratio distinct from length ratio and pooled alignment distinct from pure angle. Next: bounded D031 measurement proposal |
| MEAS-001-PLAN | Draft bounded joint-pose-inversion feasibility proposal | Codex / Omer | DONE | CALIB-REVIEW-002 | MEAS-001_PLAN.md, D032: 48 existing development families; calibration/assessment split; joint agent/block inversion; residual/ambiguity rejection; real-latent gates and synthetic controls; four-hour CPU cap; proposal only |
| MEAS-001-ENG | Implement MEAS-001 runner and run the D033 engineering pass | Claude / Omer | DONE (not ready) | MEAS-001-PLAN | MEAS-001_ENGINEERING.md, meas001-engineering.json, D033 + proposed amendment: full runner and 299 tests; 48-family membership sealed (metadata only); engineering failed setting selection (0 accepted on real endpoints for all three settings; searches stop ~5 units apart; τ rule inflated by search failure); proposal not_ready; 19.9 of 240 min charged; no MEAS family encoded |
| MEAS-REVIEW-001 | Review MEAS-001 engineering implementation and failed run | Codex / Omer | DONE | MEAS-001-ENG | MEAS-001_REVIEW.md at 8ecd208: 299 tests pass; 72 traces and diagnostics verified; four findings on input/result lineage, optional-prediction gates, final-pose restart support and identifiability claims; recommends one bounded revised engineering version, with no execution started |
| MEAS-FIX-001 | Address MEAS-REVIEW-001 before another engineering version or freeze | Claude / Omer | DONE | MEAS-REVIEW-001 | F1 lineage binding and family-hash checks, F2 full-decision prediction gate, F3 versioned classifier with independent final-pose support, F4 wording, accounting correction, charged score stage; regression tests; v1 artifacts reproduce exactly (diagnostics identical, 72/72 classifications); 343 tests pass |
| ROBUST-001 | Multi-instrument re-decoding of opened predicted latents (ridge, three MLPs, kNN) with an off-manifold score | Claude / Omer | DONE | TRAIN-FIX-002 | ROBUST-001_RESULT.md, D034, robust001-runtime.json. Exploratory; gate reproduced 1,888 CALIB-001 values (8.9e-16). Energy-ratio dose trend negative with CI < 0 for all five instruments in both scenes; PHYS-002 reserved challenge b_rev 0.543–0.574 across instruments; off-manifold flags 4–12% and falling with dose; on-manifold stratum inconclusive (ridge challenge CI crosses 0). Corroboration, not validation |
| SUPPORT-001 | Archive support map and preregistered test of attenuation vs reversion vs clamp accounts beyond the training range | Claude / Omer | DONE | ROBUST-001 | SUPPORT-001_RESULT.md, D036, support001-runtime.json. 300 fresh families × 6 doses; all six real-frame gate cells pass (incl. 0.80/1.00). Verdict: no account established (readouts agree); C beats A (−11.8 [−21.9, −3.8]), B not excluded. Compression keeps growing past the training range (block slope designed 0.71→0.57, archive 0.66→0.41 from 0.60 to 1.00). Agent probe (gated): agent motion faithful (≈0.96–1.01) where block is compressed (0.66–0.71) |
| TRAIN-002 | Support-manipulated fine-tuning: dose omission vs all doses vs archive-only, held-out 0.60 dose | Claude / Omer | DONE | TRAIN-001 | TRAIN-002_RESULT.md, D035, train002-runtime.json. Gates passed (4/4). Primary: held-out-dose deficit supported but small: dose_omitted − balanced +0.82 [0.11, 1.63]; transfer fraction 0.92 [0.85, 0.99]. Archive-only fine-tuning leaves compression unchanged (R 0.71/0.65 vs frozen 0.71/0.64). Ordinary-case cost of designed-push fine-tuning replicates (+1.08 [0.28, 1.91]) |
| MEAS-001 | Implement and assess abstaining joint-pose inversion | Claude / Omer | DONE (stopped, not ready) | MEAS-FIX-001 | D037 v2 engineering failed selection: 3/18 real accepted (coverage 0.22/0.11, restart agreement 0.33/0.11); searches converge to different minima (median 6.0 units apart); 0/24 reach the true-pose basin. Stopped per D037; 48 families stay closed; paper WP1 uses its fallback (internal controls). MEAS-001_ENGINEERING.md |
| REVIEW-CPU-001 | Evaluate the completed CPU study cycle and readiness for course/paper handoff | Codex / Omer | DONE | ROBUST-001, SUPPORT-001, TRAIN-002, MEAS-001 | CPU_STAGE_EVALUATION.md at 9060a58: 343 tests pass; TRAIN-002 full summary reproduced from 5,040 records; SUPPORT primary contrasts reproduced from 300 families; ROBUST 1,888 checks and MEAS 96 trace classifications pass; four reporting/reproduction findings; paper still has CPU work |
| REPORT-FIX-001 | Correct report figure, claim scope, support-range wording and reproduction order | Claude / Omer | DONE | REVIEW-CPU-001 | F1 account-implied slopes via support001_derived.py (D038, support001-account-slopes.json; B matches review); F2 claims rescoped in main.tex plus dated notes in result docs; F3 p99.9/413/max 0.80 macros; F4 stage-level order in REPRODUCE.md; 347 tests pass; 3-page PDF inspected; sealed evidence unchanged; see 2026-10-02 logbook |
| REPORT-001 | Bounded-study report, slides and reproduction handoff | Both | IN_PROGRESS | REPORT-FIX-001, TRAIN-002, SUPPORT-001, REPORT-FIX-002 | Written report revised after REVIEW-COURSE-001 (REPORT-FIX-002, 2026-10-13): 5-page PDF, 75 generated macros, five figures, four tables, all pages inspected; contributions covered verbally per the professor's instruction. Remaining: slides (.pptx), Inbal's clean-clone reproduction subset (REPRODUCE.md steps 1–6), final PDF export check, instructor acceptance |
| REVIEW-COURSE-001 | Professor-style assessment against supplied course requirements | Codex / Omer | DONE | REPORT-001 | PROFESSOR_REVIEW_2026-10-13.md: revised provisional 90/100 after professor's verbal-contributions guidance clarified; 347 tests pass, 46 macros match, TRAIN-002 full summary reproduced from 5,040 records; prioritized report fixes and recommendation to defer course GPU extension |
| REPORT-FIX-002 | Address REVIEW-COURSE-001 report fixes 1–5 without new experiments | Claude / Omer | DONE | REVIEW-COURSE-001 | Report expanded to 5 pages: pipeline diagram + PushT schematic, data table, fine-tuning arms table, loss equation, metric and gate definitions, TRAIN-002 results led with absolute means (new table), training/validation curves from the 12 saved runs, purpose-named experiments, wording fixes at the three flagged sentences. New `train002_recompute` (5,040 records, full summary equals sealed + tracked) and `training_curves` (hash-checked export) modules; REPRODUCE.md step 4 now runnable; 354 tests pass; sealed sources unchanged. Teammate reproduction itself still pending |
| REVIEW-COURSE-002 | Reassess merged report revision and new reproduction utilities | Codex / Omer | DONE | REPORT-FIX-002 | PROFESSOR_REVIEW_2026-10-13_FOLLOWUP.md at ba25aed: provisional 91/100; 354 tests, 75 macros, 12 hash-checked curves and full 5,040-record recomputation pass; F1–F4 identify gate-description, metric-scope/bias, flagged-latent influence and null-wording corrections; no new experiments needed |
| REPORT-FIX-003 | Address REVIEW-COURSE-002 corrections F1–F4 and presentation polish | Claude / Omer | DONE | REVIEW-COURSE-002 | F1 gates described per study (strong-push floor; SHIFT-002 pooled band/count/manipulation checks, no floor; designated TRAIN/SUPPORT cells); F2 metric subsets stated (moved vs all branches/families), R wording "normalizes … bias may remain", projection-gain wording; F3 "not driven by" removed, unflagged-only ridge −0.36 [−0.88, 0.09] reported as inconclusive (new macro); F4 "no deficit was established at 0.45". Fig. 1 redrawn (~7 pt text, explicit real-future route through the same encoder/readout, schematic label); Table 1 caption allows the intended TRAIN-001→TRAIN-002 reuse. 76 macros; 354 tests; 5-page PDF inspected; no estimates changed; F5 (PR #3 comment): strong-push paired-effect floor (`next_metrics.floor_ratio`, designated meaningful branch pairs) distinguished from the later displacement floor (`train001_metrics.cell_gate`); 5-page PDF re-inspected |
| REPORT-REDESIGN-001 | Rewrite and redesign the course report for readability: story, plain vocabulary, real examples, consistent figures | Claude / Omer | DONE (awaiting Omer's read) | REPORT-FIX-003 | D039 (example rule + vocabulary); `report_examples.py` + `report-examples.json` + 11 frame PNGs (byte-identical rerun); new figure system in `reporting.py` (teaser, method, robustness, fine-tuning); 111 numbers / 158 macros, all 76 earlier values unchanged; `main.tex` rewritten; 5-page PDF, no overfull boxes, all pages inspected (colour + greyscale); 362 tests pass; see 2026-10-13 logbook |
| REVIEW-COURSE-004 | Review the final report redesign against course requirements and saved evidence | Codex / Omer | DONE | REPORT-REDESIGN-001 | PROFESSOR_REVIEW_2026-10-14.md at 6aec9fc (same tree as merged main 87f1e9e): approximately 92/100; 362 tests pass; clean-source rebuild of 111 numbers/four figures and five PDF pages matches; F1 scope unchanged-effect claim, F2 qualify instrument/action-path exclusion, F3 correct validation caption. No new experiments needed |
| REPORT-CLARITY-001 | Explain commands, contact, gain and horizon choices before results | Codex / Omer | DONE (author understanding pending) | REPORT-FIX-004 | Rewrote abstract/introduction and first-caption definitions; explained inherited five-control grouping vs chosen two-step horizon using D009's simulator-only 88/120 pilot result; added UNDERSTANDING_THE_STUDY.md. Five-page PDF rebuilt and inspected; 111 generated numbers unchanged; no protocol/model/experiment changes |

## Next handoff

**Clarity and understanding (2026-10-14, REPORT-CLARITY-001).** Omer identified that
earlier reviews had not established whether the authors could understand and defend the
work. The report now defines command magnitude, latent/readout and motion gain before
using them; separates contact-dependent outcomes from command magnitude; explains five
controls per model step as inherited and the one-second horizon as a pilot-retained
design choice, with no optimality claim. Read UNDERSTANDING_THE_STUDY.md and work through
one command/contact/prediction example before treating the presentation as ready.
These edits are local and uncommitted. The earlier F1–F3 were fixed in abe218f.

**Final-report review (2026-10-14, REVIEW-COURSE-004).** The redesign is much clearer
and the five-page PDF rebuilds exactly. Before submission, apply the three small
corrections in PROFESSOR_REVIEW_2026-10-14.md: stable medians do not establish unchanged
effects for all demo families; probe agreement does not rule out shared instrument bias
or locate the action mechanism; Figure 4(c) uses 60 designed plus 60 demo validation
families. No new model runs are requested. Contributions remain verbal per the professor.

**Latest PR review (2026-10-13, REVIEW-COURSE-003).** PR #3 at efff5d3 reviewed:
354 tests pass, all 76 macros reproduce, five-page PDF inspected. Previous F1–F4
corrections and figure/data-reuse polish verified. One residual P2 (F5 in
PROFESSOR_REVIEW_2026-10-13_FOLLOWUP.md): distinguish PHYS-002's paired-action-effect
floor from the later displacement-error floor in the report. No new experiments
needed. Review complete; this textual correction remains before submission.

Main-study expansion remains paused. PREDICTOR-GALLERY-001 completed the
sealed comparison on 30 development and 60 reserved families. The primary
measurement gate passed and matched predictions beat mismatched actions, but
the signed challenge margin was negative and the persistence contrast was
inconclusive. The prespecified central positive-evidence rule failed; the
60 reserved families have been opened. The next step is to frame the bounded
checkpoint-specific result in light of the related-work review and decide
whether a separately scoped measurement or model question merits fresh data.
PROBING-001 recommends a bounded feasibility plan for selective agent/block concept
erasure before any mechanism claim. MotionJEPA, LeWM subspace/metric audits and
V-JEPA intervention work narrow the novelty boundary. No new experiment is frozen;
the opened confirmation families cannot confirm a revised hypothesis.
ERASURE-001_PLAN.md now specifies the next bounded feasibility study: 240 fit
families, 30 calibration and 30 assessment families from existing tuning data,
one hour of experimental execution, selective-removal gates and a stop decision.
Predictor interventions and any fresh confirmation remain separately scoped.
REVIEW-EXT-001 (EXTERNAL_REVIEW_2026-09-28.md) recommends re-measuring physics with a
100x larger readout set and a floor-relative criterion, a displacement-shrinkage test,
and simulator/action distribution checks before ERASURE-001. Note that
`concept_erasure.py` and its tests are committed although ERASURE-001 is TODO.
NEXT-001_PLAN.md turns this into DIST-001 → REANALYSIS-001 → READOUT-002 →
PHYS-002. Cloud sessions write code, mock tests and `NEXT-001_RUNBOOK.md`; all
checkpoint, archive and `outputs/` stages run on Omer's Mac. ERASURE-001 is
deferred (D021).
The code for all four tasks is implemented and tested without the model (2026-09-28),
and the two artifact-dependent tests now skip on a fresh clone. Next, run
`docs/NEXT-001_RUNBOOK.md` on the Mac and push the manifests.
The runbook was executed on the Mac on 2026-09-28 (see LOGBOOK). PHYS-002 confirmed
predictor displacement shrinkage and a floor-relative physical error on the reserved
challenge cohort; H1 (larger error at contact boundaries) was not established. Next:
interpret the results and update STUDY_SYNTHESIS, DECISIONS and ERASURE-001 scope.

**Current state (2026-09-29).** The essential-scope experiments are complete:

- PHYS-002: projection-slope shrinkage on challenge families; H1 not established.
- SHIFT-002: the slope ratio decreases with push magnitude in both scene types.
- H3-PLAN-002: H3 not established.

REVIEW-BRANCH-002 (BRANCH_REVIEW_2026-09-29.md) found overstated narratives, a
mislabelled endpoint metric and runner-provenance gaps. D026 records the corrections.
REVIEW-FIX-001 (reanalysis and independent rerun) and REVIEW-FIX-002 (provenance
validator and corrected gates) are done. Next: the course report (REPORT-001). No further confirmation is planned. A
common sealed runner is deferred until one is.

**REPORT-001 plan (2026-09-29).** See REPORT-001_PLAN.md for the course requirement
mapping, corrected claim scope, deliverables and acceptance checks. The supplied
handout requires substantial deep-learning training. READOUT-002 trained a PyTorch
MLP while JEPA stayed frozen; instructor acceptance of that scope is unresolved.
The plan proposes a separately scoped action-conditioned dynamics predictor if
additional training is required. No new experiment is frozen or started. The
completed bounded cohorts, rather than the paused original TEST-001 expansion,
are the report's evidence base. Next: instructor clarification and claim ledger.

**TRAIN-001 (2026-09-30).** Completed under D028; see TRAIN-001_RESULT.md.
- Headline after D029: a small secondary latent advantage for balanced weighting
  (−0.00684 [−0.01276, −0.00110]); ordinary-case non-inferiority not established.
- No pooled balanced-over-archive physical benefit was established. Descriptively,
  both fine-tuned predictors reduced decoded errors on the sampled constant-push
  distribution.
- The pooled physical hypothesis is not assessable, because the development archive
  floor failed.
- Next: REPORT-001 authoring from the updated plan, and the instructor clarification.

**LIT-002 (2026-10-01).** An external review asked what a main-conference paper would
need. LIT-002_REVIEW.md records the search. Key points:
- August–September 2026 preprints narrow the novelty. The closest audits LeWM intervention
  effects on Cheetah.
- The proposed question: is the distortion a predictable reversion toward the
  state-conditional typical training transition?
- Stage 1 is measurement. Add a calibration-side slope alongside the projection slope, and
  a readout-free simulator-inversion instrument.
- No experiment is scheduled or frozen.
- Next: read the seven flagged papers in full, then decide whether to open Stage 1.

**CALIB-001 (2026-10-01).** This exploratory reanalysis ran LIT-002 §3.1 under D030. See
CALIB-001_RESULT.md.
- The frozen model's shrinkage is not honest hedging: PHYS-002 challenge calibration slope
  1.33, robust to the noise assumption.
- Two components appear: compression that grows with dose, which fine-tuning removes, and
  noise-sensitive over-dispersion in designed challenge states.
- Both remain conditional on the readout being valid on predicted latents.
- Next: scope the simulator-in-the-loop inversion (LIT-002 §3.2) as a separate
  measurement study. Any claim needs fresh families.

**CALIB-REVIEW-001 (2026-10-01).** The CALIB-001 calculations reproduce exactly, but
its interpretation needs correction before reuse (CALIB-001_REVIEW.md). Decoded scalar
miscalibration does not rule out conditional-mean latent prediction; below-1 scalar gain
does not isolate excess magnitude, and the squared alignment is not an identified
resolution decomposition. The k adjustment requires explicit orthogonality assumptions.
Simulator inversion is a candidate decoder needing off-manifold rejection and ambiguity
gates. Reversion remains unestablished. Next: CALIB-FIX-001, then a bounded measurement
feasibility proposal if the publication extension proceeds. No new training is scheduled.

**CALIB-FIX-001 (2026-10-01).** All four CALIB-REVIEW-001 findings are accepted (D031).
- The interpretation is corrected and no number changed.
- What stands is descriptive: decoded block-displacement magnitude falls with dose relative
  to truth for the frozen model, with alignment roughly constant. A comparable trend
  appears in the readout-free latent change.
- No mechanism, bias-versus-uncertainty distinction or readout validity is established.
- Next: a bounded measurement-feasibility proposal with the D031 abstention and ambiguity
  gates. No training or replication yet. TRAIN-FIX-002 remains open.

**CALIB-REVIEW-002 (2026-10-01).** The uncommitted CALIB-FIX-001 changes are verified;
all four findings are addressed, with reporting qualifications in CALIB-001_REVIEW.md.
The descriptive energy-ratio trend stands. P/T is a squared RMS magnitude ratio, and
uncentered alignment also reflects variation in per-candidate gains; neither establishes
a mechanism. Next: a bounded D031 measurement-feasibility proposal, without new training.

**MEAS-001 proposal (2026-10-01).** MEAS-001_PLAN.md and D032 scope the next measurement
feasibility task: 48 existing SHIFT-002 development families, joint pose search,
rejection/ambiguity controls and a four-hour CPU cap. The proposal is complete; code,
configuration freeze and execution remain TODO. A maximum of 16 previously opened
predictions may be inspected only after the named real-latent gates pass. No new
training, replication or fresh confirmation has started. Existing CALIB changes remain
uncommitted and are preserved.

**MEAS-001 engineering pass (2026-10-01).** The runner, tests and runbook exist
(MEAS-001_RUNBOOK.md); the outcome is in MEAS-001_ENGINEERING.md.
- Engineering validation **failed** at setting selection, so the proposal is `not_ready`.
- The 48 MEAS-001 families are untouched apart from metadata.
- Diagnosis (qualified by MEAS-REVIEW-001): sampled near-truth poses fit well, but the
  searches stop short of the known low-residual solution and disagree by about 5 units.
  The D032 τ rule turns achieved fit errors into wide near-optimal sets. These finite
  search traces do not establish joint block identifiability or a smooth landscape.
- Review decisions:
  - the agent-geometry amendment;
  - whether to run engineering version 2 (phase-B families unused; about 40 min of the
    sub-cap left);
  - whether to revise the τ rule.
- MEAS-001 is blocked until then.

**MEAS-001 review (2026-10-01).** MEAS-001_REVIEW.md verifies the saved failure and
records four fixes needed before freeze. The review supports the geometry/reset-identity
corrections and recommends one bounded engineering version 2, after guard and accounting
fixes. Refine searches independently; do not use pooled initialization as restart evidence.
Develop separate fit and ambiguity thresholds on engineering data. No protocol amendment
or new engineering/calibration run was made. The 48 study families remain closed.

**Project plan (2026-10-01).** The course report is due within two weeks and has priority;
the same work should seed a main-conference paper. Sprint order:
TRAIN-FIX-002 (done) → ROBUST-001 → SUPPORT-001 and TRAIN-002 in parallel → evidence
freeze on 2026-10-07 → REPORT-001 deliverables (claims ledger, `reporting.py`, LaTeX
report, slides, reproduction subset). Anything unconfirmed at the freeze moves to the
paper track. D033's amendment is accepted (DECISIONS). MEAS-001 is off the course
critical path: even a passing MEAS-FIX-001 + version 2 ends at a freeze-ready proposal.
New runners adopt MEAS-REVIEW-001's lineage rules from the start.

**ROBUST-001 (2026-10-01).** The compression trend survives a change of instrument:
linear (ridge), saturating (three MLPs) and clamping (kNN, k = 8) readouts all give
negative energy-ratio dose contrasts with intervals below zero in challenge and archive
scenes, and the PHYS-002 projection slope is 0.54–0.57 under every instrument. High-dose
predictions are not more off-manifold. All instruments share one encoder and one frame
bank, so this is corroboration, not validation. Next: SUPPORT-001 and TRAIN-002.

**LIT-003 (2026-10-01).** The seven flagged papers were read in full
(LIT-003_POSITIONING.md). The Intervention Gap is the closest work (a capture-gated LeWM
audit on Cheetah, with excess gain). IMPLY's "typical push" concerns object identity, not
magnitude. Do-JEPA's LeWM gains follow the training intervention distribution. ARC-Bench
audits JEPA-WM, not LeWM. A positioning paragraph is drafted for the report.

**SUPPORT-001 (2026-10-01).** Preregistered account test: no account established, so the
null is reported. The dose response is confirmed a third time on fresh families and
extends to 1.00 without a plateau. Within the measurable range, the predicted agent
motion is faithful while the block effect is compressed. That locates the distortion in
the predicted interaction (descriptive). Next: TRAIN-002 gate, seal and confirmation.

**Handoff (2026-10-01 18:45).** See HANDOFF_2026-10-01.md for what is done, what is
running (TRAIN-002 training, then gate → seal → confirm → report), the MEAS-FIX-001 and
v2 design (started at Omer's request; not yet coded), and the remaining course-report
steps.

**Paper plan (2026-10-01).** The conference roadmap is in PAPER_PLAN.md. Target: ICML 2027,
with NeurIPS 2027 as fallback. Work packages:
- WP0: course-cycle evidence;
- WP1: measurement independence;
- WP2: mechanism;
- WP3: generality (Cube, TwoRooms, independent seeds);
- WP4: support-manipulated pretraining;
- WP5: planning consequence;
- WP6: writing.

GPU pretraining (SEED-001, PRETRAIN-001) is Inbal's, on Colab; the runbook is due by
Oct 20. Go/no-go dates: Nov 8, Dec 1, Dec 15.

**TRAIN-002 (2026-10-01).** Training on the designed constant pushes removes the
compression even at a held-out dose: 92% of the gain transfers, with a small
CI-separated deficit. Generic fine-tuning on archive windows does nothing. What matters
is exposure to strong interactions, not the exact dose or extra training. Next: MEAS-FIX-001
and v2, then the course report.

**MEAS-001 v2 (2026-10-01).** Stopped: the v2 engineering pass failed selection (D037).
Accepted poses are accurate, but most searches converge to different local minima. The
readout-free inversion instrument is not available. The paper's measurement argument
takes the WP1 fallback (agent/block dissociation, real-frame floors, five-readout
agreement).

**CPU-stage evaluation (2026-10-02).** CPU_STAGE_EVALUATION.md independently verifies
the latest numerical evidence and recommends closing the course experiment cycle.
REPORT-FIX-001 addresses a wrong account-B plot, categorical claims, support-range wording
and Route B's dependency order. The course report/slides/reproduction are not complete.
The conference roadmap also retains CPU mechanism/environment/planning work and an
unimplemented GPU launcher/runbook; it is not yet a GPU-only finishing stage. No new
experiment or protocol change was made by the evaluation.

**REPORT-FIX-001 done (2026-10-02).** F1–F4 are fixed without touching sealed evidence
(D038). The report compiles to three pages. Remaining for REPORT-001: author contributions,
teammate reproduction, slides and final PDF check.

**Written report final (2026-10-13).** Only the written report is due now. The PDF
(`outputs/report001/pdf/main.pdf`) is three pages with no draft markers. The authors chose
to cover individual contributions verbally in the presentation rather than in the report
or slides. Remaining for REPORT-001: slides (.pptx with PDF and speaker notes, due with the
later talk), the teammate reproduction subset and instructor acceptance.

**Course review (2026-10-13).** PROFESSOR_REVIEW_2026-10-13.md assesses the merged
main tree (2e3af92) at a provisional 88/100 for the written package; oral defense remains
ungraded. Existing twelve predictor fine-tuning runs provide a substantial training
component in the reviewer's assessment. Recommend deferring additional GPU work for the
course deadline, without cancelling the paper roadmap. The handout explicitly requires
individual contributions in the written report, so verbal coverage alone leaves that
requirement unmet. Before submission: accurate contributions, clearer methods/metrics and
training splits, stronger presentation of the valid TRAIN-002 result, remaining wording
corrections, and teammate reproduction. These are review recommendations; the report and
sealed evidence were not edited.

**Course-review correction (2026-10-13).** Omer clarified that the professor explicitly
instructed the team to discuss contributions verbally when presenting the report.
This supersedes the handout's written-contributions requirement for this submission.
No written contributions paragraph is needed; the earlier compliance finding is
withdrawn. Revised provisional written-package grade: 90/100. Remaining recommendations
concern methods, metrics, results presentation, wording and reproduction; scientific
assessment unchanged.

**Report fixes (2026-10-13, REPORT-FIX-002).** All five REVIEW-COURSE-001 fixes are
addressed in the written report, except that the teammate reproduction must be run by
Inbal: a dry run of Route A from a clean local clone is recorded in the logbook, which is
not a teammate reproduction. The report is now five pages including references. No
experiment, protocol, sealed artifact or scientific estimate changed. Remaining before
submission: Inbal runs REPRODUCE.md steps 1–6 and records them; check the exported PDF;
slides for the later talk.

**Follow-up course review (2026-10-13, REVIEW-COURSE-002).** Merged main ba25aed
is substantially clearer and the saved evidence reproduces. Provisional written-package
assessment: 91/100; contributions remain verbal per the professor. Before submission,
correct the SHIFT-002 gate description, distinguish moved-only and all-family metrics,
qualify slope normalization, remove the unsupported flagged-latent influence inference,
and say no deficit was established at 0.45. See PROFESSOR_REVIEW_2026-10-13_FOLLOWUP.md.
Improve Figure 1 readability if feasible. Teammate reproduction is recommended, not an
explicitly mandated handout procedure. Review only: report and scientific evidence unchanged.

**Follow-up fixes (2026-10-13, REPORT-FIX-003).** All four REVIEW-COURSE-002 corrections
and the Figure 1 / Table 1 polish are applied; the report stays at five pages. No
experiment, protocol, sealed artifact or estimate changed. Remaining: a fresh-machine
reproduction run (recommended, not mandated by the handout), final exported-PDF check,
slides for the later talk.
