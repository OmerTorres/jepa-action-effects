# Paper plan (approved 2026-10-01)

Copied from the approved session plan. Living document: update dates, owners and go/no-go outcomes here.


## Context

Omer asked for a plan that makes the project ready for a main-conference paper. The
course report, due within about two weeks on an unknown date, still has priority, and the
same work should feed both. Today (2026-10-01) produced SUPPORT-001, ROBUST-001 and
LIT-003, set up TRAIN-002 (training running) and scaffolded the report.
`docs/HANDOFF_2026-10-01.md` holds the details.

**What we have.** Everything comes from one checkpoint (LeWM PushT), with sealed fresh
confirmations:
- **Compression grows with command magnitude.** The predicted block displacement shrinks
  more as the push command grows. This is confirmed three times on fresh families and
  continues past the training range.
- **The trend holds across five readouts** (exploratory).
- **The agent moves as commanded while the block is compressed** (descriptive; the agent
  gate was applied after the values were seen). The distortion is in the predicted
  interaction.
- **No preregistered account predicts it.** The account-B covariate saturates.
- **Fine-tuning removes the compression** (TRAIN-001). TRAIN-002 will show whether that
  needs the high dose in training.
- **A decision cost is not established** (H3).

**What a main-conference reviewer will say is missing.** In priority order:
1. **Generality.** One checkpoint and one environment. We need at least one more
   environment and independently pretrained checkpoints, not just fine-tuning seeds.
2. **Measurement independence.** Every physical number goes through readouts fitted on
   real frames. A readout-free instrument (MEAS-001 v2) or a strong internal control is
   needed.
3. **Mechanism and cause.**
   - It is not yet shown whether command magnitude, effect size or contact speed drives
     the compression.
   - There is no causal manipulation of *pretraining* support.
4. **Consequence.** The compression has not been shown to change planning under a usable
   planner. ARC-Bench predicts that frequent replanning masks it.
5. **Novelty versus concurrent work.** The Intervention Gap, Do-JEPA and Sensitivity
   Shaping overlap. Our distinct claims must be dose response, interaction localization,
   training-support causality, generality and consequence.

**Target paper.** *Large Commands, Small Effects: action-effect compression in latent
world models.* It would claim that compression:
- grows with action magnitude past training support, across environments and pretraining
  seeds;
- is localized to the predicted interaction, not to reading the action;
- is caused by training support (a pretraining manipulation);
- harms open-loop and low-replanning control, while standard metrics (latent MSE, probes)
  miss it.

## Work packages (each confirmatory step: a D-record preregistration, sealed fresh families, family bootstrap, floor-relative gates)

**Ownership.** Omer and Inbal assign the split themselves.
- GPU work runs on Inbal's Colab and is her task: the WP3 SEED-001 and WP4 PRETRAIN-001
  runs, plus the timing pilot.
- CPU work runs on Omer's Mac with Claude: WP0, WP1, WP2, the ENV-002/003 audits, WP5.
- Claude prepares a self-contained Colab runbook and scripts so Inbal can run them
  without this session.

### WP0 — Finish the course-cycle evidence (now → evidence freeze Oct 7)
1. **TRAIN-002.** Training is complete (2026-10-01 18:58): all six runs passed the
   optimization gate, with loss 0.093–0.108 → 0.024–0.026 for dose_omitted and
   0.011–0.012 → 0.005–0.006 for archive_only.
   - Next: `train002 gate` → `seal` → `confirm` → `report`, then
     `docs/TRAIN-002_RESULT.md`, TASKS and LOGBOOK.
   - The primary is dose_omitted − balanced at the held-out 0.60 dose. This is the first
     causal training-support evidence.
2. **MEAS-FIX-001, then the MEAS-001 v2 engineering pass,** following the design in
   HANDOFF §3:
   - F1 lineage binding in `meas001._run_target`, `require_frozen`, `query_inputs` and
     `_labels`;
   - F2 `predictions_allowed` derived from the full `decision()`;
   - F3 versioned `pose_search.classify` with independent support for the final pose and
     defined nuisance outcomes;
   - F4 wording, plus the accounting correction;
   - D037 before running: independent Nelder–Mead refinement per search, separate
     τ_fit/τ_amb, a new output directory, development on phase A, validation once on
     phase B, and a stop if it fails.
3. **The course report** (REPORT-001), written as a 5-page cut of the paper storyline:
   - figures and macros from `reporting.py` (add the SUPPORT-001 and TRAIN-002 registry
     entries and a fine-tuning figure);
   - `main.tex` results and discussion, then slides, `REPRODUCE.md`, Inbal's reproduction
     subset, and the contributions written by the authors.

### WP1 — Measurement independence (Oct 14 → Nov 8)

**Update 2026-10-01: MEAS-001 v2 failed engineering (D037), so take the fallback branch
below.** Options for a second instrument, each needing its own protocol:
- a decoder route on a model that ships decoders (DINO-WM/JEPA-WM, ARCH-001);
- global search with a much larger budget (costly);
- dropping the requirement and leaning on the internal controls.
- If v2 engineering passes: freeze, then `calibrate` → `controls` → `assess` → `score`,
  then the 16-target predicted-latent diagnostic. Before the freeze, add the
  MLP/inversion disagreement, report tables and control summaries, mock-tested.
  **Gate:** the inversion agrees with the MLP readout in sign and rough size on predicted
  latents in the named cells.
- If v2 fails: report it as a measured limitation. Rely on the two strongest internal
  controls:
  - the agent/block dissociation, which uses the same readout and encoder;
  - real-frame floors at every dose.

  For the second environment, prefer environments where a pixel or state decoder is
  available.

### WP2 — Mechanism on PushT (CPU; Oct 14 → Nov 15)
- **MECH-001, command versus effect-size dissociation.** A preregistered factorial on
  fresh families: command magnitude × initial agent–block gap, so that equal commands give
  unequal block displacement and the reverse.
  - Contact onset and duration come from the stored `block_trajectories` and `contacts`.
  - Question: does compression follow command magnitude, true displacement, or agent speed
    at contact?
  - Reuse the `support001.collect` collector, adding gap offsets through
    `pilot_collection.propose_challenge` metadata, and the `support001_metrics` and
    `calib001_metrics` estimands.
- **MECH-002, interaction-level support account (B′).**
  - Build an archive map of agent speed and approach direction at contact events (from
    archive `state` columns).
  - Use a non-saturating distance-to-support covariate, which fixes SUPPORT-001's
    limitation.
  - Preregister against A and C with PhysWeep-style held-out selection, and confirm on
    fresh over-range families.
- **MECH-003 (optional): readout-free latent check.** Project latent changes onto the
  ridge block subspace against the agent subspace (exploratory).

### WP3 — Generality (Nov → mid-Dec)
- **ENV-002, LeWM Cube (OGBench manipulation; the key replication).**
  - Interface audit (pattern: `interface_audit.py`, `INTERFACE_CONTRACT.md`).
  - Seeded-replay validation (pattern: `simulator_replay.py`).
  - A readout trained on random-reset frames (pattern: `readout_scale.py`).
  - Dose families of end-effector pushes, and the effector-versus-cube dissociation.
  - Data: `quentinll/lewm-cube` (checkpoint and dataset) and swm `OGBCube-v0`.
- **ENV-003, LeWM TwoRooms (agent-only navigation; a prespecified negative-control
  prediction).** If the action pathway is faithful, there should be little compression
  of agent displacement with command magnitude. Reacher is an optional second check.
- **SEED-001.** Three independently pretrained LeWM-PushT runs with the `le-wm` recipe
  (`train.py`, `config/train/lewm.yaml`) on GPU. Re-run the SUPPORT-001 dose response on
  each, to separate pretraining seeds from fine-tuning seeds. These runs are also the
  full-support arm of WP4.
- **ARCH-001 (stretch).** DINO-WM or JEPA-WM on PushT (`facebook/jepa-wms`; separate
  simulator audit; CC-BY-NC).

### WP3/WP4 GPU handoff to Inbal (prepare by Oct 20; Claude + Omer)
- **`src/jepa_action_effects/pretrain001.py`** writes, on the Mac, sealed window-index
  manifests from the archive `action` column:
  - full data;
  - tail-removed (mean command > p90 = 0.328, from `support001` `map.json`);
  - matched random removal.

  Each manifest records hashes, the selection rule and seeds.
- **A thin training wrapper** around the pinned `le-wm` `train.py` and `jepa.py` that
  restricts its dataset to a manifest's indices. Everything else stays identical: config
  `lewm.yaml`, 100 epochs, bf16, batch 128, LR 5e-5. It writes a run record (config,
  manifest hash, git commit, GPU type, wall time, final checkpoint SHA-256).
- **`docs/PRETRAIN-001_RUNBOOK.md`.** It is the Colab notebook steps:
  - download `quentinll/lewm-pusht` from HF;
  - one-epoch timing pilot, stopping if the projection exceeds the session budget;
  - run order: SEED-001 seeds 1–3, then the tail-removed and random-removal arms × 3;
  - checkpoint every N epochs to Google Drive, with resume;
  - upload checkpoints with hashes to a private HF repo or Drive.

  Omer then pulls them to the Mac, where all evaluation (dose families, readout gates,
  agent probe) runs with the existing CPU runners.
- **Mock-tested before handoff:** a CPU smoke run of the wrapper on a tiny manifest
  subset.

**Design detail (from reading `le-wm train.py` and swm `Dataset`, 2026-10-01).**
- Training samples are `clip_indices` = (episode, start) with span 20 (4 frames ×
  frameskip 5).
- `pretrain001.py` writes a sealed boolean clip mask on the Mac from the archive `action`
  column. The clip statistic is the largest 5-control block mean ‖a‖ in the clip.
  - The tail-removed arm drops clips above the 90th percentile of that statistic.
  - The random arm drops the same number of clips, seeded.
- A launcher swaps in an `HDF5Dataset` subclass that filters `clip_indices` by the mask,
  then calls the unmodified `train.run()`. Per-column normalizers are still computed on the
  full archive, so they are identical across arms.
- The decisive contrast is tail-removed versus random-removed, matched in data and steps.
  The full-data arm (SEED-001) is the reference.

### WP4 — Cause: support-manipulated pretraining (GPU on Inbal's Colab; Nov → mid-Dec)
- **PRETRAIN-001.**
  - Tail-removed arm: drop archive windows whose mean command exceeds the 90th percentile
    (0.33).
  - Control arm: random removal matched in sample count.
  - Three seeds each, equal steps and compute, same data loader. The full-data arm is
    SEED-001.
  - Prediction: tail removal moves compression onset to lower commands, while in-range
    behaviour is unchanged.
- **Timing pilot first** on Colab (one epoch). Budget: about 6–9 runs × a few GPU-hours.
- Reuse TRAIN-002's dose-omission and archive-only results to motivate the arm design.

### WP5 — Consequence and metric blind spot (Dec)
- **PLAN-001.** LeWM's own CEM protocol (`le-wm eval.py`, `config/eval/pusht.yaml`:
  horizon 5, receding 5, 50 episodes, goal offset 25), using goals that need large versus
  small pushes.
  - Compare frozen with fine-tuned predictors (TRAIN-001 balanced, TRAIN-002 arms).
  - Vary replanning frequency (1, 5, open-loop).
  - Measure success and first-plan error. Prediction: compression hurts open-loop and
    low-replanning control, and the fine-tuned model recovers it.
- **METRIC-001.** Across arms and seeds, show that latent MSE and real-frame probe
  accuracy do not track the compression (use TRAIN-001 secondary latent results plus WP3
  checkpoints).

### WP6 — Paper (Dec 14 → submission)
- **Paper:** 8 pages plus appendix (preregistrations, gates, provenance, compute).
- **Freezes:** figure freeze Jan 10, then an internal review of the claims ledger
  against the manifests (Codex review).
- **Release:** an anonymized repo (anonymous.4open.science), merged to `main`.
- **Literature:** re-search arXiv monthly and again in the final week.
- **Target (decided):** ICML 2027 (late January; verify the exact date when the call
  appears). Fallback: NeurIPS 2027 (May), decided at the Dec 15 go/no-go.

## Calendar (ICML 2027 track)

| Window | Omer + Claude (Mac CPU) | Inbal (Colab GPU) |
| --- | --- | --- |
| Oct 1–7 | TRAIN-002 confirm; MEAS-FIX-001 + v2 engineering; evidence freeze | — |
| Oct 7–13 | Course report and slides | Reproduction subset; slides; contributions |
| Oct 14–Nov 8 | WP1 MEAS calibration/assessment; MECH-001/002; `pretrain001.py` + runbook (by Oct 20) | Timing pilot (Oct 21–25); SEED-001 seeds 1–3 |
| Nov 9–Dec 13 | ENV-002 Cube; ENV-003 TwoRooms; evaluate the pretrained checkpoints; PLAN-001 | PRETRAIN-001: tail-removed and random-removal arms × 3 |
| Dec 14–Jan 10 | METRIC-001; figure freeze; first full draft | Re-runs if needed; draft review |
| Jan 11–deadline | Internal review against the claims ledger; anonymized release; submit | Co-writing; final checks |

## Go/no-go decision points
- **Nov 8 (WP1).** If the inversion contradicts the MLP sign, drop the physical-mechanism
  framing and write a measurement paper.
- **Dec 1 (WP3).** If Cube and the independent PushT seeds show no compression, the claim
  is checkpoint-specific. Go to a workshop or TMLR.
- **Dec 15 (WP4/5).** If there is neither a causal support effect nor a planning
  consequence, slip to NeurIPS 2027 with more work.

## Readiness checklist (a conference-level paper requires all of these)
- [ ] Each headline claim is preregistered and confirmed on fresh families, with gates
  passed and ≥ 2 instruments (or an internal control).
- [ ] Generality: ≥ 2 environments and ≥ 3 independently pretrained checkpoints.
- [ ] A causal manipulation of training support (fine-tuning **and** pretraining).
- [ ] A planning consequence under the model's own planner.
- [ ] Novelty paragraph against the Intervention Gap, Do-JEPA, Sensitivity Shaping,
  ARC-Bench and IMPLY; no unsupported "first" claims.
- [ ] Limitations stated (horizon, constant pushes, readout dependence, preprints).
- [ ] Reproducible: one command per figure from the manifests; compute and seed
  statement.

## Critical files
- **New runners**, following the existing pattern (runner + `_metrics` + tests + D-record +
  manifest + RESULT doc): `src/jepa_action_effects/{mech001,mech002,env002_cube,
  pretrain001,plan001}.py`.
- **Reuse:**
  - `support001.py`: the collector, `family_units`, the gate.
  - `support001_metrics.py`: `SupportMap`, accounts, `decide`.
  - `calib001_metrics.py`: `row_statistics`, `summarize_group`.
  - `train001_metrics.paired_bootstrap`.
  - `train001.py`: `readout_decoders`, `score_model`, `build_cache`.
  - `reporting.py`: the registry and figures.
  - `meas001.py` and `pose_search.py` (WP1).
- **Docs:** `docs/DECISIONS.md` (D037+), TASKS, LOGBOOK, HANDOFF, `reports/report001/`
  (course), and a new `reports/paper/` (conference).

## Verification
- `make check` and `make test` after every Python change. Each new metric gets analytic
  fixtures that recover a known account or effect, plus lineage-mutation and
  no-write-on-failure tests, as in `tests/test_support001.py` and `tests/test_train002.py`.
- Every confirmatory run follows the order: seal before scoring, real-frame gate before
  any predictor runs, strict provenance, a reproduction gate against earlier sealed values
  where applicable.
- Report and paper: every number regenerates from tracked manifests
  (`python -m jepa_action_effects.reporting`). The PDF compiles with Tectonic. Every page
  is inspected, with page limits checked.
- At each go/no-go date, update HANDOFF/TASKS with the decision and evidence.
