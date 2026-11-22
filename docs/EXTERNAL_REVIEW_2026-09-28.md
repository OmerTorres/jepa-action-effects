# External review: JEPA action effects (2026-09-28)

Reviewer: Claude (Claude Code), at Omer's request. Scope: all tracked docs, manifests,
source and tests at `59cfa74`. The ignored `outputs/`, `data/` and `checkpoints/`
were not available, and this environment's network policy blocked Hugging Face, so
**no checkpoint inference was run in this review**. Anything marked *measured here*
was run in this session. Anything marked *hypothesis* has not been tested.

## TL;DR

1. **The engineering is solid.** Replay, contact labelling, the leakage-free
   inference path and the family-level statistics are careful and, where I checked
   them, correct. That is a reusable asset.
2. **The project's central blocker was self-imposed.** Every physical measurement
   ran into one absolute readout gate: mean ≤ 1.28 world units, which is **0.56 px
   at 224 px, or 1/25 of a ViT patch**. That gate was hit with readouts trained on
   only about 240 distinct block poses, and the learning curve was still falling
   steeply (19.96 → 15.68 → 11.68 at 60/120/240 families). By ordinary probing
   standards the block is very decodable (my rough estimate is R² ≈ 0.95–0.97; this
   was never reported). The gate was treated as a STOP, so the predictor's physical
   error was **never measured against the readout's own error floor**. That missing
   number is the one the study needed.
3. **The most interesting scientific signal is under-reported.** Predictions are
   under-dispersed: predicted pairwise spread is 64% of real spread. At the primary
   gallery scale, the predictor locates the block no better than persistence. Both
   point to one hypothesis: *the MSE-trained predictor shrinks contact-induced block
   effects toward "nothing moved"*. It can be tested cheaply.
4. **Two confounds make the positive "latent correspondence" and "contact-regime"
   results weaker than they read.** The first is the `hold` candidate in every
   challenge family. The second is action magnitudes that were never checked against
   the training archive.
5. **Next steps:** (a) scale the readout data by 100× (minutes of CPU) and use a
   floor-relative criterion; (b) run the shrinkage test; (c) run the cheap
   distribution checks (action OOD, simulator identity); (d) deprioritize
   ERASURE-001. Fresh reserved families take seconds to collect, so the opened
   confirmation set is not a real obstacle.

---

## 1. What the project did (one-paragraph recap)

The study pinned `quentinll/lewm-pusht` plus the March `stable-worldmodel` PushT. It
validated a strict CPU load and a leakage-free rollout, built deterministic
reset-and-replay branching with per-substep agent–block contact labels, and wrote a
physical-effect and regret evaluator (ENV/MODEL/SIM/EVAL-001). The 120-family pilot
stopped because the block readout failed an absolute gate. Two recovery attempts
followed (READOUT-001 with 300 more families; DELTA-001 with displacement readouts),
and both failed the same gate. The project then pivoted to readout-free measures:
global latent correspondence (LATENT-001, positive), a rendered block-pose gallery
(GALLERY-001, the encoder passes), and predictor-vs-gallery on the reserved families
(PREDICTOR-GALLERY-001, the central rule fails). After that it moved to literature
reviews and an erasure plan (PROBING-001, ERASURE-001). Total compute across all
experiments is minutes. Most of the effort went into protocol and documentation.

## 2. Findings I would stand behind

| Finding | Evidence | My confidence |
| --- | --- | --- |
| Deterministic reset+warm-up replay with exact RGB/state reproduction and pair-specific substep contacts | SIM-001 (16 seeds, 0 discrepancy); 360 reserved endpoints re-rendered exactly in PREDICTOR-GALLERY-001 | High, for this runtime |
| The inference path is aligned with the checkpoint-era LeWM API (history 3, width-10 action blocks, per-control normalization before flattening, `T = H−1+K`) | Code read (`model_inference.py`); manual AR rollout matches `model.rollout` exactly | High that it matches upstream code. Whether it matches *training* depends on the unverified simulator/data lineage |
| The frozen predictor is action-sensitive and beats latent persistence in projected-CLS MSE | LATENT-001 (A = 0.44, B = 0.52, 60 families), replicated on the 60 reserved families | High, but see confounds in §3.4. Mostly reflects agent motion |
| The encoder separates rendered block translations of 10.24 and 20.48 units at a fixed agent pose | GALLERY-001, reserved query gate 98–100% top-1 | High |
| At the primary 10.24 scale, the predictor does not locate the block better than persistence | PREDICTOR-GALLERY-001: matched − persistence margin 0.0013 [−0.008, 0.010]; top-1 53.8% vs 48.8% (chance 20%) | Medium. The contrast is diluted by stationary endpoints (§3.5) |
| Block position is linearly decodable from projected CLS, but not to sub-pixel precision with ≤ 240 poses | READOUT/DELTA-001 | High. The "failure" is about the gate, not the representation (§3.1) |
| The rendered 224-px frames carry sub-unit block information | *Measured here:* 200 random poses, random-direction shifts. The image is identical in 39.5% of cases at 0.1 units, 9% at 0.25, 1% at 0.5, and 0% at ≥ 1.0 | High. The gate is not at the pixel floor, so pixel content cannot excuse a failed readout |

## 3. Mistakes and methodological problems (ranked by impact)

### 3.1 An absolute sub-pixel gate was used as a hard STOP

`δ = max(0.01·L, 10·ε_sim) = 5.12` with `ε_sim = 0`. The gate then required mean
≤ δ/4 = 1.28 and p95 ≤ δ/2 = 2.56 world units. That is 0.56 px and 1.12 px at the
model's 224-px input, where one ViT patch covers 32 world units. The 192-d projected
CLS comes from a JEPA objective that has no reason to keep sub-pixel precision.
Nobody computed this conversion before freezing the gate.

The bigger problem is the gate's role. A readout's error only needs to be **small
relative to the quantity being compared**, not small in absolute terms. The study's
own numbers show the readout was already informative for the effects of interest. In
DELTA-001, the best readout's meaningful-effect error was 8.25, against a
zero-displacement reference of 51.84, which leaves the error at about 16% of the
effect. The right design is:

- Decode real futures and predicted futures with the *same* readout, on the same
  families.
- Report `err(readout ∘ predictor)` alongside `err(readout ∘ encoder(real future))`,
  with a paired family bootstrap.
- Claim a predictor failure only when the predicted-path error is well above the
  real-frame floor. Claim nothing when the two are close.

The plan's warning ("do not *subtract* decoder error") is right. That warning
forbids subtraction; it does not forbid a paired comparison. Because the gate was
binary, the frozen-predictor physical comparison never ran in PILOT, READOUT or
DELTA. **The study's primary quantity (H1/H2) was never computed**, even though the
code to compute it existed.

### 3.2 The readout was starved of data by design

- The readout fit set was 60 → 120 → 240 families. Block pose varies almost only
  *between* families, so the probe saw about 240 distinct (x, y, θ) block poses. It
  had to learn a 2-D position from a 192-d embedding that also entangles rotation
  and the agent.
- The learning curve had not saturated: 19.96 → 15.68 → 11.68, roughly error ∝
  N^−0.4.
- Nonlinear probes were under-trained. The pilot MLP got **300 full-batch Adam
  steps**. Two of three DELTA seeds hit the 3,000-epoch cap. Those results cannot
  count as a nonlinear failure.
- Labelled real frames are nearly free. Encoding costs about 11 ms/frame on CPU
  (MODEL-001), and 300 families were collected in 48 s. Random resets can produce
  50k–100k frames with distinct block poses in about 15 CPU-minutes. The 2.3M-frame
  training archive may also hold state/proprio columns (the interface contract
  implies it; `action_stats.py` only reads `action`, so check). Frames from resets or
  from the archive are "real encoded development observations", so AGENTS.md allows
  them.
- Unprojected CLS (8.60) beat projected CLS (11.68). The BatchNorm-MLP projector
  loses linear accessibility, and the predictor works in projected space. Report
  that; don't just note it.

### 3.3 Distribution shift was flagged as a risk but never measured

RESEARCH_PLAN §7 lists "counterfactual actions far outside training support" as a
risk, and "historical simulator identity unproven" appears in almost every entry.
Neither was tested, even though both tests are cheap and the numbers already
suggest a problem:

| Setting | Predictor MSE / persistence MSE |
| --- | --- |
| 32 archive windows (training data) | 0.0050 / 0.355 = **1.4%** |
| Seed-42 fresh simulator transition | 0.0085 / 0.0186 = 46% |
| LATENT-001 ordinary | 0.039 / 0.094 = 42% |
| LATENT-001 challenge | 0.261 / 1.020 = 26% |

That is a 20–30× relative gap between archive windows and fresh simulator
rollouts. Possible causes include memorization, a rendering or physics difference
between the pinned simulator and the one that made the data, or action/state shift.
It needs diagnosing *before* any physics claim. Suggested checks:

- **Simulator identity:** take archive episodes, reset the pinned simulator to their
  first state (or re-render recorded states), replay the recorded actions, and
  compare frames pixel by pixel. At minimum, compare background, goal overlay,
  agent radius, T colour and line anti-aliasing against archive frames.
- **Action support:** challenge candidates are *constant* 0.6-magnitude pushes held
  for 10 controls. The archive's per-coordinate action SD is 0.21, so an axis-aligned
  0.6 push is about 2.9σ. Compute archive percentiles of the candidate 5-control
  blocks, both for magnitude and for within-block constancy. Ordinary candidates are
  uniform in ±0.35, which is milder but still worth checking.
- **In-support counterfactuals:** build candidate blocks by retrieving archive action
  snippets that lead to near-contact geometry, instead of synthetic constant pushes.

### 3.4 Candidate design confounds the LATENT-001 contrasts

Every challenge family is `{hold, push, 4 × push ± ≤ 0.2 lateral}`. Under `hold`
the agent stops. Under the five pushes it moves about 160 units, almost in the same
direction each time. Consequences:

- The wrong-action mean `W` is dominated by hold-vs-push pairs, so `A` is mostly an
  "agent moved or didn't" discrimination.
- The reported **exactly_one − both contrast (0.359 [0.327, 0.392])** is largely a
  hold effect. Every `hold`–push pair tends to be exactly-one (the push contacts,
  hold does not). Almost every both-contact pair is between near-identical push variants.
  The docs call the strata "descriptive", which is fair, but this specific mechanism
  should be stated. The contrast should also be recomputed on the pre-designed
  boundary pairs only (action distance ≤ 0.2, excluding `hold`). Those are the pairs
  the challenge cohort was built for, and that analysis was never reported on its
  own.

### 3.5 The gallery comparison is diluted, and one control is confounded

- **Stationary endpoints make persistence an oracle.** In challenge families,
  `hold` and non-contact pushes usually leave the block (nearly) in place, so
  persistence scores correctly by construction. The persistence contrast matters
  only on endpoints where the block moved more than about half the negative spacing
  (> 5.12 units). The headline pools both kinds. The motion strata in
  `confirmation.json` are *pair-participant* strata (an endpoint can sit in several).
  A plain endpoint-level split into block-moved and block-static endpoints is what's
  needed. It can be done now from saved artifacts, labelled exploratory.
- **The wrong-action control is agent-confounded.** Other candidates' predictions
  put the agent somewhere else than the gallery's fixed agent: about 160 units off
  for `hold`, and tens of units for lateral variants. Their margins are therefore
  largely noise (28% top-1 against 20% chance). The matched − wrong advantage mostly
  shows agent-conditioned correspondence, not block knowledge.
- **Negatives are translation-only at the realized angle.** Pushes near contact
  mostly *rotate* the T. A predictor that gets the angle wrong is not penalized.
  Add ±rotation negatives.
- **The margin is a local linear probe with fixed directions.** `d(q,t_k) −
  d(q,t_0) = const − (2/192)·q·(t_k − t_0)`, so the metric projects the prediction
  onto template-difference directions. That suggests the continuous version in §5
  (N3).

### 3.6 The pilot's persistence baseline was oracle-initialized

`pilot.py::_predictions("persistence")` uses the **true** current block position,
while JEPA endpoints are decoded through the readout. The comparison "JEPA 24.94 vs
persistence 2.43" is therefore apples to oranges. The pilot correctly called it
inconclusive, but the fair baseline was never built. That baseline is either
`readout(z_t)` held constant, or displacement-space comparison, where the readout's
constant bias cancels.

### 3.7 Process

- **Pivot cascade.** Physical readout → displacement → global latent → novelty
  memo → gallery → predictor-gallery → probing review → erasure. Each pivot moved
  further from the original question (H1–H3) and toward method novelty, and one of
  them used up the reserved confirmation cohort. The recommended course report
  currently rests on the *weakest* positive result (global correspondence, mostly
  agent motion).
- **Reserved data is not scarce here.** Collection runs at about 6 families/s. "The
  60 families are opened" is not a blocker: reserve fresh families with a new seed
  before the next confirmatory run.
- **The overhead is out of proportion to the compute.** There are about 3,700 lines
  of docs for about 12,800 lines of code. Experiments ran in 8–112 s under
  30–60 min caps that never bound. Sealing and provenance are good habits. The
  binding constraints were data scale and the gate, and nobody revisited either.
- **Many "not X" hedges, few effect sizes.** Reports say what cannot be claimed at
  length, but leave out simple descriptive numbers such as probe R², effect-size
  distributions, and prediction-vs-truth displacement slopes. Those numbers would
  let a reader judge the result.

## 4. Bugs and engineering issues

| # | Issue | Where | Severity |
| --- | --- | --- | --- |
| B1 | Two tests **fail** (rather than skip) on a fresh clone because they read ignored artifacts (`outputs/readout001/collection/collection-manifest.json`, `outputs/pilot001/collection-manifest.json`). *Measured here:* 88 passed, 1 skipped, 2 failed. The logbook's "all pass" counts only hold on the original machine | `tests/test_gallery_audit.py::test_real_replay_pose_order_geometry_and_restoration`, `tests/test_predictor_gallery.py::test_audit_excludes_confirmation_payloads_and_rejects_changed_source` | Medium (reproducibility for Inbal/Colab). Use `pytest.skip` when the artifact is absent, as `test_model_inference.py` already does |
| B2 | Docs contradict the repo. TASKS says ERASURE-001 is "not implemented or run" and the 2026-09-28 logbook entry says "No Python/environment change". Commit `59cfa74` ("another idea") adds a 928-line `concept_erasure.py`, 253-line metrics module and tests, with no logbook entry | `docs/TASKS.md`, `docs/LOGBOOK.md` | Low–medium (project-memory integrity) |
| B3 | Under-trained nonlinear probes (300 full-batch steps in the pilot; DELTA seeds at the cap) are reported as failed diagnostics | `pilot_readouts.fit_nonlinear`, DELTA-001 | Medium (interpretation, §3.2) |
| B4 | Oracle-initialized persistence against a readout-decoded model | `pilot.py:_predictions` | Medium (interpretation, §3.6) |
| B5 | `.python-version` pins exactly 3.10.21, which uv 0.8.17 on Linux cannot fetch, so `make setup` fails. The Linux lock also pulls CUDA torch wheels (GBs). I worked around it with the system 3.10.20 for testing only | `.python-version`, `uv.lock` | Low (portability to Inbal's machine/Colab) |

Checked and found correct: action normalization per control before flattening; the
`T = H−1+K` rollout indexing and prediction slice; the independent manual rollout;
the LEACE formula (`I − Σ^{1/2} P Σ^{-1/2}`, centered affine); gallery margin and
derangement expectation; the cohort-preserving family bootstrap; and angle-before-
position pose setting. Ruff lint and format pass.

## 5. Promising next steps (ranked by value per hour)

**N1 — Re-measure physics with an adequately trained readout (about 1 day).**
- Generate 50k–100k frames from random resets with diverse block (x, y, θ) and
  agent poses, plus archive frames if they carry state. Encode them (about 15 CPU
  minutes).
- Fit ridge and a properly trained MLP (mini-batch, to convergence) on projected
  CLS. Report R², mean/p95 and the learning curve per target (block xy, angle as
  sin/cos, agent).
- Replace the absolute gate with a **floor-relative criterion**, frozen *before*
  predictor scoring. Example: the predicted-path displacement error must exceed the
  real-frame floor by a CI-separated margin before a failure is claimed.
- Then run the original H1/H2 comparison on development families, and confirm on
  freshly reserved ones.

**N2 — Test the shrinkage hypothesis (hours, on existing data).** The spread ratios
(real/pred pairwise MSE 0.72 → 0.46 overall; 1.01 → 0.63 challenge) and the
persistence-like gallery result both point to regression-to-the-mean. Estimate the
**slope** of predicted vs true block displacement:
- decoded, once N1 exists;
- in latent space, via `Δẑ_ij` regressed on `Δz_ij` for candidate pairs;
- stratified by contact regime and block-moved/static.

A slope clearly below 1 that is concentrated in contact cases would be a concrete,
mechanistically sensible finding. An MSE objective in a SIGReg-isotropic space
under-predicts uncertain, low-variance effects. That is a sharper and more
defensible claim than the current "dissociation".

**N3 — Turn the gallery into a continuous, agent-controlled decoder (1–2 days).**
- Around each realized endpoint, render finite-difference images for block ±x, ±y,
  ±θ **and agent ±x, ±y**.
- Build the local Jacobian `J` of the encoder with respect to these 5 state
  coordinates.
- Solve `Δŝ = J⁺ (ẑ − e(s_true))` for the predicted endpoint. That gives a signed
  block error in world units with the agent error absorbed as a nuisance.
- Validate linearity with persistence, whose true offset is known.

This is readout-free and handles the agent confound algebraically. It is close in
spirit to render-and-compare and state-subspace work (Li et al. 2026), so frame it as
a measurement tool, not a contribution. A global variant inverts `ẑ` by CEM search
over rendered states.

**N4 — Run the distribution checks from §3.3 (hours).** These are the simulator
re-rendering of archive episodes, archive percentiles of candidate action blocks,
and the archive-vs-fresh error gap. If the pinned simulator renders differently from
the training data, every downstream number changes meaning. This is the
highest-risk unknown in the project.

**N5 — Decision relevance without a readout (about 1 day).** H3 can be tested with
the model's *own* planning objective. Score the 6 candidates by latent distance to a
rendered goal image, as LeWM planning does, and measure simulator block-to-goal
regret on boundary families against oracle, random, and persistence-scored choice.
AD-WM overlaps, so claim it only as a checkpoint- and contact-specific result.

**N6 — Deprioritize ERASURE-001.** Removing a rank-2 linear subspace from a 192-d
isotropic embedding will almost certainly pass its own removal gate. The predictor
can still read position nonlinearly (the plan concedes this), and the intervention
pushes inputs off-manifold. The likely outcome is a null or uninterpretable effect,
and it moves further from H1–H3. Revisit only after N1–N2 establish a physical
effect worth explaining.

**N7 — Recentre the course report.** Suggested story:
1. A rigorous branching and contact-labelling harness for a frozen LeWM
   (methodological asset).
2. The encoder carries block pose. A readout's usefulness depends on the effect
   size, and absolute sub-pixel gates are the wrong criterion (lesson).
3. The predictor is action-sensitive, but its block predictions are
   persistence-like or shrunk around contact (the finding, pending N1/N2 with fresh
   confirmation).
4. Limitations: one checkpoint, simulator lineage.

## 6. What I could not verify

- Any number that depends on checkpoint inference. Hugging Face was blocked here, so
  N1–N5 are proposals, not results.
- Contents of the ignored `outputs/` (per-endpoint gallery scores, motion strata),
  so the stationary-endpoint dilution in §3.5 is inferred from the design, not
  measured.
- Whether the training archive has state columns, and what policy generated it.
- The R² estimate in the TL;DR is back-of-envelope. It assumes challenge block
  coordinates are uniform in [175, 337] (σ ≈ 47 per axis, from `PilotConfig`) plus
  the wider ordinary resets. Report the real R².
- Claims about external papers are taken from the project's own reviews. I did not
  re-open the sources.
