# GALLERY-001 — Validate a block-specific measurement and its contribution

Status: proposed next execution protocol, 2026-09-28. The preliminary renderer
checks in NOVELTY-002_MEMO.md are completed evidence; the experiment below has
not been run or frozen. This plan does not authorize predictor scoring or
opening READOUT-001 confirmation data.

## Objective and decision

Determine whether a fixed-agent, rendered block-position gallery provides a
reliable, sufficiently covered measurement in the frozen projected CLS space,
and whether using it would address a defensible gap in the closest work.
Complete the literature assessment and measurement feasibility before designing
another predictor experiment. A successful outcome is a justified next question;
a failed measurement or duplicated method is also a valid outcome.

The candidate question for a later study is: **does the frozen predictor prefer
the realized block position over plausible alternative positions when the
rendered agent pose is held fixed?** This is conditional, finite-gallery
recognition. It does not estimate continuous physical error, establish causal
contact reasoning, or restore the failed physical-readout gates.
Holding rendered agent pose fixed removes that variation from the template
bank; it does not mathematically separate agent/block features inside a nonlinear
embedding. A later predictor analysis must retain that attribution limitation.

## 1. Verify the contribution before making a novelty claim

Produce `docs/GALLERY-001_RELATED_WORK.md` from primary papers and official code.
Start with the sources recorded in LIT-001 and NOVELTY-002, including AD-WM;
verify each title, authors, version/date, source URL, and relevant equation or
code location rather than treating previous summaries as sufficient evidence.
Append corrections to historical literature claims where necessary.

Search for renderer-based latent evaluation, counterfactual object interventions,
nuisance-controlled representation evaluation, analysis-by-synthesis and latent
pose retrieval. Follow one citation hop from the closest papers and examine up
to twelve directly relevant primary works in detail. Keep a dated search log,
queries, inclusion/exclusion reasons, and unavailable sources. The bounded search
supports a scoped comparison, never a claim that no prior work exists.

For each close work, compare frozen-model evaluation, learned state decoding,
rendered pose templates, control of agent/background, nonidentical real-frame
controls, contact stratification, and demonstrated model findings. Distinguish:

- **Method novelty:** the actual measurement procedure differs materially.
- **Empirical contribution:** an established measurement answers a specific
  unresolved question for this checkpoint/distribution.
- **Replication/validation:** the method and question are substantially covered.

Write one contribution sentence and a claim-to-required-evidence table. Do not
equate a new metric name, a new simulator, or failed readouts with novelty. A
direct precedent does not invalidate careful empirical work, but must change its
framing. If the search remains incomplete, mark novelty unresolved explicitly.

## 2. Fixed scope, inputs, and runtime

- Use 30 of the existing READOUT tuning families: 20 challenge and 10 ordinary,
  all six candidates, terminal frames at the existing two-step horizon.
- Include the twelve already preselected LATENT verification families. From the
  remaining tuning pool, select fourteen challenge and four ordinary families
  uniformly without replacement. Use master seed `20260930`, with independent
  derived streams for membership, template directions, query directions, visual
  examples, and bootstrap sampling. Seal IDs before rendering or encoding.
- This is exploratory reuse of development data. Inspect partition metadata to
  exclude all confirmation IDs before reading family payloads. No collection,
  readout fitting, raw CLS diagnostics, predictor cache loading, predictor
  inference, or checkpoint training.
- Use Python 3.10.21, the locked environment, CPU float32 encoding, and float64
  metric reductions. Keep the checkpoint and simulator revisions unchanged.
- Cap experimental execution at 1,740 cumulative wall-clock seconds, with 60
  additional seconds reserved for report finalization. Include replay, rendering,
  encoding, cache validation and statistical reductions in the ledger. Time
  implementation, tests and literature work separately. Check between candidates,
  encoding batches and bootstrap batches; record unavoidable boundary overruns.
- Seal configuration, source/lock/checkpoint/input hashes and stream seeds before
  experimental work. Save completed candidate records atomically. A matching
  rerun resumes within the same budget; changed signatures or corruption fail.

Add reusable modules under `src/jepa_action_effects/` and a staged command:
`python -m jepa_action_effects.gallery_audit {audit,render,encode,report}`.
Use ignored `outputs/gallery001/`; every dependent stage validates upstream
signatures. A budget interruption yields an explicitly incomplete report.

## 3. Render controlled comparisons before encoding them

Replay each saved reset, warm-up and candidate control sequence. Verify the
terminal state and exact saved RGB. Set block angle before position and reindex
block shapes when rendering alternatives. Do not step physics after intervention.
Restore the original pose and require exact saved RGB again for every candidate.
Check that agent pose, goal, background, shape and simulator step count are
unchanged. A replay or restoration mismatch is an integrity failure and blocks
encoding; do not compensate by allowing a new pixel tolerance.

Freeze a translation-only gallery with original block orientation. Treat each
scale separately: negative block-center offsets have radius **10.24** and
**20.48** workspace units (`2 delta` and `4 delta`, with delta unchanged at 5.12).
For each endpoint/scale, generate eight directions at 45-degree intervals from
a seeded angle. In that recorded order, take the first four admissible,
RGB-distinct negatives. Include the exact true-position image as the fifth
template. Store all rejected directions and reasons; do not resample or enlarge
the bank when fewer than four negatives are valid.

Validate transformed collision geometry using the actual T-block shapes, agent
shape and walls. Alternatives must have no positive penetration beyond a fixed
`1e-6` workspace-unit numerical tolerance against agent or walls, and remain
within the rendered workspace. Record existing true-endpoint penetration
separately; do not reinterpret simulator contact as absence of motion. Geometry
and RGB checks are allowed selection rules; embedding distances are not.

The true template equals the real frame. Its nearest-template success is
therefore only an identity/collapse control. **It is not a representation-quality
gate.** Add two nonidentical positive queries per endpoint: translate the block
center by radius **1.28** (`delta/4`) along the first two admissible directions
from an independently seeded eight-direction ring. Keep agent and orientation
fixed, apply the same geometry rules, and require query RGB to differ from the
true template and from each other. Do not include queries in the gallery. If
two such queries cannot be built, mark the endpoint unavailable for the
nonidentical-query gate; never substitute the exact true frame.

Both queries retain the true template as their physical nearest position:
their displacement is at most 1.28 while every negative is at least 10.24 away
from the original center. Synthetic poses need not be dynamically reachable;
record this limitation instead of treating them as counterfactual trajectories.
Orientation and action-effect prediction remain outside this experiment.

Save geometry overlays for four families (two per cohort) chosen before
encoding, plus all integrity failures. Inspect block boundaries, held agent,
positive queries and negatives. Test that image changes lie within the union
of old/new block footprints with a fixed three-output-pixel rasterization
allowance. Unexpected background/goal changes block the run.

## 4. Test representation sensitivity without self-matching

Strict-load the frozen encoder in evaluation mode, using the existing
preprocessing and projected CLS dimension 192. Never fit a metric, normalize
features across data, whiten, or select dimensions. Reuse verified real-image
caches only with the exact existing fingerprint. Store all synthetic encodings
in the GALLERY artifact namespace keyed by image and model/preprocessing hash.

Encode each nonidentical query q and its five templates z_k. Use
`d(q,z) = ||q-z||^2 / 192`. For each query record:

- `d_true` and the minimum negative distance `d_wrong`;
- signed margin `m = d_wrong - d_true`;
- true-position top-1 retrieval, counting numerical ties as failures;
- robustness ratio `R = (d_wrong-d_true)/max(d_wrong, eta)`.

Determine numerical floor `eta = max(1e-12, 100 * e_dup)`, where `e_dup` is the
maximum duplicate/order-check embedding MSE on all templates and queries from
the twelve preselected verification families. Require maximum absolute latent
discrepancy <=1e-6. This floor measures numerical repeatability, not physical
precision. A tie means `abs(m) <= eta`; distances below the floor are reported
as collapsed, never removed from otherwise valid galleries.

Report exact-frame self-retrieval separately as an implementation control.
Use uniform gallery guessing (20% expected top-1) and a synthetic constant
embedding control. Mock constant/oracle encodings validate metric behavior;
they provide no trained-model evidence. Save per-template distances and queries
so apparent success cannot hide collapsed or unusually easy alternatives.

## 5. Predeclared feasibility gates and statistics

Apply these gates separately at each translation scale. They are proposed
engineering criteria for this gallery, not replacements for the existing
physical-error thresholds.

| Gate | Pass rule |
| --- | --- |
| Integrity | All selected branches replay exactly, all mutated poses restore exactly, nuisance checks pass, projected cache/model identity and duplicate/order checks pass |
| Geometry/query coverage | At least 80% of the 120 challenge and 60 ordinary endpoints have four valid negatives and two valid nonidentical queries; at least ten families in each cohort have four or more eligible candidates |
| Nonidentical-query retrieval | Family-averaged top-1 >=95%, separately in challenge and ordinary cohorts |
| Nonidentical-query margin | Family-weighted fifth percentile of R >=0.5, separately in each cohort |
| Completion | All 30 families processed or explicitly marked geometrically unavailable, all mandatory checks completed, within recorded runtime limits |

A margin ratio of 0.5 requires the nearest wrong-template MSE to be at least
twice the true-template MSE. This tests local stability at the prescribed
translation scales; it cannot certify the encoder at arbitrary poses or decode
predicted latents accurately.

Average the two queries within each candidate, candidates within each family,
then families. For the R quantile give each family equal total weight, each
eligible candidate equal weight within its family, and its queries equal weight.
All families with at least one eligible candidate contribute to these metrics;
the four-candidate rule is an additional coverage gate, not a score-based filter.
An empty denominator is unassessable, never zero error or a passing result.
Report exclusions against the complete preselected denominator. Bootstrap 2,000
families with replacement for exploratory percentile 95% intervals, preserving
cohort counts in pooled estimates. Point estimates determine the stated
engineering gates; intervals describe uncertainty and do not certify population
performance.

Report both cohorts and saved contact/motion labels. For pair-regime summaries
(`neither`, `exactly_one`, `both`), deduplicate participating endpoints within
each family; require ten contributing families for an assessable summary. Do
not infer new contact labels for unstepped synthetic images. Sparse regime
coverage limits any later contact claim even if pooled feasibility passes.

## 6. Required tests and evidence

Before real encoding, test angle/position ordering with the offset center of
mass, transformed geometry, tangent/penetrating/wall cases, exact restoration,
unchanged nuisance state and physics-step count, deterministic bounded proposal
lists, unavailable-gallery handling and rejection records. Retain existing
replay/contact tests.

Test nonidentical query generation and self-match separation; oracle and
constant embeddings; ties/collapse; margin and ratio boundaries; family weights,
weighted quantiles, endpoint deduplication and sparse strata. Test membership
isolation, no confirmation payload access, projected-space/cache separation,
no predictor invocation, signature mismatch rejection, interrupted recovery,
budget exhaustion and incomplete reporting.

Run and record actual outcomes of `make check`, focused tests, `make test`,
`UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv sync --locked --check`,
and `git diff --check`. Preserve original experiments and append corrections
where a newly verified defect affects earlier interpretation.

Deliver the related-work matrix, sealed configuration, recipes/geometry reasons,
RGB/latent hashes, selected overlays, distances, bootstrap summaries, runtime
and storage ledger, a tracked `docs/manifests/gallery001-runtime.json`, and
`docs/GALLERY-001_RESULT.md`. Update TASKS, append LOGBOOK evidence and add an
outcome decision distinct from this proposed protocol.

## 7. Decision and stopping point

- **Proceed to a separately scoped predictor plan:** both scales pass and the
  literature review identifies a defensible empirical question. Freeze the
  exact gallery rules before any predictor comparison. That later plan must
  compare prediction, persistence and action permutation in the same galleries,
  handle conditioned agent pose and synthetic-distribution limits explicitly,
  and reserve independent evidence. No passing result here opens confirmation
  data automatically.
- **Narrow:** only one scale passes, a cohort/regime lacks coverage, or
  prior work already covers the method. State the supported scale/population
  and frame the contribution as empirical validation or replication. No silent
  threshold or geometry changes to rescue the result.
- **Stop this measurement route:** integrity fails, neither scale passes, or
  there is no defensible remaining contribution. Report the failed measurement
  and retain the completed study as a reproducible feasibility case study.
- **Incomplete:** budget or execution prevents required work. Save completed
  artifacts and list missing work; do not call an interrupted run a failed
  representation or completed feasibility audit.

After this report, stop. Main-study expansion, learned readout recovery, physical
competence claims and confirmation-set use remain outside GALLERY-001.
