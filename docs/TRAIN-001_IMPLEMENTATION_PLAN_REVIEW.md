# TRAIN-001 implementation-plan review

Reviewed 2026-09-30 against HEAD `03c7d39` and the user-supplied implementation
plan. This is a pre-execution review, not approval of measured results. No
TRAIN-001 collection, training, protocol freeze or confirmation was performed.

The measurement-failure policy and the REPORT-001 handoff scope correctly reflect
the conversation. Resolve the implementation ambiguities below before execution.
Line references refer to the supplied `Pasted text.txt` attachment, not source code.

## Required corrections

### 1. Separate candidate rejection from fatal integrity failure

**Attachment lines 35, 72–84.** The fatal list says any identity, overlap or
duplicate violation stops the study, while collection explicitly rejects unsuitable
proposals. A proposed archive window failing the replay identity tolerance is an
expected selection rejection; SHIFT-002 counts `identity_failed` and continues.
Likewise, drawing a previously used seed or duplicate proposal should reject that
proposal, not terminate the experiment.

Define two cases: reject and count invalid proposals before admission; stop when
an already admitted artifact violates the frozen identity/split/hash contract, or
when the attempt cap prevents reaching the fixed cohort size. Test both cases.
Apply source-specific family identities: archive replays can share simulator seed
0 and must be distinguished by archive episode/window and recipe, not seed alone.

### 2. Make image identity units explicit

**Attachment line 80.** SHIFT-002 computes mean absolute image difference on raw
uint8-scale values and compares it with **1.0**, not 1/255. The proposed 1/255 is
equivalent only if images are first converted to floats in [0, 1]. Specify raw-scale
MAE <= 1.0 to match the existing implementation. Comparing its current output with
1/255 would make the gate 255 times tighter. Add a fixture checking unit equivalence.

### 3. Define validation separately for entry/resume and completion

**Attachment lines 134–140.** Requiring exact result-record coverage before the
first confirmation call would reject the empty output directory; it would also
reject legitimate partially completed runs.

- Entry/resume: require complete sealed *input* coverage and model registry;
  existing result keys must form a valid, duplicate-free subset of expected keys.
  Missing results are pending work, not a provenance failure.
- Completion/report: require exact expected result coverage, including ordinary
  windows and every model/control condition. Reject incomplete summaries.
- Reject foreign, corrupt or stale records in either state. Use atomic record
  writes and test empty, partial, complete and corrupt-partial runs.

### 4. Bind result fingerprints to the actual fine-tuned model

**Attachment lines 50, 90, 136–139.** The reused `model_fingerprint` describes the
original frozen checkpoint and hardcodes CPU metadata. It cannot by itself identify
six different fine-tuned models. Explicitly extend the prediction fingerprint with
the actual checkpoint hash, condition, training seed, inference implementation,
device and precision. Decode records additionally identify the readout artifact.

Keep real-embedding cache identity tied to the frozen encoder and input bytes;
prediction cache identity must additionally depend on the actual trained weights
and supplied controls. Test swapping two otherwise valid trained checkpoints or
records across model identities. A registry of correct checkpoint files alone does
not prove that each cached prediction came from its claimed checkpoint.

### 5. Enforce the budget within stages

**Attachment lines 193–200.** Refusing to start after 24 hours does not prevent a
stage started just before the deadline from exceeding the cap. `try/finally` alone
also loses unpersisted timing if the process is killed.

Use a cumulative ledger across run-directory revisions, persist the active stage
start, and check remaining budget at bounded work units: proposals, encoding
batches, optimizer steps and evaluation batches. Stop safely and checkpoint before
starting another unit that cannot fit the remaining allowance. Use a process-level
deadline for a hung/long unit, with a documented small shutdown allowance. Prevent
concurrent experimental stages from racing the ledger. Include failure and resume
accounting in tests; changing a run identity must not reset the study budget.

### 6. Finish the gate definitions and claim truth table

**Attachment lines 105–108, 164–176.** “Coverage projected against ... >=50” does
not define a reproducible development gate. Recommend requiring **50 observed
moved families per cell** in both the 60-per-scene validation cohort and the
120-per-scene confirmation cohort. Record this choice before collecting data.

Define the floor numerator as the mean real-frame displacement error over **all
accepted families in the cell**, and the denominator as the median true
displacement length among moved families (>5.12). Compute the projection slope on
moved families; missing/degenerate statistics fail the measurement gate. Fix the
bootstrap seed and use the selected READOUT-002 probe for eligibility.

The decision table also needs the case where development and confirmation gates
pass but the benefit criteria do not. Report **physical benefit not established**,
with signed contrasts and intervals; do not call this measurement failure. Clearly
distinguish a supported error increase from an interval spanning zero. Record
ordinary-case non-inferiority separately: failure to establish it is not proof of
regression, and a physical improvement does not automatically mean improvement
without an ordinary-case tradeoff. Use the planned <=1.28 boundary consistently.

## Additional clarifications before freeze

- **Archive population (line 81):** SHIFT-002 required an archive rest window
  (block displacement <=1.0 during the recorded history). Removing that filter
  deliberately changes the population and introduces already-moving blocks; the
  archive state omits block linear/angular velocity at reset. Recommend retaining
  the existing rest filter for this bounded study. If it is removed deliberately,
  identify this as a new population rather than claiming unchanged SHIFT-002
  collection, and report pre-existing motion separately from action effects.
- **Direction cutoff (line 77):** norm <0.05 is a substantive eligibility threshold,
  not merely numerical protection against division by zero. Give its rationale and
  log its rejection rate before interpreting archive representativeness.
- **Checkpoint reset:** explicitly reload the original checkpoint and fresh optimizer
  for the overfit diagnostic, every learning-rate trial and all six full runs.
  A resume is the only exception. Test the starting checkpoint hash; avoid pilot
  weights leaking into full training. Fix CPU thread count (the prior benchmark
  used four) and define how the last short sampler batch wraps across epochs.
- **Device fallback (line 199):** the current environment reports neither MPS nor
  CUDA available. Do not promise an MPS fallback. Use CPU, or an actually available
  separately recorded GPU environment with parity checks and the same protocol.
- **Artifacts (lines 55, 145):** `docs/TRAIN-001_PLAN.md` is not present in this
  checkout yet. Preserve the supplied protocol and record amendments separately;
  do not describe them as already saved. No `dataviz` skill is exposed in this
  session; ordinary Matplotlib is sufficient for these scientific figures.
- **Provenance scope:** hash all reused code affecting collection, inference and
  statistics, not only the new runner. Before opening confirmation, also validate
  the sealed development eligibility and checkpoint-selection lineage. A changed
  JSON plus regenerated sidecar must not silently redefine the protocol.

## Disposition and validation

**Revise before execution.** These corrections do not require relaxing the
measurement thresholds or expanding the experiment. The three-arm comparison,
paired training design, secondary latent outcome, measurement-failure policy and
separate REPORT-001 authoring task can remain as agreed.

Read-only checks: repository status/HEAD; task board and latest logbook; source
inspection of SHIFT-002 collection, original inference fingerprints, provenance
validation and per-cell gates; local GPU availability; presence of the plan file.
No scientific tests or experiments were run. Documentation whitespace validation
is recorded in LOGBOOK.
