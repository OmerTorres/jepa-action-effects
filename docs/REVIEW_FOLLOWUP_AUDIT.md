# Review of the review follow-up — 2026-09-29

## Verification of fixes at de678d9 — REVIEW-BRANCH-004

The three findings below are addressed by D027 and commit `de678d9`:

- The ratio-denominator explanation is corrected, with historical text preserved.
- The validator checks selected probe bytes and cached-row fingerprints. Strict
  mode rejects missing record metadata; historical runs use explicit legacy mode.
  The mutation tests cover the previously reproduced counterexamples.
- Zero/near-zero predicted vectors are excluded from direction statistics and
  counted; all-zero and mixed-vector fixtures pass.

Verification: `make check && make test` passed (**151 tests, no skips**, 19 existing
Gymnasium warnings; pytest 11.52 seconds). Read-only `check_completed_runs()` passed
for PHYS-002, SHIFT-002 and H3-PLAN-002 in legacy mode, with missing historical
metadata explicitly reported. Comparing the regenerated follow-up manifest against
`7e7cd78` found no change to any existing numerical field: only the metrics-source
hash changed and 31 `undefined_angle_count` fields were added, all zero.

No new actionable issue was found in this patch. This closes the three specific
findings below, not an end-to-end certification of a future experiment runner.
Runner integration remains deferred as documented: supply the current inference
fingerprint, applicable probe selection, verified family index and cache paths,
and test the actual strict-mode confirmation/resume path before new confirmation.
The full collection/training/inference pipeline was not rerun during this review.

The remainder of this document preserves the review at `e241f54`.

Reviewed by Codex at Omer's request to follow up on the branch review.
Scope: changes through local commit `e241f54`, especially
`REVIEW_FOLLOWUP_RESULT.md`, the new measurements and provenance validator.

## Assessment

The follow-up substantially addresses the original review. The absolute-endpoint
control, episode-clustered intervals, per-dose readout checks and explicit H3
corrections are useful. The length/direction analysis improves the scientific
description: projection slope is not a literal fraction of distance traveled.
Historical seals are preserved and the new analysis is labeled exploratory.

`make check && make test` passed: **148 tests, no skips**, 19 existing Gymnasium
casting warnings, 12.21 seconds. I ran `check_completed_runs()` read-only and all
three completed runs passed its implemented checks. Applying the new per-cell
readout gate to every reserved cell in the saved follow-up manifest also passed.
I inspected the rerun code and its saved zero-difference results; I did not repeat
that full checkpoint rerun in this review.

## Remaining findings

### 1. Reverse the explanation of readout bias in the ratio

`REVIEW_FOLLOWUP_RESULT.md:69–73` gets both directions backward.
The reported ratio is R = slope_pred / slope_real. Holding the numerator fixed,
a falling denominator raises R, while a rising denominator lowers it.

- Challenge slope_real falls from 1.045 to 0.993. This **opposes** the observed
  decline in R; it cannot account for that decline by itself.
- Archive slope_real rises from 0.939 to 0.967 for constant controls. This
  **contributes to** the decline in R, rather than working against it.

The raw predictor slopes still fall in both cases:

| Constant-control contrast | slope_pred, low → high | R, low → high |
| --- | --- | --- |
| Challenge 0.30 → 0.60 | 0.954 → 0.739 | 0.913 → 0.744 |
| Archive ×1 → ×2 | 0.964 → 0.855 | 1.028 → 0.884 |

These are point estimates read from `review-followup-runtime.json`, not newly
confirmed contrasts. The directional explanation needs correction; the observed
trend is not eliminated.

### 2. Provenance validation remains incomplete as a future-run safeguard

`provenance.py:87–91` verifies the selection JSON's hash, but never hashes the
probe file named by `selection['path']` against `selection['path_sha256']`.
Its cache loop at lines 113–130 also ignores a per-record inference fingerprint.

Two isolated mutation checks using the existing mock-run fixture both returned
`valid=True`:

1. Modify the selected probe's bytes, keeping its selection JSON and seal intact.
2. Give a cached record an explicitly different weights/scaler fingerprint,
   keeping its input file hash correct and re-sealing that cached record.

The existing `load_block_probe()` separately rejects modified probe bytes, so this
is not evidence that the historical scoring path silently loaded a changed probe.
It does mean the new standalone validator cannot certify everything its documented
future-run role requires. Add artifact-byte and cache-fingerprint checks, with a
strict mode for new runs that requires necessary metadata. Keep a separate explicit
legacy-audit mode for historical records that lack fingerprints/input hashes.

Deferring integration into a new common runner is reasonable when no new
confirmation is planned. Describe REVIEW-FIX-002 as a validator foundation with
remaining enforcement work, rather than a complete safeguard for future runs.

### 3. Zero predicted vectors receive a spurious direction score

`followup_metrics.py:47–55` uses atan2(cross, dot) even when predicted length is
zero. For true displacement `[10,0]` and prediction `[0,0]`, `decompose()` reports
mean/median direction error **0 degrees**. The direction is undefined; the
prediction missed the motion.

Retain zero vectors in length/projection statistics, exclude them from angular
statistics, and report a missed-motion/undefined-angle count. If no direction is
defined, return an unavailable angle rather than zero. Add a zero-prediction fixture.
This is a reproduced edge-case defect, not a finding that the reported 19-degree
checkpoint measurement contains such zero vectors.

## Scientific reading

The reported rerun strengthens confidence in reproducibility. PHYS-002 now has a
more accurate description: readout-decoded predicted displacement lengths are
about 0.63 of true lengths in aggregate, with roughly 19-degree mean direction
error, yielding a 0.56 projection slope. Those summaries use different weightings
and should not be treated as an exact multiplicative decomposition.

SHIFT-002 shows a falling slope ratio and length ratio as command scale rises.
At some doses predicted lengths exceed true lengths, so uniform shortening is
not the appropriate description. H1 and H3 remain unestablished, and neither the
new decomposition nor the original studies identify a causal mechanism.

No production code, sealed results or scientific decision was changed in this
review. Next: correct the sign explanation before the report; complete the two
small helper fixes before reusing them in a new protocol.
