# TRAIN-001 runbook

Protocol: [TRAIN-001_PLAN.md](TRAIN-001_PLAN.md) with the D028 amendments in
[DECISIONS.md](DECISIONS.md). Code: `src/jepa_action_effects/train001*.py`.

## Prerequisites

The prerequisites below are the same as for the earlier studies (see
`INTERFACE_CONTRACT.md` and `NEXT-001_RUNBOOK.md`):

- **Environment.** `make setup` gives the locked environment. Python 3.10.21 and
  torch 2.14 (CPU) come from `uv.lock`.
- **Checkpoint.** `checkpoints/lewm-pusht/{weights.pt,config.json}` at revision `22b330c2…`.
  Audit checks both hashes.
- **Archive.** The extracted HDF5 at
  `data/quentinll-lewm-pusht/655cd446…/pusht_expert_train.h5` (46,300,921,856 bytes).
  Production audit hashes it fully.
- **Action statistics.** `outputs/model001/action-stats.json`.
- **READOUT-002 artifacts.** `outputs/readout002/selection.json(.sha256)` and
  `models/mlp-2026100{4,5,6}.npz`.
- **Earlier study outputs** under `outputs/`. They form the exclusion registry, and audit
  fails if a known source is missing:
  - pilot001, readout001
  - phys002, planning001, planning002
  - shift001, shift001-prerevision, shift002
  - dist001

## Stages

Run the stages in order from the project root. Every stage is charged to the
cumulative 24-hour ledger.

```bash
export UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=$PWD/.cache/matplotlib XDG_CACHE_HOME=$PWD/.cache
```

```bash
for stage in audit collect-development cache pilot train collect-confirmation seal confirm report; do .venv/bin/python -m jepa_action_effects.train001 "$stage" || break; done
```

| Stage | Does | Main outputs (`outputs/train001/`) |
| --- | --- | --- |
| `audit` | Hashes checkpoint, archive, readouts and sources; builds the exclusion registry; seals the code-generated config | `config.json`, `audit.json`, `exclusions.json` |
| `collect-development` | 1,000-episode weight sample; 360 + 360 families; 60 ordinary windows; partitions. Simulator and archive only | `development/` |
| `cache` | Encodes real frames with the frozen encoder | `cache/development/`, `cache/development.json` |
| `pilot` | **First** seals `pilot/eligibility.json` (real-frame gates). Then runs implementation checks, rollout equivalence, the overfit diagnostic, the LR sweep and the budget projection | `pilot/` |
| `train` | Six runs (2 conditions × 3 seeds), then the calibration scalar and an exploratory development evaluation | `runs/`, `calibration.json`, `development-eval.json` |
| `collect-confirmation` | 120 + 120 fresh families and 120 ordinary windows, then encoding | `confirmation/`, `cache/confirmation*` |
| `seal` | Freezes the registry, fingerprints, hashes and expected keys. Run `make check && make test` first | `seal.json` |
| `confirm` | Strict `entry` validation, then per-record scoring, then `complete` validation and the summary | `confirm/records/`, `confirm/summary.json` |
| `report` | Manifest, figures and table draft (formatting only) | `docs/manifests/train001-runtime.json`, `docs/figures/train001/`, `docs/TRAIN-001_RESULT_DRAFT.md` |

## Exit codes

| Code | Meaning | Action |
| --- | --- | --- |
| 0 | Stage completed | Continue |
| 2 | Gate failure or data-integrity failure | Stop. Report it; do not relax thresholds |
| 3 | `budget_stop` | Resume only if budget remains; training resumes from `runs/*/resume.pt` |
| 4 | Another stage holds the ledger lock | Wait; never run stages concurrently |

## Resume and idempotence

- Sealed outputs are returned, never overwritten.
- Collection restarts deterministically from its seed streams; an unsealed phase
  directory is cleared first.
- Training resumes bit-for-bit on the same CPU and thread count. `confirm` skips existing
  records after validating them.
- **Never edit** `train001.py`, `train001_protocol.py`, `train001_budget.py`,
  `train001_collect.py`, `train001_model.py`, `train001_metrics.py` or
  `train001_provenance.py` after audit. Also leave the reused modules listed in
  `train001.sources()` and `uv.lock` unchanged. A change requires a new run identity (a
  new root). The budget ledger is shared, so a new identity does not reset the cap.
- If a record write is interrupted, the sidecar is written before the atomic move, so
  there is never an unverifiable record.

## Budget

`outputs/train001-budget/ledger.json` lists every session: stage, start, heartbeat,
end and status. A killed session is charged up to its last heartbeat plus 60 s. To
read the time used so far:

```bash
.venv/bin/python -c "from pathlib import Path; from jepa_action_effects.train001_budget import Budget; print(Budget(Path('outputs/train001-budget'), watchdog=False).used()/3600, 'h')"
```

## Implementation smoke check

`--smoke` runs every stage with tiny cohorts. It uses its own seed (20261199),
`outputs/train001-smoke/` and a separate ledger. It is implementation testing, never
evidence. The production audit adds its families and episodes to the exclusions.

## Tests

```bash
make check
```

```bash
make test
```

The TRAIN-001 tests are `tests/test_train001_{model,collect,metrics,budget,provenance}.py`.
They cover the parameter boundary, rollout alignment and gradient flow, weighted loss,
paired sampler, resume, admission vs integrity, gates, the bootstrap, the verdict
tables, calibration, the budget ledger and provenance mutations.

## Review follow-up (D029)

This step is read-only over the sealed outputs. It re-validates the run including
training lineage, compares contrasts across readouts, extracts exact cited numbers,
and regenerates the figures and table draft with descriptive labels:

```bash
.venv/bin/python -m jepa_action_effects.train001_followup
```

It writes `docs/manifests/train001-followup.json` and never modifies
`outputs/train001/`. Future TRAIN-style runners must call
`train001_followup.validate_with_lineage` before sealing and confirming. The sealed
TRAIN-001 validator does not check training-data lineage.
