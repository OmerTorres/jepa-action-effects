# Reproducing the course report

Two routes. **Route A** rebuilds the report assets (every number, figure and the PDF) from
the tracked manifests; it needs neither the checkpoint nor the archive, and it does not
recompute any scientific estimate. **Route B** reproduces the experiments; it needs the
pinned checkpoint and archive and writes ignored `outputs/`.

All commands run from the repository root on macOS or Linux with
[uv](https://docs.astral.sh/uv/) installed. Use zsh-safe exit checks (`$?`): `PIPESTATUS`
is empty in zsh.

## Setup (both routes)

```bash
make setup
```
```bash
make check
```
```bash
make test
```

`make setup` creates `.venv` with Python 3.10.21 from `uv.lock`. `make test` runs about
350 tests in about 20 s. Tests that need ignored artifacts skip on a fresh clone.

## Route A — report assets from tracked manifests (minutes)

```bash
.venv/bin/python -m jepa_action_effects.reporting
```

This writes `outputs/report001/numbers.tex` (LaTeX macros, set in math mode; interval
macros also get a point-estimate twin `\<name>Pt`), `outputs/report001/figures/` (PDF and
PNG: teaser, measurement method, readout architecture, readout robustness and fine-tuning),
`outputs/report001/inputs.json` (manifest hashes) and `docs/REPORT-001_CLAIMS.md` (claims
ledger). A missing manifest field raises an error; no placeholder is ever inserted. The
example panels read only the tracked `docs/manifests/report-examples.json` and the frames
under `docs/figures/report-examples/` (each checked against its recorded hash).

Compile the PDF with [Tectonic](https://tectonic-typesetting.github.io/) 0.17. The first
run downloads TeX packages into the cache directory, so it needs network access to
`relay.fullyjustified.net` (a TLS-intercepting proxy or sandbox makes it fail). The
expected result is a five-page PDF.

```bash
mkdir -p outputs/report001/pdf
```
```bash
cd reports/report001 && TECTONIC_CACHE_DIR="$PWD/../../.cache/tectonic" tectonic -X compile main.tex --outdir ../../outputs/report001/pdf
```

(The reporting command also creates `outputs/report001/pdf`; Tectonic refuses a missing
output directory.)

## Route B — experiments (Mac CPU)

### Prerequisites

| Item | Location | Check |
| --- | --- | --- |
| LeWM PushT checkpoint (`quentinll/lewm-pusht`, revision `22b330c2…`) | `checkpoints/lewm-pusht/{weights.pt,config.json}` | `weights.pt` SHA-256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| PushT archive (`quentinll/lewm-pusht` dataset) | `data/quentinll-lewm-pusht/<revision>/pusht_expert_train.h5` | see `docs/INTERFACE_CONTRACT.md` |
| Action statistics (MODEL-001) | `outputs/model001/action-stats.json` | `python -m jepa_action_effects.action_stats` |

### Stages, in dependency order

Each stage seals its outputs. Re-running a completed stage verifies and returns its seal;
it never overwrites. That is a provenance/reuse check, not a recomputation of the
estimates: to recompute, run on an empty `outputs/<study>/`. Tracked results are in
`docs/manifests/<study>-runtime.json`, and the result documents are in `docs/`.

Run the stages top to bottom (`python -m jepa_action_effects.<module> <stage>`). The
order is at stage level, not study level: SUPPORT-001 `collect` excludes every earlier
family, including both TRAIN-002 collections, and fails if they are missing.

| Step | Stages | Result doc |
| --- | --- | --- |
| 1. READOUT-002 (readout training) | `readout_scale collect`, `fit`, `report` | (TASKS row) |
| 2. PHYS-002 | `phys002 audit`, `collect`, `develop`, `seal`, `confirm`, `report` | PHYS-002_RESULT.md |
| 3. SHIFT-002 | `shift002 audit`, `collect`, `develop`, `seal`, `confirm`, `report` | SHIFT-002_RESULT.md |
| 4. H3-PLAN-002 | `planning002 audit`, `collect`, `develop`, `seal`, `confirm`, `report` | H3-PLAN-002_RESULT.md |
| 5. TRAIN-001 (fine-tuning) | `train001 audit`, `collect-development`, `cache`, `pilot`, `train`, `collect-confirmation`, `seal`, `confirm`, `report`; then `train001_followup` | TRAIN-001_RESULT.md, TRAIN-001_RUNBOOK.md |
| 6. CALIB-001 | `calib001 all` | CALIB-001_RESULT.md |
| 7. ROBUST-001 | `robust001 all` | ROBUST-001_RESULT.md |
| 8. TRAIN-002 collections | `train002 audit`, `collect-windows`, `collect-confirmation` | — |
| 9. SUPPORT-001 | `support001 map`, `fit`, `collect`, `cache`, `gate`, `seal`, `confirm`, `report`; then `support001_derived` (plotting summary, not sealed) | SUPPORT-001_RESULT.md |
| 10. TRAIN-002 training and test | `train002 cache`, `train`, `gate`, `seal`, `confirm`, `report`; then `training_curves` (exports the twelve runs' saved curves to `docs/manifests/training-curves.json` after checking each run hash) | TRAIN-002_RESULT.md |
| 11. Report examples (D039) | `report_examples` (after steps 9 and 10: selects one SUPPORT-001 and one TRAIN-002 designed family by true displacement, writes `docs/manifests/report-examples.json` and the example frames) | DECISIONS.md D039 |

Partial stage times measured on Omer's Mac (8 CPUs, 4 torch threads). They cover only
recent studies; historical setup, readout training and the earlier studies were not timed
end to end, so no total is claimed.
- TRAIN-001 production: 1.06 h.
- TRAIN-002 training: 47 min for six runs.
- SUPPORT-001: about 10 min (collection 6.4 min).
- ROBUST-001: under 1 min.

### Notes

- **Order matters for disjointness.** Each collector excludes every earlier study's
  seeds, archive episodes and history hashes. Running studies out of order would change
  which families are drawn.
- **Fresh families are deterministic.** Each study's collection seed is fixed in its
  protocol, so a clean re-run draws the same families from the same archive and
  simulator.
- **Provenance.** Sealed runs check source, input and checkpoint hashes, and refuse to
  score after any change.
- **Historical gaps.** Some early runs lack strict provenance fingerprints; the result
  documents disclose this. Legacy validation is not strict certification.

## Teammate reproduction subset (to be recorded in `docs/LOGBOOK.md`)

1. Route A on a clean clone. `outputs/report001/numbers.tex` and
   `docs/REPORT-001_CLAIMS.md` must regenerate identically (git diff empty). Figures are
   compared visually, because PNG bytes depend on the font and matplotlib build.
2. `robust001 all` with the shared `outputs/` artifacts: the reproduction gate must report
   1,888 CALIB-001 values matched.
3. `train002 confirm` on the completed root returns the sealed summary. This is a
   seal/provenance check only; it does not recompute the estimates.
4. Saved-record recomputation (needs the shared `outputs/train002/`, about 10 s):

   ```bash
   .venv/bin/python -m jepa_action_effects.train002_recompute
   ```

   It verifies the seal, loads all 5,040 saved confirmation records with their hash
   sidecars and identity/fingerprint/readout checks, recomputes the full summary with
   `train002_metrics.summarize` and the sealed settings, and compares every value with
   `outputs/train002/confirm/summary.json` and `docs/manifests/train002-runtime.json`.
   Expected: `"status": "match"`, `"records": 5040`, headline dose_omitted − balanced
   +0.82 [0.11, 1.63]. It runs no model and writes nothing; a match shows computational
   consistency with the saved records, not a fresh replication.
5. `python -m jepa_action_effects.support001_derived` with the shared
   `outputs/support001/`: `docs/manifests/support001-account-slopes.json` must regenerate
   identically (git diff empty).
6. With the shared `outputs/train001/runs` and `outputs/train002/runs`:
   `python -m jepa_action_effects.training_curves` must regenerate
   `docs/manifests/training-curves.json` identically (git diff empty).
7. With the shared `outputs/support001/` and `outputs/train002/`:
   `python -m jepa_action_effects.report_examples` must regenerate
   `docs/manifests/report-examples.json` and the PNGs in `docs/figures/report-examples/`
   identically (git diff empty). It selects the example families by the D039 rule (true
   displacement closest to the cohort median), verifies both seals and the family, cache and
   record hashes (the SUPPORT-001 prediction file's digest is recorded, not checked),
   and runs no world model.

Record who ran each step, the git revision, the commands, any differences and the
limitations.
