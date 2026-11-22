# JEPA action effects

A course-scale study of contact-dependent object effects in a frozen latent JEPA
world model. The deliverable is a reproducible empirical answer, including a null
result. See the [original research plan](docs/RESEARCH_PLAN.md).

## Start developing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```sh
make setup
make check
```

This creates `.venv` using Python 3.10.21 and the included `uv.lock`. The Makefile
keeps Python downloads and caches in this workspace. Select `.venv/bin/python`
in your editor. No shell activation is needed for `uv run` or Make commands.

```sh
make format                 # format and fix lint
make doctor                 # JSON report; numerical + headless plotting smoke check
make notebook               # optional notebook kernel dependencies
make test                   # pytest, once implementation tests have been added
```

`make check` verifies the development environment, not scientific validity.
ENV-001 pinned the checkpoint-era PushT candidate; MODEL-001 recovered action
statistics and validated frozen CPU inference. Python 3.10 matches LeWM's documented
environment; support for newer interpreter series has not been established. See the
interface contract before using the simulator or checkpoint.

After placing the pinned archive under the ignored `data/` path, reproduce MODEL-001:

```sh
python -m jepa_action_effects.action_stats
python -m jepa_action_effects.model_smoke
```

PILOT-001 is exposed as three manifest-checked stages (or `all`):

```sh
python -m jepa_action_effects.pilot collect
python -m jepa_action_effects.pilot evaluate
python -m jepa_action_effects.pilot report
```

The completed pilot produced a STOP-expansion decision because real-frame state
readouts failed their validation gates. Detailed generated results remain under the
ignored `outputs/pilot001/` directory; tracked provenance is in
`docs/manifests/pilot001-runtime.json`.

READOUT-001 audits that measurement chain and runs a bounded recovery study:

```sh
python -m jepa_action_effects.readout_diagnostic audit
python -m jepa_action_effects.readout_diagnostic collect
python -m jepa_action_effects.readout_diagnostic fit
python -m jepa_action_effects.readout_diagnostic report
```

Its generated evidence is in ignored `outputs/readout001/`; the tracked runtime
manifest is `docs/manifests/readout001-runtime.json`. No projected readout passed
the unchanged gate, so confirmation labels remain reserved and main expansion is
paused.

DELTA-001 tested block-displacement readouts on the same existing families:

```sh
python -m jepa_action_effects.delta_readout audit
python -m jepa_action_effects.delta_readout fit
python -m jepa_action_effects.delta_readout confirm
python -m jepa_action_effects.delta_readout evaluate
python -m jepa_action_effects.delta_readout report
python -m jepa_action_effects.delta_breakdown
```

All real-frame displacement readouts failed the measurement gate. Confirmation
and predictor evaluation were unavailable by design. See
`docs/DELTA-001_RESULT.md`, `docs/manifests/delta001-runtime.json`, and ignored
`outputs/delta001/` for the scores and artifacts.

LATENT-001 audited candidate-to-future correspondence in projected CLS on the
60 existing READOUT tuning families:

```sh
python -m jepa_action_effects.latent_audit audit
python -m jepa_action_effects.latent_audit evaluate
python -m jepa_action_effects.latent_audit report
```

Correct actions matched real future embeddings better than wrong actions, and
predictions improved over latent persistence. This does not establish block
physics because the physical readout gate failed. See
`docs/LATENT-001_RESULT.md`, `docs/STUDY_SYNTHESIS.md`, and
`docs/manifests/latent001-runtime.json`.

## Project memory

- [Task board](docs/TASKS.md): current status, proposed owners, dependencies, acceptance gates.
- [Logbook](docs/LOGBOOK.md): dated work, commands, evidence, blockers, next handoff.
- [Decisions](docs/DECISIONS.md): development choices and scientific freeze status.
- [Interface contract](docs/INTERFACE_CONTRACT.md): the first technical task.
- [Development guide](docs/DEVELOPMENT.md): dependencies, artifacts, Colab, experiments.
- [Agent instructions](AGENTS.md): shared working conventions.
- [JEPA study skill](.agents/skills/jepa-study/SKILL.md): reusable research workflow.

Source lives under `src/jepa_action_effects/`; tests live under `tests/`.
Generated `data/`, `checkpoints/`, and `outputs/` are ignored by Git. Small reports,
protocols, and manifests should be tracked under `docs/`; store large artifacts separately.

Next: prepare the course report from the bounded results. **FREEZE-001** and
main-data expansion remain paused. **LIT-001** is completed in
`docs/LIT-001_REVIEW.md`; the original research plan remains preserved as
historical context.
