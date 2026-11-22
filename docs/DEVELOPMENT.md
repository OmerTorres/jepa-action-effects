# Development environment

## Dependencies

`pyproject.toml` describes direct dependencies; `uv.lock` pins the complete resolution.
Use `make setup` after pulling changes. It uses `uv sync --locked`, which fails if
the lockfile is stale instead of silently changing it.

The base includes NumPy, SciPy, scikit-learn, Matplotlib, and PyYAML; developer tools
are Ruff and pytest. ENV-001 adds the source-pinned `stable-worldmodel` 0.0.5
candidate and the minimal PushT runtime, including the locked PyTorch stack.
MODEL-001 adds locked Transformers for the strict-loaded ViT encoder. The optional
`notebook` group adds ipykernel. Checkpoint bytes stay ignored under `checkpoints/`;
no GPU runtime or remote service is configured.

When changing dependencies, run `uv add PACKAGE` (or `uv add --dev PACKAGE`), then
`make check`, and review both metadata and lockfile. Document compatibility evidence
in the interface contract before adding model/simulator packages. Under a restricted
agent sandbox, prefix uv commands with `UV_CACHE_DIR=.cache/uv
UV_PYTHON_INSTALL_DIR=.cache/python` on the same shell command, as Make already does.

## Local and Colab roles

Develop evaluation code and run small simulator checks locally. Use Colab for
frozen inference only after profiling establishes a need. In a Colab terminal,
clone this repository, install uv, and run `make setup` and `make check`; invoke
scripts via `uv run --locked`. An existing Colab notebook kernel is a different
Python environment and does not automatically use `.venv`.

Python 3.10.21 matches LeWM's documented Python 3.10 setup, while the pinned
`stable-worldmodel` declares support for Python 3.10 or newer. ENV-001 established
immutable source and checkpoint revisions, but the missing training manifest and
action statistics mean scientific compatibility remains unproven. See
`INTERFACE_CONTRACT.md` before simulator or model work. Record OS, hardware,
accelerator, memory, package versions, throughput, and storage separately for each
measured machine. CPU, MPS, and CUDA must not be assumed numerically interchangeable
or deterministic.

## macOS editable-install repair

Python skips `.pth` files carrying macOS's hidden file flag. If that flag is present
on uv's editable-install files, the package and `jepa-doctor` console command cannot
be imported even though installation succeeded. `make setup` repairs this narrowly
by running `chflags nohidden` on `.pth` files under `.venv`; it does not change other
flags or files. Run `make repair-macos-pth` to repeat only that repair.

## Artifacts and provenance

Suggested generated paths: `data/<dataset-id>/`, `checkpoints/<model-id>/`, and
`outputs/<run-id>/`. These are ignored; a Git clone will not reproduce their contents.
Create a tracked manifest under `docs/manifests/` when real artifacts exist, recording:

- Code revision and dirty state; runtime and dependency lock hash.
- Upstream repository revision, checkpoint files/SHA-256, and licenses.
- Configuration and hashes, seeds, reset/warm-up recipes, family/split membership.
- Action units, timing, preprocessing, contact-label semantics, exclusions.
- Artifact paths, hashes, retrieval instructions, runtime, and completion status.

Save `make doctor`'s JSON output with runs (use `make -s doctor` for clean JSON).
For numerical experiments record all random seeds and deterministic settings;
verify replay rather than assuming that a seed guarantees determinism.

## Sealed experiment runners (from 2026-09-29, D026)

- **Validate before confirming.** A new sealed runner must call
  `provenance.require_valid(...)` before opening or resuming confirmation. Use the default
  strict mode; `mode="legacy"` is only for completed historical runs.
- **Per-family records** must carry `input_sha256` and the inference fingerprint.
- **Manipulation checks** should use `protocol_gates.readout_gate_per_cell` (every
  dose × structure cell used in a contrast) and the two-sided
  `protocol_gates.faithful`.
- **Completed runners are historical.** Do not edit any file listed in a completed
  run's `_sources()`: that breaks its seal. Put new logic in new modules.
- **Read-only check of completed runs** (legacy mode):
  `python -m jepa_action_effects.provenance`.

## Agent configuration

Project guidance is in `AGENTS.md`; the focused skill is in
`.agents/skills/jepa-study/SKILL.md`. It can be requested as `$jepa-study`. If it does
not appear in a current session, restart Codex or explicitly ask it to read that file.
No global skills, plugins, account connections, or agent permission settings are changed.

Setup references: [Codex skills](https://learn.chatgpt.com/docs/build-skills),
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
