# Keep downloads and plotting caches inside the workspace.
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.cache/python
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export XDG_CACHE_HOME ?= $(CURDIR)/.cache

.PHONY: setup repair-macos-pth check format doctor test notebook
setup:
	uv sync --locked
	$(MAKE) repair-macos-pth

# Python skips .pth files carrying macOS's hidden flag. Some local uv-created
# environments inherit that flag, which breaks editable installs and console scripts.
repair-macos-pth:
	@if [ "$$(uname -s)" = "Darwin" ] && [ -d .venv ]; then \
		find .venv -type f -name '*.pth' -exec chflags nohidden {} +; \
	fi

check:
	uv run --locked --no-sync ruff check .
	uv run --locked --no-sync ruff format --check .
	$(MAKE) repair-macos-pth
	uv run --locked --no-sync jepa-doctor

format:
	uv run --locked --no-sync ruff check --fix .
	uv run --locked --no-sync ruff format .

doctor:
	$(MAKE) repair-macos-pth
	uv run --locked --no-sync jepa-doctor

test:
	$(MAKE) repair-macos-pth
	uv run --locked --no-sync pytest

notebook:
	uv sync --locked --group notebook
	$(MAKE) repair-macos-pth
