# Makefile that works with either `uv` (preferred) or plain Python + venv.

UV := $(shell command -v uv 2>/dev/null)
VENV ?= .venv
PY_SYS ?= python3
PY_VENV := $(VENV)/bin/python

export PYTHONPATH := python

ifeq ($(UV),)
PY := $(if $(wildcard $(PY_VENV)),$(PY_VENV),$(PY_SYS))
PYRUN := $(PY)
SYNC_MSG := "[make] uv not found; using venv at $(VENV)"
else
PYRUN := uv run --frozen --extra dev python
SYNC_MSG := "[make] using uv"
endif

.PHONY: sync lint fmt type test check train-min eval-dev figures clean

sync:
	@echo $(SYNC_MSG)
ifeq ($(UV),)
	@$(PY_SYS) -m venv $(VENV) || (echo "Failed to create venv. On Debian/Ubuntu install python3-venv." && exit 1)
	@$(PY_VENV) -m pip install -q --upgrade pip
	@$(PY_VENV) -m pip install -q -e ".[dev]"
else
	@uv sync --frozen --extra dev
endif

lint:
	@$(PYRUN) -m ruff check python tests examples

fmt:
	@$(PYRUN) -m ruff format python tests examples

type:
	@$(PYRUN) -m mypy

test:
	@$(PYRUN) -m pytest -q python/weiss_rl/tests

check: lint fmt type test

train-min:
	@$(PYRUN) python/scripts/train.py --stack-config configs/minimal_loop.yaml

eval-dev:
	@$(PYRUN) python/scripts/eval.py --stack-config configs/minimal_loop.yaml

figures:
	@$(PYRUN) python/scripts/make_figures.py --out runs/figures/placeholder.txt

clean:
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .ruff_cache .mypy_cache
