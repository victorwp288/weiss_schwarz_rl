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
PYRUN := uv run --extra dev python
SYNC_MSG := "[make] using uv"
endif

.PHONY: sync lint fmt type test check check-placeholders train-min train-inline-smoke eval-dev figures clean

FIGURE_FORMAT_ARGS = $(foreach fmt,$(FORMATS),--format $(fmt))
FIGURE_ID_ARG = $(if $(FIG_ID),--fig-id "$(FIG_ID)")

sync:
	@echo $(SYNC_MSG)
ifeq ($(UV),)
	@$(PY_SYS) -m venv $(VENV) || (echo "Failed to create venv. On Debian/Ubuntu install python3-venv." && exit 1)
	@$(PY_VENV) -m pip install -q --upgrade pip
	@$(PY_VENV) -m pip install -q -e ".[dev]"
else
	@uv sync --extra dev
endif

check-placeholders:
	@$(PYRUN) python/scripts/check_core_placeholders.py

lint:
	@$(PYRUN) -m ruff check python tests examples

fmt:
	@$(PYRUN) -m ruff format python tests examples

type:
	@$(PYRUN) -m mypy

test:
	@$(PYRUN) -m pytest -q python/weiss_rl/tests

check: check-placeholders lint fmt type test

train-min:
	@$(PYRUN) python/scripts/train.py --stack-config configs/stack_smoke.yaml

train-inline-smoke:
	@PYTHONPATH=$(abspath ../weiss-schwarz-simulator/python)$${PYTHONPATH:+:$$PYTHONPATH} $(PYRUN) python/scripts/train.py --stack-config configs/rl_stack_locked.yaml --run-label m3_08_smoke --device cpu

eval-dev:
	@$(PYRUN) python/scripts/eval.py --stack-config configs/stack_smoke.yaml

figures:
	@test -n "$(RUN_DIR)" || { echo "Usage: make figures RUN_DIR=runs/<run_dir> [FIG_ID=seat_bias] [FORMATS=\"pdf png\"]" >&2; exit 1; }
	@$(PYRUN) python/scripts/make_figures.py --run-dir "$(RUN_DIR)" $(strip $(FIGURE_ID_ARG)) $(strip $(FIGURE_FORMAT_ARGS))

clean:
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .ruff_cache .mypy_cache
