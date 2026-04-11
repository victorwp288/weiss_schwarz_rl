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

.PHONY: sync sync-sim lint fmt fmt-check type test verify check check-placeholders package-smoke simulator-check train-min train-inline-smoke toy-public-e2e artifact-hygiene artifact-contract eval-dev figures clean

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

sync-sim:
	@echo $(SYNC_MSG)
ifeq ($(UV),)
	@$(PY_SYS) -m venv $(VENV) || (echo "Failed to create venv. On Debian/Ubuntu install python3-venv." && exit 1)
	@$(PY_VENV) -m pip install -q --upgrade pip
	@$(PY_VENV) -m pip install -q -e ".[dev,sim]"
else
	@uv sync --extra dev --extra sim
endif

check-placeholders:
	@$(PYRUN) python/scripts/check_core_placeholders.py

lint:
	@$(PYRUN) -m ruff check python tests examples python/scripts

fmt:
	@$(PYRUN) -m ruff format python tests examples python/scripts

fmt-check:
	@$(PYRUN) -m ruff format --check python tests examples python/scripts

type:
	@$(PYRUN) -m mypy

test:
	@$(PYRUN) -m pytest -q python/weiss_rl/tests

verify:
	@bash scripts/run_local_ci_parity.sh

check: verify

package-smoke: sync
	@tmpdir="$$(mktemp -d)"; \
	$(PYRUN) -m build; \
	$(PY_SYS) -m venv "$$tmpdir"; \
	"$$tmpdir/bin/python" -m pip install -q --upgrade pip; \
	"$$tmpdir/bin/python" -m pip install -q dist/*.whl; \
	"$$tmpdir/bin/python" -c "import weiss_rl; print(weiss_rl.__all__)"; \
	rm -rf "$$tmpdir"

simulator-check: sync-sim
ifeq ($(UV),)
	@$(PYRUN) -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py
else
	@uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py
endif

artifact-contract: sync
	@$(MAKE) artifact-hygiene
	@$(PYRUN) python/scripts/paper_readiness_check.py --run-dir runs/toy_public_demo_ci

train-min:
	@$(PYRUN) python/scripts/train.py --stack-config configs/stack_smoke.yaml

train-inline-smoke:
	@PYTHONPATH=$(abspath ../weiss-schwarz-simulator/python)$${PYTHONPATH:+:$$PYTHONPATH} $(PYRUN) python/scripts/train.py --stack-config configs/rl_stack_locked.yaml --run-label m3_08_smoke --device cpu

toy-public-e2e:
	@rm -rf runs/toy_public_demo_ci
	@$(PYRUN) python/scripts/train.py --stack-config configs/rl_stack_locked.yaml --public-demo --run-label toy_public_demo_ci
	@$(PYRUN) python/scripts/eval.py --stack-config configs/rl_stack_locked.yaml --public-demo --run-dir runs/toy_public_demo_ci
	@$(PYRUN) python/scripts/make_figures.py --public-demo --final-eval-dir runs/toy_public_demo_ci/eval/final_eval --out-dir runs/toy_public_demo_ci/figures

artifact-hygiene:
	@$(MAKE) toy-public-e2e
	@$(PYRUN) python/scripts/artifact_scan.py --artifact-root runs/toy_public_demo_ci

eval-dev:
	@$(PYRUN) python/scripts/eval.py --stack-config configs/stack_smoke.yaml

figures:
	@test -n "$(RUN_DIR)" || { echo "Usage: make figures RUN_DIR=runs/<run_dir> [FIG_ID=seat_bias] [FORMATS=\"pdf png\"]" >&2; exit 1; }
	@$(PYRUN) python/scripts/make_figures.py --run-dir "$(RUN_DIR)" $(strip $(FIGURE_ID_ARG)) $(strip $(FIGURE_FORMAT_ARGS))

clean:
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .ruff_cache .mypy_cache
