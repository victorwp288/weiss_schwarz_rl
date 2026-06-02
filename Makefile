# Makefile that works with either `uv` (preferred) or plain Python + venv.

UV := $(shell command -v uv 2>/dev/null)
VENV ?= .venv
PY_SYS ?= python3
PY_VENV = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(VENV)/Scripts/python.exe)

export PYTHONPATH := python

ifeq ($(UV),)
PY = $(if $(wildcard $(PY_VENV)),$(PY_VENV),$(PY_SYS))
PYRUN = $(PY)
PYSIMRUN = $(PY)
SYNC_MSG := "[make] uv not found; using venv at $(VENV)"
else
PYRUN := uv run --extra dev python
PYSIMRUN := uv run --extra dev --extra sim python
SYNC_MSG := "[make] using uv"
endif

.PHONY: sync sync-sim lint fmt fmt-check type deadcode test verify check check-placeholders repo-hygiene standard-wrapper-smoke standard-auto-gpu-wrapper-smoke package-smoke simulator-check train-min train-b1-smoke train-main-smoke eval-smoke figures-smoke thesis-smoke train-inline-smoke toy-public-e2e artifact-hygiene artifact-contract eval-dev figures clean

FIGURE_FORMAT_ARGS = $(foreach fmt,$(FORMATS),--format $(fmt))
FIGURE_ID_ARG = $(if $(FIG_ID),--fig-id "$(FIG_ID)")
B1_LABEL ?= b1_smoke
MAIN_LABEL ?= main_smoke
B1_RUN ?= runs/$(B1_LABEL)
MAIN_RUN ?= runs/$(MAIN_LABEL)
SIMULATOR_CHECK_TESTS = \
	python/weiss_rl/tests/test_simulator_contract.py \
	python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py \
	python/weiss_rl/tests/test_heuristic_public.py -k simulator_native_heuristic_pool_matches_python_oracle_across_live_steps

sync:
	@echo $(SYNC_MSG)
ifeq ($(UV),)
	@$(PY_SYS) -m venv $(VENV) || (echo "Failed to create venv. On Debian/Ubuntu install python3-venv." && exit 1)
	@$(PY_VENV) -m pip install -q --upgrade pip
	@$(PY_VENV) -m pip install -q --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[dev]"
else
	@uv sync --extra dev
endif

sync-sim:
	@echo $(SYNC_MSG)
ifeq ($(UV),)
	@$(PY_SYS) -m venv $(VENV) || (echo "Failed to create venv. On Debian/Ubuntu install python3-venv." && exit 1)
	@$(PY_VENV) -m pip install -q --upgrade pip
	@$(PY_VENV) -m pip install -q --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[dev,sim]"
else
	@uv sync --extra dev --extra sim
endif

check-placeholders:
	@$(PYRUN) -m weiss_rl.diagnostics.core_placeholder_check_entrypoint

repo-hygiene:
	@$(PYRUN) -m weiss_rl.diagnostics.repo_hygiene_check_entrypoint

lint:
	@$(PYRUN) -m ruff check python tests examples python/scripts

fmt:
	@$(PYRUN) -m ruff format python tests examples python/scripts

fmt-check:
	@$(PYRUN) -m ruff format --check python tests examples python/scripts

type:
	@$(PYRUN) -m mypy python/weiss_rl/workflows/thesis_wrapper.py python/weiss_rl/workflows/eval_entrypoint.py python/weiss_rl/human_play/play_vs_model_entrypoint.py

deadcode:
	@$(PYRUN) -m vulture python/weiss_rl python/scripts examples --min-confidence 80

test:
	@$(PYRUN) -m pytest -q python/weiss_rl/tests

standard-wrapper-smoke:
	@$(PYRUN) -m weiss_rl.workflows.thesis_wrapper --preset standard --run-label standard_surface_ci --dry-run --skip-compare

standard-auto-gpu-wrapper-smoke:
	@$(PYRUN) -m weiss_rl.workflows.thesis_wrapper --preset standard-auto-gpu --run-label standard_auto_gpu_surface_ci --dry-run --skip-compare

verify: sync
	@$(PYRUN) -m weiss_rl.workflows.verify_repo_entrypoint

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
	@$(PYRUN) -m pytest -q $(SIMULATOR_CHECK_TESTS)
else
	@uv run --extra dev --extra sim python -m pytest -q $(SIMULATOR_CHECK_TESTS)
endif

artifact-contract: sync
	@$(PYRUN) -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint

train-min:
	@$(PYSIMRUN) -m weiss_rl.cli train-b1 --run-label train_min_smoke --profile smoke

train-b1-smoke: sync-sim
	@$(PYSIMRUN) -m weiss_rl.cli train-b1 --run-label "$(B1_LABEL)" --profile smoke

train-main-smoke: sync-sim
	@$(PYSIMRUN) -m weiss_rl.cli train-main --run-label "$(MAIN_LABEL)" --b1-run "$(B1_RUN)" --profile smoke

eval-smoke: sync-sim
	@$(PYSIMRUN) -m weiss_rl.cli smoke-eval --run-dir "$(MAIN_RUN)" --b1-run "$(B1_RUN)"

figures-smoke: sync
	@$(PYRUN) -m weiss_rl.cli figures --run-dir "$(MAIN_RUN)" --format png

thesis-smoke: train-b1-smoke train-main-smoke eval-smoke figures-smoke

train-inline-smoke:
	@PYTHONPATH=$(abspath ../weiss-schwarz-simulator/python)$${PYTHONPATH:+:$$PYTHONPATH} $(PYRUN) -m weiss_rl.training.train_entrypoint --stack-config configs/presets/structured_acceptance_standard.yaml --run-label m3_08_smoke --device cpu

toy-public-e2e:
	@rm -rf runs/toy_public_demo_ci
	@$(PYRUN) -m weiss_rl.training.train_entrypoint --stack-config configs/presets/structured_acceptance_standard.yaml --public-demo --run-label toy_public_demo_ci
	@$(PYRUN) -m weiss_rl.workflows.eval_entrypoint --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/toy_public_demo_ci
	@$(PYRUN) -m weiss_rl.workflows.figures_entrypoint --public-demo --final-eval-dir runs/toy_public_demo_ci/eval/final_eval --out-dir runs/toy_public_demo_ci/figures

artifact-hygiene:
	@$(MAKE) toy-public-e2e
	@$(PYRUN) -m weiss_rl.diagnostics.artifact_scan_entrypoint --artifact-root runs/toy_public_demo_ci

eval-dev:
	@$(PYRUN) -m weiss_rl.workflows.eval_entrypoint --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml

figures:
	@test -n "$(RUN_DIR)" || { echo "Usage: make figures RUN_DIR=runs/<run_dir> [FIG_ID=seat_bias] [FORMATS=\"pdf png\"]" >&2; exit 1; }
	@$(PYRUN) -m weiss_rl.workflows.figures_entrypoint --run-dir "$(RUN_DIR)" $(strip $(FIGURE_ID_ARG)) $(strip $(FIGURE_FORMAT_ARGS))

clean:
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .ruff_cache .mypy_cache
