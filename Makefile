# project-wide commands

PYTHON ?= python

# export PYTHONPATH so tests and scripts can import the package tree
export PYTHONPATH := python

.PHONY: all lint fmt test train-min eval-dev figures-placeholder clean

# default to running quality checks and unit tests
all: lint fmt test

# linting; using ruff for fast checks – install via `pip install ruff`
lint:
	$(PYTHON) -m ruff check python

# formatting; assumes a formatter such as black is installed
fmt:
	$(PYTHON) -m black python

# run the unit tests in the repository
# the `-q` flag makes pytest quieter but you can remove it
# no need to manually set PYTHONPATH because we exported it above
test:
	$(PYTHON) -m pytest -q python/weiss_rl/tests

# minimal training invocation; change config path to suit your setup
train-min:
	$(PYTHON) python/scripts/train.py --stack-config configs/minimal_loop.yaml

# simple evaluation on the development stack config
eval-dev:
	$(PYTHON) python/scripts/eval.py --stack-config configs/minimal_loop.yaml

# placeholder target for generating figures from experiment output
figures-placeholder:
	@echo "figure generation is not implemented yet"

# clean up typical build artifacts (none are produced by this repo, but useful
# if you add bytecode, __pycache__, etc.)
clean:
	rm -rf __pycache__ python/**/__pycache__
