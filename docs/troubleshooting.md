# Troubleshooting

This page collects the most common installation and verification issues.

## Python and packaging

### Python 3.13 is not supported

This repo is intentionally narrowed to Python 3.10, 3.11, and 3.12.

Fix:

```bash
python3.12 -m venv .venv
uv sync --extra dev
```

### `uv sync` cannot resolve `torch`

The repo prefers the CUDA 12.4 PyTorch index only on the supported CPython Linux/Windows wheel targets and otherwise falls back to PyPI.

Fix:

- switch to Python 3.10-3.12
- if you are on an unsupported Linux/Windows architecture or interpreter, let `uv` use the fallback PyPI build instead of forcing a CUDA-specific install
- re-run `uv sync --extra dev`
- if you need `weiss-sim 0.8.1` too, use `uv sync --extra dev --extra sim`

## Simulator import issues

### `ModuleNotFoundError: weiss_sim`

Fix one of these ways:

```bash
uv sync --extra dev --extra sim
```

or point at a sibling checkout:

```bash
export WEISS_SIM_PYTHONPATH=/Users/vwp/code/thesis/weiss-schwarz-simulator/python
# PowerShell
$env:WEISS_SIM_PYTHONPATH="C:\\Users\\Bruger\\Desktop\\thesis-repo\\weiss-schwarz-simulator\\python"
```

If the simulator needs a different interpreter, also set:

```bash
export WEISS_SIM_PYTHON=/path/to/python3.12
# PowerShell
$env:WEISS_SIM_PYTHON="C:\\Path\\To\\python.exe"
```

## Verification issues

### `python/scripts/verify_repo.py` fails on formatting

Run the formatter first:

```bash
uv run python -m ruff format python tests examples python/scripts
```

`make verify` runs the same verification entrypoint when GNU Make is available, so the same fix applies there.

### `python/scripts/verify_repo.py` or `scripts/run_local_ci_parity.sh` fails on missing tools

Install the repo dev dependencies first:

```bash
uv sync --extra dev
```

If you are not using `uv`, make sure the editable install includes dev extras:

```bash
python -m pip install -e ".[dev]"
```

Then run the parity script through `bash`:

```bash
uv run python python/scripts/verify_repo.py
bash scripts/run_local_ci_parity.sh
```

### Artifact checks fail on the toy/demo path

Regenerate the demo tree and inspect the generated files:

```bash
make artifact-hygiene
```

Then read:

- [Artifact contract](artifact_contract.md)
- [Runtime modes](runtime_modes.md)

## Where to look next

- [Getting started](getting_started.md)
- [Docs hub](README.md)
