# Getting Started (Dev)

Goal: get a fresh clone to a successful manifest smoke run without 1:1 help.

## Prereqs
- Python >= 3.10
- `uv` installed and available in your terminal (`pip install uv`)
- Access to `weiss_sim` with `export_spec_bundle()` available.
  - If `weiss_sim` is already installed in your active Python environment, the probe uses that first.
  - Otherwise it checks `WEISS_SIM_PYTHONPATH` if set.
  - Otherwise it looks for a sibling checkout at `../weiss-schwarz-simulator/python` relative to this repo.
  - If the simulator needs a different interpreter, also set `WEISS_SIM_PYTHON=/path/to/python3.12`.

Repo paths in this guide are relative to the repo root.

## Install
From the repo root:

```bash
uv sync
```

If `uv sync` fails with `No pyproject.toml found`, you are not in the repo root. In that case, include your current directory (`pwd`) plus the full error output when asking for help.

## Manifest smoke test (<2 minutes CPU)
This does not execute training. It verifies:
- the train entrypoint runs
- the strict stack config loader accepts the dedicated smoke stack
- the run scaffold records the real simulator spec bundle and provenance

Run:

```bash
uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-id smoke_local
```

Expected console lines:
- `Loaded stack config with 0 components`
- `Wrote manifest: runs/smoke_local/manifest.json`

## Expected output files
After the smoke run, you should have:
- `runs/smoke_local/manifest.json`
- `runs/smoke_local/spec_bundle.json`
- `runs/smoke_local/config_canonical.json`

Inspection:

```bash
cat runs/smoke_local/manifest.json
```

You should see fields like `run_id256`, `spec_hash256`, `config_hash256`, `simulator`, and `spec_bundle`.

## Common errors

`ModuleNotFoundError: No module named 'weiss_rl'`
- Cause: running outside the managed environment or without installing the repo package.
- Fix: run the command via `uv run ...` and ensure `uv sync` succeeded.

`Unable to collect simulator provenance via weiss_sim.export_spec_bundle()`
- Cause: the train scaffold could not import a compatible `weiss_sim` package or find a working simulator checkout/interpreter.
- Fix: either install `weiss_sim` into the active environment, or set `WEISS_SIM_PYTHONPATH` (and, if needed, `WEISS_SIM_PYTHON`) to a working simulator environment.

`--stack-config` not found or YAML load error
- Cause: wrong working directory or incorrect config path.
- Fix: run from repo root and verify `configs/stack_smoke.yaml` exists.
