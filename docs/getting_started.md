# Getting Started (Dev)

Goal: get a fresh clone to a successful smoke run without 1:1 help.

## Prereqs
- Python >= 3.10
- `uv` installed and available in your terminal: (pip install uv)

Repo paths in this guide are relative to the repo root.

## Install
From the repo root:

```bash
uv sync
If uv sync fails with “No pyproject.toml found”, you are not in the repo root or the project metadata has not been added yet. 
In that case, ask in the group chat and include your current directory (pwd) plus the full error output.

## Smoke test (minimal_loop, <2 minutes CPU)
This does not execute training. It verifies:
- The entrypoints runs.
- The stack config loads.
- A deterministic run artifact is written.

Terminal input:
uv run python python/scripts/train.py --stack-config configs/minimal_loop.yaml --run-id smoke_local

Expected console lines:
- Loaded stack config with 0 components
- Wrote manifest: runs/smoke_local/manifest.json

## Expected output files
After the smoke run, you should have:
- runs/smoke_local/manifest.json

Inspection:
- cat runs/smoke_local/manifest.json

You should see fields like run_id, created_utc, stack_config, and counts.

## Common Errors
ModuleNotFoundError: No module named 'weiss_rl'
Cause: running outside the managed environment or without installing the repo package.
Fix: run the command via uv run ... and ensure uv sync succeeded.


error: No pyproject.toml found
Cause: you ran uv from the wrong directory.
Fix: cd into the repo root and retry uv sync.


--stack-config not found or YAML load error
Cause: wrong working directory or incorrect config path.
Fix: run from repo root and verify configs/minimal_loop.yaml exists.
