# weiss_schwarz_rl

Lean thesis RL pipeline for Weiss Schwarz.

This checkout keeps the code, configs, and retained artifacts needed to train,
evaluate, document, and defend the thesis results. Historical experiment logs,
old probe configs, debug bundles, and archive notes are intentionally excluded
from the active tree.

## Start Here

- [Getting started](docs/getting_started.md) for setup and a tiny smoke run.
- [Docs hub](docs/README.md) for the ownership map.
- [Thesis workflow](docs/thesis_workflow.md) for canonical train/eval/figure commands.
- [Artifacts](docs/artifacts.md) for retained thesis evidence and current selected runs.

## Install

```powershell
uv sync --extra dev
uv sync --extra dev --extra sim
```

The simulator-backed path expects `weiss-sim>=1.2.0,<2`.

## Verify

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

This runs repo hygiene, placeholder checks, Ruff, format checks, configured
mypy, Vulture, pytest, and wrapper dry-runs.

## Canonical Workflow

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_smoke --format png
```

Configuration details live in [docs/configuration.md](docs/configuration.md)
and [configs/README.md](configs/README.md).

## Retained Artifacts

`runs/`, `diagnostics/`, `vast_artifacts/`, and `thesis_figures_final/` contain
the report-retained evidence and adjacent provenance artifacts. Treat those
outputs as read-only unless deliberately replacing a thesis artifact.

## Repo Shape

- `python/weiss_rl/`: package code.
- `configs/`: small active config surface and seed files.
- `docs/`: concise thesis workflow, artifact, and validation docs.
- `runs/`, `diagnostics/`, `vast_artifacts/`, `thesis_figures_final/`: retained thesis evidence.
