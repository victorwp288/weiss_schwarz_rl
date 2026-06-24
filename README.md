# weiss_schwarz_rl

Lean thesis RL pipeline for Weiss Schwarz.

This checkout keeps the code, configs, and retained artifacts needed to train,
evaluate, document, and defend the thesis results. Historical experiment logs,
old probe configs, debug bundles, and archive notes are intentionally excluded
from the active tree.

## Start Here

- [Docs hub](docs/README.md) for the ownership map.
- [Thesis workflow](docs/operations/thesis_workflow.md) for setup, train/eval/figure commands, validation, and troubleshooting.
- [Training](docs/concepts/training.md), [model](docs/concepts/model.md), [rewards](docs/concepts/rewards.md), and [evaluation](docs/concepts/evaluation.md) for the explanation path.
- [Artifacts](docs/concepts/artifacts.md) for retained thesis evidence and current selected runs.

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

Configuration details live in [docs/operations/thesis_workflow.md](docs/operations/thesis_workflow.md)
and [configs/README.md](configs/README.md).

## Retained Artifacts

`runs/`, `diagnostics/`, `vast_artifacts/`, and `run_logs/` hold retained
evidence or provenance when they are present in the checkout. Treat these
outputs as read-only unless deliberately replacing a thesis artifact.

## Repo Shape

- `python/weiss_rl/`: package code.
- `configs/`: small active config surface and seed files.
- `docs/`: concise thesis workflow, artifact, and validation docs.
- `runs/`, `diagnostics/`, `vast_artifacts/`, `run_logs/`: retained evidence
  and provenance surfaces.
