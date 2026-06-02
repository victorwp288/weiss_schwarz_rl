# weiss_schwarz_rl

Lean thesis RL pipeline for Weiss Schwarz.

This checkout keeps the code, configs, and retained artifacts needed to train,
evaluate, document, and defend the thesis results. Historical experiment logs,
old probe configs, debug bundles, and archive notes are intentionally excluded
from the active tree.

## Start Here

- [Docs hub](docs/README.md)
- [Thesis workflow](docs/thesis_workflow.md)
- [Configuration](docs/configuration.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Artifacts](docs/artifacts.md)
- [Testing](docs/testing.md)

## Install

```bash
uv sync --extra dev
uv sync --extra dev --extra sim
```

The simulator-backed path expects `weiss-sim>=1.2.0`.

## Verify

```bash
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

This runs repo hygiene, placeholder checks, Ruff, format checks, configured
mypy, Vulture, pytest, and wrapper dry-runs.

## Canonical Workflow

```bash
uv run python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run python -m weiss_rl.cli figures --run-dir runs/main_smoke --format png
```

Public thesis configs live under `configs/thesis/`:

- `b1_noleague.yaml`
- `main_league.yaml`
- `main_league_auto_gpu.yaml`
- `final_eval.yaml`
- `final_eval_gpu.yaml`
- `multideck_exploratory.yaml`
- `ablations/no_gru.yaml`
- `ablations/ppo_lite.yaml`
- `ablations/terminal_only_reward.yaml`

Compatibility wrapper presets live under `configs/presets/structured_acceptance_standard*.yaml`.

## Retained Artifacts

`runs/`, `diagnostics/`, `vast_artifacts/`, and `thesis_figures_final/` contain
the report-retained evidence and adjacent provenance artifacts. Treat those
outputs as read-only unless deliberately replacing a thesis artifact.

## Repo Shape

- `python/weiss_rl/`: package code.
- `python/scripts/`: thin compatibility shims.
- `configs/`: small active config surface and seed files.
- `docs/`: concise thesis workflow, artifact, and validation docs.
- `runs/`, `diagnostics/`, `vast_artifacts/`, `thesis_figures_final/`: retained thesis evidence.
