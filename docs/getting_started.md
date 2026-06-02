# Getting Started

```powershell
uv sync --extra dev --extra sim
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

Run a tiny simulator-backed smoke:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
```

The active config list is in [configuration.md](configuration.md).
