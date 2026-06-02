# Getting Started

Run commands from the repository root.

## Install

```powershell
uv sync --extra dev --extra sim
```

The simulator-backed path expects `weiss-sim>=1.2.0,<2`. If simulator import or
spec validation fails, use [simulator_compatibility.md](simulator_compatibility.md).

## Verify

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

For focused validation commands, use [testing.md](testing.md).

## Tiny Smoke

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_smoke --format png
```

Smoke outputs are plumbing checks only. Do not cite them as model-quality
evidence.

## Next

- Use [thesis_workflow.md](thesis_workflow.md) for thesis launches and selected-run reproduction.
- Use [configuration.md](configuration.md) before adding or advertising configs.
- Use [artifacts.md](artifacts.md) before replacing retained evidence.
