# Docs

This folder is the active thesis documentation surface. It is intentionally
small; historical logs, probe notes, and archived experiment reports stay out
of the docs hub.

## Start By Task

| If you want to... | Read |
| --- | --- |
| Install, verify, train, evaluate, export figures, debug, or choose configs | [thesis_workflow.md](thesis_workflow.md) |
| Understand selected runs and retained evidence | [artifacts.md](artifacts.md) |
| Understand package layout, simulator assumptions, reproducibility, or public command boundaries | [architecture.md](architecture.md) |
| Deploy or run the human-play split frontend/backend | [human_play_deployment.md](human_play_deployment.md) |

## Local Pointers

Local READMEs under `configs/`, `tests/`, `runs/`, `deploy/`, and `web/` should
stay short. Put shared concepts in the owner docs above, then link from the
local README when needed.

## Verify

```powershell
uv run python -m pytest -q tests/weiss_rl/test_public_config_surface_docs.py
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```
