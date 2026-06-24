# Docs

This folder is the active thesis documentation surface. It is intentionally
small; historical logs, probe notes, and archived experiment reports stay out
of the docs hub.

## Start By Task

| If you want to... | Read |
| --- | --- |
| Install, verify, train, evaluate, export figures, debug, or choose configs | [operations/thesis_workflow.md](operations/thesis_workflow.md) |
| Understand what happens during B1 and main training | [concepts/training.md](concepts/training.md) |
| Explain the policy/value model and structured legal-action head | [concepts/model.md](concepts/model.md) |
| Explain the reward objective, shaping, discounting, and learner perspective | [concepts/rewards.md](concepts/rewards.md) |
| Explain smoke eval, final eval, paired seeds, payoff folding, and readiness | [concepts/evaluation.md](concepts/evaluation.md) |
| Understand selected runs and retained evidence | [concepts/artifacts.md](concepts/artifacts.md) |
| Understand package layout, simulator assumptions, reproducibility, or public command boundaries | [concepts/architecture.md](concepts/architecture.md) |
| Deploy or run the human-play split frontend/backend | [deployment/human_play.md](deployment/human_play.md) |

## Local Pointers

Local READMEs under `configs/`, `tests/`, `runs/`, `deploy/`, and `web/` should
stay short. Put shared concepts in the owner docs above, then link from the
local README when needed.

## Folder Layout

- `operations/`: commands, setup, verification, troubleshooting, and daily
  workflow.
- `concepts/`: training, model, rewards, evaluation, artifacts, architecture,
  and simulator contracts.
- `deployment/`: deployment-specific notes.

## Verify

```powershell
uv run python -m pytest -q tests/weiss_rl/test_public_config_surface_docs.py
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```
