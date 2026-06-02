# Docs

This folder is the active thesis documentation surface. Historical logs,
one-off probe notes, and archived experiment reports are intentionally outside
the docs hub.

## Start By Task

| If you want to... | Read |
| --- | --- |
| Set up the repo and run a tiny smoke | [getting_started.md](getting_started.md) |
| Launch B1/main training, final eval, figures, or diagnostics | [thesis_workflow.md](thesis_workflow.md) |
| Understand selected runs and retained evidence | [artifacts.md](artifacts.md) |
| Check what a paper-grade run tree must contain | [artifact_contract.md](artifact_contract.md) |
| Choose a public config or seed surface | [configuration.md](configuration.md) |
| Run validation, lint, tests, or simulator checks | [testing.md](testing.md) |
| Understand package layout and behavior contracts | [architecture.md](architecture.md) |
| Debug simulator/import/artifact failures | [troubleshooting.md](troubleshooting.md) |

## Ownership Map

| Topic | Source of truth |
| --- | --- |
| Operator commands and profiles | [thesis_workflow.md](thesis_workflow.md) |
| Evaluation semantics | [evaluation.md](evaluation.md) |
| Retained thesis evidence | [artifacts.md](artifacts.md) |
| Artifact layout contract | [artifact_contract.md](artifact_contract.md) |
| Config launch surface | [configuration.md](configuration.md) and [../configs/README.md](../configs/README.md) |
| Reproducibility rules | [reproducibility.md](reproducibility.md) |
| Simulator compatibility | [simulator_compatibility.md](simulator_compatibility.md) |
| Docs ownership and cleanup notes | [documentation_maintenance.md](documentation_maintenance.md) |

Local READMEs in `configs/`, `tests/`, and `runs/` should stay as
folder-specific pointers. Put concepts in the owner docs above, then link out
from local READMEs when needed.

## Verify

Full repo verifier:

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

Focused docs/config surface check:

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py
```
