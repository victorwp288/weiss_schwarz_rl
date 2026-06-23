# Contributing

This is a behavior-sensitive thesis research repository. Refactors are welcome, but public behavior must be preserved.

## Before Editing

1. Read the active task instructions, [docs/README.md](docs/README.md), and [docs/architecture.md](docs/architecture.md).
2. Identify the behavior-sensitive surface you are touching.
3. Add or identify characterization tests.
4. Avoid modifying historical run outputs, checkpoints, thesis figures, `run_logs/`, or `vast_artifacts/`.

## Validation

Start with:

```powershell
uv sync --extra dev
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

Run focused tests for the area you changed. For simulator-boundary work, also run simulator-extra tests.
See [docs/thesis_workflow.md](docs/thesis_workflow.md) for maintained validation commands.

## Pull Request Expectations

- Explain whether behavior changed. The default answer should be no.
- List validation commands and results.
- Note any artifacts created.
- Update `CHANGELOG.md` or the relevant owner doc when the public surface changes.
- Include migration notes for import path, CLI, config, or checkpoint changes.
