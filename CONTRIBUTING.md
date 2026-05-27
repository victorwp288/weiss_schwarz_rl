# Contributing

This is a behavior-sensitive thesis research repository. Refactors are welcome, but public behavior must be preserved.

## Before Editing

1. Read `AGENTS.md`, `REFACTOR_PLAN.md`, and `docs/refactor_log.md`.
2. Identify the behavior-sensitive surface you are touching.
3. Add or identify characterization tests.
4. Avoid modifying historical run outputs, checkpoints, thesis figures, `run_logs/`, or `vast_artifacts/`.

## Validation

Start with:

```powershell
uv sync --extra dev
uv run python python/scripts/verify_repo.py
```

Run focused tests for the area you changed. For simulator-boundary work, also run simulator-extra tests.

## Pull Request Expectations

- Explain whether behavior changed. The default answer should be no.
- List validation commands and results.
- Note any artifacts created.
- Update `docs/refactor_log.md`.
- Include migration notes for import path, CLI, config, or checkpoint changes.
