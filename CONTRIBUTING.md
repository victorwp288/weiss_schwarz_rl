# Contributing

This is a behavior-sensitive thesis research repository. Refactors are welcome, but public behavior must be preserved.

## Before Editing

1. Read the relevant workflow and contract docs under `docs/`.
2. Identify the behavior-sensitive surface you are touching.
3. Add or identify characterization tests.
4. Avoid modifying historical run outputs, checkpoints, and thesis figures unless the change is explicitly part of an artifact publication or cleanup.

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
- Include a concise change note when behavior, commands, config, or artifact layout changes.
- Include migration notes for import path, CLI, config, or checkpoint changes.
