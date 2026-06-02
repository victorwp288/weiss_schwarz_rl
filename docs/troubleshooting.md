# Troubleshooting

## Simulator Import

Use:

```powershell
uv sync --extra dev --extra sim
```

If using a sibling simulator checkout, ensure it is on `PYTHONPATH` before
running simulator-bound tests.

## Verification

Run the verifier first:

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

For focused failures, run the printed command directly from the repo root.

## Artifacts

Do not edit retained artifacts under `runs/`, `diagnostics/`,
`vast_artifacts/`, or `thesis_figures_final/` unless replacing a thesis artifact
on purpose.
