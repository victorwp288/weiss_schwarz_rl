# Testing

Run commands from the repository root.

## Full Verifier

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

The verifier runs repo hygiene checks, placeholder scans, Ruff, format checks,
configured mypy targets, Vulture, pytest, and wrapper dry-runs. It does not
provide a `--help` shortcut; running the module starts verification.

## Focused Checks

| Surface | Command |
| --- | --- |
| Python tests | `uv run python -m pytest -q python/weiss_rl/tests` |
| Lint | `uv run python -m ruff check python tests examples` |
| Format | `uv run python -m ruff format --check python tests examples` |
| Type checks | `uv run python -m mypy python/weiss_rl/workflows/thesis_wrapper.py python/weiss_rl/workflows/eval_entrypoint.py python/weiss_rl/human_play/play_vs_model_entrypoint.py` |

Use focused pytest files for the package you changed, then run the verifier
before treating a public-surface refactor as done.

## Simulator Boundary

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_heuristic_public.py
```

Run these after touching simulator import paths, env wrappers, legal-action
packing, deck presets, or evaluation policy resolution.

## Docs And Config Surface

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py
```

This check verifies public Markdown links and prevents docs from advertising
noncanonical thesis ablations as additional public surfaces.
