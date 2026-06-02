# Testing

Main verifier:

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

Focused checks:

```powershell
uv run python -m pytest -q python/weiss_rl/tests
uv run python -m ruff check python tests examples python/scripts
uv run python -m ruff format --check python tests examples python/scripts
uv run python -m mypy python/weiss_rl/workflows/thesis_wrapper.py python/weiss_rl/workflows/eval_entrypoint.py python/weiss_rl/human_play/play_vs_model_entrypoint.py
```

Simulator boundary checks:

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_heuristic_public.py
```
