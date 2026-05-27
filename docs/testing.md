# Testing

The main local validation entrypoint is:

```powershell
uv run python python/scripts/verify_repo.py
```

It runs placeholder checks, ruff, format check, mypy on selected scripts, vulture, the full test suite, and wrapper dry-runs.

`verify_repo.py` does not currently enforce full-package mypy. The current full-package probe is:

```powershell
uv run mypy python/weiss_rl --show-error-codes --no-error-summary
```

As of the 2026-05-11 refactor log and the latest local baseline, this command is clean. Treat selected-script mypy as the CI-enforced gate and full-package mypy as a recommended local hardening check before broad refactor PRs.

## Focused Commands

```powershell
uv run python -m pytest -q python/weiss_rl/tests
uv run python -m ruff check python tests examples python/scripts
uv run python -m ruff format --check python tests examples python/scripts
uv run python -m mypy python/scripts/thesis_run.py python/scripts/eval.py python/scripts/play_vs_model.py
uv run mypy python/weiss_rl
```

## Simulator Contract

Use the simulator extra before simulator-boundary changes:

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_heuristic_public.py
```

## Artifact Contract

Use a fresh fixture directory:

```powershell
uv run python python/scripts/write_paper_readiness_fixture.py --run-dir runs/paper_readiness_fixture_local
uv run python python/scripts/paper_readiness_check.py --run-dir runs/paper_readiness_fixture_local
```

Do not use `runs/toy_public_demo_ci` as a paper-readiness fixture.

## Smoke And Determinism Probes

Scaffold-only training smoke:

```powershell
uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label refactor_smoke_local
```

Evaluation contract smoke without rollouts:

```powershell
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml
```

Public-safe repeated eval comparison:

```powershell
uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_standard.yaml --public-demo --run-label refactor_public_demo_det
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/refactor_public_demo_det --final-eval-dir runs/refactor_public_demo_det/eval/final_eval_a
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/refactor_public_demo_det --final-eval-dir runs/refactor_public_demo_det/eval/final_eval_b
```

Compare `summary.json` from both output directories after ignoring the expected `output_dir` field. This is a public-demo determinism smoke, not a replacement for canonical simulator-backed repeated final evaluation.
