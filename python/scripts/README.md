# Script Shims

`python/scripts/` contains compatibility shims. Package modules are canonical.

Preferred entrypoints:

```powershell
uv run python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run python -m weiss_rl.workflows.eval_entrypoint --stack-config configs/thesis/final_eval.yaml --run-dir runs/main_smoke
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

Compatibility presets:

- `configs/presets/structured_acceptance_standard.yaml`
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml`
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml`
- `configs/presets/structured_acceptance_standard_multideck.yaml`
