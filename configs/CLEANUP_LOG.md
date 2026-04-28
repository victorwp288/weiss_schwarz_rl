# Config Cleanup Log

Date: 2026-04-28

## Current Public Surface

- Main: `configs/main_impala_league_server.yaml`, `configs/main_eval.yaml`
- Residual: `configs/residual_league_s1_server.yaml`, `configs/residual_eval_s1.yaml`
- Baselines: `configs/baselines/noleague_impala.yaml`, `configs/baselines/noleague_benchmark*.yaml`, `configs/baselines/noleague_fullsize_*.yaml`, `configs/baselines/norecurrence_impala.yaml`, `configs/baselines/norecurrence_noleague.yaml`, `configs/baselines/ppo_lite.yaml`, and variant no-league controls.
- Ablations: `configs/ablations/*.yaml`
- Dev/smoke/study: `configs/local.yaml`, `configs/thesis_locked.yaml`, `configs/stack_smoke.yaml`, `configs/study/metagame_sensitivity.yaml`

## Completed

- Moved the old long preset tree to `configs/archive/presets/`.
- Replaced the former active-alias directory with compact root, baseline, and ablation aliases.
- Updated wrapper/script/doc/test references away from the old public paths.
- Added a legacy loader fallback so old preset-tree command paths resolve through the archive.
- Flattened main, residual, baseline, and ablation configs so they no longer extend archive files.
- Restored compact public surfaces for benchmark, no-recurrence no-league, and ablation-control commands so legacy wrappers keep their experiment semantics.

## Verified

- `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_thesis_run_wrapper.py python/weiss_rl/tests/test_sweeps.py python/weiss_rl/tests/test_config.py python/weiss_rl/tests/test_pool_factory.py`
- `uv run python python/scripts/thesis_run.py --list-presets`
- `uv run python python/scripts/thesis_run.py --preset thesis-model-auto-gpu --run-label config_cleanup_dry_run --dry-run --skip-compare`
- `uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label config_cleanup_stack_smoke`
- Legacy fallback for the former typed-local preset path

## Next Checks

- Keep new experiment commands on the compact paths.
- Use `configs/archive/presets/` only for reproducing historical runs.
- Keep public launch configs self-contained except small dev inheritance such as `local.yaml` extending `thesis_locked.yaml`.
