# Configuration

The active config surface is small by design.

## Public Thesis Configs

- `configs/thesis/b1_noleague.yaml`
- `configs/thesis/main_league.yaml`
- `configs/thesis/main_league_auto_gpu.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/final_eval_gpu.yaml`
- `configs/thesis/multideck_exploratory.yaml`
- `configs/thesis/ablations/no_gru.yaml`
- `configs/thesis/ablations/ppo_lite.yaml`
- `configs/thesis/ablations/terminal_only_reward.yaml`

## Compatibility Presets

- `configs/presets/structured_acceptance_standard.yaml`
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml`
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml`
- `configs/presets/structured_acceptance_standard_multideck.yaml`
- `configs/presets/typed_thesis_locked.yaml`
- `configs/presets/typed_local.yaml`
- `configs/presets/typed_structured_v2.yaml`

## Seeds

Seed files under `configs/seeds/` define deterministic evaluation and promotion
surfaces. Do not casually edit them.

## Overrides

Lower-level script entrypoints still accept grouped dotted overrides:

```powershell
uv run python python/scripts/train.py `
  --stack-config configs/thesis/main_league.yaml `
  --override training.optimizer.learning_rate=0.0001
```
