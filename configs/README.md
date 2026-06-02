# Configs

This directory is intentionally small. Public thesis configs live under
`configs/thesis/`; compatibility wrapper presets live under `configs/presets/`.
Descriptions and launch context live in [../docs/configuration.md](../docs/configuration.md).

## Thesis

- `thesis/b1_noleague.yaml`
- `thesis/main_league.yaml`
- `thesis/main_league_auto_gpu.yaml`
- `thesis/final_eval.yaml`
- `thesis/final_eval_gpu.yaml`
- `thesis/multideck_exploratory.yaml`
- `thesis/ablations/no_gru.yaml`
- `thesis/ablations/ppo_lite.yaml`
- `thesis/ablations/terminal_only_reward.yaml`

`thesis/_shared/` contains only fragments needed by those public configs and by
the retained main-model provenance configs. `thesis/base_fixed_deck_structured.yaml`
is also a shared base, not a normal launch target.

## Compatibility Presets

- `presets/structured_acceptance_standard.yaml`
- `presets/structured_acceptance_standard_auto_gpu.yaml`
- `presets/structured_acceptance_standard_thesis_eval.yaml`
- `presets/structured_acceptance_standard_multideck.yaml`
- `presets/typed_thesis_locked.yaml`
- `presets/typed_local.yaml`
- `presets/typed_structured_v2.yaml`

## Studies

- `study/metagame_sensitivity.yaml`

## Seeds

- `seeds/dev_eval_seeds.txt`
- `seeds/local_dev_eval_seeds.txt`
- `seeds/local_promotion_eval_seeds.txt`
- `seeds/promotion_eval_seeds.txt`
- `seeds/report_eval_seeds.txt`
