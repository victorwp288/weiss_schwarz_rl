# Config Layout

The repo now has one public config system: grouped presets.

## Thesis Configs

Use these for standard work:

- `thesis/b1_noleague.yaml`
- `thesis/main_league.yaml`
- `thesis/main_league_auto_gpu.yaml`
- `thesis/final_eval.yaml`
- `thesis/multideck_exploratory.yaml`
- `thesis/ablations/norecurrence_impala.yaml`
- `thesis/ablations/ppo_lite.yaml`

These names are the public thesis surface. They may extend older presets
internally, but operators should not need to know those chains.
The standard B1 and main league training configs use the medium64 structured
model surface (`gru_hidden_size: 64`, `encoder_mlp_width: 64`,
`typed_feature_width: 16`) so the thesis runs are not locked to the earlier
tiny32 probe model.

## Compatibility Presets

Current compatibility presets:

- `presets/structured_acceptance_standard.yaml`
  - Canonical current training recipe.
- `presets/structured_acceptance_standard_auto_gpu.yaml`
  - Canonical Linux server variant with automatic multi-GPU actor sharding.
- `presets/structured_acceptance_standard_thesis_eval.yaml`
  - Richer final-eval companion.
- `presets/structured_acceptance_standard_multideck.yaml`
  - Deck-diversity/generalization variant.

Legacy typed presets remain available for compatibility and lower-level work:

- `presets/typed_thesis_locked.yaml`
  - Thesis-safe default.
  - Typed GRU model.
  - Conservative reward, league, and evaluation settings.
- `presets/typed_local.yaml`
  - Everyday local default.
  - Typed GRU model.
  - More exploration, anti-stall shaping, and shorter league warmup.

## Baselines

- `presets/baselines/noleague_impala.yaml`
- `presets/baselines/norecurrence_impala.yaml`
- `presets/baselines/ppo_lite.yaml`

Each baseline extends the thesis-safe typed preset and overrides only the scientific difference that defines the baseline.

## Ablations

- `presets/ablations/discount_gamma099.yaml`
- `presets/ablations/reward_shaping.yaml`

## Study Config

- `study/metagame_sensitivity.yaml`

This file is study-only. It is used by metagame/sensitivity reporting and is not part of the live training preset surface.

## Seeds

- `seeds/dev_eval_seeds.txt`
- `seeds/promotion_eval_seeds.txt`
- `seeds/report_eval_seeds.txt`

Seed files contain one unsigned 64-bit integer per line with no comments.

## Smoke Preset

`stack_smoke.yaml` is still available for scaffold checks. It is intentionally tiny and only exists to verify config loading, simulator provenance capture, and manifest writing.

## Override Style

CLI overrides now follow grouped paths, for example:

```bash
python python/scripts/train.py \
  --stack-config configs/presets/typed_local.yaml \
  --override training.optimizer.learning_rate=0.0002 \
  --override rewards.truncation.reward=-0.05 \
  --override league.warmup.first_updates=25000
```

## Path Convention

All file paths inside preset YAML are repo-root relative.
