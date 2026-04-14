# Config Layout

The repo now has one public config system: grouped presets.

## Main Presets

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
