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

Each baseline extends the frozen thesis-model surface and overrides only the scientific difference that defines the baseline.

## Thesis Ablations

- `presets/ablations/structured_acceptance_thesis_model_no_tactical_bias_auto_gpu.yaml`
- `presets/ablations/structured_acceptance_thesis_model_no_b1_cutoff_auto_gpu.yaml`
- `presets/ablations/structured_acceptance_thesis_model_teacher_fade_auto_gpu.yaml`

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
uv run python python/scripts/train.py \
  --stack-config configs/presets/typed_local.yaml \
  --override training.optimizer.learning_rate=0.0002 \
  --override rewards.truncation.reward=-0.05 \
  --override league.warmup.first_updates=25000
```

## Path Convention

All file paths inside preset YAML are repo-root relative.
