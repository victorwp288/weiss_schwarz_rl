# Config Layout

The repo now has one public config system: grouped presets.

## Thesis Configs

Use these for standard work:

- `thesis/b1_noleague.yaml`
- `thesis/main_league.yaml`
- `thesis/main_league_auto_gpu.yaml`
- `thesis/final_eval.yaml`
- `thesis/multideck_exploratory.yaml`
- `thesis/ablations/no_gru.yaml`
- `thesis/ablations/ppo_lite.yaml`
- `thesis/ablations/terminal_only_reward.yaml`

These names are the public thesis surface. They may extend older presets
internally, but operators should not need to know those chains.
Internal fragments shared by thesis configs live under `thesis/_shared/`; they
are not launch targets.
The standard B1 and main league training configs use the medium64 structured
model surface (`gru_hidden_size: 64`, `encoder_mlp_width: 64`,
`typed_feature_width: 16`) so the thesis runs are not locked to the earlier
tiny32 probe model.

## Compatibility Presets

Current compatibility presets:

- `presets/structured_acceptance_standard.yaml`
  - Compatibility preset backing the fixed-deck main thesis lane.
- `presets/structured_acceptance_standard_auto_gpu.yaml`
  - Compatibility preset backing the server-oriented main thesis lane.
- `presets/structured_acceptance_standard_thesis_eval.yaml`
  - Compatibility preset backing the richer final-eval companion.
- `presets/structured_acceptance_standard_multideck.yaml`
  - Compatibility preset backing the exploratory deck-diversity/generalization variant.

Legacy typed presets remain available for compatibility and lower-level work:

- `presets/typed_thesis_locked.yaml`
  - Thesis-safe default.
  - Typed GRU model.
  - Conservative reward, league, and evaluation settings.
- `presets/typed_local.yaml`
  - Everyday local default.
  - Typed GRU model.
  - More exploration, anti-stall shaping, and shorter league warmup.

Historical dated probes that are no longer referenced by docs, tests, or public
workflows live under `archive/`. They are retained for provenance, not for
normal thesis operation.

## Baselines

- `presets/baselines/noleague_impala.yaml`
- `presets/baselines/norecurrence_impala.yaml`
- `presets/baselines/ppo_lite.yaml`

Each baseline extends the thesis-safe typed preset and overrides only the scientific difference that defines the baseline.

## Ablations

- `thesis/ablations/README.md` lists the canonical thesis ablation surface.
- `thesis/ablations/norecurrence_impala.yaml` is the historical filename behind
  the public `thesis/ablations/no_gru.yaml` alias.
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

Lower-level direct script overrides follow grouped paths, for example:

```bash
python python/scripts/train.py \
  --stack-config configs/thesis/main_league.yaml \
  --override training.optimizer.learning_rate=0.0002 \
  --override rewards.truncation.reward=-0.05 \
  --override league.warmup.first_updates=25000
```

## Path Convention

All file paths inside preset YAML are repo-root relative.
