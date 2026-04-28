# Config Layout

The public launch surface is intentionally small. Use these files for new runs:

## Main

- `main_impala_league_server.yaml`
- `main_eval.yaml`
- `residual_league_s1_server.yaml`
- `residual_eval_s1.yaml`

## Baselines

- `baselines/noleague_impala.yaml`
- `baselines/noleague_benchmark.yaml`
- `baselines/noleague_benchmark_warmup.yaml`
- `baselines/noleague_benchmark_lowlr_continuation.yaml`
- `baselines/noleague_benchmark_eval.yaml`
- `baselines/noleague_fullsize_warmup.yaml`
- `baselines/noleague_fullsize_lowlr_continuation.yaml`
- `baselines/norecurrence_impala.yaml`
- `baselines/norecurrence_noleague.yaml`
- `baselines/ppo_lite.yaml`
- `baselines/no_tactical_bias_noleague.yaml`
- `baselines/teacher_fade_noleague.yaml`
- `baselines/multideck_noleague.yaml`
- `baselines/reward_shaping_noleague.yaml`

## Ablations

- `ablations/reward_shaping.yaml`
- `ablations/no_tactical_bias.yaml`
- `ablations/teacher_fade.yaml`
- `ablations/no_b1_cutoff.yaml`
- `ablations/multideck.yaml`

Evaluation companion aliases live beside the matching ablation when the wrapper
needs an explicit eval surface.

## Dev And Compatibility

- `local.yaml`
- `thesis_locked.yaml`
- `structured_v2.yaml`
- `typed_structured_v2.yaml`
- `stack_smoke.yaml`
- `study/metagame_sensitivity.yaml`

Main, residual, baseline, and ablation configs are self-contained launch files.
The only compact inheritance kept in the public surface is for local/dev variants,
where `local.yaml` extends `thesis_locked.yaml` and structured dev configs extend
`local.yaml`.

Historical experiment configs were moved intact to `archive/presets/`. They are
kept for reproducibility, not as a launch menu or normal dependency.
