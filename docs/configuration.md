# Configuration

The active config surface is intentionally small. Public thesis commands should
name one of the launch configs below; shared fragments and compatibility presets
exist to preserve provenance and tests.

## Public Launch Configs

| Config | Use |
| --- | --- |
| `configs/thesis/b1_noleague.yaml` | Canonical B1 no-league training. |
| `configs/thesis/main_league.yaml` | Canonical main league training. |
| `configs/thesis/main_league_auto_gpu.yaml` | Server-oriented main league lane. |
| `configs/thesis/final_eval.yaml` | Selected thesis final-eval contract. |
| `configs/thesis/final_eval_gpu.yaml` | GPU final-eval variant. |
| `configs/thesis/multideck_exploratory.yaml` | Explicitly labeled multideck exploration. |
| `configs/thesis/ablations/no_gru.yaml` | Public no-GRU IMPALA ablation. |
| `configs/thesis/ablations/ppo_lite.yaml` | Masked PPO-lite baseline. |
| `configs/thesis/ablations/terminal_only_reward.yaml` | Terminal-reward-only B1 reward ablation. |

`configs/thesis/base_fixed_deck_structured.yaml` and
`configs/thesis/_shared/` are implementation fragments. Do not present them as
launch targets unless the workflow explicitly names them.

## Compatibility Presets

| Preset | Use |
| --- | --- |
| `configs/presets/structured_acceptance_standard.yaml` | Structured acceptance compatibility. |
| `configs/presets/structured_acceptance_standard_auto_gpu.yaml` | Auto-GPU structured acceptance compatibility. |
| `configs/presets/structured_acceptance_standard_thesis_eval.yaml` | Legacy structured final-eval compatibility. |
| `configs/presets/structured_acceptance_standard_multideck.yaml` | Legacy multideck compatibility. |
| `configs/presets/typed_thesis_locked.yaml` | Locked typed thesis compatibility. |
| `configs/presets/typed_local.yaml` | Local typed diagnostic compatibility. |
| `configs/presets/typed_structured_v2.yaml` | Typed structured-v2 compatibility. |

## Seeds

Seed files under `configs/seeds/` define deterministic evaluation and promotion
surfaces. Treat them as reporting contracts, not casual tuning knobs.

## Overrides

Lower-level package entrypoints accept dotted config overrides for diagnostics:

```powershell
uv run python -m weiss_rl.training.train_entrypoint `
  --stack-config configs/thesis/main_league.yaml `
  --override training.optimizer.learning_rate=0.0001
```

Prefer `python -m weiss_rl.cli` for normal thesis runs. Use overrides only when
the changed behavior is named in the run label and recorded in the artifact
trail.
