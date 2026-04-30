# Experiment Taxonomy

## Official comparison rows

- `thesis_ablation_no_behavior_bc_b1recipe_20260430`: No behavior BC - auxiliary-loss ablation
- `thesis_ablation_no_reward_shaping_b1recipe_20260430`: No reward shaping - terminal-only reward ablation
- `thesis_ablation_norecurrence_b1recipe_20260430`: No recurrence B1 recipe - one-change architecture ablation
- `thesis_ablation_state_reward_knobs_b1recipe_20260430`: State reward knobs - reward shaping ablation
- `thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`: B1 GRU anchor - primary recurrent no-league anchor
- `thesis_baseline_ppo_b1recipe_1gpu_20260430`: PPO-lite B1 recipe - algorithm baseline
- `thesis_main_candidate_v14_b1init_anchor_stabilize_20260430`: v14 B1-init league - best pre-residual league candidate
- `thesis_main_candidate_v17e_b1_residual_guard_continue20_20260430`: v17e residual league - best constrained residual/league candidate

## Diagnostic/rescue runs retained for negative-results narrative

- `thesis_main_candidate_v10fresh_b1plus_cf81_20260429`: v10 fresh B1+CF - fresh league attempt
- `thesis_main_candidate_v13_b1init_long_variant_cf81_20260430`: v13 B1-init long variant - pre-v14 drift diagnostic
- `thesis_main_candidate_v15_frozen_trunk_b1kl_20260430`: v15 frozen trunk + B1 KL - failed rescue diagnostic
- `thesis_main_candidate_v16_b1_anchor_only_rows_20260430`: v16 B1 anchor-only rows - failed rescue diagnostic
- `thesis_main_candidate_v9c_anchor_push_relaxed_gate_delay_recent_cf81_20260429`: v9c delayed recent - promotion/drift diagnostic

## Backup/non-primary runs

- `thesis_ablation_heavy_reward_shaping_b1recipe_20260430`: Heavy reward shaping - backup shaping run without pass penalty

## Exploratory runs

All other `thesis_*` runs in `data/run_catalog.csv` are retained as exploratory audit trail and should not be presented as controlled ablations unless reviewed individually.
