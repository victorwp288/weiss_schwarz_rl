# Important Model Scores And Artifact Paths

This is the compact run log for final thesis models, official ablations, and the two appendix old-candidate baselines.

| status | label | mean5 | key mean | min key | B1 | B3 | B4 | checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| official | B1 GRU anchor | 0.8766 | 0.7943 | 0.4922 | 0.4922 | 0.8984 | 0.9922 | `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt` |
| official | No recurrence B1 recipe | 0.8836 | 0.8060 | 0.4961 | 0.4961 | 0.9219 | 1.0000 | `runs/thesis_ablation_norecurrence_b1recipe_20260430/training/checkpoints/checkpoint_160.pt` |
| official | v14 B1-init league | 0.8625 | 0.7708 | 0.5039 | 0.5039 | 0.8164 | 0.9922 | `runs/thesis_main_candidate_v14_b1init_anchor_stabilize_20260430/training/checkpoints/checkpoint_160.pt` |
| official | v17e residual league | 0.8688 | 0.7812 | 0.5000 | 0.5000 | 0.8516 | 0.9922 | `runs/thesis_main_candidate_v17e_b1_residual_guard_continue20_20260430/training/checkpoints/checkpoint_80.pt` |
| official | State reward knobs | 0.8695 | 0.7826 | 0.5039 | 0.5039 | 0.8516 | 0.9922 | `runs/thesis_ablation_state_reward_knobs_b1recipe_20260430/training/checkpoints/checkpoint_160.pt` |
| official | No reward shaping | 0.8711 | 0.7852 | 0.5078 | 0.5078 | 0.8555 | 0.9922 | `runs/thesis_ablation_no_reward_shaping_b1recipe_20260430/training/checkpoints/checkpoint_160.pt` |
| official | No behavior BC | 0.8695 | 0.7826 | 0.5117 | 0.5117 | 0.8477 | 0.9883 | `runs/thesis_ablation_no_behavior_bc_b1recipe_20260430/training/checkpoints/checkpoint_160.pt` |
| official | PPO-lite B1 recipe | 0.7336 | 0.5560 | 0.2266 | 0.2266 | 0.5039 | 0.9375 | `runs/thesis_baseline_ppo_b1recipe_1gpu_20260430/training/checkpoints/checkpoint_160.pt` |
| appendix_old_candidate | Old v2 B1-specialist | 0.9195 | 0.8659 | 0.6328 | 0.9688 | 0.6328 | 0.9961 | `runs/thesis_b1_candidate_v2_20260429/training/checkpoints/checkpoint_40.pt` |
| appendix_old_candidate | Old v3b profile-cycle | 0.9031 | 0.8385 | 0.7227 | 0.7969 | 0.7227 | 0.9961 | `runs/thesis_b1_candidate_v3b_profiles_cycle_noguard_20260429/training/checkpoints/checkpoint_40.pt` |
