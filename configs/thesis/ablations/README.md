# Thesis Ablations

Canonical thesis ablations:

- `no_gru.yaml`: public no-GRU IMPALA ablation. This is an alias for the
  older `norecurrence_impala.yaml` name so historical artifacts keep their
  original config path.
- `ppo_lite.yaml`: masked PPO-lite baseline.
- `terminal_only_reward.yaml`: terminal-reward-only B1 reward ablation. Older
  May 13 reward-shaping probes live in
  `../../archive/thesis_reward_ablations_20260513/`.

`norecurrence_impala.yaml` remains as a legacy compatibility name for historical
artifact reproduction. Do not present it as an additional public ablation; use
`no_gru.yaml` in docs and thesis-facing commands.
