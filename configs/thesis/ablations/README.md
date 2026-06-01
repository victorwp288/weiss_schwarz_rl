# Thesis Ablations

Canonical thesis ablations:

- `no_gru.yaml`: public no-GRU IMPALA ablation. This is an alias for the
  older `norecurrence_impala.yaml` name so historical artifacts keep their
  original config path.
- `ppo_lite.yaml`: masked PPO-lite baseline.
- `terminal_only_reward.yaml`: terminal-reward-only B1 reward ablation. Older
  May 13 reward-shaping probes live in
  `../../archive/thesis_reward_ablations_20260513/`.

Most other files in this directory are dated probes, rescue configs, or
targeted-confirm investigation configs. Keep them for artifact reproduction,
but do not present them as the standard thesis config surface.
