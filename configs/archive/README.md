# Config Archive

Historical configs live here when they are useful for provenance but are not
part of the public thesis workflow.

Use `configs/thesis/` for thesis-facing work and `configs/presets/` only for
compatibility presets that are still referenced by docs, tests, or wrapper
commands.

## Contents

- `presets_20260506/`: unreferenced May 6 experiment, rescue, eval, no-GRU, and
  PPO-lite preset variants moved out of `configs/presets/`.
- `thesis_reward_ablations_20260513/`: May 13 B1 reward-shaping probes moved
  out of `configs/thesis/ablations/`. They still load for reproduction and
  characterization tests, but the public thesis ablation surface keeps only the
  canonical terminal-only reward variant.
