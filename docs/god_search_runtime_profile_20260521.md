# God-Search Runtime Profile - 2026-05-21

## Purpose

Before spending confirm256 time on the K4 god-search player, we profiled whether
GPU eval should speed it up.

Result: **do not switch K4 same-world search validation to GPU as-is.** The
current implementation performs many small, sequential model calls inside
simulator rollouts. Moving those tiny forwards to CUDA made the profile slower,
not faster.

## Profile Surface

Mini-surface:

- focal: `main_league_selected`
- base run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- opponents:
  - `B1 NoLeague baseline`
  - `seed_c3aac2f9dc_policy_000004`
- paired seeds:
  - cProfile: `4`
  - wall-time check: `2`
- search:
  - `same_world_prefix_rollout`
  - `top_k = 4`
  - `rollouts_per_action = 1`
  - terminal rollouts
  - `max_search_decisions_per_game = 1`
  - rollout policy `argmax`

## Config Artifact

Added GPU overlay:

```text
configs/thesis/final_eval_gpu.yaml
```

It extends `configs/thesis/final_eval.yaml` and only changes:

```yaml
evaluation:
  eval_device: cuda:auto
```

The loaded stack preserves `model_argmax_pinned_v1` and the thesis final policy
selection contract.

## Hardware Check

```text
cuda_available True
cuda_device_count 1
cuda_device_name NVIDIA GeForce RTX 5080
```

`nvidia-smi` before profiling showed the GPU available with low utilization.

## Runtime Results

### cProfile, paired-4 mini-surface

CPU:

```text
diagnostics/god_search_profile_cpu_b1_p0004_p4_k4_20260521.prof
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_profile_cpu_b1_p0004_p4_k4_20260521/targeted_confirm4_summary.json
elapsed_seconds = 74.955
overall = 14/16
terminal_rollouts = 64
prefix_replay_failures = 0
```

GPU:

```text
diagnostics/god_search_profile_gpu_b1_p0004_p4_k4_20260521.prof
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_profile_gpu_b1_p0004_p4_k4_20260521/targeted_confirm4_summary.json
elapsed_seconds = 206.011
overall = 14/16
terminal_rollouts = 64
prefix_replay_failures = 0
```

GPU was about `2.75x` slower on the cProfile mini-surface.

### Non-cProfile wall-time check, paired-2 mini-surface

CPU:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_profile_cpu_b1_p0004_p2_k4_wall_20260521/targeted_confirm2_summary.json
elapsed_seconds = 34.904
outer measured wall time = 35.756
overall = 7/8
terminal_rollouts = 32
prefix_replay_failures = 0
```

GPU:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_profile_gpu_b1_p0004_p2_k4_wall_20260521/targeted_confirm2_summary.json
elapsed_seconds = 96.747
outer measured wall time = 97.648
overall = 7/8
terminal_rollouts = 32
prefix_replay_failures = 0
```

GPU was about `2.73x` slower in the non-cProfile wall-time check.

## Bottleneck Read

CPU cProfile top cumulative regions:

- total profiled runtime: `75.022s`
- `final_eval._run_matchup`: `58.822s`
- `SimulatorEvalRunner.run_game`: `58.625s`
- `SimulatorEvalRunner._select_action`: `57.385s`
- `SimulatorEvalRunner._model_logits_for_eval`: `50.230s`
- `model_eval_logits_for_legal_ids`: `48.219s`
- `factorized_packed_action_log_probs_seat_aware`: `47.236s`
- god-search root/branch selection:
  - `_select_action_with_god_search`: `46.732s`
  - `_run_same_world_prefix_rollout`: `46.284s`
- rollout policy model calls:
  - `_select_action_without_god_search`: `41.557s`

GPU profile showed the same call structure but slower model-eval cumulative
time:

- `SimulatorEvalRunner._model_logits_for_eval`: `178.985s`
- `model_eval_logits_for_legal_ids`: `176.392s`
- `_run_same_world_prefix_rollout`: `151.010s`

Interpretation:

- K4 search performs many tiny sequential model evals during branch rollouts.
- CUDA launch/transfer/synchronization overhead dominates those tiny forwards.
- The current GPU path is not a speedup unless the rollout model calls are
  batched or otherwise restructured.
- The simulator/prefix replay still matters, but the main visible Python
  cumulative bottleneck is repeated model evaluation inside rollouts.

## Speed Recommendations

Priority order:

1. Keep confirm128/confirm256 on CPU for the current K4 implementation.
2. If optimizing runtime, batch branch rollout model calls before trying GPU
   again. The current one-state-at-a-time GPU path is the wrong shape.
3. Add a search-specific inference cache for repeated prefix/root states only if
   diagnostics show repeated identical model surfaces. This is likely smaller
   than batching but cheaper to implement.
4. Test bounded-horizon search, for example `max_rollout_decisions = 40` or
   `80`, only as a speed/strength tradeoff. Do not replace terminal K4 unless a
   paired screen shows it preserves most gains.
5. Consider parallel workers only for exploratory throughput. Keep single-worker
   confirm evidence for selection unless the eval determinism contract is
   explicitly validated under parallel workers.

## Strength Recommendations

K4/R1 terminal/search1 is the current best search recipe because it passed
confirm64 and beat K3 by `+42` paired wins with no row regressions.

Only two last-shot strength probes are worth considering before confirm256:

1. `K5/R1/terminal/search1` on the same full13 paired-4 screen.
   Stop unless it beats K4 by at least `+3` total with no fixed row giveback.
2. `K4/R2/terminal/search1` on the same full13 paired-4 screen.
   This is more expensive and should only continue if it shows a clear learned
   gain without fixed regression.

Do not train longer before validating K4. The search layer is producing clean
paired gains without weight drift, while training has repeatedly created
fixed/learned interference.
