# Server Scaling Watchouts

Date: 2026-04-24

Scope: first Linux/L40 preflight and early real multi-GPU learner runs for the B1 no-league anchor and later thesis runs.

## Current Best Local Context

- Best local B1 throughput candidate before server scaling:
  - `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke`
  - local-only, throughput-first evidence
  - tail throughput around `140702` samples/sec
  - max/final throughput around `142620` samples/sec
  - tail learner total around `304ms`
- Autoscale/local proof artifact:
  - `runs/autoscale_gpu1_smoke`
  - resolved `8 x 64 = 512` envs on mocked `gpu1`
  - native rollout remained active
  - `collector_actor_policy_forward_ms=0`
  - `timer_learner_gradient_sync_ms ~= 0.003` in single-rank mode
- Server dry-run targets:
  - `uc1-l40-3`: `3` GPUs, `24 x 64 = 1536` envs, DDP
  - `uc1-l40-4`: `4` GPUs, `32 x 64 = 2048` envs, DDP

## Most Likely Server Bottlenecks

### 1. Learner Compute Per Rank

Expected primary bottleneck if gradient sync is small.

Watch:

- `timer_learner_total_ms`
- `timer_learner_forward_time_major_ms`
- `timer_learner_trunk_ms`
- `timer_learner_packed_scorer_ms`
- `timer_learner_public_heuristic_target_ms`
- `timer_learner_backward_ms`
- `timer_learner_optimizer_ms`

Interpretation:

- If `timer_learner_gradient_sync_ms` is small and `timer_learner_total_ms` dominates, the server is scaling correctly but still learner-compute-bound.
- If packed scorer/public heuristic target remain large, next work is structured scorer/kernel/layout optimization, not actor tuning.
- If backward dominates, next work is gradient bucket/fusion, activation layout, or model compute profiling.

### 2. Gradient Synchronization

The current multi-rank path uses explicit all-reduce gradient averaging on the raw model, not full DDP bucket overlap. This is safer for the structured model interface, but may leave communication exposed.

Watch:

- `timer_learner_gradient_sync_ms`
- `distributed_global_samples_per_sec`
- `distributed_world_size`
- per-rank `timer_learner_total_ms`

Interpretation:

- If gradient sync is under roughly `5-10%` of learner total, keep the simple path.
- If gradient sync is `15-25%+`, prioritize fused/flattened gradient buckets or a careful DDP adapter.
- If gradient sync time grows superlinearly from 3 to 4 GPUs, inspect NCCL topology, PCIe/NVLink visibility, and rank device placement.

### 3. Host Batch Build And Transfer

Local batch construction improved, but at `1536-2048` global envs the host path can reappear.

Watch:

- `timer_runtime_build_learner_batch_ms`
- `timer_runtime_batch_concat_total_ms`
- `timer_runtime_legal_concatenation_ms`
- `timer_runtime_legal_concatenation_only_ms`
- learner idle metrics:
  - `learner_idle_wait_for_batch_ms`
  - `learner_idle_wait_for_prefetch_ms`

Interpretation:

- If build/concat grows with global env count per rank, the sharding is wrong or ranks are collecting more than their shard.
- If build is high but learner GPU is idle, consider direct actor-major learner consumption or shared-memory batch layout.
- If legal concat dominates, revisit packed legal transport/layout.

### 4. Collector Starvation Or Oversubscription

Expected server actor topology:

- `uc1-l40-3`: `24` actors, `64` envs each
- `uc1-l40-4`: `32` actors, `64` envs each

Watch:

- `timer_runtime_collector_queue_wait_ms`
- `timer_runtime_fill_pending_unrolls_ms`
- `timer_runtime_collect_update_batch_total_ms`
- `queue_occupancy_p50`
- `queue_occupancy_p90`
- `collector_collect_actor_unroll_ms`
- `collector_simulator_python_native_heuristic_rollout`

Interpretation:

- Low queue occupancy plus high learner wait means collectors cannot keep up.
- High collector unroll time but low learner time means CPU/simulator/native rollout is the frontier.
- If queue occupancy is high and learner is slow, adding actors will not help.
- If Linux CPU load is saturated, reduce actor count or reserve more learner/eval CPU cores.

### 5. Rank-0 Orchestration Pauses

Rank 0 owns artifacts, checkpoint aliases, TensorBoard, eval scheduling, and snapshot registry mutation.

Watch:

- periodic spikes around checkpoint/eval updates
- `snapshot_publish_latency_ms`
- `snapshot_apply_latency_ms`
- TensorBoard/log write stalls
- checkpoint save time if profiled externally

Interpretation:

- If all ranks pause around checkpoint updates, improve checkpoint cadence or async checkpointing.
- If rank 0 lags but nonzero ranks idle at barriers, rank-0 orchestration is too expensive.
- Promotion/eval should remain async and rank-0-only unless explicitly redesigned.

### 6. Eval / Promotion Overlap Policy

The default server training preset is intentionally learner-throughput-first:

- `evaluation.async_periodic_dev_eval_enabled: false`
- `evaluation.periodic_dev_eval_batched_inference_enabled: false`
- `evaluation.periodic_dev_eval_parallel_workers: 6`
- `league.promotion.gate.async_enabled: false`
- `league.promotion.gate.parallel_workers: 6`

Interpretation:

- Periodic dev-eval and promotion are canonical scalar surfaces by default.
- They do not overlap the learner in the default server preset, so throughput claims are easier to interpret.
- If later server profiling shows eval/promotion wall-clock is the bottleneck rather than learner throughput, enable async overlap deliberately and re-check rank-0 stalls.
- Batched dev-eval remains a non-authoritative diagnostic only; do not use it for promotion/checkpoint decisions or thesis comparisons without re-anchoring the whole eval surface.

## First Server Preflight Commands

Dry-run topology only:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml \
  --runtime-mode train_async_fast \
  --autoscale-dry-run \
  --hardware-profile uc1-l40-4 \
  --max-updates 1 \
  --unroll-length 16
```

Tiny 4-GPU smoke:

```bash
torchrun --standalone --nproc_per_node=4 python/scripts/train.py \
  --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml \
  --run-label server_l40_4_ddp_smoke \
  --runtime-mode train_async_fast \
  --autoscale \
  --hardware-profile uc1-l40-4 \
  --ddp \
  --ddp-backend nccl \
  --max-updates 2 \
  --unroll-length 16 \
  --profile-timers
```

Tiny 3-GPU smoke:

```bash
torchrun --standalone --nproc_per_node=3 python/scripts/train.py \
  --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml \
  --run-label server_l40_3_ddp_smoke \
  --runtime-mode train_async_fast \
  --autoscale \
  --hardware-profile uc1-l40-3 \
  --ddp \
  --ddp-backend nccl \
  --max-updates 2 \
  --unroll-length 16 \
  --profile-timers
```

## Must-Check Artifact Fields

Inspect these before trusting any throughput number:

- `run_summary.json`
  - `autoscale_topology.total_envs`
  - `autoscale_topology.actor_count`
  - `autoscale_topology.envs_per_actor`
  - `autoscale_topology.resolved_learner_parallelism`
  - `distributed.enabled`
  - `distributed.world_size`
- `environment.json`
  - `hardware.learner_device`
  - `hardware.actor_device_layout`
  - `hardware.actor_device_unique_count`
- `training/logs/scalars.jsonl`
  - `distributed_global_samples_per_sec`
  - `distributed_global_batch_env_steps`
  - `distributed_global_total_samples_processed`
  - `runtime_actor_count`
  - `runtime_envs_per_actor`

Hard fail / do not trust result if:

- `distributed.world_size` is `1` for a claimed multi-GPU run.
- `resolved_learner_parallelism` is not `ddp` on `uc1-l40-3` or `uc1-l40-4`.
- `runtime_actor_count` does not match the per-rank shard expectation.
- `collector_actor_policy_forward_ms` is nonzero on the native heuristic rollout B1 benchmark.
- artifacts are written by multiple ranks or checkpoint files are corrupted.

## Expected Healthy First-Smoke Pattern

For `uc1-l40-4`:

- run summary says:
  - global topology: `32 x 64 = 2048`
  - `world_size=4`
- each rank should effectively collect about:
  - `8 x 64 = 512` envs
- scalar logs should show:
  - `distributed_global_batch_env_steps` about global batch size
  - `distributed_global_samples_per_sec` much higher than single-rank local, though not yet thesis-grade
  - `timer_learner_gradient_sync_ms` visible but not dominant
  - queue wait not exploding

For `uc1-l40-3`:

- global topology: `24 x 64 = 1536`
- each rank should effectively collect about:
  - `8 x 64 = 512` envs

## Decision Rules

Promote the server scaling path if:

- dry-run topology matches expected L40 shape;
- 1-2 update DDP smoke completes;
- artifacts are rank-0 clean;
- gradients sync without NaNs or parameter divergence;
- `distributed_global_samples_per_sec` scales materially over single-GPU local/server baseline;
- learning-related losses/teacher metrics are finite and sane.

Stop and debug before longer training if:

- DDP launches but world size/artifacts are wrong;
- gradient sync is a large fraction of learner time;
- collector wait dominates despite high actor count;
- rank 0 blocks all ranks on checkpoint/eval;
- native rollout path is inactive;
- B1 benchmark quality probes regress after throughput changes.

## Next Engineering Moves If Bottleneck Appears

- Gradient sync bottleneck:
  - add flattened/fused gradient buckets;
  - or build a careful DDP adapter that preserves raw structured model methods.
- Learner scorer bottleneck:
  - optimize packed scorer kernels/layout;
  - reduce public heuristic target overhead;
  - profile row restriction and candidate chunk sizes on server.
- Batch-build bottleneck:
  - direct actor-major learner consumption;
  - shared-memory or pinned-memory batch handoff;
  - reduce repeated numpy concat/copy.
- Collector bottleneck:
  - tune actor count/envs per actor structurally;
  - inspect simulator-native rollout CPU time;
  - reserve CPU cores for learners/eval.
- Rank-0 orchestration bottleneck:
  - async checkpoint save;
  - less frequent snapshot/promotion work;
  - explicit barriers only where correctness requires them.
