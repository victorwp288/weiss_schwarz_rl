# Server Multi-GPU Readiness Progress

Date: 2026-04-29

## Scope

Prepared the training stack for moving to a school multi-GPU server. The focus was DDP launch safety, rank-to-device placement, autoscale validation, and a local CPU/Gloo smoke path where possible.

## Completed Fixes

- Added backend-aware DDP scalar reductions so NCCL all-reduces use CUDA tensors instead of CPU tensors.
- Added rank-local DDP learner device resolution. Bare `cuda` and `auto` now map to `cuda:LOCAL_RANK`; mismatched fixed CUDA indices fail early.
- Added autoscale validation requiring DDP `world_size` to match the resolved learner GPU count.
- Made DDP `cuda:auto` actor placement rank-local, preventing rank 0 actors from using rank 1 learner GPUs and vice versa.
- Made DDP stop/rollback/refresh decisions collective so one rank does not leave the others stuck in the next all-reduce.
- Added PPO-lite gradient synchronization so PPO DDP does not train independent per-rank learners.
- Normalized DDP `cuda:auto` and selected Gloo for explicit CPU DDP when backend is `auto`.
- Raised the default DDP process-group timeout and exposed `--ddp-timeout-seconds` for long rank-0 eval/checkpoint gates.
- Made local hardware detection use the smallest visible GPU VRAM and label mixed visible GPU sets conservatively.
- Made gradient averaging rank-invariant when a trainable parameter has `grad=None` on one rank, avoiding mismatched NCCL collective sequences.
- Made explicit autoscale GPU counts fail if they exceed visible CUDA devices instead of silently clamping.
- Enforced `max_actor_process_count` for manual actor topologies too.
- Made DDP reject indexed learner overrides like `--device cuda:0` before process-group setup so every rank exits consistently.
- Kept DDP async eval/promotion work on rank 0's learner GPU instead of stealing learner GPUs from ranks 1+.
- Made DDP manifest actor-device layout represent the global rank-local plan instead of reporting all actors as rank 0 local CUDA.
- Prevented non-rank0 DDP opponent-pool refreshes from writing shared registry state during collective refresh.
- Added `thesis_run.py` wrapper support for `--torchrun-nproc`, `--autoscale`, `--autoscale-dry-run`, `--hardware-profile`, `--ddp`, and `--ddp-backend`.
- Made server training wrapper presets apply fast autoscale defaults unless the caller explicitly overrides them.
- Added `torchrun`/autoscale pass-through to `profile_train_job.py` and `scripts/run_thesis_queue.sh` (`TORCHRUN_NPROC=4` on the server).
- Updated the thesis recipe with the direct `torchrun` server command and wrapper equivalent.

## Verification

- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_autoscale.py`
- `uv run --extra dev python -m ruff check python/weiss_rl/distributed.py python/weiss_rl/autoscale.py python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_autoscale.py python/scripts/train.py`
- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_ppo_lite_learner.py python/weiss_rl/tests/test_autoscale.py python/weiss_rl/tests/test_runtime.py::test_resolve_actor_device_layout_can_stay_rank_local_for_ddp_cuda_auto python/weiss_rl/tests/test_thesis_run_wrapper.py`
- `uv run --extra dev python -m pytest -q python/weiss_rl/tests`
- `uv run --extra dev python -m ruff check python/scripts python/weiss_rl`
- `uv run --extra dev python -m compileall -q python`
- One-rank DDP/Gloo local smoke passed:
  `uv run --extra dev --extra sim python python/scripts/train.py --stack-config configs/baselines/noleague_impala.yaml --run-label ddp_single_gloo_smoke_20260429_v2 --num-envs 2 --unroll-length 4 --max-updates 1 --runtime-mode train_ordered --device cpu --ddp --ddp-backend gloo --override 'system.collection_backend="auto"'`

## Local Blocker

Two-rank `torchrun` on this Windows PyTorch wheel failed before repo code ran because `TCPStore` requested libuv even though this wheel lacks libuv support. The school Linux GPU box should use the direct `torchrun` path with NCCL.

## Next Server Smoke

On the server, start with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run --standalone --nproc_per_node=4 \
  python/scripts/train.py \
  --stack-config configs/main_impala_league_server.yaml \
  --autoscale \
  --hardware-profile local \
  --ddp \
  --ddp-backend nccl \
  --ddp-timeout-seconds 1800 \
  --run-label server_ddp_smoke \
  --b1-baseline-run-dir runs/b1_anchor_thesis_model_seed1 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 2
```
