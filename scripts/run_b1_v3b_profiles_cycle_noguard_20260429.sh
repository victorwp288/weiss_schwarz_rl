#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v3b_profiles_cycle_noguard_20260429 \
  --stack-config configs/baselines/noleague_impala.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 400 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30 \
  --override 'training.heuristic_native_rollout_profiles=["base","aggressive","control"]' \
  --override 'training.heuristic_native_rollout_profile_mode="cycle"' \
  --override 'curriculum.checkpoint_guard.enabled=false'
