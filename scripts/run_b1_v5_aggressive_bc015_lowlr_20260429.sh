#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429 \
  --stack-config configs/baselines/noleague_impala.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 160 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30 \
  --override 'training.heuristic_native_rollout_profile="aggressive"' \
  --override 'training.heuristic_native_rollout_profiles=[]' \
  --override 'training.heuristic_native_rollout_profile_mode="fixed"' \
  --override 'training.structured_aux.teacher_public_heuristic_profiles=["aggressive"]' \
  --override 'training.structured_aux.teacher_public_heuristic_profile_mode="cycle"' \
  --override 'training.structured_aux.teacher_public_heuristic_label_profile="aggressive"' \
  --override 'training.structured_aux.teacher_public_heuristic_temperature=8.0' \
  --override 'training.structured_aux.teacher_public_main_move_coef=0.1' \
  --override 'training.behavior_action_bc_coef=0.15' \
  --override 'training.optimizer.learning_rate=0.00005' \
  --override 'training.exploration.entropy_coef=0.015' \
  --override 'training.exploration.entropy_anneal_to=0.01' \
  --override 'curriculum.checkpoint_guard.enabled=false'
