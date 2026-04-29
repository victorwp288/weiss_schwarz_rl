#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl

ulimit -n 1048576
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

RUN_LABEL="${RUN_LABEL:-thesis_main_candidate_v12_b1init_constrained_cf81_20260429}"
MAX_UPDATES="${MAX_UPDATES:-360}"
STACK_CONFIG="${STACK_CONFIG:-configs/main_impala_league_server_v12_b1init_constrained.yaml}"
B1_DIR="runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_RESET_OPTIMIZER="${RESUME_RESET_OPTIMIZER:-0}"

EXTRA_TRAIN_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  EXTRA_TRAIN_ARGS+=(--train-arg=--resume-from --train-arg="$RESUME_FROM" --train-arg=--resume-allow-config-mismatch)
  if [[ "$RESUME_RESET_OPTIMIZER" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--train-arg=--resume-reset-optimizer)
  fi
fi

.venv/bin/python python/scripts/profile_train_job.py \
  --run-label "$RUN_LABEL" \
  --stack-config "$STACK_CONFIG" \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates "$MAX_UPDATES" \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30 \
  --train-arg=--b1-baseline-run-dir \
  --train-arg="$B1_DIR" \
  "${EXTRA_TRAIN_ARGS[@]}"
