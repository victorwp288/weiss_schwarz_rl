#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="plan"
PHASE="all"

SEED="${SEED:-20260421}"
NUM_ENVS="${NUM_ENVS:-2048}"
UNROLL_LENGTH="${UNROLL_LENGTH:-64}"
RUNTIME_MODE="${RUNTIME_MODE:-train_async_fast}"
SWEEP_UPDATES="${SWEEP_UPDATES:-10}"
SEED_SLUG="${SEED//[^[:alnum:]_-]/_}"
SEED_TAG="seed${SEED_SLUG}"

B1_BENCHMARK_MINUTES="${B1_BENCHMARK_MINUTES:-10}"
B1_BENCHMARK_MAX_UPDATES="${B1_BENCHMARK_MAX_UPDATES:-1000000}"
B1_BENCHMARK_NUM_ENVS="${B1_BENCHMARK_NUM_ENVS:-${NUM_ENVS}}"
B1_BENCHMARK_UNROLL_LENGTH="${B1_BENCHMARK_UNROLL_LENGTH:-16}"
ANCHOR_UPDATES="${ANCHOR_UPDATES:-200}"
MAIN_UPDATES="${MAIN_UPDATES:-800}"
AUX_UPDATES="${AUX_UPDATES:-400}"

RUN_REPORTING="${RUN_REPORTING:-0}"
IMPALA_LR="${IMPALA_LR:-0.0002}"
IMPALA_ENTROPY="${IMPALA_ENTROPY:-0.03}"
PPO_LR="${PPO_LR:-0.00015}"
PPO_ENTROPY="${PPO_ENTROPY:-0.005}"
PPO_CLIP="${PPO_CLIP:-0.2}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"

CANONICAL_B1_RUN_LABEL="b1_anchor_thesis_model_${SEED_TAG}"
CANONICAL_B1_RUN_DIR="runs/${CANONICAL_B1_RUN_LABEL}"
B1_BENCHMARK_RUN_LABEL="b1_anchor_benchmark_${SEED_TAG}"
MAIN_THESIS_RUN_LABEL="thesis_model_${SEED_TAG}"
NOLEAGUE_IMPALA_RUN_LABEL="noleague_impala_${SEED_TAG}"
PPO_LITE_RUN_LABEL="ppo_lite_${SEED_TAG}"
ABLATE_NO_B1_CUTOFF_RUN_LABEL="ablate_no_b1_cutoff_${SEED_TAG}"
NO_TACTICAL_CONTROL_RUN_LABEL="noleague_control_no_tactical_bias_${SEED_TAG}"
ABLATE_NO_TACTICAL_BIAS_RUN_LABEL="ablate_no_tactical_bias_${SEED_TAG}"
TEACHER_FADE_CONTROL_RUN_LABEL="noleague_control_teacher_fade_${SEED_TAG}"
ABLATE_TEACHER_FADE_RUN_LABEL="ablate_teacher_fade_${SEED_TAG}"
MULTIDECK_CONTROL_RUN_LABEL="noleague_control_multideck_${SEED_TAG}"
MULTIDECK_RUN_LABEL="thesis_model_multideck_${SEED_TAG}"
REWARD_SHAPING_CONTROL_RUN_LABEL="noleague_control_reward_shaping_${SEED_TAG}"
ABLATE_REWARD_SHAPING_RUN_LABEL="ablate_reward_shaping_${SEED_TAG}"
NORECURRENCE_B1_RUN_LABEL="b1_anchor_norecurrence_${SEED_TAG}"
NORECURRENCE_RUN_LABEL="norecurrence_impala_${SEED_TAG}"
NOLEAGUE_IMPALA_SWEEP_GROUP="thesis_prescreen_noleague_impala_${SEED_TAG}"
NORECURRENCE_SWEEP_GROUP="thesis_prescreen_norecurrence_${SEED_TAG}"
PPO_SWEEP_GROUP="thesis_prescreen_ppo_${SEED_TAG}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_thesis_queue.sh [--plan] [--run] [--phase PHASE]

Phases:
  prescreen-sweeps Quick compact sweeps to screen IMPALA/PPO settings before long runs.
  b1-benchmark     Short wall-clock B1 no-league benchmark with telemetry + eval.
  overnight-main   Main B1 anchor + main thesis wrapper run.
  matrix           Remaining baselines, ablations, multideck, and optional reporting.
  all              overnight-main followed by matrix.

Defaults:
  --plan           Print the command queue only; do not execute.

Environment overrides:
  SEED=20260421
  NUM_ENVS=2048
  UNROLL_LENGTH=64
  RUNTIME_MODE=train_async_fast
  SWEEP_UPDATES=10
  B1_BENCHMARK_MINUTES=10
  B1_BENCHMARK_MAX_UPDATES=1000000
  B1_BENCHMARK_NUM_ENVS=2048
  B1_BENCHMARK_UNROLL_LENGTH=16
  ANCHOR_UPDATES=200
  MAIN_UPDATES=800
  AUX_UPDATES=400
  RUN_REPORTING=0   Set to 1 to run paper_readiness_check.py after eval-complete jobs.
  IMPALA_LR=0.0002
  IMPALA_ENTROPY=0.03
  PPO_LR=0.00015
  PPO_ENTROPY=0.005
  PPO_CLIP=0.2
  PPO_EPOCHS=4

Notes:
  - Run `--phase prescreen-sweeps` first if you want to tune LR/entropy before the overnight queue.
  - Only change the tuned env vars if the sweep winner improves dev-eval mean by at least 0.02
    and is not more than 10% slower than the current anchor candidate.
  - `--phase matrix` expects the canonical thesis B1 anchor from `--phase overnight-main`
    to already exist for the same seed.
  - `norecurrence_impala` runs late in the queue because it needs its own matched
    feed-forward no-league anchor before the main baseline run.
  - The script is fail-fast and refuses to reuse an existing run label.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      MODE="plan"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_RUN=(.venv/bin/python)
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON_RUN=(.venv/Scripts/python.exe)
else
  echo "Could not find uv, .venv/bin/python, or .venv/Scripts/python.exe" >&2
  exit 1
fi

log_step() {
  echo
  echo "==> $1"
}

print_cmd() {
  printf '   '
  printf '%q ' "$@"
  echo
}

assert_run_dir_absent() {
  local run_label="$1"
  local run_dir="$ROOT_DIR/runs/$run_label"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing run dir: $run_dir" >&2
    exit 1
  fi
}

assert_run_dir_present() {
  local run_label="$1"
  local run_dir="$ROOT_DIR/runs/$run_label"
  if [[ "$MODE" == "run" && ! -d "$run_dir" ]]; then
    echo "Required run dir is missing: $run_dir" >&2
    echo "Run the prerequisite phase first, or point the queue at an existing canonical anchor." >&2
    exit 1
  fi
}

assert_launch_group_absent() {
  local group_label="$1"
  local group_dir="$ROOT_DIR/runs/launch_groups/$group_label"
  if [[ -e "$group_dir" ]]; then
    echo "Refusing to reuse existing launch group dir: $group_dir" >&2
    exit 1
  fi
}

run_cmd() {
  local label="$1"
  shift
  log_step "$label"
  print_cmd "$@"
  if [[ "$MODE" == "run" ]]; then
    "$@"
  fi
}

IMPALA_TRAIN_OVERRIDES=(
  --config-override "training.optimizer.learning_rate=${IMPALA_LR}"
  --config-override "training.exploration.entropy_coef=${IMPALA_ENTROPY}"
)

IMPALA_WRAPPER_OVERRIDES=(
  --train-arg=--config-override
  --train-arg="training.optimizer.learning_rate=${IMPALA_LR}"
  --train-arg=--config-override
  --train-arg="training.exploration.entropy_coef=${IMPALA_ENTROPY}"
)

PPO_TRAIN_OVERRIDES=(
  --config-override "training.optimizer.learning_rate=${PPO_LR}"
  --config-override "training.exploration.entropy_coef=${PPO_ENTROPY}"
  --config-override "training.ppo.clip_epsilon=${PPO_CLIP}"
  --config-override "training.ppo.epochs=${PPO_EPOCHS}"
)

train_job() {
  local run_label="$1"
  local stack_config="$2"
  local max_updates="$3"
  shift 3
  assert_run_dir_absent "$run_label"
  run_cmd \
    "Train $run_label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/train.py \
    --stack-config "$stack_config" \
    --run-label "$run_label" \
    --num-envs "$NUM_ENVS" \
    --unroll-length "$UNROLL_LENGTH" \
    --runtime-mode "$RUNTIME_MODE" \
    --max-updates "$max_updates" \
    --seed "$SEED" \
    "$@"
}

profiled_train_job() {
  local run_label="$1"
  local stack_config="$2"
  local max_updates="$3"
  local max_wall_clock_minutes="$4"
  local num_envs="$5"
  local unroll_length="$6"
  shift 6
  assert_run_dir_absent "$run_label"
  run_cmd \
    "Profiled train $run_label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/profile_train_job.py \
    --run-label "$run_label" \
    --stack-config "$stack_config" \
    --seed "$SEED" \
    --num-envs "$num_envs" \
    --unroll-length "$unroll_length" \
    --runtime-mode "$RUNTIME_MODE" \
    --max-updates "$max_updates" \
    --max-wall-clock-minutes "$max_wall_clock_minutes" \
    "$@"
}

thesis_job() {
  local run_label="$1"
  local preset="$2"
  local b1_run_dir="$3"
  local max_updates="$4"
  shift 4
  assert_run_dir_absent "$run_label"
  run_cmd \
    "Thesis wrapper $run_label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/thesis_run.py \
    --preset "$preset" \
    --run-label "$run_label" \
    --b1-baseline-run-dir "$b1_run_dir" \
    --num-envs "$NUM_ENVS" \
    --unroll-length "$UNROLL_LENGTH" \
    --runtime-mode "$RUNTIME_MODE" \
    --max-updates "$max_updates" \
    --seed "$SEED" \
    --skip-compare \
    "$@"
}

eval_job() {
  local run_label="$1"
  local stack_config="$2"
  shift 2
  run_cmd \
    "Eval $run_label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/eval.py \
    --stack-config "$stack_config" \
    --run-dir "runs/$run_label" \
    "$@"
}

paper_readiness_job() {
  local run_label="$1"
  run_cmd \
    "Paper readiness $run_label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/paper_readiness_check.py \
    --run-dir "runs/$run_label"
}

maybe_report() {
  local run_label="$1"
  if [[ "$RUN_REPORTING" == "1" ]]; then
    paper_readiness_job "$run_label"
  fi
}

sweep_job() {
  local label="$1"
  local preset="$2"
  local group_label="$3"
  assert_launch_group_absent "$group_label"
  run_cmd \
    "$label" \
    "${PYTHON_RUN[@]}" \
    python/scripts/sweep_experiments.py \
    --preset "$preset" \
    --group-label "$group_label" \
    --seed "$SEED" \
    --device cuda:0 \
    --device cuda:1 \
    --device cuda:2 \
    --train-arg=--num-envs \
    --train-arg="${NUM_ENVS}" \
    --train-arg=--unroll-length \
    --train-arg="${UNROLL_LENGTH}" \
    --train-arg=--runtime-mode \
    --train-arg="${RUNTIME_MODE}" \
    --train-arg=--max-updates \
    --train-arg="${SWEEP_UPDATES}"
}

phase_prescreen_sweeps() {
  sweep_job \
    "Sweep noleague IMPALA" \
    noleague_impala_compact \
    "$NOLEAGUE_IMPALA_SWEEP_GROUP"
  sweep_job \
    "Sweep no-recurrence IMPALA" \
    norecurrence_compact \
    "$NORECURRENCE_SWEEP_GROUP"
  sweep_job \
    "Sweep PPO-lite" \
    ppo_compact \
    "$PPO_SWEEP_GROUP"

  log_step "Post-sweep decision"
  echo "   Inspect runs/launch_groups/${NOLEAGUE_IMPALA_SWEEP_GROUP}/summary.json and the peer sweep summaries."
  echo "   If a winner clears the 0.02 / <=10% slowdown rule, re-run with tuned env vars such as:"
  echo "   IMPALA_LR=0.00015 IMPALA_ENTROPY=0.02 PPO_LR=0.0001 PPO_ENTROPY=0.005 PPO_CLIP=0.2 PPO_EPOCHS=4"
}

phase_b1_benchmark() {
  profiled_train_job \
    "$B1_BENCHMARK_RUN_LABEL" \
    configs/baselines/noleague_benchmark.yaml \
    "$B1_BENCHMARK_MAX_UPDATES" \
    "$B1_BENCHMARK_MINUTES" \
    "$B1_BENCHMARK_NUM_ENVS" \
    "$B1_BENCHMARK_UNROLL_LENGTH" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  eval_job \
    "$B1_BENCHMARK_RUN_LABEL" \
    configs/baselines/noleague_benchmark_eval.yaml
  maybe_report "$B1_BENCHMARK_RUN_LABEL"
}

phase_overnight_main() {
  train_job \
    "$CANONICAL_B1_RUN_LABEL" \
    configs/baselines/noleague_impala.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"

  thesis_job \
    "$MAIN_THESIS_RUN_LABEL" \
    thesis-model-server-train \
    "$CANONICAL_B1_RUN_DIR" \
    "$MAIN_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"

  maybe_report "$MAIN_THESIS_RUN_LABEL"
}

phase_matrix() {
  assert_run_dir_present "$CANONICAL_B1_RUN_LABEL"

  train_job \
    "$NOLEAGUE_IMPALA_RUN_LABEL" \
    configs/baselines/noleague_impala.yaml \
    "$AUX_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  eval_job \
    "$NOLEAGUE_IMPALA_RUN_LABEL" \
    configs/baselines/noleague_impala.yaml
  maybe_report "$NOLEAGUE_IMPALA_RUN_LABEL"

  train_job \
    "$PPO_LITE_RUN_LABEL" \
    configs/baselines/ppo_lite.yaml \
    "$AUX_UPDATES" \
    "${PPO_TRAIN_OVERRIDES[@]}" \
    --b1-baseline-run-dir "$CANONICAL_B1_RUN_DIR"
  eval_job \
    "$PPO_LITE_RUN_LABEL" \
    configs/baselines/ppo_lite.yaml \
    --b1-baseline-run-dir "$CANONICAL_B1_RUN_DIR"
  maybe_report "$PPO_LITE_RUN_LABEL"

  thesis_job \
    "$ABLATE_NO_B1_CUTOFF_RUN_LABEL" \
    ablate-no-b1-cutoff \
    "$CANONICAL_B1_RUN_DIR" \
    "$AUX_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"
  maybe_report "$ABLATE_NO_B1_CUTOFF_RUN_LABEL"

  train_job \
    "$NO_TACTICAL_CONTROL_RUN_LABEL" \
    configs/baselines/no_tactical_bias_noleague.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  thesis_job \
    "$ABLATE_NO_TACTICAL_BIAS_RUN_LABEL" \
    ablate-no-tactical-bias \
    "$CANONICAL_B1_RUN_DIR" \
    "$AUX_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"
  maybe_report "$ABLATE_NO_TACTICAL_BIAS_RUN_LABEL"

  train_job \
    "$TEACHER_FADE_CONTROL_RUN_LABEL" \
    configs/baselines/teacher_fade_noleague.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  thesis_job \
    "$ABLATE_TEACHER_FADE_RUN_LABEL" \
    ablate-teacher-fade \
    "$CANONICAL_B1_RUN_DIR" \
    "$AUX_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"
  maybe_report "$ABLATE_TEACHER_FADE_RUN_LABEL"

  train_job \
    "$MULTIDECK_CONTROL_RUN_LABEL" \
    configs/baselines/multideck_noleague.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  thesis_job \
    "$MULTIDECK_RUN_LABEL" \
    thesis-model-multideck \
    "$CANONICAL_B1_RUN_DIR" \
    "$AUX_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"
  maybe_report "$MULTIDECK_RUN_LABEL"

  train_job \
    "$REWARD_SHAPING_CONTROL_RUN_LABEL" \
    configs/baselines/reward_shaping_noleague.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  thesis_job \
    "$ABLATE_REWARD_SHAPING_RUN_LABEL" \
    ablate-reward-shaping \
    "$CANONICAL_B1_RUN_DIR" \
    "$AUX_UPDATES" \
    "${IMPALA_WRAPPER_OVERRIDES[@]}"
  maybe_report "$ABLATE_REWARD_SHAPING_RUN_LABEL"

  train_job \
    "$NORECURRENCE_B1_RUN_LABEL" \
    configs/baselines/norecurrence_noleague.yaml \
    "$ANCHOR_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}"
  train_job \
    "$NORECURRENCE_RUN_LABEL" \
    configs/baselines/norecurrence_impala.yaml \
    "$AUX_UPDATES" \
    "${IMPALA_TRAIN_OVERRIDES[@]}" \
    --b1-baseline-run-dir "runs/${NORECURRENCE_B1_RUN_LABEL}"
  eval_job \
    "$NORECURRENCE_RUN_LABEL" \
    configs/baselines/norecurrence_impala.yaml \
    --b1-baseline-run-dir "runs/${NORECURRENCE_B1_RUN_LABEL}"
  maybe_report "$NORECURRENCE_RUN_LABEL"
}

case "$PHASE" in
  prescreen-sweeps)
    phase_prescreen_sweeps
    ;;
  b1-benchmark)
    phase_b1_benchmark
    ;;
  overnight-main)
    phase_overnight_main
    ;;
  matrix)
    phase_matrix
    ;;
  all)
    phase_overnight_main
    phase_matrix
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "plan" ]]; then
  echo
  echo "Plan-only mode complete. Re-run with --run to execute."
fi
