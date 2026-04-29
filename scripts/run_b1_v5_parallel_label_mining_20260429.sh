#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl

RUN_DIR="runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429"
B1_DIR="runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429"
CKPT="u120=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt"
STACK="configs/baselines/noleague_impala.yaml"
TAG="thesis_v5_mine50_20260429"

COMMON=(
  .venv/bin/python python/scripts/b1_counterfactual_labels.py
  --stack-config "$STACK"
  --run-dir "$RUN_DIR"
  --b1-baseline-run-dir "$B1_DIR"
  --checkpoint-policy "$CKPT"
  --pairs 32
  --public-heuristic-bias-scale 1.0
  --max-target-states 240
  --max-targets-per-pair 6
  --max-actions-per-state 10
  --max-forced-replays 3000
  --progress-every 100
  --stop-after-positive-labels 15
  --randomize-target-order
  --execution-mode in_process
  --ignore-excluded-label-pair-indices
  --margin-positive-threshold 0.15
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_pass_overextend/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_attack_climax/counterfactual_labels.jsonl"
)

launch_cluster() {
  local session="$1"
  local gpu="$2"
  local log="$3"
  shift 3
  tmux new-session -d -s "$session" "$* > $log 2>&1"
}

launch_cluster "b1_labels_pass_20260429" 0 "notes/${TAG}_pass_overextend.log" \
  CUDA_VISIBLE_DEVICES=0 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_pass_overextend" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_pass_overextend" \
  --target-random-seed 52950 \
  --target-families main_play_character \
  --family-representatives-per-family 1 \
  --require-pass-legal \
  --require-baseline-family main_play_character

launch_cluster "b1_labels_nonpass_20260429" 1 "notes/${TAG}_main_nonpass.log" \
  CUDA_VISIBLE_DEVICES=1 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_main_nonpass" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_main_nonpass" \
  --target-random-seed 52951 \
  --target-families main_play_character,main_move \
  --family-representatives-per-family 3 \
  --exclude-candidate-family pass \
  --exclude-candidate-action-id 51

launch_cluster "b1_labels_attack_20260429" 2 "notes/${TAG}_attack_climax.log" \
  CUDA_VISIBLE_DEVICES=2 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_attack_climax" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_attack_climax" \
  --target-random-seed 52952 \
  --target-families attack,climax_play,main_play_event \
  --family-representatives-per-family 3

launch_cluster "b1_labels_twostep_20260429" 3 "notes/${TAG}_broad_twostep.log" \
  CUDA_VISIBLE_DEVICES=3 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_broad_twostep" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_broad_twostep" \
  --target-random-seed 52953 \
  --target-families main_play_character,main_move,attack,climax_play,main_play_event,level_up,clock_from_hand \
  --family-representatives-per-family 2 \
  --two-step-beam-targets 16 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.03 \
  --two-step-include-positive-first \
  --two-step-max-replays 3000

tmux ls | grep -E "b1_labels_(pass|nonpass|attack|twostep)_20260429"
