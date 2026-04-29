#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl

RUN_DIR="runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429"
B1_DIR="$RUN_DIR"
CKPT="u120=${RUN_DIR}/training/checkpoints/checkpoint_120.pt"
STACK="configs/baselines/noleague_impala.yaml"
TAG="thesis_v5_broad50b_20260429"

COMMON=(
  .venv/bin/python python/scripts/b1_counterfactual_labels.py
  --stack-config "$STACK"
  --run-dir "$RUN_DIR"
  --b1-baseline-run-dir "$B1_DIR"
  --checkpoint-policy "$CKPT"
  --pairs 48
  --public-heuristic-bias-scale 1.0
  --max-target-states 420
  --max-targets-per-pair 8
  --max-actions-per-state 14
  --max-forced-replays 6000
  --progress-every 200
  --stop-after-positive-labels 18
  --randomize-target-order
  --execution-mode in_process
  --ignore-excluded-label-pair-indices
  --margin-positive-threshold 0.05
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_pass_overextend/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_attack_climax/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_mine50_20260429_broad_twostep/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_broad50_20260429_attack_event/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_broad50_20260429_broad_a/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_broad50_20260429_broad_b/counterfactual_labels.jsonl"
  --exclude-labels "$RUN_DIR/eval/b1_cf_labels_s1_big_thesis_v5_broad50_20260429_pass_repair/counterfactual_labels.jsonl"
)

launch_cluster() {
  local session="$1"
  local log="$2"
  shift 2
  tmux new-session -d -s "$session" "$* > $log 2>&1"
}

launch_cluster "b1_broad50b_labels_a_20260429" "notes/${TAG}_broad_a.log" \
  CUDA_VISIBLE_DEVICES=0 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_broad_a" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_broad_a" \
  --target-random-seed 63000 \
  --target-families main_play_character,main_move,attack,climax_play,main_play_event,level_up,clock_from_hand \
  --family-representatives-per-family 2 \
  --two-step-beam-targets 16 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 6000

launch_cluster "b1_broad50b_labels_b_20260429" "notes/${TAG}_broad_b.log" \
  CUDA_VISIBLE_DEVICES=1 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_broad_b" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_broad_b" \
  --target-random-seed 63100 \
  --target-families main_play_character,main_move,main_play_event,climax_play,attack,pass \
  --family-representatives-per-family 3 \
  --two-step-beam-targets 12 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 6000

launch_cluster "b1_broad50b_labels_attack_20260429" "notes/${TAG}_attack_event.log" \
  CUDA_VISIBLE_DEVICES=2 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_attack_event" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_attack_event" \
  --target-random-seed 63200 \
  --target-families attack,climax_play,main_play_event,main_move \
  --family-representatives-per-family 4 \
  --two-step-beam-targets 10 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 6000

launch_cluster "b1_broad50b_labels_pass_20260429" "notes/${TAG}_pass_repair.log" \
  CUDA_VISIBLE_DEVICES=3 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_pass_repair" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_pass_repair" \
  --target-random-seed 63300 \
  --target-families main_play_character,main_move,pass \
  --family-representatives-per-family 3 \
  --require-pass-legal \
  --two-step-beam-targets 12 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 6000

tmux ls | grep -E "b1_broad50b_labels_(a|b|attack|pass)_20260429"
