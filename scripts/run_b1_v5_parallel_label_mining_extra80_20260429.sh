#!/usr/bin/env bash
set -euo pipefail

cd /workspace/weiss_schwarz_rl

RUN_DIR="runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429"
B1_DIR="$RUN_DIR"
CKPT="u120=${RUN_DIR}/training/checkpoints/checkpoint_120.pt"
STACK="configs/baselines/noleague_impala.yaml"
TAG="thesis_v5_extra80_20260429"

EXCLUDE_ARGS=()
while IFS= read -r labels_path; do
  EXCLUDE_ARGS+=(--exclude-labels "$labels_path")
done < <(find "$RUN_DIR/eval" -maxdepth 2 -type f -name counterfactual_labels.jsonl | sort)

COMMON=(
  .venv/bin/python python/scripts/b1_counterfactual_labels.py
  --stack-config "$STACK"
  --run-dir "$RUN_DIR"
  --b1-baseline-run-dir "$B1_DIR"
  --checkpoint-policy "$CKPT"
  --pairs 56
  --public-heuristic-bias-scale 1.0
  --max-target-states 520
  --max-targets-per-pair 8
  --max-actions-per-state 14
  --max-forced-replays 7000
  --progress-every 200
  --stop-after-positive-labels 20
  --randomize-target-order
  --execution-mode in_process
  --ignore-excluded-label-pair-indices
  --margin-positive-threshold 0.03
  "${EXCLUDE_ARGS[@]}"
)

launch_cluster() {
  local session="$1"
  local log="$2"
  shift 2
  tmux new-session -d -s "$session" "$* > $log 2>&1"
}

launch_cluster "b1_extra80_broad_a_20260429" "notes/${TAG}_broad_a.log" \
  CUDA_VISIBLE_DEVICES=0 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_broad_a" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_broad_a" \
  --target-random-seed 73400 \
  --target-families main_play_character,main_move,attack,climax_play,main_play_event,level_up,clock_from_hand \
  --family-representatives-per-family 2 \
  --two-step-beam-targets 16 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 7000

launch_cluster "b1_extra80_broad_b_20260429" "notes/${TAG}_broad_b.log" \
  CUDA_VISIBLE_DEVICES=1 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_broad_b" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_broad_b" \
  --target-random-seed 73500 \
  --target-families main_play_character,main_move,main_play_event,climax_play,attack,pass \
  --family-representatives-per-family 3 \
  --two-step-beam-targets 12 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 7000

launch_cluster "b1_extra80_attack_20260429" "notes/${TAG}_attack_event.log" \
  CUDA_VISIBLE_DEVICES=2 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_attack_event" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_attack_event" \
  --target-random-seed 73600 \
  --target-families attack,climax_play,main_play_event,main_move \
  --family-representatives-per-family 4 \
  --two-step-beam-targets 12 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 7000

launch_cluster "b1_extra80_pass_20260429" "notes/${TAG}_pass_repair.log" \
  CUDA_VISIBLE_DEVICES=3 "${COMMON[@]}" \
  --device cuda:0 \
  --artifact-dir-name "b1_cf_labels_s1_big_${TAG}_pass_repair" \
  --seed-scope "b1_cf_labels_s1_big_${TAG}_pass_repair" \
  --target-random-seed 73700 \
  --target-families main_play_character,main_move,pass \
  --family-representatives-per-family 3 \
  --require-pass-legal \
  --two-step-beam-targets 12 \
  --two-step-second-actions 6 \
  --two-step-min-first-delta 0.02 \
  --two-step-include-positive-first \
  --two-step-max-replays 7000

tmux ls | grep -E "b1_extra80_(broad_a|broad_b|attack|pass)_20260429"
