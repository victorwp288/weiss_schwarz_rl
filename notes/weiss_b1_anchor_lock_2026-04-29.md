# Weiss Thesis B1 Anchor Lock

Date: 2026-04-29

Canonical B1 anchor run:

- Run label: `thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`
- Run dir: `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`
- Canonical checkpoint: `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/best.pt`
- Source checkpoint for `best.pt`: `training/checkpoints/checkpoint_120.pt`
- Policy version: 6
- Online dev-eval aggregate at u120: 0.86875
- 64-pair confirm aggregate at u120: 0.875

64-pair confirm scores:

- B0 RandomLegal: 1.0
- B1 NoLeague baseline: 0.5078125
- B2 HeuristicPublic: 1.0
- B3 HeuristicPublicAggro: 0.875
- B4 HeuristicPublicControl: 0.9921875

Training command:

```bash
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
```

Use in downstream runs:

```bash
--b1-baseline-run-dir runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429
```

Config decision:

- Do not copy these B1-anchor training knobs into main league, baselines, or ablations globally.
- Do pass the frozen B1 run dir to downstream train/eval/mining entrypoints.
- If rerunning the B1 anchor itself, use the exact command above or create a dedicated B1-anchor config from these overrides.
