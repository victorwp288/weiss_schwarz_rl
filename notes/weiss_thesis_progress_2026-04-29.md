# Weiss RL Thesis Progress - Vast Box

Date: 2026-04-29

## Workspace

- Remote repo: `/workspace/weiss_schwarz_rl`
- Branch: `idea/server-multigpu-weiss-sim-082`

## Verified

- SSH works with `~/.ssh/vast_ed25519`.
- Hardware detected:
  - 4x RTX PRO 6000 Blackwell Max-Q, about 98 GB VRAM each.
  - 96 CPU cores.
  - about 566 GiB RAM.
- Default `ulimit -n` was 1024 and caused shared-memory collector failures.
- `ulimit -n 1048576` fixes the open-file/shared-memory startup failure.
- Standalone NCCL all-reduce works.
- Repo DDP with NCCL crashes with CUDA illegal memory access.
- Repo DDP with Gloo works with CUDA learners and 4 GPUs.

## Smoke Runs

- `vast_single_b1_smoke_ulimit_20260429_1158`
  - Single GPU smoke passed after raising `ulimit -n`.
- `vast_ddp_gloo_4gpu_smoke_20260429_1230`
  - 4-GPU Gloo DDP smoke passed.
- `vast_main_gloo_4gpu_smoke_20260429_1235`
  - Launch path worked, but failed because old copied B1 anchor does not match active main model config.

## B1 Anchor Compatibility

- Old copied anchor: `runs/b1_anchor_thesis_model_seed20260421_bestb1`
  - Not compatible with active `configs/main_impala_league_server.yaml`.
- Matching B1 config for active main:
  - `configs/baselines/noleague_impala.yaml`

## Paused Exploratory Run

- `runs/b1_anchor_noleague_impala_gloo_seed20260429`
  - Reached update 30.
  - Wrote checkpoint metadata at update 20.
  - Paused before final planning.
  - Treat as exploratory unless explicitly resumed.

## Current Next Step

Benchmark model/topology envelope, starting with width/GRU 512 and the verified launch surface:

- `ulimit -n 1048576`
- `--torchrun-nproc 4`
- `--ddp-backend gloo`
- `--autoscale --hardware-profile local`

## Benchmark Envelope Results

Raw benchmark comparison disabled periodic dev eval and raised checkpoint/snapshot intervals so the numbers measure training throughput.

Pass 1 result file:

- `notes/vast_envelope_20260429_121838_results.md`

Best/important pass 1 cases:

- width 248, 512 envs/GPU, unroll 64: mean 35.5k samples/s, max 51.8k, max VRAM about 20.1 GB.
- width 512, 512 envs/GPU, unroll 64: mean 34.3k samples/s, max 49.6k, max VRAM about 28.8 GB.
- width 512, 768 envs/GPU, unroll 64: mean 27.4k samples/s; slower despite more envs.
- width 384, 768 envs/GPU, unroll 64: mean 28.0k samples/s; slower than 512 envs/GPU.
- width 512, 1024 envs/GPU failed before metrics with a shared-memory `FileExistsError`.

Pass 2 result file:

- `notes/vast_envelope_pass2_20260429_122509_results.md`

Best/important pass 2 cases:

- width 512, 384 envs/GPU, unroll 64: mean 39.5k samples/s, max 55.7k, max VRAM about 28.3 GB.
- width 512, 512 envs/GPU, unroll 96: mean 47.4k samples/s, max 66.9k, max VRAM about 40.2 GB.
- width 512, 512 envs/GPU, unroll 128: mean 58.1k samples/s, max 80.3k, max VRAM about 44.7 GB.
- width 512, 512 envs/GPU, unroll 64, max 32 envs/actor: mean 12.1k samples/s; bad topology.
- width 768, 512 envs/GPU, unroll 64: mean 32.8k samples/s, max 47.0k, max VRAM about 38.4 GB.
- width 640, 512 envs/GPU, unroll 64 failed before metrics with a shared-memory `FileExistsError`; not treated as a model-size limit.

Shared-memory cleanup:

- After benchmarking, `/dev/shm` contained 800 leaked `weissrl_*` segments.
- No train/profile processes were running, so these were deleted.
- `/dev/shm` now has 0 `weissrl_*` segments.

## Benchmark Interpretation

- Width/GRU 512 is the sweet spot for the thesis model right now: it is much larger than the current 248 model, but at the same 512 envs/GPU and unroll 64 it was only about 3 percent slower in pass 1.
- Raising target envs/GPU above 512 is counterproductive on this runtime; 768 slowed down and 1024 hit shared-memory startup failure.
- Longer unrolls substantially increase raw samples/s for the 512 model, but they reduce updates/s and change the learning/update schedule. If we use unroll 128, use it consistently for B1/main/baselines/ablations or explicitly document it as the chosen scaled-throughput setting.
- More actors with 32 envs/actor was much worse. Keep max envs/actor 64.
- Width 768 is viable but roughly 5 percent slower than 512/u64 and much slower than 512 with longer unrolls. It is a possible "large ablation" but not the main thesis choice unless quality strongly benefits.

## Recommended Config Envelope

Use this for the main run family unless a longer confirmation run contradicts it:

- `model.gru_hidden_size=512`
- `model.encoder_mlp_width=512`
- `training.scaling.target_envs_per_gpu=512`
- `training.scaling.max_actor_process_count=64`
- `training.scaling.max_envs_per_actor=64`
- `training.rollout.unroll_length=128` for maximum sample throughput, or keep 64 if we want the lowest-risk continuation of the existing training schedule.
- Keep DDP backend `gloo`; avoid repo NCCL for now.

Conservative thesis choice:

- width 512, envs/GPU 512, unroll 64.

Aggressive throughput choice:

- width 512, envs/GPU 512, unroll 128.

I lean aggressive for the B1/main family if we apply it consistently and set max updates/sample budget accordingly, because it uses the box much better while staying far below VRAM limits.

## Fixed-512 Runtime Geometry Sweep

Result file:

- `notes/vast_runtime_geometry_20260429_123740_results.md`

All cases in this sweep used:

- `model.gru_hidden_size=512`
- `model.encoder_mlp_width=512`
- 4 GPUs
- Gloo DDP
- `max_actor_process_count=64`
- checkpoint/dev-eval overhead disabled for raw throughput

Important results:

- env/GPU 384, unroll 96, batch unrolls 64:
  - mean 63.9k samples/s, max 83.3k
  - mean updates/s 0.62
  - mean/max GPU util 27/99 percent
  - mean/max VRAM 22.5/35.5 GB
  - mean/max CPU 16.1/26.4 cores
- env/GPU 384, unroll 128, batch unrolls 64:
  - mean 76.9k samples/s, max 98.1k
  - mean updates/s 0.56
  - mean/max GPU util 15/73 percent
  - mean/max VRAM 28.8/43.1 GB
  - mean/max CPU 19.4/30.2 cores
- env/GPU 384, unroll 160, batch unrolls 64:
  - mean 88.0k samples/s, max 110.1k
  - mean updates/s 0.51
  - mean/max GPU util 47/79 percent
  - mean/max VRAM 34.7/50.1 GB
  - mean/max CPU 21.7/32.0 cores
- env/GPU 512, unroll 160, batch unrolls 64:
  - mean 80.0k samples/s, max 102.9k
  - mean updates/s 0.46
  - mean/max GPU util 16/67 percent
  - mean/max VRAM 36.4/51.9 GB
  - mean/max CPU 19.5/32.2 cores
- env/GPU 512, unroll 192, batch unrolls 64:
  - mean 88.5k samples/s, max 112.4k
  - mean updates/s 0.43
  - mean/max GPU util 52/100 percent
  - mean/max VRAM 43.9/63.0 GB
  - mean/max CPU 23.1/33.5 cores
- env/GPU 640, unroll 128, batch unrolls 64:
  - mean 64.6k samples/s, max 87.6k
  - mean updates/s 0.47
  - mean/max GPU util 47/100 percent
  - mean/max VRAM 28.4/46.2 GB
  - mean/max CPU 18.8/30.0 cores
- env/GPU 512, unroll 128, batch unrolls 128:
  - mean 109.4k samples/s, max 136.8k
  - mean updates/s 0.40
  - mean/max GPU util 35/100 percent
  - mean/max VRAM 54.8/75.4 GB
  - mean/max CPU 28.8/42.4 cores

Failed cases:

- env/GPU 256, unroll 128, batch unrolls 64:
  - failed before metrics with shared-memory `FileExistsError`; not useful as a speed point.
- env/GPU 512, unroll 128, batch unrolls 256:
  - failed with CUDA OOM, about 89 GB used per GPU. Treat batch unrolls 256 as too large.

Interpretation:

- Model size remains locked at 512; `e384`, `e512`, and `e640` are environments per GPU, not model width.
- Increasing envs above 512 is still not helpful. env/GPU 640 was slower than the best 384/512 cases.
- The best balanced stable point is env/GPU 384, unroll 160, batch unrolls 64: high samples/s, better updates/s than the larger-batch case, and only about 50 GB peak VRAM.
- The highest raw sample-throughput point is env/GPU 512, unroll 128, batch unrolls 128: about 109k mean samples/s, but it uses about 75 GB peak VRAM and changes the effective batch/update schedule more.

Current recommendation before main runs:

- Robust thesis envelope: width 512, env/GPU 384, unroll 160, batch unrolls 64.
- Aggressive throughput envelope: width 512, env/GPU 512, unroll 128, batch unrolls 128.
- I prefer robust for B1/main/baseline/ablation unless we first do a longer main-config smoke of the aggressive setting, because league/reference-policy memory could reduce the 75 GB headroom.

## Config Lock-In

Locked the robust thesis envelope into the active thesis config family:

- `model.gru_hidden_size=512`
- `model.encoder_mlp_width=512`
- `training.rollout.unroll_length=160`
- `training.rollout.batch_unrolls_per_update=64`
- `training.scaling.target_envs_per_gpu=384`
- `training.scaling.max_envs_per_actor=64`
- `training.scaling.max_actor_process_count=64`

Updated 25 configs:

- `configs/main_impala_league_server.yaml`
- `configs/main_eval.yaml`
- `configs/thesis_locked.yaml`
- Active `configs/baselines/*.yaml` thesis-family configs, excluding old `noleague_benchmark*`.
- Active `configs/ablations/*.yaml` train/eval configs.

Left alone:

- `configs/archive/**`
- `configs/baselines/noleague_benchmark*`
- local/smoke/minimal configs
- residual S1 configs

Validation:

- All 25 changed configs loaded successfully with `weiss_rl.config.load_stack_config`.
- Parser assertions confirmed the locked envelope values in every changed config.

## B1 Thesis Candidate v1

Started fresh matching B1 candidate:

- Run label: `thesis_b1_candidate_v1_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 400
- Seed: 20260429
- tmux session: `thesis_b1_v1_20260429`
- Profile log: `notes/thesis_b1_candidate_v1_20260429_profile.log`

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v1_20260429 \
  --stack-config configs/baselines/noleague_impala.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 400 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30
```

## B1 Counterfactual Label Mining

Anchor used:

- `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`
- Checkpoint policy: `u120=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt`
- Stack config: `configs/baselines/noleague_impala.yaml`

Important fix:

- `python/scripts/b1_counterfactual_labels.py` was patched so tensor sidecars can store large unsigned `episode_seed` values safely.
- The JSON label rows retain exact `episode_seed`; tensor sidecars now store a signed int64-compatible `episode_seed` plus exact `episode_seed_u64`.

Main mining script:

- Remote: `scripts/run_b1_v5_parallel_label_mining_broad_20260429.sh`
- Local copy: `run_b1_v5_parallel_label_mining_broad_20260429.sh`
- Final tag: `thesis_v5_broad50b_20260429`
- Launch command:

```bash
cd /workspace/weiss_schwarz_rl
chmod +x scripts/run_b1_v5_parallel_label_mining_broad_20260429.sh
./scripts/run_b1_v5_parallel_label_mining_broad_20260429.sh
```

Mining settings:

- Four parallel tmux jobs, one per GPU:
  - `b1_broad50b_labels_a_20260429`
  - `b1_broad50b_labels_b_20260429`
  - `b1_broad50b_labels_attack_20260429`
  - `b1_broad50b_labels_pass_20260429`
- `--pairs 48`
- `--max-target-states 420`
- `--max-targets-per-pair 8`
- `--max-actions-per-state 14`
- `--max-forced-replays 6000`
- `--stop-after-positive-labels 18`
- `--margin-positive-threshold 0.05`
- `--execution-mode in_process`
- two-step search enabled with small beams.

System utilization:

- During mining each active GPU ran at about 27-28 percent utilization and about 782 MiB VRAM.
- This is replay/simulator-bound, not VRAM-bound; parallel one-process-per-GPU mining gave much better use than the sequential wrapper.
- At completion all tmux mining jobs exited and all GPUs returned to idle.

Final label counts:

- `b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_attack_event`: 19
- `b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_broad_a`: 19
- `b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_broad_b`: 19
- `b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_pass_repair`: 19
- Fresh `broad50b` labels: 76
- Earlier usable labels:
  - `quick2_attack_climax`: 1
  - `quick2_pass_overextend`: 1
  - `mine50_broad_twostep`: 2
  - `broad50_pass_repair`: 1
- Total usable B1 counterfactual labels now validated: 81

Validation summary:

- Missing tensor refs: 0
- Tensor load errors: 0
- Winner-flip labels: 42
- Margin-positive labels: 81
- Label weights:
  - `1.0`: 42
  - `0.4`: 39
- Positive action families:
  - `pass`: 48
  - `main_move`: 33
- Selected families:
  - `main_move`: 37
  - `main_play_character`: 34
  - `attack`: 8
  - `clock_from_hand`: 2
- Score deltas:
  - min: 0.050000000000000044
  - max: 2.3
  - mean: 1.1589506172839505

Interpretation:

- The requested 50+ label target was exceeded with a validated 81-label pool.
- The pool is pass/main-move heavy, which matches where this miner found B1 counterfactual leverage.
- Because 42 labels are winner flips and the remaining 39 are lower-weight margin positives, the set is useful without being all weak labels.
- Next step should be wiring these label directories into the residual/counterfactual-positive part of the main thesis run, then launching the main league candidate.

## Main Thesis Config Wiring

Updated:

- `configs/main_impala_league_server.yaml`

The active main server config now enables:

```yaml
training:
  counterfactual_positive:
    enabled: true
    label_dirs:
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_attack_climax
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_quick2_20260429_pass_overextend
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_mine50_20260429_broad_twostep
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_broad50_20260429_pass_repair
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_attack_event
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_broad_a
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_broad_b
      - runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/b1_cf_labels_s1_big_thesis_v5_broad50b_20260429_pass_repair
    coef: 1.0
    final_coef: 0.25
    start_updates: 0
    end_updates: 120
    margin_coef: 0.2
    margin: 1.0
    max_labels: 0
```

Rationale:

- Use all 81 validated labels instead of the old `max_labels: 10` S1 setting.
- Keep the auxiliary moderate and early: it should bridge away from B1 mistakes without dominating the whole main league run.
- `max_labels: 0` means no cap in this codepath.

Validation:

- `load_stack_config(configs/main_impala_league_server.yaml)` succeeds.
- The learner counterfactual-positive loader reads all 81 records from the configured dirs.
- Loaded label weight sum: 57.600000232458115.
- Autoscale dry run with the locked B1 anchor succeeds:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_main_cf_labels_ready_dryrun_20260429 \
  --stack-config configs/main_impala_league_server.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 1 \
  --autoscale \
  --autoscale-dry-run \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --train-arg=--b1-baseline-run-dir \
  --train-arg=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429
```

Resolved dry-run topology:

- DDP backend: gloo
- learner GPUs: 4
- actor count: 24
- envs per actor: 64
- total envs: 1536
- batch unrolls per update: 64

Prepared launch script, not yet run:

- `scripts/run_thesis_main_v1_cf81_20260429.sh`
- default `RUN_LABEL=thesis_main_candidate_v1_cf81_20260429`
- default `MAX_UPDATES=400`

Launch when ready:

```bash
cd /workspace/weiss_schwarz_rl
./scripts/run_thesis_main_v1_cf81_20260429.sh
```

## Main Thesis Tuning Pass

Question: before launching the main claim run, check whether the main config is too teacher-heavy.

Finding:

- The config was mechanically ready, but `actor_heuristic_fraction` and `heuristic_public_mix_fraction` were both effectively pinned at `1.0` forever.
- That is too teacher-heavy for the main league claim because it delays/limits actual model policy data and league pressure.

Tuning changes applied to `configs/main_impala_league_server.yaml`:

- Counterfactual-positive auxiliary:
  - `coef: 1.0`
  - `final_coef: 0.1`
  - `start_updates: 0`
  - `end_updates: 160`
  - `margin_coef: 0.2`
  - `margin: 1.0`
  - `max_labels: 0`
- Actor policy mix:
  - `actor_heuristic_fraction: 1.0`
  - `actor_heuristic_start_updates: 80`
  - `actor_heuristic_end_updates: 240`
  - `actor_heuristic_final_fraction: 0.35`
  - `heuristic_actor_hidden_state_tracking: true`
- Opponent mix:
  - `heuristic_public_mix_fraction: 1.0`
  - `heuristic_public_mix_end_updates: 240`
  - `heuristic_public_final_mix_fraction: 0.35`
  - `noleague_baseline_mix_fraction: 0.15`
  - `noleague_baseline_mix_end_updates: 160`
- Early cutoff:
  - `curriculum.early_cutoff.enabled: false`

Intended schedule:

- u0: actor heuristic 1.0, heuristic opponent 1.0, B1 opponent 0.15, CF coef 1.0
- u80: actor heuristic still 1.0, heuristic opponent about 0.78, B1 opponent 0.15, CF coef 0.55
- u120: actor heuristic about 0.84, heuristic opponent about 0.68, B1 opponent 0.15, CF coef 0.325
- u160: actor heuristic about 0.675, heuristic opponent about 0.57, B1 opponent 0.0, CF coef 0.1
- u240+: actor heuristic 0.35, heuristic opponent 0.35, CF coef 0.1

Bug found during forced mixed-path smoke:

- Mixed actor/model sampling hit a recursion bug in `python/weiss_rl/model.py`.
- Cause: `_sample_packed_action_scores` monkey-patched `structured_action_head._packed_local_cdf` to a wrapper that called `structured_action_head._packed_local_cdf`, causing self-recursion.
- Fix: import `weiss_rl.structured_sampling` and have the wrapper call `structured_sampling.packed_local_cdf` directly.
- Direct packed sampling test now passes.

Smoke test:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_main_cf81_mixed_actor_smoke_20260429 \
  --stack-config configs/main_impala_league_server.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 3 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 15 \
  --override training.actor_heuristic_start_updates=0 \
  --override training.actor_heuristic_end_updates=2 \
  --override training.actor_heuristic_final_fraction=0.35 \
  --override league.sampling.heuristic_public_mix_end_updates=2 \
  --override league.sampling.heuristic_public_final_mix_fraction=0.35 \
  --override evaluation.periodic_dev_eval_interval_updates=0 \
  --override training.checkpointing.checkpoint_interval_updates=1000 \
  --override training.checkpointing.snapshot_interval_updates=1000 \
  --train-arg=--b1-baseline-run-dir \
  --train-arg=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429
```

Smoke result:

- Exit code: 0
- Updates completed: 3
- Forced mixed path reached actor heuristic fraction `0.35`.
- At update 3 opponent sampling weights were:
  - heuristic public: 0.35
  - mirror: 0.50
  - B1 no-league baseline: 0.15
- Counterfactual labels loaded: 81
- Counterfactual coef active at update 3: about 0.983
- Policy train fraction: about 0.501
- No training processes remain afterward.
- GPUs idle afterward.
- `/dev/shm` has 0 `weissrl_*` segments afterward.

Current launch recommendation:

- Run `thesis_main_candidate_v1_cf81_20260429` for 400 max updates using `scripts/run_thesis_main_v1_cf81_20260429.sh`.
- Monitor u40/u80/u120/u160 carefully.
- If u160 is still improving, extend or relaunch continuation to 600 rather than stopping at 400.

Dry-run topology immediately before launch:

- 4 GPU Gloo DDP
- actor_count: 24
- envs_per_actor: 64
- total_envs: 1536
- batch_unrolls_per_update: 64
- queue_capacity_unrolls: 256

Status:

- Stopped `thesis_b1_candidate_v1_20260429` at update 50 because online dev eval cadence was throttling training.
- v1 is diagnostic/interrupted, not the preferred final B1 anchor.

## Dev Eval Cadence Adjustment

Changed active thesis-family configs to lighter online dev eval:

- `evaluation.periodic_dev_eval_interval_updates=40`
- `evaluation.periodic_dev_eval_paired_seeds=16`

Rationale:

- periodic dev eval was already async and parallel (`async_periodic_dev_eval_enabled=true`, `periodic_dev_eval_parallel_workers=6`) for B1.
- eval every 10 updates with 32 paired seeds was still heavy enough to backlog and throttle the run.
- final/report evals remain the place for stronger seed budgets.

Validation:

- All 25 active thesis-family configs loaded successfully after the cadence change.
- Parser assertions confirmed the locked training envelope plus interval 40 / paired seeds 16.

## B1 Thesis Candidate v2

Started fresh B1 candidate with lighter online dev eval:

- Run label: `thesis_b1_candidate_v2_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 400
- Seed: 20260429
- tmux session: `thesis_b1_v2_20260429`
- Profile log: `notes/thesis_b1_candidate_v2_20260429_profile.log`

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v2_20260429 \
  --stack-config configs/baselines/noleague_impala.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 400 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30
```

Completion:

- Finished with exit code 0.
- Reached update 400.
- Run dir: `runs/thesis_b1_candidate_v2_20260429`
- Final raw checkpoint exists: `training/checkpoints/checkpoint_400.pt`
- Checkpoint guard finalized aliases back to update 40:
  - `training/checkpoints/best.pt`
  - `training/checkpoints/latest.pt`
- Best dev-eval score was update 40:
  - aggregate score: 0.85
  - paired seeds: 16
  - anchor scores:
    - B0 RandomLegal: 1.0
    - B1 NoLeague baseline: 0.5625
    - B2 HeuristicPublic: 1.0
    - B3 HeuristicPublicAggro: 0.6875
    - B4 HeuristicPublicControl: 1.0
- Final update 400 dev-eval score was lower:
  - aggregate score: 0.54375
  - paired seeds: 16
  - anchor scores:
    - B0 RandomLegal: 1.0
    - B1 NoLeague baseline: 0.4375
    - B2 HeuristicPublic: 0.75
    - B3 HeuristicPublicAggro: 0.1875
    - B4 HeuristicPublicControl: 0.34375

Throughput/health:

- Last metric update: 400.
- Wall clock: about 582 seconds.
- Final cumulative throughput: about 2823 samples/s and 0.688 updates/s.
- No stderr errors.
- GPUs idle after completion.
- `/dev/shm` cleaned up to 0 `weissrl_*` segments after exit.

Interpretation:

- Lighter online eval cadence fixed the v1 throttling problem.
- The run did not monotonically improve; checkpoint guard selected the early update-40 policy as best.
- Treat `runs/thesis_b1_candidate_v2_20260429/training/checkpoints/best.pt` as the B1 candidate unless a stronger confirmatory eval says otherwise.

## B1 Thesis Candidate v3

Started a fast diagnostic/candidate rerun to address the v2 failure mode:

- Run label: `thesis_b1_candidate_v3_profiles_cycle_noguard_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 400
- Seed: 20260429
- tmux session: `thesis_b1_v3_20260429`
- Profile log: `notes/thesis_b1_candidate_v3_profiles_cycle_noguard_20260429_profile.log`
- Main change: train rollouts cycle across `base`, `aggressive`, and `control` heuristic-native profiles instead of fixed `base`.
- Diagnostic change: disable checkpoint guard for this run so online eval does not roll aliases back to an early/noisy checkpoint.

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v3_profiles_cycle_noguard_20260429 \
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
```

## Main League v1 Probe

Run label: `thesis_main_candidate_v1_cf81_20260429`

Status: stopped at update 120. Treat this as a failed tuning probe, not a thesis candidate.

Reason:
- u40 equal-weight aggregate looked acceptable, but B1 was weak: 9/32.
- u80 declined: B1 2/32, B3 11/32, B4 21/32.
- u120 collapsed: B1 0/32, B3 2/32, B4 3/32.
- Runtime mix showed the config was over-dominated by base `heuristic_public`: no B3/B4 variant opponent lane, only 15% B1, and PFSP still not ready.

Action taken:
- Stopped the run.
- Created `configs/main_impala_league_server_v2_anchor_variant.yaml`.
- Created `scripts/run_thesis_main_v2_anchor_variant_20260429.sh`.

## Main League v2 Anchor/Variant Probe

Run label: `thesis_main_candidate_v2_anchor_variant_cf81_20260429`

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v2_anchor_variant_cf81_20260429 \
MAX_UPDATES=240 \
./scripts/run_thesis_main_v2_anchor_variant_20260429.sh
```

Key config changes versus v1:
- learning rate `0.0002 -> 0.00005`
- actor backend `heuristic_public -> model`
- actor heuristic fraction `1.0 -> 0.0`
- actor reload interval `50 -> 10`
- base heuristic opponent mix `1.0 -> 0.25`
- added B3/B4 variant opponent mix `0.35`
- B1 opponent mix `0.15 -> 0.25`, extended to update 400
- league warmup `first_updates: 200 -> 120`
- recent/champion pool sizes `24/4 -> 32/8`
- dev eval weights emphasize B1 and B3: B1 3.0, B3 3.0, B4 1.0, B0/B2 0.25

Smoke:
- `thesis_main_v2_anchor_variant_smoke2_20260429`
- 5 updates completed successfully after clearing stale shared memory.
- Early global throughput around 5.7k samples/sec; update rate around 0.19-0.24 updates/sec.

Online dev-eval:
- u40 weighted aggregate 0.5114583333333333
- u40 anchors: B0 1.0, B1 0.34375, B2 0.96875, B3 0.5, B4 0.8125
- u80 weighted aggregate 0.525
- u80 anchors: B0 1.0, B1 0.375, B2 1.0, B3 0.5, B4 0.8125
- u120 weighted aggregate 0.5958333333333333
- u120 anchors: B0 1.0, B1 0.4375, B2 1.0, B3 0.625, B4 0.78125

Interpretation:
- v2 is already healthier than v1 on the thesis-critical surfaces at u40.
- Weighted aggregate is intentionally strict because B1/B3 carry most of the weight.
- u80 is not a breakout, but it avoids the v1 collapse and slightly improves B1.
- u120 is the first strong signal: B1/B3 both improved, and PFSP became active shortly after.

Stop/resume note:
- v2 stopped after checkpoint 140 due to a promotion-gate config guard, not a training failure.
- Error: `promotion.paired_seeds` was 16 while `configs/seeds/promotion_eval_seeds.txt` contains 64 paired seeds.
- Fixed `configs/main_impala_league_server_v2_anchor_variant.yaml` to use `league.promotion.paired_seeds: 64`.
- Because that changes the config hash, continued from `checkpoint_140.pt` into a new run label.

Resume run label: `thesis_main_candidate_v2b_anchor_variant_cf81_resume140_20260429`

Exact resume command:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_main_candidate_v2b_anchor_variant_cf81_resume140_20260429 \
  --stack-config configs/main_impala_league_server_v2_anchor_variant.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 240 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30 \
  --train-arg=--b1-baseline-run-dir \
  --train-arg=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429 \
  --train-arg=--resume-from \
  --train-arg=runs/thesis_main_candidate_v2_anchor_variant_cf81_20260429/training/checkpoints/checkpoint_140.pt \
  --train-arg=--resume-allow-config-mismatch
```

v2b online dev-eval:
- u160 weighted aggregate 0.5916666666666667
- u160 anchors: B0 1.0, B1 0.4375, B2 1.0, B3 0.5625, B4 0.9375
- u200 weighted aggregate 0.5229166666666667
- u200 anchors: B0 1.0, B1 0.3125, B2 0.9375, B3 0.53125, B4 0.90625

Interpretation:
- u160 is essentially flat from u120 on the weighted target, but it is stable rather than collapsing.
- B1 held at 14/32; B3 dipped from 20/32 to 18/32; B4 rose from 25/32 to 30/32.
- u200 regressed, so stop the v2b continuation. Newly active PFSP did not improve the hard anchors under this setting.
- Running 64-pair confirmatory dev eval for v2 u120 before deciding whether this is good enough or needs another targeted tune.

Confirmatory eval:

```bash
cd /workspace/weiss_schwarz_rl
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
.venv/bin/python python/scripts/manual_dev_eval_confirm.py \
  --stack-config configs/main_impala_league_server_v2_anchor_variant.yaml \
  --run-dir runs/thesis_main_candidate_v2_anchor_variant_cf81_20260429 \
  --checkpoint runs/thesis_main_candidate_v2_anchor_variant_cf81_20260429/training/checkpoints/checkpoint_120.pt \
  --summary runs/thesis_main_candidate_v2_anchor_variant_cf81_20260429/eval/dev_eval/update_120/summary.json \
  --update 120 \
  --pairs 64 \
  --workers 8 \
  --artifact-dir-name dev_eval_confirm64_u120_allgpu
```

Result:
- weighted aggregate 0.56224
- unweighted aggregate 0.760938
- B0 RandomLegal 1.0
- B1 NoLeague baseline 0.3984375
- B2 HeuristicPublic 0.9921875
- B3 HeuristicPublicAggro 0.5546875
- B4 HeuristicPublicControl 0.859375

Interpretation:
- Better than v1 and not collapsed, but not strong enough for the main thesis model.
- Main weakness is B1 and B3. More generic PFSP/longer training did not fix this.
- Next run should be a targeted B1/B3 repair, not bigger model size.

## Main League v3 B1/B3 Repair

Run label: `thesis_main_candidate_v3_b1_b3_repair_cf81_20260429`

Goal:
- Fast iteration, not a long blind run.
- Repair the two weak surfaces from v2: B1 no-league anchor and B3 aggro heuristic.
- Preserve some self-play via warmup/recent snapshots and small champion/hard-negative lanes.

Key changes versus v2:
- B1 opponent mix `0.25 -> 0.35`
- B3/B4 variant mix held high at `0.35`
- base heuristic mix reduced `0.25 -> 0.10`
- added `warmup_snapshot_mix_fraction: 0.15`
- champion/hard-negative mixes reduced to `0.05` each
- league warmup `120 -> 80`
- entropy `0.02 -> 0.025`
- added light/fading B1 reference top-action BC:
  - `reference_policy_id: b1_noleague_baseline`
  - `reference_policy_top_action_bc_coef: 0.16`
  - `reference_policy_top_action_bc_final_coef: 0.04`
  - `reference_policy_top_action_bc_end_updates: 140`
- periodic eval workers `6 -> 8`

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v3_b1_b3_repair_cf81_20260429 \
MAX_UPDATES=140 \
./scripts/run_thesis_main_v3_b1_b3_repair_20260429.sh
```

Plan:
- Watch u40; stop immediately if worse than v2.
- Continue to u80/u120 if B1/B3 improve.
- Only run 64-pair confirm if u120 is clearly promising.

Online dev-eval:
- u40 weighted aggregate 0.565625
- u40 anchors: B0 1.0, B1 0.40625, B2 0.96875, B3 0.5625, B4 0.84375
- u80 weighted aggregate 0.5625
- u80 anchors: B0 1.0, B1 0.375, B2 1.0, B3 0.5625, B4 0.90625
- u120 weighted aggregate 0.6083333333333333
- u120 anchors: B0 1.0, B1 0.46875, B2 1.0, B3 0.59375, B4 0.875

Interpretation:
- Better than v2 u40 on the weighted target and on the two weak anchors.
- u80 is flat/slightly worse on B1, but B3 holds and B4 improves.
- PFSP/recent snapshots became active around u100, so continue to u120 as the next stop/check point.
- u120 is the best 16-pair main-model result so far, especially on B1.
- Stopped further training and started all-GPU 64-pair confirm for u120.

Confirmatory eval:
- weighted aggregate 0.573958
- unweighted aggregate 0.770312
- B0 RandomLegal 1.0
- B1 NoLeague baseline 0.4140625
- B2 HeuristicPublic 1.0
- B3 HeuristicPublicAggro 0.5625
- B4 HeuristicPublicControl 0.875

Interpretation:
- v3 confirms slightly better than v2, but only slightly.
- B1 improved from v2 confirm 0.3984375 to 0.4140625.
- B3 improved from v2 confirm 0.5546875 to 0.5625.
- This is useful signal, not thesis-final quality.
- Next strongest idea: initialize main league training from the locked B1 u120 anchor checkpoint instead of from scratch, reset optimizer, then apply the v3 B1/B3 repair league. This tests whether the issue is losing too much of the B1 anchor during fresh-student training.

## Main League v4 CF-Strong

Run label: `thesis_main_candidate_v4_cfstrong_cf82_20260429`

Reason:
- Tried a timeboxed extra80 mining pass, but it only found 1 new positive label after several minutes.
- Stopped mining rather than burning time.
- Use the clean thesis mechanism by strengthening/lengthening counterfactual supervision instead of B1 initialization.

Config:
- Starts from v3 B1/B3 repair mix.
- Counterfactual labels: 82 total, the original 81 plus one new attack/event label.
- CF coef `1.2 -> 0.2`, ending at update 220.
- Same v3 B1/B3 league mix.

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v4_cfstrong_cf82_20260429 \
MAX_UPDATES=140 \
./scripts/run_thesis_main_v4_extra_labels_20260429.sh
```

Plan:
- Compare u40/u80/u120 directly against v3.
- Stop if B1/B3 do not improve.

Online dev-eval:
- u40 weighted aggregate 0.565625
- u40 anchors: B0 1.0, B1 0.34375, B2 0.96875, B3 0.625, B4 0.84375
- u80 weighted aggregate 0.5541666666666667
- u80 anchors: B0 1.0, B1 0.40625, B2 1.0, B3 0.5, B4 0.9375

Interpretation:
- Same weighted score as v3 u40.
- B3 improved, but B1 regressed.
- Continue to u80 only because v3's B1 recovered later; stop if B1 does not recover.
- u80: B1 recovered and beats v3 u80, but B3 regressed.
- Continue to u120 because PFSP/recent snapshots are active and it is already close; v4 must beat v3 u120 to be worth confirming.
- u120 weighted aggregate 0.5958333333333333
- u120 anchors: B0 1.0, B1 0.375, B2 1.0, B3 0.65625, B4 0.875

Interpretation update:
- v4 improves B3 substantially but loses B1.
- Weighted score is below v3 u120, and B1 is too weak for main-candidate status.
- Stopped v4 without confirmatory eval.
- Current best main candidate remains v3 u120.

Interim result:

- Stopped manually around update 200 because the same post-u40 collapse appeared.
- u40: aggregate 0.8625, B1 0.53125, B2 1.0, B3 0.78125, B4 1.0
- u80: aggregate 0.7375, B1 0.5, B2 0.96875, B3 0.46875, B4 0.75
- u120: aggregate 0.6875, B1 0.5625, B2 0.875, B3 0.40625, B4 0.59375
- Interpretation: profile cycling fixed the aggro exposure problem at u40 but did not stop later policy drift.

## B1 Thesis Candidate v4

Relaunching with conservative stabilization:

- Run label: `thesis_b1_candidate_v4_profiles_cycle_bc01_lowlr_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 240
- Seed: 20260429
- tmux session: `thesis_b1_v4_20260429`
- Profile log: `notes/thesis_b1_candidate_v4_profiles_cycle_bc01_lowlr_20260429_profile.log`
- Changes from v3b:
  - Keep heuristic-native rollout profile cycling across `base`, `aggressive`, `control`.
  - Add `training.behavior_action_bc_coef=0.1` to resist policy drift from heuristic rollouts.
  - Lower `training.optimizer.learning_rate` from `0.0002` to `0.0001`.
  - Disable checkpoint guard for diagnostic visibility.

Exact launch command:

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
.venv/bin/python python/scripts/profile_train_job.py \
  --run-label thesis_b1_candidate_v4_profiles_cycle_bc01_lowlr_20260429 \
  --stack-config configs/baselines/noleague_impala.yaml \
  --seed 20260429 \
  --runtime-mode train_async_fast \
  --max-updates 240 \
  --autoscale \
  --hardware-profile local \
  --torchrun-nproc 4 \
  --ddp-backend gloo \
  --ddp-timeout-seconds 1800 \
  --sample-interval-seconds 30 \
  --override 'training.heuristic_native_rollout_profiles=["base","aggressive","control"]' \
  --override 'training.heuristic_native_rollout_profile_mode="cycle"' \
  --override 'training.behavior_action_bc_coef=0.1' \
  --override 'training.optimizer.learning_rate=0.0001' \
  --override 'curriculum.checkpoint_guard.enabled=false'
```

Interim result:

- Stopped manually around update 200 because the curve still drifted down after u120.
- u40: aggregate 0.85, B1 0.5, B2 1.0, B3 0.75, B4 1.0
- u80: aggregate 0.80625, B1 0.4375, B2 1.0, B3 0.625, B4 0.96875
- u120: aggregate 0.78125, B1 0.46875, B2 0.9375, B3 0.65625, B4 0.84375
- u160: aggregate 0.73125, B1 0.46875, B2 1.0, B3 0.4375, B4 0.75
- Interpretation: LR/BC stabilization helped relative to v3b after u40, but B1 mirror remained weak and B3 still drifted.

## B1 Thesis Candidate v5

Relaunching as an aggressive-teacher specialist/stability test based on the older warmup baseline shape.

- Run label: `thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 160
- Seed: 20260429
- tmux session: `thesis_b1_v5_20260429`
- Profile log: `notes/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429_profile.log`

Exact launch command:

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

Completion:

- Trained through update 160.
- One DDP rank hung after writing checkpoints/evals; killed leftover process after artifacts were present.
- Best online dev-eval checkpoint: `training/checkpoints/checkpoint_120.pt`
- `training/checkpoints/best.pt` points to update 120.
- `training/checkpoints/latest.pt` points to update 160.

Online dev-eval:

- u40: aggregate 0.81875, B1 0.46875, B2 1.0, B3 0.625, B4 1.0
- u80: aggregate 0.8625, B1 0.5, B2 1.0, B3 0.8125, B4 1.0
- u120: aggregate 0.86875, B1 0.5, B2 1.0, B3 0.84375, B4 1.0
- u160: aggregate 0.8625, B1 0.5, B2 1.0, B3 0.8125, B4 1.0

Confirmatory eval:

```bash
cd /workspace/weiss_schwarz_rl
.venv/bin/python python/scripts/manual_dev_eval_confirm.py \
  --stack-config configs/baselines/noleague_impala.yaml \
  --run-dir runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429 \
  --checkpoint runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt \
  --summary runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/dev_eval/update_120/summary.json \
  --update 120 \
  --pairs 64 \
  --workers 6 \
  --artifact-dir-name dev_eval_confirm64_u120
```

Confirmatory result:

- aggregate 0.875
- B0 RandomLegal: 1.0
- B1 NoLeague baseline: 0.5078125
- B2 HeuristicPublic: 1.0
- B3 HeuristicPublicAggro: 0.875
- B4 HeuristicPublicControl: 0.9921875
- Artifact dir: `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/dev_eval_confirm64_u120/update_120`

Interpretation:

- This is the first B1 candidate whose best point is not an early u40 blip.
- Aggressive-teacher + BC/lower-LR fixed the B3 failure mode and held from u80 through u160.
- B1 mirror is only slightly above 0.5 on 64 paired seeds, so call it a defensible no-league anchor, not a dominant self-play baseline.

Startup result:

- This label failed before training with a `FileExistsError` opening a stale `weissrl_*` shared-memory segment.
- No updates were produced; treat it as a failed launch, not a candidate.

## B1 Thesis Candidate v3b

Relaunching the same experiment under a fresh label after confirming no training process is alive and no `weissrl_*` shared-memory segments remain.

- Run label: `thesis_b1_candidate_v3b_profiles_cycle_noguard_20260429`
- Config: `configs/baselines/noleague_impala.yaml`
- Max updates: 400
- Seed: 20260429
- tmux session: `thesis_b1_v3b_20260429`
- Profile log: `notes/thesis_b1_candidate_v3b_profiles_cycle_noguard_20260429_profile.log`

Exact launch command:

```bash
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
```

## Main Thesis Long-Frontier v5

Launched:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v5_long_frontier_es_cf81_20260429 \
MAX_UPDATES=800 \
scripts/run_thesis_main_v5_long_frontier_20260429.sh
```

Purpose: allow training to u800 without merely extending the short v3 repair setup. The curriculum keeps B1/heuristic anchors as a floor, but increases late pressure from recent/champion/hard-negative opponents.

Changes from v3:

- Model remains locked at 512/512 GRU.
- LR lowered from 5e-5 to 4e-5 for longer-run stability.
- Entropy changes from fixed-ish 0.025 to 0.02 -> 0.012 over 800 updates.
- B1 top-action BC fades 0.16 -> 0.03 over 500 updates instead of ending at u140.
- Counterfactual-positive labels fade 1.0 -> 0.05 over 500 updates instead of u160.
- Recent pool stays 32, champion pool 8 -> 12, champion max age 120 -> 800.
- Base heuristic opponent fades 0.10 -> 0.05 by u500.
- Heuristic variant opponents fade 0.35 -> 0.20 by u500.
- B1 anchor opponent stays persistent at 0.25 (`noleague_baseline_mix_end_updates: -1`).
- Champion mix is 0.20 and hard-negative mix is 0.10, leaving roughly 0.20 late recent-snapshot pressure.
- Periodic dev eval interval widened to 80 updates to reduce eval overhead.
- Early cutoff enabled conservatively: warmup u240, patience 240 updates, min improvement 0.005, stall patience 4 evals.

Monitoring plan before trusting u800:

- u80: must still beat/roughly match v3 early anchor behavior.
- u160/u240: B1 should not collapse; B3/B4 should be healthy.
- After the run has snapshots/champions, manually confirm against recent/champion snapshots using `manual_dev_eval_confirm.py --extra-snapshot-anchor`.

Live results:

- u80 periodic dev eval:
  - aggregate 0.5875
  - B0 1.0
  - B1 0.46875
  - B2 1.0
  - B3 0.53125
  - B4 0.90625
- u120/u140 promotion gates failed with `anchor_loss_guardrail_exceeded`.
- u120 league sampling:
  - B1 anchor 0.25
  - heuristic base about 0.089-0.093
  - heuristic variants about 0.317-0.329
  - recent about 0.344-0.352
  - champion 0.0
  - hard negative 0.0
- u160 periodic dev eval:
  - aggregate 0.5958333333333333
  - B0 1.0
  - B1 0.4375
  - B2 1.0
  - B3 0.5625
  - B4 0.96875
- u160 promotion gate also failed with `anchor_loss_guardrail_exceeded`.
- u180/u200/u220/u240 promotion gates also failed with `anchor_loss_guardrail_exceeded`.
- u240 periodic dev eval:
  - aggregate 0.58125
  - B0 1.0
  - B1 0.4375
  - B2 0.9375
  - B3 0.5625
  - B4 0.875
- Stopped manually after u240/u260 window to save compute.

Interpretation:

- v5 is not collapsing; it improved aggregate and B3/B4 from u80 to u160.
- B1 retention slipped from 0.46875 to 0.4375, so anchor retention remains the gating issue.
- Recent snapshot pressure is active, but champion pool is empty because promotion is correctly rejecting snapshots that do not clear anchor guardrails.
- u240 did not justify continuing to u320: aggregate dropped versus u160, B4 regressed, and no champions/hard negatives were created.
- Do not switch to an 800-update warmup; that would delay the recent-pressure phase that is already useful. If extending beyond u800, prefer a longer curriculum with persistent B1/heuristic rails and slower BC/CF fade, not delayed league activation.

## Main Thesis Anchor-Rails v6

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v6_anchor_rails_cf81_20260429 \
MAX_UPDATES=400 \
scripts/run_thesis_main_v6_anchor_rails_20260429.sh
```

Purpose: keep recent-snapshot pressure active, but strengthen B1/heuristic rails so snapshots can clear promotion instead of repeatedly failing `anchor_loss_guardrail_exceeded`.

Changes from v5:

- LR 4e-5 -> 3.5e-5.
- Entropy 0.018 -> 0.010 over 800 updates.
- B1 top-action BC 0.22 -> 0.08 through u800.
- Counterfactual labels 1.0 -> 0.10 through u800.
- B1 opponent mix 0.25 -> 0.35, persistent.
- Base heuristic 0.08 -> 0.06 by u600.
- Heuristic variants 0.35 -> 0.26 by u600.
- Champion mix 0.20 -> 0.10.
- Hard-negative mix 0.10 -> 0.05.

Expected late league shape before champions exist:

- B1 anchor about 35%.
- Base heuristic about 6%.
- Heuristic variants about 26%.
- Recent snapshots about 33%.

Expected late league shape after champions/hard negatives exist:

- B1 anchor about 35%.
- Base heuristic about 6%.
- Heuristic variants about 26%.
- Champions about 10%.
- Hard negatives about 5%.
- Recent snapshots about 18%.

Decision rule:

- u80 should retain B1 better than v5 while not losing B3/B4 too badly.
- u160/u240 should beat v5 on B1 and ideally create at least one promoted champion.
- If promotion still fails through u240 and aggregate is not better than v5/v3, stop and move to ablations instead of chasing.

Live result:

- u80 periodic dev eval:
  - aggregate 0.6041666666666666
  - B0 1.0
  - B1 0.46875
  - B2 1.0
  - B3 0.5625
  - B4 0.9375
- Recent sampling turned on after warmup at about 24-25%, with B1 anchor 35%.
- u120/u140/u160 promotion gates failed with `anchor_loss_guardrail_exceeded`.
- u160 periodic dev eval:
  - aggregate 0.5791666666666667
  - B0 1.0
  - B1 0.40625
  - B2 1.0
  - B3 0.5625
  - B4 0.9375
- Stopped manually after u160 result.

Interpretation:

- v6 improved the u80 point versus v5, but did not survive the post-warmup recent-snapshot phase.
- Stronger B1 rails did not prevent B1 regression by u160, and still did not produce champions.
- Current best confirmed main candidate remains v3 u120; v5/v6 are useful negative tuning evidence but should not be the main run.

## Main Thesis Bootstrap-Champions v7

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v7_bootstrap_champions_cf81_20260429 \
MAX_UPDATES=400 \
scripts/run_thesis_main_v7_bootstrap_champions_20260429.sh
```

Purpose: fix the broken champion story by separating training-time champion admission from final strict evaluation. v5/v6 did fight recent snapshots, but promotion rejected all snapshots because the B1 posterior was slightly/clearly below the strict 0.45 guardrail. v7 keeps the final eval strict but allows near-anchor snapshots into the champion pool so the league can bootstrap.

Changes from v6:

- Training-time promotion guardrail relaxed:
  - `max_prob_anchor_loss_below_0_45: 0.05 -> 0.80`
- Added explicit promotion floors:
  - B1 NoLeague baseline >= 0.42
  - B3 HeuristicPublicAggro >= 0.55
  - B4 HeuristicPublicControl >= 0.80

Rationale:

- v5/v6 candidates often beat B2/B3/B4 strongly but were rejected solely because B1 was near/slightly under 0.45.
- v7 should promote snapshots like v6 u120/u140 while still rejecting bad B1 collapses like v6 u160.
- These are training champions, not final claims. Final tables should still use strict independent eval.

Result:

- v7 was stopped because the relaxed champion story was not thesis-preferred.
- It did mechanically prove the diagnosis: relaxed gate produced champions at u120/u140.
- But u80 quality was weak:
  - aggregate 0.5583333333333333
  - B1 0.40625
  - B3 0.5
  - B4 0.96875

## Main Thesis Strict Anchor-Constrained v8

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v8_strict_anchor_constrained_cf81_20260429 \
MAX_UPDATES=500 \
scripts/run_thesis_main_v8_strict_anchor_constrained_20260429.sh
```

Purpose: keep a clean thesis story with one strict/safe champion pool, while adding targeted B1-preservation pressure so strict champions can actually appear.

Changes from v6/v7:

- Strict promotion gate restored:
  - `max_prob_anchor_loss_below_0_45: 0.05`
- Added safe-champion target floors:
  - B1 NoLeague baseline >= 0.48
  - B3 HeuristicPublicAggro >= 0.55
  - B4 HeuristicPublicControl >= 0.80
- B1 opponent mix reduced from 0.35 to 0.30, but B1 reward scale added:
  - `noleague_baseline_reward_scale: 1.5`
  - effective B1 pressure about 0.45, not the 0.80-heavy version.
- Heuristic variants capped lower:
  - 0.25 -> 0.18 instead of 0.35 -> 0.26.
- Champion mix raised to 0.15 once safe champions exist.
- Warmup extended moderately:
  - first_updates 100 -> 140.
- Lower LR:
  - 3.5e-5 -> 3.0e-5.
- Stronger permanent B1/CF rails:
  - B1 BC 0.24 -> 0.14 through u1000.
  - CF 1.0 -> 0.25 through u1000.

Decision rule:

- u80 should not be much worse than v6 u80.
- u160/u180 should produce a strict/safe champion or at least clear B1 near 0.48.
- If strict promotion still fails through u220/u240, stop and treat anchor-constrained champions as not reachable in remaining time.

Result:

- Run stopped at u260 after u240 eval/gate results; no strict champions were admitted.
- GPU processes were stopped and GPUs were freed.
- Dev evals:
  - u80 aggregate 0.5625; B1 0.40625, B3 0.53125, B4 0.90625.
  - u160 aggregate 0.5791666666666667; B1 0.46875, B3 0.5, B4 0.9375.
  - u240 aggregate 0.5666666666666667; B1 0.375, B3 0.59375, B4 0.84375.
- Promotion gates:
  - u160 failed: B1 0.390625, B3 0.5625, B4 0.8671875.
  - u180 failed: B1 0.4453125, B3 0.5859375, B4 0.921875.
  - u200 failed: B1 0.453125, B3 0.5625, B4 0.9453125.
  - u220 failed: B1 0.421875, B3 0.5546875, B4 0.9296875.
  - u240 failed: B1 0.40625, B3 0.6328125, B4 0.9453125.
- Interpretation:
  - v8 improved B3/B4 and briefly approached the B1 floor at u180/u200, but did not hold B1 strongly enough for strict safe-champion promotion.
  - The strict no-relaxed-champion story is currently not bootstrapping a champion pool. The strongest confirmed main candidate remains v3 u120; v8 is useful negative evidence that targeted B1 rails plus lower variant pressure are not sufficient by themselves.

## Main Thesis Anchor-Push v9

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v9_anchor_push_cf81_20260429 \
MAX_UPDATES=320 \
scripts/run_thesis_main_v9_anchor_push_20260429.sh
```

Purpose: quick, stronger-signal test of whether strict safe champions are reachable if B1 preservation is pushed harder instead of making another small v8-style nudge.

Changes from v8:

- Strict promotion gate unchanged:
  - `max_prob_anchor_loss_below_0_45: 0.05`
  - B1 target floor remains 0.48.
- B1 direct pressure increased:
  - `noleague_baseline_mix_fraction: 0.30 -> 0.42`
  - `noleague_baseline_reward_scale: 1.5 -> 1.75`
- Heuristic pressure reduced:
  - base heuristic `0.08 -> 0.05`, final `0.05 -> 0.03`
  - variant heuristic `0.25 -> 0.16`, final `0.18 -> 0.10`
- Recents/champions/hard negatives lower until a safe champion exists:
  - warmup snapshot `0.15 -> 0.05`
  - champion `0.15 -> 0.10`
  - hard negative `0.05 -> 0.03`
- B1/label auxiliary rails stronger:
  - B1 BC `0.24 -> 0.14` becomes `0.30 -> 0.20`
  - CF `1.0 -> 0.25` becomes `1.2 -> 0.35`
- Slightly more conservative optimizer/exploration:
  - LR `3.0e-5 -> 2.5e-5`
  - entropy `0.018 -> 0.010` becomes `0.016 -> 0.008`
- Warmup slightly extended:
  - `first_updates: 140 -> 160`

Decision rule:

- If B1 still does not cross or stay very near the strict gate by u220/u240, config pressure alone is probably not enough.
- If B1 crosses but B3/B4 collapse, the tradeoff is too blunt and v9 is diagnostic rather than final.
- If a strict champion appears while B3/B4 remain above floors, continue to u320 and consider extending.

Result:

- v9 was stopped by checkpoint guard at u160 before the first real promotion attempt.
- u80 dev eval showed the B1 pressure moved the needle:
  - aggregate 0.5958333333333333
  - B1 0.5
  - B3 0.53125
  - B4 0.875
- u160 dev eval dipped:
  - aggregate 0.5739583333333333
  - B1 0.46875
  - B3 0.5
  - B4 0.90625
- The run stopped because the checkpoint guard compared u160 to u80 and rolled back before u180. That means v9 did not answer whether relaxed training admission can create champions.

## Main Thesis Anchor-Push Relaxed Gate v9b

Prepared for launch as a continuation from v9 u160:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v9b_anchor_push_relaxed_gate_cf81_20260429 \
MAX_UPDATES=320 \
RESUME_FROM=runs/thesis_main_candidate_v9_anchor_push_cf81_20260429/training/checkpoints/checkpoint_160.pt \
scripts/run_thesis_main_v9b_anchor_push_relaxed_gate_20260429.sh
```

Purpose: continue the promising v9 trajectory with a mildly relaxed training-time promotion gate and checkpoint guard disabled, so the run can reach u180/u200 promotion attempts.

Changes from v9:

- Checkpoint guard disabled.
- Promotion gate mildly relaxed for training admission:
  - B1 floor `0.48 -> 0.46`
  - B3 floor `0.55 -> 0.53`
  - B4 floor unchanged at `0.80`
  - `max_prob_anchor_loss_below_0_45: 0.05 -> 0.25`
- B1/CF pressure and opponent mix unchanged from v9.

Decision rule:

- If v9b admits a champion around u180/u220 and B1 remains near/above 0.46 with B3/B4 intact, continue.
- If it admits weak B1 snapshots or collapses B3/B4, stop and treat the relaxation as too permissive or the B1 pressure as too blunt.

Result:

- First v9b resume attempt failed immediately due stale `/dev/shm/weissrl_*` shared-memory segments from interrupted runs.
- Relaunched as v9b2 after clearing stale shared memory.
- v9b2 reached u180, but actual recent sampling weight was about 0.393, which violated the intended "delay recent pressure" idea. Stopped v9b2 before treating it as a clean result.

## Main Thesis Anchor-Push Relaxed Gate Delay-Recent v9c

Prepared for launch as a cleaner continuation from v9 u160:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v9c_anchor_push_relaxed_gate_delay_recent_cf81_20260429 \
MAX_UPDATES=320 \
RESUME_FROM=runs/thesis_main_candidate_v9_anchor_push_cf81_20260429/training/checkpoints/checkpoint_160.pt \
scripts/run_thesis_main_v9c_anchor_push_relaxed_gate_delay_recent_20260429.sh
```

Purpose: same B1-push and relaxed training gate as v9b, but with recent pressure genuinely delayed past u200.

Changes from v9b:

- `warmup.first_updates: 160 -> 220`
- `warmup_snapshot_mix_fraction: 0.05 -> 0.0`

Expected behavior:

- From resumed u160 through roughly u220, B1 pressure remains high and recent pressure should stay at zero.
- First real promotion gate should occur after warmup, around u220/u240 depending on effective update lag.

Result:

- v9c was stopped manually at u320 because the final u320 gate/eval path appeared stalled; useful artifacts through u300 were saved.
- GPUs were freed afterward.
- Sanity check:
  - At u183, recent pressure was actually zero:
    - `pfsp_sampling_weight_recent: 0.0`
    - `noleague_baseline_mix_fraction: 0.42`
    - `noleague_baseline_reward_scale_active: 1.75`
  - At u240, recent pressure had turned on after warmup, as intended:
    - `pfsp_sampling_weight_recent: ~0.401`
- Promotion / champion result:
  - u240 passed; champion pool became 1.
    - B1 0.4765625, prob below 0.45 = 0.037
    - B3 0.6171875
    - B4 0.9296875
  - u260 failed.
    - B1 0.4375, prob below 0.45 = 0.733
    - B3 0.5859375
    - B4 0.9140625
  - u280 failed.
    - B1 0.3984375, prob below 0.45 = 0.992
    - B3 0.6171875
    - B4 0.96875
  - u300 passed; champion pool became 2.
    - B1 0.4609375, prob below 0.45 = 0.235
    - B3 0.609375
    - B4 0.9375
- Periodic dev eval:
  - u240 aggregate 0.5625
  - B1 0.40625
  - B3 0.5625
  - B4 0.8125
- Interpretation:
  - Delaying recent pressure plus mildly relaxing the training admission gate successfully restored the league/champion mechanism.
  - The champion story is now viable mechanically: v9c created two promoted champions under a documented training gate.
  - However, B1 remains unstable. The promotion samples passed at u240/u300, but the cheaper u240 dev eval had weak B1. Before using v9c as the main thesis run, run larger confirmation evals on u240 and u300, and compare against v3 u120.

## Main Thesis B1-Plus Seeded-Champions v10

Prepared for launch as a continuation from v9c u300:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v10_b1plus_seeded_champions_cf81_20260429 \
MAX_UPDATES=460 \
RESUME_FROM=runs/thesis_main_candidate_v9c_anchor_push_relaxed_gate_delay_recent_cf81_20260429/training/checkpoints/checkpoint_300.pt \
scripts/run_thesis_main_v10_b1plus_seeded_champions_20260429.sh
```

Purpose: keep the two v9c promoted champions in the league while increasing B1 influence enough to test whether the post-promotion B1 drift can be stabilized.

Source champions from v9c:

- `policy_000012` from u240
- `policy_000015` from u300

Important implementation detail:

- Resuming from v9c `checkpoint_300.pt` imports the prior league snapshot pool and preserves champion status, unlike generic seed-snapshot import which copies snapshots but does not mark them as champions.

Changes from v9c:

- B1 direct pressure:
  - `noleague_baseline_mix_fraction: 0.42 -> 0.50`
  - `noleague_baseline_reward_scale: 1.75 -> 2.0`
- B1/label auxiliary rails:
  - B1 BC `0.30 -> 0.20` becomes `0.36 -> 0.26`
  - CF `1.2 -> 0.35` becomes `1.35 -> 0.45`
- Keep delayed recent / relaxed training gate:
  - `warmup.first_updates: 220`
  - `warmup_snapshot_mix_fraction: 0.0`
  - B1 floor 0.46, B3 floor 0.53, B4 floor 0.80
  - `max_prob_anchor_loss_below_0_45: 0.25`

Decision rule:

- If v10 keeps producing promoted snapshots with B1 at or above about 0.46 while B3/B4 remain strong, continue/extend.
- If B3/B4 collapse, B1 pressure is too high.
- If B1 still drifts below the gate after seeded champions, config-only B1 pressure may not be enough and we should confirm-eval v9c/v3 rather than chase more.

Partial result:

- v10 seeded continuation was stopped after u320/u340 looked B1-weak despite stronger B1 pressure.
- u320 promotion failed:
  - B1 0.4140625, prob below 0.45 = 0.95
  - B3 0.640625
  - B4 0.921875
- u320 dev eval:
  - aggregate 0.555921052631579
  - B1 0.40625
  - B3 0.5625
  - B4 0.9375
- Interpretation:
  - Resuming from v9c u300 may inherit a B1-drifted policy. Stronger B1 pressure did not quickly repair that drift.
  - Next test should be a fresh v10-style run from update 0.

## Main Thesis Fresh B1-Plus v10fresh

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v10fresh_b1plus_cf81_20260429 \
MAX_UPDATES=400 \
scripts/run_thesis_main_v10fresh_b1plus_20260429.sh
```

Purpose: clean test of the v10 B1-plus recipe without inheriting the B1-drifted v9c/v10 continuation state.

Settings:

- No resume.
- No seeded champions.
- B1 pressure:
  - `noleague_baseline_mix_fraction: 0.50`
  - `noleague_baseline_reward_scale: 2.0`
  - B1 BC `0.36 -> 0.26`
  - CF `1.35 -> 0.45`
- Delayed recent pressure:
  - `warmup.first_updates: 220`
  - `warmup_snapshot_mix_fraction: 0.0`
- Relaxed training gate unchanged:
  - B1 floor 0.46
  - B3 floor 0.53
  - B4 floor 0.80
  - `max_prob_anchor_loss_below_0_45: 0.25`

Result:

- Run stopped manually after u300 promotion artifact; GPUs freed.
- Fresh start helped compared with the v10 seeded continuation: it produced a clean first champion at u240.
- Dev evals:
  - u80 aggregate 0.5958333333333333; B1 0.46875, B3 0.5625, B4 0.875.
  - u160 aggregate 0.5739583333333333; B1 0.46875, B3 0.5, B4 0.90625.
  - u240 aggregate 0.5875; B1 0.46875, B3 0.59375, B4 0.71875.
- Promotion gates:
  - u240 passed:
    - B1 0.46875, prob below 0.45 = 0.11
    - B3 0.6171875
    - B4 0.9140625
  - u260 failed:
    - B1 0.4296875, prob below 0.45 = 0.836
    - B3 0.578125
    - B4 0.9375
  - u280 failed:
    - B1 0.4140625, prob below 0.45 = 0.955
    - B3 0.6015625
    - B4 0.9609375
  - u300 failed, but close:
    - B1 0.453125, prob below 0.45 = 0.394
    - B3 0.609375
    - B4 0.953125
- Interpretation:
  - Fresh v10 confirms that stronger B1 pressure from update 0 can create a better first champion than the seeded continuation.
  - Post-champion drift remains: after u240, B1 drops while B3/B4 stay strong.
  - Best artifact from this branch is `thesis_main_candidate_v10fresh_b1plus_cf81_20260429` checkpoint u240 / policy_000012. Next step should be larger confirm evals on v10fresh u240, v9c u240/u300, and v3 u120.

## Main Thesis Low-Damage B1-Plus v11

Prepared for launch:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v11_lowdamage_b1plus_cf81_20260429 \
MAX_UPDATES=400 \
scripts/run_thesis_main_v11_lowdamage_b1plus_20260429.sh
```

Purpose: test whether damage shaping / aggression pressure causes the post-champion B1 drift.

Changes from v10fresh:

- Damage shaping reduced:
  - `damage_reward: 0.05 -> 0.015`
- Champion pressure reduced after first champion:
  - `champion_mix_fraction: 0.10 -> 0.05`
- Kept B1-heavy settings:
  - `noleague_baseline_mix_fraction: 0.50`
  - `noleague_baseline_reward_scale: 2.0`
  - B1 BC `0.36 -> 0.26`
  - CF `1.35 -> 0.45`
  - warmup 220 and `warmup_snapshot_mix_fraction: 0.0`

Decision rule:

- If u240 promotes and u260/u280 no longer collapse on B1, reward shaping was probably part of the drift.
- If u240 fails because B3/B4 are too weak, damage shaping was helping anchor acquisition too much.
- If u240 passes but u260/u280 still lose B1, stop chasing reward tweaks and move to confirm evals.

Partial result:

- v11 was stopped early after Pro's diagnosis suggested a higher-value missing experiment.
- u80 dev eval was similar to v10fresh, not an obvious breakthrough:
  - aggregate 0.5916666666666667
  - B1 0.46875
  - B3 0.5625
  - B4 0.84375

## Main Thesis B1-Initialized Constrained Fine-Tune v12

Prepared for launch from B1 v5 u120 with optimizer reset:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v12_b1init_constrained_cf81_20260429 \
MAX_UPDATES=360 \
RESUME_FROM=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt \
RESUME_RESET_OPTIMIZER=1 \
scripts/run_thesis_main_v12_b1init_constrained_20260429.sh
```

Purpose: test Pro's main hypothesis that the best policy family already exists in B1 v5, so the league should fine-tune from that basin instead of rediscovering it from scratch.

Settings:

- Resume model weights from B1 v5 u120.
- Reset optimizer.
- Lower fine-tune LR:
  - `learning_rate: 2.0e-5`
- Lower exploration:
  - entropy `0.012 -> 0.006`
- Gentler B1 pressure than v10fresh:
  - B1 mix 0.35
  - B1 reward scale 1.25
  - B1 BC `0.20 -> 0.12`
  - CF `0.75 -> 0.25`
- Heuristic/league pressure:
  - base heuristic 0.05
  - B3/B4 variants 0.22 -> 0.15
  - recents delayed with warmup 220 and `warmup_snapshot_mix_fraction: 0.0`
  - champion mix 0.05 once champions exist
  - hard negative 0.03
- Damage shaping reduced but not removed:
  - `damage_reward: 0.025`

Decision rule:

- If v12 preserves B1 near/above the B1 anchor while maintaining B3/B4 strength, this becomes the main thesis candidate path.
- If v12 immediately damages B1, the league machinery/optimizer is perturbing the B1 basin too much and we should stop training and focus on confirm/eval story.

Result:

- Run stopped manually after u300 confirmed post-champion B1 drift; GPUs freed.
- v12 is the strongest thesis-shaped main run so far.
- It validates the B1-initialized fine-tuning hypothesis: starting from B1 v5 preserved much more B3/B4 strength while keeping B1 near the relaxed floor.
- Dev eval:
  - u160 aggregate 0.6875
    - B1 0.46875
    - B3 0.75
    - B4 1.0
  - u240 aggregate 0.6083333333333333
    - B1 0.40625
    - B3 0.625
    - B4 0.96875
- Promotion gates:
  - u240 failed only barely on uncertainty:
    - B1 0.4609375, prob below 0.45 = 0.255, max allowed 0.25
    - B3 0.7421875
    - B4 0.9921875
  - u260 passed:
    - B1 0.4765625, prob below 0.45 = 0.047
    - B3 0.6484375
    - B4 0.9765625
  - u280 failed:
    - B1 0.4375, prob below 0.45 = 0.698
    - B3 0.6953125
    - B4 0.984375
  - u300 failed:
    - B1 0.4296875, prob below 0.45 = 0.828
    - B3 0.671875
    - B4 0.96875
- Interpretation:
  - Best artifact: `thesis_main_candidate_v12_b1init_constrained_cf81_20260429` checkpoint u260 / policy_000013.
  - v12 u260 is likely the best current main thesis candidate: B1 is safer than v9/v10 promoted checkpoints, and B3/B4 are much stronger.
  - Post-champion drift still appears after u260, so do not extend blindly. Next step should be larger confirm evals on v12 u260, v12 u240, v10fresh u240, v9c u240/u300, v3 u120, and B1 v5 u120.

## Main Thesis Long B1-Initialized Variant Fine-Tune v13

Prepared for launch from B1 v5 u120 with optimizer reset:

```bash
cd /workspace/weiss_schwarz_rl
RUN_LABEL=thesis_main_candidate_v13_b1init_long_variant_cf81_20260430 \
MAX_UPDATES=560 \
RESUME_FROM=runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt \
RESUME_RESET_OPTIMIZER=1 \
scripts/run_thesis_main_v13_b1init_long_variant_20260430.sh
```

Purpose: answer whether the B1-initialized model can improve with longer fine-tuning before self-play/recents/champions turn on.

Changes from v12:

- Lower fine-tune LR:
  - `2.0e-5 -> 1.5e-5`
- Lower entropy:
  - `0.012 -> 0.006` becomes `0.010 -> 0.004`
- Longer delayed self-play window:
  - `warmup.first_updates: 220 -> 380`
- No warmup snapshot pressure:
  - `warmup_snapshot_mix_fraction: 0.0`
- More B3/B4 variant pressure before recents:
  - variants `0.22 -> 0.15` becomes `0.30 -> 0.20`
- Gentler explicit B1 opponent scaling because we start in B1 basin:
  - B1 mix `0.35 -> 0.30`
  - B1 reward scale `1.25 -> 1.10`
- Softer post-warmup league pressure:
  - champion `0.05 -> 0.03`
  - hard negative `0.03 -> 0.02`

Decision rule:

- If longer pre-self-play fine-tuning improves B3/B4 while holding B1, v13 can become the main candidate path.
- If it still does not beat B1 no-league under common confirm eval, stop training and use B1 no-league as the strongest policy plus league variants as ablations/negative results.

Result:

- Run stopped manually at u378, before recents/champions could turn on, because u320 showed clear deterioration from u160/u240.
- GPUs were freed afterward.
- Dev evals:
  - u160 aggregate 0.725
    - B1 0.5
    - B3 0.8125
    - B4 1.0
  - u240 aggregate 0.7208333333333333
    - B1 0.46875
    - B3 0.84375
    - B4 0.96875
  - u320 aggregate 0.6083333333333333
    - B1 0.4375
    - B3 0.59375
    - B4 0.96875
- Interpretation:
  - v13 u160/u240 are very strong candidates for confirm eval.
  - Longer pre-self-play fine-tuning helped through u240, but going longer to u320 degraded both B1 and B3 before recents/champions even turned on.
  - Best artifacts from this run are likely checkpoint u160 and checkpoint u240, not later checkpoints.

## Diagnostics: B1 Alignment and Heuristic Bias

Ran B1 recurrent replay-alignment diagnostics for v13 u160/u240/u320 using the existing replay inspector with full prefix replay and hidden-state updates.

Commands used the v13 stack config and compared:

- `policy_000008` / v13 u160 vs `b1_noleague_baseline`
- `policy_000012` / v13 u240 vs `b1_noleague_baseline`
- `policy_000016` / v13 u320 vs `b1_noleague_baseline`

Artifacts:

- `runs/diagnostic_b1_alignment_v13_u160_20260430b`
- `runs/diagnostic_b1_alignment_v13_u240_20260430b`
- `runs/diagnostic_b1_alignment_v13_u320_20260430b`

Results:

- u160:
  - B1 matchup games: 16 W / 16 L on the audited 16-pair dev slice.
  - Mean total variation vs B1 teacher: 0.2423.
  - Top-action match vs B1: about 0.873.
  - Top-family match vs B1: about 0.892.
  - First high-difference mismatch in B1 losses was `B1 main_play_character -> main main_move` in 16/16 inspected losses.
- u240:
  - B1 matchup games: 15 W / 17 L.
  - Mean total variation vs B1 teacher: 0.2378.
  - Top-action match vs B1: about 0.919.
  - Top-family match vs B1: about 0.952.
  - First high-difference mismatch in B1 losses was mostly `B1 main_play_character -> main main_move` in 14/17 losses.
- u320:
  - B1 matchup games: 15 W / 17 L.
  - Mean total variation vs B1 teacher: 0.2504.
  - Top-action match vs B1: about 0.908.
  - Top-family match vs B1: about 0.995.
  - First high-difference mismatch in B1 losses shifted to same-family tactical choices, especially attack-slot/type choices.

Interpretation:

- The B1 auxiliary is not totally broken. The policy remains fairly close to B1 on recurrent trajectories, especially by u240/u320.
- But it is still not close enough where it matters. Early B1 losses often start from a simple strategic divergence: B1 wants to play a character, the main policy wants to rearrange/move instead.
- By u320 the model is no longer mostly making different families; it is making different tactical choices inside the same family, especially attacks. That fits the observed "looks close but still loses" behavior.
- This is not primarily a recents/champions problem in v13 because deterioration began before recents/champions activated.

Ran public heuristic logit bias on/off audit for v13 u240:

Artifacts:

- `runs/thesis_main_candidate_v13_b1init_long_variant_cf81_20260430/eval/diagnostic_bias_on_v13_u240_20260430`
- `runs/thesis_main_candidate_v13_b1init_long_variant_cf81_20260430/eval/diagnostic_bias_off_v13_u240_20260430`

Results over 32 paired seeds / 64 games:

- Bias ON:
  - vs B1: 0.453125
  - vs B3: 0.75
  - vs B4: 0.96875
- Bias OFF:
  - vs B1: 0.046875
  - vs B3: 0.0
  - vs B4: 0.0

Interpretation:

- The heuristic logit bias is not causing the B1 weakness. It is carrying the policy.
- The raw learned logits are much weaker than the final biased policy, so the model has not internalized the heuristic scaffold well enough.
- This gives a plausible explanation for why it "finds best" early and then stalls/degrades: the biased policy is strong externally, but the trainable core is still fragile and can drift under RL/variant pressure.

Ran the same bias on/off control for the B1 v5 u120 anchor:

Artifacts:

- `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/diagnostic_bias_on_b1v5_u120_20260430`
- `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/eval/diagnostic_bias_off_b1v5_u120_20260430`

Results over 32 paired seeds / 64 games:

- Bias ON:
  - vs B1: 0.484375
  - vs B3: 0.84375
  - vs B4: 1.0
- Bias OFF:
  - vs B1: 0.59375
  - vs B3: 0.0
  - vs B4: 0.0

Interpretation:

- The bias scaffold is essential for B3/B4 performance even for the strong B1 anchor.
- Bias-off mirror/B1 behavior can look okay, but heuristic-anchor performance collapses without the scaffold.
- Therefore the v13 bias-off collapse is not a unique league bug. The broader issue is that our trained policies depend heavily on the public heuristic bias for anchor strength.

Checked v13 scalar diagnostics at u160/u240/u320/u360:

- `reference_policy_top_action_bc_coef_active` stayed active around 0.187 -> 0.174.
- `reference_policy_top_action_bc_loss` worsened from 0.556 at u160 to 0.617 at u320 and 0.669 at u360.
- `counterfactual_positive_top1_match` stayed around 0.535-0.549.
- `vtrace_rho_p99` was very large early/mid-run:
  - u160: 33770
  - u240: 8680
  - u320: 1799
  - u360: 94
- B1 reward scale was 1.10; B1 opponent mix was 0.30; heuristic variant mix was still about 0.275 at u160 and 0.262 at u240.

Interpretation:

- The objective is likely pulling away from B1 despite the BC term.
- The worsening B1 BC loss while the policy is being trained supports objective conflict rather than simple undertraining.
- The very high V-trace tail ratios suggest off-policy / recurrent-policy-lag instability may be contributing, although the current logs are not per-opponent enough to prove B1-specific critic confusion.
