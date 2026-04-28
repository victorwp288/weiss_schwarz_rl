# Rush Branch Rescue Progress - 2026-04-23

## Scope

- Source branch: `rush/snapshot-20260423-160631`
- Primary optimization target: B1-centered server training throughput and learning on the multi-GPU Linux box
- B2 status: diagnostic only on this tranche. We are not retargeting the rush branch to train for B2. We are keeping B2 in the eval/promotion plumbing because earlier thesis runs repeatedly failed there and the training pipeline should not silently drop or hide that signal.

## Code Shipped In This Tranche

- Added server-safe training presets on the thesis-model surface:
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu.yaml`
  - `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague.yaml`
- Added a dedicated short-horizon B1 benchmark surface:
  - `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml`
- Added a matching eval surface for the reduced benchmark model:
  - `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml`
- Added tracked combined teacher-fade plus no-tactical-bias presets:
  - `configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_no_tactical_bias_server_train_auto_gpu.yaml`
  - `configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_no_tactical_bias_eval_auto_gpu.yaml`
  - `configs/presets/baselines/structured_acceptance_thesis_model_teacher_fade_no_tactical_bias_server_train_auto_gpu_noleague.yaml`
- Added thesis wrapper aliases and default eval pairings for the new presets.
- Added `b1-anchor-benchmark` to `python/scripts/thesis_run.py`.
- Reduced the local-friendly benchmark model footprint from the frozen thesis-model surface to:
  - `gru_hidden_size=192`
  - `encoder_mlp_width=192`
  - `typed_feature_width=48`
  - `training.rollout.unroll_length=16`
- Fixed final policy-set selection so optional anchors such as B2 can be resolved from `anchor_scores`, not only top-level summary ids.
- Extended periodic dev-eval persistence to keep richer anchor payloads, uncertainty, warning flags, and evaluation context.
- Added a secondary `best_b2` checkpoint tracker record without changing canonical `best.pt` promotion semantics.
- Wired B2 disagreement-audit request emission into the training path.
- Added runtime timing separation for collector wait, learner wait/update, async overlap, and snapshot publish/apply.
- Refactored process collector overflow handling to spill shared pending unrolls selectively instead of eagerly materializing all shared payloads.
- Added cooperative wall-clock stopping to `python/scripts/train.py` via `--max-wall-clock-minutes`.
- Forwarded the wall-clock budget through `python/scripts/thesis_run.py` and `python/scripts/profile_train_job.py`.
- Updated `scripts/run_thesis_queue.sh` with:
  - a dedicated `b1-benchmark` phase
  - a profiled short-anchor launch path
  - `overnight-main` switching to the rush-branch server-train surfaces
- Upgraded the repo from `torch==2.5.1` on `cu124` to `torch==2.7.0` on `cu128`, including `pyproject.toml`, `uv.lock`, `Makefile`, and `scripts/run_local_ci_parity.sh`.

## Verification

### Focused test suites

- `uv run pytest python/weiss_rl/tests/test_policy_set.py python/weiss_rl/tests/test_thesis_run_wrapper.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_tensorboard_logger.py`
  - Result: `175 passed`
- `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py`
  - Result: `28 passed`
- `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_thesis_run_wrapper.py python/weiss_rl/tests/test_train_stall_monitor.py`
  - Result: `75 passed`
- `uv run python python/scripts/thesis_run.py --preset b1-anchor-benchmark --run-label b1_anchor_benchmark_wrapper_plan_smallmodel --max-wall-clock-minutes 7.5 --dry-run --skip-compare`
  - Result: wrapper now plans eval with `structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml`, so the reduced-size benchmark no longer points at the old `248/248/62` eval model

### Environment repair

- Old local state:
  - `torch 2.5.1+cu124`
  - CUDA arch list ended at `sm_90`
  - local RTX 5080 failed with `no kernel image is available for execution on the device`
- New local state:
  - `torch 2.7.0+cu128`
  - CUDA arch list now includes `sm_100` and `sm_120`
  - basic CUDA matmul and `nn.Embedding` both run successfully on the local RTX 5080

### Local smoke runs

- `runs/rescue_smoke_server_train_local_gpu_cu128_env128`
  - Command shape: thesis-model server-train preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 4`, `--max-updates 1`
  - Status: succeeded
  - Last runtime metrics:
    - `actor_env_steps_per_sec = 1485.65`
    - `timer_runtime_collect_update_batch_total_ms = 986.17`
    - `timer_runtime_fill_pending_unrolls_ms = 971.51`
    - `timer_runtime_collector_queue_wait_ms = 951.02`
    - `timer_runtime_build_learner_batch_ms = 14.04`
    - `timer_runtime_legal_concatenation_ms = 13.99`
    - `timer_runtime_shared_overflow_spill_ms = 18.77`
    - `timer_simulator_python_step_ms = 70.71`
- `runs/rescue_smoke_teacherfade_notactical_noleague_local_gpu_cu128_env128`
  - Command shape: combined teacher-fade plus no-tactical no-league preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 4`, `--max-updates 1`
  - Status: succeeded
  - Last runtime metrics:
    - `actor_env_steps_per_sec = 5639.93`
    - `timer_runtime_collect_update_batch_total_ms = 953.21`
    - `timer_runtime_fill_pending_unrolls_ms = 916.10`
    - `timer_runtime_collector_queue_wait_ms = 895.61`
    - `timer_runtime_build_learner_batch_ms = 32.74`
    - `timer_runtime_legal_concatenation_ms = 32.70`
    - `timer_runtime_shared_overflow_spill_ms = 18.82`
    - `timer_simulator_python_step_ms = 45.35`
- `runs/rescue_b1_anchor_benchmark_smoke`
  - Command shape: `profile_train_job.py` on the new benchmark preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 4`, `--max-updates 1000000`, `--max-wall-clock-minutes 0.35`
  - Status: succeeded
  - Cooperative stop:
    - wall clock budget hit at `21.14s` for a `21.00s` budget
    - final persisted B1 alias update: `12`
  - Telemetry summary:
    - mean `throughput_samples_per_sec = 6641.43`
    - mean `throughput_updates_per_sec = 0.3426`
    - max GPU util observed: `83%`
    - max GPU memory observed: `5120 MB`
  - Training-path signal:
    - steady late updates were around `11.3k-11.7k actor_env_steps/sec`
    - `timer_runtime_fill_pending_unrolls_ms` stayed around `0.98s-1.03s`
    - `timer_runtime_collector_queue_wait_ms` stayed around `0.97s-1.02s`
- `runs/rescue_b1_anchor_benchmark_smoke_smallmodel`
  - Command shape: reduced benchmark preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 32`, `--max-wall-clock-minutes 0.25`
  - Status: bounded failure mode, but still useful
  - Outcome:
    - cooperative stop still worked cleanly
    - no training updates completed before the 15s budget elapsed
    - training metrics summary stayed empty
  - Read: shrinking only the model width was not enough; the local short-budget path still needed a shorter rollout length
- `runs/rescue_b1_anchor_benchmark_smoke_smallmodel_u16`
  - Command shape: reduced benchmark preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 16`, `--max-wall-clock-minutes 0.25`
  - Status: succeeded
  - Cooperative stop:
    - wall clock budget hit at `17.69s` for a `15.00s` budget
    - final persisted B1 alias update: `2`
  - Telemetry summary:
    - mean `throughput_samples_per_sec = 8277.25`
    - mean `throughput_updates_per_sec = 0.0745`
    - max GPU util observed: `77%`
    - max GPU memory observed: `10021 MB`
  - Read:
    - `unroll_length=16` is the first reduced benchmark shape that both stays compatible with the queue path and actually completes updates on this local box
    - the smaller model plus shorter rollout is a better local iteration surface than the original frozen thesis-model shape
- `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/rescue_b1_anchor_benchmark_smoke_smallmodel_u16 --policy-id b1_noleague_baseline --b1-baseline-run-dir runs/rescue_b1_anchor_benchmark_smoke_smallmodel_u16 --paired-seed-limit 1 --skip-metagame --skip-figures --skip-readiness`
  - Result: succeeded
  - Read: the reduced benchmark eval surface can load the reduced-size checkpoint successfully; the old wrapper default to `thesis-model-eval-auto-gpu` was a real correctness bug and is now fixed
- `runs/b1_anchor_monitor_20260423_182153`
  - Command shape: reduced benchmark preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 16`, `--max-wall-clock-minutes 5`, `--runtime-mode train_async_fast`
  - Status: succeeded
  - Training outcome:
    - clean cooperative stop at `302.13s` for a `300.00s` budget
    - final update count: `63`
    - final loss: `2.1002`
    - mean `throughput_samples_per_sec = 13051.72`
    - max `throughput_samples_per_sec = 13702.70`
    - mean GPU util observed: `56.48%`
    - max GPU util observed: `96%`
    - mean GPU memory observed: `9601 MB`
    - max GPU memory observed: `10123 MB`
  - Runtime read:
    - the run stayed collector-bound
    - late `timer_runtime_fill_pending_unrolls_ms` and `timer_runtime_collector_queue_wait_ms` were still around `3.8s-4.2s`
    - learner update time stayed much smaller than collection wait, so the main local bottleneck did not move to the learner
  - Eval note:
    - the first explicit single-policy eval was a false start because it only measured self-play for `b1_noleague_baseline`
    - rerunning canonical eval with explicit policy ids `B0 RandomLegal`, `b1_noleague_baseline`, and `B2 HeuristicPublic` produced the first useful bounded anchor probe
  - Bounded local anchor probe (`paired-seed-limit=4`, still low confidence):
    - `b1_noleague_baseline` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`
    - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `8/8` wins, posterior mean `1.0`
    - `B2 HeuristicPublic` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`
  - Read:
    - this is the first local run on the downsized benchmark surface that looks both fast enough to iterate on and non-trivially competent
    - the bounded probe is still too small to count as thesis-grade evidence, but it is strong enough to justify trying the same surface on the Linux server instead of dismissing it as obviously dumb
- `runs/b1_anchor_native_rollout_20260423_1915`
  - Command shape: native-rollout benchmark preset, `--device cuda:0`, `--num-envs 128`, `--unroll-length 16`, `--max-wall-clock-minutes 5`, `--runtime-mode train_async_fast`
  - Preset delta:
    - `training.heuristic_native_rollout_enabled: true`
    - `training.heuristic_actor_hidden_state_tracking: false`
  - Status: succeeded
  - Training outcome:
    - clean cooperative stop with `367` completed updates
    - final loss: `0.1487`
    - mean `throughput_samples_per_sec = 74111.39`
    - max `throughput_samples_per_sec = 79282.69`
    - mean GPU util observed: `27.45%`
    - max GPU util observed: `95%`
    - mean GPU memory observed: `9654 MB`
    - max GPU memory observed: `10071 MB`
  - Runtime delta versus `b1_anchor_monitor_20260423_182153`:
    - mean throughput improved from `13051.72` to `74111.39` samples/sec, about `+467.8%`
    - late `timer_runtime_collector_queue_wait_ms` fell from `3877.83` to `326.34`, about `-91.6%`
    - late `timer_runtime_fill_pending_unrolls_ms` fell from `3896.78` to `343.81`, about `-91.2%`
    - late `timer_runtime_collect_update_batch_total_ms` fell from `3915.05` to `367.44`, about `-90.6%`
    - late `collector_actor_policy_forward_ms` fell from `7380` to `0`
    - late `collector_collect_actor_unroll_ms` fell from `8964` to `785`, about `-91.2%`
    - simulator-side rollout time itself got slightly worse, from `127.89ms` Python step time to `148.33ms` native rollout time, so the wall-clock win came from deleting actor-side policy-forward work rather than making the simulator intrinsically faster
  - Bounded local anchor probe (`paired-seed-limit=4`, still low confidence):
    - `b1_noleague_baseline` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`
    - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `7/8` wins, posterior mean `0.875`
  - Read:
    - this is the strongest collector-side optimization lead so far on the local box
    - the win appears to come from unlocking the already-implemented simulator-native heuristic rollout path, not from the earlier transport refactor or from additional simulator micro-optimizations
    - the bounded quality probe stayed acceptable enough to justify server validation, but the slight drop against `B2` means this should still be treated as an experiment rather than an unconditional new default
- Promotion to the standard thesis preset stack
  - Change:
    - promoted `training.heuristic_native_rollout_enabled: true`
    - promoted `training.heuristic_actor_hidden_state_tracking: false`
    - landing point: `configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml`
  - Validation:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py -q` -> `33 passed`
    - kept `structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_native_rollout.yaml` only as a compatibility alias so old commands still resolve
- `runs/b1_anchor_promoted_standard_smoke`
  - Command shape: plain benchmark preset after promotion, `--device cuda:0`, `--num-envs 128`, `--unroll-length 16`, `--max-wall-clock-minutes 1`, `--runtime-mode train_async_fast`
  - Status: succeeded
  - Outcome:
    - mean `throughput_samples_per_sec = 62037.73`
    - max `throughput_samples_per_sec = 73092.59`
    - late `timer_runtime_collector_queue_wait_ms` stayed around `323ms-436ms`
    - late `timer_runtime_fill_pending_unrolls_ms` stayed around `342ms-455ms`
    - late `collector_actor_policy_forward_ms` stayed at `0`
    - runtime logs show `timer_simulator_python_native_heuristic_rollout_ms`, confirming the plain benchmark preset now inherits the promoted fast path
  - Read:
    - the native-rollout collector optimization is no longer confined to a special-case benchmark preset
    - the ordinary B1 benchmark surface now lands in the same fast collector regime as the earlier explicit native-rollout experiment

## Failed Ideas And Bounded Blockers

- A `64`-env local smoke with the process backend still fails by design because `_resolve_actor_topology()` collapses that setup to `actor_count=1`, and the process backend requires `actor_count > 1`.
- The league-enabled combined teacher-fade plus no-tactical preset correctly rejects the old thesis-model B1 anchor because the model contract no longer matches.
- The first reduced-size benchmark attempt with `unroll_length=32` still failed to finish an update inside a 15s local budget, so `u32` is not a good local iteration default even though it may still be reasonable on the Linux server.
- Local smoke throughput is not a server throughput claim. These runs are only for correctness, artifact emission, and relative regression checks.
- The new benchmark quality verdict still needs the real Linux server run plus its post-train eval. The local profiled benchmark only proves the launch/stop/telemetry path and relative behavior.
- The bounded `B0/B1/B2` local probe is directional only. It used `4` paired seeds and explicit policy IDs, so it is useful as a quick competence check but not as a final research claim.
- The native-rollout benchmark changed collector semantics enough that its large local speedup should be treated as a mode-scoped result until the Linux server confirms it under the real multi-GPU topology.

## Current Best Read

- The local RTX 5080 is no longer blocked by PyTorch itself.
- The rescued process-collector path is alive on this branch and now emits the extra timings we needed.
- We now have a repo-native short-horizon B1 benchmark path instead of faking it with external timeouts.
- The local-friendly benchmark path now has a smaller model and shorter rollout length, and the wrapper/eval contract matches that reduced architecture.
- A full 5-minute local B1 benchmark on the reduced surface reached update `63` and looked meaningfully stronger than random and the base B2 heuristic in a bounded local probe.
- On the current local smoke, `fill_pending_unrolls` is still mostly collector wait time, which is consistent with the earlier suspicion that the next real win is still in collector transport and queue behavior, not raw simulator micro-ops.
- The strongest local collector-side win is now to bypass actor-side model forward work entirely for all-heuristic B1 anchor collection by enabling simulator-native heuristic rollout and disabling heuristic hidden-state tracking.

## Next Hypotheses

1. Run the native-rollout benchmark preset on the Linux server first and compare it directly against the plain benchmark preset on the same `5-10` minute B1 anchor task.
2. If the Linux server reproduces the collector-time collapse, treat native heuristic rollout as the new default benchmark surface for B1 anchor iteration.
3. If server quality regresses even when throughput improves, test whether re-enabling heuristic hidden-state tracking preserves enough learning while retaining some of the speedup.
4. Compare the benchmark run’s post-train eval against the older slow B1 anchor behavior before spending more budget on long runs.
5. Generate a same-contract B1 no-league anchor for the combined teacher-fade plus no-tactical branch, then boot the league-enabled combined preset against that anchor.
6. If server traces still show `fill_pending_unrolls` dominating after native rollout is enabled, continue the transport refactor into lower-copy chunk assembly and less parent-side repacking.

## 2026-04-24 Wake-Up Iteration: B1 Native-Rollout Topology And Prefetch

### Startup Read

- `AGENTS.md` was not present under `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`; the active startup/operating instructions were therefore the instructions provided in the task prompt.
- Current best known bottleneck at wake-up:
  - `runs/b1_anchor_native_rollout_20260423_1915` had already moved the B1 benchmark out of actor policy-forward bottleneck by enabling native heuristic rollout.
  - The remaining local 128-env promoted smoke still had meaningful parent-side queue wait/fill time.
- Current best learning branch:
  - native heuristic rollout remained alive because the previous 5-minute bounded probe beat B0 and went `7/8` against B2, and the promoted standard smoke preserved the fast path.
- Strong open hypotheses:
  - producer topology might be under-parallelized on local `num_envs=128`;
  - existing `training.collect_batch_prefetch_enabled` might hide remaining collection wait behind learner work;
  - benchmark post-train eval must keep using the reduced benchmark eval preset, not the train YAML.
- Suspicious active config choices:
  - `evaluation.periodic_dev_eval_interval_updates: 0` keeps the benchmark fast but makes checkpoint selection loss-based during the run;
  - `training.actor_policy_backend: heuristic_public`, `actor_heuristic_fraction: 1.0`, native rollout enabled, and hidden-state tracking disabled make this a heuristic-behavior short-horizon B1 benchmark, not ordinary self-play;
  - `scripts/run_thesis_queue.sh` still evaluated the B1 benchmark with the train benchmark YAML instead of the dedicated benchmark eval YAML.

### Changes

- Promoted collect-batch prefetch into the B1 benchmark preset only:
  - `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml`
  - `training.collect_batch_prefetch_enabled: true`
- Fixed the queue script B1 benchmark eval surface:
  - `scripts/run_thesis_queue.sh` now evaluates the `b1-benchmark` phase with `structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml`.
- Added runtime topology metrics:
  - `python/weiss_rl/runtime.py` now logs `runtime_actor_count` and `runtime_envs_per_actor` in runtime performance records.

### Commands And Results

- Topology probe:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_native_rollout_env256_smoke --device cuda:0 --num-envs 256 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final training metrics: update `99`, max cumulative throughput `96157.57` samples/sec, mean `75204.09`.
  - Late runtime read: actor throughput about `112k-120k` samples/sec; queue wait about `110-124ms`; collect-batch total about `147-160ms`.
  - Verdict: more producers helped strongly versus the prior 128-env promoted smoke.
- Larger topology probe:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_native_rollout_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final training metrics: update `124`, max cumulative throughput `107131.30` samples/sec, mean `78846.74`.
  - Late runtime read: actor throughput about `103k-150k` samples/sec; queue wait collapsed to about `1-2ms`; collect-batch total about `25-26ms`; learner update remained about `386-426ms`.
  - Verdict: local collection wait was nearly eliminated; the bottleneck moved to learner-side update/teacher work.
- Prefetch probe before promotion:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_native_rollout_env512_prefetch_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast --override training.collect_batch_prefetch_enabled=true`
  - Result: succeeded.
  - Final training metrics: update `130`, max cumulative throughput `112791.97` samples/sec, mean `83294.34`.
  - Late runtime read: learner idle wait for batch fell to about `2.1ms`; collect-batch prefetch wait stayed near `0.04ms`.
  - Verdict: modest but real local throughput win, enough to promote into the short-horizon B1 benchmark preset.
- Validation after promotion:
  - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py -q`
  - Result: `122 passed`.
  - `uv run python python/scripts/thesis_run.py --preset b1-anchor-benchmark --run-label b1_anchor_plan_check_prefetch_eval --max-wall-clock-minutes 7.5 --dry-run --skip-compare`
  - Result: planned train with the benchmark train YAML and eval with the dedicated benchmark eval YAML.
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_prefetch_default_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final training metrics: update `129`, max cumulative throughput `112417.19` samples/sec, mean `83070.24`.
  - Late runtime read: `runtime_actor_count=8`, `runtime_envs_per_actor=64`, actor throughput about `140k-151k` samples/sec, queue wait about `1.6-2.7ms`, collect-batch total about `33-39ms`, learner update about `390-445ms`.
- Bounded local quality probe:
  - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_anchor_prefetch_default_env512_smoke --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_anchor_prefetch_default_env512_smoke --paired-seed-limit 4 --skip-metagame --skip-figures --skip-readiness`
  - Result: completed after the first command timeout left the eval process running; no orphan eval process remained after waiting for completion.
  - `b1_noleague_baseline` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`.
  - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `8/8` wins, posterior mean `1.0`.
  - Verdict: local-only, small-seed directional quality check stayed strong under the prefetch-default benchmark artifact.

### Risks And Scope

- These are local Windows/RTX 5080 results only. They are useful for correctness and relative regression checks, not final throughput claims.
- The topology result says local `128` envs underfed the native-rollout benchmark; it does not prove that server `2048` envs needs changing, because the queue script already defaults B1 benchmark envs to `${NUM_ENVS}` and server default is `2048`.
- The new prefetch default is scoped to the reduced short-horizon B1 benchmark preset. It is not promoted to the main thesis training preset until server validation confirms it is stable under the multi-GPU topology.
- The benchmark still uses loss-based checkpoint choice during training because periodic dev eval is disabled; post-train explicit B0/B1/B2 eval remains required before treating a run as stronger.

### Next Hypotheses

1. Server-validate the promoted B1 benchmark preset at the usual `B1_BENCHMARK_NUM_ENVS=2048`, then compare against the previous native-rollout benchmark on the same 5-10 minute surface.
2. If server logs show collection wait is gone, move the next throughput work to learner-side costs: public heuristic target construction, teacher aux, legal-action concatenation, and lower-copy batch assembly.
3. If server logs still show collection wait, use the new `runtime_actor_count` and `runtime_envs_per_actor` metrics to test whether actor topology or queue capacity is the culprit.
4. Run a longer same-surface quality confirmation for `b1_anchor_prefetch_default_env512_smoke` or the server equivalent before using it as a thesis-grade B1 anchor.
5. Test `training.heuristic_actor_hidden_state_tracking=true` as a quality-risk ablation only if server quality regresses relative to the earlier native-rollout B2 probe.

## 2026-04-24 Iteration: Learner-Side Public-Heuristic Target Filtering

### Rationale

- After the 512-env native-rollout + prefetch B1 benchmark, collection wait was mostly gone locally:
  - `runtime_actor_count=8`, `runtime_envs_per_actor=64`;
  - queue wait about `1.6-2.7ms`;
  - collect-batch total about `33-39ms`;
  - learner update about `390-445ms`.
- Tail metrics from `runs/b1_anchor_prefetch_default_env512_smoke` showed learner-side teacher/public-heuristic work as the next scalable throughput frontier:
  - `learner_update_ms` mean `406.35`;
  - `timer_learner_public_heuristic_target_ms` mean `48.17`;
  - `timer_learner_teacher_aux_ms` mean `24.65`;
  - `teacher_valid_fraction` about `0.49`.
- Suspicious config/code mismatch:
  - the model public heuristic bias families were focused on `main_play_character`, `main_move`, and `attack`;
  - the B1 benchmark inherited empty `teacher_public_heuristic_families`, so public-heuristic distillation was active over every teacher-valid row;
  - the target scorer built public-heuristic logits before exact teacher-family row filtering, and packed soft-target CE computed all packed rows before selecting the public rows.

### Changes

- `python/weiss_rl/learners/impala_learner.py`
  - Added `_teacher_public_heuristic_rows(...)` to compute exact active public-target rows from:
    - `loss_mask`;
    - `teacher_valid`;
    - optional configured `teacher_public_heuristic_families`;
    - action-catalog family ids.
  - Threaded those active rows into `_packed_public_heuristic_target_logits(...)` and `_factorized_public_heuristic_teacher_view(...)`, so the public target scorer only sees relevant rows.
  - Added an optional `row_mask` to `_packed_soft_target_cross_entropy(...)`, so CE/entropy/student-top-mass work is also restricted to the public rows instead of computing over every packed row and slicing later.
- `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml`
  - Set B1 benchmark public-heuristic teacher families to match the active model-bias surface:
    - `main_play_character`
    - `main_move`
    - `attack`
  - Kept the change scoped to the B1 benchmark preset for now.
- `python/weiss_rl/tests/test_impala_learner.py`
  - Added coverage proving the public target scorer receives only configured-family candidate rows.

### Commands And Results

- Unit/regression tests:
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py -q`
  - Result: `43 passed`.
  - `uv run pytest python/weiss_rl/tests/test_config_loader.py -q`
  - Result: `33 passed`.
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py -q`
  - Result: `165 passed`.
- Throughput smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_publictarget_familyfilter_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final training metrics: update `135`, max cumulative throughput `116880.17` samples/sec, mean throughput `86001.42` samples/sec.
  - Comparison against `runs/b1_anchor_prefetch_default_env512_smoke`:
    - max cumulative throughput: `112417.19 -> 116880.17` samples/sec, about `+4.0%`;
    - mean throughput: `83070.24 -> 86001.42` samples/sec, about `+3.5%`.
- Tail timing comparison, 20-update means, old prefetch-default -> new family-filter:
  - `learner_update_ms`: `406.35 -> 390.95`;
  - `timer_learner_public_heuristic_target_ms`: `48.17 -> 27.91`, about `-42%`;
  - `timer_learner_teacher_aux_ms`: `24.65 -> 23.89`;
  - `timer_learner_forward_time_major_ms`: `103.93 -> 105.67`;
  - `timer_learner_trunk_ms`: `47.09 -> 47.54`;
  - `timer_learner_packed_scorer_ms`: `56.83 -> 58.12`;
  - `timer_learner_backward_ms`: `97.56 -> 101.79`;
  - `timer_learner_optimizer_ms`: `95.90 -> 93.49`;
  - batch build and collect timings stayed roughly flat.
- Bounded local quality probe:
  - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_anchor_publictarget_familyfilter_env512_smoke --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_anchor_publictarget_familyfilter_env512_smoke --paired-seed-limit 4 --skip-metagame --skip-figures --skip-readiness`
  - Result: completed after waiting for the eval process that outlived the first command timeout; no orphan Python process remained.
  - `b1_noleague_baseline` vs `B0 RandomLegal`: `7/8` wins, posterior mean `1.0`.
  - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `7/8` wins, posterior mean `0.875`, `prob_gt_half=1.0`.

### Verdict

- Promote as a B1 benchmark candidate, but not yet thesis-grade final proof.
- This is a code/default-path throughput improvement, not local micro-tuning:
  - it reduces learner public-target work before scoring and CE;
  - the saving should scale with packed legal candidates, teacher-valid rows, configured families, and batch size;
  - it is therefore much more likely to transfer to the Linux server than a tiny local-only env/topology nudge.
- Quality is still alive but slightly noisier than the previous 4-paired-seed probe:
  - previous prefetch-default local eval was `8/8` vs B2;
  - this run was `7/8` vs B2, matching the earlier native-rollout directional result.

### Risks And Scope

- Local-only Windows/RTX 5080 evidence. Treat this as correctness and relative-regression evidence, not a server throughput claim.
- The B1 quality probe is small and noisy; it should trigger a longer confirmation before replacing the current anchor.
- Scoping the teacher-public families to model-biased families changes the public distillation surface. It looks semantically cleaner and faster, but a longer run must confirm it does not remove useful broad teacher signal.
- This optimization applies to runtime paths with public-heuristic teacher targets. It does not automatically speed unrelated league/PPO paths unless they use the same teacher-public target machinery.

### Next Hypotheses

1. Run a longer local B1 candidate confirmation on the same preset to disambiguate the `7/8` vs `8/8` B2 wobble.
2. Profile the remaining learner-side target path inside `teacher_aux` and packed scorer:
   - group-log-prob / family-head work;
   - repeated packed metadata transforms;
   - legal-action concatenation and row offset handling.
3. Consider a structural row-family split cache in the learner batch if profiling shows repeated family-mask construction or packed legal slicing is meaningful at server batch sizes.
4. Keep this server candidate queued only after a longer local quality check says it still learns; server compute should not be spent merely to discover a local correctness/quality regression.

## 2026-04-24 Iteration: Strategic-Public Teacher Target Surface

### Rationale

- The previous narrow public-target filter (`main_play_character`, `main_move`, `attack`) was a valid throughput probe, but it looked semantically too restrictive:
  - `main_move` was effectively `0.0` in the B1 native-rollout data;
  - the latest run therefore distilled mostly `main_play_character + attack`;
  - public-heuristic teacher logic also scores meaningful strategic families such as clocking, climax play, event play, mulligan selection, level-up, encore, trigger order, and choice selection.
- We want the scalable code optimization from row filtering without amputating useful public teacher signal.

### Changes

- Kept the learner-side row-filtering implementation.
- Tightened public-target metric/loss rows to intersect `loss_mask > 0`, avoiding CE work on rows that cannot contribute.
- Added public-target coverage metrics:
  - `teacher_public_heuristic_selected_fraction`;
  - `teacher_public_heuristic_teacher_valid_coverage`.
- Widened the B1 benchmark public-heuristic teacher-family surface to strategic-public families:
  - `mulligan_select`
  - `clock_from_hand`
  - `main_play_character`
  - `main_play_event`
  - `climax_play`
  - `main_move`
  - `attack`
  - `level_up`
  - `encore_pay`
  - `encore_decline`
  - `trigger_order`
  - `choice_select`
- Still excluded bookkeeping/noisy families:
  - `pass`
  - `mulligan_confirm`
  - `choice_prev_page`
  - `choice_next_page`
  - `concede`

### Commands And Results

- Tests:
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py -q`
  - Result: `43 passed`.
  - `uv run pytest python/weiss_rl/tests/test_config_loader.py -q`
  - Result: `33 passed`.
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py -q`
  - Result: `165 passed`.
- Throughput smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_publictarget_strategic_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final scalar update: `131`.
  - Final cumulative throughput: `113866.71` samples/sec.
  - Tail 20-update mean throughput: `111740` samples/sec.
  - Tail `learner_update_ms`: `400.53`.
  - Tail `timer_learner_public_heuristic_target_ms`: `40.21`.
  - Tail `teacher_public_heuristic_selected_fraction`: `0.739`.
  - Tail `teacher_public_heuristic_teacher_valid_coverage`: `0.749`.
- Same-surface comparison:
  - prefetch-default/all teacher-valid semantics:
    - final cumulative throughput `112417.19`;
    - tail public-target time `48.17ms`.
  - narrow three-family filter:
    - final cumulative throughput `116880.17`;
    - tail public-target time `27.91ms`;
    - bounded eval was `7/8` vs B2.
  - strategic-public filter:
    - final cumulative throughput `113866.71`;
    - tail public-target time `40.21ms`;
    - bounded eval was `8/8` vs B2.
- Bounded local quality probe:
  - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_anchor_publictarget_strategic_env512_smoke --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_anchor_publictarget_strategic_env512_smoke --paired-seed-limit 4 --skip-metagame --skip-figures --skip-readiness`
  - Result: completed cleanly.
  - `b1_noleague_baseline` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`.
  - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `8/8` wins, posterior mean `1.0`, `prob_gt_half=1.0`.
  - No B1-vs-B2 truncations in this probe.

### Verdict

- Strategic-public is the better B1 candidate than the narrow three-family target:
  - it preserves most of the scalable target-filter speedup;
  - it covers about three quarters of teacher-valid public rows instead of mostly play/attack;
  - it recovered the bounded B2 signal to `8/8`.
- Keep strategic-public as the current candidate default for the B1 benchmark preset.
- Do not promote this as thesis-grade proof yet; it is still local-only, short-horizon, and small-eval evidence.

### Risks And Next Hypotheses

- The B1 quality probe remains noisy at 4 paired seeds. Run a longer local confirmation if this becomes the server candidate.
- If server-side learner public-target time is still meaningful, the next scalable target is reducing target scoring for strategic-public rows without falling back to narrow semantics:
  - row-family split cache;
  - cheaper packed legal slicing for selected rows;
  - avoiding repeated metadata/device transforms around public target scoring.
- If strategic-public holds quality but costs too much on server, try a middle surface:
  - `clock_from_hand`, `main_play_character`, `main_play_event`, `climax_play`, `main_move`, `attack`, `level_up`, `choice_select`;
  - leave encore/trigger-order out only if metrics show they add cost without coverage/quality value.

## 2026-04-24 Iteration: Throughput Deep Dive And Packed-Scorer Fast Path

### Startup / Bottleneck Restatement

- `AGENTS.md` was still not present under the repo root; the active instructions remain the task prompt and embedded repo instructions.
- Current best B1 candidate before this iteration:
  - `runs/b1_anchor_publictarget_strategic_env512_smoke`
  - local-only, short-horizon, strategic-public target surface
  - final cumulative throughput `113866.71` samples/sec
  - bounded eval: `8/8` vs B0 and `8/8` vs B2.
- Current bottleneck from the deep-dive metrics:
  - The current B1 surface is learner-bound, not collector-bound.
  - Tail-20 strategic baseline:
    - `learner_update_ms`: `400.53`
    - `timer_learner_total_ms`: `399.74`
    - `timer_learner_loss_and_metrics_ms`: `200.90`
    - `timer_learner_forward_time_major_ms`: `103.27`
    - `timer_learner_packed_scorer_ms`: `56.68`
    - `timer_learner_public_heuristic_target_ms`: `40.21`
    - `timer_learner_backward_ms`: `101.94`
    - `timer_learner_optimizer_ms`: `94.87`
    - `timer_runtime_collect_update_batch_total_ms`: `42.89`
    - `learner_idle_wait_for_prefetch_ms`: about `0.04`
  - Queue/fill waits are now mostly hidden by prefetch; more queue capacity is not the first lever.
- Suspicious config/code findings:
  - `model.structured_policy_contract=packed_v1`, so the factorized learner path is inactive.
  - The learner was scoring about `0.87M` packed legal candidates/update.
  - An existing plan-oriented packed scorer existed in `python/weiss_rl/model.py`, but `score_packed_candidates(...)` still called the older boolean-mask chunk scorer.
  - Public-heuristic target scoring still used the generic `candidate_scoring_chunk_size=65536`, while learner CUDA candidate scoring used `cuda_learner_candidate_scoring_chunk_size=1048576`.
  - `training.profile_timers=true` is useful for diagnostics, but a same-surface timers-off falsifier did not improve throughput locally.
  - Runtime `legal_concatenation` was a misleading aggregate: it covered broad batch rebuild, not only legal concat.

### Changes

- `python/weiss_rl/model.py`
  - Switched `score_packed_candidates(...)` to build a `_PackedScoringPlan` and call `_score_packed_candidates_chunked(...)`.
  - Aligned `_score_packed_public_heuristic_chunked(...)` with the learner CUDA chunk policy:
    - if the scoring plan is on CUDA, use `max(candidate_scoring_chunk_size, cuda_learner_candidate_scoring_chunk_size)`.
- `python/weiss_rl/runtime.py`
  - Added runtime batch rebuild diagnostic timers:
    - `timer_runtime_batch_concat_total_ms`
    - `timer_runtime_batch_core_field_concatenation_ms`
    - `timer_runtime_batch_bootstrap_field_concatenation_ms`
    - `timer_runtime_batch_teacher_field_concatenation_ms`
    - `timer_runtime_legal_concatenation_only_ms`
  - Kept legacy `timer_runtime_legal_concatenation_ms` as the aggregate for old benchmark comparability.
- `python/weiss_rl/tests/test_contracts.py`
  - Added an equivalence regression test that compares the new packed-plan scorer against the legacy chunked scorer on the same packed candidate rows.

### Commands And Results

- Falsifier for timer overhead:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_strategic_env512_notimers_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast --override training.profile_timers=false`
  - Result: no improvement.
  - Tail throughput `110290` samples/sec vs strategic baseline `111740`; final cumulative max `112514.69`.
  - Verdict: do not treat `profile_timers=true` as the main bottleneck on this surface.
- Packed-plan scorer smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_strategic_packedplan_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final cumulative throughput `117838.46` samples/sec.
  - Tail throughput `115660` samples/sec.
  - Tail `timer_learner_packed_scorer_ms`: `56.68 -> 49.81`.
  - Tail `timer_learner_total_ms`: `399.74 -> 385.19`.
  - Verdict: promote the packed-plan scorer path.
- Packed-plan + diagnostics smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_strategic_packedplan_diags_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final cumulative throughput `118073.74` samples/sec.
  - Batch rebuild split, tail means:
    - `timer_runtime_batch_core_field_concatenation_ms`: `17.23`
    - `timer_runtime_batch_bootstrap_field_concatenation_ms`: `7.56`
    - `timer_runtime_batch_teacher_field_concatenation_ms`: `1.02`
    - `timer_runtime_legal_concatenation_only_ms`: `8.22`
    - `timer_runtime_batch_concat_total_ms`: `34.04`
  - Verdict: host-side rebuild cost is broad array assembly, not only legal-row reordering.
- Public-target CUDA chunk alignment smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_strategic_packedplan_publicchunk_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Final cumulative throughput `123096.75` samples/sec.
  - Tail throughput `121104` samples/sec.
  - Tail `timer_learner_public_heuristic_target_ms`: `40.21 -> 14.13`.
  - Tail `timer_learner_total_ms`: `399.74 -> 362.49`.
  - Tail `learner_update_ms`: `400.53 -> 363.37`.
  - Tail `timer_learner_packed_scorer_ms`: `56.68 -> 49.22`.
  - Verdict: promote; this is the strongest local B1 throughput candidate so far.
- Bounded local quality probe for the final candidate:
  - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_anchor_strategic_packedplan_publicchunk_env512_smoke --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_anchor_strategic_packedplan_publicchunk_env512_smoke --paired-seed-limit 4 --skip-metagame --skip-figures --skip-readiness`
  - Result: completed cleanly.
  - `b1_noleague_baseline` vs `B0 RandomLegal`: `8/8` wins, posterior mean `1.0`.
  - `b1_noleague_baseline` vs `B2 HeuristicPublic`: `8/8` wins, posterior mean `1.0`, `prob_gt_half=1.0`.
  - No B1-vs-B2 truncations.
- Tests:
  - `uv run pytest python/weiss_rl/tests/test_heuristic_public.py -q`
  - Result: `27 passed`.
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py -q`
  - Result: `43 passed`.
  - `uv run pytest python/weiss_rl/tests/test_contracts.py -q`
  - Result after adding the equivalence test: `53 passed`.
  - `uv run pytest python/weiss_rl/tests/test_runtime.py -q`
  - Result: `89 passed`.
  - `uv run pytest python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_heuristic_public.py -q`
  - Result: `79 passed`.
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_runtime.py -q`
  - Result: `132 passed`.

### Verdict

- Promote the packed-plan scorer and public-target CUDA chunk alignment as the current B1 throughput candidate.
- Current best local B1 benchmark anchor:
  - `runs/b1_anchor_strategic_packedplan_publicchunk_env512_smoke`
  - local-only, throughput + bounded quality
  - final cumulative throughput `123096.75` samples/sec
  - tail throughput `121104` samples/sec
  - bounded eval `8/8` vs B0 and `8/8` vs B2.
- Improvement against the strategic-public baseline:
  - final cumulative throughput `113866.71 -> 123096.75`, about `+8.1%`.
  - tail learner total `399.74ms -> 362.49ms`, about `-9.3%`.
  - public-target timer `40.21ms -> 14.13ms`, about `-65%`.
  - packed scorer timer `56.68ms -> 49.22ms`, about `-13%`.
- This is a scalable code-path improvement:
  - less per-candidate boolean-mask scorer overhead;
  - fewer CUDA public-target chunks for large packed candidate batches;
  - no heavy compute knobs changed.

### Risks And Next Hypotheses

- Local-only Windows/RTX 5080 evidence. Server validation is still required before making multi-GPU throughput claims.
- The target surface still uses `packed_v1`; the factorized policy path is inactive. A later, larger structural experiment should compare `factorized_v1` against packed scoring on the same B1 anchor.
- The remaining learner bottlenecks after the final candidate:
  - forward/scorer still about `96.6ms` with packed scorer about `49.2ms`;
  - backward + optimizer about `185ms`;
  - host batch rebuild about `32-34ms`.
- Next throughput candidates:
  1. Try fused/foreach CUDA Adam or a fast steady-state AMP finite-check path; optimizer remains about `93ms`.
  2. Score only the union of policy-train rows and teacher/public rows; policy train fraction is about `0.50`, but teacher aux currently prevents the existing restricted-row scorer path from taking the full win.
  3. Design a direct learner-layout batch assembly path or actor-major learner consumption path; diagnostics show legal-only is only about `8ms` of the `34ms` batch rebuild.
  4. Add explicit active-path metrics for process-vs-central collection, native rollout fast gate, and factorized-vs-packed policy contract so stale flags are easier to catch.

## 2026-04-24 Iteration: Optimizer, Row-Union Scoring, Batch Builder, Factorized Falsifier

### Startup / Bottleneck Restatement

- `AGENTS.md` was still not present in the repo root; the active instructions were the embedded task instructions.
- Current best local B1 anchor before this iteration:
  - `runs/b1_anchor_strategic_packedplan_publicchunk_env512_smoke`
  - local-only, throughput + bounded quality
  - tail throughput `121103.56` samples/sec
  - final/max throughput `123096.75` samples/sec
  - tail `timer_learner_total_ms`: `362.49`
  - bounded 4-paired-seed eval was previously `8/8` vs B0 and `8/8` vs B2.
- Bottleneck:
  - learner-side work, not collector wait.
  - optimizer about `93.25ms`;
  - packed scorer about `49.22ms`;
  - public target about `14.13ms`;
  - host batch build about `32.46ms`.
- Suspicious choices audited:
  - optimizer implementation was implicit plain Adam even on CUDA;
  - AMP steady-state path scanned every parameter gradient every update;
  - teacher auxiliary training forced all-row packed scoring even though teacher/policy/public losses are masked to `policy_train_mask`;
  - `structured_metrics.mode=sampled` means every 10th update still needs all rows to preserve all-row summary semantics;
  - `factorized_v1` was inactive and needed a smoke-test before treating it as a throughput frontier.

### Changes

- `python/weiss_rl/learners/impala_learner.py`
  - Added live `optimizer_backend` support: `auto`, `default`, `foreach`, `fused`.
  - `auto` uses fused Adam on CUDA when available and falls back safely.
  - Removed the normal AMP per-parameter finite-gradient scan; the full scan is now only used on non-finite grad-norm failure.
  - Allowed packed learner scoring to restrict to `policy_train_mask` rows even when teacher aux is active, while preserving all-row scoring when structured summary metrics are emitted.
- `python/weiss_rl/config/models.py`, `python/weiss_rl/config/parse.py`, `python/scripts/train.py`
  - Added parsed/lived-through `training.optimizer.backend`.
- `configs/presets/typed_thesis_locked.yaml`
  - Made `optimizer.backend: auto` explicit.
- `python/weiss_rl/runtime.py`
  - Added grouped time-major, optional time-major, and batch-major concat helpers.
  - `_build_learner_batch` and `_build_ppo_batch` now allocate/fill core/bootstrap/teacher arrays in grouped passes while preserving the returned dict contract.
- Tests added/updated:
  - optimizer backend parser/live tests;
  - AMP steady-state no-scan test;
  - teacher-active packed row restriction test;
  - structured-summary all-row guard test;
  - deterministic teacher-action logits in one auxiliary-only test.

### Commands And Results

- Optimizer/AMP smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_fastamp_fusedadam_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Tail throughput `128669.61` samples/sec.
  - Max/final throughput `130551.27` samples/sec.
  - Tail `timer_learner_optimizer_ms`: `93.25 -> 67.55`.
  - Tail `timer_learner_total_ms`: `362.49 -> 337.72`.
  - Verdict: promote.
- Row-union packed scoring smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_fastamp_rowunion_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Tail throughput `140340.90` samples/sec.
  - Max/final throughput `142119.76` samples/sec.
  - Tail `timer_learner_packed_scorer_ms`: `49.22 -> 38.71`.
  - Tail `timer_learner_total_ms`: `362.49 -> 309.82`.
  - Normal non-summary updates scored about `294935` packed candidates/update instead of about `878851` total candidates/update.
  - Verdict: promote as local throughput candidate; quality needs confirmation because this changes the learner scoring surface.
- Factorized smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_factorized_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast --override model.structured_policy_contract=factorized_v1`
  - Result: succeeded but slow.
  - Tail throughput `91970.30` samples/sec.
  - Tail `timer_learner_total_ms`: `492.08`.
  - Tail `timer_learner_factorized_policy_ms`: `124.96`.
  - Tail `timer_learner_public_heuristic_student_ms`: `71.90`.
  - Verdict: do not promote for B1 throughput; keep as a later learning/architecture experiment.
- Batch-builder smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_fastamp_rowunion_batchbuilder_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Tail throughput `140368.16` samples/sec, essentially tied with row-union.
  - Tail `timer_runtime_build_learner_batch_ms`: `31.85 -> 31.32`.
  - Verdict: small local win, keep because it preserves the contract and removes repeated passes.
- Explicit optimizer-backend final smoke:
  - `uv run python python/scripts/profile_train_job.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke --device cuda:0 --num-envs 512 --unroll-length 16 --max-updates 1000000 --max-wall-clock-minutes 1 --runtime-mode train_async_fast`
  - Result: succeeded.
  - Current best local B1 throughput artifact:
    - `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke`
    - tail throughput `140702.51` samples/sec;
    - max/final throughput `142620.36` samples/sec;
    - tail `timer_learner_total_ms`: `304.40`;
    - tail `timer_learner_packed_scorer_ms`: `38.01`;
    - tail `timer_learner_optimizer_ms`: `66.98`;
    - tail `timer_runtime_build_learner_batch_ms`: `30.71`.
- Quality probe:
  - A full 4-paired-seed B0/B1/B2 eval for `runs/b1_anchor_fastamp_rowunion_env512_smoke` was stopped after the first few matchup artifacts; it did not reach the useful B1-vs-B2 direction and is not a verdict.
  - Smaller B1/B2-only probe:
    - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_anchor_fastamp_rowunion_batchbuilder_env512_smoke --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_anchor_fastamp_rowunion_batchbuilder_env512_smoke --paired-seed-limit 2 --stage1-paired-seeds 2 --max-paired-seeds 2 --skip-metagame --skip-figures --skip-readiness`
    - Result: completed.
    - B1 vs B2: `3/4` wins, `0` losses, `1` truncation; posterior mean `0.75`, `prob_gt_half=1.0`.
    - Verdict: local directional quality is acceptable for continuing throughput work, but not thesis-grade proof.
- Tests:
  - `uv run pytest python/weiss_rl/tests/test_impala_learner.py -q`
  - Result: `48 passed`.
  - `uv run pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_contracts.py -q`
  - Result: `175 passed`.

### Verdict

- Promote for local B1 throughput:
  - fused CUDA Adam via `optimizer.backend=auto`;
  - AMP steady-state no-scan path;
  - teacher-safe packed row restriction;
  - grouped runtime batch assembly.
- Current best local B1 throughput candidate:
  - `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke`
  - local-only, throughput-first with small directional quality support.
- Improvement vs previous best `publicchunk` anchor:
  - tail throughput `121103.56 -> 140702.51`, about `+16.2%`;
  - max/final throughput `123096.75 -> 142620.36`, about `+15.9%`;
  - tail learner total `362.49ms -> 304.40ms`, about `-16.0%`;
  - tail optimizer `93.25ms -> 66.98ms`, about `-28.2%`;
  - tail packed scorer `49.22ms -> 38.01ms`, about `-22.8%`.
- Do not promote `factorized_v1` as a throughput path yet.

### Risks And Next Hypotheses

- This is still local-only Windows evidence; server validation is required before multi-GPU/Linux throughput claims.
- Row restriction changes which packed rows are scored during teacher-active updates, even though masked losses should preserve semantics. The 2-seed B1/B2 quality probe is positive but too small; run a 4-seed confirmation before serious server training.
- The full 4-seed eval wrapper appeared to keep running after writing partial artifacts once; inspect eval process cleanup/replay scheduling if that repeats.
- Remaining target:
  - backward is now a larger share again (`~68ms`);
  - host batch build is only `~31ms`, so a larger direct shared-buffer redesign may not be worth doing before server profiling;
  - public-heuristic student/factorized path is too slow and should be attacked only if factorized becomes a learning requirement.

## 2026-04-24 Iteration: Autoscale + Multi-Rank Learner Scaffold

### Startup / Bottleneck Restatement

- `AGENTS.md` was still not present in the repo root; the active project instruction was the pasted long-running optimization rule.
- Current best local B1 throughput candidate remains:
  - `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke`
  - local-only throughput-first evidence;
  - tail throughput `140702.51` samples/sec;
  - max/final throughput `142620.36` samples/sec;
  - tail learner total `304.40ms`.
- Bottleneck/restated direction:
  - single-GPU local B1 is now learner-side dominated;
  - the next structural frontier is scaling learner and rollout work across L40-class multi-GPU hardware without relying on box-specific tiny config nudges.
- Suspicious choices audited:
  - old `system.actor_process_count=12` is a local/legacy cap and should not constrain server topology;
  - `auto` must be auditable and reproducible, not hidden magic;
  - direct `DistributedDataParallel(model)` is risky because the learner calls custom structured model methods, so the safer first design is raw-model interface plus explicit gradient averaging;
  - artifact/checkpoint/eval writes must be rank-0 gated before any real multi-rank server launch.

### Changes

- Added `python/weiss_rl/autoscale.py`
  - deterministic hardware profiles for `local`, `uc1-l40-3`, `uc1-l40-4`, `8gpu-l40`, and `gpu<N>`;
  - topology resolver for learner GPU count, learner parallelism, actor count, envs/actor, total envs, queue capacity, CPU/RAM/VRAM budgets;
  - server plans deliberately prefer `64` envs/actor instead of maximizing process count.
- Added parsed `training.scaling` config and canonical defaults:
  - `learner_parallelism: auto`;
  - `learner_gpu_count: auto`;
  - `target_envs_per_gpu: 512`;
  - `max_actor_process_count: 64`;
  - queue/RAM/VRAM budget fractions.
- Added train CLI:
  - `--autoscale`;
  - `--autoscale-dry-run`;
  - `--hardware-profile`;
  - `--ddp`;
  - `--ddp-backend`.
- Wired autoscale into train/runtime:
  - `build_runtime_config` can accept resolved actor/batch/queue topology exactly;
  - run summary, determinism report, and environment manifest now record `autoscale_topology` and `distributed`;
  - local `gpu1` autoscale executes as `8 x 64 = 512` envs.
- Added `python/weiss_rl/distributed.py`
  - env-derived distributed context;
  - process-group init/cleanup;
  - object broadcast/barrier helpers;
  - deterministic per-rank seed derivation;
  - explicit gradient averaging over raw model parameters.
  - scalar all-reduce helper for global DDP throughput counters.
- Updated `ImpalaLearner`:
  - optional `gradient_sync` hook;
  - gradient sync runs after backward/unscale and before grad clipping/optimizer;
  - timer metric `timer_learner_gradient_sync_ms`.
- DDP scalar metrics now include rank-aggregated counters:
  - `distributed_global_batch_env_steps`;
  - `distributed_global_total_samples_processed`;
  - `distributed_global_samples_per_sec`;
  - `distributed_local_batch_env_steps`.
- Added tests:
  - autoscale L40/generic GPU topology tests;
  - DDP sharding/seed tests;
  - config scaling parse assertions;
  - learner gradient-sync hook test.

### Commands And Results

- Focused tests:
  - `uv run pytest python/weiss_rl/tests/test_autoscale.py python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_reads_typed_thesis_preset python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_calls_gradient_sync_hook_during_update -q`
  - Result: initially caught a resolver bug where server topology chose too many actor processes (`48 x 32`, `64 x 32`) instead of the intended `64` envs/actor.
  - Fixed resolver scoring.
  - Final result: `7 passed`.
- Broader relevant tests:
  - `uv run pytest python/weiss_rl/tests/test_autoscale.py python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_impala_learner.py -q`
  - Result before global DDP scalar aggregation: `177 passed`.
  - Result after adding global scalar all-reduce: `178 passed`.
- L40-4 dry-run:
  - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --runtime-mode train_async_fast --autoscale-dry-run --hardware-profile uc1-l40-4 --max-updates 1 --unroll-length 16`
  - Result: resolved `ddp`, `4` GPUs, `32` actors, `64` envs/actor, `2048` total envs, queue capacity `256`.
- L40-3 dry-run:
  - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --runtime-mode train_async_fast --autoscale-dry-run --hardware-profile uc1-l40-3 --max-updates 1 --unroll-length 16`
  - Result: resolved `ddp`, `3` GPUs, `24` actors, `64` envs/actor, `1536` total envs, queue capacity `256`.
- Local single-GPU autoscale execution smoke:
  - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label autoscale_gpu1_smoke --runtime-mode train_async_fast --autoscale --hardware-profile gpu1 --max-updates 1 --unroll-length 16 --profile-timers`
  - Result: completed.
  - Artifact: `runs/autoscale_gpu1_smoke`.
  - Runtime metrics confirmed `runtime_actor_count=8`, `runtime_envs_per_actor=64`, `distributed_world_size=1`, native rollout active, `collector_actor_policy_forward_ms=0`, and `timer_learner_gradient_sync_ms ~= 0.003`.
- Multi-rank primitive smoke:
  - `torchrun` itself failed on this Windows PyTorch wheel before entering our script due elastic TCPStore/libuv mismatch.
  - Direct two-process Gloo smoke using `weiss_rl.distributed` succeeded:
    - rank 0 gradient after averaging: `1.500000`;
    - rank 1 gradient after averaging: `1.500000`.

### Verdict

- Keep/promote the autoscale resolver and dry-run surface.
- Keep the raw-model explicit gradient averaging path as the first safe learner-scale implementation:
  - it avoids DDP `module.` checkpoint keys;
  - it preserves custom structured model methods;
  - it is locally testable without real multi-GPU hardware.
- Local execution proof exists for the scaled-down `gpu1` path.
- Multi-rank communication proof exists for Gloo gradient averaging.
- Real `torchrun` full training remains Linux/server preflight work because this Windows wheel blocks elastic rendezvous before user code runs.

### Risks And Next Hypotheses

- This is not yet a server throughput result. It is local correctness/planning evidence plus a distributed primitive smoke.
- The current distributed learner uses all-reduce gradient averaging, not model-wrapped DDP buckets. It is safer for this model interface but may be less optimal than true DDP bucket overlap; server profiling should decide if that matters.
- Rank-0 gating is in place for the main artifact path, but rollback/checkpoint/eval synchronization across ranks needs a longer multi-rank server smoke before a serious training run.
- Next highest-value work:
  1. Add a Linux launch script/wrapper for `torchrun --nproc_per_node=$GPU_COUNT` with autoscale dry-run diff and then 1-update smoke.
  2. Add a server preflight command that writes a hardware plan artifact without starting training.
  3. Run the first Linux/Gloo or NCCL 2-rank smoke and inspect global samples/sec, gradient sync time, and rank-0 artifact integrity.
  4. If server all-reduce overhead is high, evaluate a custom fused/flattened gradient bucket path or a proper DDP adapter for only the plain forward module while preserving raw structured methods.

## 2026-04-24 Documentation: Server Scaling Watchouts

- Added `notes/server_scaling_watchouts_2026-04-24.md`.
- Purpose:
  - preserve the expected L40 bottleneck order before server time is spent;
  - list the exact metrics to watch on first Linux multi-GPU smoke;
  - define hard-fail artifact checks for DDP/autoscale correctness;
  - capture decision rules for whether to promote, debug, or redesign.
- Key expectation recorded:
  - if gradient sync is small, learner compute/scorer/backward is the next bottleneck;
  - if gradient sync is large, fused gradient buckets or a careful DDP adapter become the next target;
  - if collector wait climbs again, actor topology, host batch build, or simulator-native rollout CPU time is the frontier.

## 2026-04-24 Eval / Dev-Eval / Promotion Deep Dive

- Scope:
  - audited final eval, periodic dev-eval, async dev-eval, promotion gate, and league/promotion interactions after the B1 throughput work.
  - focus was structural correctness and server-scaling risk, not small paired-seed/config tuning.
- Changes:
  - Removed stale hard-coded `cuda:0,cuda:1,cuda:2` eval/gate worker lists from `configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml`.
    - The auto-GPU preset now lets runtime resolution choose devices instead of baking in a 3-GPU shape.
    - This avoids local 1-GPU invalid devices and L40-4 underuse / learner-device contention.
  - Fixed `python/scripts/eval.py` parallel worker resolution:
    - `eval_device=auto` now behaves like `cuda:auto`;
    - explicit CUDA worker devices are validated early against actual local CUDA availability/count;
    - `cuda:auto` falls back to CPU when CUDA is unavailable.
  - Fixed periodic dev-eval worker resolution in `python/scripts/train.py`:
    - explicit worker-device overrides are honored even for a single effective eval worker;
    - async dev-eval request construction now records the non-learner async eval-device choice;
    - explicit invalid CUDA devices fail fast instead of surfacing as worker crashes.
  - Fixed async cleanup correctness:
    - `_process_completed_periodic_dev_eval` and `_process_completed_promotion_gate` now call `future.result()` inside the `try/finally`, so failed worker futures still unpin pinned snapshots.
  - Tightened final-eval policy selection:
    - default dev-eval summary discovery now prefers current `periodic_dev_eval_summaries.json` over legacy `dev_eval_summaries.json`;
    - missing explicit or manifest-sourced snapshot/dev-eval input paths now raise instead of silently falling back;
    - manifest policy-selection reuse now requires `run_summary.canonical_eval_completed=true`, not merely a stray `final_eval/summary.json`.
  - Updated nearby stale tests:
    - periodic dev-eval context expectations include emitted matchup/episodes artifact paths;
    - minimal-training bootstrap fixture passes the now-required wall-clock budget argument.
- Validation:
  - Targeted:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_entrypoints.py::test_parallel_eval_worker_devices_treat_auto_as_cuda_auto python/weiss_rl/tests/test_entrypoints.py::test_parallel_eval_worker_devices_reject_invalid_explicit_cuda python/weiss_rl/tests/test_entrypoints.py::test_eval_entrypoint_rejects_summary_only_manifest_reuse python/weiss_rl/tests/test_entrypoints.py::test_eval_entrypoint_prefers_current_periodic_dev_eval_summaries python/weiss_rl/tests/test_entrypoints.py::test_eval_entrypoint_rejects_missing_explicit_dev_eval_summaries python/weiss_rl/tests/test_train_stall_monitor.py::test_failed_async_periodic_dev_eval_unpins_snapshots python/weiss_rl/tests/test_train_stall_monitor.py::test_failed_async_promotion_gate_unpins_snapshots python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_single_worker_honors_worker_device_override -q`
    - Result: `42 passed`.
  - Broader entrypoint/config/dev-eval:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_train_stall_monitor.py -q`
    - Result: `106 passed`.
  - Promotion/registry:
    - `uv run pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_promotion_gate.py -q`
    - Result: `37 passed`.
- Verdict:
  - Keep the config/device-selection fixes. They are server-relevant and remove stale 3-GPU assumptions from the standard auto-GPU path.
  - Keep the async cleanup fix. It closes a real artifact/registry leak if async eval or promotion workers fail.
  - Keep the final-eval selection strictness. It reduces risk of evaluating/promoting from stale or typo-induced artifacts.
- Remaining high-value targets:
  1. Eval throughput redesign: persistent/batched eval runner. Current eval still rebuilds envs per game and does batch-size-1 CUDA inference with CPU sampling.
  2. Seed-block sharding: parallel final/dev eval is capped by matchup/anchor count while paired seeds inside each matchup remain serial.
  3. Promotion pool semantics: failed promotion candidates can still enter the recent PFSP reservoir; decide whether strict promotion-gated mode should exclude rejected candidates.
  4. Async checkpoint guard semantics: async dev-eval can publish best checkpoints for older updates, but rollback/finalize behavior is weaker when results arrive after the learner has advanced.
  5. Parallel final eval startup cost: current matchup shards can duplicate policy loads across workers; device-affine model ownership or seed-block sharding would reduce RAM/VRAM churn.

## 2026-04-24 Eval Throughput Iteration: Persistent Env Reuse + Rejection Semantics

- Changed:
  - Added persistent single-env reuse for `SimulatorEvalRunner` when replay capture is disabled.
    - Final eval still uses per-game envs when replay capture/regression capture is active to avoid replay state leakage.
    - Serial and parallel final eval helper call sites now close runners explicitly when the worker/matchup finishes.
  - Added persistent env reuse to `_PeriodicDevEvalRunner` and `_PromotionGateRunner`.
    - Hidden states and deterministic per-seat PCG RNGs are still recreated per scheduled game.
    - Each game still calls `env.reset(seed=scheduled_game.episode_seed)`, preserving the existing seeded protocol.
  - Added periodic dev-eval runtime diagnostics:
    - per-anchor `evaluation_runtime.wall_clock_seconds`, `games_per_sec`, `game_count`, `persistent_env_reuse`;
    - summary-level `periodic_dev_eval_runtime`.
  - Added promotion rejection state to `SnapshotRegistry`.
    - Failed async and sync promotion gates now call `registry.reject_snapshot(candidate_policy_id)`.
    - `registry.add_champion(...)` clears any prior rejection.
    - Promotion-gated `QueueRuntime.refresh_opponent_pool()` excludes rejected snapshots from the recent reservoir.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/scripts/eval.py python/scripts/b2_disagreement_audit.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/league/registry.py python/weiss_rl/runtime.py`
    - Result: passed.
  - Focused tests:
    - `uv run pytest python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_tracks_rejections_and_clears_on_champion python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_excludes_rejected_recent_when_promotion_gate_enabled python/weiss_rl/tests/test_train_stall_monitor.py::test_failed_async_promotion_gate_marks_candidate_rejected python/weiss_rl/tests/test_snapshot_registry.py::test_simulator_eval_runner_reuses_env_when_replay_capture_disabled python/weiss_rl/tests/test_snapshot_registry.py::test_simulator_eval_runner_does_not_reuse_env_when_replay_capture_enabled python/weiss_rl/tests/test_snapshot_registry.py::test_periodic_dev_eval_runner_reuses_env_across_games -q`
    - Result: `6 passed`.
  - Broader relevant suite:
    - `uv run pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `205 passed`.
  - Real local dev-eval smoke:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_persistent_env_diag_smoke --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=1`
    - Artifact: `runs/eval_persistent_env_diag_smoke`.
    - Train update throughput: `17,459.27 samples/sec` on the single update.
    - Periodic dev-eval:
      - aggregate score `0.75`;
      - `12` games across 6 anchors;
      - `periodic_dev_eval_runtime.wall_clock_seconds=27.5355`;
      - `periodic_dev_eval_runtime.games_per_sec=0.4358`;
      - all anchor runtime payloads report `persistent_env_reuse=true`.
    - Slowest anchors:
      - `Previous recent snapshot`: `7.5622s` for 2 games, `0.2645 games/sec`;
      - `B1 NoLeague baseline`: `6.8785s` for 2 games, `0.2908 games/sec`.
- Verdict:
  - Keep persistent env reuse. It removes a real per-game simulator setup cost and is gated away from replay capture.
  - Keep rejection semantics. It fixes the promotion-gated pool contract so rejected candidates do not keep re-entering as recent PFSP opponents.
  - The local smoke shows the next eval bottleneck is not env construction anymore: learned-model anchors remain extremely slow because eval is still serial and scalar.
- Next hypotheses:
  1. Highest-value eval throughput redesign is now batched or seed-block eval for learned-model anchors.
     - Target B1/recent snapshot anchors first; they are the slowest in the smoke.
  2. Add model-forward timing inside eval runners before rewriting batching, so we can separate simulator step time from per-decision PyTorch overhead.
  3. For server scaling, dev-eval/promotion should shard paired seeds, not only anchors/matchups; current parallelism is capped by anchor count.
  4. Async checkpoint-guard semantics remain the main promotion/eval correctness issue: older async dev-eval results can publish best checkpoints but do not always drive rollback/finalization.

## 2026-04-24 Eval Throughput Iteration: Seed-Block Sharding + Runner Timing

- Changed:
  - Added periodic dev-eval runner timing counters for:
    - `model_forward`, `logits_to_cpu`, `sample_action`, `env_reset`, `env_step`;
    - decision/action counts split by focal model, opponent model, heuristic, and random-legal actions.
  - Reworked periodic dev-eval parallelism so paired seeds can shard beyond the number of anchors.
    - New job shape is `(anchor, seed block)` instead of only `anchor`.
    - Parent process writes canonical matchup summaries/episodes after workers return records, preserving global pair indices and deterministic artifact shape.
  - Fixed a dev-eval artifact collision:
    - when two anchors share the same policy id, such as `B1 NoLeague baseline` and `Previous recent snapshot`, their matchup dirs now get unique stable suffixes instead of overwriting each other.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/scripts/eval.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/league/registry.py python/weiss_rl/runtime.py`
    - Result: passed.
  - Focused:
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_single_worker_honors_worker_device_override python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_seed_block_jobs_expand_beyond_anchor_count python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_duplicate_policy_ids_get_unique_matchup_dirs -q`
    - Result: `3 passed`.
  - Broader:
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py -q`
    - Result: `75 passed`.
    - `uv run pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `207 passed`.
- Benchmarks:
  - Same local surface as the persistent-env smoke, but with `evaluation.periodic_dev_eval_paired_seeds=2`.
  - Serial:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_seedblock_serial_s2 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=1`
    - Artifact: `runs/eval_seedblock_serial_s2`.
    - Dev-eval: `24` games, `57.3708s`, `0.4183 games/sec`.
    - Slow learned anchors:
      - `B1 NoLeague baseline`: `15.579s` for 4 games; `model_forward=15.304s`.
      - `Previous recent snapshot`: `14.648s` for 4 games; `model_forward=14.374s`.
  - Anchor-parallel, 6 workers:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_seedblock_anchorparallel_s2_w6 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=6`
    - Artifact: `runs/eval_seedblock_anchorparallel_s2_w6`.
    - Dev-eval: `24` games, `35.4923s`, `0.6762 games/sec`.
    - Parallel section reports `job_count=6`, `seed_block_sharding_enabled=false`.
  - Anchor + seed-block, 12 workers:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_seedblock_w12_s2 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=12`
    - Artifact: `runs/eval_seedblock_w12_s2`.
    - Dev-eval: `24` games, `34.5889s`, `0.6939 games/sec`.
    - Parallel section reports `job_count=12`, `seed_block_sharding_enabled=true`.
    - Duplicate-policy anchors wrote distinct artifact dirs:
      - `b1_noleague_baseline__04f13886483989a9`
      - `b1_noleague_baseline__9122c87acf1cb278`
- Verdict:
  - Keep the seed-block sharding implementation and timing diagnostics.
  - Local 1-GPU speedup is modest beyond 6 workers because all eval workers contend on the same RTX 5080 and duplicate model forwards become slower:
    - 6 workers: `35.49s`;
    - 12 workers: `34.59s`.
  - The server-relevant win is structural: dev-eval can now expose more independent work than the anchor count, so a 3-4 GPU L40 machine can actually keep more eval workers/device slots busy when paired-seed counts are high.
  - The new counters identify the next eval bottleneck clearly:
    - simulator `env_step` is tiny compared with model forward;
    - learned-anchor eval is dominated by scalar recurrent model inference.
- Risks:
  - On a single local GPU, high worker counts duplicate model memory and can increase per-anchor model-forward latency.
  - The current seed-block design still loads one eval model per worker process; server profiling should watch eval worker startup and VRAM duplication.
- Next hypotheses:
  1. The next eval throughput frontier is model ownership / batching:
     - persistent per-device eval workers that own loaded models;
     - microbatch seat-aware inference across independent games where recurrent hidden states are grouped by policy/device.
  2. For server runs, test `periodic_dev_eval_parallel_workers=12` or higher only after a preflight confirms VRAM headroom on L40s.
  3. Apply the same seed-block sharding idea to promotion gate if promotion eval becomes a visible wall-clock tax.


## 2026-04-24 Eval Throughput Iteration: Experimental Batched Dev-Eval Inference

- Changed:
  - Added `evaluation.periodic_dev_eval_batched_inference_enabled` as an explicit, default-off flag.
  - Implemented a batched periodic dev-eval runner path that keeps one single-env simulator per scheduled game, preserving per-game episode seeds while batching ready model decisions into one `forward_seat_aware` call per policy/model.
  - The implementation deliberately does not use a multi-row simulator env because `DecisionBoundaryEnv.reset(seed=...)` broadcasts one seed to all rows and `reset_done(...)` cannot supply replacement episode seeds.
  - Added runner counters for `model_forward_calls` and `model_forward_rows` so we can see effective batch shape.
  - Added regression coverage that two independent focal decisions are served by one batched `forward_seat_aware` call with batch size 2 while preserving episode order and closing both envs.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py`
    - Result: passed.
  - Focused/config:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_single_worker_honors_worker_device_override python/weiss_rl/tests/test_snapshot_registry.py::test_periodic_dev_eval_runner_batches_independent_model_decisions -q`
    - Result: `36 passed`.
  - Broader relevant suite:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `242 passed`.
- Benchmarks:
  - Canonical flag-off serial S2:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_batchedflag_off_serial_s2 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=1`
    - Artifact: `runs/eval_batchedflag_off_serial_s2`.
    - Result: aggregate `0.8333`, `24` games, `58.362s`, `0.411 games/sec`.
    - This reproduced the previous scalar/canonical surface.
  - Experimental batched serial S2:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_batchedflag_on_serial_s2 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=1 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=true`
    - Artifact: `runs/eval_batchedflag_on_serial_s2`.
    - Result: aggregate `0.4583`, `24` games, `45.104s`, `0.532 games/sec`.
    - Speed improved by about `1.29x` vs flag-off serial, but the score surface changed substantially.
  - Experimental batched 6-worker S2:
    - Artifact: `runs/eval_batchedflag_on_w6_s2`.
    - Result: aggregate `0.4583`, `24` games, `29.525s`, `0.813 games/sec`.
    - Faster than old 6-worker scalar (`35.492s`, `0.676 games/sec`), but still on the changed batched stochastic surface.
  - Experimental batched 12-worker S2:
    - Artifact: `runs/eval_batchedflag_on_w12_s2`.
    - Result: aggregate `0.7917`, `24` games, `33.568s`, `0.715 games/sec`.
    - Only slightly faster than old 12-worker scalar (`34.589s`, `0.694 games/sec`) on the local one-GPU box.
- Verdict:
  - Do not promote batched dev-eval inference as the canonical/default eval path yet.
  - Keep the implementation behind the explicit default-off flag because it is useful experimental infrastructure and all tests pass.
  - The speed signal is real, but batched model inference changes the exact pinned stochastic outcomes; on short noisy evals this can move anchor scores materially.
  - Current canonical eval/progression/promotion should remain flag-off until we either:
    1. accept a new batched eval surface and re-anchor all comparisons on it, or
    2. design a stricter parity batching protocol that proves action/logit equivalence is not materially changing decisions.
- Diagnostics:
  - The speed gain comes from reducing scalar model calls; flag-on serial showed `model_forward_calls` less than `model_forward_rows` for every anchor.
  - Simulator/env time remains tiny relative to model forward.
- Risks:
  - Batched GPU math is not bit-identical to scalar GPU math, and stochastic sampling can amplify tiny logit differences into different action trajectories.
  - Running many local workers on one GPU can mask the server shape; this flag needs an L40 smoke only after deciding whether the changed eval surface is acceptable.
- Next hypotheses:
  1. If we want exact canonical parity, try a persistent per-device eval service without batching logits first: keep scalar action semantics but remove model load/process churn.
  2. If we accept a new eval surface, re-anchor all dev-eval/promotion baselines with batched inference enabled and require bounded agreement over larger paired-seed counts before promotion.
  3. Promotion gate can reuse this flag later, but only after the dev-eval semantics decision.

## 2026-04-24 Eval Throughput Iteration: Two-Lane Fast Screen + Canonical Confirmation

- Changed:
  - Added explicit eval-surface metadata to periodic dev-eval summaries:
    - `canonical_scalar`, `authoritative=true` for normal scalar eval;
    - `fast_batched_screen`, `authoritative=false` for batched inference.
  - Non-authoritative fast batched summaries are no longer eligible for:
    - checkpoint best promotion;
    - checkpoint guard rollback/finalize;
    - canonical `periodic_dev_eval_summaries.json`.
  - Added `training/logs/periodic_dev_eval_fast_screens.json` for fast-screen telemetry.
  - Added a fast-screen confirmation gate:
    - if a batched screen looks decision-relevant (`candidate_best`, `rollback_risk`, or `confidence_risk`), rerun scalar canonical dev-eval before publishing/promotion/rollback logic can use it;
    - canonical confirmation writes under `eval/dev_eval_canonical/...`;
    - regular confidence confirmatory eval remains scalar and still writes under `eval/dev_eval_confirmatory/...` when needed.
  - Batched inference remains behind `evaluation.periodic_dev_eval_batched_inference_enabled`; default remains `false`.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py`
    - Result: passed.
  - Focused:
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py::test_fast_batched_dev_eval_screen_is_non_authoritative_and_requests_scalar_confirmation python/weiss_rl/tests/test_train_stall_monitor.py::test_fast_batched_dev_eval_screen_does_not_confirm_boring_middle_score python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_single_worker_honors_worker_device_override python/weiss_rl/tests/test_snapshot_registry.py::test_periodic_dev_eval_runner_batches_independent_model_decisions -q`
    - Result: `4 passed`.
  - Broader:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `154 passed`.
- Smoke:
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_twolanefast_s2_smoke --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=1 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=true`
  - Artifact: `runs/eval_twolanefast_s2_smoke`.
  - Fast batched screen:
    - artifact: `eval/dev_eval/update_1/summary.json`;
    - surface: `fast_batched_screen`, `authoritative=false`;
    - aggregate `0.4583`;
    - `24` games, `47.0590s`, `0.5100 games/sec`.
  - Canonical scalar confirmation:
    - artifact: `eval/dev_eval_canonical/update_1/summary.json`;
    - surface: `canonical_scalar`, `authoritative=true`;
    - aggregate `0.8333`;
    - `24` games, `57.6038s`, `0.4166 games/sec`.
  - Log separation:
    - fast screen persisted to `training/logs/periodic_dev_eval_fast_screens.json`;
    - canonical confirmation persisted to `training/logs/periodic_dev_eval_summaries.json`;
    - B2 disagreement audit request used canonical episodes: `eval/dev_eval_canonical/update_1/B2 HeuristicPublic/episodes.jsonl`.
- Verdict:
  - Keep the two-lane implementation.
  - This preserves the batched speedup for screening without allowing batched stochastic drift to affect authoritative thesis/promotion/checkpoint decisions.
  - On a one-update smoke with no prior best, the fast screen immediately triggered scalar confirmation (`confidence_risk`), so total wall-clock is naturally slower than scalar-only. In real longer runs, the speedup comes from skipping scalar confirmation for boring middle updates while still confirming decision-relevant updates.
- Risks:
  - The fast-screen trigger is intentionally conservative. It may confirm more often than strictly needed early in training, especially before a stable best checkpoint exists.
  - Batched inference remains a different stochastic surface; do not compare fast-screen scores directly against canonical scalar baselines.
- Next hypotheses:
  1. Add counters for how often fast screens trigger canonical confirmation during longer local runs; tune confirmation thresholds only after seeing that rate.
  2. For server use, run fast-screen async on non-learner GPU(s) and only schedule canonical scalar confirmation when the screen is decision-relevant.
  3. If confirmation frequency is too high, require two consecutive fast-screen triggers before scalar confirmation for rollback-risk only; keep candidate-best confirmation immediate.

## 2026-04-24 Eval Throughput Iteration: Quarantine Batched Screens + Persistent Scalar Eval Pool

- Startup restatement:
  - Best known bottleneck:
    - Training throughput is no longer actor-policy-forward bound on the B1 no-league native-rollout surface.
    - Eval/dev-eval is dominated by scalar recurrent model inference, especially learned-vs-learned anchors.
  - Best known learning branch:
    - The B1 no-league native-rollout branch remains the promoted short-horizon anchor path; local quality probes were good enough to keep it alive, but not server-grade proof.
  - Strongest open hypotheses:
    1. Canonical eval throughput should improve through persistent worker ownership and less process/model churn, not through simulator/env micro-optimizations.
    2. Batched eval inference is useful for diagnostics but changes the stochastic surface enough that it should not drive promotion/checkpoint decisions.
    3. The next high-leverage eval design, if needed, is a true persistent per-device eval service that owns loaded models and can later batch only after an equivalence protocol exists.
  - Suspicious config/runtime choices:
    - `evaluation.periodic_dev_eval_batched_inference_enabled` is a dangerous production default if treated as canonical; it must remain explicit and non-authoritative.
    - A two-lane fast-screen-plus-scalar-confirm path is too much complexity if the fast screen commonly triggers scalar confirmation.
    - Sync benchmark lanes can reuse worker pools, but server presets often use async scheduling or disable eval overlap, so persistence must be live in the async path before making server claims.
- Changed:
  - Removed the automatic two-lane scalar confirmation path for batched periodic dev-eval.
    - Batched dev-eval still writes `evaluation_surface.kind=fast_batched_screen`.
    - Batched dev-eval remains `authoritative=false`.
    - Batched dev-eval is ineligible for checkpoint best promotion, checkpoint guard rollback/finalization, and B2 disagreement audit requests.
    - Fast-screen payloads only persist to `training/logs/periodic_dev_eval_fast_screens.json`.
  - Added a process-local snapshot eval model cache keyed by resolved path, mtime, size, device, observation dim, and action dim.
  - Added a reusable scalar periodic dev-eval `ProcessPoolExecutor` for the sync path.
  - Extended the same reusable scalar process pool into the async periodic dev-eval thread path, so async dev-eval can avoid recreating the seed-block worker pool every checkpoint.
  - Confirmatory scalar dev-eval now also receives the reusable pool when available.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py`
    - Result: passed.
  - Focused/config:
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py::test_fast_batched_dev_eval_screen_is_non_authoritative python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_single_worker_honors_worker_device_override python/weiss_rl/tests/test_snapshot_registry.py::test_periodic_dev_eval_runner_batches_independent_model_decisions -q`
    - Result: `3 passed`.
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py -q`
    - Result: `34 passed`.
  - Broader relevant suite:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `153 passed`.
- Benchmarks / smokes:
  - Canonical scalar, sync reusable pool, S2:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_persist_scalar_pool_w6_s2_smoke --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=2 --config-override evaluation.periodic_dev_eval_parallel_workers=6 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=false`
    - Artifact: `runs/eval_persist_scalar_pool_w6_s2_smoke`.
    - Result: aggregate `0.8333`, `24` games, `34.283s`, `0.700 games/sec`, surface `canonical_scalar`, authoritative `true`.
    - Same-surface comparison: previous scalar 6-worker S2 was `35.492s`, `0.676 games/sec`.
  - Canonical scalar, sync reusable pool across two evals, S1:
    - Artifact: `runs/eval_persist_scalar_pool_w6_s1_u2_smoke`.
    - Update 1: aggregate `0.7500`, `12` games, `19.140s`, `0.627 games/sec`.
    - Update 2: aggregate `0.5833`, `12` games, `17.727s`, `0.677 games/sec`.
  - Canonical scalar, async reusable pool across two evals, S1:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_persist_scalar_async_pool_w6_s1_u2_smoke --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 2 --checkpoint-interval-updates 1 --max-wall-clock-minutes 3 --profile-timers --config-override evaluation.async_periodic_dev_eval_enabled=true --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=6 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=false`
    - Artifact: `runs/eval_persist_scalar_async_pool_w6_s1_u2_smoke`.
    - Update 1: aggregate `0.7500`, `12` games, `19.229s`, `0.624 games/sec`.
    - Update 2: aggregate `0.5833`, `12` games, `17.465s`, `0.687 games/sec`.
  - Batched screen quarantine smoke:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label eval_batched_screen_quarantined_s1_smoke --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 2 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=1 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=1 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=true`
    - Artifact: `runs/eval_batched_screen_quarantined_s1_smoke`.
    - Result: aggregate `0.6667`, `12` games, `30.189s`, `0.397 games/sec`, surface `fast_batched_screen`, authoritative `false`.
    - No `eval/dev_eval_canonical` directory was produced.
    - No `training/logs/periodic_dev_eval_summaries.json` was produced.
    - Fast-screen payload was written to `training/logs/periodic_dev_eval_fast_screens.json`.
- Verdict:
  - Keep the quarantine cleanup.
    - It removes complexity from the production decision path while preserving batched eval as an explicit, non-authoritative diagnostic.
  - Keep the persistent scalar process pool and snapshot model cache.
    - The local speedup is small, not a breakthrough, but the design is strictly better for repeated canonical scalar evals and now applies to both sync and async periodic dev-eval paths.
  - Do not promote batched eval as a production or thesis comparison surface.
    - It remains a useful experimental flag only.
- Risks:
  - Snapshot model cache has no LRU eviction yet. That is acceptable for short local smokes and stable anchors, but a long eval service with many unique snapshot paths should get an explicit memory cap.
  - The one-GPU Windows box is still a poor proxy for L40 eval scaling; multiple local workers contend on the same GPU.
  - Persistent workers reduce churn but do not solve the dominant scalar recurrent forward bottleneck.
- Next hypotheses:
  1. If eval cost remains painful on server, build a real persistent per-device eval service with long-lived model ownership and explicit VRAM accounting.
  2. Revisit batching only if we can define a precise canonical protocol or intentionally re-anchor all eval/promotion comparisons on the new batched stochastic surface.
  3. Apply seed-block sharding and persistent workers to promotion gate if promotion wall-clock becomes visible in server profiling.

## 2026-04-24 Throughput Hardening: Promotion Seed Blocks, Cache Cap, Explicit Server Eval Policy

- Changed:
  - Promotion gate parallelism now uses `(anchor, seed block)` jobs instead of only anchor shards.
    - This removes the previous cap where `parallel_workers` could not expose more independent promotion work than the number of anchors.
    - Parent process now assembles canonical promotion episode files and `PromotionGateResult` payloads after workers return records.
    - The promotion decision surface remains scalar/canonical; this is sharding only, not batched inference.
  - Added LRU eviction to the process-local eval snapshot model cache.
    - Cache key remains resolved snapshot path, mtime, size, device, observation dim, and action dim.
    - Max entries are currently capped at `12` per process.
  - Made the server training eval/promotion overlap policy explicit in `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu.yaml`.
    - `evaluation.async_periodic_dev_eval_enabled: false`
    - `evaluation.periodic_dev_eval_interval_updates: 20`
    - `evaluation.periodic_dev_eval_parallel_workers: 6`
    - `evaluation.periodic_dev_eval_batched_inference_enabled: false`
    - `league.promotion.gate.async_enabled: false`
    - `league.promotion.gate.parallel_workers: 6`
  - Updated `notes/server_scaling_watchouts_2026-04-24.md` with the explicit eval/promotion overlap policy.
- Validation:
  - Syntax:
    - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py`
    - Result: passed.
  - Focused:
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py::test_promotion_gate_seed_block_jobs_expand_beyond_anchor_count python/weiss_rl/tests/test_train_stall_monitor.py::test_eval_snapshot_model_cache_eviction_is_lru python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets -q`
    - Result: `3 passed`.
    - `uv run pytest python/weiss_rl/tests/test_train_stall_monitor.py::test_parallel_promotion_gate_assembles_seed_block_records_parent_side -q`
    - Result: `1 passed`.
  - Promotion/config:
    - `uv run pytest python/weiss_rl/tests/test_promotion_gate.py python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_marks_passed_candidate_as_champion python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_skips_during_warmup python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_uses_effective_update_for_warmup python/weiss_rl/tests/test_snapshot_registry.py::test_promotion_gate_runner_resets_env_with_scheduled_episode_seed python/weiss_rl/tests/test_snapshot_registry.py::test_promotion_gate_runner_uses_learner_scoring_mode -q`
    - Result: `11 passed`.
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py -q`
    - Result: `34 passed`.
  - Broader relevant suite:
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_promotion_gate.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_final_eval.py -q`
    - Result: `162 passed`.
  - Autoscale sanity:
    - `uv run pytest python/weiss_rl/tests/test_autoscale.py python/weiss_rl/tests/test_distributed.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets -q`
    - Result: `7 passed`.
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --runtime-mode train_async_fast --autoscale-dry-run --hardware-profile uc1-l40-4 --max-updates 1 --unroll-length 16`
    - Result: resolved `uc1-l40-4` topology remains `32` actors x `64` envs = `2048` total envs, `4` learner GPUs, `resolved_learner_parallelism=ddp`.
- Verdict:
  - Keep the promotion seed-block sharding.
    - It is structurally server-relevant and preserves canonical scalar promotion semantics.
  - Keep the LRU eval snapshot model cache cap.
    - It closes the long-run memory-growth risk without changing eval decisions.
  - Keep the explicit no-overlap server preset defaults.
    - This makes first L40 throughput claims cleaner; async overlap can be enabled later as a deliberate server experiment if rank-0/eval wall-clock dominates.
- Risks:
  - Async promotion gate still uses its outer async worker model. The default server preset keeps promotion sync/no-overlap, so this is acceptable for first server throughput runs.
  - Promotion seed-block sharding is unit-tested locally with parent-side artifact assembly; full wall-clock speedup still needs server profiling because local one-GPU worker contention is misleading.
- Next:
  - Move back to learning-quality work unless server preflight exposes a new runtime bottleneck.

### Promotion Sharding Local Benchmark Follow-Up

- Motivation:
  - The initial hardening pass validated promotion seed-block sharding with tests, but did not yet run a wall-clock promotion benchmark.
  - This follow-up checks whether the new promotion sharding path helps locally before we treat it as a real throughput improvement.
- Setup:
  - Built a matching full-model B1 no-league baseline for import:
    - Command:
      - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague.yaml --run-label promo_bench_b1_baseline_u1 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 2 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=0`
    - Artifact: `runs/promo_bench_b1_baseline_u1`.
  - First attempted to override the server preset onto the 8-seed local promotion file from the CLI, but this failed before promotion because `stack.seed_sets['promotion_gate']` still pointed at `configs/seeds/promotion_eval_seeds.txt`.
    - Failed artifact: `runs/promo_bench_anchorcap_w2_s8`.
    - Verdict: do not use ad hoc CLI seed-file overrides for promotion timing unless `stack.seed_sets`, `evaluation.seed_files`, `league.promotion.seed_file`, and `reproducibility.seed_files` all agree.
  - Added temporary benchmark stack:
    - `runs/_promotion_bench_server_seed8.yaml`
    - Extends the server training preset, but uses `configs/seeds/local_promotion_eval_seeds.txt`, `league.warmup.first_updates=0`, and `evaluation.periodic_dev_eval_interval_updates=0`.
- Benchmark:
  - 2-worker promotion:
    - Command:
      - `uv run python python/scripts/train.py --stack-config runs/_promotion_bench_server_seed8.yaml --run-label promo_bench_seedblock_w2_s8 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 5 --profile-timers --b1-baseline-run-dir runs/promo_bench_b1_baseline_u1 --config-override league.promotion.gate.parallel_workers=2`
    - Artifact: `runs/promo_bench_seedblock_w2_s8`.
    - End-to-end process wall-clock from command runner: `223.4s`.
    - Training update wall-clock metric: about `35.99s`.
    - Approx promotion/import/post-update remainder: about `187s`.
    - Promotion wrote 6 anchor episode files x 16 games each.
  - 6-worker promotion:
    - Command:
      - `uv run python python/scripts/train.py --stack-config runs/_promotion_bench_server_seed8.yaml --run-label promo_bench_seedblock_w6_s8 --runtime-mode train_async_fast --device auto --num-envs 128 --unroll-length 16 --max-updates 1 --checkpoint-interval-updates 1 --max-wall-clock-minutes 5 --profile-timers --b1-baseline-run-dir runs/promo_bench_b1_baseline_u1 --config-override league.promotion.gate.parallel_workers=6`
    - Artifact: `runs/promo_bench_seedblock_w6_s8`.
    - End-to-end process wall-clock from command runner: `173.8s`.
    - Training update wall-clock metric: about `35.81s`.
    - Approx promotion/import/post-update remainder: about `138s`.
    - Promotion wrote the same 6 anchor episode files x 16 games each.
- Result:
  - Local end-to-end speedup: `223.4s -> 173.8s`, about `1.29x` or `22%` faster.
  - Approx promotion-heavy remainder speedup: `187s -> 138s`, about `1.36x` or `26%` faster.
- Verdict:
  - Keep the promotion seed-block sharding.
  - This is local-only and still contends on one GPU, but it is no longer just a structural hypothesis: the same promotion surface got materially faster with more seed-block workers.
- Risk / next instrumentation:
  - Promotion wall-clock is not yet written as an explicit artifact metric; the estimate above uses command wall-clock minus training update wall-clock.
  - A future cleanup should add `promotion_gate_runtime.wall_clock_seconds`, `job_count`, `worker_count`, and `seed_block_sharding_enabled` to the promotion gate record so this does not rely on shell timing.

## 2026-04-24 Learning-Quality Pivot: B1 Mulligan Teacher Contract

- Startup stance:
  - Throughput work is good enough for local learning iteration.
  - The active risk is whether the B1 no-league anchor is optimizing the right signal, not whether local collection can be made a little faster.
  - `AGENTS.md` is absent in this checkout; the thread-pasted autonomous research instruction is the active operating rule.
- Same-surface control diagnostic:
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_learning_diag_current_fast_u60_s1_20260424 --runtime-mode train_async_fast --device auto --num-envs 512 --unroll-length 16 --max-updates 60 --checkpoint-interval-updates 10 --max-wall-clock-minutes 8 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=10 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=6 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=false`
  - Artifact:
    - `runs/b1_learning_diag_current_fast_u60_s1_20260424`
  - Dev-eval curve, one paired seed, canonical scalar:
    - u10/u20/u40/u50/u60 aggregate `0.75`;
    - u30 aggregate `0.6667`;
    - B2 score stayed `1.0`, B1/self anchor mostly `0.5`, B3 mostly `0.5`.
  - Health metrics:
    - no collector no-progress/natural/tick timeout rows;
    - no eval truncations;
    - u60 `reward_abs_mean=0.01836`, `target_abs_mean=0.29785`, `value_loss=0.01266`;
    - u60 `teacher_action_accuracy=0.96835`, `teacher_family_accuracy=0.96835`;
    - u60 `structured_main_move_mass=0.00385`, `collector_main_move_actions=0`.
  - B2 audit:
    - Command needed `config_canonical.json` plus `policy_000006`; the auto-requested command using the base preset and `train_u60_p6` was not directly replayable.
    - Artifact:
      - `runs/b1_learning_diag_current_fast_u60_s1_20260424/eval/b2_disagreement_audit/update_60/audit/summary.json`
    - Result:
      - mean total variation `0.6176`;
      - top inspected family mismatch was entirely `mulligan_select` vs `mulligan_confirm` (`40/40` top examples);
      - learner probability on B2 top action was only about `0.36-0.40`.
- Failed structural config-only idea:
  - Hypothesis:
    - Adding `mulligan_confirm` to `training.structured_aux.teacher_public_heuristic_families` and `training.structured_warmstart.teacher_public_heuristic_families` might fix the audit mismatch.
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_learning_diag_mulliganconfirm_teacher_u60_s1_20260424 --runtime-mode train_async_fast --device auto --num-envs 512 --unroll-length 16 --max-updates 60 --checkpoint-interval-updates 10 --max-wall-clock-minutes 8 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=10 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=6 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=false --config-override 'training.structured_aux.teacher_public_heuristic_families=["mulligan_confirm","mulligan_select","clock_from_hand","main_play_character","main_play_event","climax_play","main_move","attack","level_up","encore_pay","encore_decline","trigger_order","choice_select"]' --config-override 'training.structured_warmstart.teacher_public_heuristic_families=["mulligan_confirm","mulligan_select","clock_from_hand","main_play_character","main_play_event","climax_play","main_move","attack","level_up","encore_pay","encore_decline","trigger_order","choice_select"]'`
  - Artifact:
    - `runs/b1_learning_diag_mulliganconfirm_teacher_u60_s1_20260424`
  - Verdict:
    - Kill this idea as insufficient.
    - Eval curve was numerically identical to the control.
    - B2 audit still showed `mulligan_select` vs `mulligan_confirm` as all top inspected mismatches.
  - Diagnosis:
    - The config key was live, but runtime teacher labeling excluded decision kind `0`, so mulligan rows were structurally unreachable for teacher labels.
- Code change:
  - Changed `_PUBLIC_TEACHER_DECISION_KINDS` in `python/weiss_rl/runtime.py` from `{1,2,3,4,5,6,7,8}` to `{0,1,2,3,4,5,6,7,8}`.
  - Added regression coverage in `python/weiss_rl/tests/test_runtime.py`:
    - decision kind `0` now labels `mulligan_confirm` and `mulligan_select`;
    - the prior unlabeled sentinel case now uses decision kind `-1`.
  - Fixed B2 audit request command emission in `python/scripts/train.py`:
    - future requests prefer `<run>/config_canonical.json`, preserving CLI overrides in the config hash;
    - future requests include `audit_policy_id=policy_000NNN` and pass that registry-resolvable policy id to the audit script.
- Tests:
  - `uv run pytest python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_mulligan_decision_kind_zero -q`
    - Result: `2 passed`.
  - `uv run pytest python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_mulligan_decision_kind_zero python/weiss_rl/tests/test_train_stall_monitor.py::test_maybe_request_b2_disagreement_audit_on_confidence_only_gate python/weiss_rl/tests/test_train_stall_monitor.py::test_maybe_request_b2_disagreement_audit_on_flatline_writes_request -q`
    - Result: `4 passed`.
- Patched same-surface diagnostic:
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_learning_diag_mulligan_teacherlive_u60_s1_20260424 --runtime-mode train_async_fast --device auto --num-envs 512 --unroll-length 16 --max-updates 60 --checkpoint-interval-updates 10 --max-wall-clock-minutes 8 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=10 --config-override evaluation.periodic_dev_eval_paired_seeds=1 --config-override evaluation.periodic_dev_eval_parallel_workers=6 --config-override evaluation.periodic_dev_eval_batched_inference_enabled=false`
  - Artifact:
    - `runs/b1_learning_diag_mulligan_teacherlive_u60_s1_20260424`
  - Dev-eval curve, one paired seed, canonical scalar:
    - u10 aggregate `0.75`;
    - u20 aggregate `0.6667`;
    - u30/u40 aggregate `0.75`;
    - u50 aggregate `1.0`;
    - u60 aggregate `0.75`, followed by checkpoint-guard rollback to u50.
  - Checkpoint tracker:
    - best: update `50`, policy version `5`, metric `dev_eval_mean=1.0`, source `training/checkpoints/checkpoint_50.pt`;
    - `best_b2`: update `50`, B2 score `1.0`.
  - Metrics:
    - teacher valid fraction rose from about `0.494` in the control to about `0.500-0.502`, confirming mulligan rows are now labeled;
    - public teacher selected fraction stayed similar because `mulligan_confirm` remains intentionally excluded from the public-target family filter;
    - throughput stayed in the same local regime, around `31-32k samples/sec` during late logged updates with periodic scalar eval overhead.
  - B2 audit at u60:
    - Artifact:
      - `runs/b1_learning_diag_mulligan_teacherlive_u60_s1_20260424/eval/b2_disagreement_audit/update_60/audit/summary.json`
    - Result:
      - mean total variation fell from control `0.6176` to `0.2735`;
      - top mismatch was no longer mulligan confirm/select;
      - family match rate rose to about `0.94`, action match rate to about `0.91`, learner probability on B2 top action to about `0.73`.
- Bounded confirmatory final eval:
  - Command:
    - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_learning_diag_mulligan_teacherlive_u60_s1_20260424 --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --b1-baseline-run-dir runs/b1_learning_diag_mulligan_teacherlive_u60_s1_20260424 --paired-seed-limit 4 --parallel-workers 6 --skip-metagame --skip-figures --skip-readiness`
  - Artifact:
    - `runs/b1_learning_diag_mulligan_teacherlive_u60_s1_20260424/eval/final_eval/summary.json`
  - Result:
    - B1 vs B0: `8/8`, mean `1.0`;
    - B1 vs B2: `8/8`, mean `1.0`;
    - B2 vs B0: `8/8`, mean `1.0`;
    - B1 self: `4/8`, mean `0.5`;
    - no truncations or timeout rows in these matchups.
  - Read:
    - Still not thesis-grade; this is only 4 paired seeds.
    - Strong enough to keep the patch and to run a larger confirmation before using server compute.
- Current best diagnosis:
  - The B1 benchmark had a real teacher-signal contract mismatch: configs requested mulligan teacher families, but runtime teacher labeling excluded decision kind `0`.
  - This left a large, audit-visible mulligan policy disagreement against B2.
  - Making mulligan labels live materially reduced B2 disagreement and produced a better local best checkpoint, without changing throughput posture.
- Risks:
  - The u50 improvement may still be seed noise; do not promote it as a thesis claim from one local 4-pair eval.
  - `main_move` remains dead in native B1 data (`collector_main_move_actions=0`), so movement may still need a separate causal pass.
  - The patched public teacher still excludes `mulligan_confirm` from the soft public-target family filter; confirm is now taught via family/action teacher losses, not via public soft-target CE.
- Next hypotheses:
  1. Run a stronger local confirmatory eval of `b1_learning_diag_mulligan_teacherlive_u60_s1_20260424` with at least 8 or 16 paired seeds before spending server budget.
  2. Run a 100-200 update local continuation with periodic eval every 10-20 updates to see whether the u50 best is stable or whether the policy decays after the mulligan fix.
  3. Add per-family teacher/action diagnostics for mulligan and other non-main families; current scalar metrics only expose play/move/attack and hide exactly the family that failed here.
  4. Separately investigate why `main_move` remains zero in B1 native-rollout data and whether this is a heuristic limitation, environment-state distribution issue, or action-contract issue.

## 2026-04-24 B1 Loop: Native Recorded-Label Contract and Current Best Candidate

- Hypothesis:
  - Simulator-native rollout records actual native heuristic actions in `trajectory.actions`, but the runtime was recomputing Python `HeuristicPublicPolicy` actions to create teacher labels.
  - If native and Python choices diverge, the learner sees behavior from one policy and supervised labels from another.
  - Labeling native-rollout teacher rows from the recorded native actions is throughput-safe and server-scalable: it removes Python teacher action recomputation and does not add actor-side model inference.
- Code change:
  - `python/weiss_rl/runtime.py`
    - Added `_public_teacher_rows(...)` for the public/focal decision-kind filter.
    - Reused it in `_teacher_labels_from_ids(...)` and `_teacher_labels_from_mask(...)`.
    - Changed `_collect_actor_unroll_all_heuristic_ids_native_rollout(...)` so native teacher labels come from `trajectory.actions[step_index][teacher_rows]` via `_teacher_labels_from_actions(...)`.
  - `python/weiss_rl/tests/test_runtime.py`
    - Added `test_public_teacher_rows_cover_mulligan_and_public_decision_kinds`.
- Tests:
  - `uv run pytest python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_mulligan_decision_kind_zero python/weiss_rl/tests/test_runtime.py::test_public_teacher_rows_cover_mulligan_and_public_decision_kinds -q`
  - Result: `3 passed`.
- Same-surface default baseline with checkpoint guard enabled:
  - Run: `runs/b1_loop_default_teacherlive_u80_s2_20260424`
  - Surface: benchmark no-league preset, 512 envs, u80, scalar dev-eval every 20 updates, 2 paired seeds.
  - Dev-eval:
    - u20 aggregate `0.9583`: B0 `1.0`, B1 `1.0`, B2 `1.0`, B3 `0.75`, B4 `1.0`, previous `1.0`.
    - u40 aggregate `0.75`: B1 `0.5`, B3 `0.5`, previous `0.5`.
  - Checkpoint behavior:
    - guard rolled back to u20 after u40 score drops.
    - best: u20, `metric_kind=dev_eval_mean`, `metric_value=0.9583333333`.
    - registry pinned `b1_noleague_baseline` to update 20.
  - Verdict:
    - current surface can produce a strong early B1 candidate, but continued training decays; guard protects the candidate but obscures the raw curve.
- Sharper public-heuristic target plus `mulligan_confirm`:
  - Run: `runs/b1_loop_sharppublic_t8_mullconfirm_u80_s2_20260424`
  - Overrides: `teacher_public_heuristic_temperature=8.0` for aux and warmstart, plus `mulligan_confirm` in public-heuristic teacher families.
  - Dev-eval:
    - u20 `0.7917`, u40 `0.7083`, u60 `0.7083`, u80 `0.7500`.
  - Metrics:
    - u80 entropy `0.3419`, exact action concentration `0.8857`, public top1 mass `0.6913`, `structured_main_move_mass=0.0030`.
  - Verdict:
    - Kill. More confidence did not improve B1/B2 quality and did not recover action-family balance.
- Older-style no-tactical-bias + teacher-public fade on the current native path:
  - Run: `runs/b1_loop_notactical_teacherfade_native_u80_s2_noguard_20260424`
  - Overrides: checkpoint guard off, no public heuristic bias families, bias fade like older runs, public teacher coefficient fades to 0 by update 10, public teacher family list empty.
  - Dev-eval:
    - u20 `0.3750`, u40 `0.4583`, u60 `0.4167`, u80 `0.6250`.
  - Metrics:
    - u20 `structured_main_move_mass=0.0182`, pass `0.2533`, main-play `0.2302`.
    - u80 `structured_main_move_mass=0.0037`, pass `0.2642`, main-play `0.1718`.
  - Verdict:
    - Kill for the current native fast path. The older recipe does not port directly; it neither restores old `main_move_mass ~0.23` nor improves dev-eval.
- Recorded native labels, default recipe, checkpoint guard disabled:
  - Run: `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424`
  - Surface: benchmark no-league preset, guard off, 512 envs, u80, scalar dev-eval every 20 updates, 2 paired seeds.
  - Dev-eval:
    - u20 aggregate `0.9583`: B0 `1.0`, B1 `1.0`, B2 `1.0`, B3 `0.75`, B4 `1.0`, previous `1.0`.
    - u40 aggregate `0.7500`.
    - u60 aggregate `0.7083`.
    - u80 aggregate `0.7083`.
  - Metrics:
    - u20 throughput `56,919 samples/sec`; loss `0.4095`; entropy `0.6972`; target abs mean `0.3057`.
    - u20 teacher valid fraction `0.5014`; teacher action accuracy `0.9496`; public top1 mass `0.4895`.
    - u20 action-family masses: pass `0.2543`, main-play-character `0.2188`, main-move `0.0297`.
    - u80 action-family masses: pass `0.2627`, main-play-character `0.1719`, main-move `0.0029`.
    - `collector_teacher_label_ms=0.0` on logged updates after the recorded-label patch.
  - Checkpoint tracker:
    - best u20, `metric_kind=dev_eval_mean`, `metric_value=0.9583333333`, `policy_version=1`, checkpoint `training/checkpoints/checkpoint_20.pt`.
  - Bounded final eval:
    - Command:
      - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424 --policy-id "B0 RandomLegal" --policy-id policy_000001 --policy-id "B2 HeuristicPublic" --paired-seed-limit 8 --parallel-workers 6 --skip-metagame --skip-figures --skip-readiness`
    - Artifact:
      - `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424/eval/final_eval/summary.json`
    - Result:
      - `policy_000001` vs B0: `16/16` wins, mean `1.0`.
      - `policy_000001` vs B2: `16/16` wins, mean `1.0`.
      - `policy_000001` self: `8/16` wins, mean `0.5`.
      - Truncations: `0` in all matrix entries.
  - B2 disagreement audit:
    - Artifact:
      - `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424/eval/b2_disagreement_audit/update_20/audit/summary.json`
    - Summary:
      - compared steps `495`;
      - mean total variation `0.304602`;
      - max total variation `0.998545`;
      - top family pairs: main-play vs main-play `47`, pass vs main-play `28`, attack vs attack `5`;
      - bundle-level family top-action match rates about `0.94-0.95`, action top match about `0.918-0.923`.
  - Verdict:
    - Keep. This is the current best B1 candidate on the local diagnostic surface.
    - Important caveat: it is an early checkpoint, not a monotonically improving run. Training past u20 currently decays locally.
- Throughput smoke after recorded-label patch:
  - Run: `runs/b1_loop_recordedlabels_throughput_smoke_u60_20260424`
  - Surface: benchmark no-league preset with periodic dev-eval off, 512 envs, u60.
  - Throughput:
    - u20 `57,286 samples/sec`;
    - u40 `87,378 samples/sec`;
    - u60 `105,845 samples/sec`;
    - tail u50-u60 average `101,511 samples/sec`;
    - max `105,845 samples/sec`.
  - Same-update comparison:
    - old `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke` at u60: `101,947 samples/sec`;
    - old run reached `140k+` only after warming to about u160.
  - Verdict:
    - No same-update throughput regression; recorded-label native path is at least neutral locally and should scale cleanly because it removes Python heuristic recomputation from native teacher labeling.
- Current best B1 stance:
  - Candidate: `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424`, `policy_000001`, update `20`, checkpoint `training/checkpoints/checkpoint_20.pt`.
  - Evidence: 2-paired-seed dev aggregate `0.9583`; 8-paired-seed final eval `16/16` vs B0 and `16/16` vs B2; zero truncations; no throughput regression.
  - Risks:
    - Local eval is still small and noisy; do not call this thesis-grade.
    - Action-family mass still does not match old structured-v2-split healthy profile; `main_move_mass` remains far below old `~0.23`.
    - Performance decays after u20, so the next research question is stability/overtraining, not initial capability.
  - Next hypotheses:
    1. Confirm `policy_000001` with 16 or 32 paired scalar seeds before promoting as the B1 anchor.
    2. Run a short server/topology smoke with the recorded-label patch and verify native rollout, DDP/rank cleanliness, and no teacher-label overhead.
    3. Investigate why post-u20 training decays: entropy collapse, over-imitation of fixed heuristic rows, checkpoint-guard rollback dynamics, and whether an explicit early-stop/frozen-best export should be the B1 anchor path.
    4. Separately audit why old non-native/structured-v2-split runs had high `main_move_mass`; current evidence says simply turning off tactical bias and fading public teacher is not sufficient on the current native fast path.

## 2026-04-24 B1 Loop Continuation: Current Best Alias and Aggro Stress Test

- User question addressed:
  - The earlier local `100k+ samples/sec` observation was real for warmed throughput-only B1 runs.
  - It was not contradicted by learning runs with periodic scalar eval and checkpoint rollback, whose per-update throughput samples include extra blocking work.
  - The dedicated recorded-label throughput smoke remains the correct same-surface local throughput check:
    - `runs/b1_loop_recordedlabels_throughput_smoke_u60_20260424`
    - u20 `57,286 samples/sec`, u40 `87,378`, u60 `105,845`;
    - tail u50-u60 average `101,511`;
    - old same-surface u60 in `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke` was `101,947`;
    - no same-update throughput regression from recorded native teacher labels.
- Wider scalar eval of the current best no-guard checkpoint:
  - Run:
    - `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424`
  - Policy:
    - `policy_000001`, update `20`, checkpoint `training/checkpoints/checkpoint_20.pt`.
  - Command:
    - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424 --policy-id "B0 RandomLegal" --policy-id policy_000001 --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 16 --parallel-workers 6 --skip-metagame --skip-figures --skip-readiness`
  - Artifact:
    - `runs/b1_loop_recordedlabels_default_u80_s2_noguard_20260424/eval/final_eval/summary.json`
  - Matrix result over 16 paired seeds / 32 games per matchup:
    - vs B0: `32/32`, mean `1.0`;
    - vs B2: `32/32`, mean `1.0`;
    - vs B3 aggro: `18/32`, mean `0.5625`;
    - vs B4 control: `29/32`, mean `0.90625`;
    - self: `16/32`, mean `0.5`;
    - truncations: `0` everywhere.
  - Read:
    - This is much better than old weak B1 behavior and beats B2 decisively on this local scalar surface.
    - B3 aggro is the remaining weak local anchor; treat it as the next stressor, not as solved.
- Structural experiment: aggressive-skewed public teacher profiles without changing the native collector path.
  - Rationale:
    - Direct B3/B4 opponent sampling would likely disable the current all-B2 simulator-native fast rollout path.
    - A learner-side public target profile skew is server-scalable and should preserve collector throughput because actor behavior remains native heuristic rollout.
  - Run:
    - `runs/b1_loop_aggressiveprofile_recordedlabels_u80_s2_noguard6_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_loop_aggressiveprofile_recordedlabels_u80_s2_noguard6_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 80 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.structured_aux.teacher_public_heuristic_profiles='["aggressive","aggressive","base","control"]' --override training.structured_warmstart.teacher_public_heuristic_profiles='["aggressive","aggressive","base","control"]'`
  - Override verification:
    - `config_canonical.json` shows the aux and warmstart profile lists as `["aggressive","aggressive","base","control"]`.
  - Dev-eval:
    - u20 aggregate `0.9583`;
    - u40 `0.7500`;
    - u60 `0.7083`;
    - u80 `0.7083`.
  - Wider final eval:
    - vs B0: `32/32`, mean `1.0`;
    - vs B2: `32/32`, mean `1.0`;
    - vs B3 aggro: `16/32`, mean `0.5`;
    - vs B4 control: `30/32`, mean `0.9375`;
    - truncations: `0` everywhere.
  - Verdict:
    - Kill. It slightly improves B4 but worsens the real weak point, B3.
    - Do not promote aggressive-profile skew as the B1 anchor recipe.
- Clean current-best guarded artifact:
  - Run:
    - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_loop_recordedlabels_guard_u80_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 80 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6`
  - Dev-eval / guard behavior:
    - u20 aggregate `0.9583`, canonical B1 alias persisted at update `20`;
    - later u40 evaluations dropped to `0.7500`/`0.7917`;
    - checkpoint guard rolled back to u20 with reason `score_drop`.
  - Registry:
    - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424/training/snapshots/registry.json`
    - pinned `b1_noleague_baseline`, update `20`;
    - alias path `training/snapshots/b1_noleague_baseline/weights.pt`.
  - Alias final eval command:
    - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_loop_recordedlabels_guard_u80_s2_20260424 --b1-baseline-run-dir runs/b1_loop_recordedlabels_guard_u80_s2_20260424 --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 16 --parallel-workers 6 --skip-metagame --skip-figures --skip-readiness`
  - Alias final eval artifact:
    - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424/eval/final_eval/summary.json`
  - Alias final eval result:
    - vs B0: `32/32`, mean `1.0`;
    - vs B2: `32/32`, mean `1.0`;
    - vs B3 aggro: `18/32`, mean `0.5625`;
    - vs B4 control: `31/32`, mean `0.96875`;
    - self: `17/32`, mean `0.53125`;
    - truncations: `0` everywhere.
  - Current best practical B1 artifact:
    - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424`
    - use `b1_noleague_baseline` / update `20` from this run for downstream local comparisons.
- Current diagnosis:
  - The main solved issue was a teacher-label contract mismatch on the native path:
    - decision kind `0` was previously excluded from public teacher labeling;
    - native rollout teacher labels were recomputed from Python heuristic instead of the actual recorded native actions.
  - Fixing both gives a strong early B1 anchor without losing the native rollout throughput posture.
  - Remaining weakness is not generic B2 imitation; it is B3 aggro robustness and post-u20 decay.
- Next hypotheses:
  1. Do not spend more local time on throughput unless profiling shows a new bottleneck; the recorded-label patch is throughput-neutral at the relevant same-update surface.
  2. Investigate B3 aggro specifically with replay/state disagreement rather than more public-profile nudges; the first profile-skew test failed.
  3. Test stability mechanisms that preserve the u20 behavior longer without changing collector data:
     - lower/decay public logit bias after warmstart;
     - reduce teacher/public CE after the first strong checkpoint;
     - explicit early-stop/frozen-best export for B1;
     - inspect value/advantage drift after u20.
  4. Before server spend, run a tiny server smoke only:
     - dry-run topology;
     - 1-2 update train;
     - verify `distributed.world_size`, native rollout active, rank-0 artifact cleanliness, no NaNs, and `collector_teacher_label_ms=0`.

## 2026-04-24 B1 Loop Continuation: B3 Disagreement Causal Pass

- Motivation:
  - The best clean B1 alias is decisive vs B0/B2 and strong vs B4, but only `18/32` vs B3 aggro.
  - The previous aggressive-profile target skew did not improve B3.
  - Need a causal replay audit before another experiment.
- Code change:
  - `python/scripts/b2_disagreement_audit.py`
    - Generalized the audit from hard-coded `B2 HeuristicPublic` to `--opponent-policy-id`, defaulting to B2 for backward compatibility.
    - The script now validates that `episodes.jsonl` matches the requested opponent and resolves/reruns the selected heuristic opponent.
- B3 audit command:
  - `uv run python python/scripts/b2_disagreement_audit.py --stack-config runs/b1_loop_recordedlabels_guard_u80_s2_20260424/config_canonical.json --run-dir runs/b1_loop_recordedlabels_guard_u80_s2_20260424 --output-run-dir runs/b1_loop_recordedlabels_guard_u80_s2_20260424/eval/b3_disagreement_audit/update_20 --episodes-jsonl runs/b1_loop_recordedlabels_guard_u80_s2_20260424/eval/final_eval/matchups/01_b1_noleague_baseline__vs__03_b3_heuristicpublicaggro/episodes.jsonl --policy-id b1_noleague_baseline --opponent-policy-id "B3 HeuristicPublicAggro" --snapshot-registry-json runs/b1_loop_recordedlabels_guard_u80_s2_20260424/training/snapshots/registry.json --top-k 25 --top-actions 5`
- B3 audit artifact:
  - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424/eval/b3_disagreement_audit/update_20/audit/summary.json`
- B3 audit result:
  - status `ok`;
  - reran `32` games across `16` paired seeds;
  - captured `32` replay bundles;
  - compared steps `4756`;
  - inspected top-difference steps `800`;
  - mean total variation `0.342573`;
  - max total variation `0.999996`.
- Top family disagreements:
  - learner/B1 `main_play_character` vs B3 `main_play_character`: `385`;
  - learner/B1 `pass` vs B3 `main_move`: `281`;
  - learner/B1 `pass` vs B3 `main_play_character`: `131`;
  - `clock_from_hand` vs `clock_from_hand`: `3`.
- Read:
  - B3 gap is not mainly a generic mulligan or attack problem.
  - The sharpest gap is exactly the previously suspicious action family: B3 wants `main_move` while B1 strongly prefers `pass`.
  - This explains why learner-side profile skew was weak: current native collector data still has `collector_main_move_actions=0` and teacher `main_move` rows are not being produced by the base native heuristic rollout.
- Cross-repo simulator finding:
  - In `C:\Users\Bruger\Desktop\this one\weiss-schwarz-simulator\weiss_core\src\env\heuristic_public.rs`, native heuristic public has one baked-in base profile:
    - `MainMove` priority `120`;
    - `Pass` priority `160`.
  - In the RL Python heuristic profile, B3/aggressive uses:
    - `move_priority=210`;
    - `pass_priority=115`.
  - So the current native rollout path cannot generate the B3-style main-move behavior without either a simulator-native profile parameter or a fallback to Python heuristic rollout.
- Current best B1 remains:
  - `runs/b1_loop_recordedlabels_guard_u80_s2_20260424`
  - policy id / alias: `b1_noleague_baseline`
  - update: `20`
  - local scalar matrix: B0 `32/32`, B2 `32/32`, B3 `18/32`, B4 `31/32`, zero truncations.
- Next structural experiment:
  - Best next change is not another target-profile knob.
  - Add profile-aware simulator-native heuristic rollout/action selection so B1 can collect B3/aggressive-style main-move data without falling back to slow Python actor orchestration.
  - Then run a small B1 no-league variant with a limited native aggressive mix and compare:
    - B3 score;
    - B0/B2/B4 retention;
    - `collector_main_move_actions`;
    - `structured_main_move_mass`;
    - same-update throughput vs the native base smoke.

## 2026-04-24 B1 Loop Continuation: Exploration/Self-Play and Native Aggressive Rollout

- User hypothesis:
  - Passing through main phase is suspicious.
  - Maybe train beyond update 20 with more exploration and/or self-play so the B1 anchor can evolve instead of freezing at the first good checkpoint.
- Blunt entropy/fade test:
  - Run:
    - `runs/b1_entropy_teacherfade_fromscratch_u80_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_entropy_teacherfade_fromscratch_u80_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 80 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.exploration.entropy_coef=0.08 --override training.exploration.entropy_anneal_to=0.04 --override training.exploration.entropy_anneal_steps_updates=200000 --override training.structured_aux.teacher_public_heuristic_coef=0.03 --override training.structured_aux.teacher_public_heuristic_final_coef=0.0 --override training.structured_aux.teacher_public_heuristic_end_updates=40 --override model.public_heuristic_logit_bias_end_updates=60 --override model.public_heuristic_logit_bias_final_scale=0.5`
  - Result:
    - u20 dev aggregate `0.9583`;
    - u40 `0.6667`;
    - u60 `0.2917`;
    - u80 `0.3750`.
  - Key metrics:
    - `actor_heuristic_fraction_active=1.0`;
    - `collector_main_move_actions=0`;
    - `teacher_public_heuristic_coef_active=0.0` by u80;
    - `public_heuristic_logit_bias_scale_active=0.5`.
  - Verdict:
    - Kill.
    - More entropy/fading the teacher does not create the missing behavior if the native collector data still never contains main-move decisions.
- Blunt model self-play / model actor continuation:
  - Code change:
    - Added explicit `--resume-allow-config-mismatch` to `python/scripts/train.py`.
    - It allows research continuations from a checkpoint when the config hash changes, while still checking spec hash and algorithm.
  - Run:
    - `runs/b1_continue_u20_modelselfplay_entropy_u40_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_continue_u20_modelselfplay_entropy_u40_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --resume-from runs/b1_loop_recordedlabels_guard_u80_s2_20260424/training/checkpoints/checkpoint_20.pt --resume-allow-config-mismatch --max-updates 40 --seed 2 --checkpoint-interval-updates 10 --override evaluation.periodic_dev_eval_interval_updates=10 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.actor_policy_backend='"model"' --override training.actor_heuristic_fraction=0.0 --override training.teacher_aux.mode='"off"' --override training.exploration.entropy_coef=0.08 --override training.exploration.entropy_anneal_to=0.04 --override training.exploration.entropy_anneal_steps_updates=200000 --override model.public_heuristic_logit_bias_scale=0.0 --override model.public_heuristic_actor_logit_bias_scale=0.0 --override model.public_heuristic_logit_bias_final_scale=0.0 --override model.public_heuristic_logit_bias_families='[]'`
  - Result:
    - u30 dev aggregate `0.0833`;
    - u40 dev aggregate `0.2500`;
    - u40 anchor scores: B0 `0.5`, old B1 `0.25`, B2 `0.0`, B3 `0.0`, B4 `0.0`, previous snapshot `0.75`.
  - Key metrics:
    - `actor_heuristic_fraction_active=0.0`;
    - `collector_actor_policy_forward_ms=26981.0`;
    - `collector_main_move_actions=217`;
    - `collector_max_consecutive_main_moves=63`;
    - `structured_main_move_mass=0.20548`;
    - `structured_pass_mass=0.06429`;
    - `vtrace_rho_p95=1206.39`;
    - `vtrace_rho_p99=1518766.25`;
    - throughput roughly `6325 samples/sec` in the run.
  - Verdict:
    - Kill.
    - True self-play creates main-move data, but in the worst way: unstable model actor collection, pathological movement loops, huge off-policy ratios, terrible eval, and a large local throughput hit.
    - This supports the diagnosis that we need richer behavior data, not unconstrained self-play.
- Structural fix:
  - Implemented profile-aware simulator-native heuristic rollout across the sibling simulator and RL runtime.
  - Simulator changes:
    - `C:\Users\Bruger\Desktop\this one\weiss-schwarz-simulator\weiss_core\src\env\heuristic_public.rs`
      - Added native `base`, `aggressive`, and `control` profile scoring values aligned to the Python public heuristic profiles.
    - `...\weiss_core\src\pool\helpers\legal_sampling.rs`
      - Added `choose_heuristic_public_profile_actions_into(...)`.
    - `...\weiss_core\src\pool\step.rs`
      - Added `rollout_heuristic_public_profile_into_i16_legal_ids(...)`; existing base rollout remains compatible.
    - `...\weiss_py\src\lib_parts\env_pool\30_settings.rs`
      - Exposed profile action selection to Python.
    - `...\weiss_py\src\lib_parts\env_pool\40_reset_step.rs`
      - Extended `rollout_heuristic_public_into_i16_legal_ids(steps, out, profile_name="base")`.
  - RL changes:
    - Added `training.heuristic_native_rollout_profile`, default `base`, choices `base/aggressive/control`.
    - Runtime passes the configured profile into native rollout.
    - Training summaries/determinism reports record the profile.
    - Promoted `training.heuristic_native_rollout_profile: aggressive` into `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague.yaml`, so the B1 no-league benchmark inherits it.
  - Validation:
    - `cargo check -p weiss_py` in the simulator repo passed.
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_prefetch_and_native_rollout_overrides -q` passed.
    - `uv run pytest python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset python/weiss_rl/tests/test_runtime.py::test_teacher_labels_from_ids_cover_mulligan_decision_kind_zero python/weiss_rl/tests/test_runtime.py::test_public_teacher_rows_cover_mulligan_and_public_decision_kinds python/weiss_rl/tests/test_runtime.py::test_collect_all_heuristic_ids_native_rollout_requires_stateless_heuristic_actor -q` passed.
- Flag smoke:
  - Run:
    - `runs/b1_native_aggressive_profile_flag_smoke_u2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_native_aggressive_profile_flag_smoke_u2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 2 --seed 2 --checkpoint-interval-updates 2 --override evaluation.periodic_dev_eval_interval_updates=999 --override training.heuristic_native_rollout_profile='"aggressive"'`
  - Result:
    - `collector_actor_policy_forward_ms=0`;
    - `collector_main_move_actions=1024` at u1/u2;
    - `collector_max_consecutive_main_moves=16`;
    - `teacher_main_move_fraction` about `0.063`;
    - native rollout profile flag is live.
- Main aggressive-native learning diagnostic:
  - Run:
    - `runs/b1_native_aggressive_profile_u80_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_native_aggressive_profile_u80_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 80 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.heuristic_native_rollout_profile='"aggressive"'`
  - Dev eval:
    - u20 aggregate `1.0000`: B0 `1.0`, old B1 `1.0`, B2 `1.0`, B3 `1.0`, B4 `1.0`, previous snapshot `1.0`;
    - u40 aggregate `0.8333`: B3 `0.75`;
    - u60 aggregate `0.7500`: B3 `0.50`;
    - u80 aggregate `0.7500`: B3 `0.50`.
  - Wider scalar eval for u20 `policy_000001`:
    - eval artifact:
      - `runs/b1_native_aggressive_profile_u80_s2_20260424/eval/final_eval/summary.json`
    - vs B0: B0 focal lost `0/32`, so policy beat B0 `32/32`;
    - vs B2: `32/32`, mean `1.0`;
    - vs B3 aggro: `22/32`, mean `0.6875`;
    - vs B4 control: `30/32`, mean `0.9375`;
    - self: `17/32`, mean `0.53125`;
    - truncations: `0`.
  - Verdict:
    - This is the first structural change that improves the B3 gap on scalar eval.
    - More updates still hurt after the early checkpoint; the profile fixes data coverage, not late-training drift.
- Throughput controls:
  - Same-build release controls:
    - `runs/b1_native_base_profile_release_throughput_smoke_u60_20260424`
    - `runs/b1_native_aggressive_profile_release_throughput_smoke_u60_20260424`
  - Tail u50-u60 throughput:
    - base profile: about `31,790 samples/sec`, final `33,596`;
    - aggressive profile: about `31,438 samples/sec`, final `33,129`;
    - same-build delta is about `-1.1%`.
  - Important caveat:
    - These absolute numbers are not comparable to the older `runs/b1_loop_recordedlabels_throughput_smoke_u60_20260424` `101,511` tail because the simulator package was rebuilt/reinstalled from the sibling checkout for profile support during this loop.
    - Do not claim a 3x regression from the profile itself; the valid local comparison is same-build base vs same-build aggressive.
    - The architecture remains server-scalable: native rollout active and `collector_actor_policy_forward_ms=0`.
- Clean guarded current-best artifact:
  - Run:
    - `runs/b1_native_aggressive_profile_guard_u80_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_native_aggressive_profile_guard_u80_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 80 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override training.heuristic_native_rollout_profile='"aggressive"'`
  - Guard behavior:
    - u20 dev aggregate `1.0000`;
    - u40 dev aggregate `0.7917` after rollback/re-eval;
    - checkpoint guard pinned `b1_noleague_baseline` at update `20`.
  - Registry:
    - `runs/b1_native_aggressive_profile_guard_u80_s2_20260424/training/snapshots/registry.json`
    - `b1_noleague_baseline` update `20`.
  - Alias final eval command:
    - `uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_eval.yaml --run-dir runs/b1_native_aggressive_profile_guard_u80_s2_20260424 --b1-baseline-run-dir runs/b1_native_aggressive_profile_guard_u80_s2_20260424 --policy-id "B0 RandomLegal" --policy-id b1_noleague_baseline --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 16 --parallel-workers 6 --skip-metagame --skip-figures --skip-readiness`
  - Alias final eval artifact:
    - `runs/b1_native_aggressive_profile_guard_u80_s2_20260424/eval/final_eval/summary.json`
  - Alias final eval result:
    - vs B0: B0 focal lost `0/32`, so B1 beat B0 `32/32`;
    - vs B2: `32/32`, mean `1.0`;
    - vs B3 aggro: `20/32`, mean `0.625`;
    - vs B4 control: `32/32`, mean `1.0`;
    - self: `16/32`, mean `0.5`;
    - truncations: `0`.
- Current best B1 update:
  - Replace the previous practical B1 anchor with:
    - `runs/b1_native_aggressive_profile_guard_u80_s2_20260424`
    - policy id / alias: `b1_noleague_baseline`
    - update: `20`
  - Compared with previous clean B1 (`runs/b1_loop_recordedlabels_guard_u80_s2_20260424`, update `20`):
    - B0 retained `32/32`;
    - B2 retained `32/32`;
    - B3 improved from `18/32` to `20/32` on the clean alias eval, and `22/32` on the unguarded u20 snapshot;
    - B4 improved from `31/32` to `32/32` on the clean alias eval;
    - no truncations in either eval.
- Interpretation:
  - The user was right that pass-only behavior was bad.
  - The correct scalable form of added exploration is not unconstrained model self-play; it is simulator-native behavioral diversity that exposes the missing main-move family while preserving native rollout.
  - The remaining post-u20 stall/decay is probably a learner/objective drift problem:
    - teacher and policy become very confident;
    - entropy falls to around `0.5`;
    - later checkpoints overfit the native aggressive/base target mix and lose robustness to the dev anchor set.
  - Do not train B1 longer after u20 without an explicit stabilizer or curriculum change; the evidence says more updates make it worse.
- Next hypotheses:
  1. Keep aggressive-native B1 as the current anchor and use checkpoint guard/early selection for downstream comparisons.
  2. Next structural experiment should target late drift, not more blind exploration:
     - stronger best-checkpoint anchoring;
     - scheduled profile mix/cycle in native rollout if implemented;
     - post-u20 lower LR or freeze auxiliary heads;
     - KL-to-best/B1 behavior regularization after first strong checkpoint;
     - entropy floor without switching to model actor collection.
  3. A server smoke is now worthwhile before any long server spend:
     - dry-run topology;
     - 1-2 update aggressive-native no-league smoke;
     - verify `training.heuristic_native_rollout_profile=aggressive`;
     - verify `distributed.world_size`;
     - verify `collector_actor_policy_forward_ms=0`;
     - verify `collector_main_move_actions>0`;
     - verify rank-0 artifacts and no NaNs.

## 2026-04-24 B1 Loop Continuation: Why It Still Does Not Improve After u20

- User question:
  - Why does the B1 anchor stop improving after update `20`?
  - Why not train to `u200/u800/u1200` with teacher fading into self-play?
- Immediate config finding:
  - The current B1 no-league surface is not actually doing teacher-fade into self-play.
  - In `runs/b1_native_aggressive_profile_curve_u200_s2_20260424/config_canonical.json`:
    - `training.actor_policy_backend=heuristic_public`;
    - `training.actor_heuristic_fraction=1.0`;
    - `training.actor_heuristic_end_updates=-1`;
    - `training.actor_heuristic_final_fraction=1.0`;
    - `training.structured_aux.teacher_public_heuristic_coef=0.1`;
    - `training.structured_aux.teacher_public_heuristic_end_updates=-1`;
    - `training.structured_aux.teacher_public_heuristic_final_coef=0.1`;
    - `training.heuristic_native_rollout_profile=aggressive`.
  - So the curve is fixed aggressive-native heuristic rollout plus constant teacher/public auxiliary, not a scheduled handoff to policy self-play.
- Longer fixed-aggressive-native curve:
  - Run:
    - `runs/b1_native_aggressive_profile_curve_u200_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_native_aggressive_profile_curve_u200_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --max-updates 200 --seed 2 --checkpoint-interval-updates 20 --override evaluation.periodic_dev_eval_interval_updates=20 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false`
  - Dev eval curve:
    - u20 aggregate `1.0000`: B0 `1.0`, old B1 `1.0`, B2 `1.0`, B3 `1.0`, B4 `1.0`, previous `1.0`;
    - u40 `0.8333`: B3 `0.75`;
    - u60 `0.7500`: B3 `0.50`;
    - u80 `0.7500`: B3 `0.50`;
    - u100 `0.8333`: B3 `1.0`;
    - u120 `0.7500`: B3 `0.50`;
    - u140 `0.8333`: B3 `1.0`;
    - u160 `0.7917`: B3 `0.75`;
    - u180 `0.7500`: B3 `0.50`;
    - u200 `0.7917`: B3 `0.75`.
  - Interpretation:
    - On tiny 2-seed dev eval, later checkpoints sometimes look okay on B3, but never recover the all-anchor u20 result.
    - The wider 16-pair scalar eval from the guarded artifact still says the reliable best is the u20 alias.
    - Extending the same training objective to u200 does not reveal a hidden upward trend.
  - Representative scalar metrics:
    - u20: entropy `0.6423`, teacher action accuracy `1.0`, structured main-move mass `0.0937`, pass mass `0.2202`;
    - u40: entropy `0.9634`, teacher action accuracy `0.8125`, main-move mass `0.0`, pass mass `0.3751`;
    - u100: entropy `0.5167`, teacher action accuracy `1.0`, main-move mass `0.0930`, pass mass `0.1883`;
    - u200: entropy `0.4686`, teacher action accuracy `1.0`, main-move mass `0.0586`, pass mass `0.2541`.
  - Read:
    - The model mostly learns the fixed native behavior distribution; loss/teacher accuracy can look good while dev robustness does not improve.
    - More updates reduce entropy and confidence-sharpen the same behavior, which is not the same as policy improvement.
- Gentle model-actor / teacher-fade continuation test:
  - Motivation:
    - Full model self-play previously failed, but maybe a small model-actor slice after u20 could create exploration without destroying the anchor.
  - Run:
    - `runs/b1_continue_u20_soft_model10_teacherfade_u60_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_continue_u20_soft_model10_teacherfade_u60_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --resume-from runs/b1_native_aggressive_profile_guard_u80_s2_20260424/training/checkpoints/checkpoint_20.pt --resume-allow-config-mismatch --max-updates 60 --seed 2 --checkpoint-interval-updates 10 --override evaluation.periodic_dev_eval_interval_updates=10 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.actor_policy_backend='"heuristic_public"' --override training.actor_heuristic_fraction=0.9 --override training.actor_heuristic_end_updates=-1 --override training.actor_heuristic_final_fraction=0.9 --override training.structured_aux.teacher_public_heuristic_coef=0.05 --override training.structured_aux.teacher_public_heuristic_final_coef=0.0 --override training.structured_aux.teacher_public_heuristic_end_updates=40 --override training.teacher_aux.mode='"always"' --override model.public_heuristic_actor_logit_bias_scale=0.5 --override model.public_heuristic_logit_bias_scale=1.0 --override model.public_heuristic_logit_bias_final_scale=0.5 --override model.public_heuristic_logit_bias_end_updates=60`
  - Dev eval:
    - u30 aggregate `0.7917`: B3 `0.75`;
    - u40 `0.4583`: B2 `0.75`, B3 `0.25`, B4 `0.25`;
    - u50 `0.2083`: B2 `0.25`, B3 `0.0`, B4 `0.0`;
    - u60 `0.0833`: B0 `0.5`, B2/B3/B4 `0.0`.
  - Metrics:
    - actor heuristic fraction active `0.9`;
    - `collector_actor_policy_forward_ms` around `19-20s` per update;
    - throughput fell from native scale to roughly `12k -> 4.7k samples/sec` over the run;
    - `teacher_public_heuristic_coef_active` reached `0.0` by u40;
    - `vtrace_rho_p99` reached `19.2` at u40 and `41.9` at u50.
  - Verdict:
    - Kill.
    - Even a 10% model-actor slice is currently destabilizing and expensive on the local path.
    - The naive teacher-fade/model-explore handoff needs more structure before it is safe to use.
- Current diagnosis:
  - There are two separate mechanisms:
    1. Fixed native heuristic training is throughput-friendly and gives a good early anchor, but it is not a self-improving policy-iteration loop.
    2. Unconstrained or lightly constrained model actor collection introduces policy-distribution shift faster than the learner can exploit it, hurting both eval and throughput.
  - So “train longer” is available and now tested to u200, but it does not solve the learning problem under the current objective/actor contract.
- Next structural direction:
  - Preserve the current u20 aggressive-native guarded B1 as the best anchor.
  - Do not run u800/u1200 of the same fixed-heuristic recipe; it is compute waste.
  - A plausible next experiment needs one of:
    - native profile schedule/mix without leaving native rollout, e.g. cycle/mix `base/aggressive/control` in the simulator rollout;
    - KL-to-best or behavior regularization when model actor lanes are introduced;
    - a much smaller and gated model-actor fraction with hard safety counters for main-move loops and pass-with-nonpass pathologies;
    - value/policy objective changes that make model-actor data stable before actor-fraction decay.

## 2026-04-24 B1 Loop Continuation: Clean Model Lanes and B3 Pressure

- Code changes:
  - Added source-aware policy train masking for mixed heuristic/model actor rows in `python/weiss_rl/runtime.py`.
    - `training.train_on_heuristic_actor_rows=false` now masks the actual heuristic-sampled focal rows instead of only masking the pure-heuristic case.
    - Added counters: `actor_model_rows`, `actor_heuristic_rows`, `policy_train_model_rows`, `policy_train_heuristic_rows`, `policy_excluded_heuristic_rows`.
    - Mixed row-level heuristic/model IMPALA collection now raises unless `training.heuristic_actor_hidden_state_tracking=true`, because otherwise recurrent state can silently go stale when a row alternates between heuristic and model behavior.
  - Added native rollout profile scheduling:
    - `training.heuristic_native_rollout_profiles`;
    - `training.heuristic_native_rollout_profile_mode` in `{fixed, cycle, random}`.
    - A cycle smoke confirmed `base/aggressive/control` profile counters while preserving `collector_actor_policy_forward_ms=0`.
  - Added `training.diverse_opponent_policy_id` so a no-league diagnostic actor lane can face a fixed opponent such as `B3 HeuristicPublicAggro` without enabling the full league.
  - Relaxed the native rollout gate so actors whose actual opponents are all B2 heuristic can still use native rollout even when other lanes use non-B2 pressure.
  - Tests run:
    - `uv run python -m py_compile python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_config_loader.py`
    - `uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_diverse_opponent_policy_id python/weiss_rl/tests/test_runtime.py::test_split_focal_actor_rows_rejects_mixed_impala_without_hidden_tracking python/weiss_rl/tests/test_runtime.py::test_policy_train_mask_for_actor_excludes_only_known_heuristic_rows python/weiss_rl/tests/test_runtime.py::test_collect_all_heuristic_ids_native_rollout_requires_stateless_heuristic_actor -q`
- Negative/diagnostic runs:
  - `runs/b1_continue_u20_model1pct_masked_trackhidden_u60_s2_20260424`
    - Clean 1% row-level model continuation from the strong u20 anchor.
    - Dev eval: u30 `0.7500`, u40 `0.7917`, u50 `0.8333`, u60 `0.6667`; B3: `0.50`, `0.50`, `0.75`, `0.25`.
    - Counters proved the mask was live: about `78-96` model rows/update and about `8.0k` heuristic rows excluded.
    - Throughput was bad locally: cumulative training throughput fell to about `4.9k samples/sec`; `collector_actor_policy_forward_ms` about `17-18s/update` because hidden-state tracking forces model advances for heuristic rows.
    - Verdict: useful bug/contract diagnostic, not a production path.
  - `runs/b1_continue_u20_actorlane_model1_u100_s2_20260424`
    - One whole model actor lane vs mirror/self-play, seven native heuristic lanes, policy loss only on model rows.
    - Dev eval: u30 `0.7500`, u40 `0.8333`, u50 `0.7500`, u60 `0.7500`, u70 `0.7500`, u80 `0.7083`, u90 `0.7917`, u100 `0.7500`.
    - B3 never trended: `0.50`, `1.00`, `0.50`, `0.75`, `0.50`, `0.75`, `0.50`, `0.50`.
    - Native lanes stayed native: tail `collector_native_rollout_profile_aggressive_unrolls=56`, about `1.0k` model train rows/update.
    - Verdict: self-play pressure alone is not the missing ingredient.
  - `runs/b1_continue_u100_actorlane_b3pressure_u200_s2_20260424`
    - Continued the promising B3-pressure u100 checkpoint to u200.
    - Dev eval: u120 `0.7500`, u140 `0.7083`, u160 `0.7917`, u180 `0.6250`, u200 `0.7083`.
    - B3: `0.50`, `0.50`, `0.75`, `0.25`, `0.50`.
    - Verdict: the u100 gain does not automatically keep improving to u200.
  - `runs/b1_continue_u100_actorlane_b3pressure_lr5e5_u200_s2_20260424`
    - Same continuation with learner LR `5e-5`.
    - Dev eval decayed similarly or worse: u120 `0.7500`, u140 `0.7083`, u160 `0.7917`, u180 `0.6250`, u200 `0.6250`.
    - Verdict: lower LR alone is not the stabilizer.
  - `runs/b1_continue_u20_actorlane_b3pressure_model2_u100_s2_20260424`
    - Two configured B3-pressure model lanes.
    - Tiny dev curve matched the one-lane run: u100 `0.8333`, B3 `1.0`.
    - Tail selected-batch counters still looked effectively like one pressure slice plus native lanes, so do not treat as a new best without deeper scheduler inspection.
- Current positive candidate:
  - Run:
    - `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark.yaml --run-label b1_continue_u20_actorlane_b3pressure_u100_s2_20260424 --autoscale --hardware-profile local --runtime-mode train_async_fast --resume-from runs/b1_native_aggressive_profile_guard_u80_s2_20260424/training/checkpoints/checkpoint_20.pt --resume-allow-config-mismatch --max-updates 100 --seed 2 --checkpoint-interval-updates 10 --override evaluation.periodic_dev_eval_interval_updates=10 --override evaluation.periodic_dev_eval_paired_seeds=2 --override evaluation.periodic_dev_eval_parallel_workers=6 --override curriculum.checkpoint_guard.enabled=false --override training.actor_policy_backend='"heuristic_public"' --override training.actor_heuristic_fraction=1.0 --override training.actor_heuristic_end_updates=-1 --override training.train_on_heuristic_actor_rows=false --override training.heuristic_actor_hidden_state_tracking=false --override training.diverse_opponent_actor_count=1 --override training.diverse_model_actor_count=1 --override training.diverse_opponent_policy_id='"B3 HeuristicPublicAggro"' --override training.diverse_opponent_batch_fraction=0.125 --override training.structured_aux.teacher_public_heuristic_coef=0.1 --override training.structured_aux.teacher_public_heuristic_final_coef=0.1 --override training.teacher_aux.mode='"always"' --override model.public_heuristic_actor_logit_bias_scale=1.0 --override model.public_heuristic_logit_bias_scale=2.0 --override model.public_heuristic_logit_bias_final_scale=2.0`
  - Tiny dev eval curve:
    - u30 `0.7500`: B3 `0.50`;
    - u40 `0.8333`: B3 `1.00`;
    - u50 `0.7500`: B3 `0.50`;
    - u60 `0.7917`: B3 `0.75`;
    - u70 `0.7917`: B3 `0.75`;
    - u80 `0.7500`: B3 `0.50`;
    - u90 `0.7500`: B3 `0.75`;
    - u100 `0.8333`: B3 `1.00`.
  - 16-pair scalar eval of u100 (`policy_000009`) on canonical scalar eval:
    - B0 RandomLegal: `32/32` for B1 (`1.0000`);
    - self: `17/32` (`0.53125`);
    - B2 HeuristicPublic: `32/32` (`1.0000`);
    - B3 HeuristicPublicAggro: `22/32` (`0.6875`);
    - B4 HeuristicPublicControl: `32/32` (`1.0000`);
    - truncations: `0`.
  - Same-surface comparison:
    - Current clean guarded u20 alias from `runs/b1_native_aggressive_profile_guard_u80_s2_20260424` was B3 `20/32` (`0.625`) on the same 16-pair scalar surface.
    - So B3-pressure u100 is a real local candidate improvement on the weak matchup, but not yet a monotone long-run solution.
  - Throughput:
    - Local cumulative training throughput around `5.3k samples/sec` by u100 is not a useful server proxy because one local GPU waits on the model actor lane.
    - Runtime counters are the scalable part: tail windows kept `collector_native_rollout_profile_aggressive_unrolls=56` and about `845-902` model policy rows/update; actor env step windows were commonly `13k-20k env steps/sec` locally.
    - Expected server translation: this actor-lane shape should map better to multi-GPU L40 than row-level mixing because native lanes stay native and model lanes can be placed on actor/learner GPU resources.
- Current diagnosis:
  - The first credible post-u20 improvement came from targeted B3 opponent pressure, not from generic longer training, profile cycling, self-play, or lower LR.
  - The stall is likely an opponent/objective mismatch: fixed aggressive-native imitation learns a useful B1 anchor, but continued RL needs relevant hard-opponent pressure; unstructured self-play is too noisy.
  - Remaining problem: B3 pressure improves the u100 checkpoint but overtraining from u100 to u200 still degrades. The next structural target should be stability/selection, e.g. checkpoint-gated pressure schedules, KL-to-u100/best anchor, pressure-lane quota verification, or promotion-style accept/reject for B1 continuation checkpoints.

## 2026-04-25 B1 Loop Continuation: Audit, Pressure-Lane Bug Fix, and Negative Stabilizers

- Current best remains:
  - `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`, checkpoint `checkpoint_100.pt`, `policy_000009`.
  - 16-pair scalar eval: B0 `32/32`, self `17/32`, B2 `32/32`, B3 `22/32`, B4 `32/32`, truncations `0`.
  - Same-surface old clean u20 anchor was B3 `20/32`, so the improvement is real but still small and not a long upward curve.
- Code changes:
  - Added `training.diverse_opponent_policy_ids` as a list form of fixed pressure opponents.
    - This allows cycling fixed pressure opponents such as `B3 HeuristicPublicAggro` and `B4 HeuristicPublicControl` without enabling the full league.
    - Kept `training.diverse_opponent_policy_id` compatible.
  - Added replay-audit escape hatches for intentional config-schema drift:
    - `python/scripts/b2_disagreement_audit.py --allow-config-hash-mismatch`.
    - `python/weiss_rl/replay/inspector.py` can allow old snapshot config hashes when the caller opts in.
  - Fixed a process-collector pressure-lane bug in `python/weiss_rl/runtime.py`.
    - Child collector runtimes use `actor_count=1` but retain the global `actor_id`.
    - The old `min(config.actor_count, requested_diverse_model_actor_count)` clamp meant only global actor `0` could become a model-pressure lane inside process collectors.
    - After the fix, global actor `1` also becomes a model lane when `training.diverse_model_actor_count=2`.
  - Tests run:
    - `uv run python -m py_compile python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/b2_disagreement_audit.py python/weiss_rl/replay/inspector.py`
    - `uv run pytest python/weiss_rl/tests/test_runtime.py::test_actor_id_force_model_policy_lane_uses_global_actor_id python/weiss_rl/tests/test_runtime.py::test_split_focal_actor_rows_forces_model_policy_on_diverse_model_lane python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_cycles_fixed_diverse_opponent_policy_ids python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_diverse_opponent_policy_ids -q`
- Audit:
  - Source eval regenerated for `policy_000009` vs `B3 HeuristicPublicAggro` with 8 paired seeds.
  - Audit run: `runs/b1_u100_vs_b3_disagreement_audit_20260425_retry2`.
  - Result: status `ok`, `16` replay bundles, `2409` compared steps, mean total variation `0.4315`, max total variation `0.9742`.
  - Important diagnosis:
    - In high-disagreement states, B1 and B3 almost always choose the same top action family.
    - Top family pairs: `main_play_character/main_play_character` `178`, `main_move/main_move` `137`, `clock_from_hand/clock_from_hand` `5`.
    - The mismatch is mostly within-family detail:
      - B1 often prefers `main_play_character(hand_index=0, stage_slot=1)` while B3 prefers `stage_slot=0` (`47` top-diff cases).
      - B1 also differs on move destinations, e.g. `main_move(from_slot=0,to_slot=1)` vs B3 `to_slot=2`.
    - Pass is not the main B3 failure mode here:
      - 8-pair B3 eval sample had `10W/6L`, average `31.75` pass actions/game, average `3.69` pass-with-nonpass/game, no truncations.
- Negative experiments:
  - `runs/b1_continue_u100_b3pressure_nativecycle_consolidate_u200_s2_20260425`
    - Fast native consolidation after the u100 pressure checkpoint.
    - Tiny dev recovered at u160 (`0.8333`, B3 `0.75`) but did not keep rising; u200 `0.7500`, B3 `0.50`.
    - 16-pair scalar eval of u160 (`policy_000012`): B3 `21/32`, worse than current best `22/32`.
  - `runs/b1_continue_u100_actorlane_b3pressure_entropy08_u200_s2_20260425`
    - High entropy after u100.
    - Tiny dev: u120 `0.7500`, u140 `0.7083`, u160 `0.8333`, u180 `0.6250`, u200 `0.7500`.
    - 16-pair scalar eval of u160 (`policy_000012`): B3 `19/32`, B4 `29/32`, worse than current best.
  - `runs/b1_continue_u20_actorlane_b3b4pressure_u120_s2_20260425`
    - Fixed B3+B4 pressure list.
    - Tiny dev peaked u60 `0.8333`, B3 `1.00`, but 16-pair scalar eval of u60 (`policy_000003`) gave B3 `18/32`, worse than current best.
  - `runs/b1_continue_u20_b3pressure_slotaux_u120_s2_20260425`
    - Audit-informed slot and same-family-action aux: `teacher_slot_coef=0.08`, `teacher_same_family_action_coef=0.03`.
    - Tiny dev never beat `0.7500`; B3 stayed `0.50`; u120 aggregate fell to `0.6667`.
    - Verdict: within-family aux as configured conflicts or over-regularizes; do not keep this idea alive without redesign.
  - `runs/b1_continue_u100_b3pressure_unbiased_entropy05_u200_s2_20260425`
    - Removed public heuristic actor/train bias and teacher target after u100, with entropy `0.05`.
    - Collapsed immediately: u120 aggregate `0.1250`; B2/B3/B4 all `0.00`; only partial recovery by u200 (`0.4167`) and B3 still `0.00`.
    - Verdict: public heuristic scaffold is still required for competence on this reduced model.
  - `runs/b1_continue_u20_b3pressure_entropy08_u120_s2_20260425`
    - High entropy from the learning phase rather than after u100.
    - Tiny dev: u40 `0.6667`, u60 `0.7500`, u80 `0.7500`, u100 `0.7500`, u120 `0.6250`; B3 fell to `0.25` at u120.
    - Verdict: more entropy is not the missing ingredient.
- Pressure-lane bug proof and rerun:
  - Pre-fix symptom:
    - `runs/b1_continue_u20_actorlane_b3pressure_model2_u100_s2_20260424` and `runs/b1_continue_u20_actorlane_b3b4pressure_u120_s2_20260425` still showed only about `850-900` model train rows/update, effectively one model-pressure actor.
  - Fix smoke:
    - `runs/b1_2lane_pressure_globalactor_fix_smoke_u22_s2_20260425`.
    - Model train rows increased to `1755` at u21 and `1976` at u22.
    - This proves the global actor-id fix is live.
  - Fair rerun after fix:
    - `runs/b1_continue_u20_b3pressure_2lane_fixed_u100_s2_20260425`.
    - Tail model rows about `1.68k-1.85k/update`, roughly 2x the one-lane pressure signal.
    - Local cumulative throughput at u100 about `7.8k samples/sec`; local-only and expected to be less punitive on the L40 server if actor GPUs/CPUs are available.
    - Tiny dev: u40 `0.6667`, u60 `0.7500`, u80 `0.7500`, u100 `0.7917`; B3 at u100 `0.75`.
    - Verdict: the bug fix is important, but simply doubling B3 pressure still does not beat the one-lane u100 best.
- Current diagnosis:
  - The main B3 weakness is not pass spam and not wrong high-level family; it is detailed tactical selection within `main_play_character` and `main_move`.
  - Public heuristic bias is required, but naive slot/same-family auxiliary losses and higher entropy hurt.
  - More pressure data is now mechanically fixed but did not solve quality by itself.
  - Best next hypotheses:
    - investigate the structured/public heuristic target itself for slot/card quality and whether B3 target labels differ from the base teacher used for aux;
    - try a target that teaches B3-style within-family choices without the failed generic slot aux, e.g. B3-profile public target on pressure lanes only;
    - add best-checkpoint retention/regularization for post-u100 continuation so longer runs do not overwrite the useful u100 behavior;
    - only then extend to u200+ with checkpoint-gated continuation instead of raw continued SGD.
- Follow-up target-profile experiment:
  - `runs/b1_continue_u20_b3pressure_aggtarget_u120_s2_20260425` failed at startup because `teacher_public_heuristic_profile_mode="fixed"` is invalid; supported modes are `cycle` and `mixture`.
  - `runs/b1_continue_u20_b3pressure_aggtarget2_u120_s2_20260425` reran with a single aggressive profile and `profile_mode=cycle`.
  - Tiny dev: u40 `0.6667`, u60 `0.7500`, u80 `0.7500`, u100 `0.7917`, u120 `0.7917`; B3 `0.50`, `0.50`, `0.50`, `0.75`, `0.75`.
  - Tail counters still one model lane: about `860-895` model rows/update; local cumulative throughput about `8.16k samples/sec` at u120.
  - Verdict: aggressive-only public teacher target did not beat the current one-lane B3-pressure best and should not be promoted.
## 2026-04-25 B1 learning loop: profile bias, shaping, and model-data fraction tests

### Hypothesis

The previous best continuation (`runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`) improved over the clean u20 anchor but did not produce a long upward curve. The B3 disagreement audit pointed at within-family detail errors, especially `main_play_character` slot choice and `main_move` source/target choices. This loop tested whether the stall was caused by:

1. profile mismatch between aggressive teacher targets and base-profile public logit bias;
2. too little model-policy training data after u20;
3. pass/stall behavior that existing simulator no-progress shaping did not penalize.

### Code changes

- Added `model.public_heuristic_logit_bias_profile` with supported values `base`, `aggressive`, `control`.
  - The structured public logit-bias raw scorer now reads the selected `HeuristicPublicScoringProfile` for play, move, attack, and encore priorities/bonuses.
  - This makes the learner/actor tactical bias alignable with the public teacher target profile instead of always using base-like constants.
- Added optional RL-side `rewards.shaping.pass_with_nonpass_penalty`.
  - This is deliberately not passed to the simulator reward payload.
  - It is applied only inside learner/ppo batch construction when the sampled action is `pass`, another legal action exists, and `policy_train_mask` is true.
  - This makes it a targeted learner-side shaping experiment for pass-with-options rows, not a default simulator contract change.
- Focused tests passed:
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py -k public_heuristic --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_heuristic_public.py -k public_bias --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_build_learner_batch_can_penalize_pass_with_nonpass_available python/weiss_rl/tests/test_runtime.py::test_build_learner_batch_does_not_double_apply_truncation_reward python/weiss_rl/tests/test_runtime.py::test_build_ppo_batch_does_not_double_apply_truncation_reward --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_applies_antistall_v2_overrides python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_public_heuristic_bias_override --tb=short`

### Runs and verdicts

- `runs/b1_aggbias_profile_smoke_u22_s2_20260425`
  - Purpose: runtime smoke for the new aggressive public logit-bias profile.
  - Result: config and active scales were live:
    - `model.public_heuristic_logit_bias_profile = aggressive`
    - `public_heuristic_actor_logit_bias_scale_active = 1.0`
    - `public_heuristic_logit_bias_scale_active = 2.0`
    - native rollout profile counters showed aggressive unrolls only.
  - Throughput was low in this pressure surface because the diverse model lane intentionally reintroduces actor model forwards.

- `runs/b1_continue_u20_b3pressure_aggbiasprofile_u120_s2_20260425`
  - Command shape: continue from `runs/b1_native_aggressive_profile_guard_u80_s2_20260424/training/checkpoints/checkpoint_20.pt`, one B3 pressure lane, aggressive teacher target, aggressive public logit-bias profile, u120.
  - Tiny dev curve:
    - u30 `0.7500`, B3 `0.50`
    - u40 `0.8333`, B3 `1.00`
    - u60 `0.7917`, B3 `0.75`
    - u100 `0.7917`, B3 `0.75`
    - u120 `0.7083`, B3 `0.50`
  - Steady local throughput, u40+: mean `5742.9`, max `9819.9`, final `5074.8` samples/sec.
  - Verdict: plumbing useful, learning result not better. Aggressive bias profile did not create an upward curve and may slightly hurt local throughput/noise on this surface.

- `runs/b1_continue_u20_b3pressure_noprogress001_u100_s2_20260425`
  - Command shape: best one-lane B3 pressure recipe plus `rewards.shaping.no_progress_penalty=0.01`.
  - Tiny dev curve essentially matched the current best:
    - u40 `0.8333`, B3 `1.00`
    - u100 `0.7917`, B3 `0.75`
  - Steady local throughput, u40+: mean `6035.3`, max `9815.3`, final `5366.6` samples/sec.
  - Verdict: existing simulator no-progress penalty does not address the suspicious pass-with-options behavior.

- `runs/b1_continue_u20_b3pressure_passpenalty002_u100_s2_20260425`
  - Command shape: best one-lane B3 pressure recipe plus new `rewards.shaping.pass_with_nonpass_penalty=0.02`.
  - Tiny dev curve:
    - u40 `0.8333`, B3 `1.00`
    - u80 `0.7500`, B3 `0.50`
    - u100 `0.8333`, B3 `1.00`
  - Steady local throughput, u40+: mean `5958.1`, max `9727.2`, final `5295.8` samples/sec.
  - 16-pair scalar eval artifact: `runs/b1_continue_u20_b3pressure_passpenalty002_u100_s2_20260425/eval/final_eval/summary.json`
    - B0 vs B1 alias: B1 wins `32/32`.
    - B1 alias mirror: `16/32`.
    - B1 alias vs B2: `32/32`.
    - B1 alias vs B3: `19/32`.
    - B1 alias vs B4: `31/32`.
  - Verdict: reject as current best. Tiny eval looked promising but scalar B3 was worse than current best `22/32`.

- `runs/b1_continue_u20_b3b4pressure_model4_u80_s2_20260425`
  - Command shape: B3+B4 fixed pressure list, `diverse_opponent_actor_count=4`, `diverse_model_actor_count=4`, `diverse_opponent_batch_fraction=0.5`, u80.
  - Model-data fraction moved as intended:
    - mean `policy_train_fraction = 0.2163`
    - mean model train rows about `3544/update`
  - Tiny dev curve still did not improve:
    - u40 `0.8333`, B3 `1.00`
    - u50 `0.6667`, B3 `0.50`
    - u80 `0.7500`, B3 `0.50`
  - Steady local throughput, u40+: mean `5259.9`, max `7969.8`, final `4708.7` samples/sec.
  - Verdict: higher model-data fraction alone is not the missing ingredient. On local hardware it also costs throughput, though this cost should be re-evaluated on the multi-GPU L40 server before making a server claim.

### Diagnosis update

- Current best remains `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`, checkpoint `checkpoint_100.pt`, `policy_000009`, with 16-pair scalar B3 `22/32`.
- The new runs did not beat that best.
- The plateau is unlikely to be explained by:
  - profile mismatch alone;
  - too few model rows alone;
  - existing no-progress shaping;
  - a naive pass-with-nonpass penalty.
- Suspicious metric: `teacher_main_move_fraction` and `teacher_move_source_supported_fraction` are repeatedly `0.0`, while the B3 replay audit found `main_move/main_move` as a major high-disagreement family.
  - Example from current best tail: update 98-100 had `teacher_main_move_fraction = 0.0`, `teacher_move_source_supported_fraction = 0.0`, while `teacher_main_play_character_fraction` was about `0.29-0.34` and `teacher_attack_fraction` about `0.08-0.11`.
  - This suggests the hard teacher-action auxiliary is not teaching main-move source/target details at all on the active model-training rows, even though main-move detail matters against B3.
- Next best structural hypothesis:
  - Add a diagnostic and/or target path that exposes public heuristic target family mass by action family, especially `main_move`, and then test a main-move-aware auxiliary/target rather than simply increasing total model rows or global teacher coefficients.
  - Do not promote the new pass penalty or aggressive bias profile into defaults yet. Keep them as explicit research knobs.

## 2026-04-25 B1 learning loop addendum: public main-move signal and teacher-row phase

### New code/diagnostics

- Added `training.structured_aux.teacher_public_main_move_coef`.
  - This is separate from the existing hard `teacher_move_source_coef`.
  - It uses the packed soft public heuristic target distribution, not the sparse hard `teacher_family == main_move` label.
  - It records target family mass diagnostics:
    - `teacher_public_heuristic_target_main_play_character_mass`
    - `teacher_public_heuristic_target_main_move_mass`
    - `teacher_public_heuristic_target_attack_mass`
    - `teacher_public_heuristic_target_pass_mass`
  - It adds a soft auxiliary over main-move source and destination groups when the public target puts mass on legal `main_move` candidates:
    - `teacher_public_main_move_loss`
    - `teacher_public_main_move_source_loss`
    - `teacher_public_main_move_slot_loss`
    - `teacher_public_main_move_source_accuracy`
    - `teacher_public_main_move_slot_accuracy`
    - `teacher_public_main_move_supported_fraction`
- Also fixed/kept from earlier in this loop:
  - `training.structured_aux.teacher_public_heuristic_label_profile`, so hard teacher labels can use `base`, `aggressive`, or `control`.
  - `model.public_heuristic_logit_bias_profile`, so public logit bias can align with the target profile.
- Focused verification:
  - `python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py`
  - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py -k "public_heuristic or move_source" --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py -k "public_heuristic or antistall" --tb=short`

### Causal diagnostics

- `runs/b1_public_mainmove_aux_smoke_u22_s2_20260425`
  - Purpose: verify the new soft public main-move auxiliary is live.
  - Result:
    - hard `teacher_main_move_fraction` was only `0.0` to `0.007`;
    - soft target `teacher_public_heuristic_target_main_move_mass` was about `0.14` to `0.19`;
    - `teacher_public_main_move_supported_fraction = 1.0`;
    - `teacher_public_main_move_loss` was nonzero.
  - Verdict: the earlier main-move failure was not absence of public target signal. It was that the hard teacher labels almost never exposed that signal.
- `runs/b3_disagreement_audit_currentbest_u100_20260425`
  - Source: current best before this loop, `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`, `policy_000009`.
  - Reran 16 games from 8 paired B3 seeds.
  - Key aggregate:
    - `mean_total_variation = 0.4315`
    - `max_total_variation = 0.9742`
    - top family pairs were same-family, not cross-family:
      - `main_play_character/main_play_character`: 97
      - `main_move/main_move`: 95
    - top label gaps:
      - B1 often preferred `main_play_character(hand_index=0, stage_slot=1)` while B3 preferred `main_play_character(hand_index=0, stage_slot=0)`.
      - B1 spread `main_move` probability across many source/target pairs while B3 was effectively one-hot on a specific move.
  - Interpretation: B1 usually knows the family, but not the sharp within-family tactical choice. This explains why generic more-data and pass-penalty experiments did not move B3 much.

### Rejected continuation experiments

- `runs/b1_continue_u20_b3pressure_publicmainmove005_u100_s2_20260425`
  - One B3 pressure lane, aggressive public target, new `teacher_public_main_move_coef=0.05`.
  - Steady u40+ throughput: mean `5957.6`, final `5293.9` samples/sec.
  - Main-move signal live:
    - `teacher_public_heuristic_target_main_move_mass` mean `0.1953`.
    - `teacher_public_main_move_loss` mean `0.4137`.
    - source accuracy mean `0.9589`, slot accuracy mean `0.7585`.
  - Tiny dev ended at `0.7917`; did not beat the old best.
  - Verdict: useful diagnostics, not a better model.
- `runs/b1_continue_u20_teacherfade_modelmix_publicmm005_u100_s2_20260425`
  - Tried a model/teacher actor fade from update 20 to 80.
  - It raised trainable model rows to about `3349/update`, but local throughput fell to mean `3313.8` samples/sec and B3 did not improve.
  - Verdict: too expensive locally and not better. This does not rule out server-side model-row scaling, but it is not the current B1 recipe.
- `runs/b1_continue_u20_b3pressure_coldpublict4_mm010_u100_s2_20260425`
  - Colder public target (`temperature=4.0`) and `teacher_public_main_move_coef=0.1`.
  - Throughput stayed comparable to the one-lane pressure recipe: mean `5951.6`, final `5277.8` samples/sec.
  - Tiny dev ended at `0.7917`.
  - Verdict: colder target alone did not solve the same-family diffusion.
- `runs/b1_continue_u20_b3pressure_aggbias5_coldt4_mm010_u100_s2_20260425`
  - Added aggressive public bias profile and larger learner bias scale (`5.0`) on top of the cold target.
  - Tiny dev ended at `0.7917`; loss was high.
  - Verdict: stronger public bias plus cold target was not enough.
- `runs/b1_continue_u100_trainheurrows_aggtarget_u160_s2_20260425`
  - Continued the teacher-row recipe from u100 to u160 without switching to model pressure.
  - Tiny dev stayed stable (`0.8333` at u120/u160), but targeted scalar B3 slipped from `14/16` at u100 to `13/16` at u160.
  - Verdict: pure teacher-row compression is useful up to about u100 on this local surface, but should not be blindly extended.
- `runs/b1_continue_u100_teacherimit_then_b3pressure_u160_s2_20260425`
  - Switched from the u100 teacher-row checkpoint to B3 pressure/RL through u160.
  - Tiny dev at u160 looked very good (`0.8750`), but targeted scalar B3 was only `12/16`.
  - Verdict: tiny dev was optimistic; reject as current best.

### New best candidate

- `runs/b1_continue_u20_trainheurrows_aggtarget_u60_s2_20260425`
  - Continue from `runs/b1_native_aggressive_profile_guard_u80_s2_20260424/training/checkpoints/checkpoint_20.pt`.
  - Main change: train on native heuristic actor rows instead of excluding them:
    - `training.train_on_heuristic_actor_rows=true`
    - `training.diverse_opponent_actor_count=0`
    - `training.diverse_model_actor_count=0`
    - `training.structured_aux.teacher_public_heuristic_profiles=["aggressive"]`
    - `training.structured_aux.teacher_public_heuristic_label_profile="aggressive"`
    - `training.structured_aux.teacher_public_heuristic_temperature=8.0`
    - `training.structured_aux.teacher_public_main_move_coef=0.1`
    - `model.public_heuristic_logit_bias_profile="aggressive"`
    - `model.public_heuristic_logit_bias_scale=3.0`
  - Then continued with the same recipe:
    - `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425`
    - checkpoint: `training/checkpoints/checkpoint_100.pt`
    - snapshot: `policy_000009`
- Why this is a real shift:
  - The previous B3-pressure recipe trained on only about `5.4%` of rows and paid actor model-forward cost.
  - The new teacher-row phase trains on about `50%` of rows, has `collector_actor_policy_forward_ms=0`, and remains simulator-native.
  - Local tail throughput for `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425`:
    - mean about `15822.5` samples/sec over the last 20 updates;
    - final `14328.7` samples/sec.
  - This is not comparable to pure no-pressure 100k+ throughput, but it is much faster than the one-lane B3 pressure learning surface and should translate better to the L40 server because it keeps actor-side native heuristic rollout.
- Small scalar anchor check, 8 paired seeds, explicit policy set:
  - Eval artifact: `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425/eval/final_eval/summary.json`
  - `policy_000009` results:
    - vs `B0 RandomLegal`: `16/16`, mean `1.0`
    - vs `B2 HeuristicPublic`: `16/16`, mean `1.0`
    - vs `B3 HeuristicPublicAggro`: `14/16`, mean `0.875`
    - vs `B4 HeuristicPublicControl`: `16/16`, mean `1.0`
    - mirror: `8/16`, mean `0.5`
  - Same B3 seed slice for the previous best `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424`, `policy_000009`: `10/16`, mean `0.625`.
  - Verdict: current best B1 candidate. It is not thesis-final evidence, but it is a real same-surface improvement and the best local anchor so far.

### Next hypotheses

- Do not blindly extend pure teacher-row training past u100; the u160 continuation slightly regressed B3 despite stable tiny dev.
- Next high-value experiment is a deliberate phase schedule:
  - teacher-row native imitation to around u80-u100;
  - then either freeze/keep the best u100 checkpoint or use a much gentler RL/self-play phase with checkpoint selection, not raw continued SGD.
- Add an explicit preset for the current best B1 teacher-row phase once confirmed on another seed or a slightly larger scalar eval.
- Before server use:
  - dry-run autoscale;
  - 1-2 update smoke;
  - verify `train_on_heuristic_actor_rows=true`, native rollout active, `collector_actor_policy_forward_ms=0`, and rank-0 artifacts only.

### Follow-up on the apparent post-u100 collapse

- User concern: the u100 candidate looked good, but u160 continuations looked worse on the first 8-paired-seed scalar checks. This would be worrying if it were a real collapse.
- Rechecked the key comparison with B3-only 16 paired seeds / 32 games:
  - u100 teacher-row candidate:
    - run: `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425`
    - policy: `policy_000009`
    - B3 scalar: `28/32`, mean `0.875`, CI roughly `[0.7625, 0.9645]`
  - u160 pure teacher-row continuation:
    - run: `runs/b1_continue_u100_trainheurrows_aggtarget_u160_s2_20260425`
    - policy: `policy_000012`
    - B3 scalar: `27/32`, mean `0.84375`, CI roughly `[0.7209, 0.9416]`
  - u160 teacher-row plus one B3 pressure lane:
    - run: `runs/b1_continue_u100_trainheurrows_plus_b3pressure_u160_s2_20260425`
    - policy: `policy_000012`
    - B3 scalar: `26/32`, mean `0.8125`, CI roughly `[0.6934, 0.9160]`
- Updated interpretation:
  - The post-u100 behavior is not a hard collapse on the more informative B3-only 32-game check. u100 and u160 pure teacher-row are separated by only one game and have strongly overlapping uncertainty.
  - It is still a real plateau. Metrics continue to look cleaner because the model is compressing the aggressive public heuristic distribution:
    - u100 tail entropy about `0.3993`, target top1 mass about `0.7187`;
    - u160 pure teacher tail entropy about `0.2925`, target top1 mass about `0.7813`;
    - vtrace clipping remains `0.0`, confirming this phase is imitation-dominated, not discovering much new RL signal.
  - Adding a small B3 model-pressure lane after u100 did not improve the plateau and cost some throughput/quality locally.
- Current best remains the u100 teacher-row candidate. The next improvement should not be "just run longer"; it should be either:
  - confirm the u100 teacher-row phase on another seed/server smoke and use best-checkpoint selection; or
  - design a genuinely new post-teacher phase with safer exploration/self-play and checkpoint gating.

### Plateau broken with low-LR teacher-row continuation

- Goal: answer the user's concern that the B1 anchor stopped improving after about u100 and that a thesis story built on a u20/u100 peak would be weak.
- Same-surface baseline before this pass:
  - `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425`
  - snapshot: `policy_000009`, update 100
  - B3-only scalar, 32 paired seeds / 64 games:
    - artifact: `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425/eval/final_eval_policy000009_u100_b3_s32_archive/summary.json`
    - vs `B3 HeuristicPublicAggro`: `54/64`, mean `0.84375`, CI roughly `[0.7635, 0.9143]`
- Rejected post-u100 continuations:
  - `runs/b1_continue_u100_native_profilemix_softaux_u160_s2_20260425`
    - native profile cycle over `base/aggressive/control`, softer aux weights
    - B3-only 16 paired seeds / 32 games: `25/32`, mean `0.78125`
    - verdict: profile diversity raised entropy but hurt B3 tactics.
  - `runs/b1_continue_u100_trainheurrows_entropyreheat_u160_s2_v2_20260425`
    - aggressive native teacher rows, lower aux weights, entropy `0.08 -> 0.04`
    - tail throughput over last 20 updates: mean `88887.1`, final `87154.4` samples/sec
    - B3-only 16 paired seeds / 32 games:
      - u120 `27/32`
      - u140 `27/32`
      - u160 `27/32`
    - verdict: simple exploration reheating preserved throughput but did not beat the plateau.
  - `runs/b1_continue_u100_trainheurrows_samefamily_u160_s2_20260425`
    - added `teacher_same_family_action_coef=0.25` and `teacher_move_source_coef=0.10`
    - tail throughput over last 20 updates: mean `86317.9`, final `84877.5` samples/sec
    - B3-only 16 paired seeds / 32 games: `27/32`, mean `0.84375`
    - verdict: the same-family B3 disagreement diagnosis was real, but direct same-family imitation alone did not improve the scalar.
  - `runs/b1_continue_u100_trainheurrows_mirrorlane_u160_s2_20260425`
    - kept teacher rows but added one model mirror/self-play lane (`diverse_opponent_actor_count=1`, `diverse_model_actor_count=1`, empty fixed opponent ids)
    - tail throughput over last 20 updates: mean `22310.1`, final `20325.5` samples/sec
    - B3-only 16 paired seeds / 32 games: `25/32`, mean `0.78125`
    - verdict: current one-lane model self-play is worse and much slower locally. Do not use as the B1 anchor continuation without redesign.
- Successful structural change:
  - run: `runs/b1_continue_u100_trainheurrows_lowlr_u200_s2_20260425`
  - continued from `runs/b1_continue_u60_trainheurrows_aggtarget_u100_s2_20260425/training/checkpoints/checkpoint_100.pt`
  - key change: keep the successful aggressive native teacher-row objective, but lower optimizer LR from `0.0002` to `0.00005`
  - retained:
    - `training.train_on_heuristic_actor_rows=true`
    - `training.diverse_opponent_actor_count=0`
    - `training.diverse_model_actor_count=0`
    - `training.heuristic_native_rollout_enabled=true`
    - `training.heuristic_actor_hidden_state_tracking=false`
    - `training.heuristic_native_rollout_profile="aggressive"`
    - `training.structured_aux.teacher_public_heuristic_profiles=["aggressive"]`
    - `training.structured_aux.teacher_public_heuristic_label_profile="aggressive"`
    - `training.structured_aux.teacher_public_heuristic_temperature=8.0`
    - `training.structured_aux.teacher_public_main_move_coef=0.1`
    - `model.public_heuristic_logit_bias_profile="aggressive"`
    - `model.public_heuristic_logit_bias_scale=3.0`
  - tail throughput over last 20 updates: mean `83989.5`, final `83131.7` samples/sec
  - snapshot: `policy_000014`, update 200
  - B3-only 16 paired seeds / 32 games:
    - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u200_s2_20260425/eval/final_eval/summary.json`
    - result: `29/32`, mean `0.90625`
  - verdict: first clear evidence that post-u100 training can continue upward without actor-lane throughput collapse.
- First improved checkpoint:
  - run: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425`
  - continued from `runs/b1_continue_u100_trainheurrows_lowlr_u200_s2_20260425/training/checkpoints/checkpoint_200.pt`
  - snapshot: `policy_000019`, update 300
  - tail throughput over last 20 updates: mean `127446.8`, final `124267.0` samples/sec
  - B3-only 16 paired seeds / 32 games:
    - artifact: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425/eval/final_eval_policy000019_u300_b3_s16_archive/summary.json`
    - result: `30/32`, mean `0.9375`
  - B3-only 32 paired seeds / 64 games:
    - artifact: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425/eval/final_eval_policy000019_u300_b3_s32_archive/summary.json`
    - result: `59/64`, mean `0.921875`, CI roughly `[0.8515, 0.9718]`
  - safety checks:
    - vs `B2 HeuristicPublic`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425/eval/final_eval_policy000019_u300_b2_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
    - vs `B4 HeuristicPublicControl`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425/eval/final_eval_policy000019_u300_b4_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
  - verdict: first strong low-LR continuation checkpoint. It beats the old u100 checkpoint on the same expanded B3 surface (`59/64` vs `54/64`) while preserving the high-throughput simulator-native training path.
- Reproducible preset:
  - Added `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_aggressive_teacher_warmup.yaml`.
  - Purpose: capture the u0/u100 aggressive native teacher-row warmup phase.
  - Added `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_lowlr_continuation.yaml`.
  - Purpose: capture the low-LR B1 continuation phase as config instead of a long CLI override list.
  - Promoted both through `python/scripts/thesis_run.py` aliases:
    - `b1-anchor-benchmark-warmup`
    - `b1-anchor-benchmark-lowlr-continuation`
    - both default to the existing reduced-surface B1 eval preset `b1-anchor-benchmark-eval-auto-gpu`.
  - Added focused regression coverage:
    - config loader verifies warmup/continuation LR, aggressive teacher profile, native rollout, no model/opponent lanes, and aggressive public bias.
    - thesis wrapper verifies both phase aliases resolve to the matching B1 benchmark eval surface.
  - Validation:
    - warmup local autoscale dry-run resolves to `8 x 64 = 512` envs, 1 learner GPU.
    - warmup `uc1-l40-4` dry-run resolves to `32 x 64 = 2048` envs, 4 learner GPUs, DDP.
    - low-LR continuation `uc1-l40-3` dry-run resolves to `24 x 64 = 1536` envs, 3 learner GPUs, DDP.
    - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_prefetch_and_native_rollout_overrides --tb=short`
    - `uv run pytest -q python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_b1_anchor_phase_presets_to_matching_eval_surface --tb=short`
  - Intended use: resume from the aggressive teacher-row warmup checkpoint around u100/u200, not necessarily train from scratch at low LR.
- Longer continuation:
  - run: `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425`
  - continued same low-LR recipe to u500
  - snapshot checks against B3, 16 paired seeds / 32 games:
    - u350 `policy_000020`: `30/32`, mean `0.9375`
    - u400 `policy_000021`: `28/32`, mean `0.875`
    - u450 `policy_000022`: `30/32`, mean `0.9375`
    - u500 `policy_000023`: `29/32`, mean `0.90625`
  - larger B3 checks, 32 paired seeds / 64 games:
    - u350 `policy_000020`: `56/64`, mean `0.875`
    - u450 `policy_000022`: `60/64`, mean `0.9375`, CI roughly `[0.8694, 0.9811]`
  - u450 safety checks:
    - vs `B2 HeuristicPublic`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/eval/final_eval_policy000022_u450_b2_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
    - vs `B4 HeuristicPublicControl`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/eval/final_eval_policy000022_u450_b4_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
  - u450 throughput:
    - updates 431-450 mean `146193.1`, final `143506.5` samples/sec
    - entropy over updates 431-450 mean `0.4082`, final `0.3146`
  - tail throughput over last 20 updates at u500: mean `131492.0`, final `129655.8` samples/sec
  - verdict: current best local B1 no-league anchor candidate is u450 (`policy_000022`), not the final u500 checkpoint. Running past u300 can improve further, but it is not monotonic; checkpoint selection matters.
- Updated interpretation:
  - The earlier post-u100 plateau was not a fundamental "cannot learn past 100 updates" problem.
  - The failed continuations suggest the bad pattern was not solved by more entropy, profile mixing, direct same-family imitation, or naive one-lane self-play.
  - The successful change points to optimizer step size / over-compression as the practical blocker after the teacher-row breakthrough. Lower LR lets the same native teacher-row objective continue improving to at least u450 with minimal throughput loss.
- Next hypotheses:
  - Promote a dedicated B1 low-LR continuation preset or phase schedule: higher LR teacher-row ramp to about u100, then LR `5e-5` continuation with checkpoint selection around u300-u500.
  - Confirm on another seed and/or server smoke before treating u450 as thesis-grade.
  - For server: this recipe should scale better than actor-lane pressure because it keeps native rollout, no model actor lanes, and no hidden-state tracking.

### Seed-3 replication and very-low-LR check

- Purpose: test whether the low-LR post-u100 improvement is repeatable, not just seed-2/local-seed luck.
- Very-low-LR continuation from the seed-2 u450 checkpoint:
  - run: `runs/b1_continue_u450_trainheurrows_vlowlr_u650_s2_20260425`
  - resumed from `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/training/checkpoints/checkpoint_450.pt`
  - changed only LR from `5e-5` to `1e-5`
  - throughput:
    - updates 551-600 mean `210727.9`, final `195263.5` samples/sec
    - updates 601-650 mean `180848.4`, final `170405.0` samples/sec
  - B3 screen, 16 paired seeds / 32 games:
    - u600 `policy_000025`: `29/32`, mean `0.90625`
    - u650 `policy_000026`: `29/32`, mean `0.90625`
  - verdict: reject as an improvement branch. It is very fast and stable, but did not beat u300/u450.
- Fresh seed-3 warmup:
  - run: `runs/b1_trainheurrows_aggtarget_u100_s3_20260425`
  - preset-equivalent recipe: aggressive native teacher-row warmup, LR `2e-4`, seed `3`, u100
  - snapshot: `policy_000002`, update 100
  - throughput over updates 81-100: mean `39263.5`, final `41040.9` samples/sec
  - B3 screen, 16 paired seeds / 32 games:
    - artifact: `runs/b1_trainheurrows_aggtarget_u100_s3_20260425/eval/final_eval_policy000002_u100_b3_s16_archive/summary.json`
    - result: `27/32`, mean `0.84375`
- Fresh seed-3 low-LR continuation:
  - run: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425`
  - resumed from `runs/b1_trainheurrows_aggtarget_u100_s3_20260425/training/checkpoints/checkpoint_100.pt`
  - continuation preset: `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_benchmark_lowlr_continuation.yaml`
  - snapshot: `policy_000009`, update 450
  - throughput over updates 431-450: mean `75852.8`, final `75651.6` samples/sec
  - B3 screens:
    - u300 `policy_000006`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/eval/final_eval_policy000006_u300_b3_s16_archive/summary.json`
      - result: `31/32`, mean `0.96875`
    - u450 `policy_000009`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/eval/final_eval_policy000009_u450_b3_s16_archive/summary.json`
      - result: `31/32`, mean `0.96875`
    - u450 `policy_000009`, 32 paired seeds / 64 games:
      - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/eval/final_eval_policy000009_u450_b3_s32_archive/summary.json`
      - result: `59/64`, mean `0.921875`, CI roughly `[0.8559, 0.9737]`
  - safety checks for seed-3 u450:
    - vs `B2 HeuristicPublic`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/eval/final_eval_policy000009_u450_b2_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
    - vs `B4 HeuristicPublicControl`, 16 paired seeds / 32 games:
      - artifact: `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/eval/final_eval_policy000009_u450_b4_s16_archive/summary.json`
      - result: `32/32`, mean `1.0`
- Updated verdict:
  - The low-LR continuation story now replicates on a fresh local seed:
    - seed 2: u100 `54/64` vs B3, u450 `118/128` vs B3
    - seed 3: u100 `27/32` vs B3, u450 `59/64` vs B3
  - Current practical recommendation: use the two-phase B1 recipe with checkpoint selection around u300-u450. Do not continue blindly to final checkpoint.

### Checkpoint soup rejection

- Hypothesis: because seed-2 u300 and u450 tied on the larger B3 surface (`118/128` each), a simple same-trajectory weight average might smooth the non-monotonic wobble.
- Artifact:
  - created averaged snapshot in `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/training/snapshots/policy_soup_u300_u450/weights.pt`
  - source snapshots:
    - u300: `runs/b1_continue_u200_trainheurrows_lowlr_u300_s2_20260425/training/snapshots/policy_000019/weights.pt`
    - u450: `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/training/snapshots/policy_000022/weights.pt`
  - averaged floating tensors 50/50; copied matching non-floating buffers.
- Eval:
  - B3 screen, 16 paired seeds / 32 games:
    - artifact: `runs/b1_continue_u300_trainheurrows_lowlr_u500_s2_20260425/eval/final_eval_policy_soup_u300_u450_b3_s16_archive/summary.json`
    - result: `28/32`, mean `0.875`
- Verdict: reject. The good u300/u450 policies are not improved by naive linear weight averaging.

## 2026-04-25 Main league transition after B1 anchor

### Why the reduced B1 anchor cannot simply be plugged into the main league

- The promoted local B1 benchmark anchors use the reduced benchmark model surface:
  - `gru_hidden_size=192`
  - `encoder_mlp_width=192`
  - `typed_feature_width=48`
- The main thesis model uses the full server surface:
  - `gru_hidden_size=248`
  - `encoder_mlp_width=248`
  - `typed_feature_width=62`
- The B1 baseline import path intentionally validates `model` and `environment` config sections and tensor shapes before importing.
- Therefore, the reduced B1 checkpoint is a learning recipe / local evidence anchor, not a directly importable opponent for the full-size league run.
- Action taken: promote matching full-size B1 phase presets, then a full-size B1-anchored league preset.

### Full-size B1 phase presets

- Added `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_aggressive_teacher_warmup.yaml`.
  - Extends the full-size no-league server preset, not the reduced benchmark preset.
  - Uses the promoted warmup recipe:
    - native aggressive rollout;
    - `train_on_heuristic_actor_rows=true`;
    - no model/opponent actor lanes;
    - aggressive public bias profile and scale `3.0`;
    - aggressive public heuristic teacher label profile;
    - public heuristic temperature `8.0`;
    - public main-move aux coefficient `0.1`;
    - LR `2e-4`.
- Added `configs/presets/baselines/structured_acceptance_thesis_model_server_train_auto_gpu_noleague_lowlr_continuation.yaml`.
  - Same full-size B1 recipe, but LR `5e-5`.
  - Intended for resume from the full-size warmup checkpoint, mirroring the reduced benchmark two-phase result.

### B1-anchored main league preset

- Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league.yaml`.
- Key differences from the previous main server train preset:
  - aligns model public heuristic bias to the B1 recipe:
    - `model.public_heuristic_logit_bias_profile=aggressive`;
    - `model.public_heuristic_logit_bias_scale=3.0`;
  - keeps the fast native teacher-row backbone:
    - `training.train_on_heuristic_actor_rows=true`;
    - `training.heuristic_native_rollout_enabled=true`;
    - `training.heuristic_actor_hidden_state_tracking=false`;
    - `training.heuristic_native_rollout_profile=aggressive`;
  - uses the low-LR continuation stance:
    - `training.optimizer.learning_rate=5e-5`;
  - aligns teacher targets to the B1 recipe:
    - aggressive public heuristic profile;
    - aggressive hard label profile;
    - public heuristic temperature `8.0`;
    - public main-move aux coefficient `0.1`;
  - reduces pure heuristic public pressure and explicitly mixes in:
    - B1 no-league baseline;
    - B2/B3/B4 heuristic-public variants;
    - promotion/champion/hard-negative machinery as it becomes available.
- Current league sampling in the new preset:
  - `heuristic_public_mix_fraction=0.55`, final `0.35`;
  - `heuristic_public_variant_mix_fraction=0.20`;
  - `noleague_baseline_mix_fraction=0.25`;
  - `champion_mix_fraction=0.20`;
  - `hard_negative_mix_fraction=0.15`.
- Rationale: avoid the old failure mode where the league model was trained under a weaker/mismatched surface than the B1 anchor, while still not jumping straight to noisy self-play pressure.

### Wrapper aliases and tests

- Added `python/scripts/thesis_run.py` aliases:
  - `b1-anchor-fullsize-warmup`;
  - `b1-anchor-fullsize-lowlr-continuation`;
  - `thesis-model-server-train-b1anchored`.
- All three default to the main thesis eval surface, not the reduced benchmark eval surface.
- Added focused tests:
  - `python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_prefetch_and_native_rollout_overrides`
    - now verifies full-size warmup, full-size low-LR continuation, and B1-anchored league config values.
  - `python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_fullsize_b1_and_b1anchored_league_presets_to_main_eval_surface`
    - verifies the wrapper aliases resolve to the expected train/eval presets.
- Test results:
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_prefetch_and_native_rollout_overrides --tb=short`
    - `1 passed`
  - `uv run pytest -q python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_fullsize_b1_and_b1anchored_league_presets_to_main_eval_surface --tb=short`
    - `1 passed`

### Local contract smokes

- Full-size B1 warmup smoke:
  - run: `runs/main_b1_fullsize_warmup_u2_contract_smoke_20260425`
  - command surface:
    - `structured_acceptance_thesis_model_server_train_auto_gpu_noleague_aggressive_teacher_warmup.yaml`
    - `--autoscale --hardware-profile local`
    - 2 updates, seed `11`
  - result:
    - completed;
    - persisted `b1_noleague_baseline`;
    - topology `8 x 64 = 512`;
    - native aggressive rollout active;
    - local throughput is not meaningful for server claims, but metrics were finite.
- B1-anchored league import smoke:
  - run: `runs/main_league_b1anchored_import_required_u1_contract_smoke_20260425`
  - command surface:
    - `structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league.yaml`
    - `--b1-baseline-run-dir runs/main_b1_fullsize_warmup_u2_contract_smoke_20260425`
    - `--autoscale --hardware-profile local`
    - 1 update, seed `11`
  - result:
    - completed;
    - imported required B1 promotion anchor:
      - `policy_id=b1_noleague_baseline`;
      - source run `runs/main_b1_fullsize_warmup_u2_contract_smoke_20260425`;
    - topology `8 x 64 = 512`;
    - native aggressive rollout active;
    - loss/entropy/value metrics finite.
- Wrapper dry-run:
  - `uv run python python/scripts/thesis_run.py --preset thesis-model-server-train-b1anchored --run-label wrapper_b1anchored_plan_smoke_20260425 --dry-run --skip-compare --skip-eval --runtime-mode train_async_fast --max-updates 1 --train-arg=--autoscale --train-arg=--hardware-profile --train-arg=local`
  - result: planned train command with `structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league.yaml`.

### Current main-model recommendation

- Do not use the reduced B1 benchmark checkpoint directly as the full league B1 opponent.
- On server:
  1. Train full-size B1 with `b1-anchor-fullsize-warmup` to the warmup boundary.
  2. Resume with `b1-anchor-fullsize-lowlr-continuation` and select the best checkpoint around the equivalent u300-u500 window.
  3. Start `thesis-model-server-train-b1anchored` from that checkpoint and pass the full-size B1 run as `--b1-baseline-run-dir`.
  4. Keep early league runs short and artifact-gated before long training.
- Big gains probably require the league phase, but the first priority is avoiding a mismatched/weak B1 anchor inside league.

## 2026-04-25 - reduced-size B1-anchored league tests before server scale-up

User direction: test the main/league changes on the smaller benchmark model sizes locally, then scale the recipe up on the multi-GPU Linux server only after the reduced surface earns it.

### Added reduced league presets

- Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark.yaml`.
  - Reduced model surface: `192/192/48`.
  - Role fixed to `experiment.role=main` so league runs do not republish themselves as `b1_noleague_baseline`.
  - B1-aligned aggressive teacher/bias recipe and low LR `5e-5`.
- Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_localpromo.yaml`.
  - Same reduced league preset, but with internally consistent local promotion seeds.
  - Fixes the seed-file contract failure seen when overriding `league.promotion.paired_seeds=2` against the 64-seed production promotion file.
  - Uses:
    - `seed_sets.promotion_gate=configs/seeds/local_promotion_eval_seeds.txt`;
    - `league.promotion.seed_file=configs/seeds/local_promotion_eval_seeds.txt`;
    - `league.promotion.paired_seeds=8`;
    - matching `evaluation.seed_files` and `reproducibility.seed_files`.
- Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_localpromo.yaml`.
  - Structural test: pure model-policy focal actors after B1 warmup.
  - Keeps model section B1-compatible for import, but changes training behavior:
    - `actor_policy_backend=model`;
    - `actor_heuristic_fraction=0.0`;
    - `heuristic_native_rollout_enabled=false`;
    - LR `2e-5`;
    - entropy `0.05 -> 0.02`.
- Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_modelbridge_localpromo.yaml`.
  - Structural bridge test: mostly heuristic-public focal rows plus a live model lane.
  - Uses:
    - `actor_policy_backend=heuristic_public`;
    - `actor_heuristic_fraction=0.85`;
    - `heuristic_native_rollout_enabled=false`;
    - `heuristic_actor_hidden_state_tracking=true`;
    - `train_on_heuristic_actor_rows=true`;
    - LR `2e-5`;
    - entropy `0.035 -> 0.015`.
- Added wrapper aliases:
  - `thesis-model-server-train-b1anchored-benchmark`;
  - `thesis-model-server-train-b1anchored-benchmark-localpromo`;
  - `thesis-model-server-train-b1anchored-benchmark-selfplay-localpromo`;
  - `thesis-model-server-train-b1anchored-benchmark-modelbridge-localpromo`.
- Loader change:
  - `python/weiss_rl/config/parse.py` now permits top-level `seed_sets` in preset YAML, not just canonical run artifacts.
- Focused tests passed:
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_b1anchored_benchmark_league_to_benchmark_eval_surface --tb=short`

### Topology checks

- Reduced local-promotion league preset local dry-run:
  - `8 x 64 = 512` envs, 1 learner GPU.
- Reduced self-play local-promotion league preset local dry-run:
  - `8 x 64 = 512` envs, 1 learner GPU.
- L40-4 dry-run without local DDP init:
  - `32 x 64 = 2048` envs, 4 learner GPUs, resolved learner parallelism `ddp`.
- Note: local Windows `--ddp` dry-run attempted process-group initialization and failed because this local PyTorch build has no NCCL. The non-DDP L40 profile dry-run is still useful topology evidence; real DDP validation must happen on Linux.

### Heuristic-row reduced league continuation

- First valid short reduced league import smoke:
  - run: `runs/b1anchored_league_benchmark_continue_u450_to_u470_s3_promosmoke_20260425`
  - resumed from `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt`
  - imported B1 anchor from `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425`
  - promotion skipped due league warmup.
  - scalar eval, 16 paired seeds / 32 games:
    - `policy_000011 vs b1_noleague_baseline`: `16/32`, mean `0.500`
    - `policy_000011 vs B3 HeuristicPublicAggro`: `29/32`, mean `0.906`
    - `b1_noleague_baseline vs B3 HeuristicPublicAggro`: `28/32`, mean `0.875`
  - Verdict: preserves B1, maybe tiny noisy B3 lift, not enough to claim improvement.
- Longer reduced league with bad CLI seed override:
  - run: `runs/b1anchored_league_benchmark_continue_u470_to_u670_s3_promosmoke_20260425`
  - failed at first active promotion with:
    - `promotion gate expected 2 paired seeds ... found 64`
  - Verdict: seed contract failure, not learning failure.
- Fixed longer reduced league with local-promotion preset:
  - run: `runs/b1anchored_league_benchmark_localpromo_u470_to_u670_s3_20260425`
  - completed updates `471-670`.
  - promotion:
    - update 550: passed;
    - update 600: passed;
    - update 650: passed.
  - registry:
    - pinned `b1_noleague_baseline`;
    - champions `policy_000013`, `policy_000014`, `policy_000015`;
    - no rejected snapshots.
  - training metrics:
    - training log throughput mean `160,271 samples/sec`, max `478,966`, final `24,795`;
    - actor env steps/sec mean `74,998`, max `397,811`, final `63,183`;
    - entropy mean `0.414`, final `0.562`;
    - loss mean `0.311`, final `0.489`;
    - vtrace clip rate `0.0`.
  - promotion-gate micro surface:
    - u550 vs B3: `14/16`;
    - u600 vs B3: `15/16`;
    - u650 vs B3: `12/16`;
    - all had `0` truncations.
  - scalar eval, 16 paired seeds / 32 games, policies `policy_000013/14/15`, B1, B3, B4:
    - all promoted policies vs B1: `16/32`, mean `0.500`;
    - all promoted policies vs B3: `28/32`, mean `0.875`;
    - all promoted policies vs B4: `32/32`, mean `1.000`;
    - B1 vs B3: `28/32`, mean `0.875`;
    - B1 vs B4: `32/32`, mean `1.000`.
  - Verdict: the heuristic-row league continuation is safe but not improving B1. It effectively preserves the imported B1 behavior.

### Pure model-policy self-play test

- Run:
  - `runs/b1anchored_league_benchmark_selfplay_localpromo_u670_to_u700_s3_v2_20260425`
  - resumed from `runs/b1anchored_league_benchmark_localpromo_u470_to_u670_s3_20260425/training/checkpoints/checkpoint_670.pt`
  - completed updates `671-700`.
- Runtime proof:
  - `heuristic_native_rollout_enabled=false`;
  - final logged `collector_native_rollout_profile_aggressive_unrolls=0`;
  - final `collector_actor_policy_forward_ms=33465`;
  - final `collector_policy_train_model_rows=8596`;
  - final `collector_policy_train_heuristic_rows=0`.
- Local throughput:
  - training log throughput mean `187,480 samples/sec`, max `561,752`, final `79,971`;
  - actor env steps/sec mean `3,620`, final `3,835`.
  - Interpretation: local model-policy actor path is much slower than native heuristic rollout; this is expected and should be judged on learning quality/server scalability, not local throughput.
- Scalar eval, 16 paired seeds / 32 games:
  - `policy_000016 vs B1`: `13/32`, mean `0.406`;
  - `policy_000017 vs B1`: `13/32`, mean `0.406`;
  - `policy_000018 vs B1`: `15/32`, mean `0.469`;
  - `policy_000016 vs B3`: `16/32`, mean `0.500`;
  - `policy_000017 vs B3`: `16/32`, mean `0.500`;
  - `policy_000018 vs B3`: `17/32`, mean `0.531`;
  - B1 vs B3 on same eval: `28/32`, mean `0.875`.
- Verdict: pure model-policy continuation is live but quality-negative. It drifts away from B1 quickly and should not be scaled.

### Mixed model-lane bridge test

- Run:
  - `runs/b1anchored_league_benchmark_modelbridge_localpromo_u670_to_u700_s3_20260425`
  - same starting checkpoint as pure self-play.
  - completed updates `671-700`.
- Runtime proof:
  - final `collector_native_rollout_profile_aggressive_unrolls=0`;
  - final `collector_actor_policy_forward_ms=34493`;
  - final `collector_policy_train_model_rows=1294`;
  - final `collector_policy_train_heuristic_rows=7049`.
- Local throughput:
  - training log throughput mean `181,193 samples/sec`, max `534,455`, final `77,738`;
  - actor env steps/sec mean `3,531`, final `3,718`.
- Scalar eval, 16 paired seeds / 32 games:
  - `policy_000016 vs B1`: `10/32`, mean `0.3125`;
  - `policy_000018 vs B1`: `12/32`, mean `0.375`;
  - `policy_000016 vs B3`: `17/32`, mean `0.531`;
  - `policy_000018 vs B3`: `19/32`, mean `0.594`;
  - B1 vs B3 on same eval: `28/32`, mean `0.875`.
- Verdict: mixed model-lane bridge is also quality-negative. It does not solve the drift problem and should not be scaled.

### Current diagnosis

- The previous "league is worse than B1" issue is now more concrete:
  - the B1-quality path is a fast heuristic-row/native-rollout teacher path;
  - simply switching to live model-policy actors makes training real, but destabilizes the B1 policy quickly;
  - keeping a small model lane inside mostly heuristic rows still destabilizes B1.
- The heuristic-row league preset is safe but cannot be expected to produce a big post-B1 improvement because it is not meaningfully evolving the model policy through self-play.
- The model-policy league path is the right structural direction for a stronger thesis model, but it needs a B1-constrained mechanism before server scale-up:
  - reference/B1 KL or distillation on collected states;
  - periodic scalar dev-eval rollback against B1/B3 during the model-policy phase;
  - or a much gentler schedule with objective evidence that B1 performance is preserved.
- Do not send pure self-play or modelbridge to the L40 server yet.
- Best current server-facing plan:
  1. Train full-size B1 using the already added full-size warmup and low-LR continuation presets.
  2. Use that full-size B1 as the required imported anchor.
  3. Before long main-model training, add/test a constrained model-policy league phase on the reduced surface.
  4. Only scale once reduced eval shows it preserves B1 while improving at least one hard surface beyond B1.

## 2026-04-25 reduced main-league constrained model-policy loop

### Structural fix: frozen B1 top-action distillation

- Problem tested:
  - Pure model-policy league and mixed model-lane bridge were live but quality-negative.
  - Moving behavior-action BC was not enough because it anchors to sampled actor behavior, not to the known-good B1 policy.
- Code changes:
  - Added `training.behavior_action_bc_coef` telemetry to learner logs.
  - Added `training.reference_policy_top_action_bc_coef` and `training.reference_policy_id`.
  - Training now attaches a frozen imported reference policy model when the reference coefficient is nonzero.
  - IMPALA learner now computes a top-action NLL against the frozen reference policy on the same legal states.
  - The reference path supports the packed legal-candidate scorer path used by the current structured model.
  - Fixed a resume-contract bug: after loading an optimizer state from a checkpoint, `_restore_learner_from_checkpoint()` now reapplies the current preset's `learning_rate` to optimizer param groups. Without this, low-LR continuation presets after resume silently kept the old checkpoint LR.
- New reduced presets:
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_bckl_localpromo.yaml`
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_bcheavy_localpromo.yaml`
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_localpromo.yaml`
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1strong_localpromo.yaml`
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1strong_lowlr_localpromo.yaml`
- Tests:
  - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state python/weiss_rl/tests/test_vtrace.py::test_impala_learner_logging_persists_returned_loss_metrics --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_b1anchored_benchmark_league_to_benchmark_eval_surface --tb=short`

### Negative controls

- `runs/b1anchored_league_benchmark_selfplay_bckl_localpromo_u670_to_u700_s3_20260425`
  - Resume: `runs/b1anchored_league_benchmark_localpromo_u470_to_u670_s3_20260425/training/checkpoints/checkpoint_670.pt`.
  - Config: moving behavior-action BC `0.15`, LR `1e-5`.
  - Scalar eval, 16 paired seeds / 32 games:
    - `policy_000016 vs B1`: `10/32`;
    - `policy_000018 vs B1`: `11/32`;
    - `policy_000016 vs B3`: `17/32`;
    - `policy_000018 vs B3`: `16/32`.
  - Verdict: not enough; moving behavior BC does not preserve B1.
- `runs/b1anchored_league_benchmark_selfplay_bcheavy_u670_to_u700_s3_20260425`
  - Config: moving behavior-action BC `1.0`, nominal LR `5e-6`.
  - Final telemetry: `behavior_action_bc_loss=7.506`, `behavior_action_bc_coef=1.0`; final actor env steps/sec `3,798`.
  - Scalar eval, 16 paired seeds / 32 games:
    - `policy_000016 vs B1`: `11/32`;
    - `policy_000018 vs B1`: `4/32`;
    - `policy_000016 vs B3`: `16/32`;
    - `policy_000018 vs B3`: `15/32`.
  - Verdict: worse; reject moving behavior BC as the main stabilizer.

### Frozen-reference experiments

- `runs/b1anchored_league_benchmark_selfplay_refb1_v2_u670_to_u700_s3_20260425`
  - Config: frozen B1 top-action BC `0.25`, LR `1e-5`.
  - Runtime proof:
    - final `reference_policy_top_action_bc_loss=0.364`, `reference_policy_top_action_bc_coef=0.25`;
    - final `collector_policy_train_model_rows=8596`;
    - final `collector_actor_policy_forward_ms=33740`;
    - final actor env steps/sec `3,799`;
    - final learner throughput `79,566 samples/sec`.
  - Scalar eval, 16 paired seeds / 32 games:
    - `policy_000016 vs B1`: `16/32`;
    - `policy_000018 vs B1`: `16/32`;
    - `policy_000016 vs B3`: `21/32`;
    - `policy_000018 vs B3`: `22/32`;
    - both vs B4: `32/32`.
  - Verdict: structural fix works, but coefficient `0.25` is too loose; still below B1 vs B3.
- `runs/b1anchored_league_benchmark_selfplay_refb1strong_u670_to_u700_s3_20260425`
  - Config: frozen B1 top-action BC `0.5`, LR effectively `1e-5`.
  - Scalar eval, 16 paired seeds / 32 games:
    - `policy_000016 vs B1`: `16/32`;
    - `policy_000016 vs B3`: `28/32`;
    - `policy_000018 vs B1`: `16/32`;
    - `policy_000018 vs B3`: `23/32`;
    - both vs B4: `32/32`.
  - Verdict: can match B1 at u680, but later updates drift down against B3.
- `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_v2_u670_to_u700_s3_20260425`
  - Config: frozen B1 top-action BC `0.5`, real LR `5e-6` after optimizer-resume LR fix.
  - Runtime proof:
    - final `reference_policy_top_action_bc_loss=0.235`, `reference_policy_top_action_bc_coef=0.5`;
    - final `collector_policy_train_model_rows=8596`;
    - final `collector_actor_policy_forward_ms=33694`;
    - final actor env steps/sec `3,782`;
    - final learner throughput `79,348 samples/sec`;
    - `collector_policy_train_heuristic_rows=0` and model-policy actor path is live.
  - Scalar eval, 16 paired seeds / 32 games:
    - `policy_000016 vs B1`: `16/32`;
    - `policy_000016 vs B3`: `28/32`;
    - `policy_000018 vs B1`: `16/32`;
    - `policy_000018 vs B3`: `30/32`;
    - both vs B4: `32/32`.
  - Confirmatory scalar eval, 32 paired seeds / 64 games, `policy_000018`:
    - `policy_000018 vs B1`: `32/64`, mean `0.500`;
    - `policy_000018 vs B3`: `58/64`, mean `0.906`;
    - `policy_000018 vs B4`: `64/64`, mean `1.000`;
    - B1 vs B3 on same confirm surface: `57/64`, mean `0.891`;
    - B1 vs B4: `64/64`, mean `1.000`;
    - `0` truncations in all listed matchups.
  - Verdict: current best reduced main-model candidate. It preserves the B1 anchor on this surface and slightly exceeds B1 vs B3 in the 64-game confirmation. Treat as promising, not thesis-final.

### Current thesis-model stance

- The old failure mode, "league worse than B1", is now addressed on the reduced model-policy surface by a frozen-B1 reference constraint plus a real low-LR continuation.
- Local throughput:
  - Native heuristic-row B1/league paths remain much faster locally.
  - Model-policy actor path is locally around `3.8k actor env steps/sec` because actor model forward dominates.
  - This is acceptable for the reduced correctness box; server translation should be judged on L40 smoke/throughput, not Windows timings.
- Next server-facing move:
  1. Use the strong B1 anchor as required import.
  2. Run a 1-2 update smoke of the refB1 strong low-LR main-model preset on L40 with DDP/autoscale.
  3. Verify `reference_policy_top_action_bc_loss` is nonzero, `distributed.world_size` is correct, model-policy rows are live, no rank artifact collisions, and optimizer LR is the configured continuation LR after resume.
  4. Then extend modestly and keep scalar eval/promotion gated against B1/B3 before any long run.

## 2026-04-26 reduced main-league continuation and server translation

### Continued reduced refB1-strong low-LR curve

- Continued from `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_v2_u670_to_u700_s3_20260425/training/checkpoints/checkpoint_700.pt`.
- Run:
  - `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_continue_u700_to_u760_s3_20260426`
  - preset: `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1strong_lowlr_localpromo.yaml`
  - completed updates `701-760`.
- Runtime proof:
  - final `reference_policy_top_action_bc_loss=0.276`, coef `0.5`;
  - final `collector_policy_train_model_rows=7863`;
  - final `collector_policy_train_heuristic_rows=0`;
  - final actor env steps/sec `3772`, cumulative `3560`;
  - final learner throughput `45,115 samples/sec`;
  - promotion gate passed at u760 for `policy_000021`.
- Warning:
  - late V-trace p99 was very large (`~167k`) while clip rate stayed around `0.083`.
  - Treat as a diagnostic warning, not an immediate quality failure; the clipped fraction and scalar eval stayed strong.
- Scalar eval, 16 paired seeds / 32 games:
  - `policy_000019` u720 vs B1/B2/B3/B4: `16/32`, `32/32`, `30/32`, `32/32`;
  - `policy_000020` u740 vs B1/B2/B3/B4: `16/32`, `32/32`, `30/32`, `32/32`;
  - `policy_000021` u760 vs B1/B2/B3/B4: `16/32`, `32/32`, `31/32`, `32/32`;
  - B1 vs B3 on same eval: `28/32`.
- Confirmatory scalar eval for `policy_000021`, 32 paired seeds / 64 games:
  - `policy_000021 vs B1`: `32/64`, mean `0.500`;
  - `policy_000021 vs B2`: `64/64`, mean `1.000`;
  - `policy_000021 vs B3`: `61/64`, mean `0.953`;
  - `policy_000021 vs B4`: `64/64`, mean `1.000`;
  - B1 vs B3 on same eval: `57/64`, mean `0.891`;
  - `0` truncations.
- Verdict:
  - This is now an actual upward reduced main-model curve through u760.
  - It preserves B1 directly and improves materially over B1 against B3.

### Further continuation shows why eval-based selection is required

- Continued from `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_continue_u700_to_u760_s3_20260426/training/checkpoints/checkpoint_760.pt`.
- Run:
  - `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_continue_u760_to_u820_s3_20260426`
  - completed updates `761-820`;
  - promotion gate passed at u820 for `policy_000024`.
- Runtime proof:
  - final `reference_policy_top_action_bc_loss=0.289`, coef `0.5`;
  - final `collector_policy_train_model_rows=7606`;
  - final `collector_policy_train_heuristic_rows=0`;
  - final actor env steps/sec `3767`, cumulative `3552`;
  - final learner throughput `48,563 samples/sec`.
- Scalar eval, 16 paired seeds / 32 games:
  - `policy_000022` u780 vs B1/B2/B3/B4: `16/32`, `32/32`, `32/32`, `32/32`;
  - `policy_000023` u800 vs B1/B2/B3/B4: `16/32`, `32/32`, `31/32`, `32/32`;
  - `policy_000024` u820 vs B1/B2/B3/B4: `16/32`, `32/32`, `29/32`, `32/32`;
  - B1 vs B3 on same eval: `28/32`.
- Confirmatory scalar eval for `policy_000022`, 32 paired seeds / 64 games:
  - `policy_000022 vs B1`: `32/64`, mean `0.500`;
  - `policy_000022 vs B2`: `64/64`, mean `1.000`;
  - `policy_000022 vs B3`: `64/64`, mean `1.000`;
  - `policy_000022 vs B4`: `64/64`, mean `1.000`;
  - B1 vs B3 on same eval: `57/64`, mean `0.891`;
  - `0` truncations.
- Important selection bug/lesson:
  - `training/checkpoints/checkpoint_tracker.json` selected u820 as best by `training_loss`.
  - Scalar eval says u780 is the best reduced snapshot.
  - Therefore do not trust training loss for final model selection in this phase.
  - Use scalar periodic dev-eval/checkpoint guard or post-run scalar final-eval selection.

### Server-facing config translation

- Added full-size server-facing preset:
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_refb1strong_lowlr.yaml`
  - It translates the reduced winner to full model size:
    - model-policy actor backend;
    - no heuristic actor training rows;
    - frozen imported B1 top-action reference coef `0.5`;
    - LR `5e-6`;
    - scalar periodic dev-eval stays enabled from the server preset (`interval=20`, `paired_seeds=32`).
- Added reduced eval-guard preset:
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1strong_lowlr_evalguard_localpromo.yaml`
  - Same reduced learning recipe, but with scalar periodic dev-eval enabled (`interval=20`, `paired_seeds=16`) so checkpoint selection is eval-based.
- Wrapper aliases added in `python/scripts/thesis_run.py`.
- Tests:
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_fullsize_b1_and_b1anchored_league_presets_to_main_eval_surface python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_b1anchored_benchmark_league_to_benchmark_eval_surface --tb=short`
  - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state python/weiss_rl/tests/test_vtrace.py::test_impala_learner_logging_persists_returned_loss_metrics --tb=short`
- Autoscale dry-runs:
  - Full-size `uc1-l40-4` dry-run resolves to `32 x 64 = 2048` envs, `4` learner GPUs, DDP.
  - Reduced local eval-guard dry-run resolves to `8 x 64 = 512` envs, single CUDA learner.

### Current best reduced candidate

- Best reduced main-model snapshot: `policy_000022` from
  - `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_continue_u760_to_u820_s3_20260426`
  - update `780`.
- It is better than the B1 anchor on the current B3 surface while preserving B1 directly.
- Next action:
  - Run a real 1-2 update L40 smoke of the full-size refB1 strong low-LR preset after importing the full-size B1 anchor.
  - Verify:
    - `distributed.world_size=4` on `uc1-l40-4` or `3` on `uc1-l40-3`;
    - nonzero `reference_policy_top_action_bc_loss`;
    - `collector_policy_train_model_rows > 0`;
    - `collector_policy_train_heuristic_rows = 0`;
    - configured LR `5e-6` survives checkpoint resume;
    - scalar periodic dev-eval writes summaries and can select best by eval, not training loss.

### League continuity and server-scaling fixes

- Motivation:
  - New-run checkpoint continuations did not automatically carry the old league registry.
  - In-place `--resume-run-dir` preserved the champion/recent pool, but `--resume-from runs/.../checkpoint_N.pt` created a fresh registry unless `--seed-snapshot-run-dir` was manually supplied.
  - This is dangerous for the server story: a long L40 continuation could silently train against a thin or empty league pool.
- Code changes:
  - `python/scripts/train.py`
    - Added auto-inference of `seed_snapshot_run_dir` from direct checkpoint paths under `runs/<source>/training/checkpoints/...` when creating a new league run.
    - Added a DDP barrier after rank-0 imports seeded snapshots, so nonzero ranks cannot build runtimes against a stale/empty registry.
    - Added `max_update` filtering to seeded pool import. When resuming from u780, only source snapshots with `snapshot.update <= 780` are imported. This avoids future-opponent leakage from the same source run, e.g. u800/u820 snapshots.
    - Initialises resumed runtimes and process collectors with `initial_learner_update=learner.update_count`.
  - `python/weiss_rl/runtime.py`
    - `league_effective_update` now reports the same reference update used by league schedules; raw `_effective_learner_update` is retained as `league_raw_effective_update`.
    - Process collectors receive the resumed learner update at construction, so the first collected batch does not look like version-0 data.
  - `python/scripts/thesis_run.py`
    - Added first-class `--seed-snapshot-run-dir` passthrough and summary recording.
- Regression tests:
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_and_champions python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation python/weiss_rl/tests/test_snapshot_registry.py::test_infer_seed_snapshot_run_dir_from_direct_resume_checkpoint python/weiss_rl/tests/test_snapshot_registry.py::test_infer_seed_snapshot_run_dir_skips_in_place_resume python/weiss_rl/tests/test_snapshot_registry.py::test_run_minimal_training_barriers_after_seed_snapshot_import python/weiss_rl/tests/test_runtime.py::test_runtime_metrics_report_window_and_cumulative_env_step_rates python/weiss_rl/tests/test_runtime.py::test_runtime_metrics_fall_back_to_current_update_for_league_reference_update python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_passes_seed_snapshot_run_dir_to_train_only python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_fullsize_b1_and_b1anchored_league_presets_to_main_eval_surface python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_defaults_b1anchored_benchmark_league_to_benchmark_eval_surface --tb=short`
  - Result: `11 passed`.
- Smoke runs:
  - Failed guard probe:
    - `runs/league_seedpool_autoinfer_u780_smoke_20260426`
    - Command intentionally resumed `checkpoint_780.pt` with `--max-updates 780`.
    - It imported the seeded pool before stopping, then correctly failed with `Resume checkpoint is already at update 780`.
  - Completed auto-seed and startup-version smoke:
    - `runs/league_seedpool_autoinfer_u780_to_u781_startver_smoke_20260426`
    - Resumed from `runs/b1anchored_league_benchmark_selfplay_refb1strong_lowlr_continue_u760_to_u820_s3_20260426/training/checkpoints/checkpoint_780.pt`.
    - Did not pass `--seed-snapshot-run-dir`; run summary recorded `seed_snapshot_run_dir_auto_inferred: true`.
    - Registry contains:
      - `b1_noleague_baseline`;
      - `seed_412cebca32_policy_000022`, update `780`;
      - no u800/u820 future snapshots.
    - Final smoke metrics:
      - `update_count=781`;
      - `loss=1.523804`;
      - `entropy=0.652444`;
      - `reference_policy_top_action_bc_loss=0.430135`, coef `0.5`;
      - `collector_policy_train_model_rows=7589`;
      - `collector_policy_train_heuristic_rows=0`;
      - `league_effective_update=780`;
      - `league_raw_effective_update=780`;
      - `league_update_lag=0`;
      - `policy_version_lag_p50=0`, `policy_version_lag_p90=0`;
      - `pfsp_pool_size=1`, `pfsp_recent_pool_size=1`;
      - local actor env steps/sec about `804`; one-update local model-actor smoke only, not a server throughput claim.
- Server topology dry-run:
  - `league_refb1strong_lowlr_uc1_l40_4_seedpool_dryrun_20260426`
  - `uc1-l40-4` resolves to:
    - `actor_count=32`;
    - `envs_per_actor=64`;
    - `total_envs=2048`;
    - `learner_gpu_count=4`;
    - `resolved_learner_parallelism=ddp`;
    - `queue_capacity_unrolls=256`.
- Verdict:
  - Keep these fixes. They are quality/scaling fixes, not local throughput polish.
  - They make league continuations more faithful and safer on multi-GPU Linux: the opponent pool is seeded automatically, DDP ranks wait for it, resumed schedules start at the correct update, and the first collected batch has clean policy-version metadata.
- Remaining risks / next hypotheses:
  - One-update smoke did not exercise actual PFSP sampling much (`pfsp_sampled_envs=0`) because it is too short. A longer 20-update evalguard continuation should verify champion/recent/hard-negative sampling proportions.
  - `policy_000022` from u780 remains the best reduced snapshot; use evalguard/full server runs for selection because training loss can still choose the wrong checkpoint.
  - Next serious run should be a 1-2 update L40 DDP smoke of the full-size preset, then a short 20-update evalguard server run if artifacts are clean.

### League PFSP active-weight and evalguard selection fixes

- Motivation:
  - The first evalguard league continuation showed a strong scalar surface but revealed two structural issues:
    - periodic dev-eval mode could still write `best` from `training_loss` before scalar dev-eval had run;
    - PFSP sampling weights reserved mass for inactive champion / hard-negative / no-league buckets, starving the active recent self-play pool.
  - This is a main-thesis scaling problem, not a local speed tweak. On the L40 server, an inactive bucket should not silently turn a league into mostly heuristic-public play.
- Code changes:
  - `python/scripts/train.py`
    - `_checkpoint_candidate_metric` no longer falls back to `training_loss` whenever periodic dev-eval is enabled.
    - Both checkpoint alias publish paths now require a non-null candidate metric before writing `best.pt`.
    - Dev-eval confidence eligibility now treats exact deterministic ties as neutral if there is no probability of being below half. This makes an 8/16 B1 or history tie eligible, while still rejecting real below-half anchors.
  - `python/weiss_rl/runtime.py`
    - `_opponent_sampling_groups` now computes recent/self-play weight from active groups only. If champion, hard-negative, or no-league groups are unavailable, their reserved mass flows back to recent snapshots instead of disappearing.
    - Added PFSP group-weight metrics earlier in this section remain useful for proving the runtime path.
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1soft_historypressure_lowlr_evalguard_localpromo.yaml`
    - New diagnostic preset: softer frozen-B1 top-action BC (`0.1`) and more champion/recent pressure.
- Regression tests:
  - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_opponent_sampling_weights_reassign_inactive_league_mass_to_recent python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_hard_negative_bucket python/weiss_rl/tests/test_runtime.py::test_runtime_metrics_report_window_and_cumulative_env_step_rates python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_wait_for_dev_eval_metric_when_periodic_eval_enabled python/weiss_rl/tests/test_snapshot_registry.py::test_publish_best_from_dev_eval_skips_null_candidate_without_existing_best python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_reasons_identify_borderline_confidence_only python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_allows_exact_tied_anchor_without_loss_probability python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_rejects_anchor_with_loss_probability python/weiss_rl/tests/test_thesis_run_wrapper.py::test_thesis_run_wrapper_passes_seed_snapshot_run_dir_to_train_only python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
  - Result: `11 passed`.
- Important runs:
  - `runs/league_evalguard_aliasgate_u780_to_u781_smoke_20260426`
    - Verified periodic-eval checkpoint alias gating.
    - `latest` pointed to u781 with null metric; `best` stayed `null`; no fake `best.pt`.
    - Promotion still passed at u781.
  - `runs/league_pfsp_activeweight_u780_to_u781_smoke_20260426`
    - Same one-update surface after PFSP active-weight fix.
    - Final PFSP group weights:
      - heuristic public `0.25`;
      - heuristic variants `0.25`;
      - recent self-play `0.50`;
      - champion / hard-negative / no-league `0.0`.
    - Promotion still passed; `policy_version_lag_p50=0`, `policy_version_lag_p90=0`.
  - `runs/league_pfsp_activeweight_u780_to_u800_evalguard_20260426`
    - 20-update scalar evalguard continuation from u780.
    - Aggregate scalar dev-eval: `0.833333`.
    - Anchor scores:
      - B0 RandomLegal `16/16`;
      - B1 NoLeague baseline `8/16`;
      - B2 HeuristicPublic `16/16`;
      - B3 HeuristicPublicAggro `16/16`;
      - B4 HeuristicPublicControl `16/16`;
      - Previous recent snapshot `8/16`.
    - Before the confidence eligibility fix this was incorrectly ineligible due exact ties having `prob_gt_half=0`.
  - `runs/league_activeweight_eligfix_u780_to_u800_evalguard_20260426`
    - Clean rerun after the evalguard eligibility fix.
    - `checkpoint_tracker.json` now has:
      - `latest.metric_kind=dev_eval_mean`, `latest.metric_value=0.833333`;
      - `best.metric_kind=dev_eval_mean`, `best.metric_value=0.833333`;
      - `best.pt` exists and points to `checkpoint_800.pt`.
    - Final PFSP group weights:
      - heuristic public `0.25`;
      - heuristic variants `0.25`;
      - recent self-play `0.50`.
    - Local model-policy league throughput:
      - mean actor env steps/sec about `3652`;
      - max about `4299`;
      - final about `3763`.
      - Local-only correctness/regression number, not an L40 claim.
  - `runs/league_pfsp_activeweight_u800_to_u820_evalguard_20260426`
    - Continued from the clean u800-style checkpoint with champion/recent active.
    - PFSP group weights:
      - heuristic public `0.25`;
      - heuristic variants `0.25`;
      - champion `0.25`;
      - recent `0.25`.
    - Promotion gate passed, but scalar dev-eval dropped to `0.758929`.
    - Anchor scores:
      - B0 `16/16`;
      - B1 `7/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `7/16`;
      - Previous recent `7/16`.
    - The patched eligibility correctly treats this as ineligible: real below-half anchors have `prob_lt_half=1.0`.
  - `runs/league_historypressure_refb1soft_u800_to_u820_evalguard_20260426`
    - Softer B1 imitation / stronger history pressure diagnostic.
    - PFSP group weights:
      - heuristic public `0.15`;
      - heuristic variants `0.15`;
      - champion `0.35`;
      - recent `0.35`.
    - Scalar dev-eval was identical to the balanced u800->u820 run: aggregate `0.758929`, B1/history `7/16`, B2/B3/B4 `16/16`.
    - Verdict: this exact 20-update continuation did not beat the plateau; do not keep tuning only these weights.
- Current reduced main-model candidate:
  - `runs/league_activeweight_eligfix_u780_to_u800_evalguard_20260426/training/checkpoints/best.pt`
  - update `800`, policy version `23`, `dev_eval_mean=0.833333`.
  - This supersedes the earlier u780 reduced candidate as the clean eval-selected artifact on the local reduced benchmark surface.
- Interpretation:
  - The league is now materially healthier:
    - seeded pool continuity works;
    - PFSP recent/champion exposure is live and visible;
    - evalguard selects by scalar dev-eval instead of loss;
    - exact ties versus B1/history no longer poison checkpoint selection.
  - But u800->u820 did not improve. The model remains excellent versus heuristic B2/B3/B4 but does not yet reliably beat B1 or its own promoted history.
- Next hypotheses:
  - Do not promote u820-style continuations as best. They are useful diagnostics showing the history plateau.
  - The next structural experiment should target policy improvement beyond B1/history, not just opponent weights:
    - stronger self-play objective or longer horizon from u800 with rollback to u800 best;
    - promotion gate/history selection that refuses successors with real below-half probability versus previous champion/recent;
    - larger server paired-seed promotion/eval to reduce local 8-seed noise;
    - investigate whether frozen B1 top-action BC, public heuristic logit bias, or teacher auxiliary losses are preventing improvement beyond the anchor.

### League guardrails for safe continued evolution

- Motivation:
  - After u800 became the clean eval-selected best, u800->u820 continuations still passed the quick promotion gate but then underperformed on scalar dev-eval.
  - A long server run must not let those weak successors enter the champion/recent pool, and rollbacks must not reuse policy ids or overwrite artifacts.
- Code changes:
  - `python/scripts/train.py`
    - New-run direct resume now seeds `checkpoint_tracker.best` from the source run only when the resumed checkpoint was that source run's scalar `dev_eval_mean` best and the checkpoint config hash matches the current run.
    - Checkpoint guard rollback now rejects snapshots newer than the restored best update, not just demotes them from champion status. This keeps rolled-back candidates out of the promotion-gated recent reservoir.
    - Mid-training rollback now restores weights without rewinding learner update / policy-version counters. This prevents repeated attempts from reusing `policy_000024` and overwriting the same `update_820` artifacts. Final selection can still restore the actual best checkpoint counters at run end.
  - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_unbiased_explore_lowlr_evalguard_localpromo.yaml`
    - Added diagnostic preset with no frozen-B1 BC, lighter teacher aux, higher entropy, and heavier champion/recent pressure. The public heuristic model-bias removal was not kept because it violated the imported B1 baseline model-config contract.
- Regression tests:
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_seed_checkpoint_tracker_from_resume_best_carries_dev_eval_best python/weiss_rl/tests/test_snapshot_registry.py::test_seed_checkpoint_tracker_from_resume_best_skips_config_mismatch python/weiss_rl/tests/test_snapshot_registry.py::test_reject_registry_snapshots_newer_than_marks_newer_refs python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_guard_rollback_restores_weights_without_rewinding_counters python/weiss_rl/tests/test_snapshot_registry.py::test_finalize_from_best_checkpoint_rewrites_latest_alias python/weiss_rl/tests/test_runtime.py::test_opponent_sampling_weights_reassign_inactive_league_mass_to_recent python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_allows_exact_tied_anchor_without_loss_probability python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_rejects_anchor_with_loss_probability --tb=short`
  - Result: `8 passed`.
- Important runs:
  - `runs/league_bestseed_reject_u800_to_u820_evalguard_20260426`
    - Resumed from `runs/league_activeweight_eligfix_u780_to_u800_evalguard_20260426/training/checkpoints/best.pt`.
    - Seeded best alias from the source eval-selected u800 best.
    - u820 quick promotion passed, but scalar dev-eval was `0.785714` vs seeded best `0.833333`.
    - Checkpoint guard rolled back to u800 and rejected `policy_000024`.
    - Registry after run:
      - champion: seeded u800 (`seed_04cb049b98_policy_000023`);
      - rejected: `policy_000024`.
  - `runs/league_explore_nobc_u800_to_u820_evalguard_20260426`
    - Diagnostic with no frozen-B1 BC, lighter teacher aux, higher entropy, 80% champion/recent pressure.
    - Did not improve: scalar dev-eval `0.776786`.
    - Anchor scores:
      - B0 `16/16`;
      - B1 `7/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Verdict: killing B1/teacher pressure hurts B1 stability and does not solve history improvement on this 20-update surface.
  - `runs/league_bestseed_reject_multiattempt_u800_to_u840_evalguard_20260426`
    - Same-config guarded multi-attempt run.
    - Two u820 attempts were made and both rolled back to u800:
      - attempt 1 scalar dev-eval `0.785714`, rollback reason `score_drop`;
      - attempt 2 scalar dev-eval `0.776786`, rollback reasons `score_drop,confidence`.
    - This exposed policy-id/artifact reuse after rollback, which was fixed afterward.
  - `runs/league_bestseed_counterfix_u800_to_u820_evalguard_20260426`
    - Verification after monotonic rollback-counter fix.
    - Mid-training rollback event:
      - `update_count=820`;
      - `restored_weight_update_count=800`;
      - `policy_version=24`;
      - `rejected_snapshots=["policy_000024"]`.
    - Final selection then restored `latest` to u800:
      - `latest.metric_kind=dev_eval_mean`;
      - `latest.metric_value=0.833333`;
      - `latest.update_count=800`.
    - Local throughput stayed in the same model-policy league band:
      - mean actor env steps/sec about `3648`;
      - max about `4287`;
      - final about `3755`.
- Current verdict:
  - League safety is now much better. A weak successor can pass the quick promotion gate, but scalar dev-eval + checkpoint guard rejects it and prevents it from contaminating champion/recent sampling.
  - The local reduced league still has not found a better-than-u800 successor. Repeated attempts and a no-BC/high-exploration diagnostic both failed.
  - This is now a credible thesis engineering story: the league machinery no longer looks worse than B1 because it is allowed to drift; it preserves the best eval-selected anchor while testing successors.
- Next hypotheses:
  - Run the guarded loop on the L40 server with larger promotion/dev-eval seed surfaces before concluding the plateau is real. The local 8-pair surface is useful for killing bad ideas, not for final promotion claims.
  - For a genuine big increase, test structural objective changes that preserve B1 while improving history:
    - keep frozen-B1 BC but add an explicit anti-history/self-play improvement auxiliary or policy KL schedule;
    - use longer same-config guarded server continuations from u800;
    - test a model-bias fade only if the B1 import/config contract is intentionally reworked, not by weakening validation ad hoc;
    - inspect game/action differences for B1/history losses, because all failed variants still crush B2/B3/B4 and mainly fail to exceed B1/history.

### League continuation: B1 disagreement, per-snapshot guidance, and balanced B1/recent pressure

- Motivation:
  - The u800 league checkpoint is still the reduced-surface best, but u800->u820 successors kept falling to `7/16` versus the promoted B1 anchor or failing to improve versus previous history.
  - The next question was whether this is a B1 action-family pathology, a public-heuristic-bias basin, or league sampling pressure disappearing after the B1 warmup window.
- B1 disagreement audits:
  - `runs/league_b1_disagreement_audit_u800_allowmismatch_20260426`
    - Compared u800 best `policy_000023` versus B1 on B1 eval episodes.
    - `compared_steps=2132`, `inspected_step_count=320`, `mean_total_variation=0.289406`.
    - Dominant disagreement: candidate `main_move` versus B1 `main_play_character`, `265/320` inspected states.
  - `runs/league_b1_disagreement_audit_u820_allowmismatch_20260426`
    - Plain u820 continuation.
    - `compared_steps=2153`, `inspected_step_count=320`, `mean_total_variation=0.293473`.
    - Dominant disagreement remained `main_move` versus `main_play_character`, `263/320`.
  - `runs/league_b1_disagreement_audit_refb1family_u820_allowmismatch_20260426`
    - Family-BC u820 candidate.
    - `compared_steps=2135`, `inspected_step_count=320`, `mean_total_variation=0.322221`.
    - Dominant disagreement worsened to `282/320` candidate `main_move` versus B1 `main_play_character`.
    - Verdict: exact/family imitation losses are live, but family BC alone does not overcome the move-heavy basin.
- Code changes:
  - `python/weiss_rl/config/models.py`, `python/weiss_rl/config/parse.py`, `python/weiss_rl/learners/impala_learner.py`, `python/scripts/train.py`
    - Added optional `training.reference_policy_top_action_family_bc_coef`.
    - Implemented learner-side family-level frozen-reference BC for dense and packed structured legal views.
    - Fixed the packed-path non-finite case by masking non-finite target/current log-probs before reduction.
  - `python/weiss_rl/runtime.py`
    - Fixed model-policy opponent snapshot loading to restore per-snapshot public-heuristic guidance payloads, matching eval snapshot loading.
    - This matters for scalable league experiments where focal/current policies and imported opponents can intentionally use different guidance scales.
  - `python/scripts/train.py`
    - B1 baseline and seed-snapshot import contracts now allow model-section differences only for payload-restored guidance scales:
      - `public_heuristic_logit_bias_scale`;
      - `public_heuristic_actor_logit_bias_scale`.
    - Architecture/environment mismatches still fail hard.
  - New diagnostic presets:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1family_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_actorbiasonly_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_b1persist_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_balanced_b1recent_lowlr_evalguard_localpromo.yaml`
- Regression tests:
  - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_family_bc_coef_adds_reference_family_nll python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
    - Result: `3 passed`.
  - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_runtime_snapshot_opponent_load_restores_model_guidance_payload python/weiss_rl/tests/test_snapshot_registry.py::test_imported_b1_contract_allows_snapshot_guidance_scale_mismatch python/weiss_rl/tests/test_snapshot_registry.py::test_imported_seed_snapshot_contract_allows_snapshot_guidance_scale_mismatch --tb=short`
    - Result: `3 passed`.
- Important runs:
  - `runs/league_refb1family_u800_to_u801_smoke_v3_20260426`
    - One-update packed-path smoke from u800.
    - Family loss was live and finite:
      - `reference_policy_top_action_bc_loss=0.405642`;
      - `reference_policy_top_action_family_bc_loss=2.429362`.
  - `runs/league_refb1family_u800_to_u820_evalguard_20260426`
    - u800->u820 with family-level B1 BC coefficient `0.75`.
    - Dev eval aggregate `0.776786`.
    - Anchors:
      - B0 `16/16`;
      - B1 `7/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Final metrics included `reference_policy_top_action_family_bc_loss=4.126007`, `actor_env_steps_per_sec=3795`.
    - Verdict: live family BC did not fix B1/history plateau and worsened B1 disagreement TV.
  - `runs/league_refb1_actorbiasonly_u800_to_u820_evalguard_v3_20260426`
    - Actor-only public heuristic bias diagnostic:
      - current/focal learner/eval bias scale `0.0`;
      - actor bias scale `1.0`;
      - imported B1/seeded opponents preserve payload scale `3.0`.
    - Promotion failed and dev eval aggregate was `0.0` across all anchors.
    - Final metrics:
      - `public_heuristic_logit_bias_scale_active=0`;
      - `reference_policy_top_action_bc_loss=6.435271`;
      - `actor_env_steps_per_sec=3794`.
    - Verdict: abrupt removal of learner/eval public bias from the u800 checkpoint destroys the evaluated policy. Any bias fade must be gradual and intentionally re-anchored, not a direct continuation flip.
  - `runs/league_refb1_b1persist_u800_to_u820_evalguard_20260426`
    - Persistent B1 opponent sampling after update 400:
      - `pfsp_sampling_weight_noleague_baseline=0.25`;
      - `pfsp_sampling_weight_recent=0.0`;
      - `pfsp_sampling_weight_champion=0.25`;
      - heuristic public and variant `0.25` each.
    - Dev eval aggregate `0.785714` on seven-anchor surface.
    - Anchors:
      - B0 `16/16`;
      - B1 `8/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Same six-anchor comparison to u800 is effectively tied; it fixes the u820 B1 regression but does not beat u800.
  - `runs/league_refb1_b1persist_u820_to_u860_evalguard_20260426`
    - Extended persistent-B1 run from the u820 candidate.
    - u840 quick dev eval `0.776786`; 32-pair confirmatory aggregate `0.783482`; B3 dropped to `15/16`.
    - u860 quick dev eval `0.767857`; 32-pair confirmatory aggregate `0.774554`; B3 dropped to `14/16`.
    - Checkpoint guard rolled final selection back to u820 best.
    - Verdict: more local updates with too much B1/champion/heuristic pressure do not break the plateau and begin eroding B3.
  - `runs/league_refb1_balanced_b1recent_u800_to_u820_evalguard_20260426`
    - Balanced late league sampling:
      - B1 `0.20`;
      - recent `0.20`;
      - champion `0.20`;
      - heuristic public `0.20`;
      - heuristic public variant `0.20`.
    - Dev eval aggregate `0.785714` on seven-anchor surface.
    - Anchors:
      - B0 `16/16`;
      - B1 `8/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Local actor env steps/sec stayed in the same model-policy league band, about `3786` final on this run.
- Current verdict:
  - Best reduced league setting from this iteration is the balanced B1/recent sampler, not family BC or actor-only bias removal.
  - It is not a big quality jump. It ties the u800 plateau while making the league pressure more structurally sound:
    - B1 remains live after warmup;
    - recent self-play remains live;
    - snapshot guidance is restored per policy in both training runtime and eval;
    - weak continuations are still rejected by checkpoint guard.
  - The local plateau appears real on this reduced 8-pair surface: B1/history stay at `8/16`, while heuristic anchors are already saturated.
- Next hypotheses:
  - Do not spend more local time on tiny sampler weights. The next quality experiment should change the objective or evaluation pressure:
    - larger server run with balanced sampler and larger paired-seed surfaces to see whether local 8-pair quantization is hiding small improvements;
    - promotion/checkpoint metric that gives explicit extra weight to B1/history improvement once B2/B3/B4 are saturated;
    - gradual public-bias fade from `3.0` rather than abrupt actor-only flip;
    - inspect losing B1/history episodes for terminal/value-target differences, because action-family audits show a persistent play-vs-move mismatch but imitation losses alone did not fix it.

### True league lane fix: sampler was live in metrics but not in actor assignment

- Motivation:
  - The previous balanced/frontier sampler experiments kept tying the plateau suspiciously exactly.
  - Per-update PFSP counters showed nonzero sampling weights but `pfsp_noleague_baseline_envs=0`, `pfsp_recent_envs=0`, and `pfsp_champion_envs=0` through the whole u800->u820 runs.
  - Root cause: the self-play league presets inherited `training.diverse_opponent_actor_count: 0`, so actors used the fallback heuristic-public opponent path instead of the league opponent sampler. The first attempt to set `-1` exposed a second bug: actor construction still used the raw `actor_id < count` comparison, so `-1` made every actor non-diverse.
- Code changes:
  - `python/weiss_rl/config/parse.py`
    - Allows `training.diverse_opponent_actor_count: -1` as an explicit "all actors" sentinel.
  - `python/weiss_rl/runtime.py`
    - Allows `-1` through runtime validation.
    - `_actor_id_is_diverse_lane()` now treats negative count as all actor ids.
    - `_build_actor_state()` now uses `_actor_id_is_diverse_lane()` instead of raw comparison.
    - Preserves `diverse_model_actor_count: 0` correctly when diverse opponent count is `-1`.
  - Presets:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_localpromo.yaml`
      - Sets `training.diverse_opponent_actor_count: -1`.
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_refb1strong_lowlr.yaml`
      - Sets `training.diverse_opponent_actor_count: -1`.
    - Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_lowlr_evalguard_localpromo.yaml`.
    - Added `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_family_lowlr_evalguard_localpromo.yaml`.
- Regression tests:
  - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_diverse_opponent_actor_count_minus_one_marks_all_actor_ids_diverse python/weiss_rl/tests/test_runtime.py::test_build_actor_state_uses_minus_one_diverse_lane_sentinel python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
    - Result: `3 passed`.
  - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_family_bc_coef_adds_reference_family_nll python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
    - Result: `2 passed`.
- Important runs:
  - `runs/league_refb1_frontier_laneall_u800_to_u801_smoke_20260426`
    - First `-1` smoke before fixing `_build_actor_state`.
    - Still fake-league: PFSP env counts all zero and all train rows remained heuristic-public shaped.
  - `runs/league_refb1_frontier_laneall_u800_to_u801_smoke_v2_20260426`
    - Smoke after fixing `_build_actor_state`.
    - Real league opponent assignment became visible by later updates in the full run; one-update smoke still had little reset exposure.
    - Local actor env steps/sec about `624` on the single update. This is local model-opponent cost, not a server claim.
  - `runs/league_refb1_frontier_laneall_u800_to_u820_evalguard_20260426`
    - True league frontier sampler without family BC.
    - Final PFSP env counts were live:
      - B1 `28`;
      - recent `34`;
      - champion `28`;
      - heuristic public `8`.
    - Dev eval aggregate `0.776786`.
    - Anchors:
      - B0 `16/16`;
      - B1 `7/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Local actor env steps/sec about `985` final.
    - Verdict: true league pressure without stronger B1 stabilizer immediately reintroduces the B1 regression.
  - `runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426`
    - True league frontier sampler plus B1 family BC `0.75`.
    - Final PFSP env counts:
      - B1 `28`;
      - recent `34`;
      - champion `28`;
      - heuristic public `8`.
    - Dev eval aggregate `0.785714`.
    - Anchors:
      - B0 `16/16`;
      - B1 `8/16`;
      - B2 `16/16`;
      - B3 `16/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Final metrics:
      - `reference_policy_top_action_bc_loss=0.292641`;
      - `reference_policy_top_action_family_bc_loss=4.905094`;
      - local `actor_env_steps_per_sec=981`.
    - Verdict: this is the best current true-league candidate. It ties the plateau on quality, but unlike prior ties, it is actually training against B1/history/champion opponents.
  - `runs/league_refb1_frontier_family_laneall_u820_to_u860_evalguard_20260426`
    - Extended the true-league family candidate from u820.
    - Command timed out after 40 minutes but wrote u860 artifacts; orphaned Python/uv process tree was stopped afterward.
    - u860 quick dev eval aggregate `0.767857`.
    - Anchors:
      - B0 `16/16`;
      - B1 `8/16`;
      - B2 `16/16`;
      - B3 `14/16`;
      - B4 `16/16`;
      - Previous champion `8/16`;
      - Previous recent `8/16`.
    - Checkpoint tracker still had best seeded from u820 at `0.785714`; latest remained u860 because the timeout interrupted final guard cleanup.
    - Verdict: continuing true-league family locally to u860 degraded B3 and did not improve B1/history.
- Current verdict:
  - The main league was not actually exercising model-policy league opponents in the reduced self-play presets. That is now fixed.
  - The best current **true-league** reduced checkpoint is:
    - `runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426/training/checkpoints/checkpoint_820.pt`
  - It does not beat the old local score plateau yet, but it is the first clean candidate where:
    - B1/history/champion opponent exposure is live;
    - B1 does not regress;
    - B2/B3/B4 remain saturated.
  - Local throughput drops sharply when the real model-opponent league path is enabled:
    - heuristic/fake-league local actor env steps/sec was around `3700-3800`;
    - true model-opponent league local actor env steps/sec is around `980`.
    - This is expected on one local GPU and should not be projected directly to L40 DDP, but it means local true-league loops are much slower.
- Next hypotheses:
  - Stop treating earlier u800->u820 sampler experiments as evidence about league sampling; they mostly trained against heuristic-public opponents.
  - Server smoke is now much more important: the true-league path is locally too slow and should scale materially better with multiple L40 GPUs/collectors.
  - Quality next steps:
    - run the true-league family frontier candidate on server with larger paired-seed eval and clean guard finalization;
    - try a lower LR or shorter checkpoint cadence after u820 to avoid B3 erosion;
    - consider explicit history/B1-weighted checkpoint scoring now that real B1/history data is present;
    - inspect B3 erosion in `u820->u860` before extending again locally.

### 2026-04-26 - league throughput/sampler audit after u820 plateau concern
- User concern:
  - League quality still looks too close to 50/50 versus history/recent snapshots.
  - League throughput is much lower than B1; verify this is not another bad flag like the earlier B1 native-rollout miss.
- Throughput flag audit:
  - Current true-league process run `runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426`:
    - startup: `collection_backend=process`, `use_process_collectors=true`, `use_central_batched_collection=false`;
    - final local `actor_env_steps_per_sec=980.64`;
    - final `collector_actor_policy_forward_ms=130442`;
    - true PFSP env counts live: B1 `28`, recent `34`, champion `28`, heuristic public `8`.
  - Earlier fake/mostly-heuristic league runs were around `3700-3800` local actor env steps/sec because PFSP model-opponent env counts were zero.
  - B1 native rollout remains a separate path:
    - `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke` around `142620` samples/sec locally;
    - league cannot inherit that because league model opponents require model inference.
- Central backend experiment:
  - One-update smoke:
    - `runs/league_refb1_frontier_family_laneall_central_u800_to_u801_smoke_20260426`;
    - only changed `system.collection_backend="central"`;
    - local `actor_env_steps_per_sec=7511.82`;
    - `collector_actor_policy_forward_ms=1320`;
    - but PFSP env counters were zero on first update because actors had not reset into fresh sampled assignments yet.
  - Sustained u820->u840 continuation:
    - `runs/league_refb1_frontier_family_auto_recentfix_u820_to_u840_evalguard_20260426`;
    - startup confirmed `collection_backend=auto`, `use_central_batched_collection=true`;
    - once true PFSP opponents became active, central fixed-opponent overwrite dominated at about `27-29s/update`;
    - tail local actor env steps/sec fell to about `537-574`;
    - dev eval aggregate `0.767857`;
    - anchors: B0 `16/16`, B1 `8/16`, B2 `16/16`, B3 `14/16`, B4 `16/16`, previous champion `8/16`, previous recent `8/16`.
  - Verdict:
    - the one-update central smoke was misleading;
    - sustained central collection is worse than process collection for true model-opponent league on the local one-GPU box;
    - do **not** switch league presets to central/auto based on the one-update smoke.
    - Process collectors are still the right local/server-scalable stance for model-opponent league unless a later server smoke proves otherwise.
- Recent-reservoir bug/fix:
  - Diagnosis:
    - In `runs/league_refb1_frontier_family_laneall_u820_to_u860_evalguard_20260426`, after promotions the registry had four champions and no rejected snapshots, but `pfsp_recent_pool_size=0`.
    - Cause: promotion-gated recent size was reduced to 2, `registry.latest_ids(2)` returned only newly promoted champions, and then the refresh path removed champion ids from recent.
    - Result: intended recent/frontier sampling weight silently collapsed to zero and was reallocated to other groups.
  - Code change:
    - `python/weiss_rl/runtime.py` now queries enough latest ids to backfill around champion/rejected/fixed exclusions.
    - If non-champion recent ids are insufficient, it backfills with the newest champion ids so the recent/frontier sampling mass stays live.
  - Test:
    - Added `test_refresh_opponent_pool_backfills_recent_when_latest_snapshots_are_champions`.
    - Passed:
      - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_small_recent_reservoir_when_champions_exist python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_backfills_recent_when_latest_snapshots_are_champions python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets --tb=short`
      - `3 passed`.
  - Process-path smoke:
    - `runs/league_recentfix_process_u860_to_u861_smoke_20260426`;
    - startup: `collection_backend=process`, `use_process_collectors=true`;
    - pool metrics: `pfsp_champion_pool_size=4`, `pfsp_recent_pool_size=2`, `pfsp_sampling_weight_recent=0.30`, `pfsp_candidate_model_count=5`;
    - first-update assignment counters remained zero because actors were still on pre-refresh active episodes, but the repaired pool contract is visible.
- Current verdict:
  - The league throughput drop is real on local Windows one-GPU when true model opponents are active; it is not just a bad B1-style native-rollout flag.
  - The process path is still preferable for the final multi-GPU Linux server because model-opponent inference can shard across collector processes/devices; local timings remain only correctness/regression evidence.
  - Quality did not improve yet: best true-league checkpoint remains u820 from `runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426`.
  - The recent-reservoir repair is important for continuing beyond u840/u860 because the old loop was losing explicit recent/frontier pressure exactly when promotion succeeded.
- Next hypotheses:
  - Re-run a short process continuation past u840/u860 with the recent-reservoir fix and scalar eval to see whether B3 erosion is reduced.
  - If quality still plateaus, add a checkpoint score/gate component that does not accept a candidate unless it preserves B1 and B3 while improving or at least not losing against recent/champion.
  - Longer-term: the league needs a stronger quality signal than raw 50/50 self-play versus recent; consider distinct roles for champion retention, frontier recent pressure, B1 anchor preservation, and hard-negative recovery.
### League B1-guidance fade and fresh-student tests - 2026-04-26

- Hypothesis:
  - The current true-league plateau may come from treating the B1 anchor as a permanent behavior ceiling.
  - Test two structural alternatives:
    1. continue the current best true-league u820 checkpoint while fading frozen-B1 exact/family BC;
    2. train a fresh league student where B1 is an early reference/opponent, then fades into self-play.
- Code/config changes:
  - Added schedulable frozen-reference BC fields:
    - `training.reference_policy_top_action_bc_final_coef`
    - `training.reference_policy_top_action_bc_start_updates`
    - `training.reference_policy_top_action_bc_end_updates`
    - `training.reference_policy_top_action_family_bc_final_coef`
    - `training.reference_policy_top_action_family_bc_start_updates`
    - `training.reference_policy_top_action_family_bc_end_updates`
  - `python/scripts/train.py` now applies these schedules each learner update and logs active coefficients through the learner metrics.
  - `python/weiss_rl/learners/impala_learner.py` added `set_reference_policy_bc_coefs()`.
  - Added reduced-size experiment presets:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_family_fade_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_earlyleague_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_slowfade_b3hold_lowlr_evalguard_localpromo.yaml`
- Validation:
  - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_guidance_anneal_overrides python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_current_thesis_facing_presets python/weiss_rl/tests/test_train_stall_monitor.py::test_reference_policy_bc_coefs_for_next_update_linearly_anneal python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_family_bc_coef_adds_reference_family_nll --tb=short`
    - Result: `5 passed`.
  - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/learners/impala_learner.py`
    - Result: passed.
- Runtime smoke:
  - `runs/league_refb1_fade_u820_to_u821_smoke_20260426`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_family_fade_lowlr_evalguard_localpromo.yaml --run-label league_refb1_fade_u820_to_u821_smoke_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426/training/checkpoints/checkpoint_820.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 821 --checkpoint-interval-updates 999 --profile-timers --override evaluation.periodic_dev_eval_interval_updates=0`
  - Proof:
    - process collectors stayed selected;
    - B1 baseline import and frozen reference attachment succeeded;
    - seeded snapshot pool imported `3` source snapshots;
    - active coefficients at u821: exact `0.494375`, family `0.740625`;
    - actor throughput on the one-update local smoke: `609 actor_env_steps/sec`.
- Continuation fade result:
  - Run:
    - `runs/league_refb1_fade_u820_to_u840_evalguard_20260426`
    - extended in place to u880.
  - Main command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_family_fade_lowlr_evalguard_localpromo.yaml --run-label league_refb1_fade_u820_to_u840_evalguard_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426/training/checkpoints/checkpoint_820.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/league_refb1_frontier_family_laneall_u800_to_u820_evalguard_20260426 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 840 --checkpoint-interval-updates 20 --profile-timers`
  - Extension:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_refb1_frontier_family_fade_lowlr_evalguard_localpromo.yaml --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-run-dir runs/league_refb1_fade_u820_to_u840_evalguard_20260426 --resume-from latest --resume-allow-config-mismatch --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 880 --checkpoint-interval-updates 20 --profile-timers`
  - Dev-eval curve:
    - u840 aggregate `0.776786`: B1 `0.5`, B3 `0.9375`, previous champion `0.5`, previous recent `0.5`.
    - u860 aggregate `0.758929`: B1 `0.5`, B3 `0.8125`, previous champion `0.5`, previous recent `0.5`.
    - u880 aggregate `0.776786`: B1 `0.5`, B3 `0.9375`, previous champion `0.5`, previous recent `0.5`.
  - Runtime:
    - u840 active exact/family BC: `0.3875` / `0.5625`.
    - u880 active exact/family BC: `0.1625` / `0.1875`.
    - true league PFSP remained live; u880 tail included B1/recent/champion sampled envs.
    - tail10 local `actor_env_steps/sec` at u840 was about `841`; tail10 by u880 was about `696`.
  - Verdict:
    - Fading the current u820 continuation did not break the plateau.
    - It remained stable but did not improve B1/history and temporarily degraded B3 at u860.
    - Do not promote this as an improvement over the existing true-league u820 candidate.
- Fresh-student result:
  - Run:
    - `runs/league_freshstudent_refb1_fade_u0_to_u60_evalguard_20260426`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_lowlr_evalguard_localpromo.yaml --run-label league_freshstudent_refb1_fade_u0_to_u60_evalguard_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 60 --checkpoint-interval-updates 20 --profile-timers`
  - Dev-eval curve:
    - u20 aggregate `0.71875`: B1 `0.1875`, B3 `0.6875`, previous recent `0.5625`.
    - u40 aggregate `0.729167`: B1 `0.25`, B3 `0.5625`, previous recent `0.6875`.
    - u60 aggregate `0.760417`: B1 `0.3125`, B3 `0.75`, previous recent `0.625`.
  - Runtime:
    - tail10 local `actor_env_steps/sec` about `1303`.
    - Promotion correctly skipped during league warmup (`threshold=200`), so this is a guided preleague curve, not yet a mature self-play league curve.
  - Verdict:
    - This is the only fresh-student evidence with a clean upward curve in this batch.
    - It is still weaker than the promoted B1 and the current u820 true-league continuation.
- Early-league fresh-student result:
  - Run:
    - `runs/league_freshstudent_refb1_fade_earlyleague_u60_to_u120_evalguard_20260426`
  - Change:
    - lowered local `league.warmup.first_updates` from `200` to `60`.
  - Dev-eval curve:
    - u80 aggregate `0.75`: B1 `0.4375`, B3 `0.5625`, previous recent `0.5`.
    - u100 aggregate `0.729167`: B1 `0.5`, B3 `0.5`, previous recent `0.5`.
    - u120 aggregate `0.739583`: B1 `0.375`, B3 `0.6875`, previous recent `0.4375`.
  - Promotion:
    - u80/u100 failed promotion via `anchor_loss_guardrail_exceeded`;
    - u120 passed aggregate promotion but did not improve the useful surface.
  - Verdict:
    - Entering true league at u60 is too early for this fresh student.
    - It makes PFSP live but destabilizes the useful B1/B3 frontier.
- Slow-fade B3-hold fresh result:
  - Run:
    - `runs/league_freshstudent_slowfade_b3hold_u60_to_u120_evalguard_20260426`
  - Change:
    - slower B1 exact/family reference fade to u320;
    - stronger aggressive public heuristic teacher/opponent pressure.
  - Dev-eval curve:
    - u80 aggregate `0.75`: B1 `0.4375`, B3 `0.5`, previous recent `0.5625`.
    - u100 aggregate `0.739583`: B1 `0.5`, B3 `0.5`, previous recent `0.5625`.
    - u120 aggregate `0.739583`: B1 `0.4375`, B3 `0.625`, previous recent `0.4375`.
  - Verdict:
    - Slower fade and extra B3 pressure did not beat the original fresh u60/u80 curve.
    - Do not keep this variant alive unless a later diagnostic shows a specific B3-collapse mechanism it fixes.
- Current interpretation:
  - Continuing the existing B1-initialized true-league checkpoint with less B1 imitation is not enough.
  - A fresh student can show an upward early curve, but it is not yet strong and early true-league pressure is harmful.
  - The league acceptance surface is still too aggregate-heavy: saturated B0/B2/B4 anchors can make a policy look passable while B1/B3/history are flat or worse.
  - Next best structural direction is not another small LR tweak. The next run should either:
    1. add a frontier-focused selection/promotion metric that weights B1, B3, previous champion, and previous recent much more heavily; or
    2. build a staged fresh-student curriculum: guided preleague to a B1/B3 minimum, then activate PFSP league, rather than fixed update-count warmup.

## 2026-04-26 - league frontier objective, seed-pool, and rollback diagnostics

- Code/preset changes:
  - Added `evaluation.periodic_dev_eval_anchor_weights` and weighted periodic dev-eval aggregation.
    - New persisted summaries now include `aggregate_score`, `unweighted_aggregate_score`, and `aggregate_weighting`.
    - Frontier-weighted local surface used:
      - B0 `0.25`
      - B1 NoLeague baseline `3.0`
      - B2 HeuristicPublic `0.25`
      - B3 HeuristicPublicAggro `3.0`
      - B4 HeuristicPublicControl `1.0`
      - Previous champion snapshot `2.0`
      - Previous recent snapshot `2.0`
  - Fixed explicit external seed-snapshot imports on resumed runs.
    - Before: resume update filtering could silently import zero B1 seed snapshots when resuming a league run from u60 and importing a B1 run whose snapshots are u150+.
    - After: auto-inferred same-run resume imports keep the max-update guard; explicit external seed pools import the full compatible pool.
  - Fixed checkpoint rollback semantics for imported seed pools.
    - Before: rollback to an older best checkpoint rejected every registry snapshot with `snapshot.update > best_update`, including `seed_*` external B1 snapshots and `b1_noleague_baseline`.
    - After: rollback rejection only marks current-run train snapshots (`policy_*`) without imported metadata; external seed pools and imported B1 anchors survive.
  - Relaxed confidence-only rollback.
    - Before: a low-seed confidence failure could roll back an equal or improving aggregate score.
    - After: confidence-only rollback requires `current_score < best_score`; equal/improving local dev-eval is not reset purely due to noisy confidence.
  - New local research presets:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_seedpool_lag10_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_seedpool_lag10_warmup200_lowlr_evalguard_localpromo.yaml`
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_hold_frontierweighted_seedpool_lag10_warmup200_lowlr_evalguard_localpromo.yaml`
- Validation:
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_seed_snapshot_import_max_update_only_limits_auto_inferred_resume_pool python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation python/weiss_rl/tests/test_train_stall_monitor.py::test_weighted_dev_eval_aggregate_prioritizes_frontier_anchors python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_guidance_anneal_overrides --tb=short`
    - `4 passed`
  - `uv run pytest -q python/weiss_rl/tests/test_train_stall_monitor.py::test_persist_periodic_dev_eval_summary_round_trips_anchor_payloads_and_b2_warnings python/weiss_rl/tests/test_snapshot_registry.py::test_seed_snapshot_import_max_update_only_limits_auto_inferred_resume_pool --tb=short`
    - `2 passed`
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_reject_registry_snapshots_newer_than_preserves_external_seed_pool python/weiss_rl/tests/test_snapshot_registry.py::test_reject_registry_snapshots_newer_than_marks_newer_refs python/weiss_rl/tests/test_train_stall_monitor.py::test_persist_periodic_dev_eval_summary_round_trips_anchor_payloads_and_b2_warnings --tb=short`
    - `3 passed`
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_guard_does_not_rollback_equal_score_on_confidence_only_noise python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_guard_does_not_rollback_improving_score_on_confidence_only_noise python/weiss_rl/tests/test_snapshot_registry.py::test_reject_registry_snapshots_newer_than_preserves_external_seed_pool --tb=short`
    - `3 passed`
  - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_snapshot_registry.py`
    - passed.
- Experiment A - frontier-weighted scoring plus explicit B1 seed pool:
  - Run:
    - `runs/league_freshstudent_frontierweighted_b1pool_u60_to_u100_evalguard_20260426`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_lowlr_evalguard_localpromo.yaml --run-label league_freshstudent_frontierweighted_b1pool_u60_to_u100_evalguard_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_freshstudent_refb1_fade_u0_to_u60_evalguard_20260426/training/checkpoints/checkpoint_60.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 100 --checkpoint-interval-updates 20 --profile-timers`
  - Startup proof:
    - imported B1 promotion anchor;
    - attached frozen B1 reference policy;
    - imported seeded snapshot pool count `7`.
  - Dev-eval:
    - u80 weighted aggregate `0.565789`: B1 `0.4375`, B3 `0.5625`, B4 `1.0`, previous recent `0.4375`.
    - u100 weighted aggregate `0.565789`: B1 `0.5`, B3 `0.5`, B4 `0.875`, previous recent `0.5`.
  - Runtime:
    - tail `actor_env_steps/sec` about `1294-1306`.
    - However `warmup_snapshot_mix_fraction_active=0.0`; imported B1 snapshots mostly sat in registry.
  - Verdict:
    - Import fix worked, but the config still did not route external B1 pool into games during warmup.
- Experiment B - low actor-policy lag plus active B1 seed-pool warmup:
  - Run:
    - `runs/league_freshstudent_frontierweighted_seedpool_lag10_u60_to_u140_evalguard_20260426`
  - Key changes:
    - `training.checkpointing.actor_reload_interval_updates: 10`
    - `league.sampling.warmup_snapshot_mix_fraction: 0.35`
    - `league.pool.recent_size: 32`
    - `league.pool.champion_size: 8`
    - `league.warmup.first_updates: 120`
  - Important runtime diagnosis:
    - With old reload interval `50`, continuation runs had `league_effective_update` stuck at the resume/reload update for long windows.
    - With lag10, update lag was typically `6-8` instead of `36+`.
    - Warmup pool was live: around u118-u120, `pfsp_pool_size=7`, `pfsp_warmup_snapshot_envs` about `147-176`.
  - Dev-eval:
    - u80 weighted aggregate `0.546053`.
    - u100 weighted aggregate `0.559211`.
    - u120 weighted aggregate `0.592105`: B1 `0.4375`, B3 `0.6875`, B4 `0.875`, previous recent `0.4375`.
  - Bad artifact side effect found:
    - At u120, checkpoint guard rolled back to u100 due confidence despite aggregate improvement, and rejected all imported B1 seed snapshots because their source updates were newer than best update.
    - This caused the seed-pool rollback fix above.
  - Verdict:
    - Low actor-policy lag plus a real warmup snapshot lane is a real structural improvement.
    - The run itself is partially contaminated after u120 by pre-fix rollback semantics; use the fixed reruns below.
- Experiment C - guard-fixed u100 to u120 continuation:
  - Run:
    - `runs/league_seedpool_lag10_guardfix_u100_to_u120_evalguard_20260426`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_seedpool_lag10_lowlr_evalguard_localpromo.yaml --run-label league_seedpool_lag10_guardfix_u100_to_u120_evalguard_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_freshstudent_frontierweighted_seedpool_lag10_u60_to_u140_evalguard_20260426/training/checkpoints/checkpoint_100.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 120 --checkpoint-interval-updates 20 --profile-timers`
  - Dev-eval:
    - u120 weighted aggregate `0.598684`, unweighted `0.75`.
    - Anchors: B1 `0.4375`, B2 `1.0`, B3 `0.6875`, B4 `0.9375`, previous recent `0.4375`.
  - Runtime:
    - tail local `actor_env_steps/sec` about `590-606`;
    - tail learner throughput about `3742-4107 samples/sec`;
    - `league_effective_update=110`, `league_update_lag=6-8`;
    - `pfsp_pool_size=7`, `pfsp_warmup_snapshot_envs=147-176`.
  - Artifact proof:
    - registry `rejected_snapshots=[]`;
    - external seed snapshots and B1 anchor survived.
  - 16-pair scalar confirm:
    - manual confirm under `eval/dev_eval_confirm16_u120_guardfix/update_120`;
    - weighted aggregate `0.575658`, unweighted `0.744792`;
    - anchors: B1 `0.4375`, B3 `0.59375`, B4 `1.0`, previous recent `0.4375`.
  - Verdict:
    - Current best small-model local league candidate from this batch.
    - It is not thesis-grade final evidence, but it is the first clean run where low lag plus active B1 seed-pool warmup improves and survives guard logic.
- Experiment D - immediate PFSP after u120:
  - Run:
    - `runs/league_seedpool_lag10_guardfix_u120_to_u140_pfsp_evalguard_20260426`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_fade_frontierweighted_seedpool_lag10_lowlr_evalguard_localpromo.yaml --run-label league_seedpool_lag10_guardfix_u120_to_u140_pfsp_evalguard_20260426 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_seedpool_lag10_guardfix_u100_to_u120_evalguard_20260426/training/checkpoints/checkpoint_120.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 140 --checkpoint-interval-updates 20 --profile-timers`
  - Runtime:
    - PFSP live: `pfsp_sampling_ready=1`, `pfsp_pool_size=7`, `pfsp_recent_envs` around `47-66`, B1/heuristic lanes also active.
  - Dev-eval:
    - u140 weighted aggregate `0.539474`, unweighted `0.708333`.
    - B1 `0.375`, B3 `0.625`, B4 `0.875`, previous recent `0.375`.
  - Promotion:
    - failed with `anchor_loss_guardrail_exceeded`.
  - Verdict:
    - Activating true PFSP at effective u120 is too early/harmful on this small surface.
    - Do not promote this variant.
- Experiment E - delayed PFSP to u200, normal reference fade:
  - Run:
    - `runs/league_seedpool_lag10_warmup200_u120_to_u140_evalguard_20260426`
  - Dev-eval:
    - u140 weighted aggregate `0.559211`, unweighted `0.71875`.
    - B1 `0.375`, B3 `0.6875`, B4 `0.875`, previous recent `0.375`.
  - Verdict:
    - Better than immediate PFSP but still loses B1/history as the B1 reference fades.
- Experiment F - delayed PFSP plus B1 reference hold after u120:
  - Runs:
    - `runs/league_seedpool_lag10_refhold_warmup200_u120_to_u140_evalguard_20260426`
    - `runs/league_seedpool_lag10_refhold_warmup200_u140_to_u160_evalguard_20260426`
  - Change:
    - held exact B1 reference BC at `0.25` and family BC at `0.35` until u180;
    - held teacher public heuristic coefficient at `0.08` until u180;
    - kept `league.warmup.first_updates=200`.
  - Dev-eval:
    - u140 weighted aggregate `0.592105`, unweighted `0.739583`; B1 `0.4375`, B3 `0.6875`, B4 `0.875`, previous recent `0.4375`.
    - u160 weighted aggregate `0.578947`, unweighted `0.739583`; B1 `0.4375`, B3 `0.625`, B4 `0.9375`, previous recent `0.4375`.
  - 16-pair scalar confirm for u140:
    - manual confirm under `eval/dev_eval_confirm16_u140_refhold/update_140`;
    - weighted aggregate `0.519737`, unweighted `0.692708`;
    - anchors: B1 `0.34375`, B3 `0.625`, B4 `0.84375`, previous recent `0.34375`.
  - Verdict:
    - Ref-hold stabilizes the 8-pair u140 surface, but larger confirm says it is weaker than the u120 guard-fixed candidate.
    - Do not treat u140 as an improvement over u120.
- Experiment G - B1 reference hold from u60:
  - Run:
    - `runs/league_seedpool_lag10_refhold_warmup200_u60_to_u120_evalguard_20260426`
  - Dev-eval:
    - u80 weighted aggregate `0.546053`: B1 `0.4375`, B3 `0.5`, B4 `1.0`, previous recent `0.4375`.
    - u100 weighted aggregate `0.559211`: B1 `0.5`, B3 `0.5`, B4 `0.8125`, previous recent `0.5`.
    - u120 weighted aggregate `0.559211`: B1 `0.4375`, B3 `0.5625`, B4 `0.9375`, previous recent `0.4375`.
  - Verdict:
    - Holding reference from u60 is too conservative or misbalanced; it underperforms the fade-to-u120 candidate.
    - Do not keep this branch alive.
- Current best and interpretation:
  - Best current small-model local candidate from this batch:
    - `runs/league_seedpool_lag10_guardfix_u100_to_u120_evalguard_20260426/training/checkpoints/checkpoint_120.pt`
  - Best local metrics:
    - 8-pair weighted aggregate `0.598684`;
    - 16-pair weighted confirm `0.575658`;
    - B1/history still only `0.4375` at 8/16-pair surfaces, so this is improved but not a final thesis story.
  - Strongest causal findings:
    - actor-policy lag was too high for league learning;
    - explicit external B1 seed pools were previously filtered out on resumed runs;
    - rollback could destroy imported seed pools;
    - confidence-only rollback was too aggressive for noisy local dev-eval;
    - immediate PFSP/self-play after u120 hurts this small student;
    - some B1 reference fade is useful, but holding it from u60 is worse.
  - Next hypotheses:
    1. Promote the low-lag + active seed-pool + guard fixes into the main league server preset, but keep final server warmup gated by a B1/B3 threshold rather than a fixed update if possible.
    2. Replace fixed `league.warmup.first_updates` with an eval-gated transition: require B1/history and B3 to clear local thresholds before PFSP starts.
    3. Try a softer post-warmup transition, e.g. PFSP recent weight ramp from `0.05` to `0.30` over 100-200 updates, instead of switching from warmup snapshots to full recent lane at once.
    4. Re-check larger model/server-surface behavior only after the transition gate is implemented; local throughput is much lower in league because true model-opponent rollouts require actor forwards and cannot use the B1 native heuristic shortcut.

## 2026-04-26 continued - eval-gated dev-target league bridge

- Structural diagnosis after the u120/u140 league branches:
  - B1 disagreement audits showed a board-development pathology, not just noisy eval:
    - `runs/league_b1_disagreement_audit_u120_guardfix_allowmismatch_20260426`: 400 inspected high-delta B1 states, policy top `pass` 104 times; `pass` vs B1 `main_play_character` 69 and vs `main_move` 35; mean TV `0.2971`.
    - `runs/league_b1_disagreement_audit_u140_refhold_allowmismatch_20260426`: 400 inspected states, policy top `pass` 99 times; `pass` vs B1 `main_play_character` 66 and vs `main_move` 33; mean TV `0.2825`.
  - Verdict:
    - B1 reference BC alone was too indirect; the league student still passed in states where B1 developed board.

- Failed quick branches from the clean u120 candidate:
  - `runs/league_seedpool_lag10_evalgated_softref_explore_u120_to_u140_evalguard_20260426`
    - u140 weighted aggregate `0.559211`, unweighted `0.71875`.
    - B1 `0.375`, B3 `0.6875`, B4 `0.875`, previous recent `0.375`.
    - Verdict: weaker than u120; killed.
  - `runs/league_seedpool_lag10_evalgated_softref_explore_passguard_u120_to_u140_evalguard_20260426`
    - Added `rewards.shaping.pass_with_nonpass_penalty=0.02`; penalty was live (`reward_nonzero_fraction` around `0.25-0.26` in that branch), but eval was unchanged.
    - u140 weighted aggregate `0.559211`; B1/history `0.375`.
    - Verdict: pass reward shaping alone did not fix the development error; killed.

- New promising branch - sharper public development target:
  - Preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_devtargetsharp_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key change:
    - teacher public heuristic coef `0.20 -> 0.08`, start/end `u180-u360`;
    - teacher temperature `8.0`;
    - `teacher_public_main_move_coef=0.12`;
    - entropy `0.05 -> 0.025`;
    - B1 reference softened but kept live (`top=0.16 -> 0.05`, family `0.22 -> 0.08`).
  - Runs and metrics:
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u120_to_u140_evalguard_20260426`
      - u140 weighted aggregate `0.598684`, unweighted `0.75`.
      - B1 `0.4375`, B2 `1.0`, B3 `0.6875`, B4 `0.9375`, previous recent `0.4375`.
      - Verdict: no longer decayed after u120.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u140_to_u160_evalguard_20260426`
      - u160 weighted aggregate `0.611842`, unweighted `0.760417`.
      - B1 `0.50`, B2 `1.0`, B3 `0.625`, B4 `0.9375`, previous recent `0.50`.
      - Gate remained closed only because B3 was under the `0.65` threshold.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u160_to_u180_evalguard_20260426`
      - u180 8-pair weighted aggregate `0.592105`, unweighted `0.75`.
      - Automatic 32-pair confirmatory eval weighted aggregate `0.620066`, unweighted `0.768229`.
      - 32-pair anchors: B1 `0.50`, B2 `1.0`, B3 `0.640625`, B4 `0.96875`, previous recent `0.50`.
      - Interpretation: local 8-pair dip was noisy; larger scalar confirm stayed near gate-open quality.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u180_to_u200_evalguard_20260426`
      - u200 weighted aggregate `0.651316`, unweighted `0.78125`.
      - B1 `0.50`, B2 `1.0`, B3 `0.75`, B4 `0.9375`, previous recent `0.50`.
      - League eval warmup gate opened.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u200_to_u220_pfsp_evalguard_20260426`
      - u220 weighted aggregate `0.657895`, unweighted `0.791667`.
      - B1 `0.50`, B2 `1.0`, B3 `0.75`, B4 `1.0`, previous recent `0.50`.
      - Runtime tail stayed in the expected local league band: cumulative actor env steps/sec about `630`, learner throughput tail about `6.9k` samples/sec.
      - Important caveat: before the resume-gate fix below, this short resumed segment still trained with the eval gate closed until the end-of-segment eval.
  - B1 audit on the stronger branch:
    - `runs/league_b1_disagreement_audit_u180_devtargetsharp_confirm32_allowmismatch_20260426`
    - 64 games / 32 paired seeds, 1600 inspected high-delta steps.
    - Policy top `pass` 334 times (`20.9%`, down from `26.0%` at u120); `pass` vs B1 `main_play_character` 202 and vs `main_move` 132; mean TV `0.2496`.
    - Verdict: sharper development target reduced the pass pathology but did not eliminate it.

- Resume/gate implementation fix:
  - Problem:
    - Short resumed league segments forgot that the previous checkpoint had already opened the eval warmup gate, so promotion/PFSP could be skipped until the next dev-eval.
  - Change:
    - Added `_load_resume_checkpoint_dev_eval_summary(...)` in `python/scripts/train.py`.
    - On resume, the runtime can seed `last_dev_eval_summary` and open/close the eval gate from the resumed checkpoint's authoritative `dev_eval_confirmatory` or `dev_eval` summary.
    - It requires config hash match by default, but honors explicit `--resume-allow-config-mismatch` for deliberate research continuations.
  - Tests:
    - `test_load_resume_checkpoint_dev_eval_summary_prefers_confirmatory`
    - `test_load_resume_checkpoint_dev_eval_summary_skips_config_mismatch`
  - Validation:
    - `uv run pytest -q ...` focused gate/config/runtime tests: `6 passed`.

- True post-gate PFSP result:
  - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u220_to_u240_pfsp_resumegatefix_20260426`
    - Resume printed `Seeded resume dev-eval summary: update=220 aggregate=0.6579`.
    - Promotion was attempted, not skipped; it failed with `anchor_loss_guardrail_exceeded`.
    - u240 weighted aggregate `0.618421`, unweighted `0.760417`.
    - B1 `0.4375`, B2 `1.0`, B3 `0.75`, B4 `0.9375`, previous recent `0.4375`.
    - Checkpoint guard rolled final selection back to u220.
  - Verdict:
    - The first true PFSP/promotion handoff after u220 is still harmful on this local small-model surface, mainly because B1/history fall back below `0.50`.

- Failed handoff follow-ups:
  - Soft warmup-snapshot bridge:
    - Temporarily tested keeping warmup seed-snapshot ballast active after the gate opened.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u220_to_u240_pfsp_softbridge2_20260426`
    - Runtime confirmed PFSP open and `pfsp_sampling_weight_warmup_snapshot=0.35`.
    - u240 still weighted aggregate `0.618421`; B1/history still `0.4375`.
    - Verdict: killed; code change was removed from the promoted path.
  - Holdrails / stricter gate:
    - Config artifact retained as failed diagnostic:
      - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_devtargetsharp_holdrails_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_holdrails_u220_to_u240_20260426`
      - u240 weighted aggregate `0.618421`; B1/history `0.4375`.
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_holdrails_u240_to_u260_20260426`
      - u260 weighted aggregate `0.605263`; B1/history still weak.
    - Verdict: killed as a promoted direction. Stronger rails/stricter gate from u220 did not recover the u220 peak.

- Current best league checkpoint:
  - Best local small-model league artifact:
    - `runs/league_seedpool_lag10_evalgated_devtargetsharp_u200_to_u220_pfsp_evalguard_20260426/training/checkpoints/checkpoint_220.pt`
  - Best local scalar metrics:
    - u220 weighted aggregate `0.657895`, unweighted `0.791667`;
    - B1 `0.50`, B2 `1.0`, B3 `0.75`, B4 `1.0`, previous recent `0.50`.
  - Learning curve on the devtargetsharp branch:
    - u140 `0.598684` -> u160 `0.611842` -> u180 confirm `0.620066` -> u200 `0.651316` -> u220 `0.657895`.
    - This is the first main-league branch in this rescue session with a real upward local curve beyond u20/u100.
  - Open risk:
    - The model still does not improve through the first true PFSP handoff; u240/u260 fall back on B1/history.
    - Next high-value experiment should change the handoff more structurally than "hold rails": e.g. PFSP/recent ramp, two-stage gate with B1/history margin, or promotion only after a larger-seed confirm has B1/history above `0.50`.

## 2026-04-26 continued - B1-initialized league bridge and warmup sampler repair

- Starting point:
  - Fresh-student devtargetsharp league was useful but still only confirmed around the low `0.62` range on larger scalar eval.
  - Current B1 no-league anchor checkpoint was much stronger, so the next structural question was whether the thesis league should initialize from B1 instead of asking a fresh small student to rediscover B1 before self-play.

- New B1-initialized bridge:
  - Preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key design:
    - resume from B1 checkpoint `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt`;
    - shift B1 reference and teacher schedules to start at update `450`;
    - keep actor and learner public heuristic bias in parity at `3.0`;
    - delay league PFSP handoff while requiring the B1/B3 eval gate.
  - First run:
    - `runs/league_b1init_devtargetsharp_actorparity3_shift450_u450_to_u470_20260426`
    - u460 8-pair weighted aggregate `0.697368`, unweighted `0.8125`.
    - Anchors: B1 `0.50`, B2 `1.0`, B3 `0.875`, B4 `1.0`, previous recent `0.50`.
    - 32-pair confirm for u460:
      - `runs/league_b1init_devtargetsharp_actorparity3_shift450_u450_to_u470_20260426/eval/dev_eval_confirmatory_manual32/update_460/summary.json`
      - weighted aggregate `0.717105`, unweighted `0.822917`.
      - B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`, previous recent `0.50`.
    - Verdict: big improvement over the fresh-student league and slightly better than the earlier local B1-initialized 8-pair surface, but B1/history remained exactly parity.

- Warmup sampler bug / misleading config fixed:
  - Problem:
    - The pre-PFSP warmup path used a snapshot-only bypass when `warmup_snapshot_mix_fraction > 0`.
    - That bypass skipped the general weighted sampler, so `noleague_baseline_mix_fraction` looked configured but did not contribute direct B1 opponent pressure before PFSP.
    - The inherited `noleague_baseline_mix_end_updates=400` had also expired before the B1-initialized u450 continuation.
  - Code/config change:
    - `python/weiss_rl/runtime.py`: `_assign_episode_roles()` now always uses `_sample_opponent_policy_ids()` for the non-fixed diverse lane, allowing B1 baseline, heuristics, mirror, and warmup snapshots to coexist before PFSP.
    - `configs/presets/...b1init...shift450...yaml`: set `league.sampling.noleague_baseline_mix_end_updates: 750`.
    - `python/weiss_rl/tests/test_runtime.py`: updated the warmup-lane test to assert the weighted sampler is used.
  - Validation:
    - Focused runtime/snapshot test command: `4 passed`.
    - Py-compile command succeeded for `runtime.py`, `train.py`, and `manual_dev_eval_confirm.py`.
    - Runtime proof:
      - `runs/league_b1init_b1warmfix_u470_to_u480_20260426`
      - tail sampling weights: B1 `0.20`, warmup snapshots `0.35`, heuristic public `0.20`, heuristic variants `0.20`, mirror `0.05`;
      - tail env counters included B1 envs and warmup snapshot envs, proving the fixed sampler was live.

- Failed / cooled follow-ups:
  - Normal LR after the fixed sampler:
    - `runs/league_b1init_b1warmfix_u470_to_u480_20260426`
      - u480 8-pair weighted aggregate `0.697368`; B1 `0.50`, B3 `0.875`, B4 `1.0`.
    - `runs/league_b1init_b1warmfix_u480_to_u500_20260426`
      - u500 8-pair weighted aggregate `0.651316`; B3 fell to `0.75`; guard rolled back to u480.
    - Verdict: fixed sampler was correct but normal `2e-5` LR continued to erode the heuristic edge.
  - Stronger direct B1 pressure:
    - Preset:
      - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_b1pressure_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
    - Mix: B1 `0.45`, warmup snapshots `0.10`, heuristics/variants `0.40`, mirror `0.05`.
    - Run:
      - `runs/league_b1init_b1warmfix_strongb1_u480_to_u500_20260426`
      - u500 weighted aggregate `0.664474`; B1 `0.50`, B3 `0.8125`, B4 `0.875`.
    - Verdict: stronger B1 pressure did not break B1 parity and hurt B4; killed as best direction.
  - Very-low LR from the later u480 point:
    - `runs/league_b1init_b1warmfix_vlowlr_u480_to_u500_20260426`
    - u500 8-pair aggregate `0.710526`, but 32-pair confirm cooled to `0.685855`.
    - 32-pair anchors: B1 `0.50`, B3 `0.84375`, B4 `0.984375`, previous recent `0.50`.
    - Verdict: better than normal LR but not better than the earlier u460 confirm.

- New best local small-model league result:
  - Preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key change:
    - same B1-initialized bridge and fixed sampler, but optimizer LR reduced to `5e-6` immediately from the confirmed-good u460 point.
  - Run:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426`
  - u480 8-pair eval:
    - weighted aggregate `0.736842`, unweighted `0.833333`.
    - B1 `0.50`, B2 `1.0`, B3 `1.0`, B4 `1.0`, previous recent `0.50`.
  - u480 32-pair confirm:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/summary.json`
    - weighted aggregate `0.722039`, unweighted `0.825521`.
    - B1 `0.50`, B2 `1.0`, B3 `0.953125`, B4 `1.0`, previous recent `0.50`.
  - Follow-up stability:
    - `runs/league_b1init_b1warmfix_vlowlr_u480best_to_u500_20260426`
    - u500 8-pair aggregate `0.717105`.
    - Automatic 32-pair confirm at u500: weighted `0.722039`, unweighted `0.825521`, anchors identical to the u480 confirm: B1 `0.50`, B3 `0.953125`, B4 `1.0`, previous recent `0.50`.
    - Guard still selected u480 because the 8-pair score dropped from `0.736842` to `0.717105`.
  - Verdict:
    - This is the current best local small-model thesis league artifact.
    - It is a real improvement over the prior fresh-student u220 confirm (`~0.6266`) and the B1-initialized u460 confirm (`0.717105`), while preserving local league throughput in the expected model-opponent range.

- PFSP handoff result:
  - Normal threshold handoff:
    - `runs/league_b1init_b1warmfix_vlowlr_u500_to_u520_20260426`
    - u520 weighted aggregate `0.677632`; B1 `0.50`, B3 `0.8125`, B4 `1.0`, previous recent `0.50`.
    - Runtime still reported `pfsp_sampling_ready=0` at the tail because effective update lag was `510` against the `520` threshold.
  - Earlier PFSP handoff:
    - Preset:
      - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_pfsp500_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
    - Run:
      - `runs/league_b1init_b1warmfix_vlowlr_pfsp500_u500_to_u520_20260426`
    - Runtime tail confirmed PFSP active: recent `0.40`, B1 `0.20`, heuristic public `0.20`, heuristic variants `0.20`, `pfsp_sampling_ready=1`.
    - u520 still weighted aggregate `0.677632`; B3 `0.8125`.
    - Verdict: merely enabling PFSP/recent at this point does not improve the small local model; it erodes the B3 edge.

- Current diagnosis:
  - The B1-initialized bridge is the right thesis-league direction for the small model.
  - The warmup sampler bug was real and is fixed.
  - The best continuation is not high-pressure self-play; it is conservative post-B1 polishing with fixed warmup sampling and `5e-6` LR.
  - The remaining ceiling is B1/history parity:
    - B1 and the imported recent u450 seed snapshot behave like duplicated B1-style anchors in eval;
    - paired scalar eval often lands exactly `0.50` there while the model dominates B3/B4;
    - breaking above B1 likely requires a new strategic signal or a larger/server model, not just more local PFSP.

- Current best artifact to carry forward:
  - Checkpoint:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Confirmatory eval:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/summary.json`
  - Local metrics:
    - 32-pair weighted aggregate `0.722039`;
    - unweighted `0.825521`;
    - B1 `0.50`, B2 `1.0`, B3 `0.953125`, B4 `1.0`, previous recent `0.50`.
  - Throughput scope:
    - Local league actor env steps/sec was in the usual model-opponent band around `600-700`; this is expected to be far below B1 native heuristic rollout throughput because league actors must run model opponents.

- Next hypotheses:
  1. Do not keep local PFSP polishing from this checkpoint unless the transition is changed structurally; current PFSP/recent handoff hurts B3 before improving B1.
  2. Add a B1/history-specific improvement signal that is not just more B1 opponent volume, e.g. B1 disagreement distillation on non-pass development states, or a small supervised patch from B1 winning/losing paired states.
  3. On server, use this checkpoint as the league starting point, run a short DDP smoke, then train with the conservative `5e-6` bridge first; only enable PFSP/recent after a larger-seed gate confirms B1/history parity and B3 above `0.90`.
  4. Consider changing the eval aggregate for development diagnostics so duplicated B1-style anchors are reported separately from true recent league snapshots; do not change thesis canonical eval without re-anchoring the comparison surface.
## 2026-04-27 - Pass 3 start: B1/recent duplication diagnosis and seed-import PFSP guard

- Starting point:
  - Current best small local league checkpoint remains:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Current best 32-pair confirm remains:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/summary.json`
    - weighted `0.722039`, unweighted `0.825521`, B1 `0.50`, B3 `0.953125`, B4 `1.0`, previous recent `0.50`.

- B1/history parity diagnosis:
  - Confirmed that B1 and `Previous recent snapshot` are not merely both `0.50`; on the u480 32-pair confirm they are outcome-identical row-by-row.
  - B1 episodes:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/b1_noleague_baseline/episodes.jsonl`
  - Previous recent episodes:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/seed_315d5e55ce_policy_000009/episodes.jsonl`
  - Exact result from the duplication audit:
    - `64/64` rows match on pair index, swap index, seed, focal seat, outcome, episode key, truncation flag, and termination reason.
    - All `32/32` paired seeds are one-win/one-loss for the focal model.
    - Pattern split is `21` WL pairs and `11` LW pairs.
  - Wrote audit artifact:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_confirmatory_manual32/update_480/b1_recent_duplication_audit.json`
  - Interpretation:
    - The development aggregate is double-counting a B1-family parity surface: B1 baseline weight `3.0` plus previous-recent weight `2.0`.
    - This does not prove the learner cannot beat B1, but it does prove the current "previous recent" diagnostic is not independent league progress for this B1-init bridge.

- Manual B1-only confirm helper:
  - Extended `python/scripts/manual_dev_eval_confirm.py` with `--only-anchor`, allowing larger scalar confirms against one named anchor without constructing a temporary config.
  - Smoke command:
    - `uv run python python/scripts/manual_dev_eval_confirm.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426 --checkpoint runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --summary runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval/update_480/summary.json --update 480 --pairs 4 --workers 2 --artifact-dir-name dev_eval_b1only_manual4_20260427 --only-anchor "B1 NoLeague baseline"`
  - Result:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_b1only_manual4_20260427/update_480/summary.json`
    - B1-only aggregate `0.500000`, `4/8` games, `4` paired seeds, canonical scalar, unbatched.

- Structural PFSP guard:
  - Root cause found in runtime/code:
    - `_import_seed_snapshot_pool()` preserved seed-import provenance only in sidecar metadata/payload.
    - `SnapshotRegistry` did not preserve source kind.
    - `QueueRuntime.refresh_opponent_pool()` treated imported seed snapshots exactly like locally trained league snapshots for recent/champion PFSP pools.
  - Code changes:
    - `python/weiss_rl/league/registry.py`
      - Added `SnapshotMeta.source_kind` with default `local`.
      - Preserved source kind across registry normalize/load/save.
    - `python/scripts/train.py`
      - Seed snapshot imports now register as `source_kind="seed_import"`.
      - B1 baseline aliases now register as `source_kind="baseline_anchor"`.
    - `python/weiss_rl/config/models.py` and `python/weiss_rl/config/parse.py`
      - Added opt-in `league.sampling.exclude_seed_snapshots_from_pfsp`.
    - `python/weiss_rl/runtime.py`
      - Added a PFSP-handoff-active filter that excludes seed-import snapshots from true recent/champion PFSP pools after warmup threshold and eval gate are open.
      - Older registries without `source_kind` are still protected by recognizing `seed_...` imported policy ids.
      - Filter is inactive before PFSP, so warmup snapshot ballast remains available.
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
    - Extends current best very-low-LR B1-init bridge and sets `league.sampling.exclude_seed_snapshots_from_pfsp: true`.

- Smoke:
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-label league_b1init_noseedpfsp_u480_to_u481_smoke_20260427 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 481 --checkpoint-interval-updates 1 --profile-timers --config-override evaluation.periodic_dev_eval_interval_updates=0`
  - Result:
    - Completed one update and wrote `runs/league_b1init_noseedpfsp_u480_to_u481_smoke_20260427`.
    - `run_summary.json`: autoscale local `8 x 64 = 512`, single CUDA, distributed world size `1`, league enabled.
    - Registry shows seed snapshots with `source_kind="seed_import"`, B1 alias with `source_kind="baseline_anchor"`, and new local `policy_000012` with `source_kind="local"`.
    - This smoke stayed pre-PFSP by design: `league_effective_update=480`, `pfsp_sampling_ready=0`, `warmup_snapshot_mix_fraction_active=0.35`.

- Validation:
  - Passed:
    - `uv run python -m py_compile python/weiss_rl/league/registry.py python/weiss_rl/runtime.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py python/scripts/manual_dev_eval_confirm.py`
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_and_champions --tb=short`
    - `uv run pytest -q python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume --tb=short`
  - Config load check:
    - New preset resolves `exclude_seed_snapshots_from_pfsp=True`, `warmup.first_updates=520`.
  - `uv run ruff check ...` was not clean, but the reported issues are existing broad style/import findings in dirty files, not specific failures of this patch path:
    - unsorted imports in `python/scripts/train.py` and `python/weiss_rl/config/parse.py`;
    - unused `PromotionGateAnchor` import in `train.py`;
    - existing runtime lint findings around `diverse_opponent_policy_id`, `getattr(..., "pass_action_id")`, and `fill_value`.

- Current verdict:
  - No new stronger checkpoint yet.
  - We do now have a concrete causal diagnosis for part of the B1/history flatline: the eval and PFSP handoff can treat imported B1-family seed snapshots as if they were independent league recents.
  - The next learning experiment should resume from u480 using the new `noseedpfsp` preset and run through at least the PFSP threshold with scalar dev eval on. If B1 remains exactly `0.50`, the next likely target is B1-specific disagreement/state audit and/or reducing B1 reference BC after parity.

## 2026-04-27 - Pass 3 continuation: clone-family diagnosis, imitation-off probe, and PFSP mirror lane

- Completed no-seed PFSP continuation:
  - Run:
    - `runs/league_b1init_noseedpfsp_u480_to_u540_20260427`
  - Command shape:
    - Resume from `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
    - Config `...shift450_vlowlr_noseedpfsp...yaml`
    - `--max-updates 540 --checkpoint-interval-updates 20 --profile-timers`
  - Results:
    - u500 8-pair dev eval: weighted `0.717105`, B1 `0.50`, B3 `0.9375`, B4 `1.0`, previous recent `0.50`.
    - u520 8-pair dev eval: weighted `0.677632`, B1 `0.50`, B3 `0.8125`, B4 `1.0`, previous recent `0.50`.
    - u540 32-pair confirm: weighted `0.712171`, unweighted `0.820313`, B1 `0.50`, B3 `0.921875`, B4 `1.0`, previous recent `0.50`.
    - Tracker selected u500 as best by small eval; u540 did not beat the previous best u480 confirm.
  - PFSP filter evidence:
    - Post-handoff rows show `league_effective_update=530`, `pfsp_sampling_ready=1`, `warmup_snapshot_mix_fraction_active=0`, `pfsp_sampling_weight_warmup_snapshot=0`, and recent weight `0.40`.
    - Registry source kinds were live: seed imports `source_kind="seed_import"`, B1 alias `source_kind="baseline_anchor"`, local `policy_000012/13/14` as `local`.
  - Important artifact:
    - `runs/league_b1init_noseedpfsp_u480_to_u540_20260427/eval/dev_eval_confirmatory/update_540/summary.json`
  - Verdict:
    - Excluding seed imports from PFSP worked mechanically, but true local recents were still B1-like and did not break B1 parity.

- B1 vs recent duplication after handoff:
  - Compared u540 confirm B1 episodes with previous-recent `policy_000013` episodes:
    - B1: `runs/league_b1init_noseedpfsp_u480_to_u540_20260427/eval/dev_eval_confirmatory/update_540/b1_noleague_baseline/episodes.jsonl`
    - Recent: `runs/league_b1init_noseedpfsp_u480_to_u540_20260427/eval/dev_eval_confirmatory/update_540/policy_000013/episodes.jsonl`
  - Result:
    - `64/64` exact row matches, including outcome, focal seat, seed, and episode key.
  - Interpretation:
    - Even a local league recent can be behaviorally/eval-identical to B1 on this surface. The previous-recent aggregate slot is still not independent evidence of league progress.

- B1 disagreement audit:
  - Initial audit without hash override failed inspection because imported B1 baseline weights carry the B1 config hash, not the league config hash.
  - Successful audit:
    - `runs/league_b1init_noseedpfsp_u540_b1_audit_allowhash_20260427/audit/summary.json`
    - Command used `python/scripts/b2_disagreement_audit.py ... --policy-id policy_000014 --opponent-policy-id b1_noleague_baseline --allow-config-hash-mismatch`.
  - u540 vs B1:
    - `64/64` games rerun, `64` replay bundles inspected.
    - `8786` compared decisions.
    - Mean total variation `0.204415`.
    - Top-action match rate `0.812656`.
    - Top-action-family match rate `0.992602`.
    - Highest-TV deviations are mostly learner `main_move` vs B1 `main_play_character`.
  - u480 best vs B1 audit:
    - `runs/league_b1init_b1warmfix_vlowlr_u480_b1_audit_allowhash_20260427/audit/summary.json`
    - `8579` compared decisions.
    - Mean total variation `0.153558`.
    - Top-action match rate `0.872013`.
    - Top-action-family match rate `1.0`.
  - Interpretation:
    - The best branch is essentially B1-family behavior. PFSP introduced slightly more distributional movement by u540, but not an exploit.

- B1-family age diagnostic:
  - Extended `python/scripts/manual_dev_eval_confirm.py` with `--extra-snapshot-anchor POLICY_ID=DISPLAY_NAME`.
  - Diagnostic command evaluated best u480 checkpoint against imported B1-family snapshots u300/u400/u450:
    - `seed_315d5e55ce_policy_000006=B1-family imported u300`
    - `seed_315d5e55ce_policy_000008=B1-family imported u400`
    - `seed_315d5e55ce_policy_000009=B1-family imported u450`
    - Artifact:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_b1family_manual16_20260427/update_480/summary.json`
  - Result:
    - 16-pair diagnostic: all three anchors scored exactly `0.50`.
    - Episode rows for u300/u400/u450 were `32/32` exact matches against each other.
  - Interpretation:
    - Imported B1 seed snapshots are not meaningful opponent diversity locally; they are a clone-family/eval-equivalent pool.

- Imitation-off probe:
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_refbcoff_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key config:
    - B1 reference top-action and family BC set to `0.0`.
    - Public heuristic teacher reduced to `0.03 -> 0.0` through update `600`.
  - Smoke:
    - `runs/league_b1init_refbcoff_u480_to_u481_smoke_20260427`
  - Probe:
    - `runs/league_b1init_refbcoff_u480_to_u500_20260427`
    - u500 8-pair dev eval weighted `0.697368`, B1 `0.50`, B3 `0.875`, B4 `1.0`, previous recent `0.50`.
    - Training metrics confirm `reference_policy_top_action_bc_coef=0.0` and `reference_policy_top_action_family_bc_coef=0.0`.
  - Verdict:
    - Turning off B1 imitation directly did not discover B1 exploitation and hurt B3. Do not continue this branch blindly.

- PFSP mirror lane structural experiment:
  - Added config/schema/runtime support:
    - `python/weiss_rl/config/models.py`
    - `python/weiss_rl/config/parse.py`
    - `python/weiss_rl/runtime.py`
    - New field: `league.sampling.mirror_mix_fraction`, default `0.0`.
    - When PFSP is ready, the configured mirror lane consumes weight before residual recent sampling, preserving old behavior by default.
  - Added test:
    - `python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_use_configured_mirror_lane_after_pfsp_ready`
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_mirror30_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
    - Sets `league.sampling.mirror_mix_fraction: 0.30`.
  - Validation:
    - `uv run python -m py_compile python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/runtime.py python/scripts/manual_dev_eval_confirm.py`
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_use_configured_mirror_lane_after_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane --tb=short`
  - Smoke:
    - `runs/league_b1init_mirror30_u500_to_u501_smoke_20260427`
  - Main mirror probe:
    - `runs/league_b1init_mirror30_u500_to_u540_20260427`
    - Resumed from `runs/league_b1init_noseedpfsp_u480_to_u540_20260427/training/checkpoints/checkpoint_500.pt`.
    - u520 8-pair dev eval: weighted `0.677632`, B1 `0.50`, B3 `0.8125`, B4 `1.0`, previous recent `0.50`.
    - u540 8-pair dev eval: weighted `0.717105`, B1 `0.50`, B3 `0.9375`, B4 `1.0`, previous recent `0.50`.
    - u540 32-pair manual confirm:
      - `runs/league_b1init_mirror30_u500_to_u540_20260427/eval/dev_eval_confirmatory_manual32/update_540/summary.json`
      - weighted `0.702303`, unweighted `0.815104`, B1 `0.50`, B2 `1.0`, B3 `0.890625`, B4 `1.0`, previous recent `0.50`.
    - PFSP-active performance rows verified:
      - `pfsp_sampling_ready=1`
      - `pfsp_sampling_weight_mirror=0.30`
      - `pfsp_sampling_weight_recent=0.10`
      - `pfsp_sampling_weight_noleague_baseline=0.20`
      - `pfsp_sampling_weight_heuristic_public=0.20`
      - `pfsp_sampling_weight_heuristic_public_variant=0.20`
  - Verdict:
    - Mechanically successful and more controlled than abrupt 40% recent pressure, but not a better checkpoint. It confirms that reducing clone-recent pressure alone is insufficient.

- Current best after this pass:
  - Still:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
    - 32-pair weighted `0.722039`, B1 `0.50`, B3 `0.953125`, B4 `1.0`.
  - New best causal diagnosis:
    - Local B1-family snapshots are eval-equivalent clones.
    - Previous-recent is often a duplicate B1-family pressure term, even after seed-import filtering.
    - Residual BC is not the only cap; removing it hurts B3 before it finds B1 exploitation.
    - PFSP/recent handoff needs stronger novelty/exploit criteria, not just different recent volume.

- Next hypotheses:
  - Add a novelty gate for promotion/PFSP recents: reject or quarantine recents whose B1-anchor paired outcomes are exactly identical to B1 or whose replay audit family-match rate exceeds a high threshold.
  - Add a B1-exploitation objective based on paired-seat counterfactuals rather than simply reducing imitation.
  - Reweight eval/promotion to report B1 and true non-B1 league recents separately; do not let previous-recent double-count B1-family clones.
  - Consider server/larger-model pilot only after a dry run plus a novelty-gated league plan; current local evidence says the unmodified league objective is mostly cloning and recycling B1-family behavior.

## 2026-04-27 - Pass 3 extra improvement attempts: BC sharpness, damage shaping, greedy exploit test, and true-recent eval fix

- Exact-action-free / family-rails probe:
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_familyrails_noexactbc_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Hypothesis:
    - u480 audit showed B1 top-action-family match `1.0` but exact top-action match about `0.872`; maybe preserving family behavior while freeing exact action choices would allow within-family B1 exploitation without losing B3/B4.
  - Smoke:
    - `runs/league_b1init_familyrails_noexactbc_u480_to_u481_smoke_20260427`
    - Log confirmed frozen reference policy `coef=0`, `family_coef=0.3`.
  - Probe:
    - `runs/league_b1init_familyrails_noexactbc_u480_to_u500_20260427`
    - u500 8-pair dev eval: weighted `0.697368`, unweighted `0.8125`, B1 `0.50`, B3 `0.875`, B4 `1.0`, previous recent `0.50`.
  - Verdict:
    - Not promising. Exact top-action BC appears stabilizing; removing only that term still erodes B3 before any B1 gain.

- Stronger damage shaping probe:
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_damage10_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key change:
    - `rewards.shaping.damage_reward: 0.10` instead of `0.05`.
  - u480 -> u500:
    - `runs/league_b1init_damage10_u480_to_u500_20260427`
    - u500 8-pair dev eval: weighted `0.717105`, unweighted `0.822917`, B1 `0.50`, B3 `0.9375`, B4 `1.0`, previous recent `0.50`.
    - This preserved the stable guard score but did not improve B1.
  - u500 -> u540:
    - `runs/league_b1init_damage10_u500_to_u540_20260427`
    - The command timed out after the run had written u540 artifacts; orphaned Python workers were identified by command line and stopped.
    - u520 8-pair dev eval: weighted `0.677632`, B1 `0.50`, B3 `0.8125`, B4 `1.0`, previous recent `0.50`.
    - u540 8-pair dev eval: weighted `0.697368`, B1 `0.50`, B3 `0.875`, B4 `1.0`, previous recent `0.50`.
  - Verdict:
    - Damage `0.10` is not an improvement and still suffers the same handoff/B3 erosion.

- Greedy focal exploit diagnostic:
  - Extended `python/scripts/manual_dev_eval_confirm.py` with diagnostic-only `--focal-action-mode greedy`.
  - This monkeypatches the scalar eval runner in-process for the focal policy only; canonical eval remains sampled.
  - Test:
    - Current best u480 vs B1 only, 16 paired seeds, serial:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_b1only_greedy_manual16_20260427/update_480/summary.json`
    - Result:
      - B1 `0.500000`.
  - Verdict:
    - The current best logits do not hide an easy greedy B1 exploit. The B1 plateau is not merely stochastic sampling failing to realize a better argmax policy.

- True-recent eval/promotion anchor fix:
  - Found structural issue:
    - `_resolve_symbolic_promotion_anchor_policy_id()` used raw `registry.latest_ids()` for `Latest recent snapshot` and `Previous recent snapshot`.
    - That can select seed imports, B1 baseline aliases, or rejected snapshots as if they were true league recents.
  - Code change:
    - `python/scripts/train.py`
      - Added `_true_local_recent_snapshot_ids()`.
      - Symbolic recent anchors now exclude:
        - `b1_noleague_baseline`;
        - rejected snapshots;
        - `seed_...` ids;
        - `source_kind in {"seed_import", "baseline_anchor"}`.
  - Test:
    - Added `python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections`.
    - Passed:
      - `uv run pytest -q python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_opponents_resolve_symbolic_snapshot_anchor_aliases python/weiss_rl/tests/test_train_stall_monitor.py::test_resolve_periodic_dev_eval_opponent_specs_returns_explicit_snapshot_specs --tb=short`
  - Corrected scalar surface artifact:
    - Reran current best u480 on true anchors only because no true previous recent exists in that run after filtering:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual16_20260427/update_480/summary.json`
    - 16-pair result:
      - weighted `0.800000`, unweighted `0.900000`.
      - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `1.0`, B4 `1.0`.
    - Important:
      - This is a corrected diagnostic/eval-surface improvement, not proof of a stronger B1-exploiting model. It removes the bogus duplicate previous-recent B1-family anchor from the aggregate.

- Current state:
  - No new checkpoint beats the current best on canonical B1. B1 remains exactly `0.50`.
  - Current best model checkpoint is still:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Current best corrected true-anchor diagnostic is:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual16_20260427/update_480/summary.json`
  - The strongest structural fixes now in code are:
    - seed imports excluded from PFSP after handoff;
    - symbolic recent eval/promotion anchors restricted to true local non-rejected recents;
    - diagnostic helper supports B1-only, extra snapshot anchors, and greedy focal action mode.

- Next highest-value improvement path:
  - Implement a novelty/exploit gate before a local snapshot can become a true recent:
    - require B1-anchor paired outcome not row-identical to B1/history, or require a minimum disagreement/novelty score from replay audit;
    - otherwise quarantine the snapshot from recent/champion/eval-history lanes.
  - Then run the B1-init bridge with true-recent filtering plus novelty gating; otherwise the league continues to recycle B1-family clones even when the aggregate no longer double-counts them.

## 2026-04-27 - Pass 3 continuation: B1 exploit gate, champion-only symbolic recents, and no-seed warmup probe

- Structural promotion-gate hardening:
  - Added configurable per-anchor promotion floors:
    - `python/weiss_rl/config/models.py`
      - `PromotionGateConfig.target_min_anchor_scores`.
    - `python/weiss_rl/config/parse.py`
      - Parses `league.promotion.gate.target_min_anchor_scores`.
    - `python/weiss_rl/league/promotion_gate.py`
      - Adds `anchor_target_floor_missed` and `anchor_target_floor_missing` rejection reasons.
  - Added test:
    - `python/weiss_rl/tests/test_promotion_gate.py::test_decision_reasons_reject_anchor_below_configured_target_floor`.
  - Validation:
    - `uv run python -m py_compile python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/league/promotion_gate.py`
    - `uv run pytest -q python/weiss_rl/tests/test_promotion_gate.py::test_decision_reasons_reject_anchor_below_configured_target_floor python/weiss_rl/tests/test_promotion_gate.py::test_decision_reasons_use_strict_overall_and_anchor_thresholds --tb=short`

- B1 exploit-gated preset:
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_b1exploitgate_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key settings:
    - Warmup eval gate requires:
      - B1 NoLeague baseline `0.5625`.
      - B3 HeuristicPublicAggro `0.90`.
    - Promotion gate `target_min_anchor_scores` requires:
      - B1 NoLeague baseline `0.5625`.
      - B3 HeuristicPublicAggro `0.90`.
  - Smoke:
    - `runs/league_b1init_b1exploitgate_u480_to_u481_smoke_20260427`
    - Confirmed resume/import worked.
    - Performance metrics showed intended lane weights:
      - `pfsp_sampling_ready=0`.
      - B1 `0.20`, heuristic public `0.20`, heuristic variants `0.20`, warmup snapshots `0.35`, mirror residual `0.05`.

- B1 exploit-gated continuation:
  - Run:
    - `runs/league_b1init_b1exploitgate_u480_to_u540_20260427`
  - u500 8-pair corrected dev eval:
    - weighted `0.775000`, unweighted `0.887500`.
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`.
    - No previous recent was resolved yet.
  - u520 / u540 before the later champion-only symbolic recent patch:
    - u520 weighted `0.677632`, B1 `0.50`, B3 `0.8125`, B4 `1.0`, Previous recent `0.50`.
    - u540 weighted `0.697368`, B1 `0.50`, B3 `0.875`, B4 `1.0`, Previous recent `0.50`.
  - Important finding:
    - Even with PFSP closed, periodic dev eval still resolved an unpromoted local checkpoint as `Previous recent snapshot`.
    - That means the earlier true-recent filter was not strict enough for promotion-gated runs: local checkpoint snapshots that failed or skipped promotion could still masquerade as league history.

- Champion-only symbolic recent fix:
  - Code change:
    - `python/scripts/train.py`
      - `_resolve_symbolic_promotion_anchor_policy_id()` now passes `promotion_gate_enabled`.
      - `_true_local_recent_snapshot_ids()` now requires local snapshot ids to be present in `registry.champion_snapshots` when promotion gating is enabled.
      - Warmup gate now ignores unavailable symbolic recent anchors, so inherited `Previous recent snapshot` thresholds do not permanently close the gate when no trusted recent exists.
  - Tests:
    - Added:
      - `python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated`
      - `python/weiss_rl/tests/test_train_stall_monitor.py::test_league_eval_warmup_gate_ignores_unavailable_symbolic_recent_anchor`
    - Passed:
      - `uv run pytest -q python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_resolve_periodic_dev_eval_opponent_specs_returns_explicit_snapshot_specs --tb=short`
      - `uv run pytest -q python/weiss_rl/tests/test_promotion_gate.py::test_decision_reasons_reject_anchor_below_configured_target_floor python/weiss_rl/tests/test_promotion_gate.py::test_decision_reasons_use_strict_overall_and_anchor_thresholds python/weiss_rl/tests/test_train_stall_monitor.py::test_league_eval_warmup_gate_ignores_unavailable_symbolic_recent_anchor python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections --tb=short`

- No-seed-warmup / higher-entropy probe:
  - New preset:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_b1exploitgate_noseedwarm_entropy08_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Key settings:
    - `league.sampling.warmup_snapshot_mix_fraction: 0.0`
    - `training.exploration.entropy_coef: 0.08`
    - `training.exploration.entropy_anneal_to: 0.04`
  - Intended lane weights before PFSP:
    - B1 `0.20`, heuristic public `0.20`, heuristic variants `0.20`, mirror `0.40`, warmup snapshots `0.0`.
  - Run:
    - `runs/league_b1init_b1exploit_noseedwarm_entropy08_u480_to_u500_20260427`
  - u500 8-pair dev eval:
    - weighted `0.775000`, unweighted `0.887500`.
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`.
  - Continuation:
    - `runs/league_b1init_b1exploit_noseedwarm_entropy08_u500_to_u540_20260427`
    - u520 weighted `0.725000`, unweighted `0.862500`, B1 `0.50`, B3 `0.8125`.
    - u540 weighted `0.750000`, unweighted `0.875000`, B1 `0.50`, B3 `0.875`.
    - Champion-only symbolic recent patch worked: no `Previous recent snapshot` anchor appeared because no candidate had passed promotion.
  - Verdict:
    - Removing imported seed warmup pressure and raising entropy did not break the B1 `0.50` plateau.
    - It did make the experiment cleaner and faster locally, and it prevented clone-like history from polluting the eval aggregate, but the learning signal still does not discover B1-exploiting deviations.

- Current conclusion:
  - New model improvement was not achieved in this block.
  - Stronger structural conclusion:
    - The suspicious `Previous recent` double-count is now mostly corrected:
      - seed imports/baseline/rejected snapshots are excluded;
      - promotion-gated symbolic recents require champion admission.
    - The B1 plateau persists under:
      - normal very-low-LR continuation;
      - stronger B1 pressure;
      - PFSP handoff;
      - seed-import filtering;
      - mirror PFSP lane;
      - no exact B1 top-action BC;
      - reference BC off;
      - stronger damage shaping;
      - greedy focal eval;
      - stricter B1 exploit gate;
      - no seed warmup plus higher entropy.
  - This increasingly points to a missing B1-exploitation learning signal or a hard local equilibrium for this small model/eval surface, not merely a sampling or promotion bug.

- Next hypotheses:
  - Add an explicit B1-loss-focused diagnostic/training signal:
    - collect paired B1 episodes, identify losing paired seeds/states, and compare learner/B1 action logits in those states;
    - build an advantage-weighted or outcome-conditioned auxiliary that rewards deviations only where they correlate with winning against B1.
  - Alternatively, move to a server-safe larger-model pilot from current best after dry-run/smoke, because local small-model B1 parity may be a capacity/eval-surface ceiling.

## 2026-04-27 Pass 3 continuation - B1 row startup repair and hard-seat probes

- Main new bug/diagnostic finding:
  - Process collectors were able to enqueue initial mirror/heuristic-only unrolls before their local opponent pools had been refreshed.
  - A first attempted child-side refresh was still insufficient because `_build_actor_state()` had already assigned non-mirror heuristic roles before model opponents were resident, and `_maybe_reassign_initial_opponents_after_pool_refresh()` refused to reassign unless every env was still mirror.
  - Fix:
    - `python/weiss_rl/runtime.py`
      - Process collectors now call `runtime.refresh_opponent_pool()` after replacing the per-process actor and before the first collection loop.
      - Initial post-refresh role reassignment now resamples all pre-collection actor envs from the real weighted opponent groups instead of skipping lanes that already held heuristic roles.
      - Added `RuntimeUnroll.b1_opponent_mask`, shared-collector transport plumbing, learner-batch propagation, and collector counters:
        - `collector_b1_opponent_env_steps`
        - `collector_b1_opponent_train_rows`
      - Added B1-specific reward scaling:
        - `league.sampling.noleague_baseline_reward_scale`
      - Added hard-seat B1 curriculum hook:
        - `league.sampling.noleague_baseline_force_focal_seat`
    - `python/weiss_rl/learners/impala_learner.py`
      - Added optional B1-opponent-row-only top-action reference BC:
        - `training.b1_opponent_reference_policy_top_action_bc_coef`
      - Metrics:
        - `b1_opponent_reference_policy_top_action_bc_loss`
        - `b1_opponent_reference_policy_top_action_bc_coef`
        - `b1_opponent_reference_policy_top_action_bc_row_fraction`
    - `python/weiss_rl/config/models.py` / `python/weiss_rl/config/parse.py`
      - Parsed the new B1 reward and hard-seat settings.
    - Tests added/updated:
      - `python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_b1_opponent_reference_bc_uses_b1_mask_only`
      - `python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_can_force_b1_baseline_focal_seat`
      - shared collector slot tests now cover the B1 mask payload.

- Validation:
  - Passed:
    - `uv run python -m py_compile python/weiss_rl/runtime.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py`
    - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_b1_opponent_reference_bc_uses_b1_mask_only python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll --tb=short`
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_shared_collector_slot_round_trip_preserves_packed_unroll_payload python/weiss_rl/tests/test_runtime.py::test_shared_pending_unroll_keeps_shared_views_until_release --tb=short`
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_can_force_b1_baseline_focal_seat python/weiss_rl/tests/test_runtime.py::test_runtime_metrics_report_window_and_cumulative_env_step_rates --tb=short`

- Corrected-smoke evidence:
  - Before the startup-role repair:
    - `runs/league_b1init_b1rowbc06_u480_to_u482_diag_20260427`
    - `collector_b1_opponent_env_steps=0`
    - `collector_b1_opponent_train_rows=0`
    - `collector_pfsp_noleague_baseline_envs=0`
    - This invalidated earlier B1-reward/B1-row-BC causal reads.
  - After the startup-role repair:
    - `runs/league_b1init_b1rowbc06_startrole_u480_to_u482_smoke_20260427`
    - u481/u482 had B1 rows live:
      - `collector_b1_opponent_train_rows` about `1430` / `1493`.
      - `b1_opponent_reference_policy_top_action_bc_row_fraction` about `0.175` / `0.177`.
  - Hard-seat smoke:
    - `runs/league_b1init_b1seat1_reward3_u480_to_u481_smoke_20260427`
    - `noleague_baseline_reward_scale_active=3.0`
    - `noleague_baseline_force_focal_seat_active=1.0`
    - `collector_b1_opponent_train_rows=1525`.

- B1-targeted continuation results from current best u480:
  - B1-row reference BC:
    - Config: `configs/presets/pass3_b1rowbc06.yaml`
    - Run: `runs/league_b1init_b1rowbc06_startrole_u480_to_u500_20260427`
    - u500 8-pair scalar:
      - weighted `0.775000`, unweighted `0.887500`
      - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`
    - Verdict: B1-row BC is now genuinely active, but copying B1 on B1 rows does not break B1 parity.
  - B1 reward scale:
    - Config: `configs/presets/pass3_b1reward3.yaml`
    - Run: `runs/league_b1init_b1reward3_startrole_u480_to_u500_20260427`
    - u500 8-pair scalar:
      - weighted `0.775000`, unweighted `0.887500`
      - B1 `0.50`, B3 `0.9375`, B4 `1.0`
    - Verdict: now a valid test after the B1-row repair, but still no B1 movement.
  - B1 reward scale plus global reference BC off:
    - Config: `configs/presets/pass3_b1reward3_refbcoff.yaml`
    - Run: `runs/league_b1init_b1reward3_refbcoff_startrole_u480_to_u500_20260427`
    - u500 8-pair scalar:
      - weighted `0.750000`, unweighted `0.875000`
      - B1 `0.50`, B3 `0.875`, B4 `1.0`
    - Verdict: removing B1 rails allowed drift but did not produce B1 exploitation.
  - Hard-seat B1 reward curriculum:
    - Config: `configs/presets/pass3_b1seat1_reward3.yaml`
    - Run: `runs/league_b1init_b1seat1_reward3_u480_to_u500_20260427`
    - u500 8-pair scalar:
      - weighted `0.775000`, unweighted `0.887500`
      - B1 `0.50`, B3 `0.9375`, B4 `1.0`
    - Continuation:
      - `runs/league_b1init_b1seat1_reward3_u500_to_u540_20260427`
      - u520: weighted `0.725000`, unweighted `0.862500`, B1 `0.50`, B3 `0.8125`, B4 `1.0`
      - u540: weighted `0.750000`, unweighted `0.875000`, B1 `0.50`, B3 `0.875`, B4 `1.0`
      - Checkpoint guard correctly selected u500 as best.
    - Verdict: concentrating B1 pressure on the hard focal seat did not move B1 and eventually eroded the B3 edge.

- B1 ceiling/artifact screen:
  - Artifact mining over league eval summaries found no canonical league run with `B1 NoLeague baseline > 0.50`; all higher B1 scores came from older no-league/local-standard surfaces.
  - Screened later no-league B1 continuation:
    - Command used `manual_dev_eval_confirm.py` against `runs/b1_continue_u450_trainheurrows_vlowlr_u650_s2_20260425/training/checkpoints/checkpoint_650.pt`
    - Artifact:
      - `runs/league_b1init_b1rowbc06_startrole_u480_to_u500_20260427/eval/dev_eval_b1u650_screen_manual16/update_650/summary.json`
    - B1-only manual16 result: `0.500000`.
  - Interpretation:
    - The current B1 anchor lineage is not obviously improved by simply using the later u650 no-league continuation.

- Current best remains:
  - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Corrected true-anchor manual16:
    - weighted `0.800000`, unweighted `0.900000`
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `1.0`, B4 `1.0`

- Updated conclusion:
  - The previous B1 pressure experiments were partly confounded by process-collector startup role assignment; that is now repaired and instrumented.
  - After repair, three valid B1-targeted mechanisms still failed to improve B1:
    - B1-row imitation,
    - B1 reward amplification,
    - hard-seat B1 curriculum.
  - The B1 50/50 wall is therefore less likely to be caused by missing B1 sampling volume and more likely caused by:
    - paired first-seat advantage requiring a very specific second-seat exploit;
    - B1 being too close to the learner/reference policy for ordinary outcome rewards to discover useful deviations locally;
    - insufficient local model/curriculum capacity;
    - or the need for explicit paired-loss/outcome-conditioned state-action training rather than online reward pressure.

- Next concrete hypotheses:
  - Build an offline B1 paired-loss dataset from scalar B1 eval episodes:
    - label paired seeds where focal loses as hard-seat states;
    - compare learner/B1 top actions and action families;
    - train an auxiliary/ranking signal only on losing-seat disagreements that correlate with the winning paired counterpart.
  - Add eval reporting for B1 seat split as a first-class metric so aggregate B1 `0.50` is decomposed into first-seat and second-seat performance every time.
  - For server work, keep the process-collector startup-role repair as mandatory and smoke-check:
    - nonzero `collector_b1_opponent_train_rows`;
    - expected `noleague_baseline_force_focal_seat_active` when configured;
    - DDP world size and rank-0 artifact cleanliness.

## 2026-04-27 Pass 3 continuation - outcome-conditioned B1 audit and weights-only resume probes

- Code/artifact work:
  - Enhanced `python/scripts/b2_disagreement_audit.py` so replay bundle summaries can carry:
    - `focal_seat`, `seat0_policy_id`, `seat1_policy_id`, `outcome`, `focal_win`, `focal_loss`, `winner_seat`.
    - outcome- and seat/outcome-conditioned aggregates:
      - `outcome_counts`
      - `seat_outcome_counts`
      - `variation_by_outcome`
      - `variation_by_seat_outcome`
      - `policy_alignment_by_outcome`
      - `policy_alignment_by_seat_outcome`
      - `top_*_by_outcome`
      - `top_*_by_seat_outcome`
    - mismatch-only top-action/family counters so real deviations are not hidden under same-action agreement.
  - Added/validated focused test coverage in `python/weiss_rl/tests/test_b2_disagreement_audit.py`.
  - Added `--resume-reset-optimizer` to `python/scripts/train.py`:
    - `_restore_learner_from_checkpoint(..., restore_optimizer_state=False)` now loads model weights/counters while starting a fresh optimizer/grad-scaler state.
    - Run metadata records `resume.reset_optimizer`, `determinism_report.resume_reset_optimizer`, and `environment.resume_reset_optimizer`.
    - Added focused checkpoint restore coverage in `python/weiss_rl/tests/test_snapshot_registry.py`.
  - Added config:
    - `configs/presets/pass3_b1seat1_reward5_refbcoff_lr2e5.yaml`
    - Purpose: hard-seat B1 exploit lane, B1 reward scale `5.0`, B1 mix `0.50`, reference top-action/family BC off, LR `2e-5`.

- Validation:
  - Passed:
    - `uv run python -m py_compile python/scripts/b2_disagreement_audit.py`
    - `uv run pytest -q python/weiss_rl/tests/test_b2_disagreement_audit.py --tb=short`
    - `uv run python -m py_compile python/scripts/train.py python/scripts/b2_disagreement_audit.py`
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state python/weiss_rl/tests/test_b2_disagreement_audit.py --tb=short`

- Outcome-conditioned B1 audit:
  - Source audit:
    - `runs/league_b1init_b1warmfix_vlowlr_u480_b1_audit_allowhash_20260427/audit/summary.json`
  - Augmented artifact:
    - `runs/league_b1init_b1warmfix_vlowlr_u480_b1_audit_allowhash_20260427/audit/summary_outcome_conditioned.json`
  - Key findings:
    - B1 audit remains exactly split over 64 games:
      - `outcome_counts`: `W=32`, `L=32`.
    - The split is strongly seat-shaped:
      - focal seat 0: `21W / 11L`
      - focal seat 1: `11W / 21L`
    - Policy alignment is almost indistinguishable across wins/losses:
      - focal seat 1 losses: top-action match rate about `0.8742`, top-family match rate `1.0`, mean probability on B1 top action about `0.7104`.
      - focal seat 1 wins: top-action match rate about `0.8730`, top-family match rate `1.0`, mean probability on B1 top action about `0.7107`.
    - Mismatch-only counters are almost entirely within-family:
      - `encore_decline(slot=0)` vs `encore_decline(slot=1)`.
      - `main_play_character(hand_index=1/2/3/5, stage_slot=...)` vs `main_play_character(hand_index=0, same stage_slot)`.
      - `top_mismatched_family_pairs` is empty in the augmented report.
  - Interpretation:
    - The B1 50/50 wall is not currently explained by pass/main_move/family-level pathology.
    - The learner is a very close B1-family clone; B1 wins/losses differ much more by seat than by obvious action-family disagreement.
    - This makes ordinary B1 pressure, B1 imitation, or simple reference-BC removal a weak route to B1 exploitation.

- Weights-only resume / optimizer-state probe:
  - Full-state hard-seat/ref-off probe:
    - Config: `configs/presets/pass3_b1seat1_reward5_refbcoff_lr2e5.yaml`
    - Smoke:
      - `runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_u480_to_u481_smoke_20260427`
      - B1 rows live: `collector_b1_opponent_train_rows=4108`.
      - `noleague_baseline_reward_scale_active=5.0`.
      - `noleague_baseline_force_focal_seat_active=1.0`.
      - reference BC metrics at zero.
    - Run:
      - `runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_u480_to_u500_20260427`
      - u500 8-pair:
        - weighted `0.725000`, unweighted `0.862500`
        - B1 `0.4375`, B3 `0.875`, B4 `1.0`
      - Verdict: with optimizer state restored, this branch drifted and damaged B1/B3.
  - Reset-optimizer hard-seat/ref-off probe:
    - Smoke:
      - `runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_resetopt_u480_to_u481_smoke_20260427`
      - Metadata confirmed `resume.reset_optimizer=true`.
      - B1 rows live: `collector_b1_opponent_train_rows=4108`.
    - Run:
      - `runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_resetopt_u480_to_u500_20260427`
      - u500 8-pair:
        - weighted `0.800000`, unweighted `0.900000`
        - B1 `0.50`, B3 `1.0`, B4 `1.0`
      - 32-pair confirm:
        - Artifact: `runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_resetopt_u480_to_u500_20260427/eval/dev_eval_trueanchors_manual32_20260427/update_500/summary.json`
        - weighted `0.756250`, unweighted `0.878125`
        - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.890625`, B4 `1.0`
      - Verdict: optimizer reset is a real experimental control; it prevented the worst drift but did not beat the current best on larger confirm.
  - Same-surface current-best confirm:
    - Run/checkpoint:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
    - Artifact:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual32_20260427/update_480/summary.json`
    - 32-pair true-anchor result:
      - weighted `0.787500`, unweighted `0.893750`
      - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.96875`, B4 `1.0`
    - Verdict: current best remains the best confirmed local checkpoint on the true-anchor surface.
  - Reset-optimizer hard-seat/reference-on probe:
    - Config: `configs/presets/pass3_b1seat1_heavy_reward5_lr2e5.yaml`
    - Run:
      - `runs/league_b1init_b1seat1_heavy_reward5_lr2e5_resetopt_u480_to_u500_20260427`
      - u500 8-pair:
        - weighted `0.775000`, unweighted `0.887500`
        - B1 `0.50`, B3 `0.9375`, B4 `1.0`
      - Verdict: keeping reference rails preserves more B3 than the full-state ref-off run, but still does not move B1 and is not worth confirmatory eval.

- Current best after this continuation:
  - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Best same-surface confirm now:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual32_20260427/update_480/summary.json`
    - weighted `0.787500`, unweighted `0.893750`, B1 `0.50`.

- Updated conclusion:
  - `--resume-reset-optimizer` should be used for future branch experiments when changing objective/LR from a checkpoint; otherwise optimizer-state carryover can confound the read.
  - The B1 wall is now better characterized:
    - not missing B1 rows;
    - not fixed by B1 reward scaling;
    - not fixed by hard-seat B1 sampling;
    - not fixed by B1-row imitation;
    - not fixed by reference BC removal;
    - not explained by obvious family-level action pathology in the audit.
  - The remaining plausible local breakthrough probably needs a search/best-response style objective that explicitly discovers second-seat deviations, or a larger/server model pilot with this repaired instrumentation.

### 2026-04-27 pass 3 continuation: B1 seat-1 auxiliary and anti-clone probes

- Goal:
  - Keep pushing on the main blocker: the league/B1-init model remains strong against B3/B4 but suspiciously pinned at `0.50` against the B1 no-league anchor.
  - Test structural B1 second-seat objectives instead of another small sampler-only tweak.

- Code/config changes:
  - Added B1-opponent/second-seat positive-advantage policy auxiliary:
    - Config key: `training.b1_second_seat_positive_advantage_policy_coef`.
    - Learner metric keys:
      - `b1_second_seat_positive_advantage_policy_loss`
      - `b1_second_seat_positive_advantage_policy_coef`
      - `b1_second_seat_positive_advantage_row_fraction`
      - `b1_second_seat_positive_advantage_mean`
    - Intent: reinforce sampled actions only on B1-opponent rows where the focal learner is acting as seat 1 and V-trace advantage is positive.
  - Added B1-opponent/second-seat reference top-action avoidance auxiliary:
    - Config key: `training.b1_second_seat_reference_top_action_avoidance_coef`.
    - Learner metric keys:
      - `b1_second_seat_reference_top_action_avoidance_loss`
      - `b1_second_seat_reference_top_action_avoidance_coef`
      - `b1_second_seat_reference_top_action_avoidance_row_fraction`
    - Intent: deliberately break exact-action cloning against the frozen B1 reference on B1 second-seat rows while leaving broader family/reference rails available.
  - New/updated files in this continuation:
    - `python/weiss_rl/config/models.py`
    - `python/weiss_rl/config/parse.py`
    - `python/scripts/train.py`
    - `python/weiss_rl/learners/impala_learner.py`
    - `python/weiss_rl/tests/test_impala_learner.py`
    - `configs/presets/pass3_b1seat1_posadv_lr2e5.yaml`
    - `configs/presets/pass3_b1seat1_posadv10_lr2e5.yaml`
    - `configs/presets/pass3_b1seat1_refavoid1_lr2e5.yaml`

- Validation:
  - Compile:
    - `uv run python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py`
  - Focused tests:
    - `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_b1_second_seat_positive_advantage_policy_aux_uses_target_rows_only python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_b1_second_seat_reference_avoidance_subtracts_reference_nll python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_b1_opponent_reference_bc_uses_b1_mask_only --tb=short`
    - Result: `3 passed`.
  - Existing focused sampler/resume tests were also kept green earlier in this pass:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume --tb=short`
    - Result: `4 passed`.

- Positive-advantage auxiliary smoke:
  - Config: `configs/presets/pass3_b1seat1_posadv_lr2e5.yaml`
  - Run:
    - `runs/league_b1init_b1seat1_posadv_lr2e5_resetopt_u480_to_u481_smoke_20260427`
  - Command shape:
    - u480 best checkpoint, `--resume-reset-optimizer`, hard B1 seat-1 lane, B1 reward scale `5.0`.
  - Smoke metrics at u481:
    - `collector_b1_opponent_train_rows=4108`
    - `noleague_baseline_force_focal_seat_active=1.0`
    - `noleague_baseline_reward_scale_active=5.0`
    - `b1_second_seat_positive_advantage_policy_loss=0.07523`
    - `b1_second_seat_positive_advantage_row_fraction=0.27334`
    - `b1_second_seat_positive_advantage_mean=0.06885`
  - Verdict: objective is live in the real async path.

- Positive-advantage auxiliary, coef `1.0`:
  - Config: `configs/presets/pass3_b1seat1_posadv_lr2e5.yaml`
  - Run:
    - `runs/league_b1init_b1seat1_posadv_lr2e5_resetopt_u480_to_u500_20260427`
  - u500 8-pair scalar:
    - weighted `0.775000`, unweighted `0.887500`
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`
    - B1 seat diagnostics unchanged from the wall pattern:
      - focal/train total wins `8/16`
      - train wins as seat0 `5/8`, as seat1 `3/8`
  - Tail metrics:
    - `b1_second_seat_positive_advantage_policy_loss` about `0.09`
    - row fraction about `0.25`
  - Verdict: live but too weak or not the right signal; no B1 movement and worse than current best.

- Positive-advantage auxiliary, coef `10.0`:
  - Config: `configs/presets/pass3_b1seat1_posadv10_lr2e5.yaml`
  - Run:
    - `runs/league_b1init_b1seat1_posadv10_lr2e5_resetopt_u480_to_u500_20260427`
  - u500 8-pair scalar:
    - weighted `0.750000`, unweighted `0.875000`
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.875`, B4 `1.0`
  - Verdict:
    - Making the positive-advantage term much stronger still does not move B1.
    - It mostly damages B3, so the problem is not simply that the coef was too small.

- B1 second-seat anti-clone / reference top-action avoidance:
  - Config: `configs/presets/pass3_b1seat1_refavoid1_lr2e5.yaml`
  - Run:
    - `runs/league_b1init_b1seat1_refavoid1_lr2e5_resetopt_u480_to_u500_20260427`
  - u500 8-pair scalar:
    - weighted `0.800000`, unweighted `0.900000`
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `1.0`, B4 `1.0`
    - B1 seat diagnostics still unchanged:
      - focal/train total wins `8/16`
      - train wins as seat0 `5/8`, as seat1 `3/8`
  - Tail metrics:
    - `b1_second_seat_reference_top_action_avoidance_loss=0.52883`
    - `b1_second_seat_reference_top_action_avoidance_row_fraction=0.50072`
  - 32-pair true-anchor confirm:
    - Artifact:
      - `runs/league_b1init_b1seat1_refavoid1_lr2e5_resetopt_u480_to_u500_20260427/eval/dev_eval_trueanchors_manual32_20260427/update_500/summary.json`
    - Result:
      - weighted `0.775000`, unweighted `0.887500`
      - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.9375`, B4 `1.0`
  - Continuation:
    - Run:
      - `runs/league_b1init_b1seat1_refavoid1_lr2e5_u500_to_u520_20260427`
    - u520 8-pair:
      - weighted `0.750000`, unweighted `0.875000`
      - B1 `0.50`, B3 `0.875`, B4 `1.0`
    - Checkpoint guard rolled back to u500:
      - current u520 score `0.7500`
      - best u500 score `0.8000`
  - Verdict:
    - Anti-clone can produce a nice 8-pair screen, but it cools on 32-pair and does not improve B1.
    - It is useful diagnostic evidence that exact top-action cloning alone is not the B1 wall.

- Current best remains unchanged:
  - Checkpoint:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Same-surface 32-pair true-anchor confirm:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual32_20260427/update_480/summary.json`
    - weighted `0.787500`, unweighted `0.893750`
    - B0 `1.0`, B1 `0.50`, B2 `1.0`, B3 `0.96875`, B4 `1.0`

- Updated conclusion / next hypothesis:
  - New structural objectives are implemented and instrumented, but neither targeted positive-advantage reinforcement nor exact-action anti-cloning broke B1 parity locally.
  - The B1 wall is now constrained further:
    - B1 rows are present and seat-forced.
    - B1 reward scale is active.
    - Resetting optimizer is necessary for fair branch probes.
    - Positive-advantage B1-seat1 RL signal is live but does not move the paired B1 score.
    - Exact B1 top-action avoidance is live but still does not move the paired B1 score.
  - Highest-value next lever is likely a real search/counterfactual best-response pipeline, not more scalar coefficients:
    - capture B1-seat1 losing states;
    - evaluate legal candidate continuations or short rollouts from cloned states;
    - train on action advantages that are explicitly counterfactual against B1, not just sampled trajectory advantages;
    - or run a server/larger-model pilot only after the instrumentation above is preserved.

### 2026-04-27 quick diagnosis: recent/champion pools are not a real league ladder yet

- User concern:
  - B1 stays at `0.50` no matter what we do; check whether snapshots/champions/recent opponents are actually updating.
- Findings:
  - Current best run:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/snapshots/registry.json`
    - `champion_snapshots` is empty.
    - `snapshots` contains imported B1-anchor history:
      - `seed_315d5e55ce_policy_000003` update 150
      - `seed_315d5e55ce_policy_000004` update 200
      - `seed_315d5e55ce_policy_000005` update 250
      - `seed_315d5e55ce_policy_000006` update 300
      - `seed_315d5e55ce_policy_000007` update 350
      - `seed_315d5e55ce_policy_000008` update 400
      - `b1_noleague_baseline` update 450
      - `seed_315d5e55ce_policy_000009` update 450
      - local `policy_000011` update 480
    - The u480 small dev eval `Previous recent snapshot` was not the new local u480 policy. It was `seed_315d5e55ce_policy_000009`, scoring `0.50`, same as B1.
    - The 32-pair true-anchor confirm intentionally did not include previous recent/champion.
  - u480-best-to-u500 continuation:
    - Registry still has no champions.
    - `policy_000012` was saved but rejected by checkpoint guard.
    - Both 8-pair and 32-pair evals used `Previous recent snapshot = seed_315d5e55ce_policy_000009`, scoring `0.50`.
  - PFSP500 handoff:
    - Runtime PFSP sampling was active with recent weight `0.40`, but tail performance metrics showed:
      - `pfsp_recent_pool_size=7`
      - `pfsp_champion_pool_size=0`
      - `pfsp_recent_envs > 0`
      - `pfsp_champion_envs=0`
    - Those 7 recents are imported seed snapshots, not a growing set of promoted league checkpoints.
    - Registry ended with `champion_snapshots: ["policy_000013"]`, but during the evaluated collection the champion pool was still empty and eval did not include a previous champion anchor.
  - Latest anti-clone branch:
    - `runs/league_b1init_b1seat1_refavoid1_lr2e5_resetopt_u480_to_u500_20260427`
    - Registry has no champions and one local `policy_000012`.
    - The eval surface for this branch excluded previous recent/champion anchors.
    - Continuation to u520 rejected `policy_000013` via checkpoint guard.
- Interpretation:
  - Yes, snapshots are being written.
  - No, the league has not yet built a meaningful champion ladder.
  - The “recent” opponent that appeared in the main u480/u500 evals is mostly an imported B1-history seed, especially `seed_315d5e55ce_policy_000009`, not a new learned exploiter.
  - This explains why B1 and previous recent often both sit at `0.50`: previous recent is effectively B1-family history.
- Next implication:
  - Fixing learning may require separating imported seed/history snapshots from true local recents/champions in both sampling and eval reporting.
  - Candidate follow-up:
    - add registry source-kind filters for `recent_local` vs `seed_import`;
    - make eval report `Previous imported seed recent` and `Previous local recent` separately;
    - gate PFSP recent sampling so imported B1-like seeds do not dominate the “recent” lane after local snapshots exist;
    - ensure champion pool is evaluated/used only after a real promotion, not inferred from imported seed history.

### 2026-04-27 repair: carry true league history across resumed continuation runs

- User concern:
  - We previously had previous/recent champion behavior, but the B1-init continuation runs were no longer behaving like a real league ladder.
  - B1 stayed at `0.50`, and previous recent/champion reporting was either absent or pointing at B1-like imported seed history.
- Root issue addressed:
  - Resumed continuation runs imported the B1 seed snapshot pool and B1 anchor, but did not carry forward true local league snapshots/champions from the run being resumed.
  - This meant the current run registry could be mostly:
    - `seed_*` imports from the B1 anchor run;
    - `b1_noleague_baseline`;
    - one newly written local candidate;
    - no prior true local recent/champion history.
- Code changes:
  - Added resume-run inference helper in `python/scripts/train.py`:
    - `_infer_run_dir_from_checkpoint_path(...)`
  - Added true league snapshot filter:
    - `_source_snapshot_is_resume_league_snapshot(...)`
    - excludes fixed anchors, rejected snapshots, `seed_*` ids, `seed_import`, and `baseline_anchor`.
  - Added resume league import helper:
    - `_import_resume_league_snapshot_pool(...)`
    - copies eligible source-run league snapshots into the new run registry;
    - preserves original policy IDs, e.g. `policy_000011`;
    - marks them as `source_kind="league_import"`;
    - preserves source champion membership;
    - validates tensor shape/dtype contract before importing.
  - Added seed import exclusion parameter:
    - `_import_seed_snapshot_pool(..., exclude_source_policy_ids=...)`
    - prevents a resume source snapshot imported as true league history from also being imported again as `seed_<hash>_<policy_id>` when seed import is auto-inferred from the same source run.
  - Wired `_import_resume_league_snapshot_pool(...)` into training startup after B1/reference setup and before seed pool import.
- Config change:
  - Updated the main B1-init very-low-LR config:
    - `configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml`
  - Added:
    - `league.sampling.exclude_seed_snapshots_from_pfsp: true`
  - Intent:
    - keep imported B1 seed snapshots as warmup ballast;
    - exclude seed imports from true PFSP after eval-gated handoff;
    - let true local `league_import`/`local` recents and champions drive the league lane.
- Tests/validation:
  - Compile:
    - `uv run python -m py_compile python/scripts/train.py python/weiss_rl/league/registry.py`
  - Focused tests:
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated --tb=short`
    - Result: `3 passed`.
  - Runtime PFSP seed-exclusion tests:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff --tb=short`
    - Result: `2 passed`.
  - Config load:
    - verified the main B1-init very-low-LR config now resolves `league.sampling.exclude_seed_snapshots_from_pfsp=True`.
- Smoke:
  - Run:
    - `runs/league_resume_registrycarry_smoke_u480_to_u481_20260427`
  - Command:
    - resumed from `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
    - used main B1-init very-low-LR config;
    - used `--resume-reset-optimizer`;
    - imported B1 seed pool and B1 baseline as before.
  - Startup evidence:
    - `Imported resume league snapshot pool: count=1 source_run_dir=.../league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426`
    - `Imported seeded snapshot pool: count=7 source_run_dir=.../b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425`
  - Registry evidence after smoke:
    - seed imports:
      - `seed_315d5e55ce_policy_000003` through `seed_315d5e55ce_policy_000009`, `source_kind=seed_import`
    - B1 anchor:
      - `b1_noleague_baseline`, `source_kind=baseline_anchor`
    - carried true league history:
      - `policy_000011`, update `480`, `source_kind=league_import`
    - new local candidate:
      - `policy_000012`, update `481`, `source_kind=local`
  - This is the desired source separation.
- External review prompt:
  - Wrote:
    - `notes/gpt_pro_league_system_review_prompt_2026-04-27.md`
  - It includes:
    - current best checkpoint/results;
    - B1 parity problem;
    - previous recent/champion pool issue;
    - recent code/config fixes;
    - failed experiments;
    - B1 audit findings;
    - commands and questions for GPT Pro.
- Next implication:
  - The previous PFSP/local continuation results should be treated as partially confounded by missing true league-history carry-forward.
  - Next experiment should rerun a short u480 continuation with the fixed registry carry-forward and seed-excluded PFSP path, then inspect:
    - registry source kinds;
    - periodic eval `Previous recent snapshot` policy id;
    - PFSP recent/champion pool composition after handoff;
    - B1, true local recent, imported seed recent, and champion scores separately.

### 2026-04-27 external review packet expansion: self-contained GPT Pro prompt

- User concern:
  - The first GPT Pro prompt was still too repo-dependent.
  - GPT Pro may only see the prompt and possibly this rescue/progress file, so it needs enough embedded code/config/context to reason without browsing the codebase.
- Updated artifact:
  - `notes/gpt_pro_league_system_review_prompt_2026-04-27.md`
- Expansion details:
  - Grew the prompt into a self-contained review packet of roughly 1.8k lines.
  - Added explicit instruction that GPT Pro cannot inspect the repo and should reason only from the prompt.
  - Added code excerpts for:
    - `SnapshotMeta` / `SnapshotRegistry` one-list registry design;
    - `_resolve_symbolic_promotion_anchor_policy_id(...)`;
    - `_true_local_recent_snapshot_ids(...)`;
    - `_seed_snapshot_policy_id(...)`;
    - `_import_seed_snapshot_pool(...)`;
    - `_source_snapshot_is_resume_league_snapshot(...)`;
    - `_import_resume_league_snapshot_pool(...)`;
    - training startup order for B1 baseline, reference attach, resume-league import, seed import, and runtime build;
    - runtime PFSP pool refresh and `exclude_seed_snapshots_from_pfsp` activation logic.
  - Added config excerpts for:
    - current best B1-init very-low-LR config;
    - shifted B1-init parent with reference/family BC and heuristic teacher schedules;
    - typed thesis locked PFSP defaults;
    - local promotion 8-pair surface;
    - B1-exploit-gated variant;
    - no-seed-warmup/higher-entropy branch.
  - Added artifact packet:
    - current best u480 registry where `Previous recent snapshot` resolved to a B1 seed import;
    - stable u480-to-u500 continuation and rejected `policy_000012`;
    - PFSP500 handoff where PFSP was active but recent pool was seed imports;
    - resume-carry smoke registry with `policy_000011` as `league_import` and `policy_000012` as local;
    - B1 disagreement audit showing exact 32/32 split and B1-family cloning.
  - Added pointed review questions around:
    - whether one sorted registry plus filters is too ambiguous;
    - whether `promotion_gate_enabled` should make "recent" champion-only;
    - whether seed-import champions should be stripped from champion state;
    - whether PFSP seed exclusion should activate earlier;
    - whether preserving source policy IDs on resume can collide;
    - whether B1 > 0.50 should be mandatory for promotion;
    - how to redesign eval to separate B1, imported seed history, true local recents, and champions.
- Purpose:
  - The prompt is intended to get structural/debugging feedback, not generic "train longer" advice.
  - It should help diagnose why the league has no upward trend, why B1 remains flat at `0.50`, and why recent/previous champion lanes did not previously show real improvement.

### 2026-04-27 repair: typed league pools and seed-history quarantine

- Trigger:
  - User pasted GPT Pro review feedback.
  - The useful actionable diagnosis was:
    - seed imports should not become active champions;
    - raw `latest_ids(...)` is too ambiguous for recent/champion semantics;
    - previous/recent/champion aliases need explicit seed/local/champion separation;
    - PFSP should not treat imported B1 seed history as true recent/champion history.
- Code changes:
  - `python/weiss_rl/league/registry.py`
    - Added typed selector APIs:
      - `snapshot_by_policy_id()`;
      - `latest_seed_history_ids(...)`;
      - `latest_local_candidate_ids(...)`;
      - `latest_active_champion_ids(...)`;
      - `latest_eligible_ids(...)`.
    - `normalize()` now strips seed imports and baseline anchors from active `champion_snapshots`.
    - Seed-history detection uses `source_kind="seed_import"` and legacy `seed_` prefix fallback.
  - `python/scripts/train.py`
    - `_true_local_recent_snapshot_ids(...)` now uses registry selectors instead of ad hoc raw-list filtering.
    - Added explicit symbolic aliases:
      - `Latest/Previous local candidate snapshot`;
      - `Latest/Previous imported seed history snapshot`;
      - `Latest/Previous promoted champion snapshot`.
    - Existing `Latest/Previous recent snapshot` and `Latest/Previous champion snapshot` now resolve through seed-safe selectors.
    - `_import_seed_snapshot_pool(...)` no longer calls `registry.add_champion(...)` for seed imports.
    - Seed-import metadata now records `source_was_champion` instead of making the seed active champion state.
    - `_import_resume_league_snapshot_pool(...)` now validates existing same-policy collisions via metadata instead of silently accepting an existing policy id.
  - `python/weiss_rl/runtime.py`
    - PFSP champion pool now uses `latest_active_champion_ids(...)`.
    - PFSP recent pool now uses `latest_local_candidate_ids(...)` and excludes active champions.
    - Added separate runtime warmup seed-history lane:
      - `_opponent_warmup_snapshot_ids`;
      - `pfsp_warmup_snapshot_pool_size`.
    - Imported seed history can still be sampled by warmup snapshot mix before PFSP opens, but is no longer labeled recent/champion.
    - After PFSP handoff / eval gate opens, seed history is no longer resident-loaded for the PFSP pool.
- Tests/validation:
  - Compile:
    - `uv run python -m py_compile python/weiss_rl/league/registry.py python/scripts/train.py python/weiss_rl/runtime.py`
  - Focused registry / train alias tests:
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated python/weiss_rl/tests/test_train_stall_monitor.py::test_symbolic_snapshot_aliases_keep_seed_history_explicit --tb=short`
    - Result: `7 passed`.
  - Focused runtime tests:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_champions_out_of_recent_lane --tb=short`
    - Result: `6 passed`.
  - Broader repaired-sampler regression:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated --tb=short`
    - Result: `6 passed`.
- Smoke:
  - Run:
    - `runs/league_typedpools_seedquarantine_smoke_u480_to_u481_20260427`
  - Command:
    - resumed from `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`;
    - used main B1-init very-low-LR config;
    - used `--resume-reset-optimizer`;
    - `--max-updates 481`;
    - B1 seed and B1 baseline run dirs were explicitly passed.
  - Startup evidence:
    - `Imported resume league snapshot pool: count=1 source_run_dir=.../league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426`
    - `Imported seeded snapshot pool: count=7 source_run_dir=.../b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425`
  - Registry evidence after smoke:
    - `champion_snapshots: []`
    - seed imports:
      - `seed_315d5e55ce_policy_000003` through `seed_315d5e55ce_policy_000009`, all `source_kind=seed_import`
    - B1 anchor:
      - `b1_noleague_baseline`, `source_kind=baseline_anchor`
    - true carried league history:
      - `policy_000011`, update `480`, `source_kind=league_import`
    - new local candidate:
      - `policy_000012`, update `481`, `source_kind=local`
    - all seed metadata showed `source_was_champion=False` for this B1 seed source.
  - Runtime metric evidence:
    - `pfsp_pool_size=1`
    - `pfsp_recent_pool_size=1`
    - `pfsp_champion_pool_size=0`
    - `pfsp_warmup_snapshot_pool_size=7`
    - `pfsp_sampling_ready=0`
    - `pfsp_sampling_weight_warmup_snapshot=0.35`
    - This confirms the naming separation:
      - true local/league history is the recent pool;
      - imported B1 seed history is now warmup snapshot history, not recent/champion history.
  - No Python training/eval processes were left running after the smoke.
- Interpretation:
  - This fixes the structural bookkeeping problem exposed by GPT Pro well enough to justify the next real topology-validation run.
  - It does not by itself prove B1 improvement; no new scalar B1 claim should be made from this one-update smoke.
- Next recommended experiment:
  - Run a real fixed-topology continuation from u480 to at least u520/u540.
  - Inspect:
    - registry source-kind timeline;
    - `pfsp_recent_pool_size`, `pfsp_warmup_snapshot_pool_size`, and `pfsp_champion_pool_size`;
    - whether `Previous local candidate snapshot` / old `Previous recent snapshot` resolve to true local/league-import policies;
    - B1, B3, B4, local recent, seed-history diagnostic, and champion scores separately.

## 2026-04-27 - Typed-pool topology validation to PFSP handoff, quality verdict

- User request:
  - Try a short empirical run after the GPT Pro-inspired topology fixes.
  - If it does not work as a quality improvement, leave a new prompt for GPT Pro.
- Code state under test:
  - `SnapshotRegistry` now has typed selectors for seed history, local candidates, active champions, and generic eligible IDs.
  - Seed imports and baseline anchors are stripped from active champion state during registry normalization.
  - Seed imports record `source_was_champion` metadata but are not made active champions.
  - Resume-carried league snapshots are imported as `source_kind=league_import`.
  - Runtime now separates true PFSP recent/champion pools from seed-history warmup snapshots.
  - Symbolic eval aliases now include explicit local-candidate, imported-seed-history, and promoted-champion aliases; legacy recent/champion aliases route through seed-safe selectors.
- Validation before longer run:
  - Compile:
    - `uv run python -m py_compile python/weiss_rl/league/registry.py python/scripts/train.py python/weiss_rl/runtime.py`
  - Registry / train alias tests:
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated python/weiss_rl/tests/test_train_stall_monitor.py::test_symbolic_snapshot_aliases_keep_seed_history_explicit --tb=short`
    - Result: `7 passed`.
  - Runtime pool tests:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_champions_out_of_recent_lane --tb=short`
    - Result: `6 passed`.
  - Repaired-sampler regression tests:
    - `uv run pytest -q python/weiss_rl/tests/test_runtime.py::test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready python/weiss_rl/tests/test_runtime.py::test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated --tb=short`
    - Result: `6 passed`.
- Topology smoke:
  - Run:
    - `runs/league_typedpools_seedquarantine_smoke_u480_to_u481_20260427`
  - Resume:
    - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
  - Key evidence:
    - Resume league import count: `1`.
    - Seed snapshot import count: `7`.
    - Registry had `policy_000011` as `source_kind=league_import`, update `480`.
    - Registry had `policy_000012` as `source_kind=local`, update `481`.
    - `champion_snapshots: []`.
    - Seed imports stayed `source_kind=seed_import`.
    - B1 anchor stayed `source_kind=baseline_anchor`.
    - Runtime pre-PFSP:
      - `pfsp_recent_pool_size=1`;
      - `pfsp_champion_pool_size=0`;
      - `pfsp_warmup_snapshot_pool_size=7`;
      - `pfsp_sampling_ready=0`;
      - `pfsp_sampling_weight_warmup_snapshot=0.35`.
  - Verdict:
    - The one-update smoke proved the semantic separation: local/league history was the recent pool, while imported B1 seed history was only warmup history.
- Short fixed-topology continuation:
  - Run:
    - `runs/league_typedpools_seedquarantine_u480_to_u500_20260427`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-label league_typedpools_seedquarantine_u480_to_u500_20260427 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --resume-allow-config-mismatch --resume-reset-optimizer --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 500 --checkpoint-interval-updates 20 --profile-timers`
  - Wall time:
    - About `684s` local Windows.
  - u500 dev eval:
    - Aggregate: `0.8000`.
    - `B0 RandomLegal`: `1.0`.
    - `B1 NoLeague baseline`: `0.50`.
    - `B2 HeuristicPublic`: `1.0`.
    - `B3 HeuristicPublicAggro`: `1.0`.
    - `B4 HeuristicPublicControl`: `1.0`.
  - Registry evidence:
    - `champion_snapshots: []`.
    - `rejected_snapshots: []`.
    - `policy_000011`: update `480`, `source_kind=league_import`.
    - `policy_000012`: update `500`, `source_kind=local`.
    - Seed imports remained seed-only.
  - Runtime evidence:
    - `league_effective_update=490`.
    - `pfsp_sampling_ready=0`.
    - `pfsp_recent_pool_size=1`.
    - `pfsp_warmup_snapshot_pool_size=7`.
    - `pfsp_sampling_weight_warmup_snapshot=0.35`.
    - `pfsp_warmup_snapshot_envs > 0`.
    - `pfsp_recent_envs=0`.
  - Verdict:
    - Still pre-PFSP because effective update lag had not crossed the configured threshold.
    - Topology remained clean; no B1 improvement evidence.
- PFSP handoff continuation:
  - Run:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-label league_typedpools_seedquarantine_u500_to_u540_20260427 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/league_typedpools_seedquarantine_u480_to_u500_20260427/training/checkpoints/checkpoint_500.pt --resume-allow-config-mismatch --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 540 --checkpoint-interval-updates 20 --profile-timers`
  - Wall time:
    - About `1362s` local Windows.
  - Startup evidence:
    - Resumed update `500`, policy version `12`.
    - Imported resume league snapshot pool count: `2`.
    - Imported seed pool count: `7`.
  - u520 dev eval:
    - Aggregate: `0.7750`.
    - `B1`: `0.50`.
    - `B3`: `0.9375`.
    - Checkpoint guard rolled back on score drop.
    - `policy_000013` rejected.
  - u540 dev eval:
    - Aggregate: `0.8000`.
    - `B1`: `0.50`.
    - `B2`: `1.0`.
    - `B3`: `1.0`.
    - `B4`: `1.0`.
    - Promotion gate passed for `policy_000014`.
  - Final registry:
    - `champion_snapshots: ['policy_000014']`.
    - `rejected_snapshots: ['policy_000013']`.
    - `policy_000011`: update `480`, `source_kind=league_import`.
    - `policy_000012`: update `500`, `source_kind=league_import`.
    - `policy_000013`: update `520`, `source_kind=local`, rejected.
    - `policy_000014`: update `540`, `source_kind=local`, active champion.
    - Seed imports stayed seed history and did not enter active champion state.
  - PFSP handoff runtime evidence after effective update crossed threshold:
    - `league_effective_update=530`.
    - `pfsp_sampling_ready=1`.
    - `pfsp_pool_size=2`.
    - `pfsp_recent_pool_size=2`.
    - `pfsp_champion_pool_size=0` during collection before the u540 promotion.
    - `pfsp_warmup_snapshot_pool_size=7`.
    - `pfsp_sampling_weight_recent=0.40`.
    - `pfsp_sampling_weight_warmup_snapshot=0.0`.
    - `pfsp_recent_envs > 0`.
    - `pfsp_warmup_snapshot_envs=0`.
  - Verdict:
    - This proves the repaired PFSP handoff is semantically alive: it sampled true local/league-import recents after handoff and did not sample imported B1 seed history as recent/champion.
- 32-pair scalar confirm for u540:
  - Command:
    - `uv run python python/scripts/manual_dev_eval_confirm.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --checkpoint runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --summary runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/dev_eval/update_540/summary.json --update 540 --pairs 32 --workers 6 --artifact-dir-name dev_eval_trueanchors_manual32_20260427`
  - Wall time:
    - About `345s` local Windows.
  - Result:
    - Weighted aggregate: `0.768750`.
    - Unweighted aggregate: `0.884375`.
    - `B0 RandomLegal`: `1.0`.
    - `B1 NoLeague baseline`: `0.50`.
    - `B2 HeuristicPublic`: `1.0`.
    - `B3 HeuristicPublicAggro`: `0.921875`.
    - `B4 HeuristicPublicControl`: `1.0`.
  - B1 seat diagnostics:
    - Current `train_u540_p14`: total wins `32/64`; wins as seat0 `14/32`; wins as seat1 `18/32`.
    - B1 baseline: total wins `32/64`; wins as seat0 `14/32`; wins as seat1 `18/32`.
    - This is exactly mirrored, not hidden B1 improvement.
  - Same-surface comparison to prior best u480 true-anchor 32-pair confirm:
    - Previous u480 weighted aggregate: `0.787500`.
    - Previous u480 unweighted aggregate: `0.893750`.
    - Previous u480 `B1`: `0.50`.
    - Previous u480 `B3`: `0.96875`.
    - Previous u480 `B4`: `1.0`.
    - New u540 weighted aggregate: `0.768750`.
    - New u540 unweighted aggregate: `0.884375`.
    - New u540 `B1`: `0.50`.
    - New u540 `B3`: `0.921875`.
  - Verdict:
    - The repair did not produce a thesis-quality learning improvement.
    - The true local PFSP branch cooled against B3 and remained exactly flat against B1.
    - Continuing this exact branch blindly is not justified as the next main move.
- GPT Pro follow-up prompt:
  - Wrote:
    - `notes/gpt_pro_followup_after_typedpools_2026-04-27.md`
  - Purpose:
    - Provide a self-contained prompt with the post-fix code summary, test evidence, run metrics, and explicit request for next structural diagnosis.
  - Core question for GPT Pro:
    - Now that the league topology is honest and PFSP samples true local recents, why does quality still fail to trend upward and why is B1 still exactly mirrored at `0.50`?
- Final local state:
  - Checked for lingering Python/uv processes after the manual confirm; none were found.
- Overall conclusion:
  - Improvement since the start of this repair pass:
    - Yes, structurally. The previous/recent/champion/seed bookkeeping problem is materially fixed and test-covered.
    - No, not yet in policy quality. B1 remains exactly `0.50`, and the fixed-topology u540 checkpoint is worse than the previous u480 best on 32-pair confirm.
  - Next hypothesis:
    - Treat B1 parity as a separate best-response / clone-equilibrium problem.
    - The next substantive work should be an explicit B1 exploiter or counterfactual rollout audit, not more blind continuation of this PFSP branch.

## 2026-04-27 - GPT Pro artifact-triage loop: B1 matrix and exact per-seed symmetry

- User request:
  - Implement and test GPT Pro's next recommendations until there is enough evidence to accept or discard the loop.
  - Write a long same-session follow-up prompt for GPT Pro with the new evidence and a final line asking what extra context it wants next.
- Implemented diagnostic artifact script:
  - Added:
    - `python/scripts/b1_artifact_matrix.py`
  - Purpose:
    - No-training artifact triage for B1/u480/u540.
    - Loads B1 baseline plus arbitrary checkpoint policies.
    - Runs ordered seat-swapped matrix matchups.
    - Emits:
      - `policy_load_manifest.json`;
      - `resolved_policies.json`;
      - `matrix_summary.json`;
      - per-matchup `episodes.jsonl`;
      - per-matchup `matchup_summary.json`;
      - per-matchup `seat_diagnostics.json`;
      - per-matchup `pair_class_summary.json`;
      - per-matchup `pair_table.jsonl`.
  - Manifest fields include:
    - source path;
    - source file SHA256;
    - loaded model object id;
    - state-dict SHA256;
    - state-dict parameter L2 norm;
    - pairwise state-dict L2 distance;
    - public heuristic learner/actor bias scales.
  - Added `--focal-action-mode greedy` to force the focal policy to use argmax over legal logits while the opponent remains normal sampled eval.
- Small registry hardening from GPT Pro:
  - File:
    - `python/weiss_rl/league/registry.py`
  - Change:
    - `latest_active_champion_ids(...)` now excludes `rejected_snapshots`.
    - `normalize()` strips rejected IDs from active champion state.
  - Test updated:
    - `python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions`
- Validation:
  - Compile:
    - `uv run python -m py_compile python/weiss_rl/league/registry.py python/scripts/b1_artifact_matrix.py`
    - Result: passed.
  - Focused selector test:
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions --tb=short`
    - Result: `1 passed`.
  - Focused topology/runtime regression:
    - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_champions_out_of_recent_lane --tb=short`
    - Result: `5 passed`.
- Matrix smoke:
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 1 --artifact-dir-name b1_artifact_matrix_smoke_p1_20260427 --device cuda:0 --include-self`
  - Artifact dir:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_smoke_p1_20260427`
  - Result:
    - All 9 ordered/self matchups split the single paired seed `1-1`.
- Main sampled matrix:
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_p8_20260427 --device cuda:0 --include-self`
  - Wall time:
    - About `472s` local Windows.
  - Artifact dir:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_p8_20260427`
  - Result:
    - `B1 NoLeague baseline vs B1 NoLeague baseline`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `B1 NoLeague baseline vs u480`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `B1 NoLeague baseline vs u540`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u480 vs B1 NoLeague baseline`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u480 vs u480`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u480 vs u540`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u540 vs B1 NoLeague baseline`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u540 vs u480`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
    - `u540 vs u540`: mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
  - Complement checks:
    - B1/u480, B1/u540, and u480/u540 forward/reverse means all summed exactly to `1.0`.
- Load manifest evidence:
  - B1:
    - source:
      - `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/snapshots/b1_noleague_baseline/weights.pt`
    - source file SHA256:
      - `8861bd04db5882ff2878a775c1487438df556cb914ff996edf81a086eb99c310`
    - state-dict SHA256:
      - `620bf243792eb2c25d40539c1f035e7e1efafe4d35a3e7e40c576d1ccf514759`
    - state-dict L2 norm:
      - `1452.0640368167014`
    - public heuristic learner/actor bias:
      - learner `3.0`, actor `1.0`.
  - u480:
    - source:
      - `runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt`
    - source file SHA256:
      - `1d23190a9c201bb35aee04745e656e2bbef48ca298e21e702036c618f1b372e4`
    - state-dict SHA256:
      - `aa67db4a97a028b65cb532d3a5415b131e4586c7ee0dd81ccd7cf12e06abea46`
    - state-dict L2 norm:
      - `1452.0640157170303`
    - public heuristic learner/actor bias:
      - learner `3.0`, actor `3.0`.
  - u540:
    - source:
      - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt`
    - source file SHA256:
      - `505372cc61b109b02a66ed941ca41fe9a6db24fa2ea0da412c14130a045c43dc`
    - state-dict SHA256:
      - `6b48fd712a03a7a6d9ba09c9642fdc55bb1fb34e562a09df6279227a6d75a1b3`
    - state-dict L2 norm:
      - `1452.0640236808495`
    - public heuristic learner/actor bias:
      - learner `3.0`, actor `3.0`.
  - Pairwise parameter L2 distances:
    - B1 vs u480:
      - `0.3942730264801399` over `2,954,524` float params.
    - B1 vs u540:
      - `0.46297084975380637` over `2,954,524` float params.
    - u480 vs u540:
      - `0.11901920451535859` over `2,954,524` float params.
  - Verdict:
    - Same-file / same-state-dict loading bug is mostly ruled out.
    - Distances are tiny relative to norm around `1452`, so functional near-cloning remains likely.
- Greedy focal matrix:
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_greedyfocal_p8_20260427 --device cuda:0 --focal-action-mode greedy`
  - Wall time:
    - About `315s` local Windows.
  - Artifact dir:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_greedyfocal_p8_20260427`
  - Result:
    - All 6 non-self ordered matchups still had mean `0.5`, wins `8`, losses `8`, pair classes `1-1: 8`.
  - Verdict:
    - Greedy focal does not reveal a hidden deterministic winning mode for u480/u540 on this matrix.
- Strongest new evidence:
  - Per-seed pattern comparison showed identical physical-seat winner patterns across:
    - `B1_NoLeague_baseline__vs__B1_NoLeague_baseline`;
    - `u480__vs__B1_NoLeague_baseline`;
    - `u540__vs__B1_NoLeague_baseline`;
    - `u480__vs__u540`;
    - `u540__vs__u480`.
  - For all those matchups, the first 8 pair outcomes were:
    - pair 0: `W/L`, winner seats `0/0`;
    - pair 1: `L/W`, winner seats `1/1`;
    - pair 2: `W/L`, winner seats `0/0`;
    - pair 3: `W/L`, winner seats `0/0`;
    - pair 4: `W/L`, winner seats `0/0`;
    - pair 5: `L/W`, winner seats `1/1`;
    - pair 6: `L/W`, winner seats `1/1`;
    - pair 7: `W/L`, winner seats `0/0`.
  - Interpretation:
    - For these B1-family model policies on these paired seeds, the physical seat/seed winner pattern is invariant to whether the loaded model is B1, u480, or u540.
    - This is stronger than "B1 is 0.50"; it suggests the current model-vs-model paired surface is dominated by near-clone policy behavior and/or seat-seed determinism.
- Accepted/discarded hypotheses after this loop:
  - Mostly discarded:
    - B1 and current are literally the same loaded weights.
    - B1 alias resolves to current checkpoint.
    - Simple swapped-label complement is broken.
    - Stochastic sampling alone hides a greedy winning mode for u480/u540.
  - Still live and now more likely:
    - B1/u480/u540 are functionally near-identical under learner public heuristic bias.
    - Policy changes are too small or too within-family to affect terminal outcomes.
    - Paired-seat eval cancels all near-clone model-vs-model changes into exact `1-1` pairs.
    - The next useful experiment should inspect action traces/logits or force counterfactual actions, not train more PFSP.
- GPT Pro follow-up prompt:
  - Wrote:
    - `notes/gpt_pro_followup_after_b1_artifact_matrix_2026-04-27.md`
  - It includes:
    - code summary;
    - commands;
    - manifests;
    - exact matrix results;
    - per-seed pattern evidence;
    - current accepted/discarded hypotheses;
    - explicit request for the next implementation target.
  - It also asks GPT Pro what exact extra context it wants in the next packet.
- Current recommendation:
  - Do not run another league/PFSP continuation yet.
  - Next likely local work:
    - no-public-heuristic-bias eval override and/or both-greedy matrix;
    - action trace digest comparison;
    - shared-state logit/action-family probe;
    - minimal forced-action counterfactual on a few invariant seat-losing states.

## 2026-04-27 - B1 matrix controls: public-bias/scoring wrapper is now the prime suspect

- Goal:
  - Act on the GPT Pro recommendation after the first B1 artifact matrix.
  - Extend the matrix to prove controls actually hit action selection, then run short diagnostics before any more PFSP/server training.
- Code changed:
  - `python/scripts/b1_artifact_matrix.py`
    - Added builtin policy support:
      - `--include-builtin B0_RandomLegal`
      - `--include-builtin B2_HeuristicPublic`
      - `--include-builtin B3_HeuristicAggro`
      - `--include-builtin B4_HeuristicControl`
    - Added targeted pair filtering:
      - `--matchup focal=opponent`
    - Added public heuristic bias controls:
      - `--disable-public-heuristic-bias`
      - `--public-heuristic-bias-scale <float>`
    - Added scoring/action controls:
      - `--scoring-mode learner|actor`
      - `--both-greedy`
      - `--seed-scope`
      - `--seed-offset`
      - `--action-rng-salt-mode shared|policy|matchup`
    - Added hard counters in every matchup summary:
      - `model_decisions`
      - `heuristic_decisions`
      - `random_legal_decisions`
      - `sample_decisions`
      - `greedy_override_decisions`
      - `fallback_to_parent_decisions`
      - `scoring_mode`
      - `greedy_policy_ids`
      - `action_rng_salt_mode`
    - Added hard failures:
      - greedy requested for a model policy but `greedy_override_decisions == 0`;
      - bias override requested but effective learner/actor scale does not match;
      - unknown builtin alias.
- Validation:
  - Compile:
    - `uv run python -m py_compile python/scripts/b1_artifact_matrix.py`
  - Control smoke:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 1 --artifact-dir-name b1_artifact_matrix_controls_smoke_p1_20260427 --device cuda:0 --disable-public-heuristic-bias --both-greedy --matchup u540=B1`
  - Smoke result:
    - Artifact:
      - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_controls_smoke_p1_20260427`
    - `u540 vs B1`: mean `0.5`, wins `1`, losses `1`, pair classes `1-1: 1`.
    - Hard-counter proof:
      - `model_decisions: 342`
      - `greedy_override_decisions: 342`
      - `sample_decisions: 0`
    - Bias override proof:
      - B1/u480/u540 effective learner and actor bias scales were all `0.0`.
- Diagnostic A: official learner scoring with contrast policies.
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_builtincontrast_p8_20260427 --device cuda:0 --include-builtin B0_RandomLegal --include-builtin B2_HeuristicPublic --include-builtin B3_HeuristicAggro --include-builtin B4_HeuristicControl --matchup B1=B0 --matchup B1=B2 --matchup B1=B3 --matchup B1=B4 --matchup u540=B0 --matchup u540=B2 --matchup u540=B3 --matchup u540=B4`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_builtincontrast_p8_20260427`
  - Results:
    - `B1 vs B0`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `B1 vs B2`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `B1 vs B3`: `0.8125`, wins/losses `13/3`, pair classes `2-0: 5`, `1-1: 3`.
    - `B1 vs B4`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `u540 vs B0`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `u540 vs B2`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `u540 vs B3`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
    - `u540 vs B4`: `1.0`, wins/losses `16/0`, pair classes `2-0: 8`.
  - Verdict:
    - The eval/action path is policy-sensitive in general.
    - The exact B1-family `0.50` is not global evaluator death.
    - u540 separates from B1 against B3 on this small contrast surface.
- Diagnostic B: no public heuristic bias, common scale `0.0`.
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_nobias_p8_20260427 --device cuda:0 --disable-public-heuristic-bias`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_nobias_p8_20260427`
  - Bias override proof:
    - B1 before learner/actor: `3.0/1.0`; effective: `0.0/0.0`.
    - u480 before learner/actor: `3.0/3.0`; effective: `0.0/0.0`.
    - u540 before learner/actor: `3.0/3.0`; effective: `0.0/0.0`.
  - Results:
    - `B1 vs u480`: `0.8125`, wins/losses `13/3`, pair classes `2-0: 5`, `1-1: 3`.
    - `B1 vs u540`: `0.9375`, wins/losses `15/1`, pair classes `2-0: 7`, `1-1: 1`.
    - `u480 vs B1`: `0.1875`, wins/losses `3/13`, pair classes `0-2: 5`, `1-1: 3`.
    - `u540 vs B1`: `0.25`, wins/losses `4/12`, pair classes `0-2: 4`, `1-1: 4`.
    - `u480 vs u540`: `0.5`, pair classes `2-0: 2`, `1-1: 4`, `0-2: 2`.
    - `u540 vs u480`: `0.5625`, pair classes `2-0: 3`, `1-1: 3`, `0-2: 2`.
  - Verdict:
    - Without the hand-coded public heuristic bias, B1 beats u480/u540 strongly.
    - u480/u540 are not stronger learned network cores on this no-bias surface.
    - Official bias scale `3.0` appears to mask learned-network weakness and collapse B1-family outcomes.
- Diagnostic C: both-greedy official learner scoring.
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_bothgreedy_p8_20260427 --device cuda:0 --both-greedy`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bothgreedy_p8_20260427`
  - Results:
    - All six ordered B1/u480/u540 model matchups remained mean `0.5`, wins/losses `8/8`, pair classes `1-1: 8`.
    - Every matchup had:
      - `model_decisions: 2128`
      - `greedy_override_decisions: 2128`
      - `sample_decisions: 0`
  - Verdict:
    - Stochastic action RNG is not the main reason official B1-family matchups lock to `0.50`.
    - Official learner scoring at bias `3.0` produces the same paired physical-seat pattern even under deterministic argmax.
- Diagnostic D: actor scoring mode, no override.
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_actor_p8_20260427 --device cuda:0 --scoring-mode actor`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_actor_p8_20260427`
  - Effective actor scales:
    - B1: `1.0`.
    - u480: `3.0`.
    - u540: `3.0`.
  - Results:
    - `B1 vs u480`: `0.125`, wins/losses `2/14`, pair classes `0-2: 6`, `1-1: 2`.
    - `B1 vs u540`: `0.125`, wins/losses `2/14`, pair classes `0-2: 6`, `1-1: 2`.
    - `u480 vs B1`: `0.9375`, wins/losses `15/1`, pair classes `2-0: 7`, `1-1: 1`.
    - `u540 vs B1`: `0.9375`, wins/losses `15/1`, pair classes `2-0: 7`, `1-1: 1`.
    - `u480 vs u540`: `0.5`, pair classes `1-1: 8`.
    - `u540 vs u480`: `0.5`, pair classes `1-1: 8`.
  - Verdict:
    - Actor mode makes u480/u540 appear much stronger than B1.
    - This is not clean thesis evidence because the wrappers are unequal: B1 actor bias `1.0`, u480/u540 actor bias `3.0`.
    - It does expose a serious actor-vs-learner effective-policy mismatch.
- Diagnostic E: common public heuristic bias scale `1.0`.
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_artifact_matrix_bias1_p8_20260427 --device cuda:0 --public-heuristic-bias-scale 1.0`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bias1_p8_20260427`
  - Effective scale proof:
    - B1/u480/u540 learner and actor scales were all forced to `1.0`.
  - Results:
    - `B1 vs u480`: `0.5`, pair classes `2-0: 1`, `1-1: 6`, `0-2: 1`.
    - `B1 vs u540`: `0.625`, pair classes `2-0: 2`, `1-1: 6`.
    - `u480 vs B1`: `0.5`, pair classes `1-1: 8`.
    - `u540 vs B1`: `0.4375`, pair classes `2-0: 1`, `1-1: 5`, `0-2: 2`.
    - `u480 vs u540`: `0.8125`, pair classes `2-0: 5`, `1-1: 3`.
    - `u540 vs u480`: `0.3125`, pair classes `1-1: 5`, `0-2: 3`.
  - Verdict:
    - Common scale `1.0` breaks the exact `1-1: 8` invariance in several matchups.
    - It still does not show a clean learned-policy win over B1.
    - u480 looks better than u540 at common scale `1.0`, which is another warning that u540 is not a better main policy.
- Updated diagnosis:
  - Mostly discarded:
    - global eval/action path dead;
    - same checkpoint loaded for B1/current;
    - greedy override not firing;
    - stochastic action RNG alone explains B1 `0.50`.
  - Now most likely:
    - Official learner public heuristic bias scale `3.0` collapses B1/u480/u540 into nearly identical effective policies.
    - u480/u540 are not stronger no-bias network policies; B1 beats them strongly without the wrapper.
    - Actor/learner scoring mismatch is large and potentially invalidates casual "B1 pressure" reasoning.
    - More PFSP with official bias `3.0` likely keeps producing wrapper-dominated clone self-play.
- GPT Pro follow-up prompt:
  - Wrote:
    - `notes/gpt_pro_followup_after_bias_matrix_2026-04-27.md`
  - It includes:
    - current artifacts;
    - code excerpts for bias override and matrix runner;
    - all commands and exact results;
    - interpretation;
    - requested next decisions around official eval, bias annealing, B1 anchor definition, actor/learner mismatch, and whether to implement traces/counterfactuals before more training.
- Current recommendation:
  - Do not run more PFSP or server training yet.
  - Next likely work after GPT Pro response:
    - decide whether to re-anchor/evaluate with common bias scales;
    - implement action trace/logit probe;
    - implement minimal forced-action counterfactual;
    - design a training branch that anneals or reduces the heuristic wrapper and gates on low-bias/no-bias B1, not only official bias `3.0`.

## 2026-04-27 - Bias sweep, trace probe, and first forced-action destructive control

- Goal:
  - Continue the GPT Pro diagnostic sequence before any more PFSP/server training.
  - Map common public-bias saturation.
  - Add per-decision trace evidence for raw network logits vs final bias-wrapped logits.
  - Verify that forced legal deviations can alter B1-family paired outcomes.
- Code changed:
  - Added:
    - `python/scripts/b1_bias_sweep_matrix.py`
  - Purpose:
    - Wrapper around `b1_artifact_matrix.py` that runs multiple common public-bias scales and aggregates into:
      - `bias_sweep_summary.json`
  - Extended:
    - `python/scripts/b1_artifact_matrix.py`
  - New trace/intervention flags:
    - `--emit-action-traces`
    - `--trace-top-k`
    - `--trace-max-decisions-per-episode`
    - `--force-pass-seat 0|1`
    - `--force-pass-max-per-episode`
  - Trace rows include:
    - `policy_id`, `pair_index`, `swap_index`, `episode_seed`, `decision_index`, `actor_seat`;
    - selected action with decoded family/label;
    - raw top-k logits with public bias temporarily set to `0.0`;
    - final top-k logits under the active surface;
    - raw-vs-final top action/family match flags;
    - effective public-bias learner/actor scales.
  - Intervention counter:
    - `forced_pass_decisions`
- Validation:
  - `uv run python -m py_compile python/scripts/b1_bias_sweep_matrix.py python/scripts/b1_artifact_matrix.py`
- Bias sweep command:
  - `uv run python python/scripts/b1_bias_sweep_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 8 --artifact-dir-name b1_bias_sweep_missing_mid_p8_20260427 --device cuda:0 --scales 0.5,1.5,2.0,3.0`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_bias_sweep_missing_mid_p8_20260427/bias_sweep_summary.json`
  - Results:
    - Common scale `0.5`:
      - `B1 vs u480`: `1.0`, pair classes `2-0: 8`.
      - `u480 vs B1`: `0.0`, pair classes `0-2: 8`.
      - `B1 vs u540`: `1.0`, pair classes `2-0: 8`.
      - `u540 vs B1`: `0.0`, pair classes `0-2: 8`.
      - `u480 vs u540`: `0.6875`, pair classes `2-0: 4`, `1-1: 3`, `0-2: 1`.
      - `u540 vs u480`: `0.3125`, pair classes `2-0: 1`, `1-1: 3`, `0-2: 4`.
    - Common scale `1.5`:
      - `B1 vs u480`: `0.5`, pair classes `2-0: 1`, `1-1: 6`, `0-2: 1`.
      - `B1 vs u540`: `0.4375`, pair classes `1-1: 7`, `0-2: 1`.
      - `u540 vs B1`: `0.5625`, pair classes `2-0: 1`, `1-1: 7`.
      - Other B1-family matchups cluster near `0.4375`-`0.5`.
    - Common scale `2.0`:
      - Most matchups cluster around `0.4375`-`0.5625`.
      - Pair classes are mostly `1-1: 5-7` with a few `2-0`/`0-2`.
    - Common scale `3.0`:
      - All six B1/u480/u540 ordered matchups return exact `0.5`, pair classes `1-1: 8`.
  - Combined with previous completed surfaces:
    - Scale `0.0`:
      - B1 strongly beats u480/u540.
    - Scale `1.0`:
      - breaks exact invariance but does not show clean u480/u540 advantage over B1.
  - Verdict:
    - The saturation threshold is real.
    - At low common scales (`0.0`/`0.5`), B1 raw/low-bias effective policy is much stronger than u480/u540.
    - At high scale (`3.0`), the wrapper collapses all B1-family outcomes to exact paired `0.50`.
    - Middle scales (`1.0`-`2.0`) expose differences, but do not show a reliable learned-policy win over B1.
- S3 trace probe command:
  - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 1 --artifact-dir-name b1_trace_probe_s3_p1_20260427 --device cuda:0 --matchup B1=u540 --emit-action-traces --trace-top-k 5 --trace-max-decisions-per-episode 80`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_trace_probe_s3_p1_20260427/B1_NoLeague_baseline__vs__u540/action_trace.jsonl`
  - Result:
    - Pair result still `0.5`, pair class `1-1`.
    - Trace rows: `160`.
    - Raw-vs-final top action:
      - match `106`;
      - mismatch `54`.
    - Raw-vs-final top family:
      - match `127`;
      - mismatch `33`.
    - Per policy:
      - B1:
        - action match rate `0.6709`;
        - family match rate `0.8608`.
      - u540:
        - action match rate `0.6543`;
        - family match rate `0.7284`.
    - Examples:
      - B1 raw `main_play_character` became final `pass`.
      - u540 raw `main_move` became final `main_play_character`.
      - B1 raw `main_move` became final `main_play_character`.
  - Verdict:
    - Official S3 wrapper is not a mild tie-breaker.
    - It frequently changes top action and sometimes changes top action family, especially for u540.
- S1 trace probe command:
  - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 1 --artifact-dir-name b1_trace_probe_s1_p1_20260427 --device cuda:0 --public-heuristic-bias-scale 1.0 --matchup B1=u540 --emit-action-traces --trace-top-k 5 --trace-max-decisions-per-episode 80`
  - Artifact:
    - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_trace_probe_s1_p1_20260427/B1_NoLeague_baseline__vs__u540/action_trace.jsonl`
  - Result:
    - Pair result `B1 vs u540 = 1.0`, pair class `2-0: 1`.
    - Trace rows: `160`.
    - Raw-vs-final top action:
      - match `116`;
      - mismatch `44`.
    - Raw-vs-final top family:
      - match `154`;
      - mismatch `6`.
    - Per policy:
      - B1:
        - action match rate `0.7531`;
        - family match rate `0.9877`.
      - u540:
        - action match rate `0.6962`;
        - family match rate `0.9367`.
  - Verdict:
    - S1 still nudges action choice, but family-level overrides are much rarer than S3.
    - This supports using S1 as a diagnostic/training surface because it exposes model differences without fully removing heuristic structure.
- Forced-action destructive control:
  - Command:
    - `uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt --pairs 1 --artifact-dir-name b1_forced_pass_seat0_s3_p1_20260427 --device cuda:0 --matchup B1=u540 --emit-action-traces --trace-top-k 3 --trace-max-decisions-per-episode 80 --force-pass-seat 0 --force-pass-max-per-episode 10`
  - Baseline same seed before forcing:
    - pair 0:
      - swap 0: focal B1 as seat0 wins, physical winner seat `0`;
      - swap 1: focal B1 as seat1 loses, physical winner seat `0`.
  - Forced-pass result:
    - pair 0:
      - swap 0: focal B1 as seat0 loses, physical winner seat `1`;
      - swap 1: focal B1 as seat1 wins, physical winner seat `1`.
    - Pair class remains `1-1`, but the physical winning seat flips from `0/0` to `1/1`.
    - `forced_pass_decisions: 20`.
  - Example forced rows:
    - forced selected action `pass` while final top action was `clock_from_hand(hand_index=0)`;
    - forced selected action `pass` while final top action was `main_play_character(hand_index=0, stage_slot=1)`.
  - Verdict:
    - The intervention path is live.
    - B1-family physical-seat outcomes are not immutable deck fate.
    - Constructive counterfactual search is now justified: legal action deviations can alter terminal winners.
- Current conclusion:
  - We are closer to the actual blocker.
  - The league did not merely have a topology problem; it is also optimizing/evaluating on a saturated wrapper-dominated surface.
  - B1 raw/low-bias policy appears stronger than u480/u540, so the next serious training branch should likely restart from B1 and train with common actor/learner bias parity around S1, while tracking S0/S3.
  - Before training, the next best tool is a constructive forced-action search over losing-seat decisions, using S1/S3 traces to find candidate action families.

## 2026-04-27 - Constructive one-action counterfactual probes from S3 trace

- User question:
  - Whether the next step is another GPT Pro prompt.
- Decision:
  - Not yet.
  - We had enough guidance to run the next concrete diagnostic: constructive counterfactual probes.
- Code changed:
  - Extended `python/scripts/b1_artifact_matrix.py` with generic force-action controls:
    - `--force-action-seat`
    - `--force-action-decision-index`
    - `--force-action-id`
    - `--force-action-swap-index`
  - Added counters:
    - `forced_action_decisions`
    - `forced_action_missed_decisions`
  - Generic forced actions can target one physical seat and optionally one swap.
- Validation:
  - `uv run python -m py_compile python/scripts/b1_artifact_matrix.py`
- Baseline trace used:
  - `runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_trace_probe_s3_p1_20260427/B1_NoLeague_baseline__vs__u540/action_trace.jsonl`
  - Baseline pair:
    - swap 0: focal B1 as seat0 wins; physical winner seat `0`.
    - swap 1: focal B1 as seat1 loses; physical winner seat `0`.
  - Target:
    - swap 0 physical seat `1` (`u540`), the losing physical seat.
- Constructive probes:
  - Ran one forced legal action per probe with:
    - `--pairs 1`
    - `--matchup B1=u540`
    - `--force-action-seat 1`
    - `--force-action-swap-index 0`
  - Candidate set from early S3 trace rows where:
    - raw network top action differed from final wrapper top action; or
    - final/raw top action differed from sampled selected action.
  - Raw-main-move candidates:
    - `decision 15 -> action 415` (`main_move(from_slot=3, to_slot=1)`)
    - `decision 16 -> action 415`
    - `decision 17 -> action 402` (`main_move(from_slot=0, to_slot=1)`)
    - `decision 18 -> action 402`
    - `decision 44 -> action 417` (`main_move(from_slot=3, to_slot=4)`)
  - Top-action candidates:
    - `decision 12 -> action 57` (`clock_from_hand(hand_index=5)`)
    - `decision 13 -> action 128` (`main_play_character(hand_index=5, stage_slot=1)`)
    - `decision 14 -> action 117` (`main_play_character(hand_index=3, stage_slot=0)`)
    - `decision 14 -> action 103` (`main_play_character(hand_index=0, stage_slot=1)`)
    - `decision 15 -> action 117`
    - `decision 16 -> action 117`
    - `decision 20 -> action 475` (`attack(slot=1, attack_type=frontal)`)
    - `decision 21 -> action 475`
- Result:
  - All 13 probes applied exactly one forced action:
    - `forced_action_decisions: 1`
    - `forced_action_missed_decisions: 0`
  - None flipped swap 0:
    - swap 0 remained focal B1 win / physical winner seat `0`.
  - Most probes left decision/tick counts unchanged or close:
    - baseline swap0 `decision_count 133`, `tick_count 416`;
    - raw-main-move probes often `decision_count 134`, `tick_count 414`;
    - `decision 14 -> action 103` gave `decision_count 135`, `tick_count 412`;
    - no terminal winner flip.
- Interpretation:
  - A single plausible constructive deviation by the losing seat is not enough on this seed.
  - This does not contradict the destructive control:
    - ten forced pass blunders for winning seat flipped the physical winner.
  - The next constructive search should use either:
    - multi-decision sequences;
    - short rollout search over top-k action alternatives at multiple high-impact decisions;
    - or repeated rollouts with common random numbers from the same trace prefix.
- Current next step:
  - Build a small constructive counterfactual search script that can apply a sequence of forced actions in one run and score candidate sequences.
  - If multi-action constructive search still finds no flips or positive terminal deltas, then write a new GPT Pro prompt with:
    - bias sweep table;
    - S3/S1 trace summaries;
    - destructive flip result;
    - failed one-action constructive probes.
  - Follow-up executed:
    - Added `--force-action-sequence` support.
    - Tried three multi-action sequences:
      - raw main-move sequence;
      - final top-action sequence;
      - mixed play/attack sequence.
    - None flipped the losing physical seat.
    - The final-top and mixed sequences changed the trajectory substantially:
      - baseline swap0 `decision_count/tick_count = 133/416`;
      - final-top sequence `125/387`;
      - mixed sequence `125/384`;
      - but physical winner stayed seat `0`.
    - Wrote GPT Pro follow-up:
      - `notes/gpt_pro_followup_after_counterfactual_sequences_2026-04-27.md`
    - Prompt asks whether to:
      - build a real state-conditioned counterfactual search;
      - switch to B1/S1 retraining;
      - protect B1 raw/S0 with auxiliary loss;
      - discard u480/u540 as main parents.

## 2026-04-27 - B1 restart with common S1 bias parity: implemented and discarded as gate-only branch

- Implemented a new B1-from-u450 S1 retraining preset:
  - `configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml`
  - Parent checkpoint:
    - `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt`
  - Main config intent:
    - current learner public heuristic bias scale: `1.0`
    - current actor public heuristic bias scale: `1.0`
    - final public heuristic bias scale: `1.0`
    - exact-action reference BC: `0.0`
    - small B1 family rail: `reference_policy_top_action_family_bc_coef: 0.03`
    - public heuristic teacher loss: `0.0`
    - LR: `1e-5`
    - entropy: `0.04 -> 0.02`
    - fixed S1 branch sampling:
      - B1 no-league baseline: `0.50`
      - heuristic public: `0.10`
      - heuristic variant: `0.15`
      - mirror: `0.25`
      - recent/champion/warmup/hard-negative: `0.0`
    - promotion and checkpoint guard disabled for this directional diagnostic branch.
- Added surface labeling to matrix diagnostics:
  - `python/scripts/b1_artifact_matrix.py`
  - new `--surface-name` argument is persisted into:
    - `policy_load_manifest.json`
    - per-matchup `evaluation_context`
    - `matrix_summary.json`
- Found and fixed a real startup wiring bug:
  - Symptom:
    - first smoke failed attaching `reference_policy_id: b1_noleague_baseline`
    - missing weights:
      - `training/snapshots/b1_noleague_baseline/weights.pt`
  - Cause:
    - `_ensure_noleague_baseline_anchor(...)` imported the B1 anchor when promotion gating required B1, but not when B1 was needed only as a frozen reference policy.
  - Fix:
    - `python/scripts/train.py`
    - `_ensure_noleague_baseline_anchor(...)` now treats B1 reference-policy BC/family-BC/B1-opponent reference BC as requiring the baseline anchor import.
  - Regression test:
    - `python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion`
- Validation:
  - `uv run python -m py_compile python/scripts/train.py python/scripts/b1_artifact_matrix.py`
  - `uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume --tb=short`
  - Result: passed.
- Smoke run:
  - Failed first attempt:
    - `runs/b1_s1_retrain_smoke_u450_to_u452_20260427`
    - failed before training because B1 reference weights were not imported.
  - Fixed smoke:
    - `runs/b1_s1_retrain_smoke_u450_to_u452_fix1_20260427`
    - completed checkpoints `451` and `452`
    - imported B1 baseline anchor and attached frozen reference:
      - `coef=0`
      - `family_coef=0.03`
    - no NaNs or timeout/fault rows in inspected tail.
  - Sampler issue found after smoke:
    - lane weights originally summed to `0.90`, leaving implicit recent weight `0.10`.
    - That made imported B1-history snapshots visible as recent pool members, although recent envs were not sampled in the inspected tail.
    - Fixed config by increasing mirror mix from `0.15` to `0.25` so fixed lanes sum to `1.0`.
- Main short branch:
  - Run:
    - `runs/b1_s1_retrain_u450_to_u460_fix2_20260427`
  - Command:
    - `uv run python python/scripts/train.py --stack-config configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml --run-label b1_s1_retrain_u450_to_u460_fix2_20260427 --runtime-mode train_async_fast --autoscale --hardware-profile local --resume-from runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt --resume-allow-config-mismatch --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --max-updates 460 --checkpoint-interval-updates 5 --profile-timers`
  - Completed checkpoints:
    - `checkpoint_455.pt`
    - `checkpoint_460.pt`
  - Runtime tail confirmed intended lane weights:
    - `pfsp_sampling_weight_noleague_baseline: 0.5`
    - `pfsp_sampling_weight_heuristic_public: 0.1`
    - `pfsp_sampling_weight_heuristic_public_variant: 0.15`
    - `pfsp_sampling_weight_mirror: 0.25`
    - `pfsp_sampling_weight_recent: 0.0`
    - `pfsp_sampling_weight_champion: 0.0`
    - `pfsp_sampling_weight_warmup_snapshot: 0.0`
    - `pfsp_sampling_weight_hard_negative: 0.0`
    - B1 pressure live:
      - tail `collector_b1_opponent_env_steps`: about `8k` per update
      - tail `collector_b1_opponent_train_rows`: about `4k` per update
  - Training tail:
    - u460 loss `0.373167`
    - policy loss `0.007907`
    - value loss `0.005522`
    - entropy `0.615247`
    - reference family BC active coefficient `0.025714`
    - vtrace p99 spiked at u460 (`353750.8`), likely update/checkpoint lag/off-policy boundary; p95 also spiked (`314.3`), so monitor if this branch is revisited.
- Multi-surface evals for `checkpoint_460.pt`:
  - S1 low-bias, 16 pairs:
    - artifact:
      - `runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s1_p16_20260427`
    - `B1 -> u460`: `0.50`, pair classes `1-1:16`
    - `u460 -> B1`: `0.50`, pair classes `1-1:16`
    - `u460 -> B3 HeuristicPublicAggro`: `0.50`, pair classes `1-1:16`
    - `u460 -> B4 HeuristicPublicControl`: `0.6875`, pair classes `2-0:6, 1-1:10`
  - S0 raw/no-bias, 8 pairs:
    - artifact:
      - `runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s0_p8_20260427`
    - `B1 -> u460`: `0.6875`
    - `u460 -> B1`: `0.1875`
    - `u460 -> B3`: `0.0`
    - `u460 -> B4`: `0.0`
  - S3 official/deployment wrapper, 8 pairs:
    - artifact:
      - `runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s3_p8_20260427`
    - `B1 -> u460`: `0.50`, pair classes `1-1:8`
    - `u460 -> B1`: `0.50`, pair classes `1-1:8`
    - `u460 -> B3`: `1.0`
    - `u460 -> B4`: `1.0`
- Verdict:
  - This branch is mechanically correct and useful as a negative result, but not a candidate.
  - It did not create S1 B1 movement:
    - still exact `0.50` with all split pairs on the intended S1 surface.
  - It did not protect raw/S0 behavior:
    - B1 still clearly beats u460 raw;
    - u460 raw collapses against B3/B4.
  - S3 remains saturated and looks good only because wrapper bias `3.0` dominates.
  - Therefore, simple B1->S1 retraining with gate-only raw protection is not enough.
- Next hypotheses:
  - Need true raw-body preservation, not just a small family rail against a reference model loaded with checkpoint guidance.
  - Current reference BC/family BC likely trains against the reference policy's effective logits and the student's current effective logits; it is not a true raw/S0 KL between B1 raw and student raw.
  - Stronger or lower-LR S1 training without true raw preservation may merely slow the same drift.
  - The next useful patch is either:
    - implement explicit raw/S0 reference distillation over legal actions/top-k actions; or
    - feed prefix-replay counterfactual positive labels into an exploiter branch.
  - A new GPT Pro prompt was written with this result and asks for guidance on exact raw-distill implementation vs counterfactual-label priority.
## 2026-04-27 - True raw/S0 B1 distillation implemented, but local B1/S1 retraining still does not improve

Context:
- Followed the GPT Pro recommendation to stop extending u480/u540 and restart from the stronger B1 parent:
  - `runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt`
- Goal was to test whether the previous B1/S1 restart failed because "rawprotect" was only a gate/family rail and not true raw/S0 B1 preservation.
- Implemented real learner-side raw/S0 B1 distillation:
  - `python/weiss_rl/learners/impala_learner.py`
    - added `raw_b1_distill_*` fields to `ImpalaLearner`;
    - temporarily sets teacher and student public heuristic bias to configured raw scales, normally `0.0`;
    - restores previous learner/actor bias scales after the auxiliary forward;
    - computes legal-action packed KL on raw/S0 logits;
    - logs `raw_b1_distill_loss`, `raw_b1_kl`, `raw_b1_top_action_ce`, `raw_b1_top1_match`, `raw_b1_topk_overlap`, `raw_b1_family_match`, bias scales, temperature, top-k, row fraction;
    - later added optional raw teacher top-action CE because soft KL preserved top-k but not game outcomes.
  - `python/weiss_rl/config/models.py`
    - added `TrainingRawB1DistillConfig`.
  - `python/weiss_rl/config/parse.py`
    - parses `training.raw_b1_distill`.
  - `python/scripts/train.py`
    - passes raw distill config into learner;
    - schedules `raw_b1_distill_coef`;
    - attaches the frozen B1 reference model when raw distill is enabled even if normal reference BC is off;
    - imports the B1 baseline anchor for raw distill.
  - `python/weiss_rl/tests/test_impala_learner.py`
    - added tests proving no-bias teacher/student forward is used and original public-bias scales are restored;
    - added test that perturbed student raw logits increase the distill loss.

Validation:
- `uv run python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py`
- `uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_uses_zero_bias_and_restores_scales python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_penalizes_perturbed_student_raw_logits --tb=short`
- Focused regression tests also passed earlier:
  - `test_impala_learner_reference_policy_top_action_bc_coef_adds_reference_nll`
  - `test_impala_learner_b1_opponent_reference_bc_uses_b1_mask_only`
  - `test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion`

Configs added:
- `configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill.yaml`
  - common S1 actor/learner public-bias parity `1.0`;
  - exact action/family reference BC off;
  - `raw_b1_distill.coef: 0.05`, `final_coef: 0.02`, raw teacher/student bias `0.0`.
- `configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill_strong.yaml`
  - same but `coef: 2.0`, `final_coef: 1.0`.
- `configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill_topce.yaml`
  - same but `coef: 1.0`, `final_coef: 0.5`, `top_action_ce_coef: 1.0`.

Runs and metrics:
- Weak KL raw distill:
  - `runs/b1_s1_rawdistill_u450_to_u455_smoke_20260427`
  - `runs/b1_s1_rawdistill_u455_to_u460_20260427`
  - u460 training metrics:
    - `raw_b1_distill_loss=0.00510567`
    - `raw_b1_distill_coef=0.0485`
    - `raw_b1_top1_match=0.9363`
    - `raw_b1_topk_overlap=0.9965`
    - `raw_b1_family_match=0.9823`
  - S1 p8:
    - B1 vs u460: `0.50`, all `1-1`
    - u460 vs B1: `0.50`, all `1-1`
    - u460 vs B3: `0.50`, all `1-1`
    - u460 vs B4: `0.625`, `2-0:2`, `1-1:6`
  - S0 p8:
    - B1 vs u460: `0.6875`
    - u460 vs B1: `0.1875`
    - u460 vs B3: `0.0`
    - u460 vs B4: `0.0`
  - S3 p8:
    - B1/u460 both directions: `0.50`, all `1-1`
    - u460 vs B3/B4: `1.0`
  - Verdict: mechanically correct but not a learning improvement; raw body still collapses.
- Strong KL raw distill:
  - `runs/b1_s1_rawdistillstrong_u450_to_u460_20260427`
  - u460 training metrics:
    - `raw_b1_distill_loss=0.0037669`
    - `raw_b1_distill_coef=1.95`
    - `raw_b1_top1_match=0.9384`
    - `raw_b1_topk_overlap=0.9967`
    - `raw_b1_family_match=0.9824`
  - S0 p8:
    - B1 vs u460: `0.625`
    - u460 vs B1: `0.1875`
    - u460 vs B3: `0.0`
    - u460 vs B4: `0.0`
  - S1 p8:
    - B1/u460 both directions: `0.50`, all `1-1`
    - u460 vs B3: `0.50`, all `1-1`
    - u460 vs B4: `0.625`
  - Verdict: not enough; stronger KL slightly changes raw reverse matchup but does not preserve candidate raw strength or move S1.
- KL + raw top-action CE:
  - `runs/b1_s1_rawdistilltopce_u450_to_u460_20260427`
  - u460 training metrics:
    - `raw_b1_distill_loss=0.9749`
    - `raw_b1_distill_coef=0.975`
    - `raw_b1_distill_top_action_ce_coef=1.0`
    - `raw_b1_kl=0.00870`
    - `raw_b1_top_action_ce=0.9662`
    - `raw_b1_top1_match=0.9158`
    - `raw_b1_topk_overlap=0.9966`
    - `raw_b1_family_match=0.9760`
  - S0 p8:
    - B1 vs u460: `0.6875`
    - u460 vs B1: `0.25`
    - u460 vs B3: `0.0`
    - u460 vs B4: `0.0`
  - S1 p8:
    - B1/u460 both directions: `0.50`, all `1-1`
    - u460 vs B3: `0.50`, all `1-1`
    - u460 vs B4: `0.625`
  - Verdict: marginal raw B1 focal improvement from `0.1875` to `0.25`, but still not useful; top-action CE appears to fight the S1 RL pressure without creating game-level improvement.

Current interpretation:
- True raw/S0 distillation is now implemented and verified, but auxiliary preservation alone is not enough.
- The S1 self-play/RL objective continues to create small raw decision shifts that are catastrophic on S0 and still do not improve S1 B1.
- S3 remains saturated: it reports B1 parity and heuristic wins even when S0 says the learned body is weak.
- The next useful lever is likely not another raw-distill coefficient tweak. We need either:
  - state-conditioned counterfactual positive labels under S1;
  - a direct supervised/rollout best-response dataset;
  - a much lower-change objective that freezes raw B1 and only updates selected heads/states;
  - or a more radical league/topology/objective redesign where the main policy and exploiters are separated and evaluated on surface-specific gates.

Next hypothesis:
- Build the prefix-replay S1 counterfactual search from the prior Pro plan and look for positive deviations. If no positive deviations are found, ordinary S1 RL from B1 is not providing a usable best-response signal. If positives are found, train a B1 exploiter with those labels rather than continuing generic self-play.

## 2026-04-27 - No-RL raw-distill-only control separates distill mechanics from destructive RL drift

Implemented a reusable `training.policy_loss_coef` switch and a control preset:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_distillonly_control.yaml
```

The control resumes B1 checkpoint 450, keeps rollout/eval on the common S1 surface, disables the RL policy loss, value loss, entropy, reference BC, behavior BC, and structured teacher auxiliary losses, and leaves only true raw/S0 B1 distillation active:

```text
policy_loss_coef = 0.0
value_loss_coef  = 0.0
entropy_coef     = 0.0
raw_b1_distill   = enabled, coef 1.0, teacher/student bias 0.0
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py
uv run pytest -q python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_policy_loss_coef_can_disable_rl_policy_loss python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_uses_zero_bias_and_restores_scales python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_penalizes_perturbed_student_raw_logits --tb=short
```

Result: `3 passed`.

Control run:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427
```

Final training metrics:

```text
loss                         9.99e-7
policy_loss                  9.85e-4
policy_loss_coef             0.0
value_loss                   0.0102
entropy                      0.6076
entropy_coef                 0.0
raw_b1_distill_loss/raw_kl    9.99e-7
raw_b1_top1_match            0.9789
raw_b1_topk_overlap          0.9997
raw_b1_family_match          0.9926
teacher_aux_loss             0.0
```

Parameter movement from parent B1 checkpoint 450 to distill-only u455:

```text
float params compared  2,965,340
L2 distance            0.00220
max abs delta          0.000102
changed tensors        187 / 291
```

Tiny matrix artifacts:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s0_p8_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s1_p8_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s3_p8_20260427
```

S0 p8:

```text
b1parent vs B1  0.625
B1 vs b1parent  0.5625
u455 vs B1      0.625
B1 vs u455      0.3125
b1parent vs B3  0.0
u455 vs B3      0.0
b1parent vs B4  0.0
u455 vs B4      0.0
```

S1 p8:

```text
b1parent vs B1  0.5625
B1 vs b1parent  0.4375
u455 vs B1      0.5
B1 vs u455      0.5
b1parent vs B3  0.5
u455 vs B3      0.5
b1parent vs B4  0.5625
u455 vs B4      0.625
```

S3 p8:

```text
b1parent vs B1  0.5
B1 vs b1parent  0.5
u455 vs B1      0.5
B1 vs u455      0.5
b1parent vs B3  0.9375
u455 vs B3      1.0
b1parent vs B4  1.0
u455 vs B4      1.0
```

Interpretation:

```text
The no-RL/distill-only branch did not reproduce the catastrophic raw/S0 closed-loop weakness seen in the normal S1 RL branches. It stayed extremely close to B1 in parameter space and retained essentially the same broad S0/S1/S3 profile as the parent on this tiny matrix.

That separates the failure mode: raw distillation plumbing is probably not the main bug. Generic S1 RL/self-play updates are the destructive component. They can move a few important decisions enough to hurt closed-loop raw play, while still failing to discover positive B1 best-response deviations.

This supports stopping coefficient tweaks and pivoting to state-conditioned positive labels plus an identity-preserving policy form: frozen B1 plus a trainable residual deviation/exploiter head.
```

## 2026-04-27 counterfactual-label smoke

Implemented a first-pass S1 counterfactual label path:

```text
python/scripts/b1_artifact_matrix.py
  + emits legal_ids in action traces
  + supports --force-action-pair-index for isolated replay interventions

python/scripts/b1_counterfactual_labels.py
  + runs S1 B1-vs-B1 baseline traces
  + targets losing physical-seat decisions
  + replays one forced legal action at a time
  + writes counterfactual_labels.jsonl and counterfactual_summary.json
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_artifact_matrix.py python/scripts/b1_counterfactual_labels.py
```

S1 destructive-control artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_destructive_pass_s1_smoke_p1_20260427
```

Baseline pair 0 had physical seat 0 winning both swaps. Forcing physical seat 0 to pass up to 3 times produced physical seat 1 winning both swaps:

```text
forced_pass_decisions = 6
baseline winner_seat pattern = 0 / 0
forced-pass winner_seat pattern = 1 / 1
```

So the S1 intervention path is live; outcomes are not immutable deck/seat fate.

Constructive one-step search artifacts:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_smoke_p1_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_focused_p1_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_focused_p3_20260427
```

Best focused run:

```text
pairs                  3
trace_rows             480
target_states          6
attempted replays      18
forced_misses          0
winner_flip_labels     0
target families        clock/main/attack/level-up/encore
```

Interpretation:

```text
The tooling works, but cheap one-step top-k/family/pass alternatives did not find positive B1-best-response labels in the small local budget. Do not read this as "B1 is unexploitable"; the destructive control shows interventions can flip outcomes. The next bottleneck is search depth and throughput: subprocess-per-action is too slow, and one-step constructive moves are probably too weak. Next credible patch is an in-process counterfactual runner and/or a small two-step beam over high-impact same-seat decisions.
```

## 2026-04-27 terminal-margin counterfactual patch

Implemented the next search-signal patch:

```text
python/weiss_rl/eval/simulator_runner.py
python/weiss_rl/eval/harness.py
python/scripts/eval.py
python/scripts/b2_disagreement_audit.py
python/scripts/b1_artifact_matrix.py
python/scripts/b1_counterfactual_labels.py
```

Changes:

```text
1. Eval GameResult/EvalGameRecord now carry terminal_summary.
2. SimulatorEvalRunner derives terminal per-seat summaries from final observation rows:
   level, clock, deck, hand, stock, waiting room, memory, climax, resolution, stage_count.
3. b1_artifact_matrix pair_table/episodes now include terminal_summary.
4. b1_counterfactual_labels now computes baseline_target_score, forced_target_score, score_delta,
   margin_positive, label_weight, and writes all forced trials to counterfactual_trials.jsonl.
5. In-process counterfactual search now replays only the targeted scheduled game for each forced action,
   instead of rerunning a full mini matrix per candidate.
6. Target selection is ranked by later/swing-ish decisions, high-impact family, raw/final disagreement,
   and low final-logit margin.
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/eval/harness.py python/weiss_rl/eval/simulator_runner.py python/scripts/b1_artifact_matrix.py python/scripts/b1_counterfactual_labels.py python/scripts/eval.py python/scripts/b2_disagreement_audit.py
```

Terminal-summary smoke:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_terminal_summary_smoke_s1_p1_20260427b
```

The pair table now includes `seat0`/`seat1` terminal summaries. The mapping source is currently:

```text
seat_mapping_source = last_acting_seat_perspective_fallback
```

because terminal batches do not expose a valid actor seat. This fallback matched the observed winner/loser terminal levels in the smoke.

Destructive-control margin check:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_destructive_pass_s1_margin_p1_20260427
```

Using the same terminal score from the losing-seat perspective:

```text
baseline seat1 score  = -1.140
forced-pass score     =  1.305
score_delta           = +2.445
winner flipped        = seat0 -> seat1
```

So the margin channel reacts strongly and consistently to a known live intervention.

Constructive ranked S1 search:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_ranked_margin_p3_t6_a4_20260427b
```

Result:

```text
pairs                  3
trace_rows             914
target_states          6
attempted replays      24
forced_misses          0
winner_flip_labels     1
margin_positive_labels 1
max score_delta        +2.165
```

Positive label:

```text
pair_index      0
swap_index      0
target_seat     1
decision_index  20
baseline action main_play_character(hand_index=0, stage_slot=3)
forced action   pass
baseline winner seat0
forced winner   seat1
score_delta     +2.165
```

Larger local search:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_ranked_margin_p8_t16_a4_20260427
```

Result:

```text
pairs                  8
trace_rows             2476
target_states          16
attempted replays      64
forced_misses          0
winner_flip_labels     1
margin_positive_labels 1
max score_delta        +2.165
```

Interpretation:

```text
This is no longer another dead end. We have a constructive one-step B1/S1 counterfactual that flips a losing-seat outcome. The label is sparse, but it proves the search direction can produce positive best-response evidence. Do not train a residual on one label as a final result yet. Next best patch is to improve label yield: broaden candidate actions, add family representatives/canonical dedupe, and then add a tiny two-step beam around high-ranked near-positive states.
```

## 2026-04-27 candidate reps and two-step beam

Implemented candidate-source metadata and family representative generation in:

```text
python/scripts/b1_counterfactual_labels.py
```

Candidate rows now include:

```text
candidate_action.action_id
candidate_action.family
candidate_action.label
candidate_action.candidate_sources
candidate_action.candidate_source_ranks
candidate_action.candidate_logits
```

The search now adds:

```text
pass_alternative
raw_topk_no_public_bias
final_topk
family_representative
```

Target ranking was fixed to use the actual action family names:

```text
main_play_event
climax_play
pass
```

instead of generic `event`/`climax`, and selected-pass states are now allowed by the default target-family list.

Candidate-rep smoke:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_candidate_reps_smoke_p3_t6_a10_20260427
```

Result:

```text
pairs                  3
target_states          6
attempted replays      52
forced_misses          0
winner_flip_labels     1
margin_positive_labels 1
```

Candidate source coverage:

```text
final_topk              41
raw_topk_no_public_bias 36
family_representative   28
pass_alternative         5
```

Larger one-step packet:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_candidate_reps_p8_t24_a6_20260427
```

Result:

```text
pairs                  8
target_states          24
attempted replays      120
forced_misses          0
winner_flip_labels     1
margin_positive_labels 1
```

Near-positive one-step states included:

```text
pair 4 swap 0 decision 133 main_move(from_slot=1,to_slot=2): score_delta +0.125
pair 6 swap 1 decision 158 main_move(from_slot=1,to_slot=2): score_delta +0.095
```

Implemented a small two-step beam:

```text
--two-step-beam-targets
--two-step-min-first-delta
--two-step-window
--two-step-second-actions
--two-step-max-replays
--two-step-include-positive-first
```

Mechanism:

```text
1. Select near-positive first-step trials.
2. Replay first forced action with an action trace.
3. Choose the next same-seat high-impact decision from the altered trajectory.
4. Replay with both forced actions.
```

Two-step sanity smoke that included positive first steps:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_twostep_smoke_p3_t6_a6_20260427
```

Result:

```text
one_step positives 1
two_step positives 4
```

However, all two-step positives shared the same first action:

```text
pair 0 swap 0 decision 20: force pass
```

So this was a machinery sanity check, not diverse data.

Two-step near-miss packet, excluding already-positive first steps:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_twostep_nearmiss_p8_t24_a6_20260427
```

Result:

```text
one_step attempted      120
two_step attempted      16
forced_misses           0
winner_flip_labels      1
margin_positive_labels  1
```

The two-step near-miss attempts did not improve beyond the first-step margin. The selected second decisions were mostly level-up choices and preserved the first-step delta:

```text
pair 4 swap 0 d133 main_move -> d149 level_up choices: score_delta stayed +0.125
pair 6 swap 1 d158 main_move -> d171 level_up choices: score_delta stayed +0.095
```

Interpretation:

```text
The counterfactual path has found a real constructive B1/S1 exploit, but label yield remains sparse.
Broad family representatives helped artifact quality, not yield.
The naive two-step beam is mechanically correct but needs better second-step selection:
  skip low-impact level-up-only branches when they do not alter margin,
  prefer subsequent main/attack/climax decisions,
  or run a phase-aware second-step queue rather than the first high-ranked next decision.
```

Next hypotheses:

```text
1. Add phase-aware second-step selection that prefers main/attack/climax over level-up when first-step score is only a near miss.
2. Add target diversity constraints so one seed/action does not dominate labels.
3. If positives remain sparse, train a tiny residual proof only on the one strong pass label as a controlled memorization/adoption test, not as a thesis result.
```

Implemented phase-aware second-step filtering:

```text
--two-step-target-families main_play_character,main_move,main_play_event,climax_play,attack,pass
```

Phase-aware packet:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_twostep_phaseaware_p8_t24_a6_20260427
```

Result:

```text
one_step attempted      120
two_step attempted      6
forced_misses           0
winner_flip_labels      1
margin_positive_labels  1
```

The second-step candidates now correctly targeted attack/pass follow-ups instead of level-up variants, but still did not improve beyond the first-step near-miss margins:

```text
pair 4 swap 0 d133 main_move -> d137 attack/pass: delta stayed around +0.075 to +0.080
pair 6 swap 1 d158 main_move -> d162 attack/pass: delta fell to +0.045
```

Updated interpretation:

```text
The current one-step search has found one strong B1/S1 exploit. The current two-step beam is mechanically valid but does not yet expand the positive-label set. The bottleneck is now target diversity/search breadth, not basic intervention plumbing.
```

Practical next decision:

```text
Either:
  A. push a wider search over more seeds and more target states to see if one-step pass/main deviations recur, or
  B. build the frozen-B1 residual supervised memorization proof on the single strong label plus synthetic/oracle labels,
     while continuing label search separately.

Under time pressure, B is now reasonable as a model-interface proof, but it should be framed as adoption/memorization evidence, not yet a B1 exploiter result.
```

## 2026-04-27 tensor capture + residual adoption proof

Implemented tensor capture for counterfactual labels:

```text
python/scripts/b1_artifact_matrix.py
  --emit-trace-tensors adds obs_float32, obs_sha256, final_legal_logits, raw_legal_logits_no_public_bias to model action traces.

python/scripts/b1_counterfactual_labels.py
  counterfactual_labels.jsonl positive rows now get tensor_ref.
  states/cf_XXXXXX.pt stores obs, actor/target seat, legal_ids, baseline_action_id, positive_action_id,
  base_s1_legal_logits, raw/base_s0 legal logits, terminal summaries, score_delta, and label metadata.
```

Tensorized rerun of the known candidate-rep search:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_candidate_reps_p8_t24_a6_tensor_20260427
```

Result:

```text
pairs                  8
target_states          24
attempted replays      120
forced_misses          0
winner_flip_labels     1
margin_positive_labels 1
score_delta max        +2.165
tensor_ref             states/cf_000000.pt
```

Tensor sanity:

```text
obs shape        (378,) float32
legal_ids shape  (32,) int64
positive action  51 pass, legal
baseline action  105 main_play_character(hand_index=0, stage_slot=3), legal
base_s1 logits   (32,) float32
```

Implemented standalone residual adoption probe:

```text
python/scripts/b1_residual_adoption_probe.py
```

Residual proof artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_cf1_20260427/residual_adoption_report.json
```

Result:

```text
residual_zero_logit_max_abs_diff_vs_stored_b1_s1 = 0.0
base_trainable_parameter_count                  = 0
base_param_delta_l2                              = 0.0
train_loss_initial                              = 20.9687
train_loss_final                                = 0.000456
base P(pass)                                    = 3.09e-8
trained residual P(pass)                        = 0.999961
base top action                                 = 421
trained top action                              = 51 pass
base pass-vs-baseline margin                    = -16.375
trained pass-vs-baseline margin                 = +19.857
adoption_rate                                   = 1/1
```

Interpretation:

```text
This proves the stored-logit frozen-B1 residual interface can adopt the discovered S1 counterfactual action without
changing any base parameters. It is a residual adoption/memorization proof, not a B1 exploiter claim.
```

Next hypotheses:

```text
1. Add natural-replay integration so the residual policy can choose the labelled pass during simulator eval rather than only offline CE.
2. Add unlabelled trace-probe drift metrics before any broader use: top-action change rate and family change rate.
3. Scale tensorized one-step search only after the residual path is runnable in eval, targeting >=10 flip labels or >=30 margin positives.
```

### 2026-04-27 - Closed-loop residual bridge probe

Implemented a live simulator bridge for the frozen-B1 residual:

```text
python/scripts/b1_residual_closed_loop_eval.py
```

The script loads the residual head from `b1_residual_adoption_probe.py`, wraps a live frozen B1 model, runs
seat-swapped S1 eval through the normal simulator runner, emits action traces/pair tables, and probes whether
the stored counterfactual label state is reached and adopted.

Important fixes discovered while looping:

```text
1. Index-only label matching was misleading. Decision 20 could be a trivial one-legal-action pass in a shifted trajectory.
   The report now requires the stored legal-action fingerprint before calling the label matched.

2. action_rng_salt_mode="shared" still inherited the base simulator seed, which includes seat_policy_id.
   Added action_rng_salt_mode="physical" to b1_artifact_matrix.py for policy-invariant seat RNG diagnostics.

3. The label came from B1-vs-B1 swap 0 with the target in physical seat 1.
   Residual-as-focal places the residual in physical seat 1 on swap 1, so the replay target was wrong.
   Added --residual-as-opponent so B1 is focal and residual is opponent, putting the residual in seat 1 on swap 0.

4. Added --alias-residual-rng-to-b1 for exact B1 prefix replay under the original stochastic action RNG.
```

Diagnostic artifacts:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_greedy_cf1_strict_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_sampled_cf1_strict_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_gated_cf1_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_gated_physical_cf1_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_gated_aliasrng_cf1_20260427
```

These showed that ungated/global residual changes early trajectory, and that wrong orientation/RNG can make the
stored label state appear under B1 rather than the residual. They are useful negative controls, not success claims.

Successful bridge artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_gated_opponent_aliasrng_cf1_20260427/closed_loop_report.json
```

Key result:

```text
surface                         lowbias_s1
matchup                         B1 NoLeague baseline vs B1 residual S1
residual_as_opponent             true
alias_residual_rng_to_b1          true
gate_to_label_obs                 true
residual_applied_rows             1
residual_suppressed_rows          160
label matched by legal fingerprint true
expected positive action          51 pass
selected action at label          51 pass
pair class                        0-2 from B1 focal perspective
B1 focal wins/losses              0 / 2
```

Interpretation:

```text
This is the first closed-loop bridge from:
  counterfactual S1 label -> frozen-B1 residual -> live simulator action adoption -> paired-seed win effect.

It is still not a general B1 exploiter. It is a one-label, gated-prefix proof. The important evidence is that the
residual can be inserted into the real eval runner without base drift and can reproduce the positive counterfactual
when the exact labelled state is reached.
```

Next hypotheses:

```text
1. Scale tensorized counterfactual search to produce more labels, now that the residual bridge is proven.
2. Train/evaluate a multi-label residual with drift metrics:
   residual application count, unlabelled top-action/family change rate, label adoption rate, pair_score.
3. Add a less brittle activation rule than exact obs hash, probably legal_fingerprint + phase/seat/window or a small
   classifier over stored label neighborhoods, before calling it a natural exploiter.
4. Keep full-model RL paused. The working path is counterfactual labels plus constrained residual first.
```

### 2026-04-27 - Counterfactual search runtime correction

Tried to scale the in-process S1 label search too aggressively:

```text
artifact_dir_name b1_cf_labels_s1_scaled_p8_t80_a16_tensor_20260427
pairs             8
max_target_states 80
max_actions       16
max_forced        1000
```

This was interrupted after roughly 30 minutes. The cause is not a hang: each candidate action is a full simulator
replay to terminal, so a 1000-replay cap is expensive. The partial run only wrote one tensor state and no final
summary because the script flushed useful JSON only at the end.

Patched `python/scripts/b1_counterfactual_labels.py`:

```text
--progress-every N
--stop-after-positive-labels N
counterfactual_progress.json
incremental counterfactual_labels.jsonl / counterfactual_trials.jsonl flushing
```

Validation smoke:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_smoke_stop1_p4_t12_a6_20260427
```

Result:

```text
runtime                  ~64s
pairs                    4
target_states            12
attempted_forced_replays 1
forced_misses            0
winner_flip_labels       1
margin_positive_labels   1
score_delta max          +2.165
stop_after_positive      1
```

Next search policy:

```text
Do not run blind 1000-replay searches interactively.
Use staged batches with progress flushing:
  4-8 pairs
  20-40 target states
  6-10 actions
  100-250 max forced replays
  progress every 10-25
  stop after 5-10 positives
Then train/evaluate a multi-label residual only when the label set actually grows.
```

### 2026-04-27 - Fast-flag correction for pass-overextension mining

Pro review correctly warned that the stop-after-positive smoke only rediscovered the known label. Added exclusion and
targeting switches to `python/scripts/b1_counterfactual_labels.py`:

```text
--exclude-labels
--exclude-pair-index
--exclude-legal-fingerprint
--exclude-action-pair
--require-pass-legal
--require-baseline-family
--randomize-target-order
--target-random-seed
```

First attempted excluded-known search used the wrong interactive scale:

```text
artifact_dir_name b1_cf_labels_s1_exclude_known_passover_p8_t24_a8_stop3_20260427
max_actions_per_state 8
attempted_forced_replays 96 before interruption
positive_labels 0
score_delta max +0.08
```

Why it was slow:

```text
The known useful pattern is main_play_character -> pass.
With --max-actions-per-state 8, the script tried pass plus seven other full-game forced rollouts per target.
Each candidate is a replay-to-terminal, so this was the wrong interactive flag.
```

Corrected fast pass:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_exclude_known_passonly_p8_t24_a1_stop3_20260427
```

Flags:

```text
--exclude-labels previous_known_label_jsonl
--require-pass-legal
--require-baseline-family main_play_character
--max-actions-per-state 1
--max-forced-replays 30
--stop-after-positive-labels 3
```

Result:

```text
runtime                  ~4.2 min
pairs                    8
target_states            21
attempted_forced_replays 21
forced_misses            0
positive_labels          0
winner_flip_labels       0
margin_positive_labels   0
score_delta max          -0.005
score_delta mean         -0.145
```

Interpretation:

```text
After excluding the known pair/fingerprint/action-pair, simple pass-overextension is not abundant in the next 7 pairs.
The first label is real, but it is not a high-yield trivial pattern. Further mining should either:
  1. use more seeds with pass-only, still cheap;
  2. target different action families with max_actions kept tiny;
  3. move to small two-step beams only for near-positive states.
```

### 2026-04-27 - First ungated two-label residual B1-exploiter signal

Ran a different-family fast search instead of asking for another prompt:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_mainmove_no_pass_p8_t20_a2_stop1_20260427
```

Flags:

```text
--exclude-labels previous_known_label_jsonl
--pass-action-id -1
--require-baseline-family main_move
--max-target-states 20
--max-actions-per-state 2
--max-forced-replays 40
--stop-after-positive-labels 1
```

Result:

```text
attempted_forced_replays 39
forced_misses            0
positive_labels          1
winner_flip_labels       1
score_delta max          +2.16
```

New label:

```text
pair_index          6
swap_index          1
target_seat         0
decision_index      106
baseline action     412 main_move(from_slot=2, to_slot=3)
positive action     402 main_move(from_slot=0, to_slot=1)
legal fingerprint   f27e229c50577a76d3152c7d7f96c86b07a0c5daf33170d9b62ec292d6e9f2c0
winner flipped      true
score_delta         +2.16
```

Patched `python/scripts/b1_residual_adoption_probe.py` so `--label-dir` can be repeated, then trained a two-label
residual:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_2label_pass_mainmove_20260427
```

Result:

```text
labels                  2
adoption_count          2
adoption_rate           1.0
base_param_delta_l2     0.0
identity max abs diff   0.0
```

Exact-gated live bridge for the new main-move label:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_gated_mainmove_20260427
```

Result:

```text
residual_applied_rows 1
selected action       402 main_move(from_slot=0, to_slot=1)
B1 focal wins/losses  7 / 9
pair classes          1 x 0-2, 7 x 1-1
```

Ungated two-label residual with alpha=1.0 overreached:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_ungated_physical_p8_20260427
B1 focal wins/losses 13 / 3
```

Trained a more conservative two-label residual:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_2label_conservative_a01_l2_20260427
alpha              0.1
lr                 0.001
residual_l2_coef   0.001
adoption_rate      2/2
```

Ungated physical S1 eval, 8 pairs:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_conservative_ungated_physical_p8_20260427
B1 focal wins/losses 4 / 12
pair classes         4 x 0-2, 4 x 1-1, 0 x 2-0
residual side score  12 / 16 = 0.75
```

Ungated physical S1 confirm, 16 pairs:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_conservative_ungated_physical_p16_20260427
B1 focal wins/losses 7 / 25
pair classes         9 x 0-2, 7 x 1-1, 0 x 2-0
residual side score  25 / 32 = 0.78125
```

Interpretation:

```text
This is the first real provisional B1-exploiter signal:
  - ungated residual
  - no RNG alias
  - physical action RNG
  - S1 surface
  - 16 paired seeds
  - residual beats B1 strongly from the opponent side

It is still not a league-integrated champion and needs swapped-label/complement sanity plus B3/B4/S3 safety evals.
But it is no longer only a gated one-label bridge.
```

Immediate next checks:

```text
1. Run swapped orientation: residual as focal vs B1 on the same 16-pair seed scope.
2. Run S3 and B3/B4 sanity for the conservative residual.
3. Register only as b1_exploiter_candidate / hard_negative_candidate if checks pass, not main champion.
```

Swapped orientation sanity:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_conservative_swapped_physical_p16_20260427
matchup                 B1 residual S1 vs B1 NoLeague baseline
action_rng_salt_mode    physical
gate_to_label_obs       false
alias_residual_rng      false
B1 residual wins/losses 27 / 5
mean                    0.84375
pair classes            11 x 2-0, 5 x 1-1, 0 x 0-2
```

This complements the previous orientation:

```text
B1 NoLeague baseline vs B1 residual S1
B1 focal wins/losses 7 / 25
residual side score  25 / 32 = 0.78125
pair classes         9 x 0-2, 7 x 1-1, 0 x 2-0
```

Current status:

```text
Status should be upgraded from residual bridge proof to provisional S1 B1 exploiter candidate.
Do not call it a main champion and do not league-integrate until S3/B3/B4 sanity and artifact packaging are done.
```

### 2026-04-28 - S3/B3/B4 sanity and candidate packaging

S3 B1 sanity:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_2label_conservative_s3_b1_p16_20260428
matchup              B1 residual S1 vs B1 NoLeague baseline
surface              S3 / public heuristic bias 3.0
action_rng_salt_mode physical
gate_to_label_obs    false
mean                 0.50
pair classes         16 x 1-1
```

Interpretation:

```text
S3 remains saturated/all-1-1, but the residual does not catastrophically collapse the S3 B1 sanity surface.
The real improvement signal remains S1, not S3.
```

S3 B3 sanity:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_2label_conservative_s3_B3_p8_20260428
matchup              B1 residual S1 vs B3 HeuristicPublicAggro
mean                 0.9375
pair classes         7 x 2-0, 1 x 1-1
```

S3 B4 sanity:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_2label_conservative_s3_B4_p8_20260428
matchup              B1 residual S1 vs B4 HeuristicPublicControl
mean                 1.0
pair classes         8 x 2-0
```

Packaged candidate manifest:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_2label_conservative_exploiter_candidate_manifest_20260428.json
```

Manifest status:

```text
candidate_id  b1_residual_2label_conservative_a01_l2_20260428
role          b1_exploiter_candidate
status        provisional_candidate_not_league_integrated
primary       lowbias_s1
S1 B1         residual focal mean 0.84375; B1 focal mean 0.21875
S3 B1         0.50 all 1-1
S3 B3         0.9375
S3 B4         1.0
```

Next implementation target:

```text
Create a runtime/train integration path for this residual policy as b1_exploiter_candidate / hard_negative_candidate.
Do not make it a main champion.
Use a low hard-negative mix against a B1/S1 main branch after the residual policy can be loaded by actors/eval as a normal policy.
```

### 2026-04-28 - Residual exploiter runtime integration smoke

Implemented the first runtime/train integration path for the provisional residual exploiter.

Code changes:

```text
python/weiss_rl/residual_policy.py
  Shared FrozenStoredLogitResidual and LiveFrozenB1Residual wrappers.

python/scripts/b1_residual_adoption_probe.py
python/scripts/b1_residual_closed_loop_eval.py
  Refactored to use the shared residual_policy module.

python/weiss_rl/config/models.py
python/weiss_rl/config/parse.py
python/weiss_rl/config/__init__.py
  Added training.residual_opponent_policies.

python/weiss_rl/runtime.py
  Added residual-opponent loading from config.
  Preloads configured residual opponents before initial actor role assignment.
  Fixed diverse_opponent_policy_ids availability check so non-B1 model policies already in _opponent_models are allowed.
  Added pfsp_residual_opponent_envs / collector_pfsp_residual_opponent_envs metrics.

configs/presets/pass3_b1_s1_main_vs_residual_hardnegative.yaml
  New B1/S1 continuation preset with b1_residual_2label_conservative_a01_l2 as an explicit diverse opponent.
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/residual_policy.py python/scripts/b1_residual_adoption_probe.py python/scripts/b1_residual_closed_loop_eval.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/runtime.py

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_runtime.py::test_load_residual_opponent_model_wraps_frozen_base python/weiss_rl/tests/test_runtime.py::test_configured_resident_opponent_policy_ids_include_residual_specs -q
```

Closed-loop refactor smoke:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_refactor_smoke_p2_20260428
residual_as_opponent       true
gate_to_label_obs          false
action_rng_salt_mode       physical
pairs                      2
residual applied rows      311
known label selected       action 51 pass at pair 0 / swap 0 / decision 20
```

Training runtime smoke:

```text
runs/b1_s1_residual_hardnegative_runtime_smoke_u451_central4_20260428
resume                     B1 checkpoint_450
max update                 451
runtime                    train_async_fast, central collection
envs                       32
unroll                     4
completed                  yes
loss                       0.310374
policy_loss                0.002116
value_loss                 0.005044
collector_pfsp_sampled_envs             53
collector_pfsp_residual_opponent_envs   53
```

Interpretation:

```text
This does not prove the main policy improves yet.
It does prove the residual candidate is no longer stranded in eval scripts:
  config can name it,
  runtime can load it,
  initial actors can assign it,
  collection can sample it,
  learner update completes with residual-opponent envs recorded.
```

Important caveat:

```text
The central collection smoke is slow on this residual path:
timer_runtime_central_fixed_opponent_overwrite_ms ~= 75s for one tiny update.
Before running a long branch, either accept a small short run or optimize/batch the residual opponent path.
```

Next hypotheses:

```text
1. Run a short 5-10 update branch only if runtime cost is acceptable, then evaluate u455/u460 against B1 on S1/S3/B3/B4.
2. Add more positive counterfactual labels to reduce overfitting risk.
3. Make residual-opponent sampling a typed league_state/hard-negative role instead of only diverse_opponent_policy_ids.
```

Fast smoke variant:

```text
runs/b1_s1_residual_hardnegative_fastbatch_smoke_u451_20260428
override                   training.rollout.batch_unrolls_per_update=32
completed                  yes
wall_clock_seconds         34.4
batch_env_steps            4096
collector_pfsp_sampled_envs             17
collector_pfsp_residual_opponent_envs   17
throughput_samples_per_sec              210231
```

Use this smaller batch override for quick local checks. It is still slow enough that long residual-opponent training should be deliberate, but it cuts the one-update smoke from about 83s to about 34s.

## 2026-04-28 - Short main-vs-residual hard-negative run

Run:

```text
runs/b1_s1_residual_hardnegative_u450_to_u455_fastbatch_20260428
resume                     B1 checkpoint_450
max update                 455
runtime                    train_async_fast, central collection
envs                       32
unroll                     4
override                   training.rollout.batch_unrolls_per_update=32
residual opponent          b1_residual_2label_conservative_a01_l2
completed                  yes
```

Training evidence:

```text
checkpoint                 training/checkpoints/checkpoint_455.pt
collector_pfsp_residual_opponent_envs at u455   37
raw_b1_top1_match at u455                       0.970464
raw_b1_family_match at u455                     0.988279
raw_b1_kl at u455                               0.000338
vtrace_rho_p99 at u455                          90295.0625
```

Eval artifacts:

```text
runs/b1_s1_residual_hardnegative_u450_to_u455_fastbatch_20260428/eval/b1_matrix_residual_hardneg_u455_s1_p8_20260428
runs/b1_s1_residual_hardnegative_u450_to_u455_fastbatch_20260428/eval/b1_matrix_residual_hardneg_u455_s3_p8_20260428
runs/b1_s1_residual_hardnegative_u450_to_u455_fastbatch_20260428/eval/b1_matrix_residual_hardneg_u455_s0_p8_20260428
```

S1 low-bias eval:

```text
u455 vs B1       mean=0.5000  pair_classes 2-0=0 1-1=8 0-2=0
B1 vs u455       mean=0.5000  pair_classes 2-0=0 1-1=8 0-2=0
u455 vs B3       mean=0.5000  pair_classes 2-0=0 1-1=8 0-2=0
u455 vs B4       mean=0.5625  pair_classes 2-0=1 1-1=7 0-2=0
```

S3 official eval:

```text
u455 vs B1       mean=0.5000  pair_classes 2-0=0 1-1=8 0-2=0
B1 vs u455       mean=0.5000  pair_classes 2-0=0 1-1=8 0-2=0
u455 vs B3       mean=1.0000
u455 vs B4       mean=1.0000
```

S0 raw eval:

```text
u455 vs B1       mean=0.5625  pair_classes 2-0=2 1-1=5 0-2=1
B1 vs u455       mean=0.3750  pair_classes 2-0=1 1-1=4 0-2=3
u455 vs B3       mean=0.0000
u455 vs B4       mean=0.0000
```

Interpretation:

```text
The residual hard-negative path is integrated into training and sampled by actors,
but five fastbatch updates did not produce a main-policy S1 improvement against B1.
This is not a residual-exploiter failure: the two-label residual remains a real S1
B1 exploiter candidate. It is a transfer/signal failure for generic main-policy RL
from sparse exposure to a tiny residual opponent.
```

Current best next hypothesis:

```text
Do not extend this main-policy branch blindly.
Mine more diverse S1 counterfactual positives with known-positive exclusion,
train a multi-label residual, and only retry main-vs-hard-negative training after
the residual has broader held-out/context-gated evidence.
```

## 2026-04-28 - Five-label residual B1 exploiter candidate

Known-positive-excluded mining:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_pass_overextend_exclude2_p8_t80_a8_stop5_20260428
execution_mode              in_process
excluded prior labels       pass label + main_move label
target filter               pass legal, baseline family main_play_character
target order                randomized
attempted_forced_replays    164 before timeout
forced_misses               0
positive_labels             3
winner_flip_labels          3
margin_positive_labels      3
```

The run timed out before writing a final summary, but `counterfactual_progress.json`
and `counterfactual_labels.jsonl` were flushed. The three new labels are all
winner flips and use action 51/pass from different pair/seed contexts than the
excluded original pass label.

Residual training:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_5label_conservative_a01_l2_20260428
label_count                                      5
alpha                                            0.1
residual_l2_coef                                 0.001
steps                                            2000
adoption_rate                                    1.0
base_param_delta_l2                              0.0
residual_zero_logit_max_abs_diff_vs_stored_b1_s1 0.0
```

Live S1 B1 closed-loop eval, ungated, physical action RNG:

```text
residual focal vs B1:
  artifact       b1_residual_closed_loop_5label_conservative_focal_physical_p16_20260428
  result         24/32 = 0.7500
  pair_classes   2-0=8 1-1=8 0-2=0

residual as opponent, B1 focal:
  artifact       b1_residual_closed_loop_5label_conservative_resopp_physical_p16_20260428
  residual side  23/32 = 0.71875
  B1 focal classes 2-0=1 1-1=7 0-2=8
```

Official S3 sanity:

```text
residual vs B1 S3:
  artifact       b1_residual_closed_loop_5label_s3_B1_p16_20260428
  result         16/32 = 0.5000
  pair_classes   all 16 pairs are 1-1

residual vs B3 S3:
  artifact       b1_residual_closed_loop_5label_s3_B3_p8_20260428
  result         14/16 = 0.8750

residual vs B4 S3:
  artifact       b1_residual_closed_loop_5label_s3_B4_p8_20260428
  result         16/16 = 1.0000
```

Raw S0 ablation:

```text
residual vs B1 S0:
  artifact       b1_residual_closed_loop_5label_s0_B1_p8_20260428
  result         10/16 = 0.6250
  pair_classes   2-0=3 1-1=4 0-2=1
```

Candidate manifest:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_5label_conservative_exploiter_candidate_manifest_20260428.json
```

Interpretation:

```text
We now have a real provisional B1 exploiter / hard-negative candidate on S1.
This is not just exact-gated replay: it is ungated, physical-RNG, live closed-loop
eval, and it beats B1 from both focal/opponent positions on 16 paired seeds.

It is still not a main champion and should not be described as solved league
learning. The immediately preceding main-policy branch showed that simply sampling
a tiny residual hard negative for five updates did not transfer into main-policy
improvement. The usable result is the exploiter itself plus a concrete hard-negative
artifact for the next league design step.
```

Next hypotheses:

```text
1. Confirm this 5-label residual on a larger independent S1 seed set, ideally 32 pairs.
2. Mine a few non-pass positives or two-step positives to avoid a pass-only critique.
3. Register the residual as b1_exploiter/hard_negative, not a champion.
4. Retry main-policy training only with a broader confirmed exploiter and explicit
   hard-negative role/gate; do not keep extending generic main RL from the failed u455 branch.
```

## 2026-04-28 - Six-label residual supersedes five-label S1 result

After the initial read, the known-positive-excluded mining artifact had flushed
one additional winner-flip label:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_pass_overextend_exclude2_p8_t80_a8_stop5_20260428
attempted_forced_replays    235
forced_misses               0
positive_labels             4
winner_flip_labels          4
margin_positive_labels      4
target_states               30
```

Retrained residual:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_6label_conservative_a01_l2_20260428
label_count                                      6
alpha                                            0.1
residual_l2_coef                                 0.001
steps                                            2000
adoption_rate                                    1.0
base_param_delta_l2                              0.0
residual_zero_logit_max_abs_diff_vs_stored_b1_s1 0.0
```

Live S1 B1 closed-loop eval, ungated, physical action RNG:

```text
residual focal vs B1:
  artifact       b1_residual_closed_loop_6label_conservative_focal_physical_p16_20260428
  result         26/32 = 0.8125
  pair_classes   2-0=10 1-1=6 0-2=0

residual as opponent, B1 focal:
  artifact       b1_residual_closed_loop_6label_conservative_resopp_physical_p16_20260428
  residual side  26/32 = 0.8125
  B1 focal classes 2-0=0 1-1=6 0-2=10
```

Official S3 sanity for the six-label residual:

```text
residual vs B1 S3:
  artifact       b1_residual_closed_loop_6label_s3_B1_p16_20260428
  result         16/32 = 0.5000
  pair_classes   all 16 pairs are 1-1

residual vs B3 S3:
  artifact       b1_residual_closed_loop_6label_s3_B3_p8_20260428
  result         14/16 = 0.8750

residual vs B4 S3:
  artifact       b1_residual_closed_loop_6label_s3_B4_p8_20260428
  result         16/16 = 1.0000
```

Raw S0 ablation:

```text
residual vs B1 S0:
  artifact       b1_residual_closed_loop_6label_s0_B1_p8_20260428
  result         6/16 = 0.3750
  pair_classes   2-0=2 1-1=2 0-2=4
```

Candidate manifest:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_6label_conservative_exploiter_candidate_manifest_20260428.json
```

Interpretation:

```text
The six-label residual is now the best current positive result. It is a real,
ungated S1 B1 exploiter candidate and beats B1 from both seat orientations on
16 paired seeds. It remains wrapper-surface-neutral on S3 and strong against
B3/B4 on S3.

It is not a main policy or thesis champion: S0 raw ablation is weak, and most
new labels are pass-overextension corrections. The correct role is:
  role=b1_exploiter
  pool_roles=[b1_exploiter, hard_negative]
  rejected_for_main=true
until a main policy learns to absorb/defend against it.
```

Immediate next hypotheses:

```text
1. Confirm the six-label residual on an independent 32-pair S1 seed set.
2. Mine non-pass/two-step positives to diversify away from pass-overextension.
3. Register this residual in league_state as a b1_exploiter/hard_negative.
4. Retry main-policy training with explicit hard-negative role and stronger/cleaner
   transfer signal; the generic u450->u455 hard-negative branch did not move S1.
```

## 2026-04-28 - Six-label residual 32-pair S1 confirmation

Confirm artifacts:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_6label_confirm32_focal_physical_20260428
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_6label_confirm32_resopp_physical_20260428
```

Both use:

```text
surface                    lowbias_s1
public_heuristic_bias      1.0
action_rng_salt_mode       physical
gate_to_label_obs          false
pairs                      32
games                      64
```

Results:

```text
residual focal vs B1:
  wins/losses              49/15
  mean                     0.765625
  pair_classes             2-0=17 1-1=15 0-2=0

residual as opponent, B1 focal:
  B1 focal wins/losses      13/51
  residual side mean        0.796875
  B1 focal pair_classes     2-0=0 1-1=13 0-2=19
```

Manifest updated:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_6label_conservative_exploiter_candidate_manifest_20260428.json
```

Interpretation:

```text
The six-label residual is confirmed as a provisional S1 B1 exploiter/hard-negative
candidate on a 32-pair independent seed scope. This is now the strongest positive
result in the rescue branch.

It still should not be called a main champion:
  S3 B1 remains 0.50/all-1-1 due saturated wrapper surface.
  S0 raw ablation is weak.
  The residual behavior is dominated by pass-overextension corrections.
```

Next actions:

```text
1. Register as b1_exploiter/hard_negative in league state or equivalent config.
2. Mine non-pass and two-step labels to reduce pass-only overfit risk.
3. Train main policy against this confirmed hard negative with explicit role logging.
4. Gate any main-policy claim on S1 B1 improvement, not S3.
```

## 2026-04-28 - Non-pass label mining and 7-label residual variant

Code patch:

```text
python/scripts/b1_counterfactual_labels.py
  added --exclude-candidate-family
  added --exclude-candidate-action-id
  added --ignore-excluded-label-pair-indices
```

Purpose:

```text
Avoid repeatedly selecting pass/action 51 and allow fresh seed scopes to reuse
numeric pair slots while still excluding known legal fingerprints/action pairs.
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_counterfactual_labels.py
```

Non-pass mining:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_nonpass_exclude6_p16_t120_a8_stop3_20260428
excluded_candidate_families      pass
excluded_candidate_action_ids    51
ignore_excluded_label_pair_indices true
attempted_forced_replays         240 before timeout
forced_misses                    0
positive_labels                  1
winner_flip_labels               1
positive action                  408 main_move(from_slot=1, to_slot=3)
baseline action                  102 main_play_character(hand_index=0, stage_slot=0)
score_delta                      2.135
```

Seven-label residual:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_7label_nonpass_a01_l2_20260428
label_count                                      7
adoption_rate                                    1.0
base_param_delta_l2                              0.0
residual_zero_logit_max_abs_diff_vs_stored_b1_s1 0.0
```

S1 eval, 16 pairs:

```text
residual focal vs B1:
  artifact       b1_residual_closed_loop_7label_nonpass_focal_physical_p16_20260428
  result         26/32 = 0.8125
  pair_classes   2-0=10 1-1=6 0-2=0

residual as opponent:
  artifact       b1_residual_closed_loop_7label_nonpass_resopp_physical_p16_20260428
  residual side  29/32 = 0.90625
  B1 focal classes 2-0=0 1-1=3 0-2=13
```

S1 confirmation, 32 pairs:

```text
residual focal vs B1:
  artifact       b1_residual_closed_loop_7label_confirm32_focal_physical_20260428
  result         47/64 = 0.734375
  pair_classes   2-0=15 1-1=17 0-2=0

residual as opponent:
  artifact       b1_residual_closed_loop_7label_confirm32_resopp_physical_20260428
  residual side  51/64 = 0.796875
  B1 focal classes 2-0=0 1-1=13 0-2=19
```

Manifest:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_7label_nonpass_exploiter_candidate_manifest_20260428.json
```

Interpretation:

```text
The 7-label variant successfully adds a non-pass/main_move winner-flip label and
remains a strong S1 B1 exploiter. It does not strictly beat the 6-label primary
on confirm32 focal score:
  6-label focal confirm32: 49/64 = 0.765625
  7-label focal confirm32: 47/64 = 0.734375
Both have residual-as-opponent confirm32 residual-side score 51/64 = 0.796875.

Use:
  primary hard negative: b1_residual_6label_conservative_a01_l2
  diversity variant:    b1_residual_7label_nonpass_a01_l2
```

## 2026-04-28 confirmed residual hard-negative main-policy smoke

Config:

```text
configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml
```

This config registers both confirmed residual B1 exploiters as resident/diverse
opponents:

```text
b1_residual_6label_conservative_a01_l2
b1_residual_7label_nonpass_a01_l2
```

Runtime smoke:

```text
run:
  runs/b1_s1_confirmed_residual_hardnegatives_runtime_smoke_u451_20260428

result:
  completed one update from B1 checkpoint_450
  collector_pfsp_residual_opponent_envs = 14
  collector_pfsp_sampled_envs = 14

note:
  the command used --num-envs 32, so runtime_actor_count stayed 1 even though
  diverse_opponent_actor_count was 2. The fixed diverse policy list still
  sampled the configured residual opponent ids. Per-policy residual opponent
  env splits are not yet logged.
```

Short main-policy branch:

```text
run:
  runs/b1_s1_confirmed_residual_hardnegatives_u450_to_u455_fastbatch_20260428

checkpoint:
  runs/b1_s1_confirmed_residual_hardnegatives_u450_to_u455_fastbatch_20260428/training/checkpoints/checkpoint_455.pt

final train summary:
  loss        0.350520
  policy_loss 0.005772
  value_loss  0.005232
  entropy     0.599954
```

S1 low-bias eval, 8 paired seeds:

```text
artifact:
  runs/b1_s1_confirmed_residual_hardnegatives_u450_to_u455_fastbatch_20260428/eval/b1_matrix_confirmed_residual_hardneg_u455_s1_p8_20260428

u455 vs B1:
  mean 0.5000, wins 8, losses 8, pair_classes 2-0=0 1-1=8 0-2=0

B1 vs u455:
  mean 0.5000, wins 8, losses 8, pair_classes 2-0=0 1-1=8 0-2=0

u455 vs B3:
  mean 0.5000, wins 8, losses 8, pair_classes 2-0=0 1-1=8 0-2=0

u455 vs B4:
  mean 0.6250, wins 10, losses 6, pair_classes 2-0=2 1-1=6 0-2=0
```

S3 official sanity eval, 8 paired seeds:

```text
artifact:
  runs/b1_s1_confirmed_residual_hardnegatives_u450_to_u455_fastbatch_20260428/eval/b1_matrix_confirmed_residual_hardneg_u455_s3_p8_20260428

u455 vs B1:
  mean 0.5000, wins 8, losses 8, pair_classes 2-0=0 1-1=8 0-2=0

B1 vs u455:
  mean 0.5000, wins 8, losses 8, pair_classes 2-0=0 1-1=8 0-2=0

u455 vs B3:
  mean 1.0000, wins 16, losses 0, pair_classes 2-0=8 1-1=0 0-2=0

u455 vs B4:
  mean 1.0000, wins 16, losses 0, pair_classes 2-0=8 1-1=0 0-2=0
```

Interpretation:

```text
The confirmed residual hard negatives are real S1 B1-exploiter artifacts, but
this very short generic main-policy IMPALA branch did not transfer their
deviations into the main policy. S3 sanity remained intact, but S1 B1 stayed
exactly all 1-1.

The next main-policy attempt should not rely only on sparse opponent exposure.
It needs direct counterfactual-label or residual-action auxiliary supervision
for the main policy, or a longer/stronger hard-negative curriculum with explicit
eval against the residual policies.
```

## 2026-04-28 counterfactual-positive main-policy auxiliary

Patch:

```text
Added training.counterfactual_positive config block.
Added ImpalaLearner counterfactual positive CE/margin auxiliary.
Added metrics:
  counterfactual_positive_loss
  counterfactual_positive_ce_loss
  counterfactual_positive_margin_loss
  counterfactual_positive_label_count
  counterfactual_positive_prob_mean
  counterfactual_positive_top1_match
  counterfactual_positive_logit_margin_mean
Added presets:
  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml
  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux_strong.yaml
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/config/__init__.py python/weiss_rl/learners/impala_learner.py python/scripts/train.py

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_counterfactual_positive_aux python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_counterfactual_positive_aux_uses_tensor_labels -q
2 passed
```

Moderate cf-aux run:

```text
run:
  runs/b1_s1_confirmed_residual_hardnegatives_cfaux_u450_to_u455_20260428

config:
  pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml

notes:
  first preset revision accidentally loaded 6 labels, not all 7.
  LR was still 1e-5 and coef annealed near 1.

u455 train metrics:
  counterfactual_positive_prob_mean rose only to 0.00080
  counterfactual_positive_top1_match stayed 0.0

S1 eval, 8 paired seeds:
  artifact:
    runs/b1_s1_confirmed_residual_hardnegatives_cfaux_u450_to_u455_20260428/eval/b1_matrix_confirmed_residual_hardneg_cfaux_u455_s1_p8_20260428

  u455 vs B1:
    mean 0.5000, all 1-1

  B1 vs u455:
    mean 0.5000, all 1-1

  u455 vs B3:
    mean 0.5000, all 1-1

  u455 vs B4:
    mean 0.6875, 2-0=3 1-1=5 0-2=0
```

Strong cf-aux run, mixed precision:

```text
run:
  runs/b1_s1_confirmed_residual_hardnegatives_cfaux_strong_u450_to_u455_20260428

config:
  pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux_strong.yaml

result:
  AMP overflow skipped every optimizer step.
  amp_grad_overflow = 1.0 every update
  grad_norm = nan
  checkpoint L2 delta from B1 checkpoint_450 = 0.0

interpretation:
  strong counterfactual CE requires FP32 or lower scale; AMP makes this
  particular bridge test a false no-op.
```

Strong cf-aux run, FP32:

```text
run:
  runs/b1_s1_confirmed_residual_hardnegatives_cfaux_strong_fp32_u450_to_u455_20260428

config:
  pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux_strong.yaml

command detail:
  training.precision.mixed_precision=false
  num_envs=8
  batch_unrolls_per_update=8

train result:
  checkpoint L2 delta from B1 checkpoint_450 = 1.0606
  counterfactual_positive_loss 18.8067 -> 7.9201
  counterfactual_positive_prob_mean 0.00092 -> 0.03991
  counterfactual_positive_logit_margin_mean -8.4238 -> +0.0171
  counterfactual_positive_top1_match was unstable and ended 0.0

S1 eval, 8 paired seeds:
  artifact:
    runs/b1_s1_confirmed_residual_hardnegatives_cfaux_strong_fp32_u450_to_u455_20260428/eval/b1_matrix_confirmed_residual_hardneg_cfaux_strong_fp32_u455_s1_p8_20260428

  u455 vs B1:
    mean 0.2500, wins 4, losses 12, pair_classes 2-0=0 1-1=4 0-2=4

  B1 vs u455:
    mean 0.8750 for B1, wins 14, losses 2, pair_classes 2-0=6 1-1=2 0-2=0

  u455 vs B3:
    mean 0.1875, wins 3, losses 13, pair_classes 2-0=0 1-1=3 0-2=5

  u455 vs B4:
    mean 0.5625, wins 9, losses 7, pair_classes 2-0=1 1-1=7 0-2=0
```

Interpretation:

```text
The full-model counterfactual CE path is wired and can move the checkpoint, but
aggressive whole-model label injection is destructive. This confirms the earlier
residual diagnosis: counterfactual positives should be consumed by a constrained
adapter/residual or a much gentler head-only/top-layer path, not by blasting the
entire B1 body.

Current best positive artifact remains the frozen-B1 residual exploiter:
  6-label residual confirm32 focal S1: 49/64 = 0.765625
  7-label residual confirm32 focal S1: 47/64 = 0.734375

Next best engineering move:
  make the trainable main policy architecture identity-preserving, e.g. frozen
  B1 + residual/adapter head as the actual policy, rather than trying to absorb
  the residual into all B1 weights through sparse IMPALA or strong full-model CE.
```

## 2026-04-28: diversified 9-label residual pass

Goal:

```text
Check whether newly mined pass-overextension labels improve the frozen-B1
residual exploiter, or whether they dilute the cleaner 6-label residual.
```

New label mining artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_pass_overextend_exclude7_p16_t100_a4_stop3_20260428
```

The mining run timed out before writing a final summary, but incremental flush
artifacts are valid:

```text
counterfactual_progress.json:
  target_states: 64
  attempted_forced_replays: 132
  forced_misses: 0
  positive_labels: 2
  winner_flip_labels: 2
  margin_positive_labels: 2
  score_delta max: 2.25

counterfactual_labels.jsonl:
  2 tensorized positive labels
```

Conservative 9-label residual:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_9label_passmix_a01_l2_20260428

training:
  labels: 9
  alpha: 0.1
  steps: 400
  residual_l2_coef: 0.01

offline adoption:
  5/9 = 0.5556
  residual-zero identity diff: 0.0
  base_param_delta_l2: 0.0
```

Natural S1 closed-loop eval:

```text
focal artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_9label_passmix_focal_physical_p16_20260428

B1 residual S1 vs B1:
  mean: 0.5625
  wins/losses: 18/14
  pair_classes: 2-0=2 1-1=14 0-2=0

as-opponent artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_9label_passmix_as_opponent_physical_p16_20260428

B1 vs B1 residual S1:
  B1 mean: 0.28125
  residual-side implied mean: 0.71875
  B1 pair_classes: 2-0=0 1-1=9 0-2=7
```

Interpretation:

```text
The conservative 9-label residual is a real S1 B1 exploiter, but weaker than
the earlier 6-label residual in focal orientation. The extra labels were not
free; with the old penalty they were partly unadopted.
```

Looser 9-label residual:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_9label_passmix_a01_l2loose_800s_20260428

training:
  labels: 9
  alpha: 0.1
  steps: 800
  residual_l2_coef: 0.002

offline adoption:
  9/9 = 1.0
  residual-zero identity diff: 0.0
  base_param_delta_l2: 0.0
```

Natural S1 closed-loop eval:

```text
focal artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_9label_passmix_l2loose_focal_physical_p16_20260428

B1 residual S1 vs B1:
  mean: 0.65625
  wins/losses: 21/11
  pair_classes: 2-0=5 1-1=11 0-2=0
  residual family drift rate: 0.02196

as-opponent artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_9label_passmix_l2loose_as_opponent_physical_p16_20260428

B1 vs B1 residual S1:
  B1 mean: 0.21875
  residual-side implied mean: 0.78125
  B1 pair_classes: 2-0=0 1-1=7 0-2=9
  residual family drift rate: 0.02312
```

Decision:

```text
The looser 9-label residual is the current best diversified hard-negative
candidate. It does not replace the 6-label residual as the cleanest focal
confirm32 artifact yet, but it is strong enough to keep as a second confirmed
S1 B1 exploiter/hard-negative candidate.

Do not continue full-model CE transfer. It is too destructive.
Do not call these main champions. They are frozen-B1 residual exploiters.
Next high-leverage path is to make this residual architecture trainable as the
main/exploiter policy form, or to train the main against confirmed residual
hard negatives with a more direct imitation/adaptation bridge than sparse IMPALA.
```

Config follow-up:

```text
Updated:
  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml
  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml

Changes:
  added b1_residual_9label_passmix_a01_l2loose as a residual hard-negative
  added the exclude7 label dir to the cf-positive label dirs
  raised cf-positive max_labels from 7 to 9

Validation:
  uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_counterfactual_positive_aux -q
  2 passed
```

## 2026-04-28: rush residual funnel wrapper

Added:

```text
python/scripts/b1_rush_residual_loop.py
```

Purpose:

```text
Automate the fast local funnel:
  optional in-process label mining
  residual adoption training
  cheap p8/p4 screen
  p16 confirm only if screen clears threshold
  compact rush_summary.json artifact

This is intended to keep iteration fast locally and map cleanly to sharded
multi-GPU Linux execution later. The final thesis-grade check still needs
p32/p64+ confirmation.
```

Validation smoke:

```text
command:
  uv run python python/scripts/b1_rush_residual_loop.py --tag smoke_nomine_p4_20260428 --skip-mining --screen-pairs 4 --confirm-if-screen-ge 0.99 --train-steps 80 --max-labels 9 --device cuda:0

result:
  adoption 9/9
  p4 residual vs B1 screen: 6/8 = 0.75
  pair_classes: 2-0=2 1-1=2 0-2=0

summary:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_rush_residual_loop_smoke_nomine_p4_20260428/rush_summary.json
```

Rush pass:

```text
command:
  uv run python python/scripts/b1_rush_residual_loop.py --tag rush_pass_screen_20260428a --screen-pairs 8 --confirm-pairs 16 --confirm-if-screen-ge 0.58 --train-steps 220 --final-train-steps 800 --rerun-final-if-screen-passes --mine-pairs 8 --mine-target-states 32 --mine-targets-per-pair 4 --mine-actions-per-state 4 --mine-max-forced-replays 180 --mine-stop-after-positives 1 --mine-target-seed 42803 --device cuda:0

mining result:
  artifact:
    runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_rush_rush_pass_screen_20260428a
  searched target_states=32, attempted_forced_replays=117
  positive_labels=0, winner_flip_labels=0
  forced_misses=0
  best score_delta=0.05

screen result:
  residual vs B1 p8: 11/16 = 0.6875
  pair_classes: 2-0=3 1-1=5 0-2=0
  passed confirm threshold

final residual:
  artifact:
    runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_rush_pass_screen_20260428a_final800
  adoption: 10/10 = 1.0
  base_param_delta_l2: 0.0

p16 confirm:
  focal artifact:
    runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_rush_pass_screen_20260428a_confirm_focal_p16
  residual vs B1: 30/32 = 0.9375
  pair_classes: 2-0=14 1-1=2 0-2=0
  residual family drift rate: 0.02593

  reverse artifact:
    runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_rush_pass_screen_20260428a_confirm_asopp_p16
  B1 vs residual: B1 scored 7/32 = 0.21875
  residual-side implied score: 25/32 = 0.78125
  B1 pair_classes: 2-0=0 1-1=7 0-2=9
  residual family drift rate: 0.02970
```

Decision:

```text
This is the strongest fast-loop residual so far. It is now registered as:

  b1_residual_rush_10label_passmix_final800

in:

  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml

The cf-positive preset max_labels is now 10.

Important: this is still a frozen-B1 residual exploiter/hard-negative, not a
main champion. The next architectural task remains making the residual form
trainable as a first-class policy/exploiter, because full-model transfer was
destructive and sparse hard-negative IMPALA did not transfer.
```

## 2026-04-28: updated hard-negative main-policy smoke

Goal:

```text
Verify the updated main-policy config can load and sample all confirmed
residual hard negatives, including b1_residual_rush_10label_passmix_final800.
This is an integration smoke, not a learning-quality claim.
```

Failed attempts:

```text
runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428
  failed before runtime because --b1-baseline-run-dir was missing.

runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v2
  imported B1 anchor, but failed because inherited system.collection_backend=process
  is unsupported for this local runtime setup.
```

Successful smoke:

```text
command:
  uv run python python/scripts/train.py --stack-config configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml --run-label b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v3 --runtime-mode train_async_fast --device auto --num-envs 8 --unroll-length 16 --max-updates 451 --checkpoint-interval-updates 1 --max-wall-clock-minutes 4 --resume-from runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt --resume-allow-config-mismatch --resume-reset-optimizer --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --config-override evaluation.periodic_dev_eval_interval_updates=0 --config-override system.collection_backend='"auto"'

artifact:
  runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v3

result:
  completed one update, checkpoint_451 written
  train/loss: 16.409204
  policy_loss: -0.006291
  value_loss: 0.008627
```

Runtime evidence:

```text
TensorBoard/config payload includes residual policies:
  b1_residual_6label_conservative_a01_l2
  b1_residual_7label_nonpass_a01_l2
  b1_residual_9label_passmix_a01_l2loose
  b1_residual_rush_10label_passmix_final800

Update 451 metrics:
  league/pfsp_residual_opponent_envs: 53
  runtime/collector_pfsp_residual_opponent_envs: 53
  league/pfsp_sampled_envs: 53
  counterfactual_positive_label_count: 10
  counterfactual_positive_prob_mean: 0.000643
  counterfactual_positive_top1_match: 0.0
```

Tiny S1 p4 eval:

```text
command:
  uv run python python/scripts/b1_artifact_matrix.py --stack-config configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml --run-dir runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v3 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --checkpoint-policy u451=runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v3/training/checkpoints/checkpoint_451.pt --matchup u451=b1_noleague_baseline --pairs 4 --artifact-dir-name b1_matrix_rushhardneg_main_smoke_u451_s1_p4_20260428 --surface-name lowbias_s1 --device cuda:0 --public-heuristic-bias-scale 1.0 --scoring-mode learner --action-rng-salt-mode physical

artifact:
  runs/b1_s1_rushhardneg_main_smoke_u450_to_u451_20260428_v3/eval/b1_matrix_rushhardneg_main_smoke_u451_s1_p4_20260428

result:
  u451 vs B1 lowbias_s1 p4: 4/8 = 0.5
  pair_classes: 2-0=0 1-1=4 0-2=0
```

Interpretation:

```text
The updated league/hard-negative plumbing works. The main policy did not move
off B1 after one tiny local update, which is expected and not evidence either
way. The important blocker remains transfer: residual exploiters are strong,
but full-model/sparse IMPALA transfer is weak. School-box work should either
run longer with this verified config or, preferably, make frozen-B1 residuals
a first-class trainable policy form.
```

## 2026-04-28: family-gated residual + full-model transfer check

Implementation:

```text
python/weiss_rl/residual_policy.py
  added residual_mode = plain | gated | family_gated
  family_gated uses action-family ids from ActionCatalog and per-family gates
  loader now backfills action_family_ids for older plain residual artifacts

python/scripts/b1_residual_adoption_probe.py
  added --residual-mode, --gate-bias, --validation-fraction
  stores residual_mode/action_family_ids/family names in residual_state.pt
  reports train/validation adoption summaries and per-family adoption

python/scripts/b1_rush_residual_loop.py
  added pass-through flags for residual mode, gate bias, and validation split

configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml
  added b1_residual_10label_familygated_a01_l2 as a diversity hard negative
```

Verification:

```text
uv run python -m py_compile python/weiss_rl/residual_policy.py python/scripts/b1_residual_adoption_probe.py python/scripts/b1_residual_closed_loop_eval.py python/scripts/b1_rush_residual_loop.py

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_counterfactual_positive_aux -q
  2 passed

loader smoke:
  old plain residual loaded: mode=plain action_family_ids=527 family_count=0
  new family-gated residual loaded: mode=family_gated action_family_ids=527 family_count=17
```

Family-gated residual training:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_10label_familygated_a01_l2_20260428

result:
  adoption: 10/10
  train adoption: 8/8
  validation adoption: 2/2
  positive families: pass 8/8, main_move 2/2
  residual-zero identity max_abs_diff: 0.0
  base delta: 0.0
```

Family-gated closed-loop S1 eval:

```text
focal artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_10label_familygated_focal_physical_p16_20260428

B1 residual S1 vs B1 NoLeague baseline:
  26/32 = 0.8125
  pair_classes: 2-0=10 1-1=6 0-2=0
  pair_score_mean: 0.8125
  residual selected-family drift: 0.03196

reverse artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_10label_familygated_asopp_physical_p16_20260428

B1 NoLeague baseline vs B1 residual S1:
  B1: 7/32 = 0.21875
  residual implied: 25/32 = 0.78125
  B1 pair_classes: 2-0=0 1-1=7 0-2=9
  residual selected-family drift: 0.02922
```

Interpretation:

```text
family_gated is a valid residual exploiter and useful diversity hard negative,
but it is not clearly better than the earlier plain rush residual:

plain rush residual p16:
  focal residual vs B1: 30/32 = 0.9375
  reverse implied residual: 25/32 = 0.78125
  family drift: about 0.026

family-gated p16:
  focal residual vs B1: 26/32 = 0.8125
  reverse implied residual: 25/32 = 0.78125
  family drift: about 0.029-0.032

Keep plain rush residual as primary; keep family_gated as diversity/backup.
```

Main-policy hard-negative league run:

```text
artifact:
  runs/b1_s1_rushhardneg_main_u450_to_u455_20260428_v5

command summary:
  B1 checkpoint_450 -> checkpoint_454
  config: pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml
  residual hard negatives enabled, counterfactual positives enabled
  local wall-clock hit before checkpoint_455, checkpoint_454 written

runtime evidence at update 454:
  collector_pfsp_residual_opponent_envs: 53-60 range in local runs
  counterfactual_positive_label_count: 10
  counterfactual_positive_prob_mean: 0.000724
  counterfactual_positive_top1_match: 0.0
  parameter L2 from B1: about 0.0155
```

S1 eval:

```text
artifact:
  runs/b1_s1_rushhardneg_main_u450_to_u455_20260428_v5/eval/b1_main_u454_s1_vs_b1_p8_20260428

B1 vs main_u454:
  8/16 = 0.5, all 1-1

main_u454 vs B1:
  8/16 = 0.5, all 1-1
```

Aggressive full-model CE transfer check:

```text
Initial AMP strong branch:
  runs/b1_s1_cfstrong_fullmodel_u450_to_u451_20260428_v1
  runs/b1_s1_cfstrong_fullmodel_u451_to_u455_20260428_v1

finding:
  amp_grad_overflow=1.0
  grad_norm=NaN
  parameter L2 from B1: exactly 0.0
  conclusion: AMP skipped optimizer steps, so this was inert.

FP32 rerun:
  runs/b1_s1_cfstrong_fp32_fullmodel_u450_to_u451_20260428_v1
  runs/b1_s1_cfstrong_fp32_fullmodel_u451_to_u455_20260428_v1

FP32 update evidence:
  u451 parameter L2 from B1: about 0.132
  u455 parameter L2 from B1: about 0.526
  counterfactual_positive_prob_mean: 0.000644 -> 0.01642
  counterfactual_positive_top1_match: 0.0 -> 0.1
```

FP32 full-model S1 eval:

```text
u451 artifact:
  runs/b1_s1_cfstrong_fp32_fullmodel_u450_to_u451_20260428_v1/eval/b1_cfstrong_fp32_u451_s1_vs_b1_p8_20260428

u451 result:
  still 0.5 both directions, all 1-1

u455 artifact:
  runs/b1_s1_cfstrong_fp32_fullmodel_u451_to_u455_20260428_v1/eval/b1_cfstrong_fp32_u455_s1_vs_b1_p8_20260428

u455 result:
  B1 vs cfstrong_fp32_u455: 9/16 = 0.5625, pair_classes 2-0=1 1-1=7
  cfstrong_fp32_u455 vs B1: 7/16 = 0.4375, pair_classes 1-1=7 0-2=1
```

Current verdict:

```text
Confirmed:
  frozen-B1 residual exploiters produce real S1 B1 pressure.
  residual hard negatives load and are sampled by the league runtime.
  family-gated residuals work and preserve base identity.
  full-model FP32 counterfactual CE can now move parameters when AMP is disabled.

Not confirmed:
  normal/full-model main-policy league improvement.

Best current positive artifact:
  frozen-B1 residual exploiter, not full-model main policy.

Next architectural step:
  make frozen-B1 residual a first-class trainable policy/learner, so league
  training updates the residual head directly instead of hoping a full recurrent
  network absorbs sparse residual labels and hard-negative pressure.
```

## 2026-04-28 trainable-live frozen-B1 residual

Implemented a simulator-facing trainable residual path:

```text
Code:
  python/weiss_rl/residual_policy.py
    added TrainableLiveFrozenB1Residual

  python/scripts/b1_trainable_residual_policy.py
    new standalone trainer for live frozen-B1 residual heads
    trains through the live B1 wrapper, but freezes B1 base params
    saves residual_state.pt compatible with b1_residual_closed_loop_eval.py

  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml
    added b1_residual_trainable_live_10label_l2x as a hard-negative candidate
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/residual_policy.py python/scripts/b1_trainable_residual_policy.py python/scripts/b1_residual_closed_loop_eval.py

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_counterfactual_positive_aux -q
  2 passed

config load smoke:
  residual_opponent_policies count = 6
  includes b1_residual_trainable_live_10label_l2x
```

Important failed/learning variant:

```text
Artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_trainable_live_residual_10label_plain_a01_l2_20260428

Training:
  lr=3e-3, alpha=0.1, residual_l2=1e-5, steps=800
  adoption 10/10
  base_param_delta_l2=0.0

Problem:
  over-drove sparse labels to positive_probability about 0.998+
  p16 focal result only 17/32 = 0.53125
  p16 reverse focal-B1 result also 17/32 for B1

Conclusion:
  live residual training works mechanically, but unchecked CE overfits too hard.
```

Confirmed improved trainable-live residual:

```text
Artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_trainable_live_residual_10label_plain_a01_lr3e4_l2x_20260428

Training:
  lr=3e-4
  alpha=0.1
  residual_l2=0.001
  early-stop adoption target=0.9
  early-stop mean positive prob target=0.70
  stopped_at_step=75
  train_loss_final=1.5900
  adoption 10/10
  validation adoption 2/2
  base_param_delta_l2=0.0
```

Closed-loop S1 eval:

```text
p8 screen:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_trainable_live_residual_10label_plain_lr3e4_l2x_focal_physical_p8_20260428
  residual vs B1: 13/16 = 0.8125
  pair classes: 2-0=5, 1-1=3, 0-2=0

p16 focal confirm:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_trainable_live_residual_10label_plain_lr3e4_l2x_focal_physical_p16_20260428
  residual vs B1: 27/32 = 0.84375
  pair classes: 2-0=11, 1-1=5, 0-2=0
  residual family drift: 0.0266

p16 reverse confirm:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_trainable_live_residual_10label_plain_lr3e4_l2x_asopp_physical_p16_20260428
  B1 vs residual: 6/32 = 0.1875
  residual implied: 26/32 = 0.8125
  B1 pair classes: 2-0=0, 1-1=6, 0-2=10
  residual family drift: 0.0359
```

Current verdict:

```text
This is now a first-class trainable-live frozen-B1 residual B1 exploiter.
It is not just a stored-logit adoption proof:
  the residual head was trained through a live frozen B1 model wrapper,
  B1 base params stayed fixed,
  the produced residual_state.pt works in live simulator eval,
  and it beats B1 strongly on S1 in both focal and reverse views.

Still not confirmed:
  full-model main-policy league improvement.

Next step:
  run a short main-policy league/hard-negative continuation using the updated
  config that includes b1_residual_trainable_live_10label_l2x. If the full
  main model remains flat, the next branch should train the residual policy
  itself further rather than trying to distill its behavior into the full model.
```

Main-policy absorption smoke with the updated hard-negative pool:

```text
Run:
  runs/b1_s1_trainablelive_hardneg_main_smoke_u450_to_u451_20260428

Command shape:
  train.py
  --stack-config configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml
  --resume-from B1 checkpoint_450
  --max-updates 451
  --num-envs 8
  --unroll-length 16
  --resume-reset-optimizer

Training completed:
  checkpoint_451 written
  loss=16.397
  throughput about 40.7k samples/sec
  vtrace_rho_p99 still huge: 1,562,222.75
  counterfactual_positive_prob_mean=0.0006428
  counterfactual_positive_top1_match=0.0
```

S1 p4 eval:

```text
Artifact:
  runs/b1_s1_trainablelive_hardneg_main_smoke_u450_to_u451_20260428/eval/b1_matrix_trainablelive_hardneg_main_u451_s1_p4_20260428

u451 vs B1:
  4/8 = 0.5
  pair classes all 1-1

B1 vs u451:
  4/8 = 0.5
  pair classes all 1-1
```

Interpretation:

```text
The trainable-live residual path works and produces a real B1 exploiter.
The full main model still does not absorb the residual/counterfactual signal
on a short update. This is consistent with previous full-model runs.

Fast next hypothesis:
  stop trying to immediately distill residual behavior into the full model;
  continue training/evolving residual policies directly, then use confirmed
  residual exploiters as hard-negative opponents for a later main-policy phase.
```

## 2026-04-28 FP32 u460 main-policy absorption check

Added a dedicated short absorption preset:

```text
configs/presets/pass3_b1_s1_main_vs_trainablelive_absorb_fp32_u460.yaml

Purpose:
  test whether the full main model can absorb confirmed residual/counterfactual
  signal over 10 updates when AMP is disabled.

Key settings:
  extends pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml
  mixed_precision=false
  checkpoint_interval_updates=5
  raw_b1_distill disabled
  counterfactual_positive coef=3.0 -> 2.0 through u470
  margin=1.5, margin_coef=0.3
  max_labels=10
```

Config validation:

```text
mixed_precision False
checkpoint_interval 5
raw_b1_distill False 0.0
counterfactual_positive True 3.0 2.0 470
residual pool includes b1_residual_trainable_live_10label_l2x

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_counterfactual_positive_aux -q
  2 passed
```

Training run:

```text
runs/b1_s1_trainablelive_absorb_fp32_u450_to_u460_20260428

Command shape:
  train.py
  --stack-config configs/presets/pass3_b1_s1_main_vs_trainablelive_absorb_fp32_u460.yaml
  --resume-from B1 checkpoint_450
  --resume-reset-optimizer
  --max-updates 460
  --num-envs 8
  --unroll-length 16
  --checkpoint-interval-updates 5

Completed:
  checkpoint_455.pt
  checkpoint_460.pt
```

Training metrics:

```text
counterfactual_positive_loss:
  u451 17.5759
  u455 10.8911
  u460  7.7023

counterfactual_positive_prob_mean:
  u451 0.000644
  u455 0.01596
  u460 0.03003

counterfactual_positive_logit_margin_mean:
  u451 -8.352
  u455 -5.807
  u460 -1.622

counterfactual_positive_top1_match:
  u451 0.0
  u455 0.1
  u460 0.0

vtrace_rho_p99 remains high:
  u451 about 1.48M
  u460 about 118k
```

S1 p4 eval:

```text
Artifact:
  runs/b1_s1_trainablelive_absorb_fp32_u450_to_u460_20260428/eval/b1_matrix_trainablelive_absorb_fp32_u455_u460_s1_p4_20260428

u455 vs B1:
  3/8 = 0.375
  pair classes: 2-0=0, 1-1=3, 0-2=1

B1 vs u455:
  4/8 = 0.5
  all 1-1

u460 vs B1:
  3/8 = 0.375
  pair classes: 2-0=0, 1-1=3, 0-2=1

B1 vs u460:
  5/8 = 0.625 for B1
  pair classes: 2-0=2, 1-1=1, 0-2=1
```

Interpretation:

```text
The FP32 u460 absorption branch did move the full model numerically toward the
counterfactual labels, but it did not produce closed-loop B1 improvement.

This rejects the immediate full-model absorption setup. It strengthens the
current decision: train/evolve frozen-B1 residual policies directly first,
confirm them as hard negatives, and only later revisit main-policy
consolidation with a better objective or architecture.
```

## 2026-04-28 trainable-live residual rush loop, mine1

Ran the end-to-end residual evolution loop with the new `trainable-live`
trainer:

```text
uv run python python/scripts/b1_rush_residual_loop.py
  --trainer trainable-live
  --tag trainlive_mine1_20260428a
  --screen-pairs 8
  --confirm-pairs 16
  --mine-pairs 16
  --mine-target-states 80
  --mine-targets-per-pair 4
  --mine-actions-per-state 4
  --mine-max-forced-replays 320
  --mine-stop-after-positives 1
  --mine-target-seed 42831
  --train-steps 800
  --lr 0.0003
  --alpha 0.1
  --residual-l2-coef 0.001
  --margin 2.0
  --validation-fraction 0.2
  --early-stop-adoption-rate 0.9
  --early-stop-mean-positive-prob 0.70
  --early-stop-check-every 25
  --confirm-if-screen-ge 0.58
```

Mining result:

```text
Artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_rush_trainlive_mine1_20260428a

target states searched: 64
attempted forced replays: 231
forced misses: 0
positive labels: 0
winner flips: 0
margin positives: 0
max score_delta: 0.085
mean score_delta: -0.0256
```

Interpretation:

```text
This did not discover a new positive label. It says this particular quick
target slice was not a useful expansion of the counterfactual dataset.
Do not read it as evidence that no more positives exist; the search was small
and still mostly one-step.
```

Trainable-live residual result:

```text
Artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_trainlive_mine1_20260428a

labels used: 10 existing positives
stopped_at_step: 75
adoption: 10/10
base_param_delta_l2: 0.0
train_loss_initial: 15.5408
train_loss_final: 1.5908
```

S1 closed-loop eval:

```text
p8 screen, residual as focal:
  artifact: runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_trainlive_mine1_20260428a_focal_p8
  residual vs B1: 13/16 = 0.8125
  pair classes: 2-0=5, 1-1=3, 0-2=0
  residual family drift: 0.0324

p16 confirm, residual as focal:
  artifact: runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_trainlive_mine1_20260428a_confirm_focal_p16
  residual vs B1: 24/32 = 0.75
  pair classes: 2-0=8, 1-1=8, 0-2=0
  residual family drift: 0.0311

p16 confirm, residual as opponent:
  artifact: runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_trainlive_mine1_20260428a_confirm_asopp_p16
  B1 vs residual: 2/32 = 0.0625
  residual implied score: 30/32 = 0.9375
  B1 pair classes: 2-0=0, 1-1=2, 0-2=14
  residual family drift: 0.0290
```

Decision:

```text
This is a second confirmed trainable-live frozen-B1 residual B1 exploiter.
It did not add label diversity, but it independently retrained from the same
10-label set and reproduced strong S1 B1 pressure with zero base drift.

Registered it as an explicit hard-negative candidate:
  b1_residual_trainable_live_mine1_retrain_l2x

Do not resume full-model main absorption yet. The last u460 branch showed
label-metric movement without closed-loop gain. The next main-policy attempt
should either use a residual/adaptor architecture for the main branch or a
stronger distillation/consolidation target from confirmed residual behavior,
not generic full-model S1 IMPALA.
```

P32 confirmation:

```text
Focal artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_trainlive_mine1_retrain_confirm_focal_p32_20260428

Residual as focal vs B1:
  52/64 = 0.8125
  pair classes: 2-0=20, 1-1=12, 0-2=0
  prob_gt_half: 1.0
  residual family change rate: 0.0318

Reverse artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_trainlive_mine1_retrain_confirm_asopp_p32_20260428

B1 as focal vs residual opponent:
  B1 score: 7/64 = 0.109375
  residual implied score: 57/64 = 0.890625
  B1 pair classes: 2-0=0, 1-1=7, 0-2=25
  prob_lt_half for B1: 1.0
  residual family change rate: 0.0308
```

Interpretation:

```text
The trainlive_mine1 retrain is confirmed at p32 in both orientations. It is
stronger evidence than the p16 loop result and should be treated as a real
S1 B1 exploiter / hard-negative artifact.

The main branch has not learned this yet. The next fast path is not another
generic full-model u460 run; it is either:
  1. a main-policy residual/adaptor branch that starts identity-equal to B1 and
     learns against confirmed residual hard negatives, or
  2. a supervised consolidation/behavior cloning pass on traces from confirmed
     residual exploiters before RL.
```
## 2026-04-28 main-residual training path smoke

Goal: stop trying to train a full u460-style model away from B1, and instead make the learner itself an identity-preserving frozen-B1 residual policy. This keeps the B1 base frozen while training only the residual head.

Code changes:

- Added `training.main_residual_policy` config support with:
  - `enabled`
  - `base_snapshot_path`
  - `initial_residual_state_path`
  - `public_heuristic_bias_scale`
  - `hidden_dim`
  - `alpha`
  - `residual_mode`
  - `gate_bias`
- Added train-time construction of `TrainableLiveFrozenB1Residual` in `python/scripts/train.py`.
- Added optional warmstart from an existing `residual_state.pt`.
- Changed IMPALA optimizer construction to optimize only `requires_grad=True` parameters, so frozen B1 base weights are not included.
- Main-residual checkpoint updates now skip normal snapshot-registry insertion/promotion, because wrapper checkpoints are not ordinary `PolicyValueModel` snapshots.
- Added `python/scripts/b1_extract_main_residual_state.py` to extract `residual_probe.*` from a main-residual training checkpoint back into `residual_state.pt` format for existing closed-loop eval scripts.
- Added presets:
  - `configs/presets/pass3_b1_s1_main_residual_vs_trainablelive_hardnegatives_smoke.yaml`
  - `configs/presets/pass3_b1_s1_main_residual_warmstarted_vs_trainablelive_hardnegatives_smoke.yaml`

Validation:

```text
uv run python -m py_compile python/scripts/train.py python/weiss_rl/residual_policy.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/weiss_rl/learners/impala_learner.py python/scripts/b1_extract_main_residual_state.py

uv run pytest python/weiss_rl/tests/test_runtime.py::test_trainable_live_residual_freezes_base_but_keeps_residual_gradients python/weiss_rl/tests/test_runtime.py::test_load_residual_opponent_model_wraps_frozen_base python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_residual_policy python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies -q
```

Result:

```text
4 passed
```

Zero-initialized main residual smoke:

```text
run: runs/b1_main_residual_smoke_u1f_20260428
updates: 1
num_envs: 2
unroll_length: 8
completed: yes
counterfactual_positive_prob_mean: 0.000644
counterfactual_positive_top1_match: 0.0
extracted residual eval p8 vs B1 S1:
  residual focal score: 8/16 = 0.50
  pair classes: 0x 2-0, 8x 1-1, 0x 0-2
```

Interpretation: zero residual main-policy training path is mechanically valid, but one live IMPALA update from zero is not enough to learn the counterfactual residual labels.

Warmstarted main residual smoke:

```text
initial_residual_state_path:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_trainlive_mine1_20260428a/residual_state.pt

run:
  runs/b1_main_residual_warm_smoke_u1_20260428

completed:
  yes

training metrics:
  counterfactual_positive_prob_mean: 0.588427
  counterfactual_positive_top1_match: 0.70
  counterfactual_positive_margin_loss: 0.0
```

Extracted warmstarted checkpoint:

```text
runs/b1_main_residual_warm_smoke_u1_20260428/eval/main_residual_extracted_u1/residual_state.pt
```

Closed-loop p8, residual as focal:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_u1_p8_20260428

score:
  residual vs B1: 13/16 = 0.8125

pair classes:
  2-0: 5
  1-1: 3
  0-2: 0

prob_gt_half:
  1.0
```

Closed-loop p8, residual as opponent:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_u1_asopp_p8_20260428

score:
  B1 vs residual: 1/16 = 0.0625
  implied residual score: 15/16 = 0.9375

pair classes from B1 perspective:
  2-0: 0
  1-1: 1
  0-2: 7

prob_lt_half for B1:
  1.0
```

Conclusion:

```text
Confirmed: the learner can now be a frozen-B1 trainable residual main policy.
Confirmed: warmstarted main-residual checkpoints preserve the B1-exploiting behavior after a live training update.
Not yet confirmed: longer self-play/league improvement, because the local smoke disabled diverse residual opponents and used only one update.
```

Next:

```text
1. Run a short u5/u10 warmstarted main-residual branch with residual hard-negative opponents enabled, after fixing/confirming residual-opponent preload in the local runtime.
2. Extract residual_state.pt at each checkpoint.
3. Evaluate p16/p32 S1 B1 both directions.
4. Only if p16/p32 stays positive, move to school Linux multi-GPU with the same warmstarted residual-main config.
```

### Hard-negative local follow-up

Issue found:

```text
No-registry/no-promotion smoke refreshes cleared _opponent_models, which wiped configured residual opponents after preload.
```

Patch:

```text
python/weiss_rl/runtime.py
  refresh_opponent_pool now reloads configured residual_opponent_policies even when no live registry is available.
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/runtime.py
uv run pytest python/weiss_rl/tests/test_runtime.py::test_configured_resident_opponent_policy_ids_include_residual_specs python/weiss_rl/tests/test_runtime.py::test_load_residual_opponent_model_wraps_frozen_base -q

2 passed
```

Run:

```text
runs/b1_main_residual_warm_hardneg_u3d_20260428

command shape:
  warmstarted main residual
  max_updates=3
  num_envs=2
  unroll_length=8
  checkpoint_interval_updates=1
  diverse_opponent_actor_count=-1
  diverse_model_actor_count=1
  diverse_opponent_policy_ids=["b1_residual_trainable_live_mine1_retrain_l2x"]

completed:
  yes
```

Important training metrics at update 3:

```text
pfsp_residual_opponent_envs: 6
counterfactual_positive_prob_mean: 0.910135
counterfactual_positive_top1_match: 1.0
counterfactual_positive_loss: 0.095318
vtrace_rho_p99: 1.95145
decision/timeouts/errors: clean
```

Extracted checkpoint:

```text
runs/b1_main_residual_warm_hardneg_u3d_20260428/eval/main_residual_extracted_u3/residual_state.pt
```

Closed-loop p16, residual as focal:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_hardneg_u3_p16_20260428

score:
  residual vs B1: 26/32 = 0.8125

pair classes:
  2-0: 11
  1-1: 4
  0-2: 1

prob_gt_half:
  1.0
```

Closed-loop p16, residual as opponent:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_hardneg_u3_asopp_p16_20260428

score:
  B1 vs residual: 3/32 = 0.09375
  implied residual score: 29/32 = 0.90625

pair classes from B1 perspective:
  2-0: 0
  1-1: 3
  0-2: 13

prob_lt_half for B1:
  1.0
```

Conclusion:

```text
The warmstarted main-residual learner can train for live updates against the confirmed residual hard negative and keep strong S1 B1 improvement.
This is now a credible short-run school-box candidate.
Next local step if time permits: p32 confirmation of u3 or a u10 local continuation.
Next server step: run the same warmstarted main-residual config with larger env count and periodic extraction/eval.
```

### P32 confirmation of u3 hard-negative branch

Closed-loop p32, residual as focal:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_hardneg_u3_p32_20260428

score:
  residual vs B1: 60/64 = 0.9375

pair classes:
  2-0: 28
  1-1: 4
  0-2: 0

prob_gt_half:
  1.0
```

Closed-loop p32, residual as opponent:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_main_residual_warm_hardneg_u3_asopp_p32_20260428

score:
  B1 vs residual: 4/64 = 0.0625
  implied residual score: 60/64 = 0.9375

pair classes from B1 perspective:
  2-0: 0
  1-1: 4
  0-2: 28

prob_lt_half for B1:
  1.0
```

Decision:

```text
The u3 warmstarted main-residual hard-negative branch is confirmed at p32 both directions.
Local Windows speed is now the bottleneck, not evidence.
Use this as the school-box launch candidate.
```

### Fast residual diversity screen after p32 confirmation

Goal:

```text
Try one or two cheap Windows-only residual variants before server scaling.
Do not start another slow mining loop unless a variant actually beats the current p32 candidate.
```

Rejected alpha 0.15 screen:

```text
command:
  uv run python python/scripts/b1_rush_residual_loop.py --trainer trainable-live --tag trainlive_alpha015_nomine_20260428 --skip-mining --screen-pairs 8 --confirm-pairs 16 --confirm-if-screen-ge 0.80 --train-steps 800 --lr 0.0003 --alpha 0.15 --residual-l2-coef 0.0005 --margin 2.0 --validation-fraction 0.2 --early-stop-adoption-rate 0.9 --early-stop-mean-positive-prob 0.75 --early-stop-check-every 25 --device cuda:0

result:
  label adoption: 10/10
  screen p8 residual vs B1: 11/16 = 0.6875
  pair classes: 3x 2-0, 5x 1-1, 0x 0-2

decision:
  reject as an improvement; too weak versus the current u3 p32 profile.
```

Accepted diversity alpha 0.10 / lower-L2 screen:

```text
command:
  uv run python python/scripts/b1_rush_residual_loop.py --trainer trainable-live --tag trainlive_alpha010_l2low_nomine_20260428 --skip-mining --screen-pairs 8 --confirm-pairs 16 --confirm-if-screen-ge 0.80 --train-steps 800 --lr 0.0003 --alpha 0.10 --residual-l2-coef 0.0005 --margin 2.0 --validation-fraction 0.2 --early-stop-adoption-rate 0.9 --early-stop-mean-positive-prob 0.75 --early-stop-check-every 25 --device cuda:0

artifacts:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_trainlive_alpha010_l2low_nomine_20260428
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_trainlive_alpha010_l2low_nomine_20260428_confirm_focal_p16
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_closed_loop_rush_trainlive_alpha010_l2low_nomine_20260428_confirm_asopp_p16

result:
  label adoption: 10/10
  focal p16 residual vs B1: 28/32 = 0.875
  focal pair classes: 12x 2-0, 4x 1-1, 0x 0-2
  reverse p16 B1 vs residual: 4/32 = 0.125
  implied residual score: 28/32 = 0.875
  reverse pair classes from B1 perspective: 0x 2-0, 4x 1-1, 12x 0-2

decision:
  keep as a confirmed diversity hard negative, not as the best warmstart.
  The current best launch candidate remains main-residual warm hardneg u3 with p32 60/64 both directions.
```

Config update:

```text
file:
  configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives.yaml

added policy:
  b1_residual_trainable_live_alpha010_l2low

state:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_trainlive_alpha010_l2low_nomine_20260428/residual_state.pt
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_rush_residual_loop.py python/scripts/b1_trainable_residual_policy.py python/scripts/train.py python/weiss_rl/residual_policy.py python/weiss_rl/runtime.py

uv run pytest python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_residual_opponent_policies python/weiss_rl/tests/test_runtime.py::test_configured_resident_opponent_policy_ids_include_residual_specs python/weiss_rl/tests/test_runtime.py::test_load_residual_opponent_model_wraps_frozen_base -q

manual config load:
  residual_opponent_policies = 8
  new policy_id = b1_residual_trainable_live_alpha010_l2low
```

Two-hard-negative runtime smoke:

```text
command:
  uv run python python/scripts/train.py --stack-config configs/presets/pass3_b1_s1_main_residual_warmstarted_vs_trainablelive_hardnegatives_smoke.yaml --run-label b1_main_residual_warm_hardneg2_u1_20260428 --max-updates 1 --num-envs 2 --unroll-length 8 --checkpoint-interval-updates 1 --device cuda:0 --runtime-mode train_ordered --override system.collection_backend='"auto"' --override training.diverse_opponent_actor_count=-1 --override training.diverse_model_actor_count=1 --override training.diverse_opponent_policy_ids='["b1_residual_trainable_live_mine1_retrain_l2x","b1_residual_trainable_live_alpha010_l2low"]' --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425

result:
  run: runs/b1_main_residual_warm_hardneg2_u1_20260428
  completed: update 1
  pfsp_residual_opponent_envs: 6
  counterfactual_positive_prob_mean: 0.588427
  counterfactual_positive_top1_match: 0.70
  vtrace_rho_p99: 2.423577
  errors/timeouts: none observed

decision:
  The widened residual hard-negative lane runs locally.
  For Windows, enough evidence exists; scale the p32-confirmed u3 path and the diversity hard-negative config on the school Linux box.
```

### Residual league automation patch

Added bigger label-mining and automatic residual-league orchestration scripts:

```text
python/scripts/b1_big_label_miner.py
python/scripts/b1_residual_league_auto.py
```

`b1_big_label_miner.py` is a clustered wrapper around the existing in-process counterfactual miner. It does not replace `b1_counterfactual_labels.py`; it runs it across label clusters while carrying forward exclusion files so we do not rediscover the same positive forever.

Built-in clusters:

```text
pass_overextend:
  target main_play_character states, require pass legal, require baseline main_play_character.

main_nonpass:
  target main_play_character/main_move states, exclude pass action/family.

attack_climax:
  target attack, climax_play, main_play_event.

level_clock:
  target level_up, clock_from_hand.

broad_twostep:
  broad high-impact target set with small two-step beam enabled.
```

Dry-run validation:

```text
uv run python python/scripts/b1_big_label_miner.py --tag dryrun_20260428 --quick --dry-run --clusters pass_overextend,main_nonpass --device cuda:0
```

Result:

```text
commands generated correctly
existing aggregate label set:
  labels: 10
  winner_flip_labels: 10
  margin_positive_labels: 10
  unique_episode_seeds: 6
  unique_legal_fingerprints: 8
  positive_families:
    pass: 8
    main_move: 2
```

`b1_residual_league_auto.py` implements the server-side loop:

```text
1. write generated iteration config
2. train warmstarted main residual for N updates
3. extract residual_state.pt from checkpoint_N.pt
4. screen eval extracted residual vs B1 S1
5. if screen passes, confirm eval with --confirm-pairs
6. optionally reverse-eval as opponent
7. promote extracted residual state into next iteration's hard-negative lane
8. write residual_league_state.json after every iteration
```

Important design choices:

```text
Promotes extracted residual states only, not full model checkpoints.
Keeps frozen B1 as the base.
Writes generated YAMLs under generated_presets/ instead of configs/ so the repo-root resolver does not mistake the artifact directory for the project root.
Uses screen-pairs for cheap filtering and confirm-pairs for promotion evidence.
```

Dry-run validation:

```text
uv run python python/scripts/b1_residual_league_auto.py --tag dryrun3_20260428 --iterations 1 --updates-per-iteration 1 --screen-pairs 1 --confirm-pairs 2 --dry-run --device cuda:0
```

Generated command sequence:

```text
train.py
b1_extract_main_residual_state.py
b1_residual_closed_loop_eval.py screen focal
b1_residual_closed_loop_eval.py confirm focal
b1_residual_closed_loop_eval.py confirm as opponent
```

Generated config loader check:

```text
root:
  C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl

opponents:
  8

initial residual:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_adoption_probe_rush_trainlive_mine1_20260428a/residual_state.pt
```

Tiny real plumbing smoke:

```text
uv run python python/scripts/b1_residual_league_auto.py --tag plumbing_smoke2_20260428 --iterations 1 --updates-per-iteration 1 --num-envs 2 --unroll-length 8 --checkpoint-interval-updates 1 --screen-pairs 1 --promote-threshold 0.75 --skip-reverse-eval --device cuda:0
```

Artifacts:

```text
runs/b1_residual_league_plumbing_smoke2_20260428_iter01
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_league_auto_plumbing_smoke2_20260428
```

Result:

```text
train block completed
residual extracted from checkpoint_1.pt
p1 closed-loop eval completed
residual_league_state.json written
no promotion expected from p1 smoke
```

Promotion-path plumbing smoke:

```text
uv run python python/scripts/b1_residual_league_auto.py --tag promotion_plumbing_20260428 --iterations 1 --updates-per-iteration 1 --num-envs 2 --unroll-length 8 --checkpoint-interval-updates 1 --screen-pairs 1 --confirm-pairs 1 --promote-threshold 0.0 --confirm-threshold 0.0 --no-require-positive-pair-score --skip-reverse-eval --device cuda:0
```

Important:

```text
This was not a strength claim. Thresholds were intentionally lowered to prove the promotion-state mutation path.
```

Result:

```text
train block completed
residual extracted
screen eval completed
confirm eval completed
residual_league_state.json appended a promoted extracted residual
opponent_count_next: 9
promoted: true
confirmed: true due to forced threshold smoke
```

Real quick label-miner smoke:

```text
uv run python python/scripts/b1_big_label_miner.py --tag quick_real_20260428 --quick --clusters main_nonpass --device cuda:0
```

Result:

```text
artifact:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_big_quick_real_20260428_main_nonpass

baseline B1-vs-B1:
  8/16 = 0.50
  pair classes: 8x 1-1

search:
  target_states: 24
  attempted_forced_replays: 72
  forced_misses: 0
  positive_labels: 0
  winner_flip_labels: 0
  margin_positive_labels: 0

decision:
  The clustered miner execution path works, including exclusion of existing labels and in-process forced replays.
  This tiny non-pass budget did not find new labels; label diversity still needs server-scale mining.
```

Real two-iteration residual-league smoke:

```text
uv run python python/scripts/b1_residual_league_auto.py --tag loop_smoke_2iter_20260428 --iterations 2 --updates-per-iteration 1 --num-envs 2 --unroll-length 8 --checkpoint-interval-updates 1 --screen-pairs 1 --promote-threshold 0.75 --skip-reverse-eval --device cuda:0
```

Artifacts:

```text
runs/b1_residual_league_loop_smoke_2iter_20260428_iter01
runs/b1_residual_league_loop_smoke_2iter_20260428_iter02
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_league_auto_loop_smoke_2iter_20260428
```

Iteration 1:

```text
train completed
extract completed
screen p1 residual vs B1: 1/2 = 0.50
promoted: false
next iteration initialized from extracted iter01 residual
```

Iteration 2:

```text
train completed
extract completed
screen p1 residual vs B1: 2/2 = 1.0
confirm p32 residual vs B1: 58/64 = 0.90625
pair classes: 26x 2-0, 6x 1-1, 0x 0-2
residual_family_drift_rate: 0.02943
promoted: true
confirmed: true
opponent_count_next: 9
promoted policy id:
  b1_residual_auto_loop_smoke_2iter_20260428_iter02
```

Conclusion:

```text
The automatic residual-league loop works end to end in a real Windows run:
  train -> extract residual -> screen -> confirm -> promote extracted residual -> update residual_league_state.json.

This is now ready for server-scale execution.
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_big_label_miner.py python/scripts/b1_residual_league_auto.py
```

Server recommendation:

```text
1. Run b1_big_label_miner.py first with broader budgets to grow labels.
2. Run b1_residual_league_auto.py with 5-10 iterations, larger env count, screen p16, confirm p32 or p64.
3. Promote only extracted residual states that pass confirm.
4. Keep final residual_league_state.json as the server-side artifact of evolving hard negatives.
```

## 2026-04-28 - final local residual eval packet for auto-promoted iter02

Candidate:

```text
policy_id:
  b1_residual_auto_loop_smoke_2iter_20260428_iter02

residual_state:
  runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_league_auto_loop_smoke_2iter_20260428/extracted/loop_smoke_2iter_20260428_iter02/residual_state.pt
```

New missing-confirm run:

```text
uv run python python/scripts/b1_residual_closed_loop_eval.py --stack-config configs/presets/pass3_b1_s1_main_vs_confirmed_residual_hardnegatives_cfaux.yaml --run-dir runs/b1_s1_distillonly_u450_to_u455_20260427 --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 --label-dir runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_candidate_reps_p8_t24_a6_tensor_20260427 --residual-state runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_residual_league_auto_loop_smoke_2iter_20260428/extracted/loop_smoke_2iter_20260428_iter02/residual_state.pt --pairs 32 --artifact-dir-name b1_final_residual_iter02_s1_reverse_p32_20260428 --seed-scope b1_final_residual_iter02_s1_reverse_p32_20260428 --device cuda:0 --public-heuristic-bias-scale 1.0 --action-rng-salt-mode physical --residual-as-opponent
```

Result:

```text
B1 as focal vs residual as opponent:
  B1 score:      9/64 = 0.140625
  residual score: 55/64 = 0.859375
  pair classes from residual perspective: 23x 2-0, 9x 1-1, 0x 0-2
  residual family drift rate: 0.02722
  engine errors/truncations/timeouts: 0/0/0
```

Additional sanity evals:

```text
S3 B1 focal p16:
  residual score: 16/32 = 0.50
  pair classes: 0x 2-0, 16x 1-1, 0x 0-2
  interpretation: S3 remains saturated, as expected.

S0 B1 focal p16:
  residual score: 9/32 = 0.28125
  pair classes: 1x 2-0, 7x 1-1, 8x 0-2
  interpretation: raw/no-bias surface remains weak, so do not frame this as raw-network improvement.

S1 B3 focal p16:
  residual score: 24/32 = 0.75
  pair classes: 8x 2-0, 8x 1-1, 0x 0-2

S1 B4 focal p16:
  residual score: 30/32 = 0.9375
  pair classes: 14x 2-0, 2x 1-1, 0x 0-2
```

Final local figure/report packet:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428/b1_residual_final_eval_summary.md
runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428/b1_residual_final_eval_summary.csv
runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428/b1_residual_final_eval_summary.json
runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428/b1_residual_final_eval.png
runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428/b1_residual_final_eval.pdf
```

Renderer added:

```text
python/scripts/b1_residual_final_eval_report.py
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_residual_final_eval_report.py
uv run python python/scripts/b1_residual_final_eval_report.py --run-dir runs/b1_s1_distillonly_u450_to_u455_20260427 --out-dir runs/b1_s1_distillonly_u450_to_u455_20260427/figures/b1_residual_final_eval_iter02_20260428 --artifact "S1 B1 focal p32=b1_residual_league_loop_smoke_2iter_20260428_iter02_confirm_focal_p32:lowbias_s1:B1:focal" --artifact "S1 B1 reverse p32=b1_final_residual_iter02_s1_reverse_p32_20260428:lowbias_s1:B1:reverse" --artifact "S3 B1 focal p16=b1_final_residual_iter02_s3_b1_focal_p16_20260428:official_s3:B1:focal" --artifact "S0 B1 focal p16=b1_final_residual_iter02_s0_b1_focal_p16_20260428:raw_s0:B1:focal" --artifact "S1 B3 focal p16=b1_final_residual_iter02_s1_b3_focal_p16_20260428:lowbias_s1:B3:focal" --artifact "S1 B4 focal p16=b1_final_residual_iter02_s1_b4_focal_p16_20260428:lowbias_s1:B4:focal"
```

Current conclusion:

```text
This is a real local positive result for the residual path:
  S1 B1 focal p32:   0.90625
  S1 B1 reverse p32: 0.859375
  S1 B3 p16:         0.75
  S1 B4 p16:         0.9375

It is not a raw/S0 improvement and not an S3 breakthrough:
  S3 B1 p16: 0.50 saturated
  S0 B1 p16: 0.28125

Thesis wording should call this a low-bias S1 frozen-B1 residual exploiter / hard-negative result,
not a broad raw-policy or S3-deployment champion yet.
```
