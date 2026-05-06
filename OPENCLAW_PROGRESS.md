# OPENCLAW_PROGRESS.md

Overnight thesis baseline task started: 2026-05-06T00:13:45+02:00
Working repo: /home/claw/autoresearch/wsrl/worktrees/rl__exp-023-early-b1-reserved-lane-ungated
Artifact root: artifacts/overnight-081-baseline-20260506-001345
Rule: prior runs/checkpoints/evals/figures/logs are locked backup artifacts; read-only for comparison/config reconstruction.

## Step 0.0 kickoff / workspace allocation

1. What was attempted
- Created fresh artifact root and progress file for the overnight run.
- Started from the prior WSRL autoresearch worktree because it contains the latest successful exp034/exp038 configs/fixes.

2. What passed
- Artifact directories created.

3. What failed
- None yet.

4. What was changed
- Created/overwrote this progress file in the working WSRL worktree.
- Created artifacts/overnight-081-baseline-20260506-001345/{manifests,reports,logs}.

5. What remains
- Freeze git state and locked old artifact manifest.
- Inspect versions, simulator, CUDA/PyTorch, GPU availability.
- Upgrade to weiss sim 0.8.1 and compatible CUDA PyTorch.

6. Exact commands and run ids
- RUN_ID=overnight-081-baseline-20260506-001345
- mkdir -p "artifacts/overnight-081-baseline-20260506-001345/manifests" "artifacts/overnight-081-baseline-20260506-001345/reports" "artifacts/overnight-081-baseline-20260506-001345/logs"
- cat > OPENCLAW_PROGRESS.md

## Step 0.1 repo/environment snapshot and locked manifest

1. What was attempted
- Captured git branch/status/remotes/HEAD and dependency metadata.
- Created a locked manifest of pre-existing run/eval/checkpoint/figure/log artifacts.

2. What passed
- Snapshot saved to artifacts/overnight-081-baseline-20260506-001345/reports/phase0_repo_env_snapshot.txt.
- Locked artifact manifest saved to artifacts/overnight-081-baseline-20260506-001345/manifests/LOCKED_PREVIOUS_ARTIFACTS.md.

3. What failed
- None in this step.

4. What was changed
- Added read-only manifest/report files under the new artifact root.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Inspect current package imports/versions and CUDA via Python.
- Preserve current uncommitted WSRL work safely before dependency/config edits.

6. Exact commands and run ids
- RUN_ID=overnight-081-baseline-20260506-001345
- git status --short --branch; git rev-parse HEAD; grep dependency lines; find old artifacts.

## Step 0.2 live package/CUDA import check

1. What was attempted
- Checked NVIDIA tooling candidates and live Python package imports through uv.
- Queried torch CUDA availability/device details and weiss_sim import/version.

2. What passed
- Report saved to artifacts/overnight-081-baseline-20260506-001345/reports/phase0_live_cuda_import_check.txt.

3. What failed
- See report for any missing NVIDIA CLI/Python import issues.

4. What was changed
- Updated OPENCLAW_PROGRESS.md only.

5. What remains
- If CUDA/imports pass: run smoke tests.
- If CUDA/imports fail: fix PyTorch/CUDA/sim compatibility before training.

6. Exact commands and run ids
- uv run python package/CUDA probe.

## Step 0.3 preserved pre-overnight uncommitted WSRL work

1. What was attempted
- Committed the pre-existing autoresearch runtime fix and exp023-exp035 config files before making dependency or clean-baseline changes.

2. What passed
- Local git commit created if staged changes existed.

3. What failed
- None.

4. What was changed
- Git history on local branch now has a preservation checkpoint.
- OPENCLAW_PROGRESS.md updated.

5. What remains
- Upgrade active environment to simulator 0.8.1 and PyTorch CUDA build compatible with RTX 5080 / sm_120.

6. Exact commands and run ids
- git add runtime/config preservation set
- git commit -m "chore: preserve autoresearch runtime and config state"
- HEAD after step: b8cbf14790b7ee9e0b55a9156c5fdf13cada726e

## Step 1.1 dependency upgrade to weiss-sim 0.8.1 + CUDA 12.8 PyTorch

1. What was attempted
- Moved weiss-sim 0.8.1 into main project dependencies so the training env cannot silently keep 0.7.0.
- Updated PyTorch from 2.5.1/cu124 to 2.9.1/cu128 for RTX 5080 / sm_120 compatibility.
- Re-locked and synced the uv environment.
- Ran a live CUDA matmul and import check.

2. What passed
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase1_dependency_upgrade.log.

3. What failed
- See log if uv/PyTorch/simulator checks failed.

4. What was changed
- Edited pyproject.toml.
- Updated uv.lock and .venv via uv lock/sync.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Run repo smoke tests and smallest train/eval smokes under the upgraded stack.

6. Exact commands and run ids
- uv lock --upgrade-package torch --upgrade-package weiss-sim
- uv sync
- uv run python CUDA/import probe

## Step 1.3-1.4 train/eval smoke failures diagnosed

1. What was attempted
- Retried train smoke with baseline no-league config and then tiny32 heuristic-opponent config.

2. What passed
- Simulator contract tests and wrapper dry-run already passed.
- Dependency/import/CUDA probes passed with torch 2.9.1+cu128 and weiss-sim 0.8.1.

3. What failed
- Standard auto-GPU and tiny32 league configs require a canonical B1 NoLeague baseline before training.
- Baseline no-league smoke failed with: missing opponent snapshot model for policy_id 'B2 HeuristicPublic'. This exposed a config/runtime mismatch where the no-league baseline inherited model-opponent behavior but attempted to route B2 as a model snapshot instead of simulator-native heuristic.

4. What was changed
- No code changed for this diagnostic; only fresh failed smoke run directories were created under the overnight run id.

5. What remains
- Create clean baseline configs that fix the B1/no-league bootstrapping issue directly.
- Run a smoke with explicit simulator-native heuristic opponent settings before starting B1 anchor training.

6. Exact commands and run ids
- Failed: runs/overnight-081-baseline-20260506-001345-cuda-train-smoke
- Failed: runs/overnight-081-baseline-20260506-001345-noleague-cuda-train-smoke
- Failed: runs/overnight-081-baseline-20260506-001345-tiny32-cuda-train-smoke
- Logs: artifacts/overnight-081-baseline-20260506-001345/reports/phase1_smoke_tests.log, phase1_train_eval_smoke_retry_noleague.log, phase1_train_eval_smoke_retry_tiny32.log

## Step 1.5 Makefile CUDA index compatibility update

1. What was attempted
- Updated Makefile pip fallback URLs from PyTorch cu124 to cu128 so non-uv setup does not install a CUDA build incompatible with RTX 5080/sm_120.

2. What passed
- Makefile edited.

3. What failed
- First attempt used bare python, which is unavailable in this shell; reran with uv run python successfully.

4. What was changed
- Makefile CUDA index fallback now points to https://download.pytorch.org/whl/cu128.

5. What remains
- Re-run smoke after clean config fixes.

6. Exact commands and run ids
- uv run python string replacement cu124 -> cu128 in Makefile.

## Step 2.2 clean config validation completed

1. What was attempted
- Re-ran config validation after quoting no-BC teacher_aux mode as a string.

2. What passed
- All clean sim-0.8.1 overnight configs loaded and produced hashes.
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase2_clean_config_validation.txt.

3. What failed
- Earlier no-BC validation failed because YAML parsed mode: off as boolean false; fixed by quoting it.

4. What was changed
- Updated configs/presets/overnight_081_clean_main_nobc_ablation_sim081.yaml.

5. What remains
- Smoke B1 config, then launch B1 anchor.

6. Exact commands and run ids
- uv run --extra dev --extra sim python config validation script.

## Step 2.4 clean config backend fix

1. What was attempted
- Fixed clean B1/main config family to use system.collection_backend=auto because train smoke showed process backend unsupported for the current single-node runtime path.
- Revalidated all clean configs.

2. What passed
- Config validation passed after backend fix.
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase2_clean_config_validation_after_backend_fix.txt.

3. What failed
- Prior B1 smoke failed before training with ValueError: system.collection_backend=process is not supported for the current runtime setup.

4. What was changed
- Added system.collection_backend: auto to B1 and main clean configs; derived ablations inherit the main fix.

5. What remains
- Retry clean B1 smoke.

6. Exact commands and run ids
- uv run --extra dev --extra sim python config validation script.

## Step 2.5 clean B1 train/eval smoke retry backend-auto

1. What was attempted
- Retried 1-update CUDA smoke using clean sim-0.8.1 B1/no-league config after switching collection_backend to auto.
- Ran tiny canonical eval smoke against policy_000001.

2. What passed
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase2_clean_b1_smoke_retry_backend_auto.log.

3. What failed
- Any failure is in the log.

4. What was changed
- Created fresh smoke run/eval directories under runs/overnight-081-baseline-20260506-001345-clean-b1-smoke2 if successful.

5. What remains
- Launch clean B1 anchor training if this smoke passed.

6. Exact commands and run ids
- train.py --stack-config configs/presets/overnight_081_clean_b1_noleague_sim081.yaml --run-label overnight-081-baseline-20260506-001345-clean-b1-smoke2 --max-updates 1
- eval.py --run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-smoke2 --policy-id policy_000001 --paired-seed-limit 2

## Step 2.6 dependency/config commit

1. What was attempted
- Committed dependency CUDA upgrade path and clean sim-0.8.1 overnight config family after smoke validation.

2. What passed
- Created commit f41fa48.
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase2_dependency_config_commit.txt.

3. What failed
- None.

4. What was changed
- Git commit includes Makefile, pyproject.toml, and clean overnight configs.

5. What remains
- Launch clean B1 anchor training.

6. Exact commands and run ids
- git commit -m "chore: add clean sim081 overnight baseline configs"

## Step 3.1 clean B1 anchor launched

1. What was attempted
- Launched clean sim-0.8.1 B1/no-league anchor training after smoke/eval passed.

2. What passed
- Startup contract passed; manifest written; training process is running under background session warm-trail.

3. What failed
- None at launch.

4. What was changed
- Active run directory: runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220.
- Training log: artifacts/overnight-081-baseline-20260506-001345/reports/phase3_clean_b1_anchor_train.log.

5. What remains
- Monitor to update 220.
- If B1 anchor completes, run final/targeted eval and then launch clean main League GRU with --b1-baseline-run-dir pointing to this run.

6. Exact commands and run ids
- train.py --stack-config configs/presets/overnight_081_clean_b1_noleague_sim081.yaml --run-label overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 --num-envs 4 --max-updates 220 --runtime-mode train_ordered --device cuda:auto --checkpoint-interval-updates 20

## Watchdog 00:43 B1 anchor status

1. What was attempted
- Checked active B1 anchor process, GPU state, checkpoint tracker, training metrics, and dev-eval artifacts.

2. What passed
- Clean B1 anchor is still running; latest observed training_metrics reached update 72/220.
- Checkpoint tracker latest checkpoint is update 60; best dev-eval checkpoint is update 25 with aggregate 0.8958333333333334.
- Dev eval summaries exist for updates 25 and 50.

3. What failed
- None requiring intervention.

4. What was changed
- Wrote watchdog status to artifacts/overnight-081-baseline-20260506-001345/reports/watchdog_0043_status.txt.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Continue monitoring until update 220 completes, then run final eval and launch clean main League GRU if B1 remains usable.

6. Exact commands and run ids
- Active run: runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220
- Active background session: warm-trail

## Watchdog 00:58 B1 anchor status

1. What was attempted
- Checked active B1 anchor process, GPU, checkpoint tracker, metrics, dev-eval summaries, and snapshot registry.

2. What passed
- Training is still running and reached at least update 162/220.
- GPU use is modest and no overlapping training was found.
- Latest checkpoint is update 160; best dev-eval remains update 25 aggregate 0.8958333333333334.

3. What failed / risk noted
- Later dev evals are weaker than update 25, with B1 anchor score drifting down: update 100 B1=0.5625, update 125 B1=0.4375, update 150 B1=0.375.
- This is not a hard failure yet because the run is still live and checkpoint_tracker preserves best update 25, but final B1 viability must be judged carefully before using it for main League GRU.

4. What was changed
- Wrote reports/watchdog_0058_status.txt and reports/watchdog_0058_b1_registry_inspection.txt.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Let current B1 finish to preserve complete evidence unless it stalls/fails.
- On completion, run final eval for both best and latest if supported/needed, then choose defensibly whether to use best checkpoint, rerun a safer B1, or proceed to main GRU.

6. Exact commands and run ids
- Active run: runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220
- Active background session: warm-trail

## Step 3.2 clean B1 anchor completed

1. What was attempted
- Waited for clean sim-0.8.1 B1/no-league anchor training to complete.

2. What passed
- Training completed 220/220.
- Latest checkpoint: checkpoint_220.pt / policy_version 11.
- Latest canonical B1 alias was persisted at update 220.
- Best dev-eval aggregate remained 0.8958333333333334 and update 200 matched aggregate 0.8958333333333334.

3. What failed / risk noted
- Mid-run dev evals dipped, especially B1-vs-B1-alias scores at updates 50/75/125/150, so final reporting must not pretend this run monotonically improved.
- The completed B1 is usable as a clean sim-0.8.1 anchor, but the thesis should report checkpoint selection/provenance clearly.

4. What was changed
- Wrote artifacts/overnight-081-baseline-20260506-001345/reports/phase3_clean_b1_completion_summary.txt.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Run B1 final eval/provenance check.
- Launch clean main League GRU with --b1-baseline-run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220.

6. Exact commands and run ids
- Completed run: runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220
- Background session warm-trail exited code 0.

## Step 3.3 clean B1 final eval

1. What was attempted
- Ran canonical final eval for clean B1 anchor latest snapshot policy_000011 with 16 paired seeds and B1 baseline run dir set to the same completed clean B1 run.

2. What passed
- See artifacts/overnight-081-baseline-20260506-001345/reports/phase3_clean_b1_final_eval.log.

3. What failed
- Any failure is in the log.

4. What was changed
- Created/updated runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220/eval/final_eval and diagnostics.

5. What remains
- Launch clean main League GRU using this B1 anchor.

6. Exact commands and run ids
- eval.py --run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 --policy-id policy_000011 --b1-baseline-run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 --paired-seed-limit 16

## Step 3.4 clean B1 final eval summary parsed

1. What was attempted
- Parsed clean B1 final eval summary.

2. What passed
- Summary written to artifacts/overnight-081-baseline-20260506-001345/reports/phase3_clean_b1_final_eval_summary.txt.

3. What failed
- None.

4. What was changed
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Launch clean main League GRU using runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 as B1 baseline anchor.

6. Exact commands and run ids
- Parsed runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220/eval/final_eval/summary.json.

## Step 4.1 clean main League GRU launched

1. What was attempted
- Launched clean sim-0.8.1 main League GRU using the completed clean B1 anchor.

2. What passed
- Startup contract passed; manifest written; B1 promotion anchor imported from runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220.
- Training is running under background session marine-fjord.

3. What failed
- None at launch.

4. What was changed
- Active run directory: runs/overnight-081-baseline-20260506-001345-clean-main-gru-u320.
- Training log: artifacts/overnight-081-baseline-20260506-001345/reports/phase4_clean_main_gru_train.log.

5. What remains
- Monitor early dev evals and B1/B2/B0 guardrails.
- On completion, run final/targeted evals and decide whether to launch ablations.

6. Exact commands and run ids
- train.py --stack-config configs/presets/overnight_081_clean_main_gru_sim081.yaml --run-label overnight-081-baseline-20260506-001345-clean-main-gru-u320 --b1-baseline-run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 --num-envs 4 --max-updates 320 --runtime-mode train_ordered --device cuda:auto --checkpoint-interval-updates 20

## Watchdog 01:13/01:16 main GRU early status

1. What was attempted
- Checked clean main League GRU process, GPU, registry, logs, and early files.

2. What passed
- Main GRU process is alive; no overlapping GPU training found.
- B1 anchor is imported and pinned in the main run registry.

3. What failed / risk noted
- No training metrics/checkpoints yet at the first check; this is still within expected startup/early-update window, so no intervention.

4. What was changed
- Wrote artifacts/overnight-081-baseline-20260506-001345/reports/watchdog_0113_status.txt and watchdog_0116_main_early_liveness.txt.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Continue monitoring until first checkpoint/dev eval; intervene only if it stalls beyond a reasonable window or guardrails collapse.

6. Exact commands and run ids
- Active run: runs/overnight-081-baseline-20260506-001345-clean-main-gru-u320
- Active background session: marine-fjord

## Watchdog 01:28 main GRU update-25 guardrail

1. What was attempted
- Checked clean main GRU early checkpoint, promotion gate, and waited for update-25 dev eval.

2. What passed
- Update-20 promotion gate passed for policy_000001.
- Update-25 dev eval aggregate is 0.8958333333333334.
- Champion sampling became active after policy_000001 was promoted; performance logs show pfsp_champion_pool_size=1 and nonzero champion envs.
- No overlapping GPU training found.

3. What failed / risk noted
- Throughput is slower than B1 anchor (~50 samples/sec), but the run is progressing.
- Need continued monitoring for B1 guardrail drift at later evals.

4. What was changed
- Wrote watchdog_0128_status.txt and watchdog_0128_main_update25_guardrail.txt.
- Updated OPENCLAW_PROGRESS.md.

5. What remains
- Continue clean main GRU to 320 unless guardrails collapse or it fails.
- On completion, run final/targeted evals, then launch matched ablations if time remains.

6. Exact commands and run ids
- Active run: runs/overnight-081-baseline-20260506-001345-clean-main-gru-u320
- Active background session: marine-fjord

## Step 4.2 guarded main GRU stopped; no-guard config added

1. What was attempted
- Inspected the first clean main GRU run after it exited with SIGTERM at update 50.
- Added and validated a clean main GRU variant with checkpoint_guard disabled.

2. What passed
- The stopped guarded run produced useful partial evidence: update 25 aggregate 0.8958333333333334 (B1 0.6875, B2/B0 1.0), update 50 aggregate 0.875 (B1 0.6875, B2 0.9375, B0 1.0), and champion/recent sampling was active.
- The no-guard config loads and hashes successfully.

3. What failed / risk noted
- The guarded run stopped after a conservative checkpoint_guard rollback at update 50 even though B1 had not collapsed. This makes it partial evidence only.

4. What was changed
- Added configs/presets/overnight_081_clean_main_gru_noguard_sim081.yaml.
- Committed the config in 6746fb7.
- Wrote artifacts/overnight-081-baseline-20260506-001345/reports/watchdog_0143_main_stopped_inspection.txt and phase4_noguard_config_validation.txt.

5. What remains
- Launch clean main GRU no-guard run from scratch using the same clean B1 anchor.

6. Exact commands and run ids
- Stopped partial run: runs/overnight-081-baseline-20260506-001345-clean-main-gru-u320
- New config: configs/presets/overnight_081_clean_main_gru_noguard_sim081.yaml

## Step 4.3 clean main GRU no-guard launched

1. What was attempted
- Launched clean sim-0.8.1 main League GRU no-guard run from scratch after the guarded run stopped at update 50.

2. What passed
- Startup contract passed; manifest written; B1 promotion anchor imported from runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220.
- Training is running under background session fast-nudibranch.

3. What failed
- None at launch.

4. What was changed
- Active run directory: runs/overnight-081-baseline-20260506-001345-clean-main-gru-noguard-u320.
- Training log: artifacts/overnight-081-baseline-20260506-001345/reports/phase4_clean_main_gru_noguard_train.log.

5. What remains
- Monitor first promotion/dev eval; continue to 320 unless real guardrail collapse/failure occurs.

6. Exact commands and run ids
- train.py --stack-config configs/presets/overnight_081_clean_main_gru_noguard_sim081.yaml --run-label overnight-081-baseline-20260506-001345-clean-main-gru-noguard-u320 --b1-baseline-run-dir runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220 --num-envs 4 --max-updates 320 --runtime-mode train_ordered --device cuda:auto --checkpoint-interval-updates 20

