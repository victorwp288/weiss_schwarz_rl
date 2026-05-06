# WSRL Emergency Handover

Date: 2026-05-06
Repo/worktree: `/home/claw/autoresearch/wsrl/worktrees/rl__exp-023-early-b1-reserved-lane-ungated`
Base branch before emergency branch: `autoresearch/exp-023-early-b1-reserved-lane-ungated-rl`
Base HEAD before handover commit: `6746fb7 chore: add clean main gru no-guard config`

## Current safety state

Victor asked to stop everything.

Stopped live WSRL/autoresearch work:
- Terminated the active `overnight-082-big-b1-gru128-robusteval-anchor-u220` training process and shell/tee wrappers.
- Disabled WSRL-related reminder/check crons:
  - `6df26252-3c84-42c6-b2cf-6a39ff0abd44` / `Check WSRL GRU128 B1 anchor midrun`
  - `f3673b03-1650-48af-9cfa-4bbef0cf2cbf` / `WSRL morning report reminder`
- Verification immediately after stopping showed no remaining WSRL/autoresearch/train/probe processes.

Do not launch new training or probes unless Victor explicitly asks.

## Important local artifact paths

These run artifacts are local and are not intended to be pushed to GitHub:

- `runs/exp-023-early-b1-reserved-lane-ungated-tier1b`
- `runs/exp-034-transplant-exp023-champion-b1-override-tier1b`
- `runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c`
- `runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220`
- `runs/overnight-082-big-b1-gru128-robusteval-anchor-u220`
- `artifacts/overnight-081-baseline-20260506-001345`
- `artifacts/overnight-082-big-b1-gru128-robusteval`

## Code/config improvements preserved on this branch line

The current branch contains the important local WSRL fixes/configs from the rescue work:

1. Runtime legal-action metadata reconstruction
   - File: `python/weiss_rl/runtime.py`
   - Reconstructs `legal_action_meta` from packed `legal_ids` when simulator/native packed layouts omit metadata.
   - This fixed structured candidate scoring / teacher-label paths that otherwise broke under packed legal-id simulator outputs.

2. RTX 5080 / CUDA compatibility update
   - Files: `pyproject.toml`, `Makefile`, `uv.lock`
   - Main deps now include `torch==2.9.1`, `weiss-sim==0.8.1`, PyTorch CUDA 12.8 index.
   - Makefile fallback URLs also point to cu128, not cu124.

3. Clean sim-0.8.1 overnight config family
   - Files under `configs/presets/overnight_081_*_sim081.yaml`
   - Includes clean B1/no-league anchor and clean main GRU/no-GRU/no-BC/reward-shaping ablation configs.
   - Uses safer `system.collection_backend: auto` where process backend was unsupported in current single-node setup.

4. Champion-pressure experiment configs
   - Files under `configs/presets/structured_acceptance_standard_auto_gpu_exp023_*` through `exp035_*`.
   - Capture the important lesson that fresh champion-pressure starts collapsed B1 unless the learner was initialized from the good exp023 weights.

5. New larger B1 robust-eval config
   - File: `configs/presets/overnight_082_big_b1_gru128_noleague_robust_eval.yaml`
   - GRU128/MLP128/typed32 B1/no-league candidate with robust B0/B2/B3/B4 dev anchors.
   - Current interrupted run did not beat prior best before Victor stopped everything.

6. Detailed scratch/progress log
   - File: `OPENCLAW_PROGRESS.md`
   - Contains the chronological overnight 081 rescue/progress details.

## Known-good findings and previous bests

### exp023: good B1 anchor, but champion-starved

Run: `runs/exp-023-early-b1-reserved-lane-ungated-tier1b`

Checkpoint tracker:
- Best/latest: update 220, policy_version 11
- Dev aggregate: `0.9479166666666666`
- B1: `0.84375`
- B2: `1.0`
- B0: `1.0`

Interpretation:
- Strong B1/B0/B2 anchor candidate.
- Not enough for champion/league robustness because champion sampling stayed starved (`pfsp_champion_envs=0`).
- This became the seed/learner base for later successful transplant work.

### exp028/029/031: champion lanes worked mechanically, but fresh learner collapsed B1

Important diagnosis:
- `exp028` fixed champion-starvation mechanics: model actor, warmstart off, diverse/model lanes active, champion/hard-negative envs active.
- But B1 collapsed to `0.625` at update 25.
- Adding small B1 mix (`exp029`) and reserved B1 lane (`exp031`) still collapsed B1 when starting from fresh learner.

Root cause:
- These runs imported exp023 snapshots only as opponents, not as learner initialization.
- The successful move was to transplant exp023 learner weights into the champion-pressure config.

### exp034: best overall thesis candidate so far

Run: `runs/exp-034-transplant-exp023-champion-b1-override-tier1b`

Core fix:
- Transplanted exp023 learner weights into the champion/B1-pressure config.
- Model actor, warmstart off, actor heuristic 0.
- B1 reserved lane active, champion pressure active, B2 training lanes disabled.

Checkpoint tracker:
- Best dev checkpoint: update 250 / policy_version 12 / `training/checkpoints/checkpoint_250.pt`
- Best dev aggregate: `0.9583333333333334`
- B1: `0.875`
- B2: `1.0`
- B0: `1.0`
- Latest checkpoint: update 320 / policy_version 16 / `training/checkpoints/checkpoint_320.pt`

Targeted champion/recent probe:
- Summary path: `runs/exp-034-transplant-exp023-champion-b1-override-tier1b/eval/champion_recent_targeted_probe_u320_fast32/targeted_probe_summary.json`
- Candidate `policy_000016` results:
  - vs B0: `32-0`
  - vs B1: `24-8`
  - vs B2: `32-0`
  - vs imported exp023 champion: `18-14`
  - vs `policy_000012`: `17-15`
  - vs `policy_000014`: `16-16`
  - vs `policy_000015`: `16-16`

Interpretation:
- `checkpoint_250.pt` is the best dev-eval checkpoint.
- `policy_000016` / checkpoint 320 is the better robustness candidate against champion/recent policies.
- Exp034 is the strongest thesis story so far: it preserved anchors and produced real self-play/league robustness, but it is balanced rather than dominant against recent policies.

### exp038: GRU32 continuation stable, but not a clean win

Run: `runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c`

Checkpoint tracker:
- Completed to update 420.
- Best dev checkpoint: update 325 / policy_version 16 / `training/checkpoints/checkpoint_325.pt`
- Best dev aggregate: `0.9583333333333334`
- Latest checkpoint: update 420 / policy_version 21 / `training/checkpoints/checkpoint_420.pt`

Dev eval signal:
- Updates 325, 350, 400 matched aggregate `0.9583333333333334` with B1 `0.875`, B2 `1.0`, B0 `1.0`.
- Update 375 dipped to aggregate `0.9375`, B1 `0.8125`.
- Latest/policy 21 promotion signal was lower than best.

Targeted probe:
- Partial summary path: `runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c/eval/champion_recent_targeted_probe_u420_fast32/targeted_probe_partial_summary.json`
- `policy_000020`: B0 `32-0`, B1 `26-6`, B2 `32-0`, imported exp034 champion `16-16`, policy19 `18-14`, policy21 `16-16`.
- `policy_000021`: B0 `32-0`, B1 `27-5`, B2 `32-0`; remaining champion/recent matchups repeatedly timed out/pathological.

Interpretation:
- GRU32 continuation is stable/viable after the bug fixes.
- Do not claim “GRU wins” from exp038.
- It is not a clean GRU-vs-no-GRU causal ablation, and it did not clearly improve over exp034.

### overnight-081 clean B1 anchor: completed but weaker

Run: `runs/overnight-081-baseline-20260506-001345-clean-b1-anchor-u220`

Checkpoint tracker:
- Completed to update 220 / policy_version 11.
- Best stayed update 25.
- Best dev aggregate: `0.8958333333333334`.

Interpretation:
- Valid clean baseline attempt, but weaker than exp023/exp034.
- Not the current best candidate.

### overnight-082 GRU128 B1 robust-eval anchor: interrupted and not better so far

Run: `runs/overnight-082-big-b1-gru128-robusteval-anchor-u220`

Command shape before stop:
- `train_ordered --num-envs 256 --max-updates 220 --device cuda:auto --checkpoint-interval-updates 25`
- Config: `configs/presets/overnight_082_big_b1_gru128_noleague_robust_eval.yaml`

State when stopped:
- Latest checkpoint: update 150 / policy_version 6.
- Best remained update 25.
- Best dev aggregate: `0.9140625`.
- Later evals observed lower than best: update 50 `0.875`, update 75 `0.90625`, update 100 `0.890625`.

Interpretation:
- Scaling model capacity to GRU128/MLP128 and using robust B0/B2/B3/B4 dev anchors did not beat exp034 before interruption.
- If resuming B1-anchor experimentation, consider a separate `train_async_fast` run with higher env count (`512` or `1024`) as a throughput/scale diagnostic, but do not confuse that with final thesis evidence.

## What to do next

Recommended priority order:

1. Package exp034 as the thesis-ready result first.
   - Freeze both `checkpoint_250.pt` as best dev-eval and `policy_000016` as robustness candidate.
   - Assemble final-eval/report artifacts cleanly.
   - Write the Results/README story around: anchor retention, champion pressure, transplant fix, targeted champion/recent robustness.

2. Only then decide whether another training run is worth it.
   - B1 anchor scale experiment may be useful: async_fast, 512/1024 envs, robust B0/B2/B3/B4 eval, hard stop if early evals regress.
   - League/champion final evidence should stay more conservative; ordered mode is cleaner and easier to defend.

3. If a clean GRU ablation is required, do a paired no-GRU continuation with the same stack/starting point.
   - Exp038 alone is not causal evidence that GRU helps.

## Commands useful for next operator

Verify no WSRL work is running:

```bash
pgrep -af 'weiss|wsrl|thesis_run|python/scripts/train|autoresearch|targeted_probe' || true
```

Inspect key trackers:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for run in [
    'exp-034-transplant-exp023-champion-b1-override-tier1b',
    'exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c',
    'overnight-082-big-b1-gru128-robusteval-anchor-u220',
]:
    p = Path('runs') / run / 'training/checkpoints/checkpoint_tracker.json'
    print('\n===', run)
    print(json.dumps(json.load(open(p)), indent=2)[:3000])
PY
```

Check current Git branch/status:

```bash
git status --short --branch
git log --oneline --decorate --max-count=10
```
