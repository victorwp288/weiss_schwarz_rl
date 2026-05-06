# Codex Handover Prompt

You are Codex taking over Victor's Weiss Schwarz RL thesis emergency handover.

Start here:

```bash
cd /home/claw/autoresearch/wsrl/worktrees/rl__exp-023-early-b1-reserved-lane-ungated
git status --short --branch
cat EMERGENCY_HANDOVER.md
```

Context:
- Victor asked Lallan to stop all live WSRL/autoresearch work, document the current findings, push an emergency branch to GitHub, and hand over.
- Do not launch training, probes, crons, or GPU-heavy jobs unless Victor explicitly asks.
- First verify nothing is running:

```bash
pgrep -af 'weiss|wsrl|thesis_run|python/scripts/train|autoresearch|targeted_probe' || true
```

Repository:
- Local worktree: `/home/claw/autoresearch/wsrl/worktrees/rl__exp-023-early-b1-reserved-lane-ungated`
- GitHub remote: `https://github.com/victorwp288/weiss_schwarz_rl.git`
- Emergency branch should already exist remotely after Lallan's handover push.

Read these files before acting:
1. `EMERGENCY_HANDOVER.md`
2. `OPENCLAW_PROGRESS.md`
3. `configs/presets/overnight_082_big_b1_gru128_noleague_robust_eval.yaml`
4. Relevant changed files from `git log --oneline origin/main..HEAD`

Current best experimental story:
- `exp034` is the strongest thesis candidate.
- Best dev-eval checkpoint: `runs/exp-034-transplant-exp023-champion-b1-override-tier1b/training/checkpoints/checkpoint_250.pt`, aggregate `0.9583333333333334`, B1 `0.875`, B2 `1.0`, B0 `1.0`.
- Robustness candidate: `policy_000016` / checkpoint 320 from exp034. Targeted champion/recent probe: B0 `32-0`, B1 `24-8`, B2 `32-0`, imported exp023 champion `18-14`, recent policies roughly tied.
- `exp038` GRU32 continuation is stable but not a clean improvement. Do not claim GRU wins.
- `overnight-082` GRU128 B1 run was stopped at Victor's request and had not beaten exp034 or even exp023 before interruption.

Recommended next task:
Package exp034 into a defensible thesis-ready result before doing any more training.

Concrete next steps:
1. Inspect exp034 artifacts and verify referenced metrics from disk.
2. Prepare a concise thesis result package/README/report section around:
   - the B1 collapse problem,
   - champion starvation diagnosis,
   - why fresh champion-pressure starts failed,
   - the exp023 learner-weight transplant fix,
   - exp034 anchor retention and champion/recent robustness,
   - caveat that robustness is balanced, not dominant.
3. If asked to run more experiments, recommend a very bounded plan first. Do not start silently.

Hard guardrails:
- Do not kill unrelated services or edit core assistant/workspace files.
- Do not push to main.
- Do not commit local `runs/` or large `artifacts/` output unless Victor explicitly asks.
- Do not claim results without checking the local JSON artifacts.
- Preserve B0/B1/B2 guardrails in any future evaluation/training plan.

Useful artifact paths:

```text
runs/exp-034-transplant-exp023-champion-b1-override-tier1b/training/checkpoints/checkpoint_tracker.json
runs/exp-034-transplant-exp023-champion-b1-override-tier1b/eval/champion_recent_targeted_probe_u320_fast32/targeted_probe_summary.json
runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c/training/checkpoints/checkpoint_tracker.json
runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c/eval/champion_recent_targeted_probe_u420_fast32/targeted_probe_partial_summary.json
runs/overnight-082-big-b1-gru128-robusteval-anchor-u220/training/checkpoints/checkpoint_tracker.json
```

Verification snippet:

```bash
python3 - <<'PY'
import json
from pathlib import Path
paths = [
 'runs/exp-034-transplant-exp023-champion-b1-override-tier1b/training/checkpoints/checkpoint_tracker.json',
 'runs/exp-034-transplant-exp023-champion-b1-override-tier1b/eval/champion_recent_targeted_probe_u320_fast32/targeted_probe_summary.json',
 'runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c/training/checkpoints/checkpoint_tracker.json',
 'runs/exp-038-gru32-gpu-robustness-continuation-from-exp034-u320-tier1c/eval/champion_recent_targeted_probe_u420_fast32/targeted_probe_partial_summary.json',
]
for p in paths:
    print('\n===', p)
    obj = json.load(open(Path(p)))
    print(json.dumps(obj, indent=2)[:4000])
PY
```
