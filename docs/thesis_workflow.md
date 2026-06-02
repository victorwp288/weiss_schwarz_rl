# Thesis Workflow

This is the canonical operator surface for the rebuild. Use the package CLI for
normal work; lower-level diagnostics now use package modules under `weiss_rl.*`.

This page owns commands and operating rules. Current evidence lives in
[artifacts.md](artifacts.md), config ownership in [configuration.md](configuration.md),
and run-tree requirements in [artifact_contract.md](artifact_contract.md).

## Setup

```powershell
uv sync --extra dev --extra sim
```

The simulator dependency is `weiss-sim>=1.2.0,<2`. Startup now checks the active
simulator version, required stepping APIs, and the three thesis deck presets.

## Standard Commands

B1 NoLeague smoke:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
```

B1 NoLeague thesis launch:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_thesis_seed1 --profile thesis-local
```

The canonical B1 config is still teacher-free and league-free. It now uses the
approved full reward shaping and action-surface guards, and its periodic dev
eval tracks B0/B2/B3/B4 so B1 quality is visible without manual side evals.

Local GPU probe:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_gpu_probe --profile gpu-probe
```

Main league smoke:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
```

`train-main --b1-run` resolves the canonical `b1_noleague_baseline` alias and
initializes from that checkpoint. It does not use chronological `latest` as a
substitute for an explicit B1 baseline alias.

Main league thesis launch:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_thesis_seed1 --b1-run runs/b1_thesis_seed1 --profile thesis-local
```

The public main league stack is `configs/thesis/main_league.yaml`. It uses the
same packed medium64 model surface as B1, imports the explicit B1 baseline
anchor, and keeps the fixed-deck thesis policy visible. The earlier guided
bootstrap/controller probes have been removed from the active workflow; their
historical results remain only in archived logs and report-retained artifacts.

Smoke eval:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
```

Smoke eval uses the packed `main_league.yaml` contract. Final eval uses the
selected factorized `final_eval.yaml` contract and should be paired with a
compatible selected-main run.

Thesis final eval:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/main_thesis_seed1 --b1-run runs/b1_thesis_seed1
```

Current selected-main final eval reproduction:

```powershell
$env:PYTHONHASHSEED='0'; uv run --extra dev --extra sim python -m weiss_rl.workflows.eval_entrypoint `
  --stack-config configs/thesis/final_eval.yaml `
  --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 `
  --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json `
  --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 `
  --policy-id main_league_selected `
  --policy-id "B0 RandomLegal" `
  --policy-id "B1 NoLeague baseline" `
  --policy-id "B2 HeuristicPublic" `
  --policy-id "B3 HeuristicPublicAggro" `
  --policy-id "B4 HeuristicPublicControl" `
  --paired-seed-limit 256 `
  --stage1-paired-seeds 64 `
  --max-paired-seeds 256 `
  --bootstrap-samples 2000
```

Figures:

```powershell
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_thesis_seed1 --format png --format pdf
```

B2 flatline/disagreement audit:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit `
  --run-dir runs/main_thesis_seed1 `
  --episodes-jsonl runs/main_thesis_seed1/eval/final_eval/episodes.jsonl `
  --policy-id policy_000200
```

Run this when B2 remains flat, suspiciously easy, or inconsistent with B0/B1
movement. It reruns the selected focal policy against `B2 HeuristicPublic`,
captures replay bundles, and writes the causal disagreement summary under
`eval/b2_disagreement/`.

Reward-component probe:

```powershell
uv run --extra dev python -m weiss_rl.diagnostics.reward_component_probe_entrypoint `
  --stack-config configs/thesis/b1_noleague.yaml `
  --num-envs 64 `
  --steps 256 `
  --output-json runs/diagnostics/reward_components/b1_noleague.json
```

Use this before launching or comparing B1 reward variants. It uses the simulator
debug output path to report the fixed reward components
`terminal, damage, level, board, no_progress`; the high-throughput training path
only carries scalar rewards.

Learning-progress diagnostic:

```powershell
uv run --extra dev python -m weiss_rl.diagnostics.learning_progress `
  --run-dir runs/b1_reward_full_shaping_probe100_20260513
```

Run this after every long B1 or main training run before promoting an alias. It
summarizes reward scale, V-trace/off-policy health, route purity, checkpoint
selection, and periodic dev-eval trend so a run that peaked at update 50 and
stalled by update 100 is visible without manual log inspection.

B1 candidate selection is currently manual in this checkout: use periodic
dev-eval, targeted confirmation, and the snapshot registry artifacts to choose
the explicit `b1_noleague_baseline` source run before launching `train-main`.
No standalone selector entrypoint is active in the package surface.

## Artifact State

Current selected runs, checkpoint hashes, final-eval evidence, figure traces,
and smoke/probe metrics live in [artifacts.md](artifacts.md). Keep this workflow
doc focused on commands and operating rules.

## Profiles

- `smoke`: CPU, 2 envs, unroll 4, 1 update, explicit B0-B4 smoke eval.
- `gpu-probe`: CUDA, 32 envs, unroll 16, 2 updates, simulator `fast`
  profile, and profile timers. Use this before a 200-update local thesis run.
- `league-probe`: CUDA, 288 envs, unroll 64, 50 updates, process collectors,
  simulator `fast` profile, profile timers, checkpoints every 5 updates. Use
  this for B1/main/guided-bootstrap early-collapse checks before a 200-update
  thesis launch.
- `thesis-local`: CUDA, 288 envs, unroll 64, 200 updates. Continue longer only
  after inspecting throughput, eval, and B2 diagnostics.
- `thesis-server`: CUDA, 4096 envs, unroll 64, process collectors.

Long thesis runs should use explicit run labels, preserved logs, and saved
checkpoints. Do not treat smoke results as model-quality evidence.

## Model Surface

The canonical B1 and main league thesis configs use the medium64 structured
model surface:

- `gru_hidden_size: 64`
- `encoder_mlp_width: 64`
- `typed_feature_width: 16`

This replaces the earlier tiny32 default for thesis training. Local probes on
2026-05-12 rejected 96/128-wide variants for routine local runs because they
ran close to the RTX 5080 VRAM ceiling.

## Fixed Deck Policy

- Focal model, B0, B1, and B2: `preset:main_deck_5hy_yotsuba_v1`.
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`.
- B4 control: `preset:control_deck_jj_s66_v1`.
- Multideck results are exploratory and must be labeled separately.
