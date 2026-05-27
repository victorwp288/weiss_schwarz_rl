# Thesis Workflow

This is the canonical operator surface for the rebuild. Use the package CLI for
normal work and keep `python/scripts/*` for compatibility and diagnostics.

## Setup

```powershell
uv sync --extra dev --extra sim
```

The simulator dependency is `weiss-sim>=1.1.0,<2`. Startup now checks the active
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

When using the locked thesis B1 artifact, `train-main --b1-run` resolves the
best-confirmed `selected_candidate` alias automatically and initializes from
that checkpoint. It does not use chronological `latest` as a substitute for
best-confirmed selection.

Locked B1 seed smoke:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main `
  --run-label main_from_locked_b1_smoke `
  --b1-run runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 `
  --profile smoke
```

Main league thesis launch:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_thesis_seed1 --b1-run runs/b1_thesis_seed1 --profile thesis-local
```

The current main league stack is
`configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml`.
It uses the factorized selected-B1 model surface, imports the explicit B1
baseline anchor, and filters seeded snapshot imports to pinned seed snapshots
so a locked selected seed does not accidentally drag an entire bootstrap lineage
into the new league pool. Its promotion/eval anchor set includes B0, B1, B2,
B3, and B4; the B1 anchor remains resident through reserved B1 exposure; and
actor checkpoint reloads are short enough that a selected continuation is not
hidden behind stale actor weights during segmented diagnostics.

Guarded selected main-league segment:

```powershell
uv run --extra dev --extra sim python python/scripts/guarded_league_bootstrap.py `
  --init-from-checkpoint runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/training/checkpoints/checkpoint_15.pt `
  --seed-snapshot-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 `
  --run-prefix guarded_main_selected_seed1 `
  --stack-config configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml `
  --segments 1 `
  --segment-updates 10 `
  --confirm-paired-seeds 64 `
  --first-init-schedule-offset-updates 0
```

The guarded controller records chronological latest, but advances only from the
published `main_league_selected` alias after targeted confirmation. Do not
extend a segment because loss improved or because it wrote a newer checkpoint.

Guarded guided-bootstrap league launch:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main-guided-bootstrap `
  --run-label main_guided_bootstrap_seed1 `
  --seed-run runs/b1_guided_seed_playstrong_factorized_auxfixed_clean_seed20260514_probe100_hashseed_20260515 `
  --init-from-checkpoint runs/main_guided_factorized_continuation_teacherfade_b2mix020_u50_20260515/training/checkpoints/best.pt `
  --profile league-probe
```

Use this only while the strict B1 NoLeague candidate gate is not cleared. It
starts from the best confirmed guided seed checkpoint, imports guided seed
snapshots at local update `0`, keeps the explicit mirror lane, and intentionally
does not require a strict `--b1-run` anchor. The `league-probe` profile stops
at update `50`, where the current failure mode is visible, instead of requiring
manual interruption of a 200-update run. Run `guard-run` after it completes,
then run `select_b1_candidate.py` or targeted confirmation before promoting any
checkpoint.

If the first guided-bootstrap probe shows large V-trace tails, use the
conservative V-trace-clipped stack instead of editing the generated command by
hand:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main-guided-bootstrap `
  --run-label main_guided_bootstrap_vtrace_seed1 `
  --seed-run runs/b1_guided_seed_playstrong_factorized_auxfixed_clean_seed20260514_probe100_hashseed_20260515 `
  --init-from-checkpoint runs/main_guided_factorized_continuation_teacherfade_b2mix020_u50_20260515/training/checkpoints/best.pt `
  --profile league-probe `
  --vtrace-clamp
```

If the first guided-bootstrap probe is off-policy healthy but drifts because no
trained snapshot passes the strict promotion gate, use the seed-champion
bootstrap probe. This marks imported seed snapshots as training-pool champions
only; it does not mark them as thesis champions or bypass `promotion_gate.json`.

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main-guided-bootstrap `
  --run-label main_guided_bootstrap_seedchampion_seed1 `
  --seed-run runs/b1_guided_seed_playstrong_factorized_auxfixed_clean_seed20260514_probe100_hashseed_20260515 `
  --init-from-checkpoint runs/main_guided_factorized_continuation_teacherfade_b2mix020_u50_20260515/training/checkpoints/best.pt `
  --profile league-probe `
  --seed-champions
```

Smoke eval:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
```

Thesis final eval:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/main_thesis_seed1 --b1-run runs/b1_thesis_seed1
```

Current selected-main final eval reproduction:

```powershell
$env:PYTHONHASHSEED='0'; uv run --extra dev --extra sim python python/scripts/eval.py `
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
uv run --extra dev python python/scripts/reward_component_probe.py `
  --stack-config configs/thesis/ablations/full_shaping_reward.yaml `
  --num-envs 64 `
  --steps 256 `
  --output-json runs/diagnostics/reward_components/full_shaping_reward.json
```

Use this before launching reward-shaping runs. It uses the simulator debug
output path to report the fixed reward components
`terminal, damage, level, board, no_progress`; the high-throughput training path
only carries scalar rewards.

Learning-progress diagnostic:

```powershell
uv run --extra dev python python/scripts/learning_progress_diagnostic.py `
  --run-dir runs/b1_reward_full_shaping_probe100_20260513
```

Run this after every long B1 or main training run before promoting an alias. It
summarizes reward scale, V-trace/off-policy health, route purity, checkpoint
selection, and periodic dev-eval trend so a run that peaked at update 50 and
stalled by update 100 is visible without manual log inspection.

Guarded league probe check:

```powershell
uv run --extra dev python -m weiss_rl.cli guard-run `
  --run-dir runs/main_league_probe `
  --max-vtrace-rho-p99 25
```

Use this after the first 25-50 updates of a main-league or guided-bootstrap
league probe. It wraps `learning_progress_diagnostic.py --league-guard`, writes
the same diagnostic JSON plus a `league_guard` section, and exits nonzero when
B2/B3/B4 latest periodic anchors collapse, promotion
gates repeatedly fail, no trained champion is admitted after enough attempts,
or the configured V-trace tail ceiling is exceeded. This is an experiment gate,
not a thesis result by itself.

B1 candidate selection:

```powershell
uv run --extra dev --extra sim python python/scripts/select_b1_candidate.py `
  --stack-config configs/thesis/b1_noleague.yaml `
  --run-dir runs/b1_thesis_seed1 `
  --output-json diagnostics/b1_candidate_selection_seed1.json
```

Use this before a B1 run is consumed by the league trainer. It ranks saved
checkpoints by B2/B3/B4 anchor performance, reports best-vs-latest falloff,
maps training policy ids to snapshot ids, and emits a deterministic
confirmation-eval command for the selected snapshot. If targeted confirmation
or checkpoint-guard confirmatory eval artifacts already exist, selector
eligibility and ranking use those higher-seed scores instead of the noisier
periodic dev-eval scores. Complete targeted-confirm artifacts are also allowed
to create candidates for checkpointed policies that fell between periodic
dev-eval intervals; this is how a confirmed update-90 checkpoint can be selected
without rewriting chronological `latest.pt`.

When comparing against the B2 heuristic baseline itself, pass a saved
anchor-vs-anchor targeted-confirm summary:

```powershell
uv run --extra dev python python/scripts/select_b1_candidate.py `
  --stack-config configs/thesis/b1_noleague.yaml `
  --run-dir runs/b1_thesis_seed1 `
  --reference-summary-json runs/main_guided_factorized_continuation_teacherfade_b2mix020_u50_20260515/eval/targeted_confirm128_b2_vs_b3b4_20260515/targeted_confirm128_summary.json `
  --reference-label "B2 HeuristicPublic reference" `
  --output-json diagnostics/b1_candidate_selection_seed1_reference_b2.json
```

`--publish-baseline-alias` is intentionally restricted to source runs marked
`experiment.role=baseline_noleague`. Guided public-teacher runs are useful
ablations, but they must not be published as the canonical B1 NoLeague anchor.
For guided/bootstrap candidates, use `--publish-selected-alias` with an explicit
alias such as `--selected-alias-policy-id guided_bootstrap_selected`; this pins
the selected checkpoint as a generic registry snapshot without changing
`latest.pt`, strict `best.pt`, or `b1_noleague_baseline`.

## Current Paper-Ready Artifacts

Locked B1 NoLeague seed:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- source policy id: `policy_000003`
- update: `15`
- weights hash: `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`
- report: `docs/b1_learning_rebuild_report_20260517.md`

Selected main fixed-deck model:

- run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected alias: `main_league_selected`
- source policy id: `main_interp_repair_a015`
- update: `5`
- weights hash: `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- report: `docs/main_league_rebuild_report_20260518.md`

Targeted confirm256 evidence for the selected main source checkpoint:

| Opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 |
| B1 NoLeague baseline | 322 | 512 | 0.628906 |
| B2 HeuristicPublic | 399 | 512 | 0.779297 |
| B3 HeuristicPublicAggro | 365 | 512 | 0.712891 |
| B4 HeuristicPublicControl | 382 | 512 | 0.746094 |

The canonical final eval for `main_league_selected` writes B0-B4 plus B1
matrix artifacts, metagame summaries, replay verification, paper figures, and a
passing `paper_readiness_summary.json` in the selected main run directory.

The current selected main checkpoint is an explicit interpolation between the
first champion/hard-negative u10 league checkpoint and a later rehearsal repair
checkpoint. It is positive against every imported learned champion/hard-negative
candidate in the 128-paired-seed panel, but that panel is supporting robustness
evidence rather than the headline selection criterion.

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

## Current Smoke Evidence

On 2026-05-12:

- `rebuild_smoke_b1_20260512_v5` completed 1 B1 update at 547.05 samples/sec.
- `rebuild_smoke_main_20260512_v2` completed 1 main league update at 446.58 samples/sec and imported the B1 anchor.
- Smoke eval on `rebuild_smoke_main_20260512_v2` resolved B0-B4 and wrote `eval/final_eval/summary.json`.
- Figure export wrote four PNG paper figures under `figures/paper/`.
- Full local verifier passed after the rebuild: `1205 passed, 2 skipped`.
- `phase2_b1_gpu_probe_20260512` completed 2 B1 updates on CUDA with
  `torch 2.11.0+cu128`, mean throughput 5634.83 samples/sec, max GPU memory
  2312 MB, and max GPU util 25%.
- `phase2_main_gpu_probe_20260512` completed 2 main league updates on CUDA,
  imported the B1 probe anchor, and reached mean throughput 5427.90 samples/sec.
- `phase2_b1_medium64_probe_20260512` completed 2 B1 updates with the medium64
  model at thesis-local shape, mean throughput 21001.79 samples/sec, max GPU
  memory 10474 MB, and max GPU util 69%.

These are plumbing and throughput smoke numbers only.
