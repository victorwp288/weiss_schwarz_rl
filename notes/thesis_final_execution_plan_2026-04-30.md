# Thesis final execution plan, 2026-04-30

## Main objective
Produce a defensible thesis data package under time pressure. We will give the v17 residual-guard main run a real chance, but we will not let it consume all remaining time if it stops showing signal.

## Current policy choices

### Locked final anchor
- Run: `runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429`
- Checkpoint: `training/checkpoints/checkpoint_120.pt`
- Confirm64: B1 `0.5078`, B3 `0.8750`, B4 `0.9922`, aggregate about `0.875`
- Role: strongest no-league/B1 anchor and likely final policy if league does not beat it.

### Best league/main fallback
- Run: `runs/thesis_main_candidate_v14_b1init_anchor_stabilize_20260430`
- Checkpoint: `training/checkpoints/checkpoint_160.pt`
- Confirm64: B1 `0.5000`, B3 `0.8594`, B4 `1.0000`, unweighted `0.8719`
- Role: best constrained league-style candidate so far if v17 fails.

### Current hail mary
- Run: `runs/thesis_main_candidate_v17e_b1_residual_guard_continue20_20260430`
- Starts from: v17d residual checkpoint u20
- Config: `configs/main_impala_league_server_v17_b1_residual_guard.yaml`
- Design: frozen B1 base + trainable residual adapter + B1 hard guard + B1 anchor-only rows + raw B1 distillation.
- Keep running only while B1 remains alive and B3/B4 do not collapse.

## v17 decision rule

At each available checkpoint:

1. Run B1-only confirm64 first.
2. If B1 is below `0.484`, stop v17 and pivot.
3. If B1 is `0.484-0.500`, run full confirm64 only if training metrics still look healthy.
4. If B1 is at least `0.500`, run full confirm64 and continue.
5. If full confirm shows B3/B4 collapse, stop or reduce pressure; do not spend hours on a dead branch.

Useful targets:
- Acceptable early: B1 >= `0.492`
- Strong: B1 >= `0.500`, B3 >= `0.875`, B4 >= `0.984`
- Thesis-useful improvement: v17 matches B1 on B1 and improves/holds B3/B4 versus B1 anchor or v14 u160.

## If v17 works

Use v17 as final main/constrained league model. Then run:

1. confirm128 or confirm256 for B1 v5 u120, v14 u160, and best v17 checkpoint.
2. Full anchor matrix against B0/B1/B2/B3/B4 plus important snapshots/champions if available.
3. Minimal v17 ablations if time permits, one change at a time:
   - no hard guard
   - no raw B1 distill
   - no B1 anchor-only rows
   - smaller/no residual alpha

## If v17 fails

Stop main-model iteration. Final story:

1. B1 v5 u120 is the primary final policy.
2. v14 u160 is the best league-style candidate.
3. v15/v16/v17 are ablations/negative results showing that unconstrained or partially constrained league updates still drift or fail to improve the anchor.

Then produce data package:

1. Confirm matrix on B1 v5 u120 and v14 u160.
2. Confirm matrix on representative failed/ablation checkpoints:
   - v14 u240 drift checkpoint
   - v15 u240 frozen trunk + KL
   - v16 u240 B1 anchor-only rows
   - best v17 checkpoint if any
3. Baselines/evals:
   - B0 RandomLegal
   - B2 HeuristicPublic
   - B3 HeuristicPublicAggro
   - B4 HeuristicPublicControl
4. Figures from `python/scripts/make_figures.py` and any matrix/report scripts.

## Config discipline

- Use the locked 512/server configs already in `configs/`.
- For any new ablation, copy from the closest final config and change only one conceptual setting.
- Save exact command lines and run labels in this notes file.
- Do not use the old queue script for final thesis runs unless absolutely necessary.

## Shutdown rule

Once final evals/figures are generated:

1. Commit/push the branch.
2. Copy/verify critical artifacts.
3. Shut down the Vast machine to avoid extra compute cost.
