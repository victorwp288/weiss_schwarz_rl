# Artifacts

Retained thesis evidence lives in these top-level directories:

- `runs/`
- `diagnostics/`
- `vast_artifacts/`
- `run_logs/`

Treat retained outputs as read-only unless deliberately replacing a thesis
artifact. Run-tree requirements are defined in
[thesis_workflow.md](thesis_workflow.md).

This page records selected evidence retained in this checkout. Commands, layout
rules, and validation commands live in [thesis_workflow.md](thesis_workflow.md).

## Retained Runs

- `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- `runs/main_champion_hardneg_long_v1_u10_20260517_seg01`
- `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01`

The referenced trajectory-BC dataset run
`runs/trajectory_bc_direct_b2_b3_b4_win_64_20260516/` is still missing from
this checkout and should be restored from backup if full provenance recreation
is needed.

## Selected Paper-Ready Artifacts

Locked B1 NoLeague seed:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- source policy id: `policy_000003`
- update: `15`
- weights hash: `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`
- supporting artifacts: final-eval outputs and paper-readiness summary inside
  the run directory.

Selected main fixed-deck model:

- run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected alias: `main_league_selected`
- source policy id: `main_interp_repair_a015`
- update: `5`
- weights hash: `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- provenance runs:
  `runs/main_champion_hardneg_long_v1_u10_20260517_seg01` and
  `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01`.

Targeted confirm256 evidence for the selected main source checkpoint:

| Opponent | Wins | Games | Win rate |
| --- | ---: | ---: | ---: |
| B0 RandomLegal | 512 | 512 | 1.000000 |
| B1 NoLeague baseline | 322 | 512 | 0.628906 |
| B2 HeuristicPublic | 399 | 512 | 0.779297 |
| B3 HeuristicPublicAggro | 365 | 512 | 0.712891 |
| B4 HeuristicPublicControl | 382 | 512 | 0.746094 |

The selected final eval for `main_league_selected` writes B0-B4 plus B1
matrix artifacts, metagame summaries, replay verification, paper figures, and a
passing `paper_readiness_summary.json` in the selected main run directory.

The selected main checkpoint is an explicit interpolation between the
first champion/hard-negative u10 league checkpoint and a later rehearsal repair
checkpoint. It is positive against every imported learned champion/hard-negative
candidate in the 128-paired-seed panel, but that panel is supporting robustness
evidence rather than the headline selection criterion.

## Diagnostics

`diagnostics/` is intentionally limited to the report/search sidecars used by
the thesis discussion and figure checks.

## Ablation Summaries

`vast_artifacts/` keeps `exp028`, `exp029`, `main`, `nogru`, and `ppo`.

## Figure Trace

Restored thesis figure bundles are not part of every checkout. When present,
treat them as read-only evidence bundles and keep commands for regenerating
figures in [thesis_workflow.md](thesis_workflow.md).

## Historical Smoke Evidence

These 2026-05-12 smoke/probe runs are historical rebuild evidence. The named
run directories are not all present in this checkout.

- `rebuild_smoke_b1_20260512_v5` completed 1 B1 update at 547.05 samples/sec.
- `rebuild_smoke_main_20260512_v2` completed 1 main league update at 446.58
  samples/sec and imported the B1 anchor.
- Smoke eval on `rebuild_smoke_main_20260512_v2` resolved B0-B4 and wrote
  `eval/final_eval/summary.json`.
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
