# Artifacts

Retained thesis evidence lives in:

- `runs/`
- `diagnostics/`
- `vast_artifacts/`
- `thesis_figures_final/`

Treat retained outputs as read-only unless deliberately replacing a thesis
artifact.

## Current Retained Runs

- `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- `runs/main_champion_hardneg_long_v1_u10_20260517_seg01`
- `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01`

The referenced trajectory-BC dataset run
`runs/trajectory_bc_direct_b2_b3_b4_win_64_20260516/` is still missing from
this checkout and should be restored from backup if full provenance recreation
is needed.

## Current Diagnostics

`diagnostics/` is intentionally limited to the report/search sidecars used by
the thesis discussion and figure checks.

## Ablation Summaries

`vast_artifacts/` keeps `exp028`, `exp029`, `main`, `nogru`, and `ppo`.
