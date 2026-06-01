# Thesis Artifact Cleanup Plan - 2026-06-01

This plan started as a cautious cleanup map for runs, configs, checkpoints,
diagnostics, and thesis figures after the submitted thesis PDF:

`../Master_Thesis_RL_WeissSchwarz_Final_2026-06-01.pdf`

The current rule is: anything directly named in the thesis, required by a
thesis-named artifact, or needed to reproduce/defend a thesis table or figure is
protected.

## Inventory Snapshot

- Run directories scanned: 1,272
- Total `runs/` size: 121.8 GiB
- Protected run directories: 5
- Protected run size: 636.6 MiB
- Unprotected run directories: 1,267
- Unprotected run size: 121.2 GiB
- Inventory files written for this audit:
  - `temp/thesis_cleanup/thesis_text.txt`
  - `temp/thesis_cleanup/pdf_path_references.md`
  - `temp/thesis_cleanup/artifact_inventory.md`
  - `temp/thesis_cleanup/protected_artifact_manifest.json`
  - `temp/thesis_cleanup/archive_dry_run_runs_20260601.csv`
  - `temp/thesis_cleanup/archive_dry_run_runs_20260601.json`
  - `temp/thesis_cleanup/archive_move_results_20260601.csv`
  - `temp/thesis_cleanup/loose_artifact_archive_results_20260601.csv`
  - `temp/thesis_cleanup/top_level_archive_results_20260601.csv`
  - `temp/thesis_cleanup/cache_cleanup_results_20260601.csv`

## Cleanup Applied

The following cleanup was applied after the protected set was verified:

- Moved 1,267 unprotected `runs/*` directories to
  `../weiss_schwarz_rl_artifact_archive_20260601/runs/`.
- Moved 121.2 GiB of run directories containing 10,431 checkpoint files.
- Left exactly 5 run directories in active `runs/`, all thesis-protected.
- Moved loose unprotected files from `runs/` to
  `../weiss_schwarz_rl_artifact_archive_20260601/runs_loose_files/`.
- Moved non-protected `run_logs/*` files to
  `../weiss_schwarz_rl_artifact_archive_20260601/run_logs/`.
- Left 4 protected K4/search logs in active `run_logs/`.
- Moved ignored top-level generated outputs `dist/`, `now/`, `now.zip`, and
  `.VSCodeCounter/` to
  `../weiss_schwarz_rl_artifact_archive_20260601/ignored_top_level/`.
- Removed reproducible local caches `.mypy_cache/`, `.pytest_cache/`, and
  `.ruff_cache/`.

No thesis-protected run, diagnostic, seed, figure, or vast artifact was moved.
The active `runs/` tree is now 636.6 MiB and contains only the protected thesis
run directories plus `runs/README.md`.

## Protected Set

Keep these in place unless a later audit proves there is a byte-identical copy
elsewhere and the thesis workflow is updated to point at it.

### Runs

- `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
  - Thesis selected no-search main run.
  - Contains the selected snapshots, `targeted_confirm256_b0_b4`,
    `targeted_confirm256_imported_champions`, K4 confirm256 eval, paper figures,
    and readiness summaries.
- `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
  - Thesis B1 NoLeague baseline run and alias source.
- `runs/main_champion_hardneg_long_v1_u10_20260517_seg01`
  - Interpolation provenance source for selected A015 model.
  - The selected model manifest points to `training/checkpoints/checkpoint_10.pt`.
- `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01`
  - Interpolation provenance source for selected A015 model.
  - The selected model manifest points to `training/checkpoints/checkpoint_5.pt`.
- `runs/trajectory_bc_direct_b2_b3_b4_win_64_20260516`
  - Dataset referenced by manifests for the B1/main selected runs.

### Configs And Seeds

- `configs/seeds/report_eval_seeds.txt`
- `configs/presets/typed_thesis_locked.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_rehearsal_probe.yaml`

### Diagnostics

- `diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json`
- `diagnostics/god_search_readiness_summary_20260521.json`
- `diagnostics/main_search_readiness_summary_20260521.json`
- `diagnostics/god_search_scorecard_confirm256_k4_r1_terminal_strict_20260521.json`
- `diagnostics/main_search_first_second_balance_20260521.json`

### Reports And Docs

The thesis names these at top-level `docs/` paths, but the current refactor
branch has archived them under `docs/archive/reports/202605/`. Keep the archived
copies and avoid deleting compatibility references until docs are updated.

- `docs/archive/reports/202605/main_league_model_lock_20260521.md`
- `docs/archive/reports/202605/main_league_rebuild_report_20260521.md`
- `docs/archive/reports/202605/god_search_k4_lock_20260521.md`

### Figures And Vast Artifacts

- `thesis_figures_final/`
- `vast_artifacts/main/`
- `vast_artifacts/nogru/`
- `vast_artifacts/ppo/`

### Run Logs

- `run_logs/god_search_confirm64_k3_r1_terminal_s1_c3_20260521.out`
- `run_logs/god_search_confirm64_k4_r1_terminal_s1_c3_20260521.out`
- `run_logs/god_search_confirm128_k4_r1_terminal_s1_c3_20260521_rerun.out`
- `run_logs/god_search_confirm256_k4_r1_terminal_s1_c3_20260521.out`

## Missing Thesis-Named Artifacts

These are named in the thesis PDF but were not found in this checkout or its
immediate parent directory. Do not clean adjacent figure artifacts until these
are either recovered or intentionally marked external.

- `thesis_figures_final/main_search_20260521/RESULTS_TRACE.json`
- `figure_data_audit_20260528.json`
- `Kandidatspeciale/Figures/final_results_20260528/`

The current local `thesis_figures_final/main_search_20260521/` contains the PNG
exports, but not `RESULTS_TRACE.json`.

## Proposed Cleanup Policy

1. Keep the protected set above frozen.
2. Recover or intentionally externalize the three missing thesis-named figure
   sidecars before deleting or moving final figure data.
3. Archive by complete run directory, not by pruning checkpoints inside a run.
   This keeps manifests, configs, eval summaries, and checkpoint numbering
   coherent.
4. Keep protected run directories in place; do not compact them internally.
5. Treat `configs/` as source, not artifact trash. Config cleanup should be a
   separate pass using `rg` reference checks plus workflow tests.
6. Treat `diagnostics/`, `run_logs/`, `thesis_figures_final/`, and
   `vast_artifacts/` as thesis evidence until the missing sidecars are resolved.

## High-Value Archive Candidates

The largest unprotected runs are old exploratory/checkpoint-heavy directories.
These are good first archive candidates once the protected set is accepted:

- `runs/b1_anchor_native_rollout_20260423_1915` - 961.2 MiB
- `runs/b1_s1_distillonly_u450_to_u455_20260427` - 868.7 MiB
- `runs/b1_native_profilecycle_u200_s2_20260424` - 569.0 MiB
- `runs/b1_native_aggressive_profile_curve_u200_s2_20260424` - 566.1 MiB
- `runs/b1_continue_u20_b3pressure_aggbiasprofile_u120_s2_20260425` - 563.4 MiB
- `runs/b1_anchor_fastamp_rowunion_batchbuilder_env512_smoke` - 508.1 MiB
- `runs/b1_anchor_fastamp_rowunion_env512_smoke` - 507.6 MiB
- `runs/b1_anchor_fastamp_rowunion_batchbuilder_explicitopt_env512_smoke` - 507.5 MiB
- `runs/b1_continue_u20_actorlane_b3pressure_u100_s2_20260424` - 477.6 MiB
- `runs/b1_continue_u20_b3pressure_passpenalty002_u100_s2_20260425` - 475.8 MiB

The generated dry-run archive manifest contained 1,267 proposed move rows,
totaling 121.2 GiB and 10,431 checkpoint files. All 1,267 rows were moved
successfully with zero failures.

## Verification

After cleanup:

- `uv run python python/scripts/paper_readiness_check.py --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
  passed.
- `uv run python python/scripts/paper_readiness_check.py --run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
  passed.

The readiness commands refreshed the two tracked `paper_readiness_summary.json`
files with the current checker format.

## Next Cleanup Step

Do a separate config/reference cleanup pass. Unlike runs and checkpoints,
`configs/` is tracked source/provenance and only 331 files, so any reduction
should be done by reference audit and tests rather than a bulk archive move.
