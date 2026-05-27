# Private Backup Manifest - 2026-05-27

This manifest records the curated artifact payload added back to
`final/progress` as a private backup branch. It is intentionally broader than
the polished public-facing branch should be.

## Branch

- Backup branch: `final/progress`
- Previous artifact-preservation commit already in this branch history:
  `b40bce61e artifacts: preserve final thesis progress runs`
- That older commit preserves the May 7 final-progress runs and baselines listed
  in `FINAL_PROGRESS_BRANCH_CONTENTS_20260507.md`.

## Added Curated Local Artifacts

These local runs were selected because the current May 21 lock docs reference
them as the thesis-selected model, B1 seed, or interpolation source artifacts:

| Path | Approx size | Why preserved |
| --- | ---: | --- |
| `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/` | 199 MB | Locked no-search main model, final evals, K4 search evals, registry, selected weights, figures, readiness outputs. |
| `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/` | 212 MB | Locked B1 NoLeague seed referenced by the main-model lock docs. |
| `runs/main_champion_hardneg_long_v1_u10_20260517_seg01/` | 118 MB | First interpolation source checkpoint for the locked selected main model. |
| `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01/` | 102 MB | Second interpolation source checkpoint for the locked selected main model. |

GitHub size guard: these checked run directories had no files above 90 MB before
staging.

## Locked Weight Hashes

- `main_league_selected`:
  `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- locked B1 `selected_candidate`:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`

These match the hashes recorded in `docs/main_league_model_lock_20260521.md` and
`docs/main_search_precleanup_lock_20260521.md`.

## Added Diagnostics

The backup includes the diagnostics directly supporting the model/search lock:

- `diagnostics/god_search_*`
- `diagnostics/main_search_*`
- `diagnostics/main_league_frontier_*`
- `diagnostics/main_champion_hardneg_interp_u10_repair_a015_selected.json`
- `diagnostics/thesis_figure_audit_20260521/`

These preserve the confirm128/confirm256 paired comparisons, scorecards,
readiness summaries, frontier audit, and figure-audit contact sheets referenced
by the May 21 docs.

## Added Sibling Figure Bundle

Copied from:

```text
C:\Users\Bruger\Desktop\this one\Kandidatspeciale\Figures\new results
```

Into:

```text
thesis_figures_final/main_search_20260521/
```

Files:

- `main_search_confirm256_delta_wins.png`
- `main_search_confirm256_group_rates.png`
- `main_search_confirm256_row_win_rates.png`
- `main_search_decision_changes.png`
- `main_search_first_second_balance.png`
- `main_search_strength_ladder.png`
- `main_search_validation_progression.png`

## Still Local Only

The full `runs/` tree is about 124.6 GB and is not suitable for a normal GitHub
branch. This backup deliberately preserves the locked/artifact-referenced runs
instead of force-adding every local experiment.

Large scratch/tool/cache outputs remain local-only:

- `.venv/`
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`
- `.VSCodeCounter/`
- `.playwright-mcp/`
- `diagnostics/` files outside the curated lock/search/frontier set
- `temp/`, `now/`, and `now.zip`
