# Main Search Pre-Cleanup Lock - 2026-05-21

This is the "future me, do not lose the thread" note before naming/config
cleanup. It locks the current strongest model artifact, the exact search
contract, the evidence surface, and the main lessons from the failed and
successful branches.

## Locked Artifacts

### Trained No-Search Main Model

- run:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected source policy id: `main_interp_repair_a015`
- published policy id: `main_league_selected`
- update: `5`
- weights hash:
  `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- registry:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json`
- imported champion/hard-negative registry:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry_with_imported_champions.json`
- selection summary:
  `diagnostics/main_champion_hardneg_interp_u10_repair_a015_selected.json`

This remains the trained thesis model. Do not replace it from confirm64,
confirm128, scalar loss, or aggregate-only evidence.

### Search-Enhanced Main Model

The strongest current player is `main_league_selected` wrapped in K4
same-world prefix-rollout decision-time search:

```text
--god-search-mode same_world_prefix_rollout
--god-search-top-k 4
--god-search-rollouts-per-action 1
--god-search-max-rollout-decisions 0
--god-search-max-search-decisions-per-game 1
--god-search-rollout-policy argmax
--god-search-trace-limit 24
```

Use the human-facing name **K4 same-world search** or
**search-enhanced main model**. Do not call it the raw trained policy. It is a
decision-time wrapper that evaluates candidate actions by replaying the sampled
hidden world.

### B1 Seed

- run:
  `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- source policy id: `policy_000003`
- update: `15`
- weights hash:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`
- report:
  `docs/b1_learning_rebuild_report_20260517.md`

### Fixed-Deck Thesis Surface

- focal/main/B0/B1/B2:
  `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro:
  `preset:aggro_deck_5hy_nino_v1`
- B4 control:
  `preset:control_deck_jj_s66_v1`

Do not mix this with multideck results unless the figure/table labels say so.

## Locked Evidence

### No-Search Main Model

- fixed B0-B4+B1 confirm256:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_b0_b4/targeted_confirm256_summary.json`
- fixed aggregate:
  `1980/2560 = 0.773438`
- imported champion/hard-negative confirm256:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`
- learned aggregate:
  `2317/4096 = 0.565674`

This is paper-ready as the trained no-search main league policy.

### K4 Search-Enhanced Main Model

- confirm128:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm128_k4_r1_terminal_s1_c3_20260521_rerun/targeted_confirm128_summary.json`
- confirm256:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm256_k4_r1_terminal_s1_c3_20260521/targeted_confirm256_summary.json`
- paired confirm256 versus selected no-search:
  `diagnostics/main_search_confirm256_vs_selected_a015_shared256_20260521.json`
- strict scorecard:
  `diagnostics/main_search_scorecard_confirm256_strict_20260521.json`
- search readiness:
  `diagnostics/main_search_readiness_summary_20260521.json`

Confirm256 result:

- overall: `5889/6656 = 0.884766`
- fixed B0-B4+B1: `2420/2560 = 0.945313`
- learned/champion/hard-negative: `3469/4096 = 0.846924`
- paired delta versus no-search selected:
  - all rows: `+1592`
  - fixed rows: `+440`
  - learned rows: `+1152`
- strict scorecard decision:
  `publishable_god_search_candidate`
- prefix replay failures: `0`
- horizon cutoffs: `0`
- truncated rollouts: `0`

## Locked Config And Command Surface

Canonical evaluation config:

```text
configs/thesis/final_eval.yaml
```

Canonical report seed file from the confirm256 summary:

```text
configs/seeds/report_eval_seeds.txt
sha256: 5db0677f8ff932a95bb86cb91e6902de3ed3460ad8303858e169e0a17c728df0
```

Canonical K4 confirm command shape:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/targeted_confirm_eval.py `
  --stack-config configs/thesis/final_eval.yaml `
  --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 `
  --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry_with_imported_champions.json `
  --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 `
  --focal-policy-id main_league_selected `
  --paired-seeds 256 `
  --workers 1 `
  --bootstrap-samples 2000 `
  --output-subdir god_search_confirm256_k4_r1_terminal_s1_c3_20260521 `
  --god-search-mode same_world_prefix_rollout `
  --god-search-top-k 4 `
  --god-search-rollouts-per-action 1 `
  --god-search-max-rollout-decisions 0 `
  --god-search-max-search-decisions-per-game 1 `
  --god-search-rollout-policy argmax `
  --god-search-trace-limit 24
```

The actual run included the 13-opponent fixed plus learned panel:
B0, B1, B2, B3, B4, and the eight imported learned/champion/hard-negative
opponents.

Runtime note: the current unbatched CUDA path was slower than CPU for this
same-world K4 search, so confirm128 and confirm256 were intentionally run on
CPU. See `docs/god_search_runtime_profile_20260521.md`.

## Figure And Thesis Artifacts

Primary thesis figure pack:

```text
C:/Users/Bruger/Desktop/this one/Kandidatspeciale/Figures/main_search_20260521
```

Important files:

- `paper/main_search_strength_ladder.png`
- `paper/main_search_confirm256_group_rates.png`
- `paper/main_search_confirm256_row_win_rates.png`
- `paper/main_search_confirm256_delta_wins.png`
- `paper/main_search_validation_progression.png`
- `paper/main_search_decision_changes.png`
- `paper/main_search_seat_balance.png`
- `paper/main_search_first_second_balance.png`
- `main_search_figure_snippets.tex`

Data sidecars are in:

```text
C:/Users/Bruger/Desktop/this one/Kandidatspeciale/Figures/main_search_20260521/data
```

Restyled contact sheet:

```text
diagnostics/thesis_figure_audit_20260521/main_search_restyled_contact_sheet.png
```

## Seat And Turn-Order Semantics

Do not interpret `seat0` and `seat1` as "going first" and "going second".
They are evaluation assignment slots.

The simulator sets the starting player from seed parity:

```text
starting_player = episode_seed & 1
even seed -> seat0 starts
odd seed  -> seat1 starts
```

Because the confirm eval uses paired seat swaps, each seed gives the focal
policy one game as first player and one game as second player. The correct
turn-order diagnostic is:

```text
diagnostics/main_search_first_second_balance_20260521.json
```

K4 confirm256 turn-order result:

- first player: `2896/3328 = 0.870192`
- second player: `2993/3328 = 0.899339`
- first minus second: `-2.915 percentage points`

Interpretation: on this specific K4-vs-panel surface, the search-enhanced focal
model did better when going second. This is not a universal Weiss Schwarz claim;
it is an eval result for this model, deck policy, opponent panel, and simulator
rules. The game implementation also limits the starting player to one attack on
the first turn, which makes "second can be better" plausible.

## Main Lessons

1. B1 was a good seed, but not enough by itself.

   The B1 locked artifact made the main league phase reproducible and gave a
   stable baseline, but direct continuation did not automatically produce a
   stronger publishable policy.

2. The trained league policy hit a row-level plateau.

   Many branches improved learned/champion/hard-negative aggregate results but
   lost one or two games on B2, B4, policy0004, or another row. That matters
   because the thesis selection contract is row-aware, not scalar-loss-aware.

3. Replay and repair objectives were useful diagnostics but not the final win.

   Grouped replay, hard-negative loss-state repair, paired-swing repair,
   conservative preservation, opponent-context adapters, and interpolation all
   taught us where the pressure was, but none produced a strict confirm256
   successor to the selected no-search model.

4. Search was the decisive bolt-on.

   One-decision K4 same-world prefix-rollout search converted many paired games
   without regressing any row at confirm256. It is the strongest current player
   and the cleanest "impressive model" result.

5. The search result must be labeled honestly.

   It is thesis-defensible as a decision-time search ablation or
   search-enhanced model. It is not defensible as the same thing as a blind
   public-information learned policy, because it replays the sampled hidden
   world.

6. Bigger algorithms are not automatically better right now.

   PPO or another learner would add scope and risk. The evidence says the
   current final result should be protected first: locked model, locked search
   wrapper, clean names, figures, and reproducible commands.

## Cleanup Guardrails

- Preserve chronological `latest` semantics.
- Preserve selected/best as best-confirmed checkpoints.
- Preserve the exact `main_league_selected` registry entry and hash unless a
  new model passes confirm256 no-regression.
- Preserve the K4 search CLI contract exactly if names are cleaned.
- Keep no-search and search-enhanced claims separate.
- Keep fixed-deck and multideck claims separate.
- If files are renamed, update this document, `docs/god_search_k4_lock_20260521.md`,
  `docs/main_god_search_rebuild_report_20260521.md`, and the thesis figure
  snippets together.
