# God-Search Track Plan

Date: 2026-05-21

This is a separate experimental strong-player track. It does not replace the
locked thesis main model:

- locked run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- locked alias: `main_league_selected`
- locked thesis evidence remains the fixed-deck confirm256 and final eval
  artifacts documented in `docs/main_league_model_lock_20260521.md`

## Goal

Build a stronger decision-time player by keeping the locked model weights fixed
and adding search around its legal-action logits.

The search player is evaluated as a named `god_search` artifact, not as a new
blind RL checkpoint unless later evidence supports that claim.

## Search Mode

Implemented first:

- mode: `same_world_prefix_rollout`
- base policy: `main_league_selected`
- action candidates: top-K legal actions by model logit
- branch method: replay the current episode from seed plus prior action prefix,
  force the candidate root action, then roll out to terminal/truncation
- continuation policy: configurable; initial probes use argmax
- default thesis-safe label: exploratory same-world search

Defensibility note: this is reproducible and useful as a strong-player/search
ablation, but it is not the same as a blind public-belief policy. The branch
rollout advances the simulator's sampled hidden world. Any thesis text must
call it same-world decision-time search unless a later public/belief sampling
mode is implemented.

## Loose God-Player Gate

The search track uses an explicit aggregate objective rather than strict
no-regression:

- improve total paired wins versus the selected argmax baseline;
- keep fixed rows from catastrophic collapse;
- keep learned/champion/hard-negative aggregate nonnegative;
- allow small row regressions only when total gains are clear;
- never use this loose gate to replace the locked thesis model.

Initial loose gate defaults are implemented in
`python/scripts/god_search_scorecard.py`:

- `min_all_delta_wins = 1`
- `min_fixed_delta_wins = -2`
- `min_learned_delta_wins = 0`
- `max_fixed_row_drop_wins = 2`
- `max_any_row_drop_wins = 4`

For deeper confirms, raise `min_all_delta_wins` so a candidate must justify its
extra compute cost.

## Fast Loop

1. Mechanistic smoke:
   - verify prefix replay has zero mismatches;
   - verify search evaluates candidates and sometimes changes the chosen action;
   - verify diagnostics are saved under matchup `diagnostics.json`.
2. Sentinel:
   - B2, B4;
   - `seed_c3aac2f9dc_policy_000001`;
   - `seed_c3aac2f9dc_main_league_selected`;
   - `seed_c3aac2f9dc_policy_000003`;
   - `seed_c3aac2f9dc_policy_000004`;
   - `seed_c3aac2f9dc_policy_000005`.
3. Compare against selected argmax on the same seeds with
   `paired_outcome_compare.py`.
4. Apply `god_search_scorecard.py`.
5. Escalate only survivors:
   - sentinel pass -> confirm64;
   - confirm64 pass -> confirm128;
   - confirm128 pass -> confirm256;
   - confirm256 pass -> report as `god_search`, not as a locked thesis model.

## Current Best Probe

Best current setting:

- mode: `same_world_prefix_rollout`
- `top_k = 3`
- `rollouts_per_action = 1`
- `max_rollout_decisions = 0` (terminal)
- `max_search_decisions_per_game = 1`
- `rollout_policy = argmax`

Mechanistic B2 smoke:

- baseline selected argmax, one paired seed:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_baseline_b2_pair1_argmax_20260521/targeted_confirm1_summary.json`
  scored `1/2`.
- search K2 terminal, one paired seed:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_smoke_b2_pair1_k2_r1_terminal_s1_20260521/targeted_confirm1_summary.json`
  scored `2/2`.
- diagnostics showed `changed_decisions = 1`, `prefix_replay_failures = 0`,
  and terminal rollouts for all candidates.

Sentinel4 K3 result:

- baseline:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_baseline_sentinel4_argmax_c3_20260521/targeted_confirm4_summary.json`
  scored `42/56 = 0.750000`.
- search:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_sentinel4_k3_r1_terminal_s1_c3_20260521/targeted_confirm4_summary.json`
  scored `48/56 = 0.857143`.
- paired compare:
  `diagnostics/god_search_sentinel4_k3_r1_terminal_vs_argmax_20260521.json`
  reported `+6` total, `+2` fixed, `+4` learned.
- loose scorecard:
  `diagnostics/god_search_scorecard_sentinel4_k3_r1_terminal_20260521.json`
  recommended `run_confirm64`.

No sentinel row regressed at sentinel4. This is still tiny evidence and should
not be described as a model-quality claim.

Full13 paired-4 screen:

- baseline:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_baseline_full13screen4_argmax_c3_20260521/targeted_confirm4_summary.json`
  scored `78/104 = 0.750000`.
- search:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_full13screen4_k3_r1_terminal_s1_c3_20260521/targeted_confirm4_summary.json`
  scored `91/104 = 0.875000`.
- paired compare:
  `diagnostics/god_search_full13screen4_k3_r1_terminal_vs_argmax_20260521.json`
  reported `+13` total, `+6` fixed, `+7` learned.
- loose scorecard:
  `diagnostics/god_search_scorecard_full13screen4_k3_r1_terminal_20260521.json`
  recommended `run_confirm64`.

Full13 paired-4 row deltas:

| Opponent | Delta wins |
|---|---:|
| B0 RandomLegal | 0 |
| B1 NoLeague baseline | +2 |
| B2 HeuristicPublic | +1 |
| B3 HeuristicPublicAggro | +2 |
| B4 HeuristicPublicControl | +1 |
| seed_c3aac2f9dc_policy_000001 | 0 |
| seed_c3aac2f9dc_policy_000002 | +1 |
| seed_c3aac2f9dc_checkpoint_000025 | +1 |
| seed_c3aac2f9dc_main_bestresponse_u25_devbest | +1 |
| seed_c3aac2f9dc_main_league_selected | +1 |
| seed_c3aac2f9dc_policy_000003 | +1 |
| seed_c3aac2f9dc_policy_000004 | +1 |
| seed_c3aac2f9dc_policy_000005 | +1 |

This is the first broad screen that is actually exciting. It is still only
four paired seeds, so the next real test is full13 confirm64.

## Next Commands

Recommended confirm64 screen:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/targeted_confirm_eval.py `
  --stack-config configs/thesis/final_eval.yaml `
  --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 `
  --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry_with_imported_champions.json `
  --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 `
  --focal-policy-id main_league_selected `
  --opponent "B0 RandomLegal" `
  --opponent "B1 NoLeague baseline" `
  --opponent "B2 HeuristicPublic" `
  --opponent "B3 HeuristicPublicAggro" `
  --opponent "B4 HeuristicPublicControl" `
  --opponent seed_c3aac2f9dc_policy_000001 `
  --opponent seed_c3aac2f9dc_policy_000002 `
  --opponent seed_c3aac2f9dc_checkpoint_000025 `
  --opponent seed_c3aac2f9dc_main_bestresponse_u25_devbest `
  --opponent seed_c3aac2f9dc_main_league_selected `
  --opponent seed_c3aac2f9dc_policy_000003 `
  --opponent seed_c3aac2f9dc_policy_000004 `
  --opponent seed_c3aac2f9dc_policy_000005 `
  --paired-seeds 64 `
  --workers 1 `
  --bootstrap-samples 2000 `
  --output-subdir god_search_confirm64_k3_r1_terminal_s1_c3_20260521 `
  --god-search-mode same_world_prefix_rollout `
  --god-search-top-k 3 `
  --god-search-rollouts-per-action 1 `
  --god-search-max-rollout-decisions 0 `
  --god-search-max-search-decisions-per-game 1 `
  --god-search-rollout-policy argmax `
  --god-search-trace-limit 24
```

Run a same-seed selected-argmax baseline if a matching summary is not already
available for the exact opponent set and seed count, then compare with
`paired_outcome_compare.py` and gate with `god_search_scorecard.py`.
