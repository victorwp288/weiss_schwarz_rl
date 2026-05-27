# God-Search K4 Candidate Lock - 2026-05-21

## Status

This file locks the current best search-track player as the strongest current
main-model artifact. The trainable base model remains the locked selected
main-league policy; the final player is that policy plus the K4 same-world
decision-time search wrapper.

- Base model run:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- Base policy id:
  `main_league_selected`
- Base checkpoint/model:
  locked selected-a015 thesis model; unchanged by this search work
- Search candidate name:
  `god_search_k4_r1_terminal_s1_c3_20260521`
- Search mode:
  `same_world_prefix_rollout`
- Thesis label:
  same-world decision-time search ablation around the locked selected model
- Defensibility caveat:
  this mode replays from the sampled simulator seed/prefix and rolls forward in
  that sampled hidden world. It must not be described as a blind public-belief
  player.

## Locked Search Contract

The K4 candidate is defined by this exact decision-time wrapper:

```text
--god-search-mode same_world_prefix_rollout
--god-search-top-k 4
--god-search-rollouts-per-action 1
--god-search-max-rollout-decisions 0
--god-search-max-search-decisions-per-game 1
--god-search-rollout-policy argmax
--god-search-trace-limit 24
```

Semantics:

- Apply search to the focal `main_league_selected` player only.
- At the first focal decision with more than one legal candidate, take the
  model's top 4 legal candidates.
- For each candidate, replay the same game prefix from the original episode
  seed, force that candidate action, then roll out to terminal with argmax
  policies.
- Pick the candidate with the best terminal score from the focal seat.
- Verify prefix replay and fail on mismatch.

## Confirm64 Evidence

Primary confirm64 artifact:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm64_k4_r1_terminal_s1_c3_20260521/targeted_confirm64_summary.json
```

Overall:

- `1473/1664 = 0.885216`
- fixed B0-B4 rows: `599/640 = 0.935938`
- learned/champion/hard-negative rows: `874/1024 = 0.853516`
- elapsed: `3789.34` seconds

Per-row wins:

| Opponent | Wins |
| --- | ---: |
| B0 RandomLegal | 128/128 |
| B1 NoLeague baseline | 112/128 |
| B2 HeuristicPublic | 124/128 |
| B3 HeuristicPublicAggro | 119/128 |
| B4 HeuristicPublicControl | 116/128 |
| seed_c3aac2f9dc_policy_000001 | 114/128 |
| seed_c3aac2f9dc_policy_000002 | 111/128 |
| seed_c3aac2f9dc_checkpoint_000025 | 107/128 |
| seed_c3aac2f9dc_main_bestresponse_u25_devbest | 107/128 |
| seed_c3aac2f9dc_main_league_selected | 110/128 |
| seed_c3aac2f9dc_policy_000003 | 110/128 |
| seed_c3aac2f9dc_policy_000004 | 106/128 |
| seed_c3aac2f9dc_policy_000005 | 109/128 |

## Paired Comparisons

Versus selected-a015 argmax on shared64:

```text
diagnostics/god_search_confirm64_k4_r1_terminal_vs_selected_a015_shared64_20260521.json
```

- all rows: `+376` wins
- fixed rows: `+107` wins
- learned rows: `+269` wins
- no row regressed on the shared64 surface

Versus K3/R1 terminal/search1 confirm64:

```text
diagnostics/god_search_confirm64_k4_vs_k3_r1_terminal_s1_20260521.json
```

- all rows: `+42` wins
- fixed rows: `+13` wins
- learned rows: `+29` wins
- zero K3-to-K4 row regressions

Loose gate:

```text
diagnostics/god_search_scorecard_confirm64_k4_r1_terminal_20260521.json
```

- decision: `run_confirm128`
- reason: `confirm64_loose_gate_passed`

## Mechanistic Diagnostics

Across the 13 confirm64 rows:

- searched decisions: `1664`
- changed decisions: `376`
- terminal rollouts: `6656`
- prefix replay failures: `0`
- horizon cutoffs: `0`
- truncated rollouts: `0`

The search is doing real work, not acting as a no-op. Changed-decision counts
were largest on learned mirror/recent-policy rows:

- `seed_c3aac2f9dc_main_league_selected`: `40`
- `seed_c3aac2f9dc_policy_000003`: `40`
- `seed_c3aac2f9dc_policy_000004`: `36`
- `seed_c3aac2f9dc_policy_000005`: `33`

## Current Validation State

Historical pre-confirm256 state:

- K4 passed sentinel and full13 paired-4 screens.
- K4 passed confirm64 and beat K3 confirm64 cleanly.
- The first K4 confirm128 attempt was intentionally stopped before producing a
  row summary or final summary:
  `god_search_confirm128_k4_r1_terminal_s1_c3_20260521`.
- K4 was then rerun through confirm128 and confirm256; the final locked
  artifacts are listed below.

## Historical Before Confirm256 Notes

These were the checks required before the confirm256 run. They are preserved so
future cleanup can reconstruct why K4 was escalated.

1. Runtime/GPU profiling:
   completed in `docs/god_search_runtime_profile_20260521.md`. The current
   unbatched CUDA path is slower than CPU for K4 same-world search.
2. Optional last cheap strength probes:
   try only tiny sentinel or paired-4 screens for K5/R1 and K4/R2. Do not run
   confirm64 unless the tiny screen beats K4 by a meaningful margin without row
   givebacks.
3. Confirm128:
   finish or rerun K4 confirm128 before any confirm256 claim.
4. Confirm256:
   only run after confirm128 passes the loose aggregate gate and shows no
   catastrophic fixed-anchor or learned-panel collapse.

## Exact Confirm128 Command

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
  --paired-seeds 128 `
  --workers 1 `
  --bootstrap-samples 2000 `
  --output-subdir god_search_confirm128_k4_r1_terminal_s1_c3_20260521 `
  --god-search-mode same_world_prefix_rollout `
  --god-search-top-k 4 `
  --god-search-rollouts-per-action 1 `
  --god-search-max-rollout-decisions 0 `
  --god-search-max-search-decisions-per-game 1 `
  --god-search-rollout-policy argmax `
  --god-search-trace-limit 24
```

## Final Confirm128 and Confirm256 Lock Update

The provisional confirm64 K4 recipe above has now passed confirm128 and
confirm256. Treat this as the locked search-enhanced main model:

```text
base policy: main_league_selected
base run: runs/main_champion_hardneg_interp_u10_repair_a015_20260517
search mode: same_world_prefix_rollout
top_k: 4
rollouts_per_action: 1
max_rollout_decisions: 0
max_search_decisions_per_game: 1
rollout_policy: argmax
```

Important thesis caveat: this is a same-world decision-time search player. It is
the strongest current fixed-deck model artifact, but it should be described as a
search-enhanced/oracle-style decision-time model, not as a blind public-belief
policy.

### Confirm128

Artifact:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm128_k4_r1_terminal_s1_c3_20260521_rerun/targeted_confirm128_summary.json
```

Score:

- overall: `2923/3328 = 0.8783052884615384`
- fixed B0-B4+B1: `1200/1280 = 0.9375`
- learned/champion/hard-negative: `1723/2048 = 0.84130859375`

Paired comparison versus selected-a015 no-search confirm128 prefix:

```text
diagnostics/god_search_confirm128_k4_r1_terminal_vs_selected_a015_shared128_20260521.json
```

- all rows: `+749` wins
- fixed rows: `+208` wins
- learned rows: `+541` wins
- every row was non-regressing

Gate:

```text
diagnostics/god_search_scorecard_confirm128_k4_r1_terminal_20260521.json
```

- decision: `run_confirm256`

### Confirm256

Artifact:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm256_k4_r1_terminal_s1_c3_20260521/targeted_confirm256_summary.json
```

Score:

- overall: `5889/6656 = 0.884765625`
- fixed B0-B4+B1: `2420/2560 = 0.9453125`
- learned/champion/hard-negative: `3469/4096 = 0.846923828125`

Paired comparison versus selected-a015 no-search confirm256:

```text
diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json
```

- all rows: `+1592` wins
- fixed rows: `+440` wins
- learned rows: `+1152` wins
- every row was non-regressing under the strict scorecard

Strict gate:

```text
diagnostics/god_search_scorecard_confirm256_k4_r1_terminal_strict_20260521.json
```

- decision: `publishable_god_search_candidate`

Search-specific readiness summary:

```text
diagnostics/god_search_readiness_summary_20260521.json
```

### Confirm256 Row Scores

| Opponent | selected-a015 no-search | K4 search | Delta |
|---|---:|---:|---:|
| B0 RandomLegal | 512/512 | 512/512 | +0 |
| B1 NoLeague baseline | 322/512 | 455/512 | +133 |
| B2 HeuristicPublic | 399/512 | 498/512 | +99 |
| B3 HeuristicPublicAggro | 365/512 | 480/512 | +115 |
| B4 HeuristicPublicControl | 382/512 | 475/512 | +93 |
| seed_c3aac2f9dc_policy_000001 | 328/512 | 454/512 | +126 |
| seed_c3aac2f9dc_policy_000002 | 299/512 | 447/512 | +148 |
| seed_c3aac2f9dc_checkpoint_000025 | 289/512 | 431/512 | +142 |
| seed_c3aac2f9dc_main_bestresponse_u25_devbest | 289/512 | 431/512 | +142 |
| seed_c3aac2f9dc_main_league_selected | 274/512 | 424/512 | +150 |
| seed_c3aac2f9dc_policy_000003 | 274/512 | 424/512 | +150 |
| seed_c3aac2f9dc_policy_000004 | 291/512 | 433/512 | +142 |
| seed_c3aac2f9dc_policy_000005 | 273/512 | 425/512 | +152 |

### Confirm256 Search Diagnostics

Across the 13 confirm256 rows:

- searched decisions: `6656`
- changed decisions: `1592`
- candidate evaluations: `26624`
- terminal rollouts: `26624`
- prefix replay failures: `0`
- horizon cutoffs: `0`
- truncated rollouts: `0`

### Figures and Tables

Search-specific paper figures and data exports:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_row_win_rates.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_delta_wins.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_group_rates.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_rows.csv
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_group_summary.json
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_row_table.md
```

Figure command:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/make_god_search_figures.py `
  --compare-json diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json `
  --out-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures `
  --figure-prefix god_search_k4_confirm256
```
