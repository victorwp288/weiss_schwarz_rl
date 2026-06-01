# Main God-Search Rebuild Report - 2026-05-21

## Executive Summary

The strongest current main model is the locked `main_league_selected` policy
from:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517
```

wrapped with K4/R1 same-world decision-time search:

```text
mode = same_world_prefix_rollout
top_k = 4
rollouts_per_action = 1
max_rollout_decisions = 0
max_search_decisions_per_game = 1
rollout_policy = argmax
```

This is now confirm256 validated on the fixed-deck thesis surface plus imported
champion/hard-negative rows. It is the strongest current fixed-deck artifact,
but it must be described carefully: it is a search-enhanced same-world
decision-time player, not a blind public-belief policy.

## Why We Pivoted to Search

The trained no-search league policy reached a useful but stubborn plateau. The
best unconditioned training branches repeatedly showed the same pattern:
learned/champion/hard-negative rows could improve, but strict row-level
no-regression against B0/B1/B2/B3/B4 or individual learned rows did not hold
reproducibly at confirm128/confirm256.

We tested and diagnosed several structural directions before locking search:

- conservative fixed-anchor continuations protected B0-B4 but did not move the
  learned rows enough;
- grouped replay and loss-state repair moved learned rows but taxed fixed
  anchors;
- paired-swing and context/adapter ideas exposed opponent-dependent conflict,
  but did not produce a clean publishable no-regression policy in the remaining
  time;
- GPU execution did not speed up K4 search because the current search loop
  performs many tiny sequential model forwards inside simulator rollouts.

Decision-time same-world search was the strongest thesis-defensible final move:
it leaves the locked selected policy unchanged, makes the extra mechanism
explicit, and gives a reproducible strength result rather than another
borderline trained-policy continuation.

## Locked Base Model

Base selected model:

```text
run: runs/main_champion_hardneg_interp_u10_repair_a015_20260517
policy id: main_league_selected
source selected candidate: main_interp_repair_a015
selected update: 5
weights hash: 1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26
```

Locked B1 reference model:

```text
run: runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01
policy id: selected_candidate
source policy id: policy_000003
update: 15
weights hash: 66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685
```

Fixed deck policy:

- focal, B0, B1, B2: `preset:main_deck_5hy_yotsuba_v1`
- B3: `preset:aggro_deck_5hy_nino_v1`
- B4: `preset:control_deck_jj_s66_v1`

## Confirm256 Result

Primary artifact:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm256_k4_r1_terminal_s1_c3_20260521/targeted_confirm256_summary.json
```

Comparison artifact:

```text
diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json
```

Strict scorecard:

```text
diagnostics/god_search_scorecard_confirm256_k4_r1_terminal_strict_20260521.json
```

Search-specific readiness summary:

```text
diagnostics/god_search_readiness_summary_20260521.json
```

Group results:

| Group | selected-a015 no-search | K4 search | Delta |
|---|---:|---:|---:|
| Fixed B0-B4+B1 | 1980/2560 = 0.7734375 | 2420/2560 = 0.9453125 | +440 |
| Learned/champion/hard-negative | 2317/4096 = 0.565673828125 | 3469/4096 = 0.846923828125 | +1152 |
| All 13 rows | 4297/6656 = 0.6455829326923077 | 5889/6656 = 0.884765625 | +1592 |

Row results:

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

The strict confirm256 scorecard decision is:

```text
publishable_god_search_candidate
```

## Earlier Gates and Ablations

K3/R1 terminal/search1 confirm64:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm64_k3_r1_terminal_s1_c3_20260521/targeted_confirm64_summary.json
```

- overall: `1431/1664 = 0.8599759615384616`
- fixed: `586/640 = 0.915625`
- learned: `845/1024 = 0.8251953125`

K4/R1 terminal/search1 confirm64:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm64_k4_r1_terminal_s1_c3_20260521/targeted_confirm64_summary.json
```

- overall: `1473/1664 = 0.8852163461538461`
- fixed: `599/640 = 0.9359375`
- learned: `874/1024 = 0.853515625`
- K4 beat K3 by `+42` paired wins, with no row regression.

K4 confirm128:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/god_search_confirm128_k4_r1_terminal_s1_c3_20260521_rerun/targeted_confirm128_summary.json
```

- overall: `2923/3328 = 0.8783052884615384`
- fixed: `1200/1280 = 0.9375`
- learned: `1723/2048 = 0.84130859375`
- paired versus selected-a015 prefix128: `+749` all, `+208` fixed,
  `+541` learned.

CPU/GPU runtime profile:

```text
docs/god_search_runtime_profile_20260521.md
```

- CPU cProfile paired-4 mini surface: `74.955s`
- GPU cProfile paired-4 mini surface: `206.011s`
- GPU was about `2.75x` slower in the current unbatched search path.

## Search Diagnostics

Confirm256 search counters:

- searched decisions: `6656`
- changed decisions: `1592`
- candidate evaluations: `26624`
- terminal rollouts: `26624`
- prefix replay failures: `0`
- horizon cutoffs: `0`
- truncated rollouts: `0`

This confirms the search mechanism is active and clean: it changed outcomes
through explicit candidate rollout selection, not through silent replay failure
or truncation.

## Figures and Tables

Generated search-specific figure/data artifacts:

```text
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_row_win_rates.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_delta_wins.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/god_search_k4_confirm256_group_rates.png
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_rows.csv
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_group_summary.json
runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/data/god_search_k4_confirm256_row_table.md
```

Exporter command:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/make_god_search_figures.py `
  --compare-json diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json `
  --out-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures `
  --figure-prefix god_search_k4_confirm256
```

## Reproduction Command

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

## Tests

Focused validation run:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python -m pytest -q `
  python/weiss_rl/tests/test_god_search.py `
  python/weiss_rl/tests/test_god_search_scorecard.py `
  python/weiss_rl/tests/test_god_search_figures.py `
  python/weiss_rl/tests/test_targeted_confirm_prefix.py
```

Result:

```text
8 passed
```

## Remaining Defensibility Notes

Use this wording in thesis material:

> The final search-enhanced main model is the locked main league policy with a
> one-decision K4 same-world prefix-rollout search wrapper. It is evaluated as a
> decision-time search model on the fixed-deck thesis surface. Because the
> wrapper replays the sampled hidden world, it is not a blind public-belief
> policy and should be reported separately from the no-search trained policy.

The canonical paper-readiness pipeline was originally built for ordinary
policy-vs-policy final-eval matrices. The K4 search artifact has targeted
confirm256 summaries, row tables, figures, diagnostics, and a search-specific
readiness summary, but a future cleanup could add a canonical final-eval wrapper
mode so `paper_readiness_summary.json` can represent search-enhanced focal
policies directly.
