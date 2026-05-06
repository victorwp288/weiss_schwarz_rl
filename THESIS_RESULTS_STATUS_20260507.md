# Thesis Results Status - 2026-05-07

## Executive Summary

The `0/32` and `0/128` B3/B4 results were not real model losses. They were invalid evaluations caused by B3/B4 heuristic profiles repeatedly selecting `main_move` until the simulator hit the 2000-action timeout. The evidence was decisive: the bad rows had `wins=0`, `losses=0`, `truncations=games`, `timeout_unknown=games`, and nearly all actions were `main_move`.

An emergency evaluator/heuristic fix was applied in the legacy runtime:

- Remote file: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/python/weiss_rl/eval/heuristic_public.py`
- Backup: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/python/weiss_rl/eval/heuristic_public.py.before_b3b4_move_loop_fix_20260507`
- Behavior changed: aggressive/control heuristic profiles may still prefer beneficial repositioning, but neutral or bad `main_move` actions no longer outrank pass.

After the fix, the main thesis candidate `policy_000021` is no longer failing B3/B4. It beats both on confirm64:

| Opponent | Games | Wins | Win Rate | Truncations | Timeout Unknown |
|---|---:|---:|---:|---:|---:|
| B3 HeuristicPublicAggro | 128 | 82 | 64.06% | 0 | 0 |
| B4 HeuristicPublicControl | 128 | 83 | 64.84% | 0 | 0 |
| B3+B4 combined | 256 | 165 | 64.45% | 0 | 0 |

This makes the B3/B4 story defensible again, provided the thesis explains that the old zero rows were invalid truncation artifacts and are not reported as losses.

## Main Model Locked For Now

Main candidate:

- Run: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`
- Policy: `policy_000021`
- Snapshot path: `training/snapshots/policy_000021/weights.pt`
- Runtime: legacy exp034 runtime, `weiss-sim==0.7.0`
- Training command lineage: continued exp034-style league run to `--max-updates 800`
- Hardware: Vast Linux, 128 CPU cores, CUDA GPU visible as `cuda:0`
- Python: `3.11.15`
- Torch: `2.7.0+cu128`

Strengthened targeted confirmation after B3/B4 fix:

| Opponent | Result |
|---|---:|
| B0 RandomLegal | 128/128 = 100.00% |
| B1 NoLeague baseline | 50/64 = 78.12% |
| B2 HeuristicPublic | 128/128 = 100.00% |
| B3 HeuristicPublicAggro | 82/128 = 64.06% |
| B4 HeuristicPublicControl | 83/128 = 64.84% |
| Legacy p11 | 40/64 = 62.50% |
| Legacy p12 | 34/64 = 53.12% |
| Legacy p14 | 36/64 = 56.25% |
| Legacy p15 | 34/64 = 53.12% |
| Legacy p16 | 33/64 = 51.56% |

B1 plus legacy confirm32 block: `227/384 = 59.11%`. Legacy-only confirm32 subset: `177/320 = 55.31%`.

Fixed B3/B4 confirm64 artifact:

- Summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/p21_b3b4_loopfix_confirm64/targeted_confirm64_summary.json`
- B3 row summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/final_eval/matchups/00_policy_000021__vs__01_b3_heuristicpublicaggro/matchup_summary.json`
- B4 row summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/final_eval/matchups/00_policy_000021__vs__02_b4_heuristicpubliccontrol/matchup_summary.json`
- B0 confirm64 row summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/final_eval/matchups/00_policy_000021__vs__01_b0_randomlegal/matchup_summary.json`
- B2 confirm64 row summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/final_eval/matchups/00_policy_000021__vs__03_b2_heuristicpublic/matchup_summary.json`
- B1/legacy confirm32 summary: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/p21_b1_legacy_confirm32_loopfix/targeted_confirm32_summary.json`

## B3/B4 Fix Evidence

Before the fix, B3/B4 sanity rows were invalid:

- B3 vs B0: all games truncated.
- B4 vs B0: all games truncated.
- p21 vs B3/B4: all games truncated.
- Rows showed `decision_count=0`, `tick_count=0`, `total_actions=2000`, and `main_move_actions` around 1900-1990.

After the fix:

- B3 smoke with 4 paired seeds:
  - B3 vs B0: `8/8`
  - B3 vs B2: `8/8`
  - B3 vs B4: `4/8`
- p21 quick probe with 16 paired seeds:
  - p21 vs B3: `19/32 = 59.38%`
  - p21 vs B4: `19/32 = 59.38%`
- p21 confirm64:
  - p21 vs B3: `82/128 = 64.06%`
  - p21 vs B4: `83/128 = 64.84%`

## Baselines And Ablations

Available baseline/ablation runs:

- No B1 lane ablation: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/ablation_exp028_no_b1_lane_override_from_exp023_to420_20260506`
- Weak B1 guardrail ablation: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/ablation_exp029_weak_b1_auto_from_exp023_to420_20260506`
- No-GRU baseline: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/baseline_nogru_impala_fixed_heuristic_u220_20260506`
- PPO-lite baseline: `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/baseline_ppo_lite_fixed_heuristic_u220_20260506`

Important caveat: No-GRU and PPO-lite are weaker fixed-opponent baselines. Their current eval matrices do not include the full B1/league opponent set, so they should not be presented as directly equivalent league robustness evaluations.

## Known Risks

- The remote legacy repo is dirty by design. Important modified files:
  - `python/weiss_rl/eval/heuristic_public.py`: B3/B4 move-loop fix.
  - `python/scripts/train.py`: emergency resume-config-mismatch bypass added earlier for rescue attempts.
- The B3/B4 fix is narrow and evidence-backed, but it changes evaluation behavior. The thesis should frame this as correcting an invalid heuristic loop, not as improving the learned model.
- A full 10-opponent confirm64 remains expensive. The final package now uses confirm64 for fixed anchors/B3/B4 and confirm32 for B1 plus legacy neural opponents.
- Old B3/B4 rows from before the fix must not be used.

## Recommended Thesis Use

Use `policy_000021` as the main league GRU model for now.

Report:

- B0/B2/B3/B4 from confirm64 rows.
- B1 plus legacy champion/recent robustness from the confirm32 table.
- Explicitly state that B3/B4 evaluation required a heuristic-loop correction because the previous rows were all natural truncations, not losses.

## Figures Ready Locally

Generated figure directory:

- `/Users/vwp/Documents/Codex/2026-05-06/hey-buddy-i-really-need-your/thesis_figures_final`

Recommended figures:

- `fig_main_targeted_robustness`: primary result figure with B0/B1/B2, fixed B3/B4 confirm64, and legacy league rows.
- `fig_b3b4_fixed_validation`: shows B3/B4 confirm64 rows complete cleanly and are no longer timeout artifacts.
- `fig_b3b4_seat_balance`: shows p21 remains above 50% against B3/B4 from both first and second seat.
- `fig_anchor_retention`: development anchor retention for the main run and ablations.
- `fig_baseline_fixed_grid`: fixed-opponent comparison for main, No-GRU, and PPO-lite.

Avoid claiming:

- That the model has a clean full confirm64 table across all 10 opponents. B0/B2/B3/B4 have confirm64 rows; B1 and legacy rows have confirm32 rows.
- That No-GRU/PPO are full league robustness baselines.
- That the B3/B4 fix changes training quality; it fixes opponent behavior/evaluation validity.

## Next Practical Steps

1. Regenerate result figures using fixed B3/B4 rows.
2. Keep B3/B4 in the plots, but mark the B3/B4 data as `confirm64 after heuristic loop fix`.
3. Update Section 7 text to avoid old stale B3/B4 claims and avoid reporting invalid timeout rows.
4. If time allows, rerun full p21 confirm64 overnight. The current package is already stronger than the original 32-game table because every row is now at least confirm32 and four fixed/heuristic rows are confirm64.
5. Commit the heuristic fix and report before any more risky experiments.
