# Section 7 Results Draft Notes - 2026-05-07

## Main Result Framing

The primary thesis model is the league-trained recurrent policy `policy_000021` from:

`runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`

This model should be described as the main League GRU model, trained in the legacy exp034 simulator/runtime for comparability with the strongest previous local result. It is not a clean current-simulator model. That limitation is worth stating plainly.

## Core Claim

The main model substantially solves the fixed public anchors and retains positive robustness against previously promoted league opponents.

Recommended wording:

> The selected League GRU policy maintains perfect performance against the random and public heuristic anchors in higher-seed confirmation (`128/128` against both B0 RandomLegal and B2 HeuristicPublic). It also beats the no-league B1 baseline in confirm64 (`100/128`, 78.1%). Against older promoted league snapshots, performance is more modest but remains above parity in each confirm64 row, with individual win rates between 50.8% and 59.4%.

For the two closest promoted snapshots, add the following caveat if space allows:

> Because p15 and p16 were close to parity in the uniform confirm64 table, they were rerun with 128 paired seeds. Both remained slightly above parity (`129/256`, 50.4%), but the confidence intervals overlap 50%, so these rows should be interpreted as narrow robustness evidence rather than decisive dominance.

## B3/B4 Correction

The B3/B4 results need an explicit methodological caveat.

Recommended wording:

> An initial B3/B4 evaluation produced apparent 0% win rates, but inspection showed that these rows were not losses: all games truncated at the action limit with no wins, no losses, and nearly all actions classified as `main_move`. This exposed a deterministic loop in the aggressive/control public heuristic profiles, where neutral or bad movement could outrank pass indefinitely. After correcting the heuristic so only beneficial repositioning can outrank pass, B3/B4 evaluations completed normally with zero truncations.

Then report:

- B3 HeuristicPublicAggro: `82/128 = 64.06%`
- B4 HeuristicPublicControl: `83/128 = 64.84%`
- B3/B4 combined: `165/256 = 64.45%`
- Truncations: `0/256`

Do not describe the old B3/B4 rows as model failures. They were invalid evaluation artifacts.

## Baselines And Ablations

Recommended conservative framing:

> The No-GRU and PPO-lite baselines are included as fixed-opponent sanity baselines rather than full league-robustness comparisons. Their current matrices do not include the B1/league-opponent rows used for the main model, so they should not be interpreted as equivalent full final evaluations.

For the B1-pressure ablations:

> The no-B1-lane and weak-B1 ablations are most useful for supporting the importance of retaining no-league pressure during league training. They should be discussed through their anchor-retention curves and final anchor behavior rather than overinterpreted as fully independent model families.

## Recommended Figures

Use these figures from `thesis_figures_final`:

- `fig_main_targeted_robustness.pdf`: primary quantitative result.
- `fig_b3b4_fixed_validation.pdf`: documents B3/B4 completion and win rates after the heuristic-loop fix.
- `fig_b3b4_seat_balance.pdf`: shows B3/B4 wins are not from only one seat.
- `fig_close_legacy_stress.pdf`: documents the p15/p16 confirm128 stress check.
- `fig_anchor_retention.pdf`: supports B1-pressure/ablation discussion.
- `fig_baseline_fixed_grid.pdf`: compact fixed-opponent baseline comparison.

## Claims To Avoid

- Do not overstate the close legacy rows. p15 and p16 are above parity, but only by `65/128`.
- Do not claim No-GRU/PPO were evaluated against the same full opponent set.
- Do not imply the B3/B4 fix improved the learned model. It repaired invalid opponent behavior in evaluation.
- Do not claim current-simulator generality; this main result is legacy-simulator comparable.

## Candidate Selection Note

Later snapshots up to `policy_000033` were checked after the p21 package was assembled. `policy_000033` looked promising in a very small sweep, but confirm32 showed it was not a better thesis model overall: it scored `293/512 = 57.23%` on B1, B3/B4, and legacy neural opponents, with weaker B1 and B3/B4 performance than p21. It improved p12 and p16 but regressed p11, p14, and p15. Therefore p21 remains the recommended main model because it gives the stronger broad robustness profile.

## Current Best Numerical Summary

| Opponent | Wins | Games | Win rate | Status |
|---|---:|---:|---:|---|
| B0 RandomLegal | 128 | 128 | 100.00% | confirm64 |
| B1 NoLeague baseline | 100 | 128 | 78.12% | confirm64 |
| B2 HeuristicPublic | 128 | 128 | 100.00% | confirm64 |
| B3 HeuristicPublicAggro | 82 | 128 | 64.06% | fixed confirm64 |
| B4 HeuristicPublicControl | 83 | 128 | 64.84% | fixed confirm64 |
| Legacy p11 | 76 | 128 | 59.38% | confirm64 |
| Legacy p12 | 69 | 128 | 53.91% | confirm64 |
| Legacy p14 | 69 | 128 | 53.91% | confirm64 |
| Legacy p15 | 65 | 128 | 50.78% | confirm64 |
| Legacy p16 | 65 | 128 | 50.78% | confirm64 |
