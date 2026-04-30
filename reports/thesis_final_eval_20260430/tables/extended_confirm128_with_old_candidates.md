# Extended Confirm128 Matrix With Old Candidate Baselines

These two old runs are included as appendix/tradeoff baselines. They should not replace the official final rows: they are early u40/debug configs and mainly show B1-specialization versus B3 robustness tradeoff.

| rank by mean5 | candidate | family | mean5 | key mean | min key | B1 | B3 | B4 | note |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Old v2 B1-specialist | Old/debug | 0.9195 | 0.8659 | 0.6328 | 0.9688 | 0.6328 | 0.9961 | Old early B1-focused config; very strong vs B1 but weak vs B3 aggro. |
| 2 | Old v3b profile-cycle | Old/debug | 0.9031 | 0.8385 | 0.7227 | 0.7969 | 0.7227 | 0.9961 | Old profile-cycle/no-guard config; B1-heavy but trades off B3 strength. |
| 3 | No recurrence B1 recipe | Architecture | 0.8836 | 0.8060 | 0.4961 | 0.4961 | 0.9219 | 1.0000 | Feed-forward one-change ablation; strongest five-anchor mean. |
| 4 | B1 GRU anchor | Anchor | 0.8766 | 0.7943 | 0.4922 | 0.4922 | 0.8984 | 0.9922 | Locked recurrent no-league anchor; strongest recurrent baseline. |
| 5 | No reward shaping | Reward | 0.8711 | 0.7852 | 0.5078 | 0.5078 | 0.8555 | 0.9922 | Terminal-only reward; similar to shaped runs, slightly better than heavy shaping. |
| 6 | State reward knobs | Reward | 0.8695 | 0.7826 | 0.5039 | 0.5039 | 0.8516 | 0.9922 | Higher damage/level/board/no-progress/pass penalty; did not clearly help. |
| 7 | No behavior BC | Auxiliary | 0.8695 | 0.7826 | 0.5117 | 0.5117 | 0.8477 | 0.9883 | Removes behavior cloning; B1 improves slightly, B3/B4 trade off. |
| 8 | v17e residual league | League | 0.8688 | 0.7812 | 0.5000 | 0.5000 | 0.8516 | 0.9922 | Best constrained residual/league candidate; stabilizes B1, does not surpass anchors. |
| 9 | v14 B1-init league | League | 0.8625 | 0.7708 | 0.5039 | 0.5039 | 0.8164 | 0.9922 | Best pre-residual league checkpoint; B1-safe but weaker vs B3. |
| 10 | PPO-lite B1 recipe | Algorithm | 0.7336 | 0.5560 | 0.2266 | 0.2266 | 0.5039 | 0.9375 | Algorithm baseline; substantially worse and unstable, supports IMPALA/V-trace choice. |

## Recommended interpretation

- Old v2 is a useful B1-specialist baseline: B1 is extremely high, but B3 is much weaker.
- Old v3b is a softer version of the same tradeoff.
- These are useful appendix evidence that optimizing the anchor too directly can sacrifice robustness to aggressive heuristic opponents.
- Use `min key` and B1/B3 tradeoff plots when discussing them; aggregate alone overstates them because B0/B2/B4 are saturated.
