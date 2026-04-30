# Consolidated Thesis Evaluation Artifacts

Confirm128 rows use 128 paired seeds with seat swap (256 games per anchor).

| rank | candidate | family | mean 5 | B1/B3/B4 mean | B0 | B1 | B2 | B3 | B4 | note |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | No recurrence B1 recipe | Architecture | 0.8836 | 0.8060 | 1.0000 | 0.4961 | 1.0000 | 0.9219 | 1.0000 | Feed-forward one-change ablation; strongest five-anchor mean. |
| 2 | B1 GRU anchor | Anchor | 0.8766 | 0.7943 | 1.0000 | 0.4922 | 1.0000 | 0.8984 | 0.9922 | Locked recurrent no-league anchor; strongest recurrent baseline. |
| 3 | No reward shaping | Reward | 0.8711 | 0.7852 | 1.0000 | 0.5078 | 1.0000 | 0.8555 | 0.9922 | Terminal-only reward; similar to shaped runs, slightly better than heavy shaping. |
| 4 | State reward knobs | Reward | 0.8695 | 0.7826 | 1.0000 | 0.5039 | 1.0000 | 0.8516 | 0.9922 | Higher damage/level/board/no-progress/pass penalty; did not clearly help. |
| 5 | No behavior BC | Auxiliary | 0.8695 | 0.7826 | 1.0000 | 0.5117 | 1.0000 | 0.8477 | 0.9883 | Removes behavior cloning; B1 improves slightly, B3/B4 trade off. |
| 6 | v17e residual league | League | 0.8688 | 0.7812 | 1.0000 | 0.5000 | 1.0000 | 0.8516 | 0.9922 | Best constrained residual/league candidate; stabilizes B1, does not surpass anchors. |
| 7 | v14 B1-init league | League | 0.8625 | 0.7708 | 1.0000 | 0.5039 | 1.0000 | 0.8164 | 0.9922 | Best pre-residual league checkpoint; B1-safe but weaker vs B3. |
| 8 | PPO-lite B1 recipe | Algorithm | 0.7336 | 0.5560 | 1.0000 | 0.2266 | 1.0000 | 0.5039 | 0.9375 | Algorithm baseline; substantially worse and unstable, supports IMPALA/V-trace choice. |

## Takeaways

- Best five-anchor mean: no-recurrence B1 recipe; confirm256 shows it is effectively tied with the locked GRU anchor on B1/B3/B4.
- Best recurrent/GRU no-league policy: B1 GRU anchor.
- Best league-style policy: v17e residual league; stable but below the anchor family.
- PPO-lite is clearly worse than IMPALA/V-trace.
- Reward shaping and no-BC mostly trade B1 against B3/B4 rather than lifting all anchors.
