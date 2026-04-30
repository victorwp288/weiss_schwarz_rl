# Model Cards / Experiment Summary

## B1 GRU anchor

- Family: Anchor
- Checkpoint: B1 v5 u120
- Mean five-anchor win rate: 0.8766
- B1/B3/B4 mean: 0.7943
- B1: 0.4922
- B3: 0.8984
- B4: 0.9922
- Interpretation: Locked recurrent no-league anchor; strongest recurrent baseline.

## No recurrence B1 recipe

- Family: Architecture
- Checkpoint: u160
- Mean five-anchor win rate: 0.8836
- B1/B3/B4 mean: 0.8060
- B1: 0.4961
- B3: 0.9219
- B4: 1.0000
- Interpretation: Feed-forward one-change ablation; strongest five-anchor mean.

## v14 B1-init league

- Family: League
- Checkpoint: u160
- Mean five-anchor win rate: 0.8625
- B1/B3/B4 mean: 0.7708
- B1: 0.5039
- B3: 0.8164
- B4: 0.9922
- Interpretation: Best pre-residual league checkpoint; B1-safe but weaker vs B3.

## v17e residual league

- Family: League
- Checkpoint: u80
- Mean five-anchor win rate: 0.8688
- B1/B3/B4 mean: 0.7812
- B1: 0.5000
- B3: 0.8516
- B4: 0.9922
- Interpretation: Best constrained residual/league candidate; stabilizes B1, does not surpass anchors.

## State reward knobs

- Family: Reward
- Checkpoint: u160
- Mean five-anchor win rate: 0.8695
- B1/B3/B4 mean: 0.7826
- B1: 0.5039
- B3: 0.8516
- B4: 0.9922
- Interpretation: Higher damage/level/board/no-progress/pass penalty; did not clearly help.

## No reward shaping

- Family: Reward
- Checkpoint: u160
- Mean five-anchor win rate: 0.8711
- B1/B3/B4 mean: 0.7852
- B1: 0.5078
- B3: 0.8555
- B4: 0.9922
- Interpretation: Terminal-only reward; similar to shaped runs, slightly better than heavy shaping.

## No behavior BC

- Family: Auxiliary
- Checkpoint: u160
- Mean five-anchor win rate: 0.8695
- B1/B3/B4 mean: 0.7826
- B1: 0.5117
- B3: 0.8477
- B4: 0.9883
- Interpretation: Removes behavior cloning; B1 improves slightly, B3/B4 trade off.

## PPO-lite B1 recipe

- Family: Algorithm
- Checkpoint: u160
- Mean five-anchor win rate: 0.7336
- B1/B3/B4 mean: 0.5560
- B1: 0.2266
- B3: 0.5039
- B4: 0.9375
- Interpretation: Algorithm baseline; substantially worse and unstable, supports IMPALA/V-trace choice.
