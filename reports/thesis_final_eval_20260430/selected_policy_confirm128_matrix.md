# Thesis Final Eval Snapshot - 2026-04-30
Confirm128 results use 128 paired seeds, seat-swapped for 256 games per anchor.

## Selected Policies
| candidate | mean 5-anchor | B1/B3/B4 mean | B0 | B1 | B2 | B3 | B4 | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 v5 no-league anchor | 0.8766 | 0.7943 | 1.0000 | 0.4922 | 1.0000 | 0.8984 | 0.9922 | Primary final/simple policy. Strongest overall mean; trained without league continuation. |
| v14 B1-init anchor-stabilized league | 0.8625 | 0.7708 | 1.0000 | 0.5039 | 1.0000 | 0.8164 | 0.9922 | Best pre-residual league-style checkpoint; slightly best B1 score but weaker B3. |
| v17e B1 residual guard | 0.8688 | 0.7812 | 1.0000 | 0.5000 | 1.0000 | 0.8516 | 0.9922 | Best constrained residual/league candidate; first approach to stabilize B1 while keeping B3 reasonably high. |

## Current Decision
- Use B1 v5 u120 as the primary final policy: it has the best five-anchor mean and strongest B3 score.
- Use v17e u80 as the main league/constrained residual candidate: it is B1-safe at 0.5000 and beats v14 on B3.
- Use v14 u160 as the pre-residual league comparator: slightly better on B1, weaker on B3.
- Thesis framing: league/self-play did not clearly surpass the B1 anchor under time constraints, but the residual guard solved most of the B1 drift failure and gives a defensible constrained-league ablation.

## Important Negative Results
- v14 continued past u160 drifted on B1.
- v15 frozen trunk + B1 KL did not recover the anchor; u240 B1-only was about 0.4766.
- v16 B1 anchor-only rows did not beat v14; confirm64 u240 had B1 0.4609, B3 0.7266, B4 0.9609.
- v17e later checkpoints preserved B1 for a while but traded off B3: u120 B3 0.8125, u140 B3 0.8047, u160 B1 0.4922.
