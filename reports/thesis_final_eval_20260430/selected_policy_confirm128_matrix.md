# Thesis Final Eval Snapshot - 2026-04-30
Confirm128 results use 128 paired seeds, seat-swapped for 256 games per anchor.

## Selected Policies And Baselines
| candidate | mean 5-anchor | B1/B3/B4 mean | B0 | B1 | B2 | B3 | B4 | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 v5 no-league GRU anchor | 0.8766 | 0.7943 | 1.0000 | 0.4922 | 1.0000 | 0.8984 | 0.9922 | Locked recurrent GRU/no-league policy; strongest B3 among recurrent candidates. |
| No-recurrence B1-recipe baseline | 0.8836 | 0.8060 | 1.0000 | 0.4961 | 1.0000 | 0.9219 | 1.0000 | One-change ablation from B1 recipe: recurrent_core=none; strongest mean on this anchor suite. |
| v14 B1-init anchor-stabilized league | 0.8625 | 0.7708 | 1.0000 | 0.5039 | 1.0000 | 0.8164 | 0.9922 | Best pre-residual league-style checkpoint; slightly best B1 score but weaker B3. |
| v17e B1 residual guard | 0.8688 | 0.7812 | 1.0000 | 0.5000 | 1.0000 | 0.8516 | 0.9922 | Best constrained residual/league candidate; stabilizes B1 while retaining reasonable B3. |

## Current Decision
- Treat the no-recurrence B1-recipe run as a strong baseline/ablation, not yet as the only final answer; it is a one-shot result that should be described carefully.
- B1 v5 remains the locked recurrent anchor and strongest recurrent no-league policy.
- v17e u80 remains the best league-style/constrained residual candidate.
- Thesis framing: the strongest policies came from anchor/no-league training; ordinary league continuation drifted, while residual guard stabilized B1 but did not surpass the best no-league baselines.

## Important Negative Results
- v14 continued past u160 drifted on B1.
- v15 frozen trunk + B1 KL did not recover the anchor; u240 B1-only was about 0.4766.
- v16 B1 anchor-only rows did not beat v14; confirm64 u240 had B1 0.4609, B3 0.7266, B4 0.9609.
- v17e later checkpoints preserved B1 for a while but traded off B3: u120 B3 0.8125, u140 B3 0.8047, u160 B1 0.4922.

## Top-Two Confirm256 Check
Confirm256 uses 256 paired seeds, seat-swapped for 512 games per anchor, on the key B1/B3/B4 slice.

| candidate | B1 | B3 | B4 | mean |
| --- | ---: | ---: | ---: | ---: |
| B1 v5 no-league GRU anchor | 0.4980 | 0.8887 | 0.9980 | 0.7949 |
| No-recurrence B1-recipe baseline | 0.4980 | 0.9004 | 0.9902 | 0.7962 |

Interpretation: the no-recurrence one-change ablation and the locked GRU B1 anchor are effectively tied on the key confirm256 slice. No-recurrence is slightly stronger on B3, while the GRU anchor is slightly stronger on B4. This should be framed as a strong baseline/ablation result rather than a reason to reopen main-model training.
