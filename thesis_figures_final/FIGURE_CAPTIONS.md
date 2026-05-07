# Thesis Figure Captions

## fig_main_targeted_robustness
Targeted evaluation of the selected main league GRU model (`policy_000021`) against fixed anchors and legacy league opponents. B0/B2/B3/B4 use completed confirm64 rows; B1 and legacy league rows use completed confirm32 rows. The B1 plus legacy block scores 227/384 (59.1%), and B3/B4 score 165/256 (64.5%).

## fig_anchor_retention
Periodic development evaluation on fixed anchors for the main model and B1-pressure ablations. Solid lines show aggregate anchor score; dashed lines show B1 no-league retention. Use this to argue that the selected model retains anchor strength while entering league training.

## fig_fast_matrix_sanity
Fast final matrix with 4 paired seeds per matchup. This is useful as a qualitative sanity check and figure for the artifact set, but it is not strong enough as the headline quantitative claim.

## fig_baseline_sanity_fixed
Fixed-opponent baseline sanity check against B0 and B2 only. B1 is intentionally excluded here because the No-GRU and PPO-lite matrices were produced against fixed non-neural opponents only; B1 evidence for the main model belongs in the targeted robustness figure.

## fig_baseline_fixed_grid
Table-style version of the fixed-opponent baseline sanity check against B0 and B2 only. This is clearer than bars because the No-GRU and PPO-lite B2 results are true 0/32 outcomes.

## fig_training_loss_diagnostic
Smoothed training loss diagnostic for the main run. Keep this out of the main results argument unless needed for transparency; actor-critic loss is noisy and evaluation win rates are more meaningful.

## fig_b3b4_fixed_validation
Validation figure for the corrected B3/B4 evaluation rows. It shows that all B3/B4 confirm64 games completed without truncation and that `policy_000021` wins 82/128 against B3 and 83/128 against B4.

## fig_b3b4_seat_balance
Seat-swapped B3/B4 robustness diagnostic. `policy_000021` remains above 50% both when moving first and when moving second, which supports using the paired-seat evaluation rather than a single-seat result.

## fig_p21_seat_advantage
Headline-table seat sensitivity diagnostic for `policy_000021`. Positive values mean the policy won more often from the second seat than the first seat for that opponent. Across the headline table, p21 scored 313/448 from first seat and 335/448 from second seat, indicating a modest second-seat advantage in these artifacts.

## fig_p21_seat_advantage
Headline-table seat sensitivity diagnostic for `policy_000021`. Positive values mean the policy won more often from the second seat than the first seat for that opponent. The aggregate row is first seat 313/448 and second seat 335/448, so there is a modest second-seat advantage in these artifacts.
