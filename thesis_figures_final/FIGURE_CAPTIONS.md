# Thesis Figure Captions

## fig_main_targeted_robustness
Targeted evaluation of the selected main League GRU model (`policy_000021`) against fixed anchors, corrected B3/B4 public heuristics, and legacy league opponents. Evidence level: confirm64 rows. Overall targeted table: 865/1280 (67.6%). B1 no-league: 100/128 (78.1%). B3/B4 combined: 165/256 (64.5%). Legacy neural subset: 344/640 (53.8%).

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
Headline-table seat sensitivity diagnostic for `policy_000021`. Positive values mean the policy won more often from the second seat than the first seat for that opponent. The evaluations are paired and seat-swapped, so the seat split is a diagnostic rather than a confound.

## fig_close_legacy_stress
Stress check for the two closest legacy neural rows. Both p15 and p16 remain slightly above parity at confirm128 (`129/256` each), but their bootstrap intervals overlap 50%, so they should be described as narrow positive margins rather than decisive wins.

## fig_result_decomposition
Aggregate view of the confirm64 table by opponent group. This is useful for a high-level results paragraph: fixed anchors `256/256`, B1 `100/128`, B3/B4 `165/256`, and legacy neural snapshots `344/640`.

## fig_legacy_margin_ladder
Per-legacy-opponent margin above 50% parity with confidence intervals. This is the clearest caveat figure: every legacy row is positive, but p15/p16 are close.

## fig_candidate_p21_vs_p33
Candidate selection diagnostic comparing selected p21 against later p33 on common targeted opponents. p21 uses confirm64 and p33 uses confirm32, so it should support model selection discussion rather than serve as a formal equal-seed comparison.

## fig_anchor_ablation_endpoints
Endpoint fixed-anchor retention for the main model and B1-pressure ablations. Useful for the ablation discussion because it summarizes how B1 pressure affected anchor retention.
