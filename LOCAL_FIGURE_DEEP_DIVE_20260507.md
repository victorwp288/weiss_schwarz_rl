# Local Figure Deep Dive - 2026-05-07

## What can be analyzed locally

The figure work is CPU-only. No GPU or remote training is needed for the current plots because the relevant evaluation outputs have already been copied into `vast_artifacts`.

Available local evidence:

- Full p21 confirm64 targeted table: `vast_artifacts/main/p21_b1_legacy_confirm64_summary.json`, `vast_artifacts/main/p21_b3b4_loopfix_confirm64_summary.json`, and `vast_artifacts/main/confirm64_rows/`.
- p15/p16 confirm128 stress check: `vast_artifacts/main/p21_p15_p16_confirm128_summary.json`.
- Seat diagnostics: `vast_artifacts/main/seat_diagnostics/`.
- B3/B4 diagnostics: `vast_artifacts/main/diagnostics/`.
- Development anchor curves: `vast_artifacts/main/dev_eval_summaries.json`, `vast_artifacts/exp028/dev_eval_summaries.json`, and `vast_artifacts/exp029/dev_eval_summaries.json`.
- Baseline fixed-opponent matrices: `vast_artifacts/nogru/final_summary.json` and `vast_artifacts/ppo/final_summary.json`.
- Candidate-selection check for p33: `vast_artifacts/main/p33_b1_b3b4_legacy_confirm32_summary.json`.

## New local figures added

- `fig_result_decomposition`: high-level result grouped into fixed anchors, B1, B3/B4, and legacy neural opponents.
- `fig_legacy_margin_ladder`: per-legacy-opponent margin above 50% parity with confidence intervals.
- `fig_candidate_p21_vs_p33`: selection diagnostic showing why p21 remains preferable to later p33 overall. Caveat: p21 is confirm64; p33 is confirm32.
- `fig_anchor_ablation_endpoints`: endpoint fixed-anchor retention for the main run and B1-pressure ablations.

## Best figure set for Section 7

Use these as the main thesis result set:

- `fig_main_targeted_robustness`
- `main_p21_results_table`
- `fig_result_decomposition`
- `fig_legacy_margin_ladder`
- `fig_close_legacy_stress`
- `fig_b3b4_fixed_validation`
- `fig_p21_seat_advantage`

Use these for baselines/ablations:

- `fig_anchor_retention`
- `fig_anchor_ablation_endpoints`
- `fig_baseline_fixed_grid`
- `fig_candidate_p21_vs_p33`

Use cautiously or omit unless space is needed:

- `fig_fast_matrix_sanity`: low-game sanity matrix only.
- `fig_training_loss_diagnostic`: optimization diagnostic only; win-rate evals are more meaningful.

## Main caveat

The local data is deep enough for plots and diagnostics, but not for inventing new training claims. The strongest defensible claims are still the confirm64 targeted table, the B3/B4 validity repair, the p15/p16 confirm128 stress check, and the fixed-opponent baseline caveats.
