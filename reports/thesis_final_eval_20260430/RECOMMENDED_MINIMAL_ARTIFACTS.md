# Recommended Minimal Artifact Set

Use the full artifact folder as backup only. For the thesis/slides, the strongest concise set is:

## Core figures
1. `figures/13_compact_final_story.png`
   - Best single overview: no-recurrence/GRU/v17e/PPO in one place.
2. `figures/17_key_anchor_winrate_matrix.png`
   - Clean win-rate matrix for B1/B3/B4 without saturated B0/B2 columns.
3. `figures/15_delta_heatmap_vs_gru_anchor.png`
   - Best zoom for small differences relative to the locked GRU anchor.

## Optional appendix figures
4. `figures/06_impala_vs_ppo_key_anchors.png`
   - Use if the text needs an explicit algorithm-baseline visual.
5. `figures/09_confirm256_top_two_key_anchors.png`
   - Use if the text needs the top-two robustness check.
6. `figures/20_competitive_cluster_tradeoff.png`
   - Use if discussing B1/B3 tradeoffs in detail.

## Core tables/data
- `selected_policy_confirm128_matrix.md`
- `selected_policy_confirm128_matrix.csv`
- `data/all_confirm128_rows.json`
- `data/eval_score_manifest.csv` as backup/raw index, not a thesis table.

## Do not lead with
- Full contact sheet.
- Run catalog of exploratory runs.
- All 20 figures.
- Exploratory early v1-v13/v9/v10 runs as formal ablations unless individually reviewed.

