# Thesis Final Eval Artifact Pack

This folder is self-contained for plotting from the consolidated data files.

Important files:
- `data/all_confirm128_rows.csv/json`: official comparison rows.
- `data/eval_score_manifest.csv`: raw eval score index for all discovered thesis eval summaries.
- `data/run_catalog.csv/json`: important and exploratory run catalog.
- `tables/model_cards.md`: model-card style summary.
- `tables/experiment_taxonomy.md`: official vs diagnostic vs exploratory run labels.
- `scripts/recreate_core_figures.py`: local no-GPU script to recreate two core figures from `data/all_confirm128_rows.json`.
- `slides/suggested_slide_map.md`: suggested mapping from thesis claims to figures.

To recreate core figures locally:

```bash
cd thesis_final_eval_20260430
python3 scripts/recreate_core_figures.py
```
