# Final Thesis Result Trace

This directory contains the compact, thesis-facing figure exports for the May 21
final result surface. `RESULTS_TRACE.json` records the exact table values,
search contract, group totals, diagnostics, and source artifact paths used by the
thesis results chapter.

The full raw evidence remains intentionally outside `main` because it includes
large run directories, raw episode logs, checkpoints, and posterior samples. The
source artifact branch is `origin/final/progress`; the key lock notes are:

- `docs/main_league_model_lock_20260521.md`
- `docs/god_search_k4_lock_20260521.md`
- `diagnostics/main_search_readiness_summary_20260521.json`
- `diagnostics/god_search_confirm256_k4_r1_terminal_vs_selected_a015_shared256_20260521.json`
- `diagnostics/main_search_first_second_balance_20260521.json`

The compact trace is the audit bridge for readers of the thesis repo: it makes
the reported numbers auditable from source history without committing bulky raw
training artifacts.
