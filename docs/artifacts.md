# Artifacts

Existing historical outputs under `runs/`, `run_logs/`, `vast_artifacts/`, and
`thesis_figures_final/` are protected. Do not migrate or delete them during the
rebuild unless explicitly asked.

Canonical new run outputs live under `runs/<run_label>/` and include:

- `manifest.json`
- `environment.json`
- `run_summary.json`
- `determinism_report.json`
- `config_canonical.json`
- `spec_bundle.json`
- `spec_hash256.txt`
- `training/checkpoints/`
- `training/snapshots/registry.json`
- `training/logs/training_metrics.jsonl`
- `training/logs/performance.jsonl`
- `eval/final_eval/`
- `eval/diagnostics/`
- `eval/metagame/` for thesis eval
- `eval/b2_disagreement/` when B2 diagnosis is run
- `replays/`
- `figures/paper/`
- `figures/data/` when figure data exports are produced

Smoke/demo artifacts must stay clearly labeled and must not be cited as thesis
evidence.
