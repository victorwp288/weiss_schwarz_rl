# Final Progress Branch Contents - 2026-05-07

Branch: `final/progress`

Purpose: preserve the thesis experiment state, including configs, evaluation data, diagnostic figures, and model/run artifacts that are small enough to keep in GitHub.

## Included

Documentation and analysis:

- `THESIS_EXPERIMENT_FINDINGS_AND_CONFIG_NOTES_20260507.md`
- `THESIS_FINAL_READINESS_AUDIT_20260507.md`
- `THESIS_RESULTS_STATUS_20260507.md`
- `SECTION7_RESULTS_DRAFT_20260507.md`
- `LOCAL_FIGURE_DEEP_DIVE_20260507.md`

Figures and local summaries:

- `thesis_figures_final/`
- `vast_artifacts/`
- `scripts/make_thesis_figures.py`

Configs and run logs:

- Emergency/ablation/baseline configs under `configs/presets/`
- `run_logs/`

Full run artifacts included despite `runs/*` being normally ignored:

- `runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`
- `runs/ablation_exp028_no_b1_lane_override_from_exp023_to420_20260506`
- `runs/ablation_exp029_weak_b1_auto_from_exp023_to420_20260506`
- `runs/baseline_nogru_impala_fixed_heuristic_u220_20260506`
- `runs/baseline_ppo_lite_fixed_heuristic_u220_20260506`

These contain checkpoints, snapshots, eval outputs, TensorBoard files, logs, manifests, and canonical configs.

## Excluded

- `.venv-exp034/`
- backup scratch files such as `*.before_*`
- old symlinked transferred run dirs with zero local size
- large external desktop archives

## Size Check

The included full run directories are approximately:

- main thesis run: `264M`
- exp028 ablation: `180M`
- exp029 ablation: `180M`
- No-GRU baseline: `52M`
- PPO-lite baseline: `94M`

No individual included file was above GitHub's 100MB file limit at the time of staging.
