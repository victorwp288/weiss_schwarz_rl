# Repository Map

This is the source-of-truth map for where work belongs after the thesis cleanup.
When adding or moving code, prefer extending these lanes instead of creating a
new top-level area or another path-based script implementation.

## Public Surface

- `python -m weiss_rl.cli`: canonical package CLI for thesis workflows.
- `python -m weiss_rl.training.train_entrypoint`: low-level training command.
- `python -m weiss_rl.workflows.eval_entrypoint`: low-level evaluation command.
- `python -m weiss_rl.workflows.verify_repo_entrypoint`: release-facing local verification.
- `python/scripts/*.py`: compatibility shims only. These keep old run history,
  notebooks, and shell commands working, but package modules own behavior.

## Source Layout

- `python/weiss_rl/config/`: strict config models, loading, overrides, and config hashes.
- `python/weiss_rl/core/`: simulator boundary contract, action catalog, masks,
  legal action handling, observation layout, and invariant helpers.
- `python/weiss_rl/envs/`: simulator-backed environment wrappers.
- `python/weiss_rl/runtime_components/`: queue runtime internals behind the
  `python/weiss_rl/runtime.py` facade.
- `python/weiss_rl/learners/`: learner implementations such as IMPALA/V-trace
  and PPO-lite.
- `python/weiss_rl/models/`: policy/value modules and structured action scoring
  behind the `python/weiss_rl/model.py` facade.
- `python/weiss_rl/training/`: reusable training orchestration, checkpointing,
  and train entrypoint helpers.
- `python/weiss_rl/eval/`: deterministic evaluation, policy resolution,
  uncertainty, paper-readiness checks, and eval diagnostics.
- `python/weiss_rl/league/`: snapshot registry, PFSP, promotion gates, and
  opponent pools.
- `python/weiss_rl/workflows/`: package CLI parser construction, workflow
  profiles, thesis wrapper, command builders, figure/export orchestration, and
  verification plans.
- `python/weiss_rl/diagnostics/`: repository gates, artifact scans, training
  probes, replay audits, and analysis commands that do not own training state.
- `python/weiss_rl/experiments/`: bounded experiment launchers, scorecards, and
  one-off thesis investigation entrypoints that still need to remain reproducible.

## Config Layout

- `configs/thesis/`: primary thesis workflows and fixed-deck comparison configs.
- `configs/thesis/ablations/`: small documented ablation surface.
- `configs/presets/`: compatibility and lower-level stack presets used by the
  public thesis wrappers.
- `configs/seeds/`: committed seed files and deterministic input fixtures.

Do not add another config tree without documenting why it is a distinct public
surface.

## Artifact Layout

- `runs/`: protected thesis run artifacts and new reproducible runs.
- `run_logs/`: protected external run logs that are part of the thesis evidence.
- `vast_artifacts/`: downloaded remote artifacts that are intentionally kept.
- `thesis_figures_final/`: final figure outputs used by thesis/report material.
- `temp/`, caches, build output, ad-hoc zips, and scratch exports: untracked
  local workspace material. Archive outside the checkout if it must be kept.

Existing historical artifacts are evidence, not implementation. Treat them as
read-only unless a cleanup or reproduction task explicitly promotes a new result.

## Refactor Rules

- New behavior belongs in package modules under `python/weiss_rl/`.
- New path-based scripts are not allowed unless they are thin compatibility
  shims that delegate to package entrypoints.
- Parser behavior has one owner. Prefer composing parser modules under
  `python/weiss_rl/workflows/` over duplicating full command trees.
- Generated outputs should not appear as tracked top-level paths.
- When a module grows into several responsibilities, split by thesis workflow
  boundary: config, runtime, training, eval, diagnostics, or reporting.

## Hygiene Gate

Run this before review:

```bash
uv run python -m weiss_rl.diagnostics.repo_hygiene_check_entrypoint
```

The gate checks that tracked files stay within the documented top-level layout,
that generated root outputs are not tracked, and that `python/scripts/*.py`
files remain thin compatibility shims. It is also part of
`python -m weiss_rl.workflows.verify_repo_entrypoint`.
