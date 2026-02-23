# Config Layout

This project keeps thesis defaults in split, domain-focused YAML files under `configs/`.

## Grouping

- Core runtime:
  - `system_locked.yaml`
  - `model_locked.yaml`
  - `training_family_a_locked.yaml`
  - `environment_locked.yaml`
- Self-play and evaluation:
  - `league_locked.yaml`
  - `evaluation_locked.yaml`
  - `reproducibility_locked.yaml`
  - `metagame_locked.yaml`
  - `sensitivity_locked.yaml`
- Budgeting and optional ablations:
  - `compute_budget_locked.yaml`
  - `family_b_discount_ablation_locked.yaml`
  - `family_c_shaping_ablation_locked.yaml`
- Seed sets:
  - `seeds/dev_eval_seeds.txt`
  - `seeds/promotion_eval_seeds.txt`
  - `seeds/report_eval_seeds.txt`

## Consolidated Index

Use `rl_stack_locked.yaml` as the canonical entrypoint when wiring a loader.
It points to all split files and to committed seed-set files.

## Path Convention

All file paths in YAML configs are **repo-root relative**.

## Seed Files

Seed files contain one unsigned 64-bit integer per line with no comments.
This keeps parsing simple and deterministic across tooling.
