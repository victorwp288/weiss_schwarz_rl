# AGENTS.md

## Project

This is the Weiss Schwarz reinforcement learning repo for a computer science master thesis.

The goal is to make the repo a lean, clean, reproducible research system that can train, evaluate, document, and defend:

1. B0 RandomLegal
2. B1 NoLeague
3. B2 HeuristicPublic
4. B3 HeuristicPublicAggro
5. B4 HeuristicPublicControl
6. Main league thesis model
7. Baselines and ablations

Use `weiss-schwarz-simulator` version 1.1.0 or newer as the standard simulator dependency.

## Current objective

Follow `RL_REBUILD_PLAN.md`.

The rebuild should produce:

1. Simple standard training commands
2. Simple standard evaluation commands
3. A small number of clear configs
4. A documented artifact layout
5. A trainable and evaluable B1 path
6. A trainable and evaluable main league path
7. Trainable baselines and ablations
8. Reproducible thesis eval and figure export commands

## Fixed deck policy

Primary thesis comparisons use this policy:

1. Focal model, B0, B1, and B2 use `preset:main_deck_5hy_yotsuba_v1`
2. B3 aggro uses `preset:aggro_deck_5hy_nino_v1`
3. B4 control uses `preset:control_deck_jj_s66_v1`
4. Multideck is exploratory, not the primary thesis comparison

Do not mix fixed deck and multideck results without clearly labeling them.

## Code style

Prefer:

1. Small focused modules
2. Clear names
3. Explicit types
4. Explicit errors
5. Simple control flow
6. Readable research code
7. Config driven workflows
8. Tests around important invariants

Avoid:

1. Giant files
2. Duplicate training systems
3. Config maze
4. Flag soup
5. Hidden fallbacks
6. Silent legality fixes
7. Half migrated old and new systems
8. Clever abstractions that make thesis defense harder

## Standard commands

Discover actual commands from `pyproject.toml`, README, scripts, and existing docs.

Expected validation commands may include:

1. `uv sync`
2. `uv run python -m pytest`
3. `uv run ruff check .`
4. `uv run ruff format --check .`
5. `uv run mypy python`

If a command is not configured, document that instead of pretending it passed.

## Rebuild log

Update `docs/rebuild_log.md` after major milestones.

Each entry should include:

1. What changed
2. Commands run
3. Results
4. Tests added
5. Failures found
6. Fixes applied
7. Behavior changes, if any
8. Performance numbers, if any
9. Remaining risks
10. Next action

## Evaluation rules

Do not overclaim from tiny local evals.

Any thesis relevant model claim must be backed by saved artifacts.

If models flatlines or behaves suspiciously, diagnose:

1. Deck selection
2. Observation encoding
3. Legal action mapping
4. Action masks
5. Candidate ordering
6. Reward handling
7. Opponent loading
8. Seat swap logic
9. Seed pairing
10. Replay traces

Document findings in `docs/rebuild_log.md`.

## Definition of done

The rebuild is not complete until:

1. `RL_REBUILD_PLAN.md` has been executed or updated with justified changes
2. The repo has a small clear thesis workflow
3. Standard training and eval commands work without flag soup
4. Simulator 1.1.0 optimized path is the default
5. Fixed deck policy is implemented and tested
6. B1 baseline path is trainable and evaluable
7. Main league path is trainable and evaluable
8. Eval, export, and figure pipeline works
9. Tests and lint pass, or exceptions are documented
10. A fresh reader can reproduce the thesis workflow from README and docs