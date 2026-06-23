# Changelog

This project is thesis-run driven. Summarize changes here when they affect
contributors, public commands, docs, or compatibility surfaces.

## Unreleased

### Public Workflow And Docs

- Kept `python -m weiss_rl.cli` as the public six-command thesis surface:
  `train-b1`, `train-main`, `smoke-eval`, `eval-final`, `figures`, and
  `b2-audit`.
- Retired path-based `python/scripts/*.py` command wrappers. Use package
  modules under `python -m weiss_rl...`.
- Removed stale root example and script surfaces in favor of maintained package
  commands, tests, and Make targets.
- Consolidated docs navigation, artifact ownership, local README pointers, and
  contributor guidance around the active docs hub.
- Split human-play deployment guidance across the high-level deployment doc,
  the frontend README, and the backend container README.

### Package Surface Cleanup

- Replaced broad lazy aliases and root-level compatibility facades with concrete
  imports across workflow, training, evaluation, readiness, targeted-confirm,
  parallel-final-eval, and learner packages.
- Kept live compatibility only where tests or retained callers still require a
  stable import surface.
- Removed orphan smoke/facade modules that no longer owned behavior, including
  actor-worker smoke wrappers, final-eval worker facades, selection shims, and
  readiness fixture/check facades.

### Training, Runtime, And Evaluation Refactors

- Split training entrypoint, checkpointing, snapshot registry, periodic
  dev-eval, promotion gates, warmstarts, replay data, and learner construction
  into focused modules with characterization tests.
- Split runtime collection, batching, opponent sampling, legal-action handling,
  policy inference, shared transport, and debug validation into smaller modules
  while preserving behavior-sensitive runtime contracts.
- Split evaluation policy resolution, B1 baseline lookup, snapshot lookup,
  final-eval planning, targeted confirmation, metagame outputs, and
  paper-readiness checks into concrete packages.
- Preserved public behavior around legal-action order, run manifests, snapshot
  paths, policy ordering, paired seeds, payoff folding, and paper-readiness
  artifact layout.

### Models And Learners

- Extracted structured-model observation, candidate, scoring, public-heuristic,
  tensor, and sampling helpers into concrete model modules.
- Extracted IMPALA/V-trace action log-probability, packed-row, structured
  teacher, auxiliary metric, optimizer, and update-stage helpers into concrete
  learner modules.
- Kept the public policy/value model and learner entry surfaces stable while
  moving private helper imports to their owner modules.

### Verification And Typing

- Added or tightened characterization tests around refactored runtime,
  checkpoint, config, evaluation, model, learner, and docs/config surfaces.
- Reduced broad mypy and lint gaps in several refactor slices. Re-run the
  verifier before treating the full dirty worktree as release-ready.
