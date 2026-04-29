# Refactor Plan - 2026-04-28

This is a behavior-preserving refactor roadmap for making the thesis codebase easier to explain, safer to change, and less dependent on long files with hidden control flow. The goal is not to make the code pretty at the expense of the result pipeline. The goal is to expose the real architecture already present in the project: config, simulator boundary, runtime collection, learner updates, league/snapshot policy management, evaluation, and thesis artifacts.

## Current Evidence

Static scan, excluding `runs/`, found 385 source/config/doc files and 920 collected tests under `python/weiss_rl/tests`.

Largest source hotspots:

| File | Lines | Main issue |
| --- | ---: | --- |
| `python/scripts/train.py` | 9,075 by PowerShell line count; 9,744 in direct agent scan | CLI, manifest setup, training loop, eval, promotion, checkpointing, rollback, and finalization are all in one entrypoint. |
| `python/weiss_rl/runtime.py` | 7,928 by PowerShell line count; 8,287 in direct agent scan | `QueueRuntime` owns environment setup, collector backends, league routing, opponent policies, transport, batch building, metrics, and teacher labels. |
| `python/weiss_rl/learners/impala_learner.py` | 5,734 | Learner update, structured auxiliary metrics, reference-policy losses, distillation, and fault handling are mixed together. |
| `python/weiss_rl/model.py` | 4,851 | Model, structured legal head, packed scorer, heuristic bias logic, and multiple scoring modes live in one module. |
| `python/weiss_rl/config/parse.py` | 2,601 | Long hand-written section parser; defaults, compatibility, schema, and validation are intertwined. |

Longest functions/classes found:

- `compute_structured_teacher_auxiliary_metrics()` in `python/weiss_rl/learners/impala_learner.py:823`, about 1,296 lines.
- `_run_minimal_training()` in `python/scripts/train.py:8023`, about 1,192 lines.
- `_parse_training_config()` in `python/weiss_rl/config/parse.py:488`, about 1,094 lines.
- `QueueRuntime` in `python/weiss_rl/runtime.py:1480`, about 6,315 lines.
- `ImpalaLearner` in `python/weiss_rl/learners/impala_learner.py:2559`, about 3,467 lines.
- `_StructuredLegalActionHead` in `python/weiss_rl/model.py:1031`, about 3,404 lines.

Verification signals from this pass:

- `uv run --extra dev python -m pytest -q --collect-only python/weiss_rl/tests` collected 920 tests.
- `uv run --extra dev python -m ruff check python tests examples python/scripts` currently reports 36 lint errors. Many are import sorting, but there are real signals too: missing `Mapping` imports in `eval/harness.py`, duplicate metric key in `impala_learner.py`, unused locals in `runtime.py`, and one suspicious module-level import placement in `train.py`.
- `uv run --extra dev python -m vulture python/weiss_rl python/scripts examples --min-confidence 80` reports one 100 percent unreachable-code finding in `python/weiss_rl/runtime.py:5342`.

## Do Not Break These Contracts

These are the behavior surfaces that should remain stable unless explicitly changed and documented:

- Public CLI flags and warnings for `python/scripts/train.py`, `python/scripts/eval.py`, and `python/scripts/thesis_run.py`.
- Run identity computation, manifest fields, config hash stability, and simulator spec-bundle validation.
- Artifact tree shape documented in `docs/artifact_contract.md`: `training/`, `eval/final_eval/`, `eval/diagnostics/`, `eval/metagame/`, `replays/`, and `figures/paper/`.
- Snapshot registry schema, snapshot paths, aliases, `policy_meta.json`, `weights.pt`, source roots, and B1 no-league baseline import behavior.
- Periodic dev-eval and promotion gate seed formulas, matchup ordering, anchor aliases, confidence/truncation guard behavior, and rollback semantics.
- TensorBoard scalar names, training metric names, JSON/CSV/JSONL output shapes, and replay bundle references.
- Runtime modes from `docs/runtime_modes.md`: `paper_eval_pinned`, `train_ordered`, `train_async_fast`, and `public_demo`.
- Packed legality semantics, empty-legal PASS fallback, legal ID ordering, and simulator-backed `DecisionBoundaryEnv` contracts.

## Code Style Bar

The refactor should make the code look intentional and thesis-explainable. The target style is boring in the best way: explicit names, narrow modules, typed data objects, small orchestration layers, and comments only where they explain decisions that are not obvious from the code.

Guidelines:

- Prefer names that explain the domain role: `PromotionGateExecutor`, `SnapshotImportService`, `CanonicalEvalRequest`, `PolicyResolutionResult`, `TrainingLoopState`, `RuntimeCollectorMetrics`.
- Keep scripts thin. `python/scripts/*.py` files should parse arguments, call package code, print user-facing summaries, and preserve compatibility flags.
- Use dataclasses or small typed result objects for long argument chains. A function taking 12 loosely related parameters is a refactor smell.
- Split modules by responsibility, not by convenience. Good module boundaries are config parsing, bootstrap, training session, snapshot artifacts, runtime collection, learner losses, eval orchestration, policy resolution, artifact contracts, and compatibility.
- Keep pure helpers pure. Anything that computes config hashes, seed schedules, policy IDs, metrics, or artifact specs should avoid hidden file writes, global mutation, and device side effects.
- Make compatibility explicit. Deprecated flags, legacy config paths, old artifact aliases, and fallback behavior should live in `compat`-named helpers or clearly marked sections.
- Avoid clever abstractions. Add a class or protocol only when it names a real concept in the thesis pipeline or removes repeated control flow.
- Prefer clear control flow over dense expressions in training/runtime code. Long RL pipelines are already hard enough; a few extra lines are fine when they make ordering visible.
- Use comments for invariants, not narration. Good comments explain why a seed formula, artifact path, rollout ordering, or compatibility alias must remain stable.
- Keep public metric keys and artifact field names stable. When a name is ugly but already public, wrap it with a better internal name rather than silently changing output.
- Write tests next to the responsibility being protected. If a refactor extracts `policy_resolution`, its tests should assert policy identity, source roots, legacy aliases, and model-loading boundaries directly.
- Avoid sweeping formatting-only commits mixed with behavior changes. Style cleanup is useful, but it should not hide semantic diffs.
- Prefer package imports over script imports as code moves. Scripts can remain compatibility wrappers, but package modules should not depend on `python/scripts/train.py`.
- Keep error messages concrete. They should name the config key, policy ID, artifact path, or runtime mode that caused the failure.
- Let docs mirror code concepts. If the code has `TrainingSession`, `SnapshotImportService`, and `CanonicalEvalRequest`, the docs should use those same names.

File-size targets are guidelines, not hard rules:

- Entry scripts: under 250 lines when feasible.
- Regular modules: usually under 800 lines.
- Classes: usually under 300 lines.
- Functions: usually under 80 lines, with exceptions for carefully structured numerical kernels or unavoidable compatibility adapters.
- Tests may be longer, but large fixture builders should be extracted and named.

## Immediate Risks To Pin Before Refactoring

These should be characterized and fixed separately before moving large blocks of code:

1. `python/weiss_rl/runtime.py:5285` `_central_sample_policy_rows_ids()` returns before the `any_model_rows` branch at `runtime.py:5342`. In mixed heuristic/model actor rows, model row sampling may be skipped.
2. `python/weiss_rl/runtime.py:4444` `_apply_policy_rows_mask()` lacks the `source_label` argument passed by a caller around `runtime.py:4318`; the ids path has the argument around `runtime.py:4597`. Mask and ids paths have drifted.
3. Central collection fills `teacher_move_source` around `runtime.py:5834` and `runtime.py:6220`, but the `RuntimeUnroll` construction around `runtime.py:6315` omits it. Other collector paths pass it around `runtime.py:6569`, `runtime.py:6908`, and `runtime.py:7316`.
4. `python/weiss_rl/eval/harness.py` uses `Mapping` in annotations without importing it.
5. `python/weiss_rl/learners/impala_learner.py:3097` and `:3098` repeat the same `"raw_b1_top_action_ce"` key.

Treat these as bug-fix or characterization PRs, not as part of broad extraction. That keeps behavior changes auditable.

## Target Architecture

The codebase should end up with thin entrypoints and explicit service boundaries:

- `python/scripts/train.py`: argument parsing and compatibility wrapper only.
- `weiss_rl.training.bootstrap`: CLI-to-session setup, config loading, run identity, distributed/autoscale setup, manifest writing.
- `weiss_rl.training.session`: `TrainingSession` / `TrainingLoopState`, one-update loop, checkpoint tick, eval/promotion tick, finalization.
- `weiss_rl.training.snapshots`: snapshot writes, imports, aliases, retention, registry coordination.
- `weiss_rl.training.eval_tasks`: shared periodic dev-eval and promotion-gate executors.
- `weiss_rl.runtime`: public runtime facade, with internals split into env adapter, collector implementations, batch assembly, league role planner, transport/shared memory, and metrics.
- `weiss_rl.learners.losses`: structured teacher auxiliary, reference-policy BC, raw B1 distillation, counterfactual/residual losses, and fault-bundle helpers.
- `weiss_rl.model.structured`: structured legal action head and packed scorer internals.
- `weiss_rl.eval.canonical`: package-level canonical eval request/result; `python/scripts/eval.py` becomes dispatch only.
- `weiss_rl.eval.policy_resolution`: registry/source-root policy resolution separate from model loading.
- `weiss_rl.artifact_contract`: declarative canonical and compatibility artifact specs used by readiness checks and fixtures.
- `weiss_rl.compat`: deprecated CLI/config/path aliases and old naming bridges, so legacy support is explicit rather than scattered.

## Staged Plan

### Stage 0 - Baseline And Characterization

Goal: make the current behavior observable before moving code.

Actions:

- Add targeted tests for the three runtime hazards above: mixed heuristic/model central rows, mask-vs-ids policy-row parity, and central `teacher_move_source`.
- Add golden tests for config parse/hash stability for current public config surfaces.
- Add golden tests for B1 baseline import, registry-root snapshot resolution, periodic dev-eval payloads, promotion payloads, and checkpoint rollback behavior.
- Capture current `thesis_run.py --list-presets` output and representative dry-run plans as fixtures.

Verification:

- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_snapshot_registry.py`
- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_thesis_run_wrapper.py python/weiss_rl/tests/test_entrypoints.py`
- `uv run --extra dev python -m ruff check python tests examples python/scripts`

### Stage 1 - Fix Known Pre-Refactor Hazards

Goal: start from a clean behavioral baseline.

Actions:

- Fix the unreachable model-row sampling branch in `_central_sample_policy_rows_ids()`.
- Align `_apply_policy_rows_mask()` signature/diagnostics with the ids path.
- Preserve central `teacher_move_source` in `RuntimeUnroll`.
- Fix missing imports and duplicate metric keys.
- Decide whether obvious unused locals in `runtime.py` are truly dead or should be wired.

Verification:

- Targeted tests from Stage 0.
- Full unit suite when feasible: `uv run --extra dev python -m pytest -q python/weiss_rl/tests`.
- Record before/after `performance.jsonl` or a compact profile for any runtime-touching fix.

### Stage 2 - Thin Training Entrypoint

Goal: make `train.py` readable without changing its public behavior.

Actions:

- Move CLI/config/bootstrap helpers into `weiss_rl.training.bootstrap`.
- Keep `python/scripts/train.py` as the public compatibility script.
- Preserve all CLI flags, deprecated `--run-id` behavior, public-demo behavior, scaffold path behavior, run identity, manifest fields, and printed startup banner.
- Introduce `TrainingSession` / `TrainingLoopState` as a wrapper around the existing `_run_minimal_training()` control flow, without changing ordering yet.

Verification:

- `python/weiss_rl/tests/test_entrypoints.py`
- `python/weiss_rl/tests/test_thesis_run_wrapper.py`
- `uv run --extra dev python python/scripts/thesis_run.py --list-presets`
- `uv run --extra dev python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label refactor_stack_smoke`

### Stage 3 - Split Training Loop Responsibilities

Goal: turn `_run_minimal_training()` into understandable lifecycle steps.

Actions:

- Extract runtime setup, learner setup, one-update execution, checkpoint tick, eval tick, promotion tick, rollback, async completion, and finalization.
- Keep update ordering, prefetch timing, checkpoint cadence, registry writes, TensorBoard scalar names, and metric payloads stable.
- Use explicit dataclasses for loop state instead of long argument chains.

Verification:

- `test_train_stall_monitor.py`
- `test_snapshot_registry.py`
- `test_training_logger.py`
- `test_entrypoints.py`
- One deterministic smoke and one async smoke, with `performance.jsonl` compared to baseline.

### Stage 4 - Unify Eval And Promotion Executors

Goal: remove duplicated policy resolution and sync/async result handling.

Actions:

- Share model loading, pinned sampling, seed expansion, anchor/opponent resolution, and result persistence between `_PeriodicDevEvalRunner` and `_PromotionGateRunner`.
- Make async and sync paths use one completion handler.
- Preserve `_periodic_dev_eval_rng_seed`, `_promotion_gate_rng_seed`, matchup ordering, anchor IDs, champion aliases, pin/unpin behavior, and rejection writes.

Verification:

- `test_train_stall_monitor.py`
- `test_snapshot_registry.py`
- Specific tests around confirmatory dev eval, confidence-only gates, B2 flatline audit requests, and promotion rollback.

### Stage 5 - Extract Snapshot And League Artifact Services

Goal: put snapshot persistence and league state in one understandable ownership layer.

Actions:

- Move B1 import, seed snapshot imports, resume imports, hard negative snapshots, alias refresh, retention, and registry save/load coordination out of `train.py`.
- Keep registry paths and metadata exactly stable.
- Clarify which league logic belongs in `weiss_rl.league` versus `weiss_rl.training`.

Verification:

- `test_snapshot_registry.py`
- `test_heuristic_public.py` tests that depend on registry-backed eval resolution.
- `test_thesis_run_wrapper.py` for B1 baseline wiring.

### Stage 6 - Split Config Parsing

Goal: make configs thesis-explainable.

Actions:

- Split `config/parse.py` by section: experiment, system, environment, model, training, league, evaluation, metagame.
- Keep dataclass shapes and canonical hashes stable.
- Move legacy path resolution and legacy field aliases into explicit compat helpers.
- Audit under-wired/no-op fields like `opponent_sampling`, warmup gates, and `promotion.threshold`; either wire them or mark them deprecated.

Verification:

- `test_config.py`
- `test_config_loader.py`
- `test_config_overrides.py`
- `test_study_config.py`
- `test_sweeps.py`
- `uv run --extra dev python python/scripts/thesis_run.py --list-presets`

### Stage 7 - Package Canonical Eval

Goal: make `eval.py` a thin CLI and make eval logic reusable.

Actions:

- Introduce `CanonicalEvalRequest` and `CanonicalEvalResult`.
- Move `_run_canonical_eval_pipeline()` into package code.
- Split `resolve_eval_policies()` into registry/source-root location, policy identity resolution, and model loading.
- Keep public-demo mode, completed-manifest reuse behavior, TensorBoard output, metagame calls, figure calls, readiness calls, and policy-set selection details stable.

Verification:

- `test_entrypoints.py`
- `test_final_eval.py`
- `test_policy_set.py`
- `test_heuristic_public.py`
- `test_paper_readiness.py`

### Stage 8 - Declarative Artifact Contract

Goal: make thesis artifact expectations visible in code, not duplicated by path strings.

Actions:

- Create canonical and compatibility artifact spec objects.
- Make paper readiness, fixture generation, and docs consume the same spec.
- Separate artifact presence checks from statistical guardrails.
- Keep compatibility paths explicit and temporary.

Verification:

- `make artifact-contract` or equivalent Python commands.
- `test_paper_readiness.py`
- `test_artifact_hygiene.py`
- Readiness check on one generated fixture and one real completed run.

### Stage 9 - Peel Runtime Internals

Goal: reduce `QueueRuntime` without changing collector semantics or throughput.

Actions:

- Extract pure contract helpers and batch concatenation first.
- Extract env adapter and `DecisionBoundaryEnv` packing contract; stop importing private `_pack_batch` from runtime.
- Extract collector implementations: central packed, scalar, native rollout, process/shared-memory transport.
- Extract league role planner and opponent policy residency.
- Extract metrics/counter aggregation.
- Keep central/scalar/native behavior benchmarked separately. Do not generalize performance gains across no-league and league modes without evidence.

Verification:

- `test_runtime.py`
- `test_actor_worker.py`
- `test_decision_env.py`
- `test_trajectory_buffers.py`
- `profile_structured_hotpaths.py --mode structured`
- `profile_structured_hotpaths.py --mode heuristic`
- `profile_train_job.py` against a compact no-league benchmark config and one league smoke config.

### Stage 10 - Modularize Learner And Model

Goal: make the learning objective explainable.

Actions:

- Move structured teacher auxiliary metrics into a dedicated module.
- Move reference-policy BC, raw B1 distillation, counterfactual/residual losses, V-trace glue, and numeric fault bundle writing into named components.
- Split `_StructuredLegalActionHead` into feature preparation, scoring, candidate planning, public heuristic bias, and sampling.
- Preserve checkpoint compatibility and every public metric key.

Verification:

- `test_impala_learner.py`
- `test_vtrace.py`
- `test_training_logger.py`
- `test_contracts.py`
- A short fixed-batch learner regression confirming loss decreases and metric keys are unchanged.

### Stage 11 - Quarantine Or Remove Legacy Surfaces

Goal: delete only what has proven unused, otherwise move it behind clear compatibility names.

Candidates:

- `ActorWorker` and legacy actor-worker scaffolding.
- Duplicate unroll contracts across runtime, actors, and trajectory modules.
- `RuntimeUnroll.behavior_logits` if it is always `None`.
- `minimal_batch` if it is only test-only smoke wiring.
- Legacy config paths and deprecated `--run-id`.
- Legacy payoff matrix exports and `eval/selection.py` re-export.

Rule:

- No deletion until there is either test coverage proving the active replacement or a compatibility module that preserves old imports/paths.

## Verification Ladder

Use a ladder rather than one giant all-or-nothing check:

1. Static: `ruff`, `mypy` on touched surfaces, `vulture` after obvious cleanup.
2. Unit/contract: targeted tests for touched modules.
3. Entry point: train/eval/thesis wrapper dry-runs and public-demo smoke.
4. Artifact: fixture readiness and artifact hygiene.
5. Simulator: `make simulator-check` or equivalent `uv run --extra dev --extra sim`.
6. Performance: structured hotpath profile plus compact train job profile.
7. Thesis confidence: one representative run tree passes readiness and produces expected eval/metagame/figure outputs.

## Progress Log

Current best metrics from this planning pass:

- 920 tests collected.
- Largest refactor target: `QueueRuntime`, about 6,315 lines in one class.
- Largest single function target: `compute_structured_teacher_auxiliary_metrics()`, about 1,296 lines.
- Static tooling already found actionable hazards: 36 Ruff findings and one 100 percent confidence unreachable block from Vulture.

Failed ideas / avoided shortcuts:

- Do not start by deleting "dead-looking" code. Several old surfaces are still compatibility paths or test scaffolds.
- Do not start with broad formatting churn. It would obscure behavioral changes.
- Do not split runtime first. It is the highest-risk area and needs characterization plus performance baselines.
- Do not trust unit tests alone for runtime changes. Simulator smoke and telemetry are needed.

Next hypotheses:

- A small pre-refactor bug-fix slice will pay off more than immediate module movement.
- Most thesis-readability gain will come from extracting explicit training/eval/session/snapshot concepts, not from renaming internals.
- Runtime can be made explainable once policy-row routing, collector backends, and league role planning are separated.
- Learner/model explainability depends on naming the objective components: V-trace, teacher auxiliary, reference BC, distillation, residual/counterfactual losses, and structured legal action scoring.
