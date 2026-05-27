# Architecture

This repository is a behavior-sensitive thesis RL pipeline for Weiss Schwarz. The code is organized around explicit contracts rather than hidden global state.

## Top-Level Flow

1. A stack config is loaded from `configs/` through `weiss_rl.config`.
2. `python/scripts/train.py` creates a run manifest, verifies simulator/config hashes, builds the model and learner, and drives `QueueRuntime`.
3. `QueueRuntime` collects decision-boundary rollouts from simulator-backed environments and packages learner batches.
4. Learners update policy/value models and write checkpoints, metrics, snapshots, and TensorBoard logs.
5. `python/scripts/eval.py` resolves deterministic policy sets, runs pinned final evaluation, writes summaries, and optionally drives metagame/readiness reporting.

## Main Packages

- `weiss_rl.config`: strict stack config models, parsing, overrides, and canonical hashes.
- `weiss_rl.config.parsing_utils`: strict YAML/JSON document loading, scalar/list validators, unknown-key rejection, preset inheritance merge helpers, and repo path resolution used by the stack config parser.
- `weiss_rl.config.sections_core`: experiment and system section parsers used by the stack config parser.
- `weiss_rl.config.sections_curriculum`: curriculum section parser and simulator payload normalizer used by the stack config parser.
- `weiss_rl.config.sections_environment`: environment and reward section parsers used by the stack config parser.
- `weiss_rl.config.sections_evaluation`: evaluation section parser and hard-fail legal-fingerprint mismatch policy guard used by the stack config parser.
- `weiss_rl.config.sections_league`: league pool, sampling, warmup, and promotion section parser used by the stack config parser.
- `weiss_rl.config.sections_model`: model section parser and model choice constants used by the stack config parser.
- `weiss_rl.config.sections_reproducibility`: reproducibility section parser and fail-fast spec mismatch policy enforcement used by the stack config parser.
- `weiss_rl.config.sections_training`: training algorithm, PPO, profiling, backend, public-heuristic, diversity, structured-metrics, teacher-auxiliary, warmstart, and precision section parser used by the stack config parser.
- `weiss_rl.config.seed_sets`: seed-set path resolution and canonical run-artifact seed override parsing used by the stack config parser.
- `weiss_rl.core`: simulator/domain contracts, legal-action batches, action catalog decoding, masking, observation layout, schedules, and termination classification.
- `weiss_rl.artifacts`: run artifact layout, manifest writing, and reproducibility/hash helpers.
- `weiss_rl.diagnostics`: action diagnostics, artifact hygiene, job telemetry, TensorBoard and JSONL training log helpers, and CLI startup banner formatting.
- `weiss_rl.experiments`: no-league baseline helpers, experiment launch plans, sweep presets, structured-acceptance helpers, and public-demo scaffolding.
- `weiss_rl.envs`: simulator pool construction and `DecisionBoundaryEnv` wrappers.
- `weiss_rl.runtime`: compatibility facade for the training collection runtime and learner-batch assembly.
- `weiss_rl.runtime_components`: runtime internals used by `weiss_rl.runtime`, including actor state/model helpers, batching, collector commands, counters, IPC, metrics, opponent sampling, process collectors, shared transport, topology, and thread setup.
- `weiss_rl.learners`: IMPALA/V-trace and PPO-lite learner implementations.
- `weiss_rl.learners.action_logp`: dense masked and packed learner action log-probability/entropy helpers used by the IMPALA learner.
- `weiss_rl.learners.batch_fields`: model-device batch field conversion and validation helpers used by the IMPALA learner.
- `weiss_rl.learners.faults`: numeric fault bundle payload, finite tensor, and gradient diagnostic helpers used by the IMPALA learner.
- `weiss_rl.learners.legal_fields`: observation/action/legal-mask validators and packed legality resolution helpers used by the IMPALA learner.
- `weiss_rl.learners.logging`: checkpoint metadata and training metric record helpers used by the IMPALA learner.
- `weiss_rl.learners.structured_auxiliary`: public-heuristic profile normalization, structured action-catalog metadata, packed structured legal-view helpers, and packed auxiliary probability helpers used by the IMPALA learner.
- `weiss_rl.learners.structured_policy_metrics`: structured policy metric summaries used by IMPALA learner logging.
- `weiss_rl.learners.tensor_ops`: pure segment reductions, grouped sums, weighted means, and nonfinite diagnostics used by the IMPALA learner.
- `weiss_rl.learners.vtrace_torch`: torch V-trace target computation used by the IMPALA learner.
- `weiss_rl.learners.vtrace_diagnostics`: V-trace percentile and clipping metric summaries used by the IMPALA learner.
- `weiss_rl.model`: compatibility facade for the policy/value model and structured action logits.
- `weiss_rl.models`: model internals used by `weiss_rl.model`, including typed encoders, observation contracts, action plans/tables, dense/factorized/packed scoring, public heuristic scoring, deterministic sampling, tensor helpers, and snapshot loading.
- `weiss_rl.eval`: deterministic evaluation, policy resolution, payoff folding, uncertainty, diagnostics, and paper readiness.
- `weiss_rl.league`: snapshot registry, PFSP sampling, promotion gates, and opponent outcomes.
- `weiss_rl.training`: reusable training orchestration helpers extracted from public scripts, including CLI parsing, startup checks, paths, input validation, checkpoint/snapshot helpers, guidance schedules, environment builders, and minimal-batch utilities.
- `weiss_rl.training.manifest_layout`: training manifest actor-device layout helpers used by `python/scripts/train.py`.
- `python/scripts`: path-based public CLI entrypoints. These paths are compatibility surfaces.

## Behavior-Sensitive Boundaries

- Legal action IDs must keep stable ordering and indexing.
- Observation encoding, rewards, done/truncation semantics, and rollout packing are simulator contracts.
- Eval seeds, seat swaps, policy ordering, payoff folding, and uncertainty summaries are reporting contracts.
- Checkpoint schemas and snapshot registry paths are compatibility contracts.
- CLI defaults and config override behavior are public behavior.

## Compatibility Quarantine

The package CLI in `weiss_rl.cli` is the standard thesis surface. The
path-based `python/scripts/*` commands remain as compatibility wrappers,
diagnostic tools, or lower-level implementation entrypoints. Dated campaign,
rescue, sweep, and targeted-confirm scripts are quarantined from the thesis
workflow: do not use them for primary claims unless the reason and resulting
artifacts are recorded in `docs/rebuild_log.md`.

The current rebuild moved stale top-level facades into domain packages such as
`weiss_rl.core`, `weiss_rl.artifacts`, `weiss_rl.diagnostics`,
`weiss_rl.experiments`, `weiss_rl.models`, and `weiss_rl.runtime_components`.
Future extractions should stay small, preserve behavior with tests, and keep the
public thesis commands unchanged.
