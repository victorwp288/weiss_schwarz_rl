# Runtime Components Package Map

Use this package for simulator-backed rollout collection and queue-runtime
helpers. `queue_runtime.py` is the outer orchestrator; this package owns the
pieces it delegates to.

## Root Files

- `config.py`, `runtime_configuration.py`, `training_settings.py`,
  `teacher_settings.py`, `topology.py`, `devices.py`, `types.py`: runtime
  configuration, device/topology decisions, and shared types.
- `startup_logging.py`, `lifecycle.py`, `support.py`, `hashing.py`: startup,
  lifecycle, and support utilities.
- `opponent_mixin.py`, `opponent_startup.py`, `opponent_context.py`,
  `opponent_rows.py`, `opponent_pool_refresh_log.py`, `outcomes.py`,
  `policy_ids.py`: opponent state, league sampling, and logging adapters.
- `heuristic_fast_path.py`, `heuristic_policy_setup.py`,
  `heuristic_public_actions.py`, `heuristic_rollouts.py`,
  `teacher_heuristic_mixin.py`, `teacher_labels.py`,
  `structured_warmstart.py`: heuristic rollout, teacher-label, and warmstart
  support.

## Subpackages

- `actors/`: actor startup, state, routing, unroll execution, and
  actor-side policy rows.
- `central/`: centralized actor collection, policy phases, row partitioning,
  unroll assembly, and finalization.
- `collection/`: collector state, action execution, legal-action steps, unroll
  storage, batch collection, and terminal resets.
- `batching/`: batch-level bootstrapping, counters, legal batching, metrics,
  and reward backfill helpers.
- `actions/`: action catalog setup, dense/packed action surfaces, pass/mulligan
  guards, legal metadata, and policy row/output records.
- `rewards/`: reward flow maps and reward-shaping plans/counters.
- `policy_inference/`, `opponents/`, `opponent_policies/`, `shared_memory/`,
  `ipc_shared/`, `heuristic_rollout/`: lower-level helpers for the named
  runtime subsystems.
