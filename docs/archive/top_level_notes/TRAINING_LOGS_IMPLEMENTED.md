# Training logs implementation snapshot

Short reviewer-facing status note for the learner-side JSONL logger.
For usage details and examples, see `docs/training_logs.md`.

## Confirmed scope today

- `python/weiss_rl/training_logger.py` writes structured JSONL records
- `python/weiss_rl/learners/impala_learner.py` can emit throughput and masked V-trace metrics
- `python/weiss_rl/learners/vtrace.py` requires a legality surface for diagnostics and will not compute health metrics from illegal actions
- `examples/training_logs_example.py` demonstrates standalone usage

## Explicit non-claims

These items are **not** implemented in the current repo entrypoints:

- `train.py` wiring
- full end-to-end actor/learner integration
- aggregated actor-lag logging from a live training loop

## Logged core fields

Every valid record includes:

- `update_count`
- `wall_clock_seconds`
- `wall_clock_ms`
- `policy_version`

Optional checkpoint-sync lag fields are named:

- `checkpoint_lag_updates`
- `checkpoint_lag_percentile_p50`
- `checkpoint_lag_percentile_p90`

## Checkpoint semantics

- learner checkpoints are written as `checkpoint_<update_count>.pt`
- actor lag in this scaffold is therefore checkpoint-update lag, not arbitrary per-step learner staleness

## Validation

Relevant tests:

- `python/weiss_rl/tests/test_training_logger.py`
- `python/weiss_rl/tests/test_actor_worker.py`
