# Training logs status

This branch contains the learner-side JSONL logging scaffold only.

## Confirmed scope on this stack

- `python/weiss_rl/training_logger.py` writes structured JSONL records
- `python/weiss_rl/learners/impala_learner.py` can emit throughput and masked V-trace metrics
- `python/weiss_rl/learners/vtrace.py` now requires a legality surface for diagnostics and will not compute health metrics from illegal actions
- `examples/training_logs_example.py` demonstrates standalone usage

## Explicit non-claims

These items are **not** implemented on this branch:

- `train.py` wiring
- full end-to-end actor/learner integration
- aggregated actor lag logging from the live training loop

That broader integration belongs to M3-08 scope.

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
