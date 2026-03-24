# Training logs

This branch provides a standalone JSONL logger for learner-side training metrics.
It does **not** wire the logger into `python/scripts/train.py`; that integration belongs to the broader training loop work in M3-08.

## What works on this stack

- `TrainingLogger` writes append-only JSONL records to `runs/<run>/logs/training_metrics.jsonl`
- `ImpalaLearner` can emit throughput and masked V-trace health metrics when `logs_dir` is configured
- `compute_vtrace_metrics()` respects the masking contract when the batch includes:
  - `logits`
  - `behavior_logits`
  - `actions`
  - either `legal_mask` or `legal_ids` + `legal_offsets`

If the legality surface is missing, V-trace diagnostics intentionally fall back to zero instead of computing from illegal actions.

## What is intentionally not claimed here

- no `train.py` integration on this branch
- no actor-to-learner lag aggregation on this branch
- no promise that checkpoint logging equals full end-to-end training readiness

## Record shape

`TrainingMetrics` always stores these fields:

- `update_count`
- `wall_clock_seconds`
- `wall_clock_ms`
- `policy_version`

Common learner-written fields:

- `loss`
- `throughput_samples_per_sec`
- `throughput_updates_per_sec`
- `vtrace_rho_mean`
- `vtrace_rho_p50`
- `vtrace_rho_p90`
- `vtrace_rho_p99`
- `vtrace_clip_rate`
- `vtrace_c_clipped_rate`
- `kl_divergence`
- `entropy`

Optional caller-supplied fields exist for checkpoint-based actor sync lag:

- `checkpoint_lag_updates`
- `checkpoint_lag_percentile_p50`
- `checkpoint_lag_percentile_p90`

## Minimal usage

```python
from pathlib import Path

import numpy as np

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.training_logger import TrainingLogger

learner = ImpalaLearner(
    checkpoint_dir=Path("runs/example/checkpoints"),
    logs_dir=Path("runs/example/logs"),
    logging_interval_updates=10,
)

batch = {
    "logits": np.random.randn(8, 2, 5),
    "behavior_logits": np.random.randn(8, 2, 5),
    "actions": np.random.randint(0, 5, size=(8, 2)),
    "legal_mask": np.ones((8, 2, 5), dtype=bool),
    "rewards": np.random.randn(8, 2),
}

for _ in range(20):
    learner.update(batch)

records = TrainingLogger.read_jsonl(Path("runs/example/logs/training_metrics.jsonl"))
print(records[-1]["vtrace_rho_p90"])
```

## Validation

```python
from pathlib import Path
from weiss_rl.training_logger import TrainingLogger

is_valid, message = TrainingLogger.validate_jsonl(
    Path("runs/example/logs/training_metrics.jsonl")
)
print(is_valid, message)
```

## Notes on semantics

- `policy_version` is the learner-side checkpoint version counter
- checkpoint files are written as `checkpoint_<update_count>.pt`
- actor sync lag should be interpreted in checkpoint-update units, not as full learner-step staleness between arbitrary updates

## Tests

Relevant tests live in:

- `python/weiss_rl/tests/test_training_logger.py`
- `python/weiss_rl/tests/test_actor_worker.py`
