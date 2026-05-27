# Training logs

Learner-side JSONL logging is implemented as a standalone component and is now also exercised by the minimal inline `train.py` smoke path.

`make train-min` remains a manifest/provenance smoke run. The inline training path only activates when `train.py` receives a full training stack and the active interpreter can import a simulator runtime with stepping APIs.

For the current run-root layout and the canonical separation between training metrics, deterministic evaluation outputs, and artifact-level readiness files, see [Artifact contract](artifact_contract.md).

## What works today

- `TrainingLogger` writes append-only JSONL records to `runs/<run>/logs/training_metrics.jsonl`
- the minimal inline `train.py` path also writes `runs/<run>/training/logs/scalars.jsonl` for the master-plan style scalar stream
- `ImpalaLearner` can emit throughput and masked V-trace health metrics when `logs_dir` is configured
- `compute_vtrace_metrics()` respects the masking contract when the batch includes:
  - `logits`
  - `behavior_logits`
  - `actions`
  - either `legal_mask` or `legal_ids` + `legal_offsets`

If the legality surface is missing, V-trace diagnostics intentionally fall back to zero instead of computing from illegal actions.

## What is intentionally not claimed

- no claim that this logging page fully documents the multi-actor `train.py` pipeline
- no claim that every actor-to-learner lag metric is emitted in every runtime mode
- no claim that the inline smoke path equals full end-to-end thesis training readiness

## Minimal usage

Run the standalone example from the repo root:

```bash
uv run python examples/training_logs_example.py
```

If you are bypassing installation for a quick local check:

```bash
PYTHONPATH=python python examples/training_logs_example.py
```

Code sketch:

```python
from pathlib import Path

import numpy as np

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.diagnostics.training_logger import TrainingLogger

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

## Validation

```python
from pathlib import Path
from weiss_rl.diagnostics.training_logger import TrainingLogger

is_valid, message = TrainingLogger.validate_jsonl(
    Path("runs/example/logs/training_metrics.jsonl")
)
print(is_valid, message)
```

## Notes on semantics

- `policy_version` is the learner-side checkpoint version counter
- checkpoint files are written as `checkpoint_<update_count>.pt`
- actor sync lag should be interpreted in checkpoint-update units, not as full learner-step staleness between arbitrary updates
- training metrics and run-root provenance are separate concerns; the latter now belongs in `manifest.json`, `environment.json`, and `run_summary.json`

## Tests

Relevant tests live in:

- `python/weiss_rl/tests/test_training_logger.py`
- `python/weiss_rl/tests/test_actor_worker.py`
