from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.diagnostics.logging.training_logger import TrainingLogger
from weiss_rl.learners.impala import ImpalaLearner


def _masked_batch(*, time_steps: int = 4, batch_size: int = 2, action_space: int = 5) -> dict[str, np.ndarray]:
    return {
        "logits": np.random.randn(time_steps, batch_size, action_space),
        "behavior_logits": np.random.randn(time_steps, batch_size, action_space),
        "actions": np.random.randint(0, action_space, size=(time_steps, batch_size)),
        "legal_mask": np.ones((time_steps, batch_size, action_space), dtype=bool),
        "rewards": np.random.randn(time_steps, batch_size),
    }


def test_impala_learner_logs_masked_metrics_and_uses_update_count_checkpoint_metadata(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
        logs_dir=tmp_path / "logs",
        logging_interval_updates=1,
    )

    for _ in range(4):
        result = learner.update(_masked_batch())
        assert set(result) == {
            "entropy_coef",
            "loss",
            "throughput_samples_per_sec",
            "throughput_updates_per_sec",
        }

    records = TrainingLogger.read_jsonl(tmp_path / "logs" / "training_metrics.jsonl")
    assert [record["update_count"] for record in records] == [1, 2, 3, 4]
    assert all(record["entropy"] > 0.0 for record in records)
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_2.json").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_logs_vtrace_metrics_for_object_batches(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        logs_dir=tmp_path / "logs",
        logging_interval_updates=1,
    )

    result = learner.update(SimpleNamespace(**_masked_batch(time_steps=1, batch_size=1, action_space=3)))

    assert set(result) == {
        "entropy_coef",
        "loss",
        "throughput_samples_per_sec",
        "throughput_updates_per_sec",
    }

    [record] = TrainingLogger.read_jsonl(tmp_path / "logs" / "training_metrics.jsonl")
    assert math.isfinite(record["vtrace_rho_mean"])
    assert math.isfinite(record["kl_divergence"])
    assert math.isfinite(record["entropy"])
    assert record["custom_metrics"]["vtrace_batch_metrics_available"] == 1.0
    assert record["custom_metrics"]["vtrace_entropy"] == pytest.approx(record["entropy"])
