from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from weiss_rl.runtime.components.startup_logging import log_runtime_startup


class _PerformanceLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, record: dict[str, object]) -> None:
        self.records.append(record)


def test_log_runtime_startup_records_collection_and_batch_shape() -> None:
    logger = _PerformanceLogger()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            unroll_length=4,
            envs_per_actor=8,
            batch_unrolls_per_update=3,
            queue_capacity_unrolls=9,
            actor_count=2,
            total_envs=16,
            actor_sampling_temperature=0.75,
        ),
        _performance_logger=logger,
        _device=torch.device("cpu"),
        _process_actor_device_names=("cpu", "cpu"),
        _use_process_collectors=False,
        _compile_actor_inference=True,
        _fixed_opponent_backend="python_batched",
        _actor_policy_backend="model",
        _actor_heuristic_fraction=0.25,
        _collection_backend="central",
        _league_enabled=True,
        _structured_fixed_opponents_expected=True,
        _use_central_batched_collection=True,
    )

    log_runtime_startup(
        runtime,
        model_kind="structured_v2",
        training_config=SimpleNamespace(structured_warmstart_enabled=True),
        structured_warmstart_cfg=SimpleNamespace(enabled=False),
    )

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record["kind"] == "runtime_startup_v1"
    assert record["actor_device_layout"] == ["cpu"]
    assert record["runtime_rows_per_actor_unroll"] == 32
    assert record["runtime_batch_env_steps"] == 96
    assert record["collection_backend"] == "central"
    assert record["model_kind"] == "structured_v2"
    assert record["structured_warmstart_enabled"] is True
    assert record["structured_warmstart_flag_enabled"] is False
    assert record["use_central_batched_collection"] is True
    assert record["use_process_collectors"] is False
    assert record["actor_sampling_temperature"] == pytest.approx(0.75)
