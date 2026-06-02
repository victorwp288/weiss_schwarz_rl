from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import RuntimeBatch as RuntimeBatchFromRuntime
from weiss_rl.runtime import RuntimeUnroll as RuntimeUnrollFromRuntime
from weiss_rl.runtime.components.types import RuntimeBatch, RuntimeUnroll


def _runtime_unroll() -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=1,
        unroll_seq=2,
        behavior_policy_version=3,
        unroll_hash="1:2:3",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
    )


def test_runtime_unroll_container_preserves_defaults_frozen_slots_and_runtime_import() -> None:
    unroll = _runtime_unroll()

    assert RuntimeUnrollFromRuntime is RuntimeUnroll
    assert unroll.teacher_family is None
    assert unroll.behavior_logits is None
    assert unroll.counters is None
    assert getattr(unroll, "__dict__", None) is None
    with pytest.raises(FrozenInstanceError):
        unroll.actor_id = 10  # type: ignore[misc]


def test_runtime_batch_container_preserves_frozen_slots_and_runtime_import() -> None:
    batch = RuntimeBatch(learner_batch={"obs": object()}, runtime_metrics={"fps": 12.5})

    assert RuntimeBatchFromRuntime is RuntimeBatch
    assert batch.learner_batch.keys() == {"obs"}
    assert batch.runtime_metrics == {"fps": 12.5}
    assert getattr(batch, "__dict__", None) is None
    with pytest.raises(FrozenInstanceError):
        batch.runtime_metrics = cast_metrics({})  # type: ignore[misc]


def cast_metrics(value: dict[str, float]) -> Any:
    return value
