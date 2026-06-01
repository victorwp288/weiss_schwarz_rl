from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.runtime_components.central_step_inputs import prepare_central_step_inputs
from weiss_rl.runtime_components.collector_state import allocate_collector_unroll_state


class _ContextModel:
    def opponent_context_indices_for_policy_ids(
        self,
        policy_ids: list[object],
        *,
        batch_size: int,
    ) -> np.ndarray:
        assert batch_size == len(policy_ids)
        mapping = {"B2 HeuristicPublic": 3, "policy_000001": 7}
        return np.asarray([mapping.get(str(policy_id), 0) for policy_id in policy_ids], dtype=np.int64)


def _state() -> object:
    return allocate_collector_unroll_state(
        time_steps=3,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float16,
        seat_hidden=torch.zeros((1, 2), dtype=torch.float32),
        trajectory_retention_enabled=False,
    )


def test_prepare_central_step_inputs_filters_snapshots_and_writes_opponent_context() -> None:
    states = {1: _state(), 2: _state()}
    actors = [
        SimpleNamespace(
            actor_id=1,
            model=_ContextModel(),
            opponent_policy_id_by_env=["B2 HeuristicPublic", "unknown"],
        ),
        SimpleNamespace(
            actor_id=2,
            model=_ContextModel(),
            opponent_policy_id_by_env=["policy_000001", "B2 HeuristicPublic"],
        ),
    ]
    original_batches = [
        SimpleNamespace(
            obs=np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16),
            actor=np.asarray([0, 1], dtype=np.int8),
        ),
        SimpleNamespace(
            obs=np.asarray([[5.5, 6.5], [7.5, 8.5]], dtype=np.float16),
            actor=np.asarray([1, 0], dtype=np.int8),
        ),
    ]
    calls: list[tuple[object, object, object]] = []

    def filter_batch(batch: object, *, counters: object, action_sequence_state: object) -> object:
        calls.append((batch, counters, action_sequence_state))
        batch_obj = batch
        return SimpleNamespace(
            obs=np.asarray(batch_obj.obs, dtype=np.float16) + np.float16(10.0),
            actor=np.asarray(batch_obj.actor, dtype=np.int16) + np.int16(2),
        )

    prepared = prepare_central_step_inputs(
        actors=actors,
        batches=original_batches,
        states_by_actor=states,
        step_index=1,
        batch_size=2,
        filter_action_surface_for_batch=filter_batch,
    )

    assert len(calls) == 2
    assert calls[0][0] is original_batches[0]
    assert calls[0][1] is states[1].counters
    assert calls[0][2] is states[1].action_sequence_state
    assert calls[1][0] is original_batches[1]
    assert calls[1][1] is states[2].counters
    assert calls[1][2] is states[2].action_sequence_state
    assert prepared.batches[0] is not original_batches[0]
    assert prepared.batches[1] is not original_batches[1]
    np.testing.assert_array_equal(
        prepared.obs_storage_steps[0],
        np.asarray([[11.5, 12.5], [13.5, 14.5]], dtype=np.float16),
    )
    assert prepared.obs_storage_steps[0].dtype == np.float16
    np.testing.assert_array_equal(
        prepared.obs_steps[1],
        np.asarray([[15.5, 16.5], [17.5, 18.5]], dtype=np.float32),
    )
    assert prepared.obs_steps[1].dtype == np.float32
    np.testing.assert_array_equal(prepared.actor_steps[0], np.asarray([2, 3], dtype=np.int64))
    assert prepared.actor_steps[0].dtype == np.int64
    np.testing.assert_array_equal(states[1].opponent_context_index[1], np.asarray([3, 0], dtype=np.int16))
    np.testing.assert_array_equal(states[2].opponent_context_index[1], np.asarray([7, 3], dtype=np.int16))

    prepared.batches[0].obs[0, 0] = np.float16(99.0)
    prepared.batches[0].actor[0] = np.int16(99)
    assert prepared.obs_storage_steps[0][0, 0] == np.float16(11.5)
    assert prepared.obs_steps[0][0, 0] == np.float32(11.5)
    assert prepared.actor_steps[0][0] == np.int64(2)


def test_prepare_central_step_inputs_rejects_actor_batch_mismatch() -> None:
    states = {1: _state(), 2: _state()}
    actors = [
        SimpleNamespace(actor_id=1, model=object(), opponent_policy_id_by_env=["a", "b"]),
        SimpleNamespace(actor_id=2, model=object(), opponent_policy_id_by_env=["a", "b"]),
    ]
    batches = [
        SimpleNamespace(obs=np.zeros((2, 2), dtype=np.float32), actor=np.zeros((2,), dtype=np.int8)),
    ]

    with pytest.raises(ValueError, match="zip"):
        prepare_central_step_inputs(
            actors=actors,
            batches=batches,
            states_by_actor=states,
            step_index=0,
            batch_size=2,
            filter_action_surface_for_batch=lambda batch, **_: batch,
        )
