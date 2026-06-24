from __future__ import annotations

import numpy as np
import torch
from weiss_rl.runtime.components.collection.collector_state import allocate_collector_unroll_state


def test_allocate_collector_unroll_state_sets_expected_dtypes_and_shapes() -> None:
    seat_hidden = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6)

    state = allocate_collector_unroll_state(
        time_steps=3,
        batch_size=2,
        observation_dim=5,
        obs_dtype=np.float16,
        seat_hidden=seat_hidden,
        trajectory_retention_enabled=True,
    )

    assert state.obs.shape == (3, 2, 5)
    assert state.obs.dtype == np.float16
    assert state.actions.dtype == np.uint16
    assert state.rewards.dtype == np.float32
    assert state.terminated.dtype == np.bool_
    assert state.truncated.dtype == np.bool_
    assert state.to_play_seat.dtype == np.int8
    assert state.behavior_logp.dtype == np.float32
    assert state.values.dtype == np.float32
    assert state.episode_seed.dtype == np.uint64
    assert state.policy_train_mask.dtype == np.bool_
    assert state.opponent_context_index.dtype == np.int16
    assert state.teacher_family.dtype == np.int32
    assert state.teacher_family.tolist() == [[-1, -1], [-1, -1], [-1, -1]]
    assert state.teacher_valid.dtype == np.bool_
    assert state.trajectory_retention_valid is not None
    assert state.trajectory_retention_valid.shape == (3, 2)
    assert state.packed_offsets[0].dtype == np.uint32
    assert state.packed_offsets[0].tolist() == [0]
    assert state.counters["actor_env_step_ms"] == 0
    assert state.action_sequence_state.consecutive_main_moves_by_env.tolist() == [0, 0]
    np.testing.assert_array_equal(state.initial_hidden_state, seat_hidden.numpy())

    seat_hidden.fill_(99.0)
    assert state.initial_hidden_state[0, 0, 0] == 0.0


def test_allocate_collector_unroll_state_uses_independent_mutable_state() -> None:
    hidden = torch.zeros((1, 2), dtype=torch.float32)

    first = allocate_collector_unroll_state(
        time_steps=1,
        batch_size=1,
        observation_dim=1,
        obs_dtype=np.float32,
        seat_hidden=hidden,
        trajectory_retention_enabled=False,
    )
    second = allocate_collector_unroll_state(
        time_steps=1,
        batch_size=1,
        observation_dim=1,
        obs_dtype=np.float32,
        seat_hidden=hidden,
        trajectory_retention_enabled=False,
    )

    first.packed_ids.append(np.array([7], dtype=np.uint32))
    first.packed_offsets.append(np.array([1], dtype=np.uint32))
    first.counters["actor_env_step_ms"] = 12
    first.action_sequence_state.consecutive_main_moves_by_env[0] = 3

    assert second.packed_ids == []
    assert len(second.packed_offsets) == 1
    assert second.counters["actor_env_step_ms"] == 0
    assert second.action_sequence_state.consecutive_main_moves_by_env.tolist() == [0]
    assert first.trajectory_retention_valid is None
    assert second.trajectory_retention_valid is None
