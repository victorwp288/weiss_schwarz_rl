from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from weiss_rl.diagnostics.action_diagnostics import make_action_sequence_state
from weiss_rl.runtime.components.actor_unroll_step_inputs import prepare_actor_unroll_step_inputs


class _ContextModel:
    def opponent_context_indices_for_policy_ids(self, policy_ids: list[object], *, batch_size: int) -> np.ndarray:
        assert batch_size == len(policy_ids)
        return np.asarray([3 + index for index, _policy_id in enumerate(policy_ids)], dtype=np.int64)


def _actor() -> SimpleNamespace:
    return SimpleNamespace(
        model=_ContextModel(),
        opponent_policy_id_by_env=np.asarray(["a", "b", "c"], dtype=object),
        focal_seat_by_env=np.asarray([0, 1, 0], dtype=np.int64),
    )


def _batch(*, obs: np.ndarray | None = None, actor: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((3, 4), dtype=np.float16) if obs is None else obs,
        actor=np.asarray([0, 0, 1], dtype=np.int8) if actor is None else actor,
    )


def test_prepare_actor_unroll_step_inputs_filters_copies_and_records_context_and_role_counts() -> None:
    counters: dict[str, int] = {}
    action_sequence_state = make_action_sequence_state(3)
    opponent_context_index = np.zeros((2, 3), dtype=np.int16)
    input_batch = _batch(obs=np.full((3, 4), 1.5, dtype=np.float16))
    filtered_batch = _batch(obs=np.full((3, 4), 2.5, dtype=np.float16))
    filter_calls: list[dict[str, Any]] = []

    def filter_batch(batch: Any, **kwargs: Any) -> Any:
        filter_calls.append({"batch": batch, **kwargs})
        kwargs["counters"]["filtered"] = 1
        return filtered_batch

    prepared = prepare_actor_unroll_step_inputs(
        actor=_actor(),
        batch=input_batch,
        step_index=1,
        batch_size=3,
        observation_dim=4,
        opponent_context_index=opponent_context_index,
        counters=counters,
        action_sequence_state=action_sequence_state,
        filter_action_surface_for_batch=filter_batch,
    )

    assert prepared.batch is filtered_batch
    assert filter_calls == [
        {
            "batch": input_batch,
            "counters": counters,
            "action_sequence_state": action_sequence_state,
        }
    ]
    assert prepared.obs_storage_step.dtype == np.float16
    assert prepared.obs_step.dtype == np.float32
    assert prepared.actor_step.dtype == np.int64
    assert prepared.focal_rows.tolist() == [True, False, False]
    assert counters["filtered"] == 1
    assert counters["focal_row_count"] == 1
    assert counters["opponent_row_count"] == 2
    assert opponent_context_index[1].tolist() == [3, 4, 5]

    filtered_batch.obs[0, 0] = 99
    filtered_batch.actor[0] = 1
    assert prepared.obs_storage_step[0, 0] == np.float16(2.5)
    assert prepared.obs_step[0, 0] == np.float32(2.5)
    assert prepared.actor_step.tolist() == [0, 0, 1]


def test_prepare_actor_unroll_step_inputs_rejects_unexpected_obs_shape() -> None:
    with pytest.raises(RuntimeError, match=r"unexpected actor obs shape: \(2, 4\)"):
        prepare_actor_unroll_step_inputs(
            actor=_actor(),
            batch=_batch(obs=np.zeros((2, 4), dtype=np.float32), actor=np.zeros((2,), dtype=np.int64)),
            step_index=0,
            batch_size=3,
            observation_dim=4,
            opponent_context_index=np.zeros((1, 3), dtype=np.int16),
            counters={},
            action_sequence_state=make_action_sequence_state(3),
            filter_action_surface_for_batch=lambda batch, **_: batch,
        )


def test_prepare_actor_unroll_step_inputs_rejects_non_live_actor_rows() -> None:
    with pytest.raises(RuntimeError, match=r"actor runtime only supports live seat rows, got \[0, 2, 1\]"):
        prepare_actor_unroll_step_inputs(
            actor=_actor(),
            batch=_batch(actor=np.asarray([0, 2, 1], dtype=np.int64)),
            step_index=0,
            batch_size=3,
            observation_dim=4,
            opponent_context_index=np.zeros((1, 3), dtype=np.int16),
            counters={},
            action_sequence_state=make_action_sequence_state(3),
            filter_action_surface_for_batch=lambda batch, **_: batch,
        )
