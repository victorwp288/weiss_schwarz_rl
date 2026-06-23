from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_central_outputs_test_support import (
    AdvanceOnlyModel,
    RecordingHeuristicPolicy,
    central_actor,
    central_opponent_runtime,
)


def test_batched_heuristic_public_overwrite_batches_rows_and_legal_metadata_across_actors() -> None:
    heuristic_policy = RecordingHeuristicPolicy()
    runtime = central_opponent_runtime(
        heuristic_policies={HEURISTIC_PUBLIC_POLICY_ID: heuristic_policy},
        action_dim=32,
    )
    shared_model = AdvanceOnlyModel()
    actor_a = central_actor(
        focal_seats=[0, 0],
        opponent_policy_ids=[HEURISTIC_PUBLIC_POLICY_ID, MIRROR_OPPONENT_POLICY_ID],
        hidden_width=3,
        model=shared_model,
        compiled_model=None,
        seat_hidden=torch.zeros((2, 3)),
    )
    actor_b = central_actor(
        focal_seats=[1, 1],
        opponent_policy_ids=[MIRROR_OPPONENT_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
        hidden_width=3,
        model=shared_model,
        compiled_model=None,
        seat_hidden=torch.zeros((2, 3)),
    )
    batch_a = _legal_batch(
        legal_ids=[10, 11, 12],
        legal_offsets=[0, 2, 3],
        legal_action_meta=[
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 2, 2, 0],
        ],
    )
    batch_b = _legal_batch(
        legal_ids=[20, 21, 22],
        legal_offsets=[0, 1, 3],
        legal_action_meta=[
            [3, 0, 0, 0],
            [4, 1, 0, 0],
            [5, 2, 0, 0],
        ],
    )
    obs_a = np.asarray([[1, 0, 0], [9, 9, 9]], dtype=np.float32)
    obs_b = np.asarray([[8, 8, 8], [2, 0, 0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)
    logits_a = np.full((2, 32), -5.0, dtype=np.float32)
    logits_b = np.full((2, 32), -5.0, dtype=np.float32)
    values_a = np.ones((2,), dtype=np.float32)
    values_b = np.ones((2,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[batch_a, batch_b],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 21, 22], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 2, 4], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(
        call_meta,
        np.asarray(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [4, 1, 0, 0],
                [5, 2, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert actor_a.seat_hidden[0].tolist() == [1.0, 1.0, 1.0]
    assert actor_b.seat_hidden[1].tolist() == [1.0, 1.0, 1.0]
    assert values_a.tolist() == [0.0, 1.0]
    assert values_b.tolist() == [1.0, 0.0]
    assert logits_a[0, 10] == pytest.approx(0.0)
    assert logits_a[0, 11] < 0.0
    assert logits_b[1, 21] == pytest.approx(0.0)
    assert logits_b[1, 22] < 0.0


def _legal_batch(
    *,
    legal_ids: list[int],
    legal_offsets: list[int],
    legal_action_meta: list[list[int]],
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            ids_offsets=(
                np.asarray(legal_ids, dtype=np.uint32),
                np.asarray(legal_offsets, dtype=np.uint32),
            ),
            legal_action_meta=np.asarray(legal_action_meta, dtype=np.uint16),
            mask=None,
        ),
    )
