from __future__ import annotations

from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_central_outputs_test_support import (
    SNAPSHOT_POLICY_ID,
    SequentialSeatAwareOpponentModel,
    central_actor,
    central_opponent_runtime,
)


def test_batched_snapshot_opponent_overwrite_groups_rows_across_actors() -> None:
    model = SequentialSeatAwareOpponentModel(hidden_increment=2.0)
    runtime = central_opponent_runtime(snapshot_model=model)
    actor_a = central_actor(
        focal_seats=[0, 0],
        opponent_policy_ids=[SNAPSHOT_POLICY_ID, MIRROR_OPPONENT_POLICY_ID],
        hidden_width=3,
    )
    actor_b = central_actor(
        focal_seats=[1, 1],
        opponent_policy_ids=[MIRROR_OPPONENT_POLICY_ID, SNAPSHOT_POLICY_ID],
        hidden_width=3,
    )
    logits_a = np.zeros((2, 5), dtype=np.float32)
    logits_b = np.zeros((2, 5), dtype=np.float32)
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((2,), dtype=np.float32)
    obs_a = np.asarray([[1.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    obs_b = np.asarray([[8.0, 8.0], [2.0, 0.0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[cast(Any, object()), cast(Any, object())],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1, 0], dtype=np.int64))
    assert np.all(logits_a[0] == np.asarray([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.all(logits_b[1] == np.asarray([5, 6, 7, 8, 9], dtype=np.float32))
    assert values_a.tolist() == [10.0, 0.0]
    assert values_b.tolist() == [0.0, 11.0]
    assert actor_a.opponent_hidden[0].tolist() == [2.0, 2.0, 2.0]
    assert actor_b.opponent_hidden[1].tolist() == [2.0, 2.0, 2.0]
