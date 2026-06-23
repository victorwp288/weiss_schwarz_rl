from __future__ import annotations

from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

from .runtime_opponent_central_outputs_test_support import (
    SNAPSHOT_POLICY_ID,
    ConstantSeatAwareOpponentModel,
    central_actor,
    central_opponent_runtime,
)


def test_single_actor_overwrite_only_touches_non_mirror_opponent_rows() -> None:
    model = ConstantSeatAwareOpponentModel(logit_value=-7.0, value=42.0, hidden_increment=1.0)
    runtime = central_opponent_runtime(snapshot_model=model)
    actor = central_actor(
        focal_seats=[0, 0, 0],
        opponent_policy_ids=[MIRROR_OPPONENT_POLICY_ID, SNAPSHOT_POLICY_ID, MIRROR_OPPONENT_POLICY_ID],
        hidden_width=4,
        layout_name="i16_legal_ids",
        rng=np.random.default_rng(7),
    )
    batch = cast(
        Any,
        object(),
    )
    obs_step = np.zeros((3, 8), dtype=np.float32)
    actor_step = np.asarray([1, 1, 1], dtype=np.int64)
    logits = np.zeros((3, 5), dtype=np.float32)
    values = np.zeros((3,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_opponents(
        runtime,
        actor=actor,
        batch=batch,
        obs_step=obs_step,
        actor_step=actor_step,
        logits_out=logits,
        values_out=values,
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.zeros((1, 8), dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1], dtype=np.int64))
    assert values.tolist() == [0.0, 42.0, 0.0]
    assert np.all(logits[0] == 0.0)
    assert np.all(logits[1] == -7.0)
    assert np.all(logits[2] == 0.0)
    assert actor.opponent_hidden[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert actor.opponent_hidden[0].tolist() == [0.0, 0.0, 0.0, 0.0]
