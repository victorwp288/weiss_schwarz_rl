from __future__ import annotations

from typing import Any

import numpy as np
from weiss_rl.runtime.components.central_actor_step import execute_central_actor_step

from .central_actor_step_test_support import (
    StepEnv,
    central_step_actor,
    central_step_inputs,
    central_step_next_batch,
    central_step_packed_batch,
    central_step_runtime_context,
    central_step_state,
)


def test_execute_central_actor_step_returns_reset_batch_after_terminal_rows() -> None:
    state = central_step_state(trajectory_retention_enabled=False)
    terminal_batch = central_step_next_batch(terminated=[True, False])
    reset_batch = central_step_next_batch()
    env = StepEnv(terminal_batch)
    actor = central_step_actor(env)
    outcome_calls: list[dict[str, Any]] = []
    assigned_done: list[np.ndarray] = []
    reset_done: list[np.ndarray] = []

    returned = execute_central_actor_step(
        actor=actor,
        batch=central_step_packed_batch(),
        state=state,
        inputs=central_step_inputs(
            step_index=0,
            structured_action_steps=[np.asarray([1, 2], dtype=np.int64)],
            structured_logp_steps=[np.asarray([-0.3, -0.4], dtype=np.float32)],
        ),
        runtime=central_step_runtime_context(
            update_outcomes=lambda **kwargs: outcome_calls.append(kwargs),
            assign_episode_roles=lambda actor, done, **_: assigned_done.append(done.copy()),
            reset_done_rows=lambda actor, done: reset_done.append(done.copy()) or reset_batch,
        ),
    )

    assert returned is reset_batch
    assert outcome_calls[0]["terminal_batch"] is terminal_batch
    assert outcome_calls[0]["done"].tolist() == [True, False]
    assert assigned_done[0].tolist() == [True, False]
    assert reset_done[0].tolist() == [True, False]
    assert state.terminated[0].tolist() == [True, False]
    assert state.actions[0].tolist() == [1, 2]
    assert state.counters["actor_done_reset_ms"] >= 0
    np.testing.assert_allclose(actor.seat_hidden[0].numpy(), np.asarray([7.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(actor.opponent_hidden[0].numpy(), np.asarray([3.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(actor.seat_hidden[1].numpy(), np.asarray([0.0, 0.0], dtype=np.float32))
