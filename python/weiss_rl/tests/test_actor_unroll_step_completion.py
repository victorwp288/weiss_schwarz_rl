from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.runtime.components.actor_unroll_policy_execution import ActorPolicyExecutionResult
from weiss_rl.runtime.components.actor_unroll_step_completion import (
    ActorUnrollStepCompletionCallbacks,
    ActorUnrollStepCompletionInputs,
    complete_generic_actor_unroll_step,
    shaped_generic_actor_step_rewards,
)
from weiss_rl.runtime.components.actor_unroll_step_inputs import ActorUnrollStepInputs
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state
from weiss_rl.runtime.components.counters import collector_counter_template
from weiss_rl.runtime.components.teacher_labels import teacher_label_arrays


class _ContextModel:
    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_policy_ids: object | None = None,
    ) -> torch.Tensor:
        fill = 7.0 if opponent_policy_ids is not None else 3.0
        return torch.full((int(batch_size), 2), fill, dtype=torch.float32, device=device)


def _state(*, retention: bool = True) -> Any:
    return allocate_collector_unroll_state(
        time_steps=2,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float16,
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        trajectory_retention_enabled=retention,
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        pass_action_id=0,
        pass_with_nonpass_penalty=0.25,
        mulligan_select_with_confirm_penalty=0.0,
    )


def _actor() -> SimpleNamespace:
    return SimpleNamespace(
        model=_ContextModel(),
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((2, 2), dtype=torch.float32),
        opponent_policy_id_by_env=np.asarray(["p0", "p1"], dtype=object),
        focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
    )


def _step_inputs() -> ActorUnrollStepInputs:
    return ActorUnrollStepInputs(
        batch=SimpleNamespace(),
        obs_storage_step=np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16),
        obs_step=np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        actor_step=np.asarray([0, 0], dtype=np.int64),
        focal_rows=np.asarray([True, False], dtype=np.bool_),
    )


def _next_batch(*, terminated: list[bool] | None = None) -> SimpleNamespace:
    terminated_array = np.asarray([False, False] if terminated is None else terminated, dtype=np.bool_)
    return SimpleNamespace(
        reward=np.asarray([1.0, 1.0], dtype=np.float32),
        terminated=terminated_array,
        truncated=np.asarray([False, False], dtype=np.bool_),
        episode_seed=np.asarray([101, 202], dtype=np.uint64),
        engine_status=np.asarray([0, 0], dtype=np.int64),
        decision_count=np.asarray([3, 4], dtype=np.int32),
        tick_count=np.asarray([30, 40], dtype=np.int32),
        no_progress_count=np.asarray([0, 0], dtype=np.int32),
    )


def _executed_policy(*, next_batch: Any | None = None) -> ActorPolicyExecutionResult:
    labels = teacher_label_arrays(2)
    labels[4][:] = np.asarray([8, 9], dtype=np.int32)
    labels[5][:] = np.asarray([True, False], dtype=np.bool_)
    return ActorPolicyExecutionResult(
        next_batch=_next_batch() if next_batch is None else next_batch,
        action_step=np.asarray([0, 2], dtype=np.int64),
        logp_step=np.asarray([-0.1, -0.2], dtype=np.float32),
        teacher_labels=labels,
        reward_legal_ids=None,
        reward_legal_offsets=None,
        reward_legal_meta=None,
        reward_legal_mask=np.asarray([[True, True, False], [False, False, True]], dtype=np.bool_),
    )


def test_shaped_generic_actor_step_rewards_preserves_mask_and_packed_reward_surfaces() -> None:
    counters = collector_counter_template()
    rewards = shaped_generic_actor_step_rewards(
        executed_policy=_executed_policy(),
        counters=counters,
        config=_config(),
        action_family_index={},
    )

    assert rewards.tolist() == [0.75, 1.0]
    assert counters["pass_with_nonpass_penalty_count"] == 1

    packed = _executed_policy()._replace(
        reward_legal_ids=np.asarray([0, 1, 2], dtype=np.int64),
        reward_legal_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        reward_legal_mask=None,
    )
    packed_counters = collector_counter_template()
    packed_rewards = shaped_generic_actor_step_rewards(
        executed_policy=packed,
        counters=packed_counters,
        config=_config(),
        action_family_index={},
    )

    assert packed_rewards.tolist() == [0.75, 1.0]
    assert packed_counters["pass_with_nonpass_penalty_count"] == 1


def test_complete_generic_actor_unroll_step_shapes_rewards_and_stores_nonterminal_step() -> None:
    state = _state()
    retention_calls: list[np.ndarray] = []
    returned = complete_generic_actor_unroll_step(
        inputs=ActorUnrollStepCompletionInputs(
            actor=_actor(),
            state=state,
            step_index=1,
            step_inputs=_step_inputs(),
            executed_policy=_executed_policy(),
            value_step=np.asarray([0.25, 0.75], dtype=np.float32),
            config=_config(),
            action_family_index={},
            timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
            device=torch.device("cpu"),
        ),
        callbacks=ActorUnrollStepCompletionCallbacks(
            trajectory_retention_mask_for_actor=lambda *, actor, focal_rows: (
                retention_calls.append(focal_rows.copy()) or np.asarray([False, True], dtype=np.bool_)
            ),
            update_outcomes=lambda **_: None,
            assign_episode_roles=lambda *_, **__: None,
            reset_done_rows=lambda *_: None,
        ),
    )

    assert returned.episode_seed.tolist() == [101, 202]
    assert retention_calls[0].tolist() == [True, False]
    assert state.actions[1].tolist() == [0, 2]
    assert state.rewards[1].tolist() == [0.75, 1.0]
    assert state.behavior_logp[1].tolist() == [-0.10000000149011612, -0.20000000298023224]
    assert state.values[1].tolist() == [0.25, 0.75]
    assert state.teacher_action[1].tolist() == [8, 9]
    assert state.teacher_valid[1].tolist() == [True, False]
    assert state.trajectory_retention_valid is not None
    assert state.trajectory_retention_valid[1].tolist() == [False, True]
    assert state.counters["trajectory_retention_rows"] == 1
    assert state.counters["pass_with_nonpass_penalty_count"] == 1


def test_complete_generic_actor_unroll_step_resets_terminal_rows_after_storage() -> None:
    state = _state(retention=False)
    actor = _actor()
    terminal_batch = _next_batch(terminated=[True, False])
    reset_batch = _next_batch()
    outcome_calls: list[dict[str, Any]] = []
    assigned_done: list[np.ndarray] = []
    reset_done: list[np.ndarray] = []

    returned = complete_generic_actor_unroll_step(
        inputs=ActorUnrollStepCompletionInputs(
            actor=actor,
            state=state,
            step_index=0,
            step_inputs=_step_inputs(),
            executed_policy=_executed_policy(next_batch=terminal_batch),
            value_step=np.asarray([0.0, 0.0], dtype=np.float32),
            config=_config(),
            action_family_index={},
            timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
            device=torch.device("cpu"),
        ),
        callbacks=ActorUnrollStepCompletionCallbacks(
            trajectory_retention_mask_for_actor=lambda *, actor, focal_rows: None,
            update_outcomes=lambda **kwargs: outcome_calls.append(kwargs),
            assign_episode_roles=lambda actor, done, **_: assigned_done.append(done.copy()),
            reset_done_rows=lambda actor, done: reset_done.append(done.copy()) or reset_batch,
        ),
    )

    assert returned is reset_batch
    assert state.terminated[0].tolist() == [True, False]
    assert state.actions[0].tolist() == [0, 2]
    assert outcome_calls[0]["terminal_batch"] is terminal_batch
    assert outcome_calls[0]["done"].tolist() == [True, False]
    assert assigned_done[0].tolist() == [True, False]
    assert reset_done[0].tolist() == [True, False]
    assert state.counters["actor_done_reset_ms"] >= 0
    np.testing.assert_allclose(actor.seat_hidden[0].numpy(), np.asarray([7.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(actor.opponent_hidden[0].numpy(), np.asarray([3.0, 3.0], dtype=np.float32))
