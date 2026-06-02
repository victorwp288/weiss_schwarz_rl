from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.batching.bootstrap_values import runtime_bootstrap_fields
from weiss_rl.runtime.components.batching.learner_batch_inputs import (
    prepare_impala_learner_batch_inputs,
    prepare_ppo_learner_batch_inputs,
    prepare_shared_learner_batch_inputs,
)


def _unroll(
    *,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray | None = None,
    policy_train_mask: np.ndarray | None = None,
    values: np.ndarray | None = None,
    to_play_seat: np.ndarray | None = None,
    bootstrap_actor: np.ndarray | None = None,
    bootstrap_value: np.ndarray | None = None,
) -> SimpleNamespace:
    reward_array = np.asarray(rewards, dtype=np.float32)
    time_steps, batch_size = reward_array.shape
    return SimpleNamespace(
        obs=np.zeros((time_steps, batch_size, 2), dtype=np.float32),
        actions=np.zeros((time_steps, batch_size), dtype=np.int64),
        rewards=reward_array,
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=(
            np.zeros((time_steps, batch_size), dtype=np.bool_)
            if truncated is None
            else np.asarray(truncated, dtype=np.bool_)
        ),
        to_play_seat=(
            np.zeros((time_steps, batch_size), dtype=np.int64)
            if to_play_seat is None
            else np.asarray(to_play_seat, dtype=np.int64)
        ),
        behavior_logp=np.zeros((time_steps, batch_size), dtype=np.float32),
        values=(
            np.zeros((time_steps, batch_size), dtype=np.float32)
            if values is None
            else np.asarray(values, dtype=np.float32)
        ),
        initial_hidden_state=np.zeros((batch_size, 3), dtype=np.float32),
        final_hidden_state=np.ones((batch_size, 3), dtype=np.float32),
        policy_train_mask=(
            np.ones((time_steps, batch_size), dtype=np.bool_)
            if policy_train_mask is None
            else np.asarray(policy_train_mask, dtype=np.bool_)
        ),
        legal_actions=LegalActionBatch.from_mask(np.ones((time_steps, batch_size, 2), dtype=np.bool_)),
        bootstrap_obs=np.ones((batch_size, 2), dtype=np.float32),
        bootstrap_actor=(
            np.zeros((batch_size,), dtype=np.int64)
            if bootstrap_actor is None
            else np.asarray(bootstrap_actor, dtype=np.int64)
        ),
        bootstrap_value=(
            np.zeros((batch_size,), dtype=np.float32)
            if bootstrap_value is None
            else np.asarray(bootstrap_value, dtype=np.float32)
        ),
        opponent_context_index=None,
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )


def test_prepare_impala_learner_batch_inputs_applies_backfills_and_discount_contract() -> None:
    timers: list[str] = []
    unroll = _unroll(
        rewards=np.asarray([[0.0], [-1.0], [0.0]], dtype=np.float32),
        terminated=np.asarray([[False], [True], [False]], dtype=np.bool_),
        policy_train_mask=np.asarray([[True], [False], [True]], dtype=np.bool_),
        bootstrap_value=np.asarray([0.75], dtype=np.float32),
    )

    prepared = prepare_impala_learner_batch_inputs(
        [unroll],
        action_dim=2,
        gamma=0.9,
        terminal_outcome_backfill_reward=0.5,
        terminal_outcome_trace_backfill_reward=0.25,
        record_batch_timer_ms=lambda name, _elapsed: timers.append(name),
    )

    assert timers == ["legal_concatenation"]
    assert prepared.fields.obs.shape == (3, 1, 2)
    assert prepared.bootstrap.value.tolist() == pytest.approx([0.75])
    assert prepared.bootstrap.actor.dtype == np.int64
    assert prepared.bootstrap.final_hidden_state.tolist() == [[1.0, 1.0, 1.0]]
    assert prepared.rewards[:, 0].tolist() == pytest.approx([0.75, -1.0, 0.0])
    assert prepared.discounts[:, 0].tolist() == pytest.approx([0.9, 0.0, 0.9])
    assert prepared.reset_before_step[:, 0].tolist() == [False, False, True]
    assert prepared.backfill_metrics.outcome_count == 1
    assert prepared.backfill_metrics.outcome_total_micros == 500_000
    assert prepared.backfill_metrics.trace_count == 1
    assert prepared.backfill_metrics.trace_total_micros == 250_000


def test_prepare_shared_learner_batch_inputs_owns_done_discount_and_reset_semantics() -> None:
    unroll = _unroll(
        rewards=np.asarray([[0.0], [0.0], [0.0]], dtype=np.float32),
        terminated=np.asarray([[False], [True], [False]], dtype=np.bool_),
        truncated=np.asarray([[False], [False], [True]], dtype=np.bool_),
        to_play_seat=np.asarray([[0], [1], [0]], dtype=np.int64),
        bootstrap_actor=np.asarray([1], dtype=np.int64),
    )
    bootstrap = runtime_bootstrap_fields([unroll])

    prepared = prepare_shared_learner_batch_inputs(
        [unroll],
        action_dim=2,
        gamma=0.5,
        bootstrap=bootstrap,
    )

    assert prepared.bootstrap is bootstrap
    assert prepared.done[:, 0].tolist() == [False, True, True]
    assert prepared.discounts[:, 0].tolist() == pytest.approx([-0.5, 0.0, 0.0])
    assert prepared.reset_before_step[:, 0].tolist() == [False, False, True]


def test_prepare_ppo_learner_batch_inputs_computes_gae_returns_from_shared_runtime_fields() -> None:
    unroll = _unroll(
        rewards=np.asarray([[1.0], [0.5]], dtype=np.float32),
        terminated=np.asarray([[False], [False]], dtype=np.bool_),
        values=np.asarray([[0.2], [0.3]], dtype=np.float32),
        bootstrap_value=np.asarray([0.4], dtype=np.float32),
    )

    prepared = prepare_ppo_learner_batch_inputs(
        [unroll],
        action_dim=2,
        gamma=1.0,
        gae_lambda=0.95,
    )

    expected_last = 0.5 + 0.4 - 0.3
    expected_first = (1.0 + 0.3 - 0.2) + (0.95 * expected_last)
    assert prepared.bootstrap.value.tolist() == pytest.approx([0.4])
    assert prepared.discounts[:, 0].tolist() == pytest.approx([1.0, 1.0])
    assert prepared.reset_before_step[:, 0].tolist() == [False, False]
    assert prepared.advantages[:, 0].tolist() == pytest.approx([expected_first, expected_last])
    assert prepared.returns[:, 0].tolist() == pytest.approx([expected_first + 0.2, expected_last + 0.3])
