from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime

from .runtime_test_support import _make_runtime_unroll


def test_build_learner_batch_preserves_teacher_labels() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 3), dtype=np.bool_)),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        teacher_family=np.array([[1], [2]], dtype=np.int32),
        teacher_slot=np.array([[0], [-1]], dtype=np.int32),
        teacher_move_source=np.array([[-1], [2]], dtype=np.int32),
        teacher_attack_type=np.array([[-1], [1]], dtype=np.int32),
        teacher_action=np.array([[7], [9]], dtype=np.int32),
        teacher_valid=np.array([[True], [False]], dtype=np.bool_),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert np.array_equal(batch["teacher_family"], cast(np.ndarray, unroll.teacher_family))
    assert np.array_equal(batch["teacher_slot"], cast(np.ndarray, unroll.teacher_slot))
    assert np.array_equal(batch["teacher_move_source"], cast(np.ndarray, unroll.teacher_move_source))
    assert np.array_equal(batch["teacher_attack_type"], cast(np.ndarray, unroll.teacher_attack_type))
    assert np.array_equal(batch["teacher_action"], cast(np.ndarray, unroll.teacher_action))
    assert np.array_equal(batch["teacher_valid"], cast(np.ndarray, unroll.teacher_valid))


def test_build_learner_batch_fills_missing_trajectory_retention_labels() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    labeled = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 3), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((2, 1), dtype=np.uint64),
        policy_train_mask=np.ones((2, 1), dtype=np.bool_),
        trajectory_retention_valid=np.array([[True], [False]], dtype=np.bool_),
    )
    unlabeled = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.int64),
        rewards=np.zeros((2, 2), dtype=np.float32),
        terminated=np.zeros((2, 2), dtype=np.bool_),
        truncated=np.zeros((2, 2), dtype=np.bool_),
        to_play_seat=np.zeros((2, 2), dtype=np.int64),
        behavior_logp=np.zeros((2, 2), dtype=np.float32),
        values=np.zeros((2, 2), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 2, 3), dtype=np.bool_)),
        bootstrap_obs=np.zeros((2, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((2,), dtype=np.int64),
        bootstrap_value=np.zeros((2,), dtype=np.float32),
        initial_hidden_state=np.zeros((2, 1), dtype=np.float32),
        final_hidden_state=np.zeros((2, 1), dtype=np.float32),
        episode_seed=np.zeros((2, 2), dtype=np.uint64),
        policy_train_mask=np.ones((2, 2), dtype=np.bool_),
        trajectory_retention_valid=None,
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [labeled, unlabeled],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["trajectory_retention_valid"] is not None
    assert batch["trajectory_retention_valid"].shape == (2, 3)
    assert np.array_equal(batch["trajectory_retention_valid"][:, :1], labeled.trajectory_retention_valid)
    assert np.array_equal(batch["trajectory_retention_valid"][:, 1:], np.zeros((2, 2), dtype=np.bool_))
