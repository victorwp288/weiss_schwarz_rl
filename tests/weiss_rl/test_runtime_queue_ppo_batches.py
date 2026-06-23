from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.batching import build_ppo_learner_batch

from .runtime_test_support import _make_runtime_unroll


def test_build_ppo_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    runtime_any._bootstrap_values = lambda unroll: np.zeros((1,), dtype=np.float32)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_ppo_batch(
        runtime,
        [unroll],
        gamma=0.99,
        gae_lambda=0.95,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])


def test_build_ppo_batch_uses_stored_behavior_bootstrap_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    runtime_any._bootstrap_values = lambda unroll: np.array([9.0], dtype=np.float32)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=3),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 2), dtype=np.bool_)),
        bootstrap_value=np.array([0.25], dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_ppo_batch(
        runtime,
        [unroll],
        gamma=1.0,
        gae_lambda=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
    )

    assert batch["advantages"][:, 0].tolist() == pytest.approx([0.25])
    assert batch["returns"][:, 0].tolist() == pytest.approx([0.25])


def test_build_ppo_batch_preserves_shared_auxiliary_labels() -> None:
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
        teacher_action=np.asarray([[4], [5]], dtype=np.int32),
        teacher_valid=np.asarray([[True], [False]], dtype=np.bool_),
        trajectory_retention_valid=np.asarray([[False], [True]], dtype=np.bool_),
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
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )

    batch = build_ppo_learner_batch(
        [labeled, unlabeled],
        action_dim=3,
        gamma=0.99,
        gae_lambda=0.95,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
    )

    assert batch["teacher_action"] is not None
    assert batch["teacher_action"].tolist() == [[4, -1, -1], [5, -1, -1]]
    assert batch["teacher_valid"] is not None
    assert batch["teacher_valid"].tolist() == [[True, False, False], [False, False, False]]
    assert batch["trajectory_retention_valid"] is not None
    assert batch["trajectory_retention_valid"].tolist() == [[False, False, False], [True, False, False]]
