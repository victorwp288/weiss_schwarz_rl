from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime

from .runtime_test_support import _make_runtime_unroll


def test_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
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
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])
    assert batch["reset_before_step"][:, 0].tolist() == [False, False]


def test_build_learner_batch_signs_discount_across_actor_perspectives() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        bootstrap_actor=np.asarray([0], dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
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

    assert batch["discounts"][:, 0].tolist() == pytest.approx([-0.99, -0.99])


def test_build_learner_batch_zeros_timeout_discount_when_bootstrap_state_is_post_reset() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((3, 1, 1), dtype=np.float32),
        actions=np.zeros((3, 1), dtype=np.int64),
        rewards=np.zeros((3, 1), dtype=np.float32),
        terminated=np.zeros((3, 1), dtype=np.bool_),
        truncated=np.array([[False], [True], [False]], dtype=np.bool_),
        to_play_seat=np.zeros((3, 1), dtype=np.int64),
        behavior_logp=np.zeros((3, 1), dtype=np.float32),
        values=np.zeros((3, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((3, 1, 2), dtype=np.bool_)),
        bootstrap_value=np.asarray([99.0], dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
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

    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0, 0.99])
    assert batch["reset_before_step"][:, 0].tolist() == [False, False, True]
