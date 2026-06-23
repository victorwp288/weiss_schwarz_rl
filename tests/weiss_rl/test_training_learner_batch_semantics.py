from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from weiss_rl.config import load_stack_config
from weiss_rl.training.batches import build_learner_batch

from .training_batches_test_support import minimal_rollout, repo_root


def test_build_learner_batch_preserves_truncation_reward_and_discount_semantics() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = minimal_rollout()

    batch = build_learner_batch(
        stack,
        rollout,
        np.zeros((1,), dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert stack.config.rewards is not None
    assert batch["discounts"][:, 0].tolist() == pytest.approx([float(stack.config.rewards.gamma), 0.0])
    assert batch["reset_before_step"][:, 0].tolist() == [False, False]
    assert batch["actor"] is batch["to_play_seat"]
    assert batch["behavior_logits"] is batch["logits"]


def test_build_learner_batch_signs_discount_when_next_value_is_opponent_perspective() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = replace(
        minimal_rollout(),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        bootstrap_actor=np.asarray([0], dtype=np.int64),
    )

    batch = build_learner_batch(
        stack,
        rollout,
        np.zeros((1,), dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert stack.config.rewards is not None
    gamma = float(stack.config.rewards.gamma)
    assert batch["discounts"][:, 0].tolist() == pytest.approx([-gamma, -gamma])


def test_build_learner_batch_marks_reset_after_done_rows() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = replace(
        minimal_rollout(),
        obs=np.zeros((3, 1, 1), dtype=np.float32),
        legal_mask=np.ones((3, 1, 2), dtype=np.bool_),
        actions=np.zeros((3, 1), dtype=np.int64),
        rewards=np.zeros((3, 1), dtype=np.float32),
        terminated=np.asarray([[False], [True], [False]], dtype=np.bool_),
        truncated=np.zeros((3, 1), dtype=np.bool_),
        to_play_seat=np.zeros((3, 1), dtype=np.int64),
        behavior_logp=np.zeros((3, 1), dtype=np.float32),
        logits=np.zeros((3, 1, 2), dtype=np.float32),
        values=np.zeros((3, 1), dtype=np.float32),
    )

    batch = build_learner_batch(
        stack,
        rollout,
        np.zeros((1,), dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert batch["reset_before_step"][:, 0].tolist() == [False, False, True]
    assert stack.config.rewards is not None
    gamma = float(stack.config.rewards.gamma)
    assert batch["discounts"][:, 0].tolist() == pytest.approx([gamma, 0.0, gamma])


def test_build_learner_batch_threads_bootstrap_value_into_vtrace_targets() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    behavior_logp = np.full((1, 1), -np.log(2.0), dtype=np.float32)
    rollout = replace(
        minimal_rollout(),
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        legal_mask=np.ones((1, 1, 2), dtype=np.bool_),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=behavior_logp,
        logits=np.zeros((1, 1, 2), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.asarray([0], dtype=np.int64),
    )

    batch = build_learner_batch(
        stack,
        rollout,
        np.asarray([2.0], dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert stack.config.rewards is not None
    expected_target = float(stack.config.rewards.gamma) * 2.0
    assert batch["vtrace_result"].vs[:, 0].tolist() == pytest.approx([expected_target])
    assert batch["vtrace_result"].pg_advantages[:, 0].tolist() == pytest.approx([expected_target])
