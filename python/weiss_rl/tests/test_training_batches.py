from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.config import load_stack_config
from weiss_rl.training.batches import (
    MinimalRollout,
    bootstrap_values,
    build_learner_batch,
    collect_training_batch,
)


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _minimal_rollout(*, truncated_bootstrap_actor: np.ndarray | None = None) -> MinimalRollout:
    bootstrap_actor = np.array([0], dtype=np.int64) if truncated_bootstrap_actor is None else truncated_bootstrap_actor
    return MinimalRollout(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_mask=np.ones((2, 1, 2), dtype=np.bool_),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        logits=np.zeros((2, 1, 2), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        bootstrap_obs=np.arange(len(bootstrap_actor), dtype=np.float32).reshape(len(bootstrap_actor), 1),
        bootstrap_actor=bootstrap_actor,
    )


class _RuntimeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def collect_update_batch(self, **kwargs: object) -> str:
        self.calls.append(("impala", kwargs))
        return "impala_batch"

    def collect_policy_batch(self, **kwargs: object) -> str:
        self.calls.append(("ppo", kwargs))
        return "ppo_batch"


def _training_config() -> SimpleNamespace:
    return SimpleNamespace(
        vtrace_rho_bar=1.25,
        vtrace_c_bar=0.75,
        ppo_gae_lambda=0.92,
    )


def _rewards_config() -> SimpleNamespace:
    return SimpleNamespace(
        gamma=0.99,
        truncation=SimpleNamespace(
            reward=-0.5,
            bootstrap_value=True,
        ),
    )


def test_collect_training_batch_dispatches_impala_with_vtrace_arguments() -> None:
    runtime = _RuntimeRecorder()

    batch = collect_training_batch(
        runtime=runtime,
        algorithm="impala_vtrace_structured_v1",
        training_config=_training_config(),
        rewards_config=_rewards_config(),
    )

    assert batch == "impala_batch"
    assert runtime.calls == [
        (
            "impala",
            {
                "gamma": 0.99,
                "truncation_reward": -0.5,
                "truncation_bootstrap_value": True,
                "vtrace_rho_bar": 1.25,
                "vtrace_c_bar": 0.75,
            },
        )
    ]


def test_collect_training_batch_dispatches_ppo_with_gae_arguments() -> None:
    runtime = _RuntimeRecorder()

    batch = collect_training_batch(
        runtime=runtime,
        algorithm="ppo_lite_masked_v1",
        training_config=_training_config(),
        rewards_config=_rewards_config(),
    )

    assert batch == "ppo_batch"
    assert runtime.calls == [
        (
            "ppo",
            {
                "gamma": 0.99,
                "gae_lambda": 0.92,
                "truncation_reward": -0.5,
                "truncation_bootstrap_value": True,
            },
        )
    ]


def test_collect_training_batch_rejects_unknown_algorithm() -> None:
    with pytest.raises(RuntimeError, match="Unsupported training.algorithm: unknown"):
        collect_training_batch(
            runtime=_RuntimeRecorder(),
            algorithm="unknown",
            training_config=_training_config(),
            rewards_config=_rewards_config(),
        )


def test_build_learner_batch_preserves_truncation_reward_and_discount_semantics() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = _minimal_rollout()

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
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = replace(
        _minimal_rollout(),
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


def test_bootstrap_values_only_evaluates_live_actor_rows() -> None:
    class _Model:
        def forward_seat_aware(self, obs, actor, hidden):
            assert obs.cpu().numpy().reshape(-1).tolist() == [0.0, 1.0]
            assert actor.cpu().numpy().tolist() == [0, 1]
            assert tuple(hidden.shape) == (2, 2)
            return None, torch.tensor([3.0, 4.0], dtype=torch.float32), None

    values = bootstrap_values(
        _Model(),
        _minimal_rollout(truncated_bootstrap_actor=np.array([0, 1, -1], dtype=np.int64)),
        torch.zeros((3, 2), dtype=torch.float32),
        device=torch.device("cpu"),
    )

    assert values.tolist() == pytest.approx([3.0, 4.0, 0.0])


def test_build_learner_batch_requires_training_and_rewards_config_blocks() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(training=None, rewards=None))

    with pytest.raises(RuntimeError, match="requires training and rewards config blocks"):
        build_learner_batch(
            stack,
            _minimal_rollout(),
            np.zeros((1,), dtype=np.float32),
            action_dim=2,
            initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
            pass_action_id=1,
        )
