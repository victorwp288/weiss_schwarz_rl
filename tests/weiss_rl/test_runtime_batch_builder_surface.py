from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from weiss_rl.runtime.components.batching import (
    build_impala_learner_batch,
    build_ppo_learner_batch,
    impala_learner_batch,
    ppo_learner_batch,
)

from .runtime_test_support import _make_runtime_unroll


def test_runtime_batching_facade_reexports_algorithm_payload_builders() -> None:
    assert build_impala_learner_batch is impala_learner_batch.build_impala_learner_batch
    assert build_ppo_learner_batch is ppo_learner_batch.build_ppo_learner_batch
    assert build_impala_learner_batch.__module__ == "weiss_rl.runtime.components.batching.impala_learner_batch"
    assert build_ppo_learner_batch.__module__ == "weiss_rl.runtime.components.batching.ppo_learner_batch"


def test_build_impala_batch_exposes_stable_learner_payload_contract() -> None:
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        behavior_logp=np.asarray([[-0.25]], dtype=np.float32),
        values=np.asarray([[0.5]], dtype=np.float32),
        bootstrap_value=np.asarray([0.75], dtype=np.float32),
        bootstrap_obs=np.asarray([[3.0]], dtype=np.float32),
        bootstrap_actor=np.asarray([1], dtype=np.int64),
        final_hidden_state=np.asarray([[[2.0]]], dtype=np.float32),
    )

    batch = build_impala_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.25,
        vtrace_c_bar=0.75,
    )

    assert set(batch) == {
        "obs",
        "actions",
        "legal_actions",
        "legal_mask",
        "legal_action_meta",
        "to_play_seat",
        "actor",
        "initial_hidden_state",
        "rewards",
        "discounts",
        "reset_before_step",
        "policy_train_mask",
        "opponent_context_index",
        "teacher_family",
        "teacher_slot",
        "teacher_move_source",
        "teacher_attack_type",
        "teacher_action",
        "teacher_valid",
        "trajectory_retention_valid",
        "bootstrap_obs",
        "bootstrap_actor",
        "final_hidden_state",
        "behavior_logp",
        "behavior_values",
        "bootstrap_value",
        "vtrace_rho_bar",
        "vtrace_c_bar",
        "terminal_outcome_backfill_count",
        "terminal_outcome_backfill_total_micros",
        "terminal_outcome_trace_backfill_count",
        "terminal_outcome_trace_backfill_total_micros",
    }
    assert batch["actor"] is batch["to_play_seat"]
    assert np.allclose(batch["behavior_logp"], np.asarray([[-0.25]], dtype=np.float32))
    assert np.allclose(batch["behavior_values"], np.asarray([[0.5]], dtype=np.float32))
    assert batch["bootstrap_value"].tolist() == pytest.approx([0.75])
    assert np.allclose(batch["bootstrap_obs"], np.asarray([[3.0]], dtype=np.float32))
    assert batch["bootstrap_actor"].tolist() == [1]
    assert np.allclose(batch["final_hidden_state"], np.asarray([[[2.0]]], dtype=np.float32))
    assert batch["vtrace_rho_bar"] == pytest.approx(1.25)
    assert batch["vtrace_c_bar"] == pytest.approx(0.75)
    assert batch["terminal_outcome_backfill_count"] == 0
    assert batch["terminal_outcome_trace_backfill_total_micros"] == 0


def test_build_ppo_batch_exposes_stable_learner_payload_contract() -> None:
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        behavior_logp=np.asarray([[-0.25]], dtype=np.float32),
        values=np.asarray([[0.5]], dtype=np.float32),
        rewards=np.asarray([[0.25]], dtype=np.float32),
        bootstrap_value=np.asarray([0.75], dtype=np.float32),
    )

    batch = build_ppo_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        gae_lambda=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
    )

    assert set(batch) == {
        "obs",
        "actions",
        "legal_actions",
        "legal_mask",
        "legal_action_meta",
        "to_play_seat",
        "actor",
        "initial_hidden_state",
        "rewards",
        "discounts",
        "reset_before_step",
        "policy_train_mask",
        "opponent_context_index",
        "teacher_family",
        "teacher_slot",
        "teacher_move_source",
        "teacher_attack_type",
        "teacher_action",
        "teacher_valid",
        "trajectory_retention_valid",
        "old_logp",
        "old_values",
        "returns",
        "advantages",
    }
    assert batch["actor"] is batch["to_play_seat"]
    assert np.allclose(batch["old_logp"], np.asarray([[-0.25]], dtype=np.float32))
    assert np.allclose(batch["old_values"], np.asarray([[0.5]], dtype=np.float32))
    assert np.allclose(batch["advantages"], np.asarray([[0.5]], dtype=np.float32))
    assert np.allclose(batch["returns"], np.asarray([[1.0]], dtype=np.float32))
