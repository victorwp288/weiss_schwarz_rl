from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.collector_unroll_storage import (
    build_collector_runtime_unroll,
    estimate_collector_copied_bytes,
)


def test_build_collector_runtime_unroll_packages_fields_and_copied_bytes() -> None:
    obs = np.ones((1, 2, 3), dtype=np.float32)
    actions = np.array([[1, 2]], dtype=np.uint16)
    rewards = np.array([[0.5, -0.25]], dtype=np.float32)
    terminated = np.array([[False, True]], dtype=np.bool_)
    truncated = np.array([[False, False]], dtype=np.bool_)
    to_play = np.array([[0, 1]], dtype=np.int8)
    logp = np.array([[-0.1, -0.2]], dtype=np.float32)
    values = np.array([[0.3, 0.4]], dtype=np.float32)
    seeds = np.array([[10, 20]], dtype=np.uint64)
    policy_train = np.array([[True, False]], dtype=np.bool_)
    opponent_context = np.array([[0, 1]], dtype=np.int16)
    teacher = np.array([[1, -1]], dtype=np.int32)
    teacher_valid = np.array([[True, False]], dtype=np.bool_)
    retention = np.array([[False, True]], dtype=np.bool_)
    bootstrap_obs = np.ones((2, 3), dtype=np.float64)
    bootstrap_actor = np.array([0, 1], dtype=np.int32)
    bootstrap_value = np.array([0.7, 0.8], dtype=np.float64)
    counters = {"copied_bytes_estimate": 5}

    unroll = build_collector_runtime_unroll(
        actor_id=3,
        unroll_seq=4,
        behavior_policy_version=5,
        layout_name="i16_legal_ids",
        action_dim=8,
        obs=obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        to_play_seat=to_play,
        behavior_logp=logp,
        values=values,
        packed_ids=[np.array([1, 2], dtype=np.uint32)],
        packed_offsets=[np.array([0, 1, 2], dtype=np.uint32)],
        packed_meta=[],
        mask_steps=[],
        bootstrap_obs=bootstrap_obs,
        bootstrap_actor=bootstrap_actor,
        bootstrap_value=bootstrap_value,
        initial_hidden_state=np.zeros((2, 4), dtype=np.float32),
        final_hidden_state=np.ones((2, 4), dtype=np.float32),
        episode_seed=seeds,
        policy_train_mask=policy_train,
        opponent_context_index=opponent_context,
        teacher_family=teacher,
        teacher_slot=teacher,
        teacher_move_source=teacher,
        teacher_attack_type=teacher,
        teacher_action=teacher,
        teacher_valid=teacher_valid,
        trajectory_retention_valid=retention,
        counters=counters,
        copy_counters=True,
    )

    expected_added = estimate_collector_copied_bytes(
        obs,
        actions,
        rewards,
        terminated,
        truncated,
        to_play,
        logp,
        values,
        seeds,
        policy_train,
        opponent_context,
        teacher,
        teacher,
        teacher,
        teacher,
        teacher,
        teacher_valid,
        retention,
        np.asarray(bootstrap_obs, dtype=np.float32),
        np.asarray(bootstrap_actor, dtype=np.int64),
        np.asarray(bootstrap_value, dtype=np.float32),
    )
    assert counters["copied_bytes_estimate"] == 5 + expected_added
    assert unroll.counters is not counters
    assert unroll.counters is not None
    assert unroll.counters["copied_bytes_estimate"] == counters["copied_bytes_estimate"]
    assert unroll.bootstrap_obs.dtype == np.float32
    assert unroll.bootstrap_actor.dtype == np.int64
    assert unroll.bootstrap_value.dtype == np.float32
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.ids.tolist() == [1, 2]
    assert unroll.unroll_hash


def test_build_collector_runtime_unroll_allows_unlabeled_heuristic_rollout_fields() -> None:
    counters = {"copied_bytes_estimate": 0}
    unroll = build_collector_runtime_unroll(
        actor_id=1,
        unroll_seq=2,
        behavior_policy_version=3,
        layout_name="i16_legal_ids",
        action_dim=4,
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        actions=np.array([[1]], dtype=np.uint16),
        rewards=np.array([[0.0]], dtype=np.float32),
        terminated=np.array([[False]], dtype=np.bool_),
        truncated=np.array([[False]], dtype=np.bool_),
        to_play_seat=np.array([[0]], dtype=np.int8),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        packed_ids=[np.array([1], dtype=np.uint32)],
        packed_offsets=[np.array([0, 1], dtype=np.uint32)],
        packed_meta=[],
        mask_steps=[],
        bootstrap_obs=np.zeros((1, 2), dtype=np.float32),
        bootstrap_actor=np.array([0], dtype=np.int64),
        bootstrap_value=np.array([0.0], dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1, 2), dtype=np.float32),
        final_hidden_state=np.ones((1, 1, 2), dtype=np.float32),
        episode_seed=np.array([[99]], dtype=np.uint64),
        policy_train_mask=np.array([[True]], dtype=np.bool_),
        opponent_context_index=np.array([[0]], dtype=np.int16),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
        counters=counters,
        copy_counters=False,
    )

    assert unroll.teacher_family is None
    assert unroll.teacher_valid is None
    assert unroll.counters is counters
    assert counters["copied_bytes_estimate"] > 0
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.ids.tolist() == [1]
