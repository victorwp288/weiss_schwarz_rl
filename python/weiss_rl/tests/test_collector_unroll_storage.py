from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.runtime_components.collector_unroll_storage import (
    CollectorStepPayload,
    CollectorStepStorage,
    build_collector_runtime_unroll,
    estimate_collector_copied_bytes,
    legal_actions_from_collector_steps,
    normalize_collector_bootstrap_arrays,
    store_collector_step,
    write_collector_step,
)


def _teacher_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([5, 6], dtype=np.int32),
        np.array([7, 8], dtype=np.int32),
        np.array([9, 10], dtype=np.int32),
        np.array([True, False], dtype=np.bool_),
    )


def test_store_collector_step_writes_dtypes_and_retention_counter() -> None:
    obs = np.zeros((1, 2, 3), dtype=np.float32)
    actions = np.zeros((1, 2), dtype=np.uint16)
    rewards = np.zeros((1, 2), dtype=np.float32)
    terminated = np.zeros((1, 2), dtype=np.bool_)
    truncated = np.zeros((1, 2), dtype=np.bool_)
    to_play = np.zeros((1, 2), dtype=np.int8)
    logp = np.zeros((1, 2), dtype=np.float32)
    values = np.zeros((1, 2), dtype=np.float32)
    seeds = np.zeros((1, 2), dtype=np.uint64)
    teacher_family = np.full((1, 2), -1, dtype=np.int32)
    teacher_slot = np.full((1, 2), -1, dtype=np.int32)
    teacher_move_source = np.full((1, 2), -1, dtype=np.int32)
    teacher_attack_type = np.full((1, 2), -1, dtype=np.int32)
    teacher_action = np.full((1, 2), -1, dtype=np.int32)
    teacher_valid = np.zeros((1, 2), dtype=np.bool_)
    retention = np.zeros((1, 2), dtype=np.bool_)
    counters = {"trajectory_retention_rows": 0}

    store_collector_step(
        step_index=0,
        obs_storage=obs,
        actions_storage=actions,
        rewards_storage=rewards,
        terminated_storage=terminated,
        truncated_storage=truncated,
        to_play_seat_storage=to_play,
        behavior_logp_storage=logp,
        values_storage=values,
        episode_seed_storage=seeds,
        teacher_family_storage=teacher_family,
        teacher_slot_storage=teacher_slot,
        teacher_move_source_storage=teacher_move_source,
        teacher_attack_type_storage=teacher_attack_type,
        teacher_action_storage=teacher_action,
        teacher_valid_storage=teacher_valid,
        trajectory_retention_storage=retention,
        obs_step=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        actions=np.array([11, 12], dtype=np.int64),
        rewards=np.array([0.25, -0.5], dtype=np.float64),
        terminated=np.array([False, True]),
        truncated=np.array([True, False]),
        actor_step=np.array([0, 1], dtype=np.int64),
        behavior_logp=np.array([-0.1, -0.2], dtype=np.float64),
        values=np.array([0.3, 0.4], dtype=np.float64),
        episode_seed=np.array([101, 202], dtype=np.int64),
        teacher_labels=_teacher_labels(),
        retention_valid=np.array([True, False]),
        counters=counters,
    )

    assert actions.dtype == np.uint16
    assert actions[0].tolist() == [11, 12]
    assert rewards[0].tolist() == [0.25, -0.5]
    assert terminated[0].tolist() == [False, True]
    assert truncated[0].tolist() == [True, False]
    assert to_play[0].tolist() == [0, 1]
    np.testing.assert_allclose(logp[0], np.array([-0.1, -0.2], dtype=np.float32))
    np.testing.assert_allclose(values[0], np.array([0.3, 0.4], dtype=np.float32))
    assert seeds[0].tolist() == [101, 202]
    assert teacher_family[0].tolist() == [1, 2]
    assert teacher_slot[0].tolist() == [3, 4]
    assert teacher_move_source[0].tolist() == [5, 6]
    assert teacher_attack_type[0].tolist() == [7, 8]
    assert teacher_action[0].tolist() == [9, 10]
    assert teacher_valid[0].tolist() == [True, False]
    assert retention[0].tolist() == [True, False]
    assert counters["trajectory_retention_rows"] == 1


def test_write_collector_step_uses_explicit_storage_and_payload_contract() -> None:
    storage = CollectorStepStorage(
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.uint16),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int8),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        teacher_family=np.full((1, 1), -1, dtype=np.int32),
        teacher_slot=np.full((1, 1), -1, dtype=np.int32),
        teacher_move_source=np.full((1, 1), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, 1), -1, dtype=np.int32),
        teacher_action=np.full((1, 1), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, 1), dtype=np.bool_),
        trajectory_retention=np.zeros((1, 1), dtype=np.bool_),
    )
    payload = CollectorStepPayload(
        obs=np.array([[3.0, 4.0]], dtype=np.float32),
        actions=np.array([9], dtype=np.int64),
        rewards=np.array([0.75], dtype=np.float64),
        terminated=np.array([False]),
        truncated=np.array([True]),
        actor_step=np.array([1], dtype=np.int64),
        behavior_logp=np.array([-0.5], dtype=np.float64),
        values=np.array([0.25], dtype=np.float64),
        episode_seed=np.array([44], dtype=np.int64),
        teacher_labels=tuple(label[:1] for label in _teacher_labels()),
        retention_valid=np.array([True]),
    )
    counters = {"trajectory_retention_rows": 0}

    write_collector_step(step_index=0, storage=storage, payload=payload, counters=counters)

    assert storage.obs[0].tolist() == [[3.0, 4.0]]
    assert storage.actions[0].tolist() == [9]
    assert storage.rewards[0].tolist() == [0.75]
    assert storage.truncated[0].tolist() == [True]
    assert storage.to_play_seat[0].tolist() == [1]
    assert storage.teacher_family is not None
    assert storage.teacher_family[0].tolist() == [1]
    assert storage.trajectory_retention is not None
    assert storage.trajectory_retention[0].tolist() == [True]
    assert counters["trajectory_retention_rows"] == 1


def test_write_collector_step_requires_teacher_storage_when_labels_are_present() -> None:
    storage = CollectorStepStorage(
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.uint16),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int8),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention=None,
    )
    payload = CollectorStepPayload(
        obs=np.array([[1.0]], dtype=np.float32),
        actions=np.array([1], dtype=np.int64),
        rewards=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor_step=np.array([0], dtype=np.int64),
        behavior_logp=np.array([0.0], dtype=np.float32),
        values=np.array([0.0], dtype=np.float32),
        episode_seed=np.array([1], dtype=np.uint64),
        teacher_labels=tuple(label[:1] for label in _teacher_labels()),
        retention_valid=None,
    )

    with pytest.raises(ValueError, match="teacher storage arrays"):
        write_collector_step(
            step_index=0,
            storage=storage,
            payload=payload,
            counters={"trajectory_retention_rows": 0},
        )


def test_store_collector_step_allows_optional_teacher_storage_for_heuristic_paths() -> None:
    obs = np.zeros((1, 1, 2), dtype=np.float32)
    actions = np.zeros((1, 1), dtype=np.uint16)
    rewards = np.zeros((1, 1), dtype=np.float32)
    terminated = np.zeros((1, 1), dtype=np.bool_)
    truncated = np.zeros((1, 1), dtype=np.bool_)
    to_play = np.zeros((1, 1), dtype=np.int8)
    logp = np.zeros((1, 1), dtype=np.float32)
    values = np.zeros((1, 1), dtype=np.float32)
    seeds = np.zeros((1, 1), dtype=np.uint64)
    counters = {"trajectory_retention_rows": 0}

    store_collector_step(
        step_index=0,
        obs_storage=obs,
        actions_storage=actions,
        rewards_storage=rewards,
        terminated_storage=terminated,
        truncated_storage=truncated,
        to_play_seat_storage=to_play,
        behavior_logp_storage=logp,
        values_storage=values,
        episode_seed_storage=seeds,
        teacher_family_storage=None,
        teacher_slot_storage=None,
        teacher_move_source_storage=None,
        teacher_attack_type_storage=None,
        teacher_action_storage=None,
        teacher_valid_storage=None,
        trajectory_retention_storage=None,
        obs_step=np.array([[8.0, 9.0]], dtype=np.float32),
        actions=np.array([5], dtype=np.int64),
        rewards=np.array([1.25], dtype=np.float32),
        terminated=np.array([False], dtype=np.bool_),
        truncated=np.array([True], dtype=np.bool_),
        actor_step=np.array([1], dtype=np.int64),
        behavior_logp=np.array([0.0], dtype=np.float32),
        values=np.array([0.0], dtype=np.float32),
        episode_seed=np.array([77], dtype=np.uint64),
        teacher_labels=None,
        retention_valid=None,
        counters=counters,
    )

    assert obs[0].tolist() == [[8.0, 9.0]]
    assert actions[0].tolist() == [5]
    assert rewards[0].tolist() == [1.25]
    assert terminated[0].tolist() == [False]
    assert truncated[0].tolist() == [True]
    assert to_play[0].tolist() == [1]
    assert seeds[0].tolist() == [77]
    assert counters["trajectory_retention_rows"] == 0


def test_normalize_collector_bootstrap_arrays_preserves_runtime_unroll_dtypes() -> None:
    bootstrap = normalize_collector_bootstrap_arrays(
        bootstrap_obs=np.ones((2, 3), dtype=np.float64),
        bootstrap_actor=np.array([0, 1], dtype=np.int32),
        bootstrap_value=np.array([0.1, 0.2], dtype=np.float64),
    )

    assert bootstrap.obs.dtype == np.float32
    assert bootstrap.actor.dtype == np.int64
    assert bootstrap.value.dtype == np.float32
    assert bootstrap.obs.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert bootstrap.actor.tolist() == [0, 1]


def test_legal_actions_from_collector_steps_preserves_packed_and_mask_layouts() -> None:
    packed = legal_actions_from_collector_steps(
        layout_name="i16_legal_ids",
        action_dim=8,
        packed_ids=[np.array([1, 3], dtype=np.uint32), np.array([2], dtype=np.uint32)],
        packed_offsets=[
            np.array([0], dtype=np.uint32),
            np.array([1, 2], dtype=np.uint32),
            np.array([3], dtype=np.uint32),
        ],
        packed_meta=[
            np.array([[10, 0], [30, 0]], dtype=np.uint16),
            np.array([[20, 0]], dtype=np.uint16),
        ],
        mask_steps=[],
    )

    assert packed.ids is not None
    assert packed.offsets is not None
    assert packed.meta is not None
    assert packed.ids.tolist() == [1, 3, 2]
    assert packed.offsets.tolist() == [0, 1, 2, 3]
    assert packed.meta[:, 0].tolist() == [10, 30, 20]

    mask = legal_actions_from_collector_steps(
        layout_name="dense_mask",
        action_dim=3,
        packed_ids=[],
        packed_offsets=[np.array([0], dtype=np.uint32)],
        packed_meta=[],
        mask_steps=[
            np.array([[True, False, False]], dtype=np.bool_),
            np.array([[False, True, True]], dtype=np.bool_),
        ],
    )

    assert mask.mask is not None
    assert mask.mask.shape == (2, 1, 3)
    assert mask.mask.tolist() == [[[True, False, False]], [[False, True, True]]]


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
