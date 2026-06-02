from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime, RuntimeUnroll
from weiss_rl.runtime.components import impala_learner_batch, ppo_learner_batch
from weiss_rl.runtime.components.batching import (
    actor_perspective_discounts,
    build_impala_learner_batch,
    build_ppo_learner_batch,
    gae_advantages,
)
from weiss_rl.runtime.components.legal_batching import (
    concatenate_batch_legal_actions,
    concatenate_legal_actions,
    require_ids_offsets,
    slice_packed_rows_with_meta,
    structured_legal_batch_from_packed,
)


def _make_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=f"{actor_id}:{unroll_seq}:{behavior_policy_version}",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        behavior_logits=None,
        counters=counters,
    )


def test_runtime_batching_facade_reexports_algorithm_payload_builders() -> None:
    assert build_impala_learner_batch is impala_learner_batch.build_impala_learner_batch
    assert build_ppo_learner_batch is ppo_learner_batch.build_ppo_learner_batch
    assert build_impala_learner_batch.__module__ == "weiss_rl.runtime.components.impala_learner_batch"
    assert build_ppo_learner_batch.__module__ == "weiss_rl.runtime.components.ppo_learner_batch"


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


def test_runtime_batching_concatenates_packed_legal_actions_in_time_major_order() -> None:
    unroll_a = SimpleNamespace(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 3, 2, 4], dtype=np.uint32),
            np.asarray([0, 2, 4], dtype=np.uint32),
            action_space=8,
        ),
    )
    unroll_b = SimpleNamespace(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([5, 7, 6], dtype=np.uint32),
            np.asarray([0, 2, 3], dtype=np.uint32),
            action_space=8,
        ),
    )

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=8)

    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.ids.tolist() == [1, 3, 5, 7, 2, 4, 6]
    assert combined.offsets.tolist() == [0, 2, 4, 6, 7]


def test_build_impala_batch_concatenates_opponent_context_index() -> None:
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        opponent_context_index=np.asarray([[1]], dtype=np.int16),
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        opponent_context_index=np.asarray([[2]], dtype=np.int16),
    )

    batch = build_impala_learner_batch(
        [unroll_a, unroll_b],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["opponent_context_index"].dtype == np.int16
    assert batch["opponent_context_index"].tolist() == [[1, 2]]


def test_build_impala_batch_backfills_terminal_outcome_to_last_train_row() -> None:
    unroll = SimpleNamespace(
        obs=np.zeros((4, 1, 1), dtype=np.float32),
        actions=np.zeros((4, 1), dtype=np.int64),
        rewards=np.asarray([[0.0], [-1.0], [0.0], [-1.0]], dtype=np.float32),
        terminated=np.asarray([[False], [True], [False], [True]], dtype=np.bool_),
        truncated=np.zeros((4, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [1], [0], [1]], dtype=np.int64),
        behavior_logp=np.zeros((4, 1), dtype=np.float32),
        values=np.zeros((4, 1), dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        policy_train_mask=np.asarray([[True], [False], [True], [False]], dtype=np.bool_),
        legal_actions=LegalActionBatch.from_mask(np.ones((4, 1, 1), dtype=np.bool_)),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )

    batch = build_impala_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        terminal_outcome_backfill_reward=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert batch["terminal_outcome_backfill_count"] == 2
    assert batch["terminal_outcome_backfill_total_micros"] == 2_000_000


def test_build_impala_batch_trace_backfills_terminal_outcome_to_episode_suffix() -> None:
    unroll = SimpleNamespace(
        obs=np.zeros((5, 1, 1), dtype=np.float32),
        actions=np.zeros((5, 1), dtype=np.int64),
        rewards=np.asarray([[0.0], [0.0], [-1.0], [0.0], [-1.0]], dtype=np.float32),
        terminated=np.asarray([[False], [False], [True], [False], [True]], dtype=np.bool_),
        truncated=np.zeros((5, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [0], [1], [0], [0]], dtype=np.int64),
        behavior_logp=np.zeros((5, 1), dtype=np.float32),
        values=np.zeros((5, 1), dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        policy_train_mask=np.asarray([[True], [True], [False], [True], [True]], dtype=np.bool_),
        legal_actions=LegalActionBatch.from_mask(np.ones((5, 1, 1), dtype=np.bool_)),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )

    batch = build_impala_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        terminal_outcome_trace_backfill_reward=0.25,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.25, 0.25, -1.0, -0.25, -1.0])
    assert batch["terminal_outcome_trace_backfill_count"] == 3
    assert batch["terminal_outcome_trace_backfill_total_micros"] == 750_000


def test_runtime_batching_concatenate_legal_actions_keeps_packed_ids_fast_path() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([0, 2, 1, 2], dtype=np.uint32),
        np.array([0, 2, 4], dtype=np.uint32),
        action_space=64,
    )
    unroll_a = replace(_make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)
    unroll_b = replace(_make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 2, 4]
    assert combined.ids.tolist() == [0, 2, 0, 2]


def test_runtime_batching_concatenate_legal_actions_reorders_packed_rows_to_match_time_major_layout() -> None:
    packed_a = LegalActionBatch.from_packed(
        np.array([10, 11, 20, 21], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    packed_b = LegalActionBatch.from_packed(
        np.array([30, 31, 40, 41], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_a,
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_b,
    )

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert combined.ids.tolist() == [10, 11, 30, 31, 20, 21, 40, 41]


def test_runtime_batching_mixed_legal_payloads_fall_back_to_dense_mask() -> None:
    packed_unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 3, 2], dtype=np.uint32),
            np.asarray([0, 2, 3], dtype=np.uint32),
            action_space=5,
        ),
    )
    mask_unroll = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(
            np.asarray([[[True, False, False, False, True]], [[False, True, False, True, False]]], dtype=np.bool_),
            action_space=5,
        ),
    )

    combined = concatenate_legal_actions([packed_unroll, mask_unroll], action_space=5)

    assert combined.ids is None
    assert combined.offsets is None
    assert combined.mask is not None
    assert combined.mask.tolist() == [
        [[False, True, False, True, False], [True, False, False, False, True]],
        [[False, False, True, False, False], [False, True, False, True, False]],
    ]


def test_runtime_batching_slices_packed_rows_with_meta() -> None:
    ids, offsets, meta = slice_packed_rows_with_meta(
        np.asarray([1, 2, 3, 4, 5], dtype=np.uint32),
        np.asarray([0, 2, 3, 5], dtype=np.uint32),
        np.asarray([2, 0], dtype=np.int64),
        legal_action_meta=np.asarray([[10], [11], [12], [13], [14]], dtype=np.uint16),
    )

    assert ids.tolist() == [4, 5, 1, 2]
    assert offsets.tolist() == [0, 2, 4]
    assert meta is not None
    assert meta.tolist() == [[13], [14], [10], [11]]


def test_runtime_batching_structured_legal_batch_from_packed_preserves_meta() -> None:
    batch = structured_legal_batch_from_packed(
        np.asarray([1, 2, 3], dtype=np.uint32),
        np.asarray([0, 1, 3], dtype=np.uint32),
        np.asarray([1], dtype=np.int64),
        np.asarray([[10], [11], [12]], dtype=np.uint16),
    )

    assert batch.ids is not None
    assert batch.offsets is not None
    assert batch.meta is not None
    assert batch.ids.tolist() == [2, 3]
    assert batch.offsets.tolist() == [0, 2]
    assert batch.meta.tolist() == [[11], [12]]


def test_legal_action_batch_uses_metadata_to_expand_packed_payloads() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([1, 3], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        action_space=5,
    )

    mask = packed.to_mask(expected_shape=(1, 2))

    assert packed.action_space == 5
    assert np.array_equal(
        mask,
        np.array([[[False, True, False, False, False], [False, False, False, True, False]]], dtype=np.bool_),
    )


def test_runtime_batching_concatenates_decision_batches_or_rejects_missing_packed_legality() -> None:
    first = SimpleNamespace(
        mask=None,
        ids_offsets=(np.asarray([1, 2], dtype=np.uint32), np.asarray([0, 2], dtype=np.uint32)),
        legal_action_meta=np.asarray([[10], [11]], dtype=np.uint16),
    )
    second = SimpleNamespace(
        mask=None,
        ids_offsets=(np.asarray([3], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
        legal_action_meta=np.asarray([[12]], dtype=np.uint16),
    )

    combined = concatenate_batch_legal_actions([first, second], action_space=8)

    assert combined is not None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.meta is not None
    assert combined.ids.tolist() == [1, 2, 3]
    assert combined.offsets.tolist() == [0, 2, 3]
    assert combined.meta.tolist() == [[10], [11], [12]]

    with pytest.raises(RuntimeError, match="requires ids_offsets"):
        require_ids_offsets(SimpleNamespace(ids_offsets=None))


def test_gae_advantages_matches_manual_discounted_deltas() -> None:
    rewards = np.asarray([[1.0], [0.5]], dtype=np.float32)
    values = np.asarray([[0.2], [0.3]], dtype=np.float32)
    discounts = np.asarray([[1.0], [0.0]], dtype=np.float32)
    bootstrap = np.asarray([0.4], dtype=np.float32)

    advantages = gae_advantages(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        discounts=discounts,
        gae_lambda=0.95,
    )

    expected_last = 0.5 - 0.3
    expected_first = (1.0 + 0.3 - 0.2) + (0.95 * expected_last)
    assert advantages[:, 0].tolist() == pytest.approx([expected_first, expected_last])


def test_actor_perspective_discounts_flip_when_next_value_is_opponent_perspective() -> None:
    done = np.asarray([[False], [False], [False]], dtype=np.bool_)
    to_play_seat = np.asarray([[0], [1], [1]], dtype=np.int64)
    bootstrap_actor = np.asarray([0], dtype=np.int64)

    discounts = actor_perspective_discounts(
        done=done,
        to_play_seat=to_play_seat,
        bootstrap_actor=bootstrap_actor,
        gamma=0.99,
    )

    assert discounts[:, 0].tolist() == pytest.approx([-0.99, 0.99, -0.99])


def test_actor_perspective_discounts_ignore_invalid_bootstrap_actor_on_done_rows() -> None:
    discounts = actor_perspective_discounts(
        done=np.asarray([[True]], dtype=np.bool_),
        to_play_seat=np.asarray([[0]], dtype=np.int64),
        bootstrap_actor=np.asarray([-1], dtype=np.int64),
        gamma=0.99,
    )

    assert discounts[:, 0].tolist() == pytest.approx([0.0])


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


def test_build_learner_batch_preserves_bootstrap_inputs_for_learner_values() -> None:
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
        bootstrap_obs=np.array([[3.0]], dtype=np.float32),
        bootstrap_actor=np.array([1], dtype=np.int64),
        final_hidden_state=np.array([[[1.0, 2.0]]], dtype=np.float32),
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

    assert np.array_equal(batch["bootstrap_obs"], unroll.bootstrap_obs)
    assert np.array_equal(batch["bootstrap_actor"], unroll.bootstrap_actor)
    assert np.array_equal(batch["final_hidden_state"], unroll.final_hidden_state)


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
