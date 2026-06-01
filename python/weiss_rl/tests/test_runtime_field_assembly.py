from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime_components.field_assembly import (
    OptionalTimeMajorFieldSpec,
    RequiredTimeMajorFieldSpec,
    base_runtime_learner_payload,
    concat_optional_time_major_fields,
    concat_required_time_major_fields,
    concat_runtime_batch_fields,
    concat_time_major_field,
    time_major_batch_layout,
)


def _unroll(
    *,
    width: int,
    teacher_action: np.ndarray | None = None,
    teacher_valid: np.ndarray | None = None,
    trajectory_retention_valid: np.ndarray | None = None,
    opponent_context_index: np.ndarray | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((2, width, 1), dtype=np.float32),
        actions=np.arange(2 * width, dtype=np.int64).reshape(2, width),
        rewards=np.zeros((2, width), dtype=np.float32),
        terminated=np.zeros((2, width), dtype=np.bool_),
        truncated=np.zeros((2, width), dtype=np.bool_),
        to_play_seat=np.zeros((2, width), dtype=np.int64),
        behavior_logp=np.zeros((2, width), dtype=np.float32),
        values=np.zeros((2, width), dtype=np.float32),
        initial_hidden_state=np.zeros((width, 1), dtype=np.float32),
        policy_train_mask=np.ones((2, width), dtype=np.bool_),
        opponent_context_index=opponent_context_index,
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        trajectory_retention_valid=trajectory_retention_valid,
        legal_actions=LegalActionBatch.from_mask(np.ones((2, width, 3), dtype=np.bool_)),
    )


def test_concat_runtime_batch_fields_fills_missing_auxiliary_labels_with_sentinels() -> None:
    labeled = _unroll(
        width=1,
        teacher_action=np.asarray([[4], [5]], dtype=np.int32),
        teacher_valid=np.asarray([[True], [False]], dtype=np.bool_),
        trajectory_retention_valid=np.asarray([[False], [True]], dtype=np.bool_),
    )
    unlabeled = _unroll(width=2)

    fields = concat_runtime_batch_fields(
        [labeled, unlabeled],
        action_dim=3,
        record_batch_timer_ms=None,
    )

    assert fields.teacher_action is not None
    assert fields.teacher_action.dtype == np.int32
    assert fields.teacher_action.tolist() == [[4, -1, -1], [5, -1, -1]]
    assert fields.teacher_valid is not None
    assert fields.teacher_valid.tolist() == [[True, False, False], [False, False, False]]
    assert fields.trajectory_retention_valid is not None
    assert fields.trajectory_retention_valid.tolist() == [[False, False, False], [True, False, False]]


def test_field_assembly_specs_preserve_required_and_optional_contracts() -> None:
    labeled = _unroll(
        width=1,
        teacher_action=np.asarray([[4], [5]], dtype=np.int32),
        teacher_valid=np.asarray([[True], [False]], dtype=np.bool_),
    )
    unlabeled = _unroll(width=2)

    required = concat_required_time_major_fields(
        [labeled, unlabeled],
        (RequiredTimeMajorFieldSpec("actions"), RequiredTimeMajorFieldSpec("policy_train_mask")),
    )
    optional = concat_optional_time_major_fields(
        [labeled, unlabeled],
        (
            OptionalTimeMajorFieldSpec("teacher_action", -1),
            OptionalTimeMajorFieldSpec("teacher_valid", False),
        ),
    )

    assert required["actions"].dtype == np.int64
    assert required["actions"].tolist() == [[0, 0, 1], [1, 2, 3]]
    assert required["policy_train_mask"].tolist() == [[True, True, True], [True, True, True]]
    assert optional["teacher_action"] is not None
    assert optional["teacher_action"].tolist() == [[4, -1, -1], [5, -1, -1]]
    assert optional["teacher_valid"] is not None
    assert optional["teacher_valid"].tolist() == [[True, False, False], [False, False, False]]


def test_time_major_layout_is_shared_across_required_and_optional_fields() -> None:
    first = _unroll(
        width=2,
        teacher_action=np.asarray([[10, 11], [12, 13]], dtype=np.int32),
    )
    second = _unroll(width=1)
    layout = time_major_batch_layout([first, second], "obs")

    actions = concat_required_time_major_fields(
        [first, second],
        (RequiredTimeMajorFieldSpec("actions"),),
    )["actions"]
    teacher_action = concat_optional_time_major_fields(
        [first, second],
        (OptionalTimeMajorFieldSpec("teacher_action", -1),),
    )["teacher_action"]

    assert layout.batch_offsets == (0, 2, 3)
    assert layout.total_batch == 3
    assert actions.tolist() == [[0, 1, 0], [2, 3, 1]]
    assert teacher_action is not None
    assert teacher_action.tolist() == [[10, 11, -1], [12, 13, -1]]


def test_optional_time_major_layout_rejects_labels_with_wrong_batch_width() -> None:
    labeled = _unroll(
        width=2,
        teacher_action=np.asarray([[4], [5]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"teacher_action must have leading shape \(2, 2\)"):
        concat_optional_time_major_fields(
            [labeled],
            (OptionalTimeMajorFieldSpec("teacher_action", -1),),
        )


def test_concat_runtime_batch_fields_leaves_absent_auxiliary_labels_as_none() -> None:
    fields = concat_runtime_batch_fields(
        [_unroll(width=1), _unroll(width=2)],
        action_dim=3,
        record_batch_timer_ms=None,
    )

    assert fields.teacher_action is None
    assert fields.teacher_valid is None
    assert fields.trajectory_retention_valid is None
    assert fields.opponent_context_index is None


def test_concat_runtime_batch_fields_records_legal_concatenation_timer() -> None:
    timer_events: list[tuple[str, float]] = []

    fields = concat_runtime_batch_fields(
        [_unroll(width=1)],
        action_dim=3,
        record_batch_timer_ms=lambda name, elapsed: timer_events.append((name, elapsed)),
    )

    assert fields.legal_mask is not None
    assert len(timer_events) == 1
    assert timer_events[0][0] == "legal_concatenation"
    assert timer_events[0][1] >= 0.0


def test_base_runtime_learner_payload_preserves_actor_alias_and_auxiliary_arrays() -> None:
    fields = concat_runtime_batch_fields(
        [_unroll(width=1, opponent_context_index=np.asarray([[7], [8]], dtype=np.int16))],
        action_dim=3,
        record_batch_timer_ms=None,
    )
    discounts = np.ones((2, 1), dtype=np.float32)
    reset_before_step = np.asarray([[False], [True]], dtype=np.bool_)

    payload = base_runtime_learner_payload(
        fields=fields,
        rewards=fields.rewards,
        discounts=discounts,
        reset_before_step=reset_before_step,
    )

    assert payload["actor"] is fields.to_play_seat
    assert payload["to_play_seat"] is fields.to_play_seat
    assert payload["opponent_context_index"].dtype == np.int16
    assert payload["opponent_context_index"].tolist() == [[7], [8]]
    assert payload["discounts"] is discounts
    assert payload["reset_before_step"] is reset_before_step


def test_concat_time_major_field_rejects_empty_unrolls() -> None:
    try:
        concat_time_major_field([], "obs")
    except ValueError as exc:
        assert str(exc) == "unrolls must be non-empty"
    else:
        raise AssertionError("empty unroll concatenation should fail")
