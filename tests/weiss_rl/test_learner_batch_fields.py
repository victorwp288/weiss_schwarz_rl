from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.batch_fields import (
    float_target,
    optional_batch_seat_field,
    optional_time_major_bool_field,
    optional_time_major_float_field,
    optional_time_major_index_field,
    optional_time_major_loss_mask,
    optional_time_major_seat_field,
    prepare_acting_seat_batch,
    prepare_legacy_hidden_state,
    prepare_seat_hidden_state,
    tensor_on_device,
)


def _reference() -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32)


def test_tensor_on_device_requires_value_and_converts_dtype() -> None:
    converted = tensor_on_device([1, 2], reference=_reference(), dtype=torch.float64)

    assert converted.dtype == torch.float64
    torch.testing.assert_close(converted, torch.tensor([1.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="batch field is required"):
        tensor_on_device(None, reference=_reference(), dtype=torch.float32)


def test_float_target_validates_shape_and_uses_like_dtype() -> None:
    like = torch.zeros((2, 1), dtype=torch.float64)

    target = float_target([[1.0], [2.0]], expected_shape=torch.Size((2, 1)), like=like, reference=_reference())

    assert target.dtype == torch.float64
    torch.testing.assert_close(target, torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match=r"target must have shape \(2, 1\), got \(2,\)"):
        float_target([1.0, 2.0], expected_shape=torch.Size((2, 1)), like=like, reference=_reference())


def test_optional_batch_seat_field_validates_integer_shape_and_binary_values() -> None:
    assert optional_batch_seat_field(None, field_name="actor", expected_batch_size=2, reference=_reference()) is None
    seat = optional_batch_seat_field([0, 1], field_name="actor", expected_batch_size=2, reference=_reference())
    assert seat is not None
    assert seat.tolist() == [
        0,
        1,
    ]

    with pytest.raises(ValueError, match="actor must be integer-valued"):
        optional_batch_seat_field([0.0, 1.0], field_name="actor", expected_batch_size=2, reference=_reference())
    with pytest.raises(ValueError, match=r"actor must have shape \(2,\), got \(1, 2\)"):
        optional_batch_seat_field([[0, 1]], field_name="actor", expected_batch_size=2, reference=_reference())
    with pytest.raises(ValueError, match="actor values must be 0 or 1"):
        optional_batch_seat_field([0, 2], field_name="actor", expected_batch_size=2, reference=_reference())


def test_hidden_state_helpers_validate_legacy_and_seat_aware_shapes() -> None:
    like = torch.zeros((), dtype=torch.float64)

    legacy = prepare_legacy_hidden_state([[1, 2], [3, 4]], batch_size=2, like=like, reference=_reference())
    assert legacy is not None and legacy.dtype == torch.float64
    assert prepare_legacy_hidden_state(None, batch_size=2, like=like, reference=_reference()) is None
    with pytest.raises(ValueError, match="must be 2D"):
        prepare_legacy_hidden_state([1, 2], batch_size=2, like=like, reference=_reference())
    with pytest.raises(ValueError, match="batch mismatch: expected 3, got 2"):
        prepare_legacy_hidden_state([[1, 2], [3, 4]], batch_size=3, like=like, reference=_reference())

    seat = prepare_seat_hidden_state(torch.zeros((2, 2, 3)), batch_size=2, like=like, reference=_reference())
    assert seat is not None and seat.shape == (2, 2, 3)
    assert prepare_seat_hidden_state(None, batch_size=2, like=like, reference=_reference()) is None
    with pytest.raises(ValueError, match="must be 3D"):
        prepare_seat_hidden_state(torch.zeros((2, 3)), batch_size=2, like=like, reference=_reference())
    with pytest.raises(ValueError, match="seat mismatch: expected 2, got 3"):
        prepare_seat_hidden_state(torch.zeros((2, 3, 4)), batch_size=2, like=like, reference=_reference())


def test_time_major_seat_and_acting_seat_helpers() -> None:
    expected_shape = torch.Size((2, 2))

    seat = optional_time_major_seat_field(
        [[0, 1], [1, 0]],
        field_name="to_play_seat",
        expected_shape=expected_shape,
        reference=_reference(),
    )
    assert seat is not None and seat.tolist() == [[0, 1], [1, 0]]
    assert (
        optional_time_major_seat_field(
            None,
            field_name="actor",
            expected_shape=expected_shape,
            reference=_reference(),
        )
        is None
    )

    with pytest.raises(ValueError, match="to_play_seat must be integer-valued"):
        optional_time_major_seat_field(
            [[0.0, 1.0], [1.0, 0.0]],
            field_name="to_play_seat",
            expected_shape=expected_shape,
            reference=_reference(),
        )
    with pytest.raises(ValueError, match="to_play_seat values must be 0 or 1"):
        optional_time_major_seat_field(
            [[0, 2], [1, 0]],
            field_name="to_play_seat",
            expected_shape=expected_shape,
            reference=_reference(),
        )
    with pytest.raises(ValueError, match="actor must match to_play_seat"):
        prepare_acting_seat_batch(
            [[0, 1], [1, 0]],
            actor=[[0, 1], [0, 1]],
            expected_shape=expected_shape,
            reference=_reference(),
        )
    acting_seat = prepare_acting_seat_batch(
        None,
        actor=[[0, 1], [1, 0]],
        expected_shape=expected_shape,
        reference=_reference(),
    )
    assert acting_seat is not None
    assert acting_seat.tolist() == [[0, 1], [1, 0]]


def test_time_major_loss_mask_index_and_bool_fields() -> None:
    expected_shape = torch.Size((2, 2))
    like = torch.zeros(expected_shape)

    mask = optional_time_major_loss_mask(
        [[-1.0, 0.5], [2.0, 1.0]],
        expected_shape=expected_shape,
        like=like,
        reference=_reference(),
    )
    assert mask is not None
    torch.testing.assert_close(mask, torch.tensor([[0.0, 0.5], [1.0, 1.0]]))
    assert optional_time_major_loss_mask(None, expected_shape=expected_shape, like=like, reference=_reference()) is None
    with pytest.raises(ValueError, match=r"policy_train_mask must have shape \(2, 2\), got \(2,\)"):
        optional_time_major_loss_mask([1.0, 2.0], expected_shape=expected_shape, like=like, reference=_reference())

    weights = optional_time_major_float_field(
        [[0.25, 2.0], [8.0, 1.0]],
        field_name="preference_pair_weight",
        expected_shape=expected_shape,
        like=like,
        reference=_reference(),
    )
    assert weights is not None
    torch.testing.assert_close(weights, torch.tensor([[0.25, 2.0], [8.0, 1.0]]))
    assert (
        optional_time_major_float_field(
            None,
            field_name="preference_pair_weight",
            expected_shape=expected_shape,
            like=like,
            reference=_reference(),
        )
        is None
    )
    with pytest.raises(ValueError, match=r"preference_pair_weight must have shape \(2, 2\), got \(2,\)"):
        optional_time_major_float_field(
            [1.0, 2.0],
            field_name="preference_pair_weight",
            expected_shape=expected_shape,
            like=like,
            reference=_reference(),
        )

    indices = optional_time_major_index_field(
        [[1, 2], [3, 4]],
        field_name="teacher_action",
        expected_shape=expected_shape,
        reference=_reference(),
    )
    assert indices is not None and indices.dtype == torch.long
    with pytest.raises(ValueError, match="teacher_action must be integer-valued"):
        optional_time_major_index_field(
            [[1.0, 2.0], [3.0, 4.0]],
            field_name="teacher_action",
            expected_shape=expected_shape,
            reference=_reference(),
        )

    bools = optional_time_major_bool_field(
        [[1, 0], [0, 1]],
        field_name="teacher_valid",
        expected_shape=expected_shape,
        reference=_reference(),
    )
    assert bools is not None and bools.tolist() == [[True, False], [False, True]]
    with pytest.raises(ValueError, match=r"teacher_valid must have shape \(2, 2\), got \(2,\)"):
        optional_time_major_bool_field(
            [1, 0],
            field_name="teacher_valid",
            expected_shape=expected_shape,
            reference=_reference(),
        )
