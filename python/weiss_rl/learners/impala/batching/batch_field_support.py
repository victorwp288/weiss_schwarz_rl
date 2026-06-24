"""IMPALA learner batch field and tensor conversion helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

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


class ImpalaBatchFieldSupportMixin:
    """Convert optional batch fields onto the learner model device."""

    def _float_target(self: Any, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        return float_target(value, expected_shape=expected_shape, like=like, reference=self._model_parameter())

    def _optional_batch_seat_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_batch_size: int,
    ) -> Tensor | None:
        return optional_batch_seat_field(
            value,
            field_name=field_name,
            expected_batch_size=expected_batch_size,
            reference=self._model_parameter(),
        )

    def _prepare_legacy_hidden_state(self: Any, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        return prepare_legacy_hidden_state(value, batch_size=batch_size, like=like, reference=self._model_parameter())

    def _prepare_seat_hidden_state(self: Any, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        return prepare_seat_hidden_state(value, batch_size=batch_size, like=like, reference=self._model_parameter())

    def _prepare_acting_seat_batch(
        self: Any,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_seat_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_seat_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_loss_mask(
        self: Any,
        value: Any,
        *,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        return optional_time_major_loss_mask(
            value,
            expected_shape=expected_shape,
            like=like,
            reference=self._model_parameter(),
        )

    def _optional_time_major_index_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_index_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_float_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        return optional_time_major_float_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            like=like,
            reference=self._model_parameter(),
        )

    def _optional_time_major_bool_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_bool_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _float_input(self: Any, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self: Any, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self: Any, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self: Any, value: Any, *, dtype: torch.dtype) -> Tensor:
        return tensor_on_device(value, reference=self._model_parameter(), dtype=dtype)

    def _model_parameter(self: Any) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter


__all__ = ["ImpalaBatchFieldSupportMixin"]
