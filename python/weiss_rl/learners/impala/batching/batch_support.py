"""IMPALA learner batch field, legality, and bootstrap support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.bootstrap import (
    current_model_bootstrap_value,
    has_raw_vtrace_inputs,
    resolve_vtrace_bootstrap_value,
)
from weiss_rl.learners.impala.batching.batch_access import batch_value as _batch_value
from weiss_rl.learners.impala.batching.batch_field_support import ImpalaBatchFieldSupportMixin
from weiss_rl.learners.legal_fields import (
    has_legal_actions,
    require_actions,
    require_legal_mask,
    require_obs,
    resolve_legal_mask,
    resolve_packed_legal_actions_with_meta,
)
from weiss_rl.learners.packed_rows import (
    packed_candidate_positions_for_rows,
    packed_legal_action_view,
    scatter_packed_candidate_values,
    slice_packed_legal_rows_with_meta,
    subset_observation_context_rows,
)


class ImpalaBatchSupportMixin(ImpalaBatchFieldSupportMixin):
    def _require_obs(self: Any, value: Any) -> Tensor:
        return require_obs(value, reference=self._model_parameter())

    def _require_actions(self: Any, value: Any, *, expected_shape: torch.Size) -> Tensor:
        return require_actions(value, expected_shape=expected_shape, reference=self._model_parameter())

    def _require_legal_mask(self: Any, value: Any, *, expected_shape: torch.Size) -> Tensor:
        return require_legal_mask(value, expected_shape=expected_shape, reference=self._model_parameter())

    def _has_legal_actions(self: Any, batch: Any) -> bool:
        return has_legal_actions(batch, batch_value=_batch_value)

    def _resolve_legal_mask(self: Any, batch: Any, *, expected_shape: torch.Size, action_dim: int) -> Tensor:
        return resolve_legal_mask(
            batch,
            expected_shape=expected_shape,
            action_dim=action_dim,
            reference=self._model_parameter(),
            batch_value=_batch_value,
        )

    def _resolve_packed_legal_actions(
        self: Any,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor] | None:
        resolved = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if resolved is None:
            return None
        return resolved[0], resolved[1]

    def _resolve_packed_legal_actions_with_meta(
        self: Any,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        return resolve_packed_legal_actions_with_meta(
            batch,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
            batch_value=_batch_value,
            supports_legal_candidate_scoring=bool(getattr(self.model, "supports_legal_candidate_scoring", False)),
        )

    def _packed_legal_action_view(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
    ) -> Any:
        return packed_legal_action_view(packed_legal)

    def _slice_packed_legal_rows_with_meta(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        return slice_packed_legal_rows_with_meta(packed_legal, row_indices)

    def _packed_candidate_positions_for_rows(
        self: Any,
        offsets: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        return packed_candidate_positions_for_rows(offsets, row_indices)

    def _scatter_packed_candidate_values(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
        subset_values: Tensor,
        *,
        fill_value: float = 0.0,
    ) -> Tensor:
        return scatter_packed_candidate_values(packed_legal, row_indices, subset_values, fill_value=fill_value)

    def _subset_observation_context_rows(
        self: Any,
        observation_context: Mapping[str, Tensor],
        row_indices: Tensor,
        *,
        row_count: int,
    ) -> dict[str, Tensor]:
        return subset_observation_context_rows(observation_context, row_indices, row_count=row_count)

    def _has_raw_vtrace_inputs(self: Any, batch: Any) -> bool:
        return has_raw_vtrace_inputs(batch, batch_value=_batch_value)

    def _resolve_vtrace_bootstrap_value(
        self: Any,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor:
        return resolve_vtrace_bootstrap_value(
            batch,
            batch_size=batch_size,
            like=like,
            model=self.model,
            compiled_model=self.compiled_model,
            reference_parameter=self._model_parameter,
            batch_value=_batch_value,
        )

    def _current_model_bootstrap_value(
        self: Any,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor | None:
        return current_model_bootstrap_value(
            batch,
            batch_size=batch_size,
            like=like,
            model=self.model,
            compiled_model=self.compiled_model,
            reference_parameter=self._model_parameter,
            batch_value=_batch_value,
        )


__all__ = ["ImpalaBatchFieldSupportMixin", "ImpalaBatchSupportMixin", "_batch_value"]
