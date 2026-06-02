"""Shared batch normalization for IMPALA paired auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

PackedLegalWithMeta = tuple[Tensor, Tensor, Tensor | None]


@dataclass(frozen=True)
class PairedAuxiliaryBatchInputs:
    obs: Tensor
    expected_shape: torch.Size
    packed_legal: PackedLegalWithMeta
    loss_mask: Tensor


def batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def resolve_paired_auxiliary_batch_inputs(
    learner: Any,
    batch: Any,
    *,
    packed_legal_error: str,
) -> PairedAuxiliaryBatchInputs:
    """Resolve the common time-major batch contract used by paired replay losses."""

    obs = learner._require_obs(batch_value(batch, "obs"))
    expected_shape = obs.shape[:2]
    packed_legal = learner._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
    if packed_legal is None:
        raise ValueError(packed_legal_error)
    loss_mask = learner._optional_time_major_loss_mask(
        batch_value(batch, "policy_train_mask"),
        expected_shape=expected_shape,
        like=obs[..., 0],
    )
    if loss_mask is None:
        loss_mask = torch.ones(expected_shape, device=obs.device, dtype=obs.dtype)
    return PairedAuxiliaryBatchInputs(
        obs=obs,
        expected_shape=expected_shape,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
    )


def resolve_paired_auxiliary_reset_before_step(
    learner: Any,
    batch: Any,
    *,
    expected_shape: torch.Size,
) -> Tensor | None:
    return learner._optional_time_major_bool_field(
        batch_value(batch, "reset_before_step"),
        field_name="reset_before_step",
        expected_shape=expected_shape,
    )


__all__ = [
    "PackedLegalWithMeta",
    "PairedAuxiliaryBatchInputs",
    "batch_value",
    "resolve_paired_auxiliary_batch_inputs",
    "resolve_paired_auxiliary_reset_before_step",
]
