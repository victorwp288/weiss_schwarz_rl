"""Mask and forward-flag resolution for IMPALA loss inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaLossMasks:
    loss_mask: Tensor
    reset_before_step: Tensor | None
    trajectory_retention_valid: Tensor | None
    trajectory_retention_active: Tensor | None


@dataclass(frozen=True, slots=True)
class ImpalaLossForwardFlags:
    teacher_aux_active: bool
    emit_structured_metrics: bool
    restrict_packed_policy_rows: bool


def resolve_impala_loss_masks(
    *,
    learner: Any,
    batch: Any,
    obs: Tensor,
    batch_value: BatchValueGetter,
) -> ImpalaLossMasks:
    loss_mask = _resolve_policy_loss_mask(learner=learner, batch=batch, obs=obs, batch_value=batch_value)
    reset_before_step = learner._optional_time_major_loss_mask(
        batch_value(batch, "reset_before_step"),
        expected_shape=obs.shape[:2],
        like=obs[..., 0],
    )
    if reset_before_step is not None:
        reset_before_step = reset_before_step.to(dtype=torch.bool)
    trajectory_retention_valid = learner._optional_time_major_loss_mask(
        batch_value(batch, "trajectory_retention_valid"),
        expected_shape=obs.shape[:2],
        like=obs[..., 0],
    )
    trajectory_retention_active = (
        None
        if trajectory_retention_valid is None or float(learner.trajectory_retention_coef) == 0.0
        else trajectory_retention_valid.to(dtype=torch.bool)
    )
    return ImpalaLossMasks(
        loss_mask=loss_mask,
        reset_before_step=reset_before_step,
        trajectory_retention_valid=trajectory_retention_valid,
        trajectory_retention_active=trajectory_retention_active,
    )


def resolve_impala_loss_forward_flags(
    *,
    learner: Any,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    loss_mask: Tensor,
) -> ImpalaLossForwardFlags:
    teacher_aux_active = isinstance(
        getattr(learner.model, "action_catalog", None),
        ActionCatalog,
    ) and learner._teacher_aux_active(auxiliary_update=False)
    emit_structured_metrics = learner._should_emit_structured_metrics(auxiliary_update=False)
    restrict_packed_policy_rows = bool(
        packed_legal is not None
        and bool((loss_mask <= 0.0).any().item())
        and not teacher_aux_active
        and not emit_structured_metrics
    )
    return ImpalaLossForwardFlags(
        teacher_aux_active=teacher_aux_active,
        emit_structured_metrics=emit_structured_metrics,
        restrict_packed_policy_rows=restrict_packed_policy_rows,
    )


def _resolve_policy_loss_mask(*, learner: Any, batch: Any, obs: Tensor, batch_value: BatchValueGetter) -> Tensor:
    loss_mask = learner._optional_time_major_loss_mask(
        batch_value(batch, "policy_train_mask"),
        expected_shape=obs.shape[:2],
        like=obs[..., 0],
    )
    if loss_mask is None:
        loss_mask = torch.ones(obs.shape[:2], dtype=obs.dtype, device=obs.device)
    return loss_mask


__all__ = [
    "ImpalaLossForwardFlags",
    "ImpalaLossMasks",
    "resolve_impala_loss_forward_flags",
    "resolve_impala_loss_masks",
]
