"""IMPALA teacher-auxiliary application and loss context finalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala.teacher_auxiliary_request import compute_impala_teacher_auxiliary

BatchValueGetter = Callable[[Any, str], Any]
DenseMaskResolver = Callable[[Any, torch.Size, int], Tensor]


@dataclass(frozen=True, slots=True)
class ImpalaLossFinalization:
    total_loss: Tensor
    teacher_metrics: dict[str, float]


def apply_impala_teacher_auxiliary(
    *,
    learner: Any,
    batch: Any,
    total_loss: Tensor,
    context: dict[str, Any],
    teacher_aux_active: bool,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    loss_mask: Tensor,
    action_catalog: Any,
    expected_shape: torch.Size,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    packed_view: Any,
    factorized_result: Any,
    public_heuristic_target_logits: Tensor | None,
    resolve_legal_mask: DenseMaskResolver,
    batch_value: BatchValueGetter,
) -> ImpalaLossFinalization:
    teacher_metrics: dict[str, float] = {}
    if not teacher_aux_active:
        return ImpalaLossFinalization(total_loss=total_loss, teacher_metrics=teacher_metrics)

    assert isinstance(action_catalog, ActionCatalog)
    structured_legal_mask = _resolve_teacher_aux_legal_mask(
        batch=batch,
        logits=logits,
        legal_mask=legal_mask,
        expected_shape=expected_shape,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        resolve_legal_mask=resolve_legal_mask,
    )
    teacher_aux_result = compute_impala_teacher_auxiliary(
        learner=learner,
        batch=batch,
        logits=logits,
        legal_mask=structured_legal_mask,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        expected_shape=expected_shape,
        packed_legal=packed_legal,
        packed_view=packed_view,
        factorized_result=factorized_result,
        public_heuristic_target_logits=public_heuristic_target_logits,
        batch_value=batch_value,
    )
    context.update(teacher_aux_result.context)
    return ImpalaLossFinalization(
        total_loss=total_loss + teacher_aux_result.loss,
        teacher_metrics=teacher_aux_result.metrics,
    )


def finalize_impala_loss_context(
    *,
    learner: Any,
    batch: Any,
    context: dict[str, Any],
    policy_loss: Tensor,
    value_loss: Tensor,
    entropy_mean: Tensor,
    total_loss: Tensor,
    policy_anchor_loss: Tensor | None,
    factorized_result: Any,
) -> None:
    context["policy_loss"] = policy_loss.detach()
    context["value_loss"] = value_loss.detach()
    context["entropy_mean"] = entropy_mean.detach()
    if policy_anchor_loss is not None:
        context["policy_anchor_loss"] = policy_anchor_loss.detach()
    context["total_loss"] = total_loss.detach()
    if factorized_result is not None:
        context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
    learner._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
    learner._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
    learner._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
    learner._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)


def _resolve_teacher_aux_legal_mask(
    *,
    batch: Any,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    expected_shape: torch.Size,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    factorized_result: Any,
    resolve_legal_mask: DenseMaskResolver,
) -> Tensor | None:
    if factorized_result is not None:
        return None
    if legal_mask is not None:
        return legal_mask
    if packed_legal is not None and packed_legal[2] is not None:
        return None
    return resolve_legal_mask(batch, expected_shape, cast(Tensor, logits).shape[-1])


__all__ = ["ImpalaLossFinalization", "apply_impala_teacher_auxiliary", "finalize_impala_loss_context"]
