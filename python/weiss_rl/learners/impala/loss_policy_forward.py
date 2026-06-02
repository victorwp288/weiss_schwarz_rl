"""Policy-forward evaluation for IMPALA loss-input preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaPolicyForwardResult:
    factorized_result: Any
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None
    logits: Tensor | None
    packed_logits: Tensor | None
    values: Tensor
    forward_observation_context: Mapping[str, Tensor] | None


def evaluate_impala_policy_forward(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
    forward_model: Any,
    obs: Tensor,
    actions: Tensor,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    loss_mask: Tensor,
    reset_before_step: Tensor | None,
    trajectory_retention_active: Tensor | None,
    restrict_packed_policy_rows: bool,
) -> ImpalaPolicyForwardResult:
    factorized_result = None
    forward_observation_context: Mapping[str, Tensor] | None = None
    if learner._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
        factorized_result, packed_legal = learner._evaluate_factorized_time_major(
            batch,
            obs=obs,
            actions=actions,
            extra_active_mask=trajectory_retention_active,
        )
        return ImpalaPolicyForwardResult(
            factorized_result=factorized_result,
            packed_legal=packed_legal,
            logits=None,
            packed_logits=None,
            values=factorized_result.values,
            forward_observation_context=forward_observation_context,
        )

    packed_forward_mask = loss_mask
    if trajectory_retention_active is not None:
        packed_forward_mask = torch.logical_or(
            loss_mask > 0.0,
            trajectory_retention_active,
        ).to(dtype=loss_mask.dtype)
    forward = learner._forward_time_major(
        obs,
        initial_hidden_state=batch_value(batch, "initial_hidden_state"),
        to_play_seat=batch_value(batch, "to_play_seat"),
        actor=batch_value(batch, "actor"),
        legal_actions=batch_value(batch, "legal_actions"),
        policy_train_mask=packed_forward_mask if restrict_packed_policy_rows else None,
        reset_before_step=reset_before_step,
        opponent_context_index=batch_value(batch, "opponent_context_index"),
    )
    return ImpalaPolicyForwardResult(
        factorized_result=factorized_result,
        packed_legal=packed_legal,
        logits=forward.logits,
        packed_logits=forward.packed_logits,
        values=forward.values,
        forward_observation_context=forward.observation_context,
    )


__all__ = ["ImpalaPolicyForwardResult", "evaluate_impala_policy_forward"]
