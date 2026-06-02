"""IMPALA policy-anchor loss-stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs


@dataclass(frozen=True, slots=True)
class ImpalaPolicyAnchorStage:
    total_loss: Tensor
    policy_anchor_loss: Tensor | None
    policy_anchor_metrics: dict[str, float]


def apply_impala_policy_anchor_stage(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    total_loss: Tensor,
) -> ImpalaPolicyAnchorStage:
    policy_anchor_loss, policy_anchor_metrics = learner._policy_anchor_loss_and_metrics(
        batch,
        obs=inputs.obs,
        loss_mask=inputs.loss_mask,
        packed_legal=inputs.packed_legal,
        factorized_result=inputs.factorized_result,
        forward_model=inputs.forward_model,
        reset_before_step=inputs.reset_before_step,
    )
    if policy_anchor_loss is not None:
        total_loss = total_loss + policy_anchor_loss
    return ImpalaPolicyAnchorStage(
        total_loss=total_loss,
        policy_anchor_loss=policy_anchor_loss,
        policy_anchor_metrics=policy_anchor_metrics,
    )


__all__ = [
    "ImpalaPolicyAnchorStage",
    "apply_impala_policy_anchor_stage",
]
