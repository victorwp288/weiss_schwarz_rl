"""IMPALA final loss-context stage orchestration."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_finalization import finalize_impala_loss_context
from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs
from weiss_rl.learners.impala.loss_policy_anchor_stage import ImpalaPolicyAnchorStage
from weiss_rl.learners.impala.objective_loss import ImpalaObjectiveLosses


def finalize_impala_loss_context_stage(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    total_loss: Tensor,
    objective_losses: ImpalaObjectiveLosses,
    policy_anchor_stage: ImpalaPolicyAnchorStage,
) -> dict[str, Any]:
    finalize_impala_loss_context(
        learner=learner,
        batch=batch,
        context=inputs.context,
        policy_loss=objective_losses.policy_loss,
        value_loss=objective_losses.value_loss,
        entropy_mean=objective_losses.entropy_mean,
        total_loss=total_loss,
        policy_anchor_loss=policy_anchor_stage.policy_anchor_loss,
        factorized_result=inputs.factorized_result,
    )
    return inputs.context


__all__ = ["finalize_impala_loss_context_stage"]
