"""Initial IMPALA loss batch-input resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaLossBatchInputs:
    vtrace_result: Any
    obs: Tensor
    actions: Tensor
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None
    forward_model: Any


def resolve_impala_loss_batch_inputs(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
) -> ImpalaLossBatchInputs:
    vtrace_result = batch_value(batch, "vtrace_result")
    obs = learner._require_obs(batch_value(batch, "obs"))
    actions = learner._require_actions(batch_value(batch, "actions"), expected_shape=obs.shape[:2])
    packed_legal = learner._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
    forward_model = learner.compiled_model if learner.compiled_model is not None else learner.model
    return ImpalaLossBatchInputs(
        vtrace_result=vtrace_result,
        obs=obs,
        actions=actions,
        packed_legal=packed_legal,
        forward_model=forward_model,
    )


__all__ = ["ImpalaLossBatchInputs", "resolve_impala_loss_batch_inputs"]
