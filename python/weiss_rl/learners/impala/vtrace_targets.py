"""IMPALA learner V-trace target resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_torch import compute_vtrace_targets_torch

BatchValueGetter = Callable[[Any, str], Any]


class FloatTargetConverter(Protocol):
    def __call__(self, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor: ...


class BootstrapValueResolver(Protocol):
    def __call__(self, batch: Any, *, batch_size: int, like: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class ImpalaResolvedVTraceTargets:
    action_logp: Tensor
    behavior_logp_for_mask: Tensor | None
    targets: Tensor
    advantages: Tensor
    rhos_for_metrics: Tensor
    rewards_for_metrics: Tensor


def resolve_impala_vtrace_targets(
    *,
    batch: Any,
    vtrace_result: Any,
    values: Tensor,
    action_logp: Tensor,
    loss_mask: Tensor,
    rho_bar: float,
    c_bar: float,
    float_target: FloatTargetConverter,
    resolve_bootstrap_value: BootstrapValueResolver,
    batch_value: BatchValueGetter,
) -> ImpalaResolvedVTraceTargets:
    raw_behavior_logp = batch_value(batch, "behavior_logp")
    behavior_logp_for_mask = None
    if raw_behavior_logp is not None:
        behavior_logp_for_mask = float_target(
            raw_behavior_logp,
            expected_shape=values.shape,
            like=values,
        )

    if behavior_logp_for_mask is not None and bool((loss_mask <= 0.0).any().item()):
        action_logp = torch.where(loss_mask > 0.0, action_logp, behavior_logp_for_mask)

    if isinstance(vtrace_result, VTraceTargets):
        targets = float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
        advantages = float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)
        rhos_for_metrics = float_target(vtrace_result.rhos, expected_shape=values.shape, like=values)
        raw_rewards = batch_value(batch, "rewards")
        if raw_rewards is None:
            rewards_for_metrics = torch.zeros_like(values)
        else:
            rewards_for_metrics = float_target(raw_rewards, expected_shape=values.shape, like=values)
    else:
        rewards = float_target(batch_value(batch, "rewards"), expected_shape=values.shape, like=values)
        discounts = float_target(batch_value(batch, "discounts"), expected_shape=values.shape, like=values)
        if behavior_logp_for_mask is None:
            raise ValueError("raw V-trace batches must include behavior_logp")
        bootstrap_value = resolve_bootstrap_value(
            batch,
            batch_size=int(values.shape[1]),
            like=values,
        )
        full_values = torch.cat([values.detach(), bootstrap_value.detach().unsqueeze(0)], dim=0)
        # Use the current learner policy log-prob for the V-trace target policy.
        # Passing behavior_logp twice silently forces rho=1 and disables off-policy correction.
        targets, advantages, rhos_for_metrics = compute_vtrace_targets_torch(
            rewards,
            full_values,
            discounts,
            behavior_logp_for_mask,
            action_logp,
            rho_bar=rho_bar,
            c_bar=c_bar,
        )
        rewards_for_metrics = rewards

    return ImpalaResolvedVTraceTargets(
        action_logp=action_logp,
        behavior_logp_for_mask=behavior_logp_for_mask,
        targets=targets.detach(),
        advantages=advantages.detach(),
        rhos_for_metrics=rhos_for_metrics.detach(),
        rewards_for_metrics=rewards_for_metrics,
    )


__all__ = ["ImpalaResolvedVTraceTargets", "resolve_impala_vtrace_targets"]
