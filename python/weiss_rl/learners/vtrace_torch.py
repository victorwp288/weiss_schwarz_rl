"""Torch V-trace target computation used by the IMPALA learner."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

_MAX_LOG_RHO_TORCH = float(np.log(np.finfo(np.float32).max))


def _importance_weights_from_logp(behavior_logp: Tensor, target_logp: Tensor) -> Tensor:
    log_rhos = torch.clamp(target_logp - behavior_logp, max=_MAX_LOG_RHO_TORCH)
    return torch.exp(log_rhos).clamp(max=torch.finfo(torch.float32).max)


def _clip_importance_weights(rhos: Tensor, *, rho_bar: float, c_bar: float) -> tuple[Tensor, Tensor]:
    rho_bar_tensor = torch.full((), float(rho_bar), dtype=rhos.dtype, device=rhos.device)
    c_bar_tensor = torch.full((), float(c_bar), dtype=rhos.dtype, device=rhos.device)
    return torch.minimum(rho_bar_tensor, rhos), torch.minimum(c_bar_tensor, rhos)


def compute_vtrace_targets_torch(
    rewards: Tensor,
    values: Tensor,
    discounts: Tensor,
    behavior_logp: Tensor,
    target_logp: Tensor,
    *,
    rho_bar: float,
    c_bar: float,
) -> tuple[Tensor, Tensor, Tensor]:
    rewards = rewards.detach()
    values = values.detach()
    discounts = discounts.detach()
    behavior_logp = behavior_logp.detach()
    target_logp = target_logp.detach()
    rhos = _importance_weights_from_logp(behavior_logp=behavior_logp, target_logp=target_logp)
    clipped_rhos, clipped_cs = _clip_importance_weights(rhos, rho_bar=rho_bar, c_bar=c_bar)

    acc = torch.zeros_like(values[-1])
    vs_minus_v_xs = torch.zeros_like(rewards)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = clipped_rhos[t] * (rewards[t] + discounts[t] * values[t + 1] - values[t])
        acc = delta + discounts[t] * clipped_cs[t] * acc
        vs_minus_v_xs[t] = acc

    vs = values[:-1] + vs_minus_v_xs
    next_vs = torch.cat((vs[1:], values[-1:].clone()), dim=0)
    pg_advantages = clipped_rhos * (rewards + discounts * next_vs - values[:-1])
    return vs, pg_advantages, rhos
