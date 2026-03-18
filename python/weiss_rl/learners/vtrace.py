"""V-trace targets for IMPALA-style off-policy correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VTraceTargets:
    vs: np.ndarray
    pg_advantages: np.ndarray
    rhos: np.ndarray


def _validate_time_major_inputs(
    rewards: np.ndarray,
    values: np.ndarray,
    discounts: np.ndarray,
    behavior_logp: np.ndarray,
    target_logp: np.ndarray,
) -> None:
    if rewards.shape != discounts.shape:
        raise ValueError("rewards and discounts must have identical shapes")
    if rewards.shape != behavior_logp.shape or rewards.shape != target_logp.shape:
        raise ValueError("rewards, behavior_logp, and target_logp must have identical shapes")
    if values.ndim != rewards.ndim:
        raise ValueError("values must have the same rank as rewards")
    if values.shape[1:] != rewards.shape[1:]:
        raise ValueError("values must match rewards on all non-time dimensions")
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must have one extra bootstrap step on the time axis")


def _compute_vtrace_from_rhos(
    rewards: np.ndarray,
    values: np.ndarray,
    discounts: np.ndarray,
    rhos: np.ndarray,
    *,
    rho_bar: float,
    c_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards64 = np.asarray(rewards, dtype=np.float64)
    values64 = np.asarray(values, dtype=np.float64)
    discounts64 = np.asarray(discounts, dtype=np.float64)
    rhos64 = np.asarray(rhos, dtype=np.float64)

    clipped_rhos = np.minimum(rho_bar, rhos64)
    clipped_cs = np.minimum(c_bar, rhos64)

    vs_minus_v_xs = np.zeros_like(rewards64, dtype=np.float64)
    acc = np.zeros_like(values64[-1], dtype=np.float64)
    for t in range(rewards64.shape[0] - 1, -1, -1):
        delta = clipped_rhos[t] * (rewards64[t] + discounts64[t] * values64[t + 1] - values64[t])
        acc = delta + discounts64[t] * clipped_cs[t] * acc
        vs_minus_v_xs[t] = acc

    vs = values64[:-1] + vs_minus_v_xs
    next_vs = np.concatenate((vs[1:], values64[-1:]), axis=0)
    pg_advantages = clipped_rhos * (rewards64 + discounts64 * next_vs - values64[:-1])
    return vs, pg_advantages


def compute_vtrace_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    discounts: np.ndarray,
    behavior_logp: np.ndarray,
    target_logp: np.ndarray,
    *,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> VTraceTargets:
    """Compute time-major V-trace value targets and policy advantages.

    Args:
        rewards: Time-major rewards with shape ``[T, B...]``.
        values: Baseline values with shape ``[T + 1, B...]``.
        discounts: Per-step discounts with shape ``[T, B...]``.
        behavior_logp: Behavior-policy action log-probs with shape ``[T, B...]``.
        target_logp: Learner-policy action log-probs with shape ``[T, B...]``.
        rho_bar: Importance ratio clip used for value correction and policy advantages.
        c_bar: Trace clip used inside the backward recursion.
    """
    if rho_bar < 0.0:
        raise ValueError("rho_bar must be non-negative")
    if c_bar < 0.0:
        raise ValueError("c_bar must be non-negative")

    _validate_time_major_inputs(rewards, values, discounts, behavior_logp, target_logp)

    log_rhos = np.asarray(target_logp, dtype=np.float64) - np.asarray(behavior_logp, dtype=np.float64)
    rhos = np.exp(log_rhos)
    vs, pg_advantages = _compute_vtrace_from_rhos(
        rewards,
        values,
        discounts,
        rhos,
        rho_bar=rho_bar,
        c_bar=c_bar,
    )
    return VTraceTargets(
        vs=np.asarray(vs, dtype=np.float32),
        pg_advantages=np.asarray(pg_advantages, dtype=np.float32),
        rhos=np.asarray(rhos, dtype=np.float32),
    )
