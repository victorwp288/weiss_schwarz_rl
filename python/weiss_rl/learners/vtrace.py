"""V-trace targets for IMPALA-style off-policy correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.masking import (
    assert_strictly_increasing_legal_ids,
    masked_log_softmax,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
)

_MAX_LOG_RHO = float(np.log(np.finfo(np.float32).max))
_UNAVAILABLE_METRIC = float("nan")


@dataclass(slots=True)
class VtraceMetrics:
    """V-trace health metrics for monitoring learning."""

    rho_mean: float = 0.0
    rho_p50: float = 0.0
    rho_p90: float = 0.0
    rho_p99: float = 0.0
    clip_rate: float = 0.0
    c_clipped_rate: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0


@dataclass(frozen=True, slots=True)
class VTraceTargets:
    vs: np.ndarray
    pg_advantages: np.ndarray
    rhos: np.ndarray


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _unavailable_vtrace_metrics() -> VtraceMetrics:
    return VtraceMetrics(
        rho_mean=_UNAVAILABLE_METRIC,
        rho_p50=_UNAVAILABLE_METRIC,
        rho_p90=_UNAVAILABLE_METRIC,
        rho_p99=_UNAVAILABLE_METRIC,
        clip_rate=_UNAVAILABLE_METRIC,
        c_clipped_rate=_UNAVAILABLE_METRIC,
        kl_divergence=_UNAVAILABLE_METRIC,
        entropy=_UNAVAILABLE_METRIC,
    )


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
    """Compute time-major V-trace value targets and policy advantages."""
    if rho_bar < 0.0:
        raise ValueError("rho_bar must be non-negative")
    if c_bar < 0.0:
        raise ValueError("c_bar must be non-negative")

    _validate_time_major_inputs(rewards, values, discounts, behavior_logp, target_logp)

    log_rhos = np.asarray(target_logp, dtype=np.float64) - np.asarray(behavior_logp, dtype=np.float64)
    safe_log_rhos = np.minimum(log_rhos, _MAX_LOG_RHO)
    rhos = np.minimum(np.exp(safe_log_rhos), np.finfo(np.float32).max)
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


def compute_vtrace_metrics(
    batch: Any,
    rho_bar: float = 2.4,
    c_bar: float = 1.0,
    *,
    pass_action_id: int | None = None,
) -> VtraceMetrics:
    """Compute masked V-trace health metrics from a training batch."""
    metrics = _unavailable_vtrace_metrics()

    logits_value = _batch_value(batch, "logits")
    if logits_value is None:
        return metrics

    try:
        flat_logits, legal_mask = _flatten_logits_and_legality(logits_value, batch)
        metrics.entropy = _mean_masked_entropy(flat_logits, legal_mask)
    except Exception:
        return metrics

    behavior_logits_value = _batch_value(batch, "behavior_logits")
    actions_value = _batch_value(batch, "actions")
    if behavior_logits_value is None or actions_value is None:
        return metrics

    try:
        if np.asarray(behavior_logits_value).shape != np.asarray(logits_value).shape:
            raise ValueError("behavior_logits must match logits")

        behavior_logits, _ = _flatten_logits_and_legality(behavior_logits_value, batch)
        actions = _flatten_actions(actions_value, expected_shape=np.asarray(logits_value).shape[:-1])
        current_logp = _masked_action_logp(flat_logits, batch, actions, pass_action_id=pass_action_id)
        behavior_action_logp = _masked_action_logp(behavior_logits, batch, actions, pass_action_id=pass_action_id)
        rho = np.exp(np.clip(current_logp - behavior_action_logp, a_min=-20.0, a_max=20.0))

        metrics.rho_mean = float(np.mean(rho))
        metrics.rho_p50 = float(np.percentile(rho, 50))
        metrics.rho_p90 = float(np.percentile(rho, 90))
        metrics.rho_p99 = float(np.percentile(rho, 99))
        metrics.clip_rate = float(np.mean(rho > rho_bar))
        metrics.c_clipped_rate = float(np.mean(rho > c_bar))
        metrics.kl_divergence = _mean_masked_kl(behavior_logits, flat_logits, legal_mask)
    except Exception:
        return metrics

    return metrics


def _flatten_logits_and_legality(logits: Any, batch: Any) -> tuple[np.ndarray, np.ndarray]:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim < 2:
        raise ValueError("logits must include an action dimension")

    row_count = int(np.prod(logits_array.shape[:-1]))
    action_space = int(logits_array.shape[-1])
    flat_logits = logits_array.reshape(row_count, action_space)
    legal_mask = _flatten_legal_mask(
        batch,
        expected_shape=logits_array.shape,
        row_count=row_count,
        action_space=action_space,
    )
    return flat_logits, legal_mask


def _flatten_legal_mask(
    batch: Any,
    *,
    expected_shape: tuple[int, ...],
    row_count: int,
    action_space: int,
) -> np.ndarray:
    legal_mask_value = _batch_value(batch, "legal_mask")
    if legal_mask_value is not None:
        legal_mask = np.asarray(legal_mask_value)
        if legal_mask.shape != expected_shape:
            raise ValueError("legal_mask must match logits")
        return legal_mask.reshape(row_count, action_space) != 0

    legal_ids_value = _batch_value(batch, "legal_ids")
    legal_offsets_value = _batch_value(batch, "legal_offsets")
    if legal_ids_value is None or legal_offsets_value is None:
        raise ValueError("masked V-trace metrics require legal_mask or legal_ids/legal_offsets")

    legal_ids = np.asarray(legal_ids_value)
    legal_offsets = np.asarray(legal_offsets_value)
    if legal_offsets.ndim != 1 or legal_offsets.shape[0] != row_count + 1:
        raise ValueError("legal_offsets must have one entry per row plus a sentinel")
    if legal_offsets[0] != 0:
        raise ValueError("legal_offsets must start at 0")
    if np.any(legal_offsets[1:] < legal_offsets[:-1]):
        raise ValueError("legal_offsets must be nondecreasing")
    if legal_offsets[-1] != legal_ids.shape[0]:
        raise ValueError("legal_offsets must end at len(legal_ids)")

    dense_mask = np.zeros((row_count, action_space), dtype=bool)
    for row_index in range(row_count):
        start = int(legal_offsets[row_index])
        end = int(legal_offsets[row_index + 1])
        row_legal_ids = np.asarray(legal_ids[start:end])
        if row_legal_ids.size == 0:
            continue
        assert_strictly_increasing_legal_ids(row_legal_ids)
        dense_mask[row_index, row_legal_ids.astype(np.intp, copy=False)] = True
    return dense_mask


def _flatten_actions(actions: Any, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    actions_array = np.asarray(actions)
    if actions_array.shape != expected_shape:
        raise ValueError("actions must match logits on all non-action dimensions")
    return actions_array.reshape(-1)


def _masked_action_logp(
    flat_logits: np.ndarray,
    batch: Any,
    actions: np.ndarray,
    *,
    pass_action_id: int | None,
) -> np.ndarray:
    legal_mask_value = _batch_value(batch, "legal_mask")
    if legal_mask_value is not None:
        legal_mask = np.asarray(legal_mask_value).reshape(flat_logits.shape[0], flat_logits.shape[1])
        return masked_logp_from_mask(flat_logits, legal_mask, actions, pass_action_id=pass_action_id)

    legal_ids = np.asarray(_batch_value(batch, "legal_ids"))
    legal_offsets = np.asarray(_batch_value(batch, "legal_offsets"))
    return masked_logp_from_legal_ids(
        flat_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def _mean_masked_entropy(flat_logits: np.ndarray, legal_mask: np.ndarray) -> float:
    log_probs = masked_log_softmax(flat_logits, legal_mask)
    safe_log_probs = np.where(legal_mask, log_probs, 0.0)
    probs = np.where(legal_mask, np.exp(safe_log_probs), 0.0)
    entropy = -np.sum(probs * safe_log_probs, axis=1)
    return float(np.mean(entropy))


def _mean_masked_kl(behavior_logits: np.ndarray, logits: np.ndarray, legal_mask: np.ndarray) -> float:
    behavior_log_probs = masked_log_softmax(behavior_logits, legal_mask)
    current_log_probs = masked_log_softmax(logits, legal_mask)
    safe_behavior_log_probs = np.where(legal_mask, behavior_log_probs, 0.0)
    safe_current_log_probs = np.where(legal_mask, current_log_probs, 0.0)
    behavior_probs = np.where(legal_mask, np.exp(safe_behavior_log_probs), 0.0)
    log_ratio = safe_behavior_log_probs - safe_current_log_probs
    kl = np.sum(behavior_probs * log_ratio, axis=1)
    return float(np.mean(kl))
