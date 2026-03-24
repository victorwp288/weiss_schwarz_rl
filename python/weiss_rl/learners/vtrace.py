"""V-trace helper scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.masking import (
    assert_strictly_increasing_legal_ids,
    masked_log_softmax,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
)


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


def compute_vtrace_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    discounts: np.ndarray,
) -> np.ndarray:
    """Placeholder V-trace target calculation.

    This simple bootstrap is a scaffold only; replace with full IMPALA V-trace.
    """
    if rewards.shape != values.shape or rewards.shape != discounts.shape:
        raise ValueError("rewards, values, discounts must have identical shapes")
    return rewards + discounts * values


def compute_vtrace_metrics(
    batch: dict[str, Any],
    rho_bar: float = 2.4,
    c_bar: float = 1.0,
    *,
    pass_action_id: int | None = None,
) -> VtraceMetrics:
    """Compute masked V-trace health metrics from a training batch."""
    metrics = VtraceMetrics()
    if not isinstance(batch, dict):
        return metrics

    try:
        logits, legal_mask = _flatten_logits_and_legality(batch["logits"], batch)
        metrics.entropy = _mean_masked_entropy(logits, legal_mask)
    except Exception:
        return metrics

    try:
        behavior_logits, _ = _flatten_logits_and_legality(batch["behavior_logits"], batch)
        actions = _flatten_actions(batch["actions"], expected_shape=np.asarray(batch["logits"]).shape[:-1])
        current_logp = _masked_action_logp(logits, batch, actions, pass_action_id=pass_action_id)
        behavior_logp = _masked_action_logp(behavior_logits, batch, actions, pass_action_id=pass_action_id)
        rho = np.exp(np.clip(current_logp - behavior_logp, a_min=-20.0, a_max=20.0))

        metrics.rho_mean = float(np.mean(rho))
        metrics.rho_p50 = float(np.percentile(rho, 50))
        metrics.rho_p90 = float(np.percentile(rho, 90))
        metrics.rho_p99 = float(np.percentile(rho, 99))
        metrics.clip_rate = float(np.mean(rho > rho_bar))
        metrics.c_clipped_rate = float(np.mean(rho > c_bar))
        metrics.kl_divergence = _mean_masked_kl(behavior_logits, logits, legal_mask)
    except Exception:
        pass

    return metrics


def _flatten_logits_and_legality(logits: Any, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim < 2:
        raise ValueError("logits must include an action dimension")

    row_count = int(np.prod(logits_array.shape[:-1]))
    action_space = int(logits_array.shape[-1])
    flat_logits = logits_array.reshape(row_count, action_space)
    legal_mask = _flatten_legal_mask(batch, row_count=row_count, action_space=action_space)
    return flat_logits, legal_mask


def _flatten_legal_mask(batch: dict[str, Any], *, row_count: int, action_space: int) -> np.ndarray:
    if "legal_mask" in batch:
        legal_mask = np.asarray(batch["legal_mask"])
        expected_shape = np.asarray(batch["logits"]).shape
        if legal_mask.shape != expected_shape:
            raise ValueError("legal_mask must match logits")
        return legal_mask.reshape(row_count, action_space) != 0

    if "legal_ids" not in batch or "legal_offsets" not in batch:
        raise ValueError("masked V-trace metrics require legal_mask or legal_ids/legal_offsets")

    legal_ids = np.asarray(batch["legal_ids"])
    legal_offsets = np.asarray(batch["legal_offsets"])
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
    batch: dict[str, Any],
    actions: np.ndarray,
    *,
    pass_action_id: int | None,
) -> np.ndarray:
    if "legal_mask" in batch:
        legal_mask = np.asarray(batch["legal_mask"]).reshape(flat_logits.shape[0], flat_logits.shape[1])
        return masked_logp_from_mask(flat_logits, legal_mask, actions, pass_action_id=pass_action_id)

    legal_ids = np.asarray(batch["legal_ids"])
    legal_offsets = np.asarray(batch["legal_offsets"])
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
