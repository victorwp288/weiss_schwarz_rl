"""Pinned evaluation sampling helper math."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.core.masking import masked_logp_from_legal_ids

_CDF_RENORMALIZE_TOL = 1e-6


def coerce_eval_logits(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 1:
        raise ValueError("logits must be a 1D array")
    return logits_array


def coerce_sampling_temperature(temperature: float) -> float:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    return value


def coerce_eval_legal_ids(legal_ids: np.ndarray, *, action_space: int) -> np.ndarray:
    legal_ids_array = np.asarray(legal_ids)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.dtype == np.bool_ or not np.issubdtype(legal_ids_array.dtype, np.integer):
        raise ValueError("legal_ids must be an integer array")

    signed = legal_ids_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("legal_ids must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"legal_ids must be < action_space ({action_space})")
    return signed.astype(np.intp, copy=False)


def require_pass_action(pass_action_id: int | None, *, action_space: int) -> int:
    if pass_action_id is None:
        raise ValueError("pass_action_id is required when legal_ids is empty")
    if pass_action_id < 0 or pass_action_id >= action_space:
        raise ValueError(f"pass_action_id must be in [0, {action_space})")
    return int(pass_action_id)


def legal_probs_for_cdf(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    anomalies: Any | None = None,
) -> np.ndarray:
    legal_logits = logits[legal_ids]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")

    row_max = np.max(legal_logits)
    shifted = legal_logits - row_max
    weights = np.exp(shifted)
    denom = np.sum(weights, dtype=np.float32)
    probs64 = np.asarray(weights / denom, dtype=np.float64)
    return normalize_cdf_probs(probs64, anomalies=anomalies)


def normalize_cdf_probs(
    probs64: np.ndarray,
    *,
    anomalies: Any | None = None,
) -> np.ndarray:
    prob_sum = float(np.sum(probs64, dtype=np.float64))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("legal probabilities must sum to a finite positive value")
    if abs(prob_sum - 1.0) > _CDF_RENORMALIZE_TOL:
        probs64 = probs64 / prob_sum
        if anomalies is not None:
            anomalies.cdf_renormalizations += 1
    return probs64


def sample_cdf_index(probs64: np.ndarray, *, rng: Any) -> int:
    cdf = np.cumsum(probs64, dtype=np.float64)
    cdf[-1] = 1.0
    draw = float(rng.next_float())
    if not np.isfinite(draw) or draw < 0.0 or draw > 1.0:
        raise ValueError("rng.next_float() must return a finite value in [0.0, 1.0]")
    return min(int(np.searchsorted(cdf, draw, side="right")), cdf.size - 1)


def selected_logp(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    action: int,
    *,
    pass_action_id: int | None,
) -> np.float32:
    legal_offsets = np.array([0, legal_ids.size], dtype=np.int64)
    actions = np.array([action], dtype=np.int64)
    logp = masked_logp_from_legal_ids(
        logits[np.newaxis, :],
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )
    return np.float32(logp[0])


__all__ = [
    "coerce_eval_legal_ids",
    "coerce_eval_logits",
    "coerce_sampling_temperature",
    "legal_probs_for_cdf",
    "normalize_cdf_probs",
    "require_pass_action",
    "sample_cdf_index",
    "selected_logp",
]
