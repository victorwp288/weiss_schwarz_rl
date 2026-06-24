"""Pinned action sampling helpers for deterministic evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from weiss_rl.core.masking import (
    assert_strictly_increasing_legal_ids,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
)
from weiss_rl.eval.sampling.sampling_helpers import coerce_eval_legal_ids as _coerce_eval_legal_ids
from weiss_rl.eval.sampling.sampling_helpers import coerce_eval_logits as _coerce_eval_logits
from weiss_rl.eval.sampling.sampling_helpers import coerce_sampling_temperature as _coerce_sampling_temperature
from weiss_rl.eval.sampling.sampling_helpers import legal_probs_for_cdf as _legal_probs_for_cdf
from weiss_rl.eval.sampling.sampling_helpers import normalize_cdf_probs
from weiss_rl.eval.sampling.sampling_helpers import require_pass_action as _require_pass_action
from weiss_rl.eval.sampling.sampling_helpers import sample_cdf_index as _sample_cdf_index
from weiss_rl.eval.sampling.sampling_helpers import selected_logp as _selected_logp
from weiss_rl.eval.simulator.records import EvalSamplerAnomalies


class _FloatRng(Protocol):
    def next_float(self) -> float: ...


def eval_sampler_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def eval_sampler_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def _normalize_cdf_probs(
    probs64: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    return normalize_cdf_probs(probs64, anomalies=anomalies)


def sample_action_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    rng: _FloatRng,
    pass_action_id: int | None = None,
    anomalies: EvalSamplerAnomalies | None = None,
    temperature: float = 1.0,
) -> tuple[int, np.float32]:
    """Sample one action from a single packed legal-id row with pinned CPU CDF math."""
    logits_array = _coerce_eval_logits(logits)
    temperature_value = _coerce_sampling_temperature(temperature)
    if temperature_value != 1.0:
        logits_array = logits_array / np.float32(temperature_value)
    legal_ids_array = _coerce_eval_legal_ids(legal_ids, action_space=logits_array.shape[0])

    if legal_ids_array.size == 0:
        action = _require_pass_action(pass_action_id, action_space=logits_array.shape[0])
        logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=action)
        return action, logp

    assert_strictly_increasing_legal_ids(legal_ids_array)
    probs64 = _legal_probs_for_cdf(logits_array, legal_ids_array, anomalies=anomalies)
    action_index = _sample_cdf_index(probs64, rng=rng)
    action = int(legal_ids_array[action_index])
    logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=pass_action_id)
    return action, logp


def select_action_argmax_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> tuple[int, np.float32]:
    """Select the highest-logit legal action with the same logp math as pinned eval."""

    logits_array = _coerce_eval_logits(logits)
    legal_ids_array = _coerce_eval_legal_ids(legal_ids, action_space=logits_array.shape[0])

    if legal_ids_array.size == 0:
        action = _require_pass_action(pass_action_id, action_space=logits_array.shape[0])
        logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=action)
        return action, logp

    assert_strictly_increasing_legal_ids(legal_ids_array)
    legal_logits = logits_array[legal_ids_array]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")
    action_index = int(np.argmax(legal_logits))
    action = int(legal_ids_array[action_index])
    logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=pass_action_id)
    return action, logp


__all__ = [
    "eval_sampler_logp_from_legal_ids",
    "eval_sampler_logp_from_mask",
    "sample_action_pinned",
    "select_action_argmax_pinned",
]
