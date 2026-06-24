"""Single-source masking utilities for legal-action handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weiss_rl.core.action_ids import PASS_ACTION_ID as PASS_ACTION_ID
from weiss_rl.core.action_ids import resolve_pass_action_id
from weiss_rl.core.packed_masking import (
    assert_strictly_increasing_legal_ids as assert_strictly_increasing_legal_ids,
)
from weiss_rl.core.packed_masking import masked_logp_from_legal_ids as masked_logp_from_legal_ids
from weiss_rl.core.packed_masking import sample_actions_from_legal_ids as sample_actions_from_legal_ids
from weiss_rl.core.packed_masking import select_argmax_from_legal_ids as select_argmax_from_legal_ids

_NEG_INF = np.float32(-np.inf)


@dataclass(slots=True)
class MaskingAnomalyCounters:
    empty_legal: int = 0


def masked_log_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Compute row-wise masked log-softmax without NaNs on fully-masked rows."""
    logits_array = _coerce_logits(logits)
    legal_mask_array = _coerce_legal_mask(legal_mask, expected_shape=logits_array.shape)
    _require_finite_legal_logits_from_mask(logits_array, legal_mask_array)
    return _masked_log_softmax_from_inputs(logits_array, legal_mask_array)


def empty_legal_guard(
    legal_mask: np.ndarray,
    *,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the PASS fallback payload for rows with no legal actions."""
    legal_mask_array = np.asarray(legal_mask)
    if legal_mask_array.ndim != 2:
        raise ValueError("legal_mask must be 2D (batch, action)")
    legal_mask_bool = legal_mask_array != 0
    empty_rows = ~np.any(legal_mask_bool, axis=1)
    if counters is not None and np.any(empty_rows):
        counters.empty_legal += int(np.sum(empty_rows))

    resolved_pass_action_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    batch_size = legal_mask_bool.shape[0]
    actions = np.full((batch_size,), resolved_pass_action_id, dtype=np.int64)
    logp = np.zeros((batch_size,), dtype=np.float32)
    entropy = np.zeros((batch_size,), dtype=np.float32)
    return empty_rows, actions, logp, entropy


def apply_empty_legal_action_fallback(
    actions: np.ndarray,
    legal_mask: np.ndarray,
    *,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Force PASS on rows with no legal actions and report which rows were empty."""
    action_array = np.asarray(actions)
    if action_array.ndim != 1:
        raise ValueError("actions must be 1D (batch,)")

    legal_mask_array = np.asarray(legal_mask)
    if legal_mask_array.ndim != 2:
        raise ValueError("legal_mask must be 2D (batch, action)")
    if action_array.shape[0] != legal_mask_array.shape[0]:
        raise ValueError("actions batch dimension must match legal_mask")

    empty_rows, fallback_actions, _, _ = empty_legal_guard(
        legal_mask_array,
        counters=counters,
        pass_action_id=pass_action_id,
    )
    adjusted_actions = action_array.copy()
    if np.any(empty_rows):
        adjusted_actions[empty_rows] = fallback_actions[empty_rows].astype(adjusted_actions.dtype, copy=False)
    return empty_rows, adjusted_actions


def masked_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Compute log-prob of selected actions under the dense masked distribution."""
    logits_array = _coerce_logits(logits)
    legal_mask_array = _coerce_legal_mask(legal_mask, expected_shape=logits_array.shape)
    actions_array = _coerce_actions(actions, num_rows=logits_array.shape[0], action_space=logits_array.shape[1])
    _require_finite_legal_logits_from_mask(logits_array, legal_mask_array)

    empty_rows = ~np.any(legal_mask_array, axis=1)
    row_ids = np.arange(logits_array.shape[0], dtype=np.intp)
    illegal_rows = (~empty_rows) & (~legal_mask_array[row_ids, actions_array])
    if np.any(illegal_rows):
        row_index = int(np.flatnonzero(illegal_rows)[0])
        action = int(actions_array[row_index])
        raise ValueError(f"illegal action {action} for row {row_index}")

    log_probs = _masked_log_softmax_from_inputs(logits_array, legal_mask_array)
    selected = log_probs[row_ids, actions_array].astype(np.float32, copy=True)
    _apply_empty_row_pass_logp(
        selected,
        empty_rows,
        actions_array,
        pass_action_id=pass_action_id,
        action_space=logits_array.shape[1],
    )
    return selected


def sample_actions_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample masked actions row-wise and return action, behavior logp, and entropy."""
    logits_array = _coerce_logits(logits)
    temperature_value = _coerce_sampling_temperature(temperature)
    legal_mask_array = _coerce_legal_mask(legal_mask, expected_shape=logits_array.shape)
    _require_finite_legal_logits_from_mask(logits_array, legal_mask_array)
    sampling_logits = logits_array if temperature_value == 1.0 else logits_array / np.float32(temperature_value)

    batch_size = logits_array.shape[0]
    pass_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    empty_rows = ~np.any(legal_mask_array, axis=1)
    actions = np.empty((batch_size,), dtype=np.int64)
    logp = np.empty((batch_size,), dtype=np.float32)
    entropy = np.empty((batch_size,), dtype=np.float32)

    non_empty_rows = np.flatnonzero(~empty_rows)
    if non_empty_rows.size > 0:
        logp_all = masked_log_softmax(sampling_logits[non_empty_rows], legal_mask_array[non_empty_rows])
        probs = np.exp(logp_all, dtype=np.float32)

        for local_row, row_index in enumerate(non_empty_rows.tolist()):
            row_probs = probs[local_row]
            action = int(rng.choice(row_probs.shape[0], p=row_probs))
            actions[row_index] = action
            logp[row_index] = np.float32(logp_all[local_row, action])
            positive_probs = row_probs[row_probs > 0.0]
            entropy[row_index] = np.float32(-np.sum(positive_probs * np.log(positive_probs), dtype=np.float32))

    if np.any(empty_rows):
        if counters is not None:
            counters.empty_legal += int(np.sum(empty_rows))
        actions[empty_rows] = pass_id
        logp[empty_rows] = 0.0
        entropy[empty_rows] = 0.0

    return actions.astype(np.int64, copy=False), logp, entropy


def select_argmax_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Select the highest-logit legal action for each dense legal-mask row."""

    logits_array = _coerce_logits(logits)
    legal_mask_array = _coerce_legal_mask(legal_mask, expected_shape=logits_array.shape)
    _require_finite_legal_logits_from_mask(logits_array, legal_mask_array)
    pass_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    actions = np.empty((logits_array.shape[0],), dtype=np.int64)
    for row_index in range(logits_array.shape[0]):
        legal_indices = np.flatnonzero(legal_mask_array[row_index])
        if legal_indices.size == 0:
            actions[row_index] = pass_id
            continue
        actions[row_index] = int(legal_indices[int(np.argmax(logits_array[row_index, legal_indices]))])
    return actions


def logits_for_sampling_temperature(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    """Return logits scaled for behavior sampling, preserving the original array at temperature 1."""
    logits_array = _coerce_logits(logits)
    temperature_value = _coerce_sampling_temperature(temperature)
    if temperature_value == 1.0:
        return logits_array
    return (logits_array / np.float32(temperature_value)).astype(np.float32, copy=False)


def _coerce_logits(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 2:
        raise ValueError("logits must be a 2D array")
    return logits_array


def _coerce_sampling_temperature(temperature: float) -> float:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    return value


def _coerce_legal_mask(legal_mask: np.ndarray, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    legal_mask_array = np.asarray(legal_mask)
    if legal_mask_array.shape != expected_shape:
        raise ValueError("logits and legal_mask shapes must match")
    return legal_mask_array != 0


def _coerce_actions(actions: np.ndarray, *, num_rows: int, action_space: int) -> np.ndarray:
    actions_array = np.asarray(actions)
    if actions_array.ndim != 1 or int(actions_array.shape[0]) != num_rows:
        raise ValueError(f"actions must have shape ({num_rows},)")
    if actions_array.dtype == np.bool_ or not np.issubdtype(actions_array.dtype, np.integer):
        raise ValueError("actions must be an integer array")

    signed = actions_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("actions must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"actions must be < action_space ({action_space})")
    return signed.astype(np.intp, copy=False)


def _require_finite_legal_logits_from_mask(logits: np.ndarray, legal_mask: np.ndarray) -> None:
    if not np.all(np.isfinite(logits[legal_mask])):
        raise ValueError("legal logits must be finite")


def _masked_log_softmax_from_inputs(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    safe_logits = np.where(legal_mask, logits, _NEG_INF)
    has_legal = np.any(legal_mask, axis=1, keepdims=True)
    row_max = np.max(safe_logits, axis=1, keepdims=True)
    row_max = np.where(has_legal, row_max, 0.0)
    shifted = np.where(legal_mask, logits - row_max, _NEG_INF)
    exp_shifted = np.where(legal_mask, np.exp(shifted), 0.0)
    denom = np.sum(exp_shifted, axis=1, keepdims=True, dtype=np.float32)
    log_denom = np.zeros_like(denom, dtype=np.float32)
    np.log(denom, out=log_denom, where=has_legal)
    return np.where(legal_mask, shifted - log_denom, _NEG_INF).astype(np.float32, copy=False)


def _apply_empty_row_pass_logp(
    logp: np.ndarray,
    empty_rows: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None,
    action_space: int,
) -> None:
    if not np.any(empty_rows):
        return

    if pass_action_id is None:
        raise ValueError("pass_action_id is required when legality contains empty rows")
    if pass_action_id < 0 or pass_action_id >= action_space:
        raise ValueError(f"pass_action_id must be in [0, {action_space})")

    illegal_rows = empty_rows & (actions != int(pass_action_id))
    if np.any(illegal_rows):
        row_index = int(np.flatnonzero(illegal_rows)[0])
        action = int(actions[row_index])
        raise ValueError(f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}")
    logp[empty_rows] = 0.0
