"""Single-source masking utilities for legal-action handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS_ACTION_ID = 51
_NEG_INF = np.float32(-np.inf)


@dataclass(slots=True)
class MaskingAnomalyCounters:
    empty_legal: int = 0


def resolve_pass_action_id() -> int:
    """Return the contract PASS action id, validating weiss_sim when available."""
    try:
        import weiss_sim
    except Exception:
        return PASS_ACTION_ID

    try:
        simulator_pass_action_id = int(weiss_sim.PASS_ACTION_ID)
    except AttributeError as exc:
        raise RuntimeError("weiss_sim is missing PASS_ACTION_ID") from exc

    if simulator_pass_action_id != PASS_ACTION_ID:
        raise RuntimeError(
            "PASS_ACTION_ID mismatch between weiss_rl and weiss_sim: "
            f"expected {PASS_ACTION_ID}, got {simulator_pass_action_id}"
        )
    return PASS_ACTION_ID


def assert_strictly_increasing_legal_ids(legal_ids: np.ndarray) -> None:
    """Require strictly increasing legal ids with no duplicates."""
    legal_ids_array = np.asarray(legal_ids)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.size <= 1:
        return
    if np.any(legal_ids_array[1:] <= legal_ids_array[:-1]):
        raise ValueError("legal_ids must be strictly increasing")


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


def masked_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Compute log-prob of selected actions from packed legal-id slices."""
    logits_array = _coerce_logits(logits)
    action_space = logits_array.shape[1]
    legal_ids_array = _coerce_legal_ids(legal_ids, action_space=action_space)
    actions_array = _coerce_actions(actions, num_rows=logits_array.shape[0], action_space=action_space)
    legal_offsets_array = _coerce_legal_offsets(
        legal_offsets,
        num_rows=logits_array.shape[0],
        legal_count=legal_ids_array.shape[0],
    )

    logp = np.empty(logits_array.shape[0], dtype=np.float32)
    empty_rows = np.zeros(logits_array.shape[0], dtype=bool)

    for row_index in range(logits_array.shape[0]):
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        if start == end:
            empty_rows[row_index] = True
            continue

        row_legal_ids = legal_ids_array[start:end]
        assert_strictly_increasing_legal_ids(row_legal_ids)

        row_legal_logits = logits_array[row_index, row_legal_ids]
        if not np.all(np.isfinite(row_legal_logits)):
            raise ValueError(f"legal logits must be finite for row {row_index}")

        action = int(actions_array[row_index])
        action_position = int(np.searchsorted(row_legal_ids, action))
        is_legal = action_position < row_legal_ids.size and int(row_legal_ids[action_position]) == action
        if not is_legal:
            raise ValueError(f"illegal action {action} for row {row_index}")

        row_max = np.max(row_legal_logits)
        shifted = row_legal_logits - row_max
        logp[row_index] = np.float32(shifted[action_position] - np.log(np.sum(np.exp(shifted), dtype=np.float32)))

    _apply_empty_row_pass_logp(
        logp,
        empty_rows,
        actions_array,
        pass_action_id=pass_action_id,
        action_space=action_space,
    )
    return logp


""" Sample per-env actions from packed legal-id slices (ids+offsets layout) 
    and return (action, behavior_logp, entropy) for actor unroll storage.
"""
def sample_actions_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    rng: np.random.Generator,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample actions row-wise from categorical distribution restricted to packed legal-id slices.

    Returns:
      actions: (B,) int64
      logp:    (B,) float32
      entropy: (B,) float32
    """
    logits_array = _coerce_logits(logits)
    action_space = logits_array.shape[1]
    legal_ids_array = _coerce_legal_ids(legal_ids, action_space=action_space)
    legal_offsets_array = _coerce_legal_offsets(
        legal_offsets,
        num_rows=logits_array.shape[0],
        legal_count=legal_ids_array.shape[0],
    )

    B = logits_array.shape[0]
    actions = np.empty((B,), dtype=np.int64)
    logp = np.empty((B,), dtype=np.float32)
    entropy = np.empty((B,), dtype=np.float32)

    resolved_pass = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    empty_rows = np.zeros((B,), dtype=bool)

    for row_index in range(B):
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        if start == end:
            empty_rows[row_index] = True
            actions[row_index] = resolved_pass
            logp[row_index] = 0.0
            entropy[row_index] = 0.0
            continue

        row_legal = legal_ids_array[start:end]
        assert_strictly_increasing_legal_ids(row_legal)

        row_logits = logits_array[row_index, row_legal]
        if not np.all(np.isfinite(row_logits)):
            raise ValueError(f"legal logits must be finite for row {row_index}")

        # softmax over legal slice
        m = float(np.max(row_logits))
        shifted = (row_logits - m).astype(np.float32, copy=False)
        exps = np.exp(shifted, dtype=np.float32)
        denom = float(np.sum(exps, dtype=np.float32))
        if denom <= 0.0:
            raise ValueError(f"row {row_index} has zero denom in softmax over legal slice")

        p = exps / denom
        k = int(rng.choice(p.shape[0], p=p))
        a = int(row_legal[k])

        actions[row_index] = a
        logp[row_index] = np.float32(np.log(p[k]))
        pr = p[p > 0.0]
        entropy[row_index] = np.float32(-np.sum(pr * np.log(pr), dtype=np.float32))

    if counters is not None and np.any(empty_rows):
        counters.empty_legal += int(np.sum(empty_rows))

    return actions, logp, entropy

"""
    Actor-side helper: sample an action from masked logits per env and return (action, behavior_logp),
    using PASS + anomaly counter on empty-legal rows.
"""
def sample_actions_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample actions from masked categorical distribution.

    Returns:
        (actions, behavior_logp)
    """
    logits_array = np.asarray(logits, dtype=np.float32)
    legal_mask_array = np.asarray(legal_mask)

    if logits_array.ndim != 2:
        raise ValueError("logits must be 2D")
    if legal_mask_array.shape != logits_array.shape:
        raise ValueError("legal_mask must match logits shape")

    pass_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)

    # Sample only for non-empty rows; empty rows become PASS via fallback.
    empty_rows = ~np.any(legal_mask_array != 0, axis=1)
    actions = np.empty((logits_array.shape[0],), dtype=np.int64)

    if np.any(~empty_rows):
        logp_all = masked_log_softmax(logits_array[~empty_rows], (legal_mask_array[~empty_rows] != 0))
        probs = np.exp(logp_all, dtype=np.float32)
        # Normalize guard (should already sum to 1 on legal support)
        probs = probs / np.sum(probs, axis=1, keepdims=True, dtype=np.float32)

        for i, p in enumerate(probs):
            actions[np.flatnonzero(~empty_rows)[i]] = int(rng.choice(p.shape[0], p=p))

    # Force PASS on empty rows and count anomaly
    if np.any(empty_rows):
        if counters is not None:
            counters.empty_legal += int(np.sum(empty_rows))
        actions[empty_rows] = pass_id

    behavior_logp = masked_logp_from_mask(
        logits_array,
        legal_mask_array,
        actions,
        pass_action_id=pass_id,
    )
    return actions.astype(np.int64, copy=False), behavior_logp


def _coerce_logits(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 2:
        raise ValueError("logits must be a 2D array")
    return logits_array


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


def _coerce_legal_ids(legal_ids: np.ndarray, *, action_space: int) -> np.ndarray:
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


def _coerce_legal_offsets(legal_offsets: np.ndarray, *, num_rows: int, legal_count: int) -> np.ndarray:
    legal_offsets_array = np.asarray(legal_offsets)
    if legal_offsets_array.ndim != 1 or int(legal_offsets_array.shape[0]) != num_rows + 1:
        raise ValueError(f"legal_offsets must have shape ({num_rows + 1},)")
    if legal_offsets_array.dtype == np.bool_ or not np.issubdtype(legal_offsets_array.dtype, np.integer):
        raise ValueError("legal_offsets must be an integer array")

    signed = legal_offsets_array.astype(np.int64, copy=False)
    if int(signed[0]) != 0:
        raise ValueError("legal_offsets must start at 0")
    if np.any(signed < 0):
        raise ValueError("legal_offsets must be >= 0")
    if np.any(signed[1:] < signed[:-1]):
        raise ValueError("legal_offsets must be nondecreasing")
    if int(signed[-1]) > legal_count:
        raise ValueError("legal_offsets cannot point past legal_ids")
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
