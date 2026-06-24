"""Packed legal-id masking utilities."""

from __future__ import annotations

import numpy as np

from weiss_rl.core.action_ids import resolve_pass_action_id


def assert_strictly_increasing_legal_ids(legal_ids: np.ndarray) -> None:
    """Require strictly increasing legal ids with no duplicates."""
    legal_ids_array = np.asarray(legal_ids)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.size <= 1:
        return
    if np.any(legal_ids_array[1:] <= legal_ids_array[:-1]):
        raise ValueError("legal_ids must be strictly increasing")


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


def sample_actions_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    rng: np.random.Generator,
    counters: object | None = None,
    pass_action_id: int | None = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample actions row-wise from categorical distribution restricted to packed legal-id slices.

    Returns:
      actions: (B,) int64
      logp:    (B,) float32
      entropy: (B,) float32
    """
    logits_array = _coerce_logits(logits)
    temperature_value = _coerce_sampling_temperature(temperature)
    action_space = logits_array.shape[1]
    legal_ids_array = _coerce_legal_ids(legal_ids, action_space=action_space)
    legal_offsets_array = _coerce_legal_offsets(
        legal_offsets,
        num_rows=logits_array.shape[0],
        legal_count=legal_ids_array.shape[0],
    )

    batch_size = logits_array.shape[0]
    actions = np.empty((batch_size,), dtype=np.int64)
    logp = np.empty((batch_size,), dtype=np.float32)
    entropy = np.empty((batch_size,), dtype=np.float32)

    resolved_pass = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    empty_rows = np.zeros((batch_size,), dtype=bool)

    for row_index in range(batch_size):
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
        if temperature_value != 1.0:
            row_logits = row_logits / np.float32(temperature_value)
        if not np.all(np.isfinite(row_logits)):
            raise ValueError(f"legal logits must be finite for row {row_index}")

        row_max = np.max(row_logits)
        shifted = (row_logits - row_max).astype(np.float32, copy=False)
        exps = np.exp(shifted, dtype=np.float32)
        denom = np.sum(exps, dtype=np.float32)
        if float(denom) <= 0.0:
            raise ValueError(f"row {row_index} has zero denom in softmax over legal slice")

        row_logp = shifted - np.log(denom)
        probs = np.exp(row_logp, dtype=np.float32)
        choice = int(rng.choice(probs.shape[0], p=probs))

        actions[row_index] = int(row_legal[choice])
        logp[row_index] = np.float32(row_logp[choice])
        positive_probs = probs[probs > 0.0]
        entropy[row_index] = np.float32(-np.sum(positive_probs * np.log(positive_probs), dtype=np.float32))

    if counters is not None and np.any(empty_rows):
        counters.empty_legal += int(np.sum(empty_rows))

    return actions, logp, entropy


def select_argmax_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Select the highest-logit legal action for each packed legal-id row."""

    logits_array = _coerce_logits(logits)
    action_space = logits_array.shape[1]
    legal_ids_array = _coerce_legal_ids(legal_ids, action_space=action_space)
    legal_offsets_array = _coerce_legal_offsets(
        legal_offsets,
        num_rows=logits_array.shape[0],
        legal_count=legal_ids_array.shape[0],
    )
    pass_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    actions = np.empty((logits_array.shape[0],), dtype=np.int64)
    for row_index in range(logits_array.shape[0]):
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        if start == end:
            actions[row_index] = pass_id
            continue
        row_legal = legal_ids_array[start:end]
        assert_strictly_increasing_legal_ids(row_legal)
        row_logits = logits_array[row_index, row_legal]
        if not np.all(np.isfinite(row_logits)):
            raise ValueError(f"legal logits must be finite for row {row_index}")
        actions[row_index] = int(row_legal[int(np.argmax(row_logits))])
    return actions


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


__all__ = [
    "assert_strictly_increasing_legal_ids",
    "masked_logp_from_legal_ids",
    "sample_actions_from_legal_ids",
    "select_argmax_from_legal_ids",
]
