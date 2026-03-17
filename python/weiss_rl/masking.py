"""Single-source masking utilities for legal-action handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS_ACTION_ID = 51


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
    if legal_ids.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids.size <= 1:
        return
    if np.any(legal_ids[1:] <= legal_ids[:-1]):
        raise ValueError("legal_ids must be strictly increasing")


def masked_log_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Compute log-softmax over legal actions and keep illegal entries at -inf."""
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if logits.ndim != 2:
        raise ValueError("logits and legal_mask must be 2D (batch, action)")

    legal = legal_mask != 0
    any_legal = np.any(legal, axis=1, keepdims=True)

    safe_logits = np.where(legal, logits, -np.inf)
    row_max = np.where(any_legal, np.max(safe_logits, axis=1, keepdims=True), 0.0)
    shifted = np.where(any_legal, safe_logits - row_max, -np.inf)

    exp_shifted = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    denom = np.sum(exp_shifted, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = shifted - np.log(denom)

    return np.where(legal & any_legal, log_probs, -np.inf)


def empty_legal_guard(
    legal_mask: np.ndarray,
    *,
    counters: MaskingAnomalyCounters | None = None,
    pass_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the PASS fallback payload for rows with no legal actions."""
    if legal_mask.ndim != 2:
        raise ValueError("legal_mask must be 2D (batch, action)")

    empty_rows = ~np.any(legal_mask != 0, axis=1)
    if counters is not None and np.any(empty_rows):
        counters.empty_legal += int(np.sum(empty_rows))

    resolved_pass_action_id = resolve_pass_action_id() if pass_action_id is None else int(pass_action_id)
    batch_size = legal_mask.shape[0]
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
    if action_array.shape[0] != legal_mask.shape[0]:
        raise ValueError("actions batch dimension must match legal_mask")

    empty_rows, fallback_actions, _, _ = empty_legal_guard(
        legal_mask,
        counters=counters,
        pass_action_id=pass_action_id,
    )
    adjusted_actions = action_array.copy()
    if np.any(empty_rows):
        adjusted_actions[empty_rows] = fallback_actions[empty_rows].astype(adjusted_actions.dtype, copy=False)
    return empty_rows, adjusted_actions
