"""Single-source masking utilities for legal-action handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


"""Best-effort PASS_ACTION_ID resolution.

    Prefer weiss_sim.PASS_ACTION_ID when installed, else fall back to 0 so
    weiss_rl remains importable in dev/CI environments without weiss_sim.
"""

def _pass_action_id() -> int:
    try:
        import weiss_sim  # type: ignore
        return int(weiss_sim.PASS_ACTION_ID)
    except Exception:
        return 0


@dataclass(slots=True)
class MaskingAnomalyCounters:
    empty_legal: int = 0


def assert_strictly_increasing_legal_ids(legal_ids: np.ndarray) -> None:
    if legal_ids.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids.size <= 1:
        return
    if np.any(legal_ids[1:] <= legal_ids[:-1]):
        raise ValueError("legal_ids must be strictly increasing")


"""Compute log-softmax over legal actions, setting illegal entries to -inf.

    For empty-legal rows, returns all -inf. Callers must apply PASS fallback.
 """

def masked_log_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:

    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if logits.ndim != 2:
        raise ValueError("logits and legal_mask must be 2D (batch, action)")

    legal = (legal_mask != 0)
    any_legal = np.any(legal, axis=1, keepdims=True)

    safe_logits = np.where(legal, logits, -np.inf)
    row_max = np.where(any_legal, np.max(safe_logits, axis=1, keepdims=True), 0.0)
    shifted = np.where(any_legal, safe_logits - row_max, -np.inf)

    exp_shifted = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    denom = np.sum(exp_shifted, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = shifted - np.log(denom)

    return np.where(legal & any_legal, log_probs, -np.inf)


 """Guard for the no-legal edge case (master plan §6.6).
    Returns:
      empty_rows: bool (B,)
      fallback_actions: int64 (B,) filled with PASS_ACTION_ID
      fallback_logp: float32 (B,) filled with 0.0
      fallback_entropy: float32 (B,) filled with 0.0
    """

def empty_legal_guard(
    legal_mask: np.ndarray,
    *,
    counters: MaskingAnomalyCounters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    if legal_mask.ndim != 2:
        raise ValueError("legal_mask must be 2D (batch, action)")
    empty_rows = ~np.any(legal_mask != 0, axis=1)
    if counters is not None and np.any(empty_rows):
        counters.empty_legal += int(np.sum(empty_rows))

    bsz = legal_mask.shape[0]
    pid = _pass_action_id()
    actions = np.full((bsz,), pid, dtype=np.int64)
    logp = np.zeros((bsz,), dtype=np.float32)
    entropy = np.zeros((bsz,), dtype=np.float32)
    return empty_rows, actions, logp, entropy
