"""Single-source masking utilities for legal-action handling."""

from __future__ import annotations

import numpy as np


def assert_strictly_increasing_legal_ids(legal_ids: np.ndarray) -> None:
    """Require strictly increasing legal ids with no duplicates."""
    if legal_ids.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids.size <= 1:
        return
    if np.any(legal_ids[1:] <= legal_ids[:-1]):
        raise ValueError("legal_ids must be strictly increasing")


def masked_log_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Compute log-softmax over legal actions, setting illegal entries to -inf."""
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask shapes must match")
    safe_logits = np.where(legal_mask != 0, logits, -np.inf)
    row_max = np.max(safe_logits, axis=1, keepdims=True)
    shifted = safe_logits - row_max
    exp_shifted = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    denom = np.sum(exp_shifted, axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_probs = shifted - np.log(denom)
    return np.where(legal_mask != 0, log_probs, -np.inf)
