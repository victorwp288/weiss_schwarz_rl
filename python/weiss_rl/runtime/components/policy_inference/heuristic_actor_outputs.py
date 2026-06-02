"""Heuristic actor output-writing helpers for queue runtime collection."""

from __future__ import annotations

import numpy as np

from weiss_rl.runtime.components.policy_inference.deterministic_logits import (
    write_deterministic_logits,
    write_deterministic_logits_from_packed,
)


def legal_action_ids_from_mask_rows(*, legal_mask: np.ndarray, row_indices: np.ndarray) -> list[np.ndarray]:
    """Return per-row legal action ids using the mask-layout runtime contract."""

    return [
        np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(np.uint32, copy=False)
        for row_index in row_indices.tolist()
    ]


def write_heuristic_actor_outputs_mask(
    *,
    logits_out: np.ndarray | None,
    row_indices: np.ndarray,
    chosen_actions: np.ndarray,
    legal_mask: np.ndarray,
    actions_out: np.ndarray | None,
    logp_out: np.ndarray | None,
    action_dim: int,
) -> None:
    """Scatter deterministic heuristic actor outputs for mask-backed legal actions."""

    if logits_out is not None:
        write_deterministic_logits(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_action_ids=legal_action_ids_from_mask_rows(legal_mask=legal_mask, row_indices=row_indices),
            action_dim=int(action_dim),
        )
    if actions_out is not None:
        actions_out[row_indices] = chosen_actions
    if logp_out is not None:
        logp_out[row_indices] = 0.0


def write_heuristic_actor_outputs_ids(
    *,
    logits_out: np.ndarray | None,
    row_indices: np.ndarray,
    chosen_actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions_out: np.ndarray | None,
    logp_out: np.ndarray | None,
) -> None:
    """Scatter deterministic heuristic actor outputs for packed legal-action ids."""

    if logits_out is not None:
        write_deterministic_logits_from_packed(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
    if actions_out is not None:
        actions_out[row_indices] = chosen_actions
    if logp_out is not None:
        logp_out[row_indices] = 0.0


__all__ = [
    "legal_action_ids_from_mask_rows",
    "write_heuristic_actor_outputs_ids",
    "write_heuristic_actor_outputs_mask",
]
