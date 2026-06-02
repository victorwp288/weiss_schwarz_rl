"""Lazy legal-action hook helpers for central actor-row forwarding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch


def concatenate_batch_legal_actions(batches: Sequence[DecisionBoundaryBatch], *, action_space: int) -> Any:
    # Resolve lazily through weiss_rl.runtime so the compatibility wrapper remains the public hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._concatenate_batch_legal_actions(batches, action_space=action_space)


def optional_legal_action_meta(batch: DecisionBoundaryBatch) -> np.ndarray | None:
    from weiss_rl import runtime as runtime_module

    return runtime_module._optional_legal_action_meta(batch)


def require_ids_offsets(batch: DecisionBoundaryBatch) -> tuple[np.ndarray, np.ndarray]:
    from weiss_rl import runtime as runtime_module

    return runtime_module._require_ids_offsets(batch)


def slice_packed_rows_with_meta(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    *,
    legal_action_meta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from weiss_rl import runtime as runtime_module

    return runtime_module._slice_packed_rows_with_meta(
        legal_ids,
        legal_offsets,
        row_indices,
        legal_action_meta=legal_action_meta,
    )


__all__ = [
    "concatenate_batch_legal_actions",
    "optional_legal_action_meta",
    "require_ids_offsets",
    "slice_packed_rows_with_meta",
]
