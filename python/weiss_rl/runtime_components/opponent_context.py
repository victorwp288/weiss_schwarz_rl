"""Opponent-context helpers for optional recurrent-policy conditioning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from inspect import Parameter, signature
from typing import Any

import numpy as np
import torch


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if keyword in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def opponent_context_indices_for_model(
    model: Any,
    opponent_policy_ids: Iterable[object],
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    policy_ids = list(opponent_policy_ids)
    expected_size = len(policy_ids) if batch_size is None else int(batch_size)
    if len(policy_ids) != expected_size:
        raise ValueError(f"opponent_policy_ids must have length {expected_size}, got {len(policy_ids)}")
    index_fn = getattr(model, "opponent_context_indices_for_policy_ids", None)
    if not callable(index_fn):
        return np.zeros((expected_size,), dtype=np.int16)
    indices = np.asarray(index_fn(policy_ids, batch_size=expected_size), dtype=np.int64)
    if indices.shape != (expected_size,):
        raise ValueError(f"opponent context index shape must be {(expected_size,)}, got {indices.shape}")
    return np.clip(indices, 0, np.iinfo(np.int16).max).astype(np.int16, copy=False)


def initial_seat_hidden_for_opponents(
    model: Any,
    batch_size: int,
    *,
    device: torch.device,
    opponent_policy_ids: Iterable[object] | None = None,
    opponent_context_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    initial_hidden = model.initial_seat_hidden
    kwargs: dict[str, Any] = {"device": device}
    if opponent_context_indices is not None and _call_accepts_keyword(initial_hidden, "opponent_context_indices"):
        kwargs["opponent_context_indices"] = opponent_context_indices
    elif opponent_policy_ids is not None and _call_accepts_keyword(initial_hidden, "opponent_policy_ids"):
        kwargs["opponent_policy_ids"] = opponent_policy_ids
    return initial_hidden(int(batch_size), **kwargs)


def eval_policy_uses_opponent_context(model: Any, policy_id: str) -> bool:
    enabled_fn = getattr(model, "should_apply_opponent_context_for_eval_policy", None)
    return bool(callable(enabled_fn) and enabled_fn(str(policy_id)))


__all__ = [
    "eval_policy_uses_opponent_context",
    "initial_seat_hidden_for_opponents",
    "opponent_context_indices_for_model",
]
