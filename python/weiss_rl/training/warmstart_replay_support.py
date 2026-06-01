"""Shared replay-batch helpers for auxiliary warmstart workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _opponent_context_indices_for_episodes(model: Any, dataset: Any, *, episode_indices: list[int]) -> np.ndarray:
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return np.zeros((len(episode_indices),), dtype=np.int64)
    return np.asarray(model.opponent_context_indices_for_policy_ids(selected_policy_ids), dtype=np.int64).reshape(-1)


def _source_opponent_policy_ids_by_episode(dataset: Any) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, Mapping) else None
        ids.append(str(raw_id or "").strip())
    return ids


def _sample_episode_indices(
    rng: np.random.Generator,
    *,
    episode_count: int,
    batch_episodes: int,
) -> list[int]:
    count = int(episode_count)
    batch_size = int(batch_episodes)
    if count <= 0:
        raise ValueError("episode_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_episodes must be positive")
    replace = count < batch_size
    indices = rng.choice(count, size=batch_size, replace=replace)
    return [int(index) for index in np.asarray(indices, dtype=np.int64).reshape(-1).tolist()]


def _initial_hidden_state(
    model: Any,
    *,
    batch_size: int,
    device: torch.device,
    opponent_context_indices: np.ndarray,
) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
    try:
        hidden = model.initial_seat_hidden(
            int(batch_size),
            device=device,
            opponent_context_indices=opponent_context_indices,
        )
    except TypeError:
        hidden = model.initial_seat_hidden(int(batch_size), device=device)
    return hidden.detach().cpu().numpy()


__all__ = [
    "_initial_hidden_state",
    "_opponent_context_indices_for_episodes",
    "_sample_episode_indices",
    "_source_opponent_policy_ids_by_episode",
]
