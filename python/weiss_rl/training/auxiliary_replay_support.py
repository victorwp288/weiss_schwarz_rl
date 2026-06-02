"""Shared helpers for in-training auxiliary replay regularizers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def trajectory_bc_compatible_training_config(
    *,
    structured_aux: Any,
    dataset_path_text: str,
    every_updates: int,
    field_prefix: str,
    seed_default: int,
    include_focus_fields: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            trajectory_bc_dataset_path=dataset_path_text,
            trajectory_bc_every_updates=int(every_updates),
            trajectory_bc_aux_updates=int(getattr(structured_aux, f"{field_prefix}_aux_updates", 1)),
            trajectory_bc_batch_episodes=int(getattr(structured_aux, f"{field_prefix}_batch_episodes", 8)),
            trajectory_bc_seed=int(getattr(structured_aux, f"{field_prefix}_seed", seed_default)),
            trajectory_bc_focus_source_labels=(
                tuple(getattr(structured_aux, f"{field_prefix}_focus_source_labels", ()))
                if include_focus_fields
                else ()
            ),
            trajectory_bc_focus_fraction=(
                float(getattr(structured_aux, f"{field_prefix}_focus_fraction", 0.0)) if include_focus_fields else 0.0
            ),
            trajectory_bc_focus_groups=(
                tuple(getattr(structured_aux, f"{field_prefix}_focus_groups", ())) if include_focus_fields else ()
            ),
        )
    )


def opponent_context_indices_for_episodes(
    model: Any,
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: list[int],
) -> np.ndarray | None:
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return None
    opponent_ids = source_opponent_policy_ids_by_episode(dataset)
    if not opponent_ids:
        return None
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    indices = model.opponent_context_indices_for_policy_ids(selected_policy_ids)
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, dict) else None
        ids.append(str(raw_id or "").strip())
    return ids


def initial_hidden_state(
    model: Any,
    *,
    batch_size: int,
    device: torch.device,
    opponent_context_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
    kwargs: dict[str, Any] = {"device": device}
    if opponent_context_indices is not None:
        kwargs["opponent_context_indices"] = opponent_context_indices
    try:
        hidden = model.initial_seat_hidden(int(batch_size), **kwargs)
    except TypeError:
        hidden = model.initial_seat_hidden(int(batch_size), device=device)
    return hidden.detach().cpu().numpy()


__all__ = [
    "initial_hidden_state",
    "opponent_context_indices_for_episodes",
    "source_opponent_policy_ids_by_episode",
    "trajectory_bc_compatible_training_config",
]
