"""In-training paired outcome preference replay regularizer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, replay_trajectory_bc_batch
from weiss_rl.training.trajectory_bc_replay import TrajectoryBcReplayState

_AGGREGATIONS = frozenset({"mean", "sum"})


@dataclass(slots=True)
class PairedOutcomePreferenceReplayState:
    sampler: TrajectoryBcReplayState
    beta: float
    coef: float
    aggregation: str
    group_balance: bool
    complete_pair_count: int

    @classmethod
    def from_training_config(
        cls, training_config: Any, *, repo_root: Path
    ) -> PairedOutcomePreferenceReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "paired_outcome_preference_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "paired_outcome_preference_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        beta = float(getattr(structured_aux, "paired_outcome_preference_beta", 0.1))
        if beta <= 0.0:
            raise ValueError("paired_outcome_preference_beta must be > 0.0")
        coef = float(getattr(structured_aux, "paired_outcome_preference_coef", 0.05))
        if coef < 0.0:
            raise ValueError("paired_outcome_preference_coef must be >= 0.0")
        aggregation = str(getattr(structured_aux, "paired_outcome_preference_aggregation", "mean")).strip().lower()
        if aggregation not in _AGGREGATIONS:
            raise ValueError("paired_outcome_preference_aggregation must be one of: mean, sum")
        group_balance = bool(getattr(structured_aux, "paired_outcome_preference_group_balance", False))

        sampler = TrajectoryBcReplayState.from_training_config(
            _trajectory_bc_compatible_training_config(
                structured_aux=structured_aux,
                dataset_path_text=dataset_path_text,
                every_updates=every_updates,
            ),
            repo_root=repo_root,
        )
        if sampler is None:
            return None
        complete_pair_count = paired_outcome_preference_complete_pair_count(sampler.dataset)
        if complete_pair_count <= 0:
            raise ValueError(
                f"paired outcome preference dataset has no complete preferred/rejected pairs: {dataset_path_text}"
            )
        return cls(
            sampler=sampler,
            beta=beta,
            coef=coef,
            aggregation=aggregation,
            group_balance=group_balance,
            complete_pair_count=complete_pair_count,
        )


def maybe_run_paired_outcome_preference_replay(
    *,
    state: PairedOutcomePreferenceReplayState | None,
    learner: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured paired outcome preference auxiliary steps after an RL update."""

    if state is None:
        return
    sampler = state.sampler
    if int(update_count) <= 0 or int(update_count) % int(sampler.every_updates) != 0:
        return
    updater = getattr(learner, "paired_outcome_preference_update", None)
    if not callable(updater):
        raise ValueError("learner does not support paired_outcome_preference_update")

    aux_metrics: dict[str, float] = {}
    total_batch_episodes = 0
    total_context_episodes = 0
    for _ in range(int(sampler.aux_updates)):
        indices = sampler.next_episode_indices()
        total_batch_episodes += len(indices)
        opponent_context_indices = _opponent_context_indices_for_episodes(
            learner.model,
            sampler.dataset,
            episode_indices=indices,
        )
        if opponent_context_indices is not None:
            total_context_episodes += int(np.count_nonzero(opponent_context_indices))
        hidden = _initial_hidden_state(
            learner.model,
            batch_size=len(indices),
            device=device,
            opponent_context_indices=opponent_context_indices,
        )
        batch = replay_trajectory_bc_batch(
            sampler.dataset,
            episode_indices=indices,
            initial_hidden_state=hidden,
            opponent_context_indices=opponent_context_indices,
        )
        preference_group_indices = _preference_group_indices_for_episodes(sampler.dataset, episode_indices=indices)
        if preference_group_indices is not None:
            batch["preference_group_id"] = np.broadcast_to(
                np.asarray(preference_group_indices, dtype=np.int64).reshape(1, -1),
                np.asarray(batch["actions"]).shape,
            ).copy()
        aux_metrics = updater(
            batch,
            beta=float(state.beta),
            coef=float(state.coef),
            aggregation=state.aggregation,
            group_balance=bool(state.group_balance),
        )

    latest_metrics["paired_outcome_preference_replay_aux_updates"] = float(sampler.aux_updates)
    latest_metrics["paired_outcome_preference_replay_batch_episodes"] = float(total_batch_episodes)
    latest_metrics["paired_outcome_preference_replay_dataset_train_rows"] = float(
        sampler.dataset.metadata["train_rows"]
    )
    latest_metrics["paired_outcome_preference_replay_complete_pair_count"] = float(state.complete_pair_count)
    latest_metrics["paired_outcome_preference_replay_beta"] = float(state.beta)
    latest_metrics["paired_outcome_preference_replay_coef"] = float(state.coef)
    latest_metrics["paired_outcome_preference_replay_aggregation_sum"] = 1.0 if state.aggregation == "sum" else 0.0
    latest_metrics["paired_outcome_preference_replay_group_balance"] = 1.0 if state.group_balance else 0.0
    latest_metrics["paired_outcome_preference_replay_opponent_context_episodes"] = float(total_context_episodes)
    for key, value in aux_metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            latest_metrics[f"paired_outcome_preference_replay_{key}"] = float(value)


def paired_outcome_preference_complete_pair_count(dataset: ReplayTrajectoryDataset) -> int:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return 0
    roles_by_pair: dict[int, set[int]] = {}
    train_rows_by_episode = np.asarray(dataset.policy_train_mask, dtype=np.bool_).any(axis=0)
    for episode_index, bundle in enumerate(bundles):
        if not train_rows_by_episode[int(episode_index)]:
            continue
        if not isinstance(bundle, Mapping):
            continue
        if "preference_pair_id" not in bundle or "preference_role" not in bundle:
            continue
        pair_id = int(bundle["preference_pair_id"])
        role = int(bundle["preference_role"])
        if pair_id < 0 or role not in {0, 1}:
            continue
        roles_by_pair.setdefault(pair_id, set()).add(role)
    return sum(1 for roles in roles_by_pair.values() if {0, 1}.issubset(roles))


def _trajectory_bc_compatible_training_config(
    *, structured_aux: Any, dataset_path_text: str, every_updates: int
) -> Any:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            trajectory_bc_dataset_path=dataset_path_text,
            trajectory_bc_every_updates=every_updates,
            trajectory_bc_aux_updates=int(getattr(structured_aux, "paired_outcome_preference_aux_updates", 1)),
            trajectory_bc_batch_episodes=int(getattr(structured_aux, "paired_outcome_preference_batch_episodes", 8)),
            trajectory_bc_seed=int(getattr(structured_aux, "paired_outcome_preference_seed", 20260520)),
            trajectory_bc_focus_source_labels=(),
            trajectory_bc_focus_fraction=0.0,
            trajectory_bc_focus_groups=(),
        )
    )


def _opponent_context_indices_for_episodes(
    model: Any,
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: list[int],
) -> np.ndarray | None:
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return None
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    if not opponent_ids:
        return None
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    indices = model.opponent_context_indices_for_policy_ids(selected_policy_ids)
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def _source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, dict) else None
        ids.append(str(raw_id or "").strip())
    return ids


def _preference_group_indices_for_episodes(
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: list[int],
) -> np.ndarray | None:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return None
    labels: list[str] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            labels.append("")
            continue
        labels.append(str(bundle.get("merge_source_dataset_label") or bundle.get("source_dataset_label") or ""))
    nonempty_labels = sorted({label for label in labels if label})
    if not nonempty_labels:
        return None
    label_to_index = {label: index for index, label in enumerate(nonempty_labels)}
    indices = [
        label_to_index.get(labels[int(index)] if int(index) < len(labels) else "", -1) for index in episode_indices
    ]
    return np.asarray(indices, dtype=np.int64)


def _initial_hidden_state(
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
    "PairedOutcomePreferenceReplayState",
    "maybe_run_paired_outcome_preference_replay",
    "paired_outcome_preference_complete_pair_count",
]
