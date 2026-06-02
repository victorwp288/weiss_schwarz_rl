"""Paired-outcome preference dataset metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


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


def preference_group_indices_for_episodes(
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


__all__ = [
    "paired_outcome_preference_complete_pair_count",
    "preference_group_indices_for_episodes",
]
