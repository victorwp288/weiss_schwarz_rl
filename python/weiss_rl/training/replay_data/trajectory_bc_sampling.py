"""Trajectory replay dataset sampling and focus-group selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.trajectory_bc_focus import (
    TrajectoryBcReplayFocusGroupState,
    episode_indices_by_source_label,
    focus_group_configs_from_structured_aux,
    focus_group_counts,
    focus_groups_by_source_label,
    metric_key_fragment,
    source_labels_by_episode,
    take_from_order,
)


@dataclass(slots=True)
class TrajectoryBcReplayState:
    dataset: ReplayTrajectoryDataset
    rng: np.random.Generator
    batch_episodes: int
    aux_updates: int
    every_updates: int
    order: np.ndarray
    focus_source_labels: tuple[str, ...] = ()
    focus_fraction: float = 0.0
    focus_indices: np.ndarray | None = None
    nonfocus_indices: np.ndarray | None = None
    focus_order: np.ndarray | None = None
    nonfocus_order: np.ndarray | None = None
    focus_groups: tuple[TrajectoryBcReplayFocusGroupState, ...] = ()
    cursor: int = 0
    focus_cursor: int = 0
    nonfocus_cursor: int = 0
    last_focus_episode_count: int = 0
    last_nonfocus_episode_count: int = 0

    @classmethod
    def from_training_config(cls, training_config: Any, *, repo_root: Path) -> TrajectoryBcReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "trajectory_bc_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "trajectory_bc_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        dataset_path = Path(dataset_path_text)
        if not dataset_path.is_absolute():
            dataset_path = Path(repo_root) / dataset_path
        dataset = load_replay_trajectory_bc_dataset(dataset_path)
        if int(dataset.metadata.get("train_rows", 0)) <= 0:
            raise ValueError(f"trajectory BC dataset has no trainable rows: {dataset_path}")
        rng = np.random.default_rng(int(getattr(structured_aux, "trajectory_bc_seed", 20260516)))
        order = rng.permutation(dataset.episode_count)
        focus_source_labels = tuple(
            str(label).strip()
            for label in getattr(structured_aux, "trajectory_bc_focus_source_labels", ())
            if str(label).strip()
        )
        focus_fraction = float(getattr(structured_aux, "trajectory_bc_focus_fraction", 0.0))
        if focus_fraction < 0.0 or focus_fraction > 1.0:
            raise ValueError("trajectory_bc_focus_fraction must be between 0.0 and 1.0")
        focus_group_configs = focus_group_configs_from_structured_aux(structured_aux)
        focus_groups, grouped_nonfocus_indices = focus_groups_by_source_label(
            dataset,
            focus_groups=focus_group_configs,
            dataset_path=dataset_path,
            rng=rng,
        )
        if focus_groups:
            focus_labels = tuple(label for group in focus_groups for label in group.source_labels)
            focus_source_labels = focus_labels
            focus_fraction = sum(group.fraction for group in focus_groups)
            focus_indices = np.concatenate([group.indices for group in focus_groups], axis=0)
            nonfocus_indices = grouped_nonfocus_indices
            focus_order = None
            nonfocus_order = None if nonfocus_indices is None else rng.permutation(nonfocus_indices)
        else:
            focus_indices, nonfocus_indices = episode_indices_by_source_label(
                dataset,
                source_labels=focus_source_labels,
                dataset_path=dataset_path,
            )
            focus_order = None if focus_indices is None else rng.permutation(focus_indices)
            nonfocus_order = None if nonfocus_indices is None else rng.permutation(nonfocus_indices)
        return cls(
            dataset=dataset,
            rng=rng,
            batch_episodes=int(getattr(structured_aux, "trajectory_bc_batch_episodes", 8)),
            aux_updates=int(getattr(structured_aux, "trajectory_bc_aux_updates", 1)),
            every_updates=every_updates,
            order=order,
            focus_source_labels=focus_source_labels,
            focus_fraction=focus_fraction,
            focus_indices=focus_indices,
            nonfocus_indices=nonfocus_indices,
            focus_order=focus_order,
            nonfocus_order=nonfocus_order,
            focus_groups=focus_groups,
        )

    def next_episode_indices(self) -> list[int]:
        if self.focus_groups:
            return self._next_grouped_episode_indices()
        if self.focus_indices is not None and self.focus_fraction > 0.0:
            return self._next_stratified_episode_indices()
        if self.cursor >= int(self.order.shape[0]):
            self.order = self.rng.permutation(self.dataset.episode_count)
            self.cursor = 0
        end = min(self.cursor + int(self.batch_episodes), int(self.order.shape[0]))
        indices = self.order[self.cursor : end].astype(np.int64).tolist()
        self.cursor = end
        self.last_focus_episode_count = 0
        self.last_nonfocus_episode_count = len(indices)
        if indices:
            return indices
        return self.next_episode_indices()

    def _next_stratified_episode_indices(self) -> list[int]:
        batch_size = int(self.batch_episodes)
        if batch_size <= 0:
            raise ValueError("trajectory BC batch_episodes must be >= 1")
        focus_available = 0 if self.focus_indices is None else int(self.focus_indices.shape[0])
        nonfocus_available = 0 if self.nonfocus_indices is None else int(self.nonfocus_indices.shape[0])
        focus_count = min(focus_available, int(np.ceil(batch_size * float(self.focus_fraction))))
        if focus_count <= 0 and focus_available > 0:
            focus_count = 1
        nonfocus_count = batch_size - focus_count
        if nonfocus_available <= 0:
            focus_count = batch_size
            nonfocus_count = 0
        elif nonfocus_count > nonfocus_available and focus_available > focus_count:
            extra_focus = min(focus_available - focus_count, nonfocus_count - nonfocus_available)
            focus_count += extra_focus
            nonfocus_count -= extra_focus
        focus = self._take_focus_episodes(focus_count)
        nonfocus = self._take_nonfocus_episodes(nonfocus_count)
        indices = [*focus, *nonfocus]
        if not indices:
            raise ValueError("trajectory BC stratified sampler produced no episode indices")
        self.rng.shuffle(indices)
        self.last_focus_episode_count = len(focus)
        self.last_nonfocus_episode_count = len(nonfocus)
        return [int(index) for index in indices]

    def _next_grouped_episode_indices(self) -> list[int]:
        batch_size = int(self.batch_episodes)
        if batch_size <= 0:
            raise ValueError("trajectory BC batch_episodes must be >= 1")
        target_focus_count = min(batch_size, int(np.ceil(batch_size * float(self.focus_fraction))))
        nonfocus_available = 0 if self.nonfocus_indices is None else int(self.nonfocus_indices.shape[0])
        if nonfocus_available <= 0:
            target_focus_count = batch_size
        group_counts = focus_group_counts(
            batch_size=batch_size,
            target_focus_count=target_focus_count,
            fractions=tuple(float(group.fraction) for group in self.focus_groups),
        )
        focus: list[int] = []
        for group, count in zip(self.focus_groups, group_counts, strict=True):
            taken = self._take_focus_group_episodes(group, int(count))
            group.last_episode_count = len(taken)
            focus.extend(taken)
        nonfocus_count = batch_size - len(focus)
        nonfocus = self._take_nonfocus_episodes(nonfocus_count)
        indices = [*focus, *nonfocus]
        if not indices:
            raise ValueError("trajectory BC grouped sampler produced no episode indices")
        self.rng.shuffle(indices)
        self.last_focus_episode_count = len(focus)
        self.last_nonfocus_episode_count = len(nonfocus)
        return [int(index) for index in indices]

    def _take_focus_group_episodes(self, group: TrajectoryBcReplayFocusGroupState, count: int) -> list[int]:
        if count <= 0:
            return []
        taken, group.order, group.cursor = take_from_order(
            order=group.order,
            cursor=group.cursor,
            count=int(count),
            source_indices=group.indices,
            rng=self.rng,
        )
        return taken

    def _take_focus_episodes(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if self.focus_indices is None or int(self.focus_indices.shape[0]) <= 0:
            return []
        if self.focus_order is None or int(self.focus_order.shape[0]) <= 0:
            self.focus_order = self.rng.permutation(self.focus_indices)
            self.focus_cursor = 0
        taken, self.focus_order, self.focus_cursor = take_from_order(
            order=self.focus_order,
            cursor=self.focus_cursor,
            count=int(count),
            source_indices=self.focus_indices,
            rng=self.rng,
        )
        return taken

    def _take_nonfocus_episodes(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if self.nonfocus_indices is None or int(self.nonfocus_indices.shape[0]) <= 0:
            return []
        if self.nonfocus_order is None or int(self.nonfocus_order.shape[0]) <= 0:
            self.nonfocus_order = self.rng.permutation(self.nonfocus_indices)
            self.nonfocus_cursor = 0
        taken, self.nonfocus_order, self.nonfocus_cursor = take_from_order(
            order=self.nonfocus_order,
            cursor=self.nonfocus_cursor,
            count=int(count),
            source_indices=self.nonfocus_indices,
            rng=self.rng,
        )
        return taken


__all__ = [
    "TrajectoryBcReplayFocusGroupState",
    "TrajectoryBcReplayState",
    "episode_indices_by_source_label",
    "focus_group_configs_from_structured_aux",
    "focus_group_counts",
    "focus_groups_by_source_label",
    "metric_key_fragment",
    "source_labels_by_episode",
    "take_from_order",
]
