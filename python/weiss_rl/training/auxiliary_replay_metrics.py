"""Shared metric accounting for post-update auxiliary replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState, metric_key_fragment


@dataclass(slots=True)
class AuxiliaryReplayMetricAccumulator:
    sampler: TrajectoryBcReplayState
    total_batch_episodes: int = 0
    total_focus_episodes: int = 0
    total_nonfocus_episodes: int = 0
    total_context_episodes: int = 0
    total_focus_group_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.total_focus_group_counts = {group.name: 0 for group in self.sampler.focus_groups}

    def record_sampled_episodes(self, episode_indices: list[int]) -> None:
        self.total_batch_episodes += len(episode_indices)
        self.total_focus_episodes += int(self.sampler.last_focus_episode_count)
        self.total_nonfocus_episodes += int(self.sampler.last_nonfocus_episode_count)
        for group in self.sampler.focus_groups:
            self.total_focus_group_counts[group.name] = self.total_focus_group_counts.get(group.name, 0) + int(
                group.last_episode_count
            )

    def record_context_indices(self, opponent_context_indices: np.ndarray | None) -> None:
        if opponent_context_indices is not None:
            self.total_context_episodes += int(np.count_nonzero(opponent_context_indices))

    def emit_common_metrics(
        self,
        latest_metrics: dict[str, float],
        *,
        prefix: str,
        include_focus: bool = False,
        include_context: bool = False,
    ) -> None:
        latest_metrics[f"{prefix}_aux_updates"] = float(self.sampler.aux_updates)
        latest_metrics[f"{prefix}_batch_episodes"] = float(self.total_batch_episodes)
        latest_metrics[f"{prefix}_dataset_train_rows"] = float(self.sampler.dataset.metadata["train_rows"])
        if include_focus:
            latest_metrics[f"{prefix}_focus_fraction"] = float(self.sampler.focus_fraction)
            latest_metrics[f"{prefix}_focus_batch_episodes"] = float(self.total_focus_episodes)
            latest_metrics[f"{prefix}_nonfocus_batch_episodes"] = float(self.total_nonfocus_episodes)
            self.emit_focus_group_metrics(latest_metrics, prefix=prefix)
        if include_context:
            latest_metrics[f"{prefix}_opponent_context_episodes"] = float(self.total_context_episodes)

    def emit_focus_group_metrics(self, latest_metrics: dict[str, float], *, prefix: str) -> None:
        if not self.sampler.focus_groups:
            return
        latest_metrics[f"{prefix}_focus_group_count"] = float(len(self.sampler.focus_groups))
        for group in self.sampler.focus_groups:
            key = metric_key_fragment(group.name)
            latest_metrics[f"{prefix}_focus_group_{key}_batch_episodes"] = float(
                self.total_focus_group_counts.get(group.name, 0)
            )


def emit_finite_aux_metrics(
    latest_metrics: dict[str, float],
    *,
    prefix: str,
    aux_metrics: dict[str, Any],
) -> None:
    for key, value in aux_metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            latest_metrics[f"{prefix}_{key}"] = float(value)


__all__ = [
    "AuxiliaryReplayMetricAccumulator",
    "emit_finite_aux_metrics",
]
