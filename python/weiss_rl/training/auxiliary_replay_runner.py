"""Shared execution loop for in-training auxiliary replay updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_metrics import AuxiliaryReplayMetricAccumulator, emit_finite_aux_metrics
from weiss_rl.training.auxiliary_replay_support import initial_hidden_state, opponent_context_indices_for_episodes
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

ReplayBatchFactory = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AuxiliaryReplayBatchContext:
    """Per-aux-update context derived before a replay batch is handed to a learner."""

    episode_indices: list[int]
    opponent_context_indices: np.ndarray | None


@dataclass(frozen=True, slots=True)
class AuxiliaryReplayRunResult:
    """Metrics from a scheduled auxiliary replay run."""

    aux_metrics: dict[str, float]
    sampled_metrics: AuxiliaryReplayMetricAccumulator


def auxiliary_replay_update_is_due(*, every_updates: int, update_count: int) -> bool:
    return int(update_count) > 0 and int(update_count) % int(every_updates) == 0


def auxiliary_replay_sampler_is_due(sampler: TrajectoryBcReplayState, *, update_count: int) -> bool:
    return auxiliary_replay_update_is_due(every_updates=int(sampler.every_updates), update_count=update_count)


def require_auxiliary_replay_updater(learner: Any, *, method_name: str, error_message: str) -> Callable[..., Any]:
    updater = getattr(learner, method_name, None)
    if not callable(updater):
        raise ValueError(error_message)
    return updater


def run_auxiliary_replay_updates(
    *,
    sampler: TrajectoryBcReplayState,
    learner: Any,
    device: torch.device,
    update_batch: Callable[[dict[str, Any], AuxiliaryReplayBatchContext], Mapping[str, float]],
    batch_factory: ReplayBatchFactory = replay_trajectory_bc_batch,
    use_opponent_context: bool = False,
) -> AuxiliaryReplayRunResult:
    """Sample replay episodes, build learner batches, and apply auxiliary updates."""

    aux_metrics: dict[str, float] = {}
    sampled_metrics = AuxiliaryReplayMetricAccumulator(sampler)
    for _ in range(int(sampler.aux_updates)):
        episode_indices = sampler.next_episode_indices()
        sampled_metrics.record_sampled_episodes(episode_indices)
        model = learner.model
        opponent_context_indices = _opponent_context_indices(
            model=model,
            dataset=sampler.dataset,
            episode_indices=episode_indices,
            enabled=use_opponent_context,
        )
        if use_opponent_context:
            sampled_metrics.record_context_indices(opponent_context_indices)
        hidden = initial_hidden_state(
            model,
            batch_size=len(episode_indices),
            device=device,
            opponent_context_indices=opponent_context_indices,
        )
        batch_kwargs: dict[str, Any] = {
            "episode_indices": episode_indices,
            "initial_hidden_state": hidden,
        }
        if use_opponent_context:
            batch_kwargs["opponent_context_indices"] = opponent_context_indices
        batch = batch_factory(sampler.dataset, **batch_kwargs)
        context = AuxiliaryReplayBatchContext(
            episode_indices=episode_indices,
            opponent_context_indices=opponent_context_indices,
        )
        aux_metrics = dict(update_batch(batch, context))
    return AuxiliaryReplayRunResult(aux_metrics=aux_metrics, sampled_metrics=sampled_metrics)


def emit_auxiliary_replay_run_metrics(
    latest_metrics: dict[str, float],
    *,
    prefix: str,
    replay_result: AuxiliaryReplayRunResult,
    include_focus: bool = False,
    include_context: bool = False,
) -> None:
    emit_auxiliary_replay_sampled_metrics(
        latest_metrics,
        prefix=prefix,
        include_focus=include_focus,
        include_context=include_context,
        replay_result=replay_result,
    )
    emit_auxiliary_replay_aux_metrics(latest_metrics, prefix=prefix, replay_result=replay_result)


def emit_auxiliary_replay_sampled_metrics(
    latest_metrics: dict[str, float],
    *,
    prefix: str,
    replay_result: AuxiliaryReplayRunResult,
    include_focus: bool = False,
    include_context: bool = False,
) -> None:
    replay_result.sampled_metrics.emit_common_metrics(
        latest_metrics,
        prefix=prefix,
        include_focus=include_focus,
        include_context=include_context,
    )


def emit_auxiliary_replay_aux_metrics(
    latest_metrics: dict[str, float],
    *,
    prefix: str,
    replay_result: AuxiliaryReplayRunResult,
) -> None:
    emit_finite_aux_metrics(
        latest_metrics,
        prefix=prefix,
        aux_metrics=replay_result.aux_metrics,
    )


def _opponent_context_indices(
    *,
    model: Any,
    dataset: ReplayTrajectoryDataset,
    episode_indices: list[int],
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    return opponent_context_indices_for_episodes(
        model,
        dataset,
        episode_indices=episode_indices,
    )


__all__ = [
    "AuxiliaryReplayBatchContext",
    "AuxiliaryReplayRunResult",
    "ReplayBatchFactory",
    "auxiliary_replay_sampler_is_due",
    "auxiliary_replay_update_is_due",
    "emit_auxiliary_replay_aux_metrics",
    "emit_auxiliary_replay_run_metrics",
    "emit_auxiliary_replay_sampled_metrics",
    "require_auxiliary_replay_updater",
    "run_auxiliary_replay_updates",
]
