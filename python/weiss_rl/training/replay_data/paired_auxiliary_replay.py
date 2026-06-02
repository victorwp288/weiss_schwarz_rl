"""Shared orchestration for configured paired auxiliary replay updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from weiss_rl.replay.trajectory_bc import replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_runner import (
    AuxiliaryReplayBatchContext,
    AuxiliaryReplayRunResult,
    ReplayBatchFactory,
    auxiliary_replay_sampler_is_due,
    emit_auxiliary_replay_aux_metrics,
    emit_auxiliary_replay_sampled_metrics,
    require_auxiliary_replay_updater,
    run_auxiliary_replay_updates,
)
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

AuxiliaryReplayUpdater = Callable[..., Any]
AuxiliaryReplayUpdateBatch = Callable[[dict[str, Any], AuxiliaryReplayBatchContext], Mapping[str, float]]
AuxiliaryReplayUpdateBatchFactory = Callable[[AuxiliaryReplayUpdater], AuxiliaryReplayUpdateBatch]


def run_due_paired_auxiliary_replay(
    *,
    state: Any | None,
    learner: Any,
    device: torch.device,
    update_count: int,
    updater_method_name: str,
    updater_error_message: str,
    make_update_batch: AuxiliaryReplayUpdateBatchFactory,
    batch_factory: ReplayBatchFactory = replay_trajectory_bc_batch,
    use_opponent_context: bool = True,
) -> AuxiliaryReplayRunResult | None:
    if state is None:
        return None
    sampler = _state_sampler(state)
    if not auxiliary_replay_sampler_is_due(sampler, update_count=update_count):
        return None
    updater = require_auxiliary_replay_updater(
        learner,
        method_name=updater_method_name,
        error_message=updater_error_message,
    )
    return run_auxiliary_replay_updates(
        sampler=sampler,
        learner=learner,
        device=device,
        update_batch=make_update_batch(updater),
        batch_factory=batch_factory,
        use_opponent_context=use_opponent_context,
    )


def emit_paired_auxiliary_replay_metrics(
    latest_metrics: dict[str, float],
    *,
    prefix: str,
    replay_result: AuxiliaryReplayRunResult,
    static_metrics: Mapping[str, float],
    include_focus: bool = False,
    include_context: bool = True,
) -> None:
    emit_auxiliary_replay_sampled_metrics(
        latest_metrics,
        prefix=prefix,
        replay_result=replay_result,
        include_focus=include_focus,
        include_context=include_context,
    )
    latest_metrics.update(static_metrics)
    emit_auxiliary_replay_aux_metrics(
        latest_metrics,
        prefix=prefix,
        replay_result=replay_result,
    )


def _state_sampler(state: Any) -> TrajectoryBcReplayState:
    sampler = getattr(state, "sampler", None)
    if not isinstance(sampler, TrajectoryBcReplayState):
        raise TypeError("paired auxiliary replay state must expose a TrajectoryBcReplayState sampler")
    return sampler


__all__ = [
    "AuxiliaryReplayUpdateBatch",
    "AuxiliaryReplayUpdateBatchFactory",
    "AuxiliaryReplayUpdater",
    "emit_paired_auxiliary_replay_metrics",
    "run_due_paired_auxiliary_replay",
]
