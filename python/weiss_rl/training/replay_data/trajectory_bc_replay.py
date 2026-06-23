"""In-training replay trajectory behavior-cloning regularizer."""

from __future__ import annotations

from typing import Any

import torch

from weiss_rl.replay.trajectory_bc import replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_runner import (
    auxiliary_replay_sampler_is_due,
    emit_auxiliary_replay_run_metrics,
    run_auxiliary_replay_updates,
)
from weiss_rl.training.replay_data import trajectory_bc_sampling as _trajectory_bc_sampling
from weiss_rl.training.replay_data.trajectory_bc_teacher_state import (
    apply_trajectory_bc_teacher_aux_state,
    capture_teacher_aux_state,
    restore_teacher_aux_state,
)


def maybe_run_trajectory_bc_replay(
    *,
    state: _trajectory_bc_sampling.TrajectoryBcReplayState | None,
    learner: Any,
    training_config: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured replay-BC auxiliary steps after an RL update."""

    if state is None:
        return
    if not auxiliary_replay_sampler_is_due(state, update_count=update_count):
        return
    structured_aux = training_config.structured_aux
    previous = capture_teacher_aux_state(learner)
    apply_trajectory_bc_teacher_aux_state(learner, structured_aux)
    try:
        replay_result = run_auxiliary_replay_updates(
            sampler=state,
            learner=learner,
            device=device,
            update_batch=lambda batch, _context: learner.auxiliary_update(batch),
            batch_factory=replay_trajectory_bc_batch,
        )
        emit_auxiliary_replay_run_metrics(
            latest_metrics,
            prefix="trajectory_bc_replay",
            replay_result=replay_result,
            include_focus=True,
        )
    finally:
        restore_teacher_aux_state(learner, previous)


__all__ = [
    "maybe_run_trajectory_bc_replay",
]
