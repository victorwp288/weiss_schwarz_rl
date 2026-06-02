"""Replay-state lifecycle and post-update dispatch for training."""

from __future__ import annotations

from typing import Any

import torch

from weiss_rl.training.replay_data.paired_outcome_preference_replay import (
    maybe_run_paired_outcome_preference_replay as _maybe_run_paired_outcome_preference_replay,
)
from weiss_rl.training.replay_data.paired_swing_replay import (
    maybe_run_paired_swing_replay as _maybe_run_paired_swing_replay,
)
from weiss_rl.training.replay_data.training_replay_paths import (
    PostUpdateReplayPath,
    build_post_update_replay_paths,
)
from weiss_rl.training.replay_data.training_replay_paths import (
    post_update_replay_path_specs as _post_update_replay_path_specs,
)
from weiss_rl.training.replay_data.training_replay_states import (
    TrainingReplayStates,
    reset_policy_anchor_for_fresh_preference_replay,
    reset_policy_anchor_to_current_model,
    training_replay_states_from_config,
)
from weiss_rl.training.replay_data.trajectory_bc_replay import (
    maybe_run_trajectory_bc_replay as _maybe_run_trajectory_bc_replay,
)


def post_update_replay_paths() -> tuple[PostUpdateReplayPath, ...]:
    return build_post_update_replay_paths(
        trajectory_bc_runner=maybe_run_trajectory_bc_replay,
        paired_swing_runner=maybe_run_paired_swing_replay,
        paired_outcome_preference_runner=maybe_run_paired_outcome_preference_replay,
    )


def post_update_replay_path_specs() -> tuple[tuple[str, str, str, bool], ...]:
    return _post_update_replay_path_specs(post_update_replay_paths())


def maybe_run_trajectory_bc_replay(**kwargs: Any) -> None:
    _maybe_run_trajectory_bc_replay(**kwargs)


def maybe_run_paired_swing_replay(**kwargs: Any) -> None:
    _maybe_run_paired_swing_replay(**kwargs)


def maybe_run_paired_outcome_preference_replay(**kwargs: Any) -> None:
    _maybe_run_paired_outcome_preference_replay(**kwargs)


def run_post_update_replay(
    *,
    replay_states: TrainingReplayStates,
    learner: Any,
    training_config: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
    profile_timers: bool,
    learner_torch_threads: int | None,
    profile_block: Any,
    torch_num_threads_scope: Any,
) -> None:
    for path in post_update_replay_paths():
        _run_post_update_replay_path(
            path=path,
            replay_states=replay_states,
            learner=learner,
            training_config=training_config,
            device=device,
            update_count=update_count,
            latest_metrics=latest_metrics,
            profile_timers=profile_timers,
            learner_torch_threads=learner_torch_threads,
            profile_block=profile_block,
            torch_num_threads_scope=torch_num_threads_scope,
        )


def _run_post_update_replay_path(
    *,
    path: PostUpdateReplayPath,
    replay_states: TrainingReplayStates,
    learner: Any,
    training_config: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
    profile_timers: bool,
    learner_torch_threads: int | None,
    profile_block: Any,
    torch_num_threads_scope: Any,
) -> None:
    with (
        profile_block(profile_timers, path.metric_name),
        torch_num_threads_scope(learner_torch_threads),
    ):
        kwargs: dict[str, Any] = {
            "state": getattr(replay_states, path.state_attr),
            "learner": learner,
            "device": device,
            "update_count": update_count,
            "latest_metrics": latest_metrics,
        }
        if path.include_training_config:
            kwargs["training_config"] = training_config
        path.runner(**kwargs)


__all__ = [
    "PostUpdateReplayPath",
    "TrainingReplayStates",
    "post_update_replay_path_specs",
    "post_update_replay_paths",
    "reset_policy_anchor_for_fresh_preference_replay",
    "reset_policy_anchor_to_current_model",
    "run_post_update_replay",
    "training_replay_states_from_config",
]
