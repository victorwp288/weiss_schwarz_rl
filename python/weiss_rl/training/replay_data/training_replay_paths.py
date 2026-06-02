"""Post-update replay path registry for training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostUpdateReplayPath:
    metric_name: str
    state_attr: str
    runner_name: str
    runner: Callable[..., None]
    include_training_config: bool = False


def build_post_update_replay_paths(
    *,
    trajectory_bc_runner: Callable[..., None],
    paired_swing_runner: Callable[..., None],
    paired_outcome_preference_runner: Callable[..., None],
) -> tuple[PostUpdateReplayPath, ...]:
    return (
        PostUpdateReplayPath(
            metric_name="trajectory_bc_replay",
            state_attr="trajectory_bc",
            runner_name="maybe_run_trajectory_bc_replay",
            runner=trajectory_bc_runner,
            include_training_config=True,
        ),
        PostUpdateReplayPath(
            metric_name="paired_swing_replay",
            state_attr="paired_swing",
            runner_name="maybe_run_paired_swing_replay",
            runner=paired_swing_runner,
        ),
        PostUpdateReplayPath(
            metric_name="paired_outcome_preference_replay",
            state_attr="paired_outcome_preference",
            runner_name="maybe_run_paired_outcome_preference_replay",
            runner=paired_outcome_preference_runner,
        ),
    )


def post_update_replay_path_specs(paths: tuple[PostUpdateReplayPath, ...]) -> tuple[tuple[str, str, str, bool], ...]:
    return tuple((path.metric_name, path.state_attr, path.runner_name, path.include_training_config) for path in paths)


__all__ = [
    "PostUpdateReplayPath",
    "build_post_update_replay_paths",
    "post_update_replay_path_specs",
]
