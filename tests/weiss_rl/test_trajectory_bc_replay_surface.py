from __future__ import annotations

import weiss_rl.training.replay_data.trajectory_bc_replay as trajectory_bc_replay


def test_trajectory_bc_replay_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "trajectory_bc_replay")
    assert not hasattr(training, "trajectory_bc_sampling")
    assert not hasattr(training, "trajectory_bc_teacher_state")


def test_trajectory_bc_replay_does_not_reexport_sampler_state() -> None:
    assert trajectory_bc_replay.__all__ == ["maybe_run_trajectory_bc_replay"]
    assert not hasattr(trajectory_bc_replay, "TrajectoryBcReplayFocusGroupState")
    assert not hasattr(trajectory_bc_replay, "TrajectoryBcReplayState")
