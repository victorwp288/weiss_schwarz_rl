from __future__ import annotations

import weiss_rl.training.replay_data.training_replay_paths as training_replay_paths


def test_training_replay_path_builder_preserves_order_and_injected_runners() -> None:
    def trajectory_bc_runner(**_kwargs: object) -> None:
        return None

    def paired_swing_runner(**_kwargs: object) -> None:
        return None

    def paired_outcome_preference_runner(**_kwargs: object) -> None:
        return None

    paths = training_replay_paths.build_post_update_replay_paths(
        trajectory_bc_runner=trajectory_bc_runner,
        paired_swing_runner=paired_swing_runner,
        paired_outcome_preference_runner=paired_outcome_preference_runner,
    )

    assert [path.runner for path in paths] == [
        trajectory_bc_runner,
        paired_swing_runner,
        paired_outcome_preference_runner,
    ]
    assert training_replay_paths.post_update_replay_path_specs(paths) == (
        ("trajectory_bc_replay", "trajectory_bc", "maybe_run_trajectory_bc_replay", True),
        ("paired_swing_replay", "paired_swing", "maybe_run_paired_swing_replay", False),
        (
            "paired_outcome_preference_replay",
            "paired_outcome_preference",
            "maybe_run_paired_outcome_preference_replay",
            False,
        ),
    )
