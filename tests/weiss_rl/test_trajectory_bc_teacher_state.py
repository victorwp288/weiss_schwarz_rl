from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import weiss_rl.training.replay_data.trajectory_bc_replay as trajectory_bc_replay
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.trajectory_bc_replay import maybe_run_trajectory_bc_replay
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState
from weiss_rl.training.replay_data.trajectory_bc_teacher_state import (
    apply_trajectory_bc_teacher_aux_state,
    capture_teacher_aux_state,
    restore_teacher_aux_state,
)

from .trajectory_bc_replay_test_support import (
    ReplayLearner,
    dataset_with_source_labels,
    trajectory_bc_training_config,
)


def test_trajectory_bc_teacher_state_round_trips_public_guidance_fields() -> None:
    learner = ReplayLearner()
    learner.teacher_aux_mode = "always"
    learner.set_teacher_aux_coefs(
        family=0.11,
        slot=0.12,
        hand=0.13,
        move_source=0.14,
        attack_type=0.15,
        action=0.16,
        same_family_action=0.17,
        action_margin=0.18,
        action_margin_value=0.19,
        same_family_action_margin=0.20,
        same_family_action_margin_value=0.21,
        exact_action_families=("play", "attack"),
        public_heuristic=0.22,
        public_heuristic_temperature=4.0,
        public_nonpass_over_pass=0.23,
        public_nonpass_over_pass_margin=0.24,
        public_heuristic_families=("main_play_character",),
        public_heuristic_profiles=("base", "aggressive"),
        public_heuristic_profile_mode="cycle",
        public_heuristic_profiles_end_updates=30,
    )
    captured = capture_teacher_aux_state(learner)

    apply_trajectory_bc_teacher_aux_state(
        learner,
        SimpleNamespace(
            trajectory_bc_teacher_family_coef=0.31,
            trajectory_bc_teacher_slot_coef=0.32,
            trajectory_bc_teacher_move_source_coef=0.33,
            trajectory_bc_teacher_attack_type_coef=0.34,
            trajectory_bc_teacher_action_coef=0.35,
            trajectory_bc_teacher_same_family_action_coef=0.36,
            trajectory_bc_teacher_same_family_action_margin_coef=0.37,
            trajectory_bc_teacher_same_family_action_margin=0.38,
        ),
    )

    assert learner.teacher_aux_mode == "warmstart_only"
    assert learner.teacher_family_coef == pytest.approx(0.31)
    assert learner.teacher_hand_coef == pytest.approx(0.13)
    assert learner.teacher_public_heuristic_profiles == ("base", "aggressive")

    restore_teacher_aux_state(learner, captured)

    assert capture_teacher_aux_state(learner) == captured


def test_trajectory_bc_replay_restores_teacher_state_after_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(dataset_path, dataset_with_source_labels(["old", "repair_a"]))
    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(dataset_path=dataset_path, batch_episodes=2),
        repo_root=tmp_path,
    )
    assert state is not None
    monkeypatch.setattr(
        trajectory_bc_replay,
        "replay_trajectory_bc_batch",
        lambda dataset, *, episode_indices, initial_hidden_state: {"episode_indices": tuple(episode_indices)},
    )
    learner = ReplayLearner()
    learner.teacher_aux_mode = "always"
    learner.set_teacher_aux_coefs(
        family=0.41,
        slot=0.42,
        hand=0.43,
        action_margin_value=0.44,
        public_heuristic=0.45,
        public_heuristic_temperature=3.0,
        public_heuristic_profiles=("base",),
        public_heuristic_profile_mode="mixture",
        public_heuristic_profiles_end_updates=12,
    )
    previous = capture_teacher_aux_state(learner)

    maybe_run_trajectory_bc_replay(
        state=state,
        learner=learner,
        training_config=trajectory_bc_training_config(dataset_path=dataset_path, batch_episodes=2),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics={},
    )

    assert capture_teacher_aux_state(learner) == previous
