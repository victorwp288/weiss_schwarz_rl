from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset
from weiss_rl.training.auxiliary_replay_support import (
    initial_hidden_state,
    opponent_context_indices_for_episodes,
    trajectory_bc_compatible_training_config,
)


def test_trajectory_bc_compatible_training_config_preserves_prefixed_replay_fields() -> None:
    group = SimpleNamespace(name="repair", source_labels=("fixed",), fraction=0.25)
    structured_aux = SimpleNamespace(
        paired_swing_aux_updates=3,
        paired_swing_batch_episodes=5,
        paired_swing_seed=11,
        paired_swing_focus_source_labels=("fixed",),
        paired_swing_focus_fraction=0.5,
        paired_swing_focus_groups=(group,),
    )

    config = trajectory_bc_compatible_training_config(
        structured_aux=structured_aux,
        dataset_path_text="data/swing.npz",
        every_updates=7,
        field_prefix="paired_swing",
        seed_default=20260519,
        include_focus_fields=True,
    )

    assert config.structured_aux.trajectory_bc_dataset_path == "data/swing.npz"
    assert config.structured_aux.trajectory_bc_every_updates == 7
    assert config.structured_aux.trajectory_bc_aux_updates == 3
    assert config.structured_aux.trajectory_bc_batch_episodes == 5
    assert config.structured_aux.trajectory_bc_seed == 11
    assert config.structured_aux.trajectory_bc_focus_source_labels == ("fixed",)
    assert config.structured_aux.trajectory_bc_focus_fraction == 0.5
    assert config.structured_aux.trajectory_bc_focus_groups == (group,)


def test_trajectory_bc_compatible_training_config_can_suppress_focus_fields() -> None:
    structured_aux = SimpleNamespace(
        paired_outcome_preference_aux_updates=2,
        paired_outcome_preference_batch_episodes=4,
        paired_outcome_preference_seed=13,
        paired_outcome_preference_focus_source_labels=("ignored",),
        paired_outcome_preference_focus_fraction=0.75,
    )

    config = trajectory_bc_compatible_training_config(
        structured_aux=structured_aux,
        dataset_path_text="data/preference.npz",
        every_updates=9,
        field_prefix="paired_outcome_preference",
        seed_default=20260520,
        include_focus_fields=False,
    )

    assert config.structured_aux.trajectory_bc_aux_updates == 2
    assert config.structured_aux.trajectory_bc_batch_episodes == 4
    assert config.structured_aux.trajectory_bc_seed == 13
    assert config.structured_aux.trajectory_bc_focus_source_labels == ()
    assert config.structured_aux.trajectory_bc_focus_fraction == 0.0
    assert config.structured_aux.trajectory_bc_focus_groups == ()


def test_opponent_context_indices_and_hidden_state_preserve_legacy_model_fallback() -> None:
    dataset = _dataset_with_opponents(["B2 HeuristicPublic", "seed_policy"])
    model = _LegacyContextModel()

    context = opponent_context_indices_for_episodes(model, dataset, episode_indices=[1, 0])
    hidden = initial_hidden_state(
        model,
        batch_size=2,
        device=torch.device("cpu"),
        opponent_context_indices=context,
    )

    assert context is not None
    assert context.tolist() == [7, 3]
    assert model.context_calls == [["seed_policy", "B2 HeuristicPublic"]]
    assert model.hidden_calls == [2]
    assert hidden is not None
    assert hidden.shape == (1, 2, 2, 4)


class _LegacyContextModel:
    def __init__(self) -> None:
        self.context_calls: list[list[str]] = []
        self.hidden_calls: list[int] = []

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> np.ndarray:
        self.context_calls.append(policy_ids)
        return np.asarray([3 if policy_id == "B2 HeuristicPublic" else 7 for policy_id in policy_ids], dtype=np.int64)

    def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        self.hidden_calls.append(int(batch_size))
        return torch.zeros((1, 2, int(batch_size), 4), device=device)


def _dataset_with_opponents(opponents: list[str]) -> ReplayTrajectoryDataset:
    episode_count = len(opponents)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    return ReplayTrajectoryDataset(
        obs=np.zeros((1, episode_count, 4), dtype=np.float32),
        actor=np.zeros((1, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int8),
        actions=np.ones((1, episode_count), dtype=np.int64),
        legal_ids=legal_ids,
        legal_offsets=np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32),
        legal_action_meta=np.zeros((legal_ids.shape[0], 3), dtype=np.uint16),
        teacher_family=np.ones((1, episode_count), dtype=np.int32),
        teacher_slot=np.ones((1, episode_count), dtype=np.int32),
        teacher_move_source=np.ones((1, episode_count), dtype=np.int32),
        teacher_attack_type=np.ones((1, episode_count), dtype=np.int32),
        teacher_action=np.ones((1, episode_count), dtype=np.int32),
        teacher_valid=np.ones((1, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((1, episode_count), dtype=np.bool_),
        reset_before_step=np.zeros((1, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "train_rows": episode_count,
            "selected_bundles": [
                {"source_dataset_label": f"episode_{index}", "source_opponent_policy_id": opponent}
                for index, opponent in enumerate(opponents)
            ],
        },
    )
