from __future__ import annotations

import numpy as np
import torch
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState


class ContextModel:
    def __init__(self, policy_index_by_id: dict[str, int]) -> None:
        self.policy_index_by_id = policy_index_by_id
        self.initial_contexts: list[list[int]] = []

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> np.ndarray:
        return np.asarray([self.policy_index_by_id[policy_id] for policy_id in policy_ids], dtype=np.int64)

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_context_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        assert device == torch.device("cpu")
        assert opponent_context_indices is not None
        self.initial_contexts.append(opponent_context_indices.tolist())
        return torch.zeros((int(batch_size), 3), dtype=torch.float32, device=device)


class PlainModel:
    def __init__(self) -> None:
        self.initial_batch_sizes: list[int] = []

    def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        self.initial_batch_sizes.append(int(batch_size))
        return torch.zeros((int(batch_size), 2), dtype=torch.float32, device=device)


class Learner:
    def __init__(self, *, model: object) -> None:
        self.model = model


def sampler(
    opponent_policy_ids: list[str],
    *,
    batch_episodes: int,
    aux_updates: int,
    every_updates: int,
) -> TrajectoryBcReplayState:
    return TrajectoryBcReplayState(
        dataset=dataset_with_opponents(opponent_policy_ids),
        rng=np.random.default_rng(5),
        batch_episodes=batch_episodes,
        aux_updates=aux_updates,
        every_updates=every_updates,
        order=np.arange(len(opponent_policy_ids), dtype=np.int64),
    )


def dataset_with_opponents(opponent_policy_ids: list[str]) -> ReplayTrajectoryDataset:
    time_steps = 1
    episode_count = len(opponent_policy_ids)
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int64)
    to_play_seat = np.zeros((time_steps, episode_count), dtype=np.int64)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    teacher = np.ones((time_steps, episode_count), dtype=np.int32)
    valid = np.ones((time_steps, episode_count), dtype=np.bool_)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    metadata = {
        "format": "weiss_rl_replay_trajectory_bc_v1",
        "train_rows": episode_count,
        "selected_bundles": [
            {
                "source_dataset_label": f"episode_{index}",
                "source_opponent_policy_id": policy_id,
            }
            for index, policy_id in enumerate(opponent_policy_ids)
        ],
    }
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32),
        legal_action_meta=np.zeros((episode_count * 2, 3), dtype=np.uint16),
        teacher_family=teacher,
        teacher_slot=teacher,
        teacher_move_source=teacher,
        teacher_attack_type=teacher,
        teacher_action=teacher,
        teacher_valid=valid,
        policy_train_mask=valid,
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata=metadata,
    )
