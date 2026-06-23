from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def build_training_config(
    *,
    dataset_path: Path,
    batch_episodes: int = 2,
    aux_updates: int = 1,
    focus_groups: tuple[SimpleNamespace, ...] = (),
    conflict_filter: str = "none",
    compare_to: str = "negative",
) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            paired_swing_dataset_path=dataset_path.as_posix(),
            paired_swing_every_updates=1,
            paired_swing_aux_updates=aux_updates,
            paired_swing_batch_episodes=batch_episodes,
            paired_swing_seed=7,
            paired_swing_focus_source_labels=(),
            paired_swing_focus_fraction=0.0,
            paired_swing_focus_groups=focus_groups,
            paired_swing_margin=0.25,
            paired_swing_coef=0.05,
            paired_swing_positive_action_source="teacher_action",
            paired_swing_negative_action_source="actions",
            paired_swing_conflict_filter=conflict_filter,
            paired_swing_loss_scope="row",
            paired_swing_compare_to=compare_to,
        )
    )


class ContextModel:
    def __init__(self) -> None:
        self.last_initial_context = np.zeros((0,), dtype=np.int64)

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> np.ndarray:
        return np.asarray(
            [3 if str(policy_id) == "B2 HeuristicPublic" else 7 for policy_id in policy_ids],
            dtype=np.int64,
        )

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_context_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        self.last_initial_context = np.asarray(opponent_context_indices, dtype=np.int64).reshape(-1)
        return torch.zeros((1, 2, int(batch_size), 4), device=device)


class ReplayLearner:
    def __init__(self, model: object | None = None) -> None:
        self.model = model
        self.last_batch: object | None = None

    def paired_swing_update(self, batch: object, **kwargs: object) -> dict[str, float]:
        self.last_batch = batch
        assert batch
        assert kwargs["positive_action_source"] == "teacher_action"
        assert kwargs["negative_action_source"] == "actions"
        assert kwargs["loss_scope"] == "row"
        assert kwargs["compare_to"] == "negative"
        return {"loss": 0.125, "paired_swing_rows": 2.0}


def dataset_with_action_pairs(
    *,
    labels: list[str],
    actions: list[int],
    teacher_actions: list[int],
    legal_rows: list[list[int]] | None = None,
    opponents: list[str] | None = None,
) -> ReplayTrajectoryDataset:
    time_steps = 1
    episode_count = len(labels)
    assert len(actions) == episode_count
    assert len(teacher_actions) == episode_count
    opponent_ids = opponents or ["" for _ in labels]
    legal_row_values = legal_rows or [[1, 2] for _ in labels]
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    offsets = [0]
    cursor = 0
    for row in legal_row_values:
        row_ids = np.asarray(row, dtype=np.uint32)
        legal_ids_parts.append(row_ids)
        legal_meta_parts.append(np.zeros((row_ids.shape[0], 4), dtype=np.uint16))
        cursor += int(row_ids.shape[0])
        offsets.append(cursor)
    valid = np.ones((time_steps, episode_count), dtype=np.bool_)
    metadata = {
        "format": "weiss_rl_replay_trajectory_bc_v1",
        "train_rows": episode_count,
        "selected_bundles": [
            {
                "source_dataset_label": label,
                "source_opponent_policy_id": opponent_ids[index],
                "pair_index": index,
                "swap_index": 0,
                "outcome": "W",
            }
            for index, label in enumerate(labels)
        ],
    }
    return ReplayTrajectoryDataset(
        obs=np.zeros((time_steps, episode_count, 4), dtype=np.float32),
        actor=np.zeros((time_steps, episode_count), dtype=np.int64),
        to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int64),
        actions=np.asarray([actions], dtype=np.int64),
        legal_ids=np.concatenate(legal_ids_parts).astype(np.uint32),
        legal_offsets=np.asarray(offsets, dtype=np.uint32),
        legal_action_meta=np.concatenate(legal_meta_parts, axis=0).astype(np.uint16),
        teacher_family=np.ones((time_steps, episode_count), dtype=np.int32),
        teacher_slot=np.ones((time_steps, episode_count), dtype=np.int32),
        teacher_move_source=np.ones((time_steps, episode_count), dtype=np.int32),
        teacher_attack_type=np.ones((time_steps, episode_count), dtype=np.int32),
        teacher_action=np.asarray([teacher_actions], dtype=np.int32),
        teacher_valid=valid,
        policy_train_mask=valid,
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata=metadata,
    )
