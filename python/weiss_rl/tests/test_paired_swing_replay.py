from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset
from weiss_rl.training.paired_swing_replay import (
    PairedSwingReplayState,
    filter_paired_swing_conflict_rows,
    maybe_run_paired_swing_replay,
    paired_swing_distinct_train_row_count,
)


def test_paired_swing_replay_state_reuses_grouped_sampler_and_counts_distinct_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_action_pairs(
            labels=["old", "fixed", "hardneg", "fixed", "old", "hardneg"],
            actions=[1, 1, 1, 1, 1, 1],
            teacher_actions=[1, 2, 2, 2, 1, 2],
        ),
    )

    state = PairedSwingReplayState.from_training_config(
        _training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            focus_groups=(
                SimpleNamespace(name="fixed", source_labels=("fixed",), fraction=0.25),
                SimpleNamespace(name="hardneg", source_labels=("hardneg",), fraction=0.25),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    assert state.distinct_train_rows == 4
    indices = state.sampler.next_episode_indices()
    labels = [state.sampler.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 4
    assert sum(label == "fixed" for label in labels) == 1
    assert sum(label == "hardneg" for label in labels) == 1


def test_paired_swing_replay_rejects_dataset_without_distinct_action_pairs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_action_pairs(labels=["old", "old"], actions=[1, 1], teacher_actions=[1, 1]),
    )

    with pytest.raises(ValueError, match="no trainable rows where positive and negative actions differ"):
        PairedSwingReplayState.from_training_config(
            _training_config(dataset_path=dataset_path),
            repo_root=tmp_path,
        )


def test_maybe_run_paired_swing_replay_reports_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_action_pairs(labels=["fixed", "hardneg"], actions=[1, 1], teacher_actions=[2, 2]),
    )
    state = PairedSwingReplayState.from_training_config(
        _training_config(dataset_path=dataset_path, batch_episodes=2),
        repo_root=tmp_path,
    )
    assert state is not None
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=_ReplayLearner(),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["paired_swing_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_swing_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_dataset_distinct_train_rows"] == 2.0
    assert latest_metrics["paired_swing_replay_loss_scope_episode_mean"] == 0.0
    assert latest_metrics["paired_swing_replay_loss"] == pytest.approx(0.125)
    assert latest_metrics["paired_swing_replay_paired_swing_rows"] == 2.0


def test_maybe_run_paired_swing_replay_threads_opponent_context(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing_context.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_action_pairs(
            labels=["fixed", "hardneg"],
            actions=[1, 1],
            teacher_actions=[2, 2],
            opponents=["B2 HeuristicPublic", "seed_c3aac2f9dc_policy_000005"],
        ),
    )
    state = PairedSwingReplayState.from_training_config(
        _training_config(dataset_path=dataset_path, batch_episodes=2),
        repo_root=tmp_path,
    )
    assert state is not None
    model = _ContextModel()
    learner = _ReplayLearner(model=model)
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert learner.last_batch is not None
    context = np.asarray(learner.last_batch["opponent_context_index"], dtype=np.int64)
    source_label_id = np.asarray(learner.last_batch["source_label_id"], dtype=np.int64)
    assert context.shape == (1, 2)
    assert sorted(context.reshape(-1).tolist()) == [3, 7]
    assert source_label_id.shape == (1, 2)
    assert sorted(source_label_id.reshape(-1).tolist()) == [0, 1]
    assert sorted(model.last_initial_context.tolist()) == [3, 7]
    assert latest_metrics["paired_swing_replay_opponent_context_episodes"] == 2.0


def test_paired_swing_distinct_train_row_count_requires_legal_positive_and_negative_actions() -> None:
    dataset = _dataset_with_action_pairs(
        labels=["good", "bad"],
        actions=[1, 1],
        teacher_actions=[2, 3],
        legal_rows=[[1, 2], [1, 2]],
    )

    assert paired_swing_distinct_train_row_count(dataset) == 1


def test_paired_swing_conflict_filter_masks_current_state_reverse_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing_conflict.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_action_pairs(
            labels=["fixed", "learned", "safe"],
            actions=[104, 124, 1],
            teacher_actions=[124, 104, 2],
            legal_rows=[[104, 124], [104, 124], [1, 2]],
        ),
    )

    state = PairedSwingReplayState.from_training_config(
        _training_config(dataset_path=dataset_path, conflict_filter="current_state"),
        repo_root=tmp_path,
    )

    assert state is not None
    assert state.distinct_train_rows == 1
    assert state.conflict_filter_summary is not None
    assert state.conflict_filter_summary["dropped_train_rows"] == 2
    assert state.conflict_filter_summary["kept_train_rows"] == 1
    assert int(np.count_nonzero(state.sampler.dataset.policy_train_mask)) == 1


def test_paired_swing_conflict_filter_rejects_unknown_mode() -> None:
    dataset = _dataset_with_action_pairs(labels=["safe"], actions=[1], teacher_actions=[2])

    with pytest.raises(ValueError, match="paired-swing conflict filter mode"):
        filter_paired_swing_conflict_rows(dataset, mode="timeline")


def _training_config(
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


class _ContextModel:
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


class _ReplayLearner:
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


def _dataset_with_action_pairs(
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
