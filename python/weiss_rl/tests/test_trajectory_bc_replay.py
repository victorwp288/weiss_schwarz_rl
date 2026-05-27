from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset
from weiss_rl.training import trajectory_bc_replay
from weiss_rl.training.trajectory_bc_replay import TrajectoryBcReplayState, maybe_run_trajectory_bc_replay


def test_trajectory_bc_replay_stratifies_focus_source_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_labels(["old", "old", "repair_a", "repair_b", "old", "repair_a"]),
    )

    state = TrajectoryBcReplayState.from_training_config(
        _training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 4
    assert sum(label in {"repair_a", "repair_b"} for label in labels) == 2
    assert state.last_focus_episode_count == 2
    assert state.last_nonfocus_episode_count == 2


def test_trajectory_bc_replay_rejects_missing_focus_source_label(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(dataset_path, _dataset_with_labels(["old", "repair_a"]))

    with pytest.raises(ValueError, match="trajectory BC focus source labels not found"):
        TrajectoryBcReplayState.from_training_config(
            _training_config(
                dataset_path=dataset_path,
                focus_source_labels=("missing_repair",),
                focus_fraction=0.5,
            ),
            repo_root=tmp_path,
        )


def test_trajectory_bc_replay_metrics_accumulate_focus_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_labels(["old", "old", "repair_a", "repair_b", "old", "repair_a", "old", "repair_b"]),
    )
    state = TrajectoryBcReplayState.from_training_config(
        _training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            aux_updates=2,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        repo_root=tmp_path,
    )
    assert state is not None

    monkeypatch.setattr(
        trajectory_bc_replay,
        "replay_trajectory_bc_batch",
        lambda dataset, *, episode_indices, initial_hidden_state: {"episode_indices": tuple(episode_indices)},
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_trajectory_bc_replay(
        state=state,
        learner=_ReplayLearner(),
        training_config=_training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            aux_updates=2,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["trajectory_bc_replay_aux_updates"] == 2.0
    assert latest_metrics["trajectory_bc_replay_batch_episodes"] == 8.0
    assert latest_metrics["trajectory_bc_replay_focus_batch_episodes"] == 4.0
    assert latest_metrics["trajectory_bc_replay_nonfocus_batch_episodes"] == 4.0


def test_trajectory_bc_replay_reserves_named_focus_group_fractions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_labels(
            [
                "old",
                "learned_a",
                "fixed_a",
                "old",
                "learned_b",
                "fixed_b",
                "old",
                "old",
            ]
        ),
    )

    state = TrajectoryBcReplayState.from_training_config(
        _training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            focus_groups=(
                SimpleNamespace(name="learned", source_labels=("learned_a", "learned_b"), fraction=0.25),
                SimpleNamespace(name="fixed", source_labels=("fixed_a", "fixed_b"), fraction=0.25),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 8
    assert sum(label in {"learned_a", "learned_b"} for label in labels) == 2
    assert sum(label in {"fixed_a", "fixed_b"} for label in labels) == 2
    assert sum(label == "old" for label in labels) == 4
    assert state.last_focus_episode_count == 4
    assert state.last_nonfocus_episode_count == 4
    assert [group.last_episode_count for group in state.focus_groups] == [2, 2]


def test_trajectory_bc_replay_grouped_focus_fills_batch_when_no_nonfocus(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_labels(["learned_a", "learned_b", "fixed_a", "fixed_b"]),
    )

    state = TrajectoryBcReplayState.from_training_config(
        _training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            focus_groups=(
                SimpleNamespace(name="learned", source_labels=("learned_a", "learned_b"), fraction=0.16),
                SimpleNamespace(name="fixed", source_labels=("fixed_a", "fixed_b"), fraction=0.16),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 8
    assert sum(label in {"learned_a", "learned_b"} for label in labels) == 4
    assert sum(label in {"fixed_a", "fixed_b"} for label in labels) == 4
    assert state.last_focus_episode_count == 8
    assert state.last_nonfocus_episode_count == 0
    assert [group.last_episode_count for group in state.focus_groups] == [4, 4]


def test_trajectory_bc_replay_metrics_accumulate_named_focus_group_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_labels(["old", "old", "learned", "fixed", "old", "learned", "fixed", "old"]),
    )
    focus_groups = (
        SimpleNamespace(name="learned_repair", source_labels=("learned",), fraction=0.25),
        SimpleNamespace(name="fixed_repair", source_labels=("fixed",), fraction=0.25),
    )
    state = TrajectoryBcReplayState.from_training_config(
        _training_config(dataset_path=dataset_path, batch_episodes=8, aux_updates=2, focus_groups=focus_groups),
        repo_root=tmp_path,
    )
    assert state is not None

    monkeypatch.setattr(
        trajectory_bc_replay,
        "replay_trajectory_bc_batch",
        lambda dataset, *, episode_indices, initial_hidden_state: {"episode_indices": tuple(episode_indices)},
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_trajectory_bc_replay(
        state=state,
        learner=_ReplayLearner(),
        training_config=_training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            aux_updates=2,
            focus_groups=focus_groups,
        ),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["trajectory_bc_replay_focus_group_count"] == 2.0
    assert latest_metrics["trajectory_bc_replay_focus_batch_episodes"] == 8.0
    assert latest_metrics["trajectory_bc_replay_focus_group_learned_repair_batch_episodes"] == 4.0
    assert latest_metrics["trajectory_bc_replay_focus_group_fixed_repair_batch_episodes"] == 4.0


def test_trajectory_bc_replay_focus_group_counts_distribute_tied_remainders() -> None:
    assert trajectory_bc_replay._focus_group_counts(
        batch_size=16,
        target_focus_count=11,
        fractions=(0.20, 0.15, 0.15, 0.15),
    ) == (3, 3, 3, 2)
    assert trajectory_bc_replay._focus_group_counts(
        batch_size=16,
        target_focus_count=12,
        fractions=(0.16, 0.12, 0.14, 0.14, 0.14),
    ) == (3, 2, 3, 2, 2)


def _training_config(
    *,
    dataset_path: Path,
    batch_episodes: int = 4,
    aux_updates: int = 1,
    focus_source_labels: tuple[str, ...] = (),
    focus_fraction: float = 0.0,
    focus_groups: tuple[SimpleNamespace, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            trajectory_bc_dataset_path=dataset_path.as_posix(),
            trajectory_bc_every_updates=1,
            trajectory_bc_aux_updates=aux_updates,
            trajectory_bc_batch_episodes=batch_episodes,
            trajectory_bc_seed=7,
            trajectory_bc_focus_source_labels=focus_source_labels,
            trajectory_bc_focus_fraction=focus_fraction,
            trajectory_bc_focus_groups=focus_groups,
        )
    )


class _ReplayLearner:
    model = None
    teacher_aux_mode = "always"
    teacher_family_coef = 0.0
    teacher_slot_coef = 0.0
    teacher_hand_coef = 0.0
    teacher_move_source_coef = 0.0
    teacher_attack_type_coef = 0.0
    teacher_action_coef = 0.0
    teacher_same_family_action_coef = 0.0
    teacher_action_margin_coef = 0.0
    teacher_action_margin = 0.5
    teacher_same_family_action_margin_coef = 0.0
    teacher_same_family_action_margin = 0.5
    teacher_exact_action_families: tuple[str, ...] = ()
    teacher_public_heuristic_coef = 0.0
    teacher_public_heuristic_temperature = 32.0
    teacher_public_nonpass_over_pass_coef = 0.0
    teacher_public_nonpass_over_pass_margin = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = ()
    teacher_public_heuristic_profiles: tuple[str, ...] = ()
    teacher_public_heuristic_profile_mode = ""
    teacher_public_heuristic_profiles_end_updates = -1

    def set_teacher_aux_coefs(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, f"teacher_{key}_coef", value)

    def auxiliary_update(self, batch: object) -> dict[str, float]:
        assert batch
        return {"teacher_action_acc": 1.0}


def _dataset_with_labels(labels: list[str]) -> ReplayTrajectoryDataset:
    time_steps = 1
    episode_count = len(labels)
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int64)
    to_play_seat = np.zeros((time_steps, episode_count), dtype=np.int64)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    teacher = np.ones((time_steps, episode_count), dtype=np.int32)
    valid = np.ones((time_steps, episode_count), dtype=np.bool_)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    legal_action_meta = np.zeros((legal_ids.shape[0], 3), dtype=np.uint16)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    metadata = {
        "format": "weiss_rl_replay_trajectory_bc_v1",
        "train_rows": episode_count,
        "selected_bundles": [
            {
                "source_dataset_label": label,
                "pair_index": index,
                "swap_index": 0,
                "outcome": "W",
            }
            for index, label in enumerate(labels)
        ],
    }
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
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
