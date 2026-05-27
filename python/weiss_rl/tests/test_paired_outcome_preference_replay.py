from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset
from weiss_rl.training.paired_outcome_preference_replay import (
    PairedOutcomePreferenceReplayState,
    paired_outcome_preference_complete_pair_count,
)


def test_paired_outcome_preference_complete_pair_count_requires_both_roles() -> None:
    dataset = _preference_dataset(
        [
            {"preference_pair_id": 5, "preference_role": 1},
            {"preference_pair_id": 5, "preference_role": 0},
            {"preference_pair_id": 6, "preference_role": 1},
        ]
    )

    assert paired_outcome_preference_complete_pair_count(dataset) == 1


def test_paired_outcome_preference_replay_state_loads_explicit_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "preference.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _preference_dataset(
            [
                {"preference_pair_id": 5, "preference_role": 1},
                {"preference_pair_id": 5, "preference_role": 0},
            ]
        ),
    )

    state = PairedOutcomePreferenceReplayState.from_training_config(
        _training_config(dataset_path),
        repo_root=tmp_path,
    )

    assert state is not None
    assert state.complete_pair_count == 1
    assert state.beta == pytest.approx(0.2)
    assert state.coef == pytest.approx(0.07)
    assert state.aggregation == "sum"
    assert state.group_balance is True


def test_paired_outcome_preference_replay_state_rejects_incomplete_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "preference.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _preference_dataset([{"preference_pair_id": 5, "preference_role": 1}]),
    )

    with pytest.raises(ValueError, match="no complete preferred/rejected pairs"):
        PairedOutcomePreferenceReplayState.from_training_config(
            _training_config(dataset_path),
            repo_root=tmp_path,
        )


def _training_config(dataset_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            paired_outcome_preference_dataset_path=dataset_path.as_posix(),
            paired_outcome_preference_every_updates=1,
            paired_outcome_preference_aux_updates=2,
            paired_outcome_preference_batch_episodes=2,
            paired_outcome_preference_seed=123,
            paired_outcome_preference_coef=0.07,
            paired_outcome_preference_beta=0.2,
            paired_outcome_preference_aggregation="sum",
            paired_outcome_preference_group_balance=True,
        )
    )


def _preference_dataset(selected_bundles: list[dict]) -> ReplayTrajectoryDataset:
    episode_count = len(selected_bundles)
    obs = np.zeros((1, episode_count, 4), dtype=np.float32)
    actions = np.ones((1, episode_count), dtype=np.int64)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    legal_meta = np.stack([legal_ids, legal_ids + 10, legal_ids + 20], axis=1).astype(np.uint16)
    offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    bundles = []
    for index, bundle in enumerate(selected_bundles):
        bundles.append(
            {
                "source_dataset_label": f"episode_{index}",
                "source_opponent_policy_id": "policy_000001",
                **bundle,
            }
        )
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((1, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=offsets,
        legal_action_meta=legal_meta,
        teacher_family=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((1, episode_count), dtype=np.bool_),
        reset_before_step=np.zeros((1, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": episode_count,
            "requested_bundle_count": episode_count,
            "include_outcomes": ["ALL"],
            "pass_action_id": 0,
            "spec_hash256": "ab" * 32,
            "train_rows": episode_count,
            "selected_bundles": bundles,
        },
    )
