from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.paired_swing_conflict_filter import (
    filter_paired_swing_conflict_rows,
    normalize_paired_swing_action_source,
    paired_swing_distinct_train_row_count,
)
from weiss_rl.training.replay_data.paired_swing_replay import PairedSwingReplayState

from .paired_swing_replay_test_support import build_training_config, dataset_with_action_pairs


def test_paired_swing_distinct_train_row_count_requires_legal_positive_and_negative_actions() -> None:
    dataset = dataset_with_action_pairs(
        labels=["good", "bad"],
        actions=[1, 1],
        teacher_actions=[2, 3],
        legal_rows=[[1, 2], [1, 2]],
    )

    assert paired_swing_distinct_train_row_count(dataset) == 1


def test_normalize_paired_swing_action_source_preserves_old_validation() -> None:
    assert normalize_paired_swing_action_source(" Teacher_Action ", field_name="source") == "teacher_action"

    with pytest.raises(ValueError, match="source must be one of: actions, teacher_action"):
        normalize_paired_swing_action_source("policy", field_name="source")


def test_paired_swing_conflict_filter_masks_current_state_reverse_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing_conflict.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_action_pairs(
            labels=["fixed", "learned", "safe"],
            actions=[104, 124, 1],
            teacher_actions=[124, 104, 2],
            legal_rows=[[104, 124], [104, 124], [1, 2]],
        ),
    )

    state = PairedSwingReplayState.from_training_config(
        build_training_config(dataset_path=dataset_path, conflict_filter="current_state"),
        repo_root=tmp_path,
    )

    assert state is not None
    assert state.distinct_train_rows == 1
    assert state.conflict_filter_summary is not None
    assert state.conflict_filter_summary["dropped_train_rows"] == 2
    assert state.conflict_filter_summary["kept_train_rows"] == 1
    assert int(np.count_nonzero(state.sampler.dataset.policy_train_mask)) == 1


def test_paired_swing_conflict_filter_rejects_unknown_mode() -> None:
    dataset = dataset_with_action_pairs(labels=["safe"], actions=[1], teacher_actions=[2])

    with pytest.raises(ValueError, match="paired-swing conflict filter mode"):
        filter_paired_swing_conflict_rows(dataset, mode="timeline")
