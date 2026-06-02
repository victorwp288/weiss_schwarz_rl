from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset
from weiss_rl.training.auxiliary_replay_metrics import AuxiliaryReplayMetricAccumulator, emit_finite_aux_metrics
from weiss_rl.training.replay_data.trajectory_bc_sampling import (
    TrajectoryBcReplayFocusGroupState,
    TrajectoryBcReplayState,
)


def test_auxiliary_replay_metrics_emit_focus_groups_context_and_finite_aux_metrics() -> None:
    group = TrajectoryBcReplayFocusGroupState(
        name="Hard Negative/Repair",
        source_labels=("repair",),
        fraction=0.25,
        indices=np.asarray([1, 3], dtype=np.int64),
        order=np.asarray([1, 3], dtype=np.int64),
        last_episode_count=2,
    )
    sampler = TrajectoryBcReplayState(
        dataset=_dataset(episode_count=4),
        rng=np.random.default_rng(7),
        batch_episodes=3,
        aux_updates=5,
        every_updates=2,
        order=np.asarray([0, 1, 2, 3], dtype=np.int64),
        focus_fraction=0.25,
        focus_groups=(group,),
    )
    sampler.last_focus_episode_count = 2
    sampler.last_nonfocus_episode_count = 1
    metrics = AuxiliaryReplayMetricAccumulator(sampler)

    metrics.record_sampled_episodes([0, 1, 2])
    metrics.record_context_indices(np.asarray([0, 4, 5], dtype=np.int64))
    latest: dict[str, float] = {}
    metrics.emit_common_metrics(latest, prefix="example_replay", include_focus=True, include_context=True)
    emit_finite_aux_metrics(
        latest,
        prefix="example_replay",
        aux_metrics={"loss": 0.125, "nan_loss": float("nan"), "label": "ignored"},
    )

    assert latest["example_replay_aux_updates"] == 5.0
    assert latest["example_replay_batch_episodes"] == 3.0
    assert latest["example_replay_dataset_train_rows"] == 4.0
    assert latest["example_replay_focus_batch_episodes"] == 2.0
    assert latest["example_replay_nonfocus_batch_episodes"] == 1.0
    assert latest["example_replay_focus_group_hard_negative_repair_batch_episodes"] == 2.0
    assert latest["example_replay_opponent_context_episodes"] == 2.0
    assert latest["example_replay_loss"] == pytest.approx(0.125)
    assert "example_replay_nan_loss" not in latest
    assert "example_replay_label" not in latest


def test_auxiliary_replay_metrics_can_emit_batch_metrics_without_focus_fields() -> None:
    sampler = TrajectoryBcReplayState(
        dataset=_dataset(episode_count=2),
        rng=np.random.default_rng(11),
        batch_episodes=2,
        aux_updates=1,
        every_updates=1,
        order=np.asarray([0, 1], dtype=np.int64),
        focus_fraction=0.5,
    )
    sampler.last_focus_episode_count = 1
    sampler.last_nonfocus_episode_count = 1
    metrics = AuxiliaryReplayMetricAccumulator(sampler)

    metrics.record_sampled_episodes([0, 1])
    latest: dict[str, float] = {}
    metrics.emit_common_metrics(latest, prefix="preference_replay", include_focus=False)

    assert latest == {
        "preference_replay_aux_updates": 1.0,
        "preference_replay_batch_episodes": 2.0,
        "preference_replay_dataset_train_rows": 2.0,
    }


def _dataset(*, episode_count: int) -> ReplayTrajectoryDataset:
    obs = np.zeros((1, episode_count, 4), dtype=np.float32)
    actions = np.ones((1, episode_count), dtype=np.int64)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    teacher = np.ones((1, episode_count), dtype=np.int32)
    valid = np.ones((1, episode_count), dtype=np.bool_)
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((1, episode_count), dtype=np.int64),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int64),
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=np.zeros((legal_ids.shape[0], 3), dtype=np.uint16),
        teacher_family=teacher,
        teacher_slot=teacher,
        teacher_move_source=teacher,
        teacher_attack_type=teacher,
        teacher_action=teacher,
        teacher_valid=valid,
        policy_train_mask=valid,
        reset_before_step=np.zeros((1, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "train_rows": episode_count,
            "selected_bundles": [{"source_dataset_label": ""} for _ in range(episode_count)],
        },
    )
