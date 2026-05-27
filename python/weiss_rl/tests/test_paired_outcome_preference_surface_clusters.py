from __future__ import annotations

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_surface_clusters import (
    PairedOutcomePreferenceSurfaceClusterConfig,
    build_paired_outcome_preference_surface_cluster_report,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def test_surface_cluster_report_classifies_cross_opponent_reverse_as_context_resolvable(tmp_path):
    dataset_path = tmp_path / "cross_opponent.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_reversed_edges(["B2 HeuristicPublic", "B2 HeuristicPublic", "policy_000004", "policy_000004"]),
    )

    report = build_paired_outcome_preference_surface_cluster_report(
        PairedOutcomePreferenceSurfaceClusterConfig(
            dataset_path=dataset_path,
            opponent_context_policy_ids=("B2 HeuristicPublic", "policy_000004"),
            max_examples=10,
        )
    )

    assert report["surface_conflict_count"] == 1
    assert report["opponent_context_resolvable_conflict_count"] == 1
    assert report["replay_only_required_conflict_count"] == 0
    assert report["public_surface_separable"] is True
    conflict = report["surface_conflicts"][0]
    assert conflict["classification"] == "opponent_context_resolvable"
    assert {row["opponent_context_index"] for row in conflict["opponent_rows"]} == {1, 2}


def test_surface_cluster_report_classifies_same_opponent_reverse_as_replay_only(tmp_path):
    dataset_path = tmp_path / "same_opponent.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        _dataset_with_reversed_edges(
            ["B2 HeuristicPublic", "B2 HeuristicPublic", "B2 HeuristicPublic", "B2 HeuristicPublic"]
        ),
    )

    report = build_paired_outcome_preference_surface_cluster_report(
        PairedOutcomePreferenceSurfaceClusterConfig(
            dataset_path=dataset_path,
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            max_examples=10,
        )
    )

    assert report["surface_conflict_count"] == 1
    assert report["replay_only_required_conflict_count"] == 1
    assert report["public_surface_separable"] is False
    assert report["surface_conflicts"][0]["classification"] == "replay_only_required"


def test_surface_cluster_report_treats_same_opponent_different_states_as_separable(tmp_path):
    dataset_path = tmp_path / "different_states.npz"
    dataset = _dataset_with_reversed_edges(
        ["B2 HeuristicPublic", "B2 HeuristicPublic", "B2 HeuristicPublic", "B2 HeuristicPublic"]
    )
    dataset.obs[0, 2:] = 2.0
    save_replay_trajectory_bc_dataset(dataset_path, dataset)

    report = build_paired_outcome_preference_surface_cluster_report(
        PairedOutcomePreferenceSurfaceClusterConfig(
            dataset_path=dataset_path,
            opponent_context_policy_ids=("B2 HeuristicPublic",),
        )
    )

    assert report["surface_conflict_count"] == 0
    assert report["same_public_surface_different_action_count"] == 2
    assert report["public_surface_separable"] is True
    assert report["unconditioned_replay_safe"] is True


def _dataset_with_reversed_edges(opponent_ids: list[str]) -> ReplayTrajectoryDataset:
    episode_count = 4
    obs = np.ones((1, episode_count, 3), dtype=np.float32)
    actions = np.asarray([[124, 104, 104, 124]], dtype=np.int64)
    legal_ids = np.tile(np.asarray([104, 124], dtype=np.uint32), episode_count)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    legal_meta = np.zeros((legal_ids.shape[0], 4), dtype=np.uint16)
    bundles = [
        {
            "source_opponent_policy_id": opponent_ids[0],
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 0,
            "preference_role": 1,
            "preference_role_label": "preserve",
        },
        {
            "source_opponent_policy_id": opponent_ids[1],
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 0,
            "preference_role": 0,
            "preference_role_label": "repair_loss",
        },
        {
            "source_opponent_policy_id": opponent_ids[2],
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 1,
            "preference_role": 1,
            "preference_role_label": "repair",
        },
        {
            "source_opponent_policy_id": opponent_ids[3],
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 1,
            "preference_role": 0,
            "preference_role_label": "preserve_loss",
        },
    ]
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((1, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        teacher_family=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((1, episode_count), dtype=np.bool_),
        reset_before_step=np.ones((1, episode_count), dtype=np.bool_),
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
