from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_surface_clusters import (
    PairedOutcomePreferenceSurfaceClusterConfig,
    build_paired_outcome_preference_surface_cluster_report,
)
from weiss_rl.experiments.paired_outcome_preference_surface_clusters_reporting import (
    paired_outcome_preference_surface_cluster_output_line,
    paired_outcome_preference_surface_cluster_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_surface_clusters_runtime import (
    paired_outcome_preference_surface_cluster_config_from_args,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def test_surface_cluster_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_surface_clusters,
        paired_outcome_preference_surface_clusters_cli,
        paired_outcome_preference_surface_clusters_entrypoint,
        paired_outcome_preference_surface_clusters_runtime,
    )

    assert paired_outcome_preference_surface_clusters_entrypoint._build_parser is (
        paired_outcome_preference_surface_clusters_cli.build_paired_outcome_preference_surface_cluster_parser
    )
    assert paired_outcome_preference_surface_clusters_entrypoint.run_paired_outcome_preference_surface_cluster is (
        paired_outcome_preference_surface_clusters_runtime.run_paired_outcome_preference_surface_cluster
    )
    assert paired_outcome_preference_surface_clusters_entrypoint.PairedOutcomePreferenceSurfaceClusterConfig is (
        paired_outcome_preference_surface_clusters.PairedOutcomePreferenceSurfaceClusterConfig
    )
    assert (
        paired_outcome_preference_surface_clusters_entrypoint.build_paired_outcome_preference_surface_cluster_report
        is (paired_outcome_preference_surface_clusters.build_paired_outcome_preference_surface_cluster_report)
    )
    assert (
        paired_outcome_preference_surface_clusters_entrypoint.write_paired_outcome_preference_surface_cluster_report
        is (paired_outcome_preference_surface_clusters.write_paired_outcome_preference_surface_cluster_report)
    )


def test_surface_cluster_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_surface_clusters_cli import (
        build_paired_outcome_preference_surface_cluster_parser,
    )

    args = build_paired_outcome_preference_surface_cluster_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "preference.npz"),
            "--output-json",
            str(tmp_path / "surface_clusters.json"),
        ]
    )

    assert args.dataset == tmp_path / "preference.npz"
    assert args.spec_bundle_json is None
    assert args.stack_config is None
    assert args.opponent_context_policy_id == []
    assert args.max_examples == 25
    assert args.output_json == tmp_path / "surface_clusters.json"


def test_surface_cluster_runtime_maps_args_without_resolving_paths(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "preference.npz",
        spec_bundle_json=tmp_path / "spec_bundle.json",
        stack_config=tmp_path / "stack.yaml",
        opponent_context_policy_id=["B2 HeuristicPublic", "policy_000004"],
        max_examples=7,
    )

    config = paired_outcome_preference_surface_cluster_config_from_args(args)

    assert config == PairedOutcomePreferenceSurfaceClusterConfig(
        dataset_path=tmp_path / "preference.npz",
        spec_bundle_json=tmp_path / "spec_bundle.json",
        stack_config_path=tmp_path / "stack.yaml",
        opponent_context_policy_ids=("B2 HeuristicPublic", "policy_000004"),
        max_examples=7,
    )


def test_surface_cluster_reporting_preserves_compact_console_json(tmp_path: Path) -> None:
    report = {
        "aligned_different_action_count": 8,
        "same_public_surface_different_action_count": 6,
        "surface_conflict_count": 2,
        "opponent_context_resolvable_conflict_count": 1,
        "opponent_context_required_missing_mapping_count": 0,
        "replay_only_required_conflict_count": 1,
        "public_surface_separable": False,
        "unconditioned_replay_safe": False,
    }

    assert paired_outcome_preference_surface_cluster_output_payload(
        output_json=tmp_path / "surface_clusters.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "surface_clusters.json").as_posix(),
        "aligned_different_action_count": 8,
        "same_public_surface_different_action_count": 6,
        "surface_conflict_count": 2,
        "opponent_context_resolvable_conflict_count": 1,
        "opponent_context_required_missing_mapping_count": 0,
        "replay_only_required_conflict_count": 1,
        "public_surface_separable": False,
        "unconditioned_replay_safe": False,
    }
    assert paired_outcome_preference_surface_cluster_output_line(
        output_json=tmp_path / "surface_clusters.json",
        report=report,
    ) == (
        '{"aligned_different_action_count": 8, "opponent_context_required_missing_mapping_count": 0, '
        '"opponent_context_resolvable_conflict_count": 1, '
        f'"output_json": "{(tmp_path / "surface_clusters.json").as_posix()}", '
        '"public_surface_separable": false, "replay_only_required_conflict_count": 1, '
        '"same_public_surface_different_action_count": 6, "surface_conflict_count": 2, '
        '"unconditioned_replay_safe": false}'
    )


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
