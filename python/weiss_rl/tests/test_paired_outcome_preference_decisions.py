from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_decisions import (
    PairedOutcomePreferenceDecisionConfig,
    build_paired_outcome_preference_decision_report,
)
from weiss_rl.experiments.paired_outcome_preference_decisions_reporting import (
    paired_outcome_preference_decisions_output_line,
    paired_outcome_preference_decisions_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_decisions_runtime import (
    paired_outcome_preference_decision_config_from_args,
)
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    save_replay_trajectory_bc_dataset,
)


def test_paired_outcome_preference_decisions_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_decisions,
        paired_outcome_preference_decisions_cli,
        paired_outcome_preference_decisions_entrypoint,
        paired_outcome_preference_decisions_runtime,
    )

    assert paired_outcome_preference_decisions_entrypoint._build_parser is (
        paired_outcome_preference_decisions_cli.build_paired_outcome_preference_decisions_parser
    )
    assert paired_outcome_preference_decisions_entrypoint.run_paired_outcome_preference_decisions is (
        paired_outcome_preference_decisions_runtime.run_paired_outcome_preference_decisions
    )
    assert paired_outcome_preference_decisions_entrypoint.PairedOutcomePreferenceDecisionConfig is (
        paired_outcome_preference_decisions.PairedOutcomePreferenceDecisionConfig
    )
    assert paired_outcome_preference_decisions_entrypoint.build_paired_outcome_preference_decision_report is (
        paired_outcome_preference_decisions.build_paired_outcome_preference_decision_report
    )
    assert paired_outcome_preference_decisions_entrypoint.write_paired_outcome_preference_decision_report is (
        paired_outcome_preference_decisions.write_paired_outcome_preference_decision_report
    )


def test_paired_outcome_preference_decisions_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_decisions_cli import (
        build_paired_outcome_preference_decisions_parser,
    )

    args = build_paired_outcome_preference_decisions_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "preference.npz"),
            "--output-json",
            str(tmp_path / "decisions.json"),
        ]
    )

    assert args.dataset == tmp_path / "preference.npz"
    assert args.spec_bundle_json is None
    assert args.max_examples == 25
    assert args.top_action_edges == 25
    assert args.output_json == tmp_path / "decisions.json"


def test_paired_outcome_preference_decisions_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "preference.npz",
        spec_bundle_json=tmp_path / "spec_bundle.json",
        max_examples=7,
        top_action_edges=9,
    )

    config = paired_outcome_preference_decision_config_from_args(args)

    assert config.dataset_path == tmp_path / "preference.npz"
    assert config.spec_bundle_json == tmp_path / "spec_bundle.json"
    assert config.max_examples == 7
    assert config.top_action_edges == 9


def test_paired_outcome_preference_decisions_reporting_preserves_compact_console_json(tmp_path: Path) -> None:
    report = {
        "preference_pair_count": 4,
        "complete_pair_count": 3,
        "aligned_different_action_count": 2,
        "same_current_state_edge_count": 8,
        "same_current_state_different_action_edge_count": 1,
        "current_state_conflict_count": 5,
        "history_conflict_count": 6,
    }

    assert paired_outcome_preference_decisions_output_payload(
        output_json=tmp_path / "decisions.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "decisions.json").as_posix(),
        "preference_pair_count": 4,
        "complete_pair_count": 3,
        "aligned_different_action_count": 2,
        "same_current_state_edge_count": 8,
        "same_current_state_different_action_edge_count": 1,
        "current_state_conflict_count": 5,
        "history_conflict_count": 6,
    }
    assert paired_outcome_preference_decisions_output_line(
        output_json=tmp_path / "decisions.json",
        report=report,
    ) == (
        '{"aligned_different_action_count": 2, "complete_pair_count": 3, '
        '"current_state_conflict_count": 5, "history_conflict_count": 6, '
        f'"output_json": "{(tmp_path / "decisions.json").as_posix()}", '
        '"preference_pair_count": 4, "same_current_state_different_action_edge_count": 1, '
        '"same_current_state_edge_count": 8}'
    )


def test_paired_outcome_preference_decision_report_detects_reversed_same_state_conflict(tmp_path: Path) -> None:
    dataset_path = tmp_path / "preference.npz"
    save_replay_trajectory_bc_dataset(dataset_path, _dataset_with_reversed_edges())

    report = build_paired_outcome_preference_decision_report(
        PairedOutcomePreferenceDecisionConfig(dataset_path=dataset_path, max_examples=10)
    )

    assert report["preference_pair_count"] == 2
    assert report["complete_pair_count"] == 2
    assert report["aligned_step_count"] == 2
    assert report["aligned_different_action_count"] == 2
    assert report["same_current_state_edge_count"] == 2
    assert report["same_current_state_different_action_edge_count"] == 2
    assert report["current_state_conflict_count"] == 1
    conflict = report["current_state_conflicts"][0]
    assert conflict["preferred_actions"] == [104, 124]
    assert conflict["rejected_actions"] == [104, 124]
    assert conflict["exact_reverse_pair_count"] == 1
    assert conflict["source_pair_indices"] == [205]
    assert {example["source_opponent_policy_id"] for example in conflict["examples"]} == {
        "B2 HeuristicPublic",
        "policy_000004",
    }
    edges = {
        (edge["preferred_action"], edge["rejected_action"]): edge["count"] for edge in report["action_edge_counts"]
    }
    assert edges[(124, 104)] == 1
    assert edges[(104, 124)] == 1


def test_paired_outcome_preference_decision_report_marks_incomplete_pairs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "incomplete.npz"
    dataset = _dataset_with_reversed_edges()
    dataset.metadata["selected_bundles"][1]["preference_pair_id"] = 99
    save_replay_trajectory_bc_dataset(dataset_path, dataset)

    report = build_paired_outcome_preference_decision_report(
        PairedOutcomePreferenceDecisionConfig(dataset_path=dataset_path)
    )

    assert report["complete_pair_count"] == 1
    assert report["incomplete_pair_count"] == 2


def test_paired_outcome_preference_decision_report_ignores_same_action_conflicts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "same_action.npz"
    dataset = _dataset_with_reversed_edges()
    dataset.actions[:] = 104
    save_replay_trajectory_bc_dataset(dataset_path, dataset)

    report = build_paired_outcome_preference_decision_report(
        PairedOutcomePreferenceDecisionConfig(dataset_path=dataset_path)
    )

    assert report["same_current_state_edge_count"] == 2
    assert report["same_current_state_different_action_edge_count"] == 0
    assert report["current_state_conflict_count"] == 0


def _dataset_with_reversed_edges() -> ReplayTrajectoryDataset:
    episode_count = 4
    obs = np.ones((1, episode_count, 3), dtype=np.float32)
    actions = np.asarray([[124, 104, 104, 124]], dtype=np.int64)
    legal_ids = np.tile(np.asarray([104, 124], dtype=np.uint32), episode_count)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    legal_meta = np.zeros((legal_ids.shape[0], 4), dtype=np.uint16)
    bundles = [
        {
            "source_opponent_policy_id": "B2 HeuristicPublic",
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 0,
            "preference_role": 1,
            "preference_role_label": "fixed_preserve",
        },
        {
            "source_opponent_policy_id": "B2 HeuristicPublic",
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 0,
            "preference_role": 0,
            "preference_role_label": "learned_loss",
        },
        {
            "source_opponent_policy_id": "policy_000004",
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 1,
            "preference_role": 1,
            "preference_role_label": "learned_repair",
        },
        {
            "source_opponent_policy_id": "policy_000004",
            "source_pair_index": 205,
            "episode_seed": 14210367516666939508,
            "preference_pair_id": 1,
            "preference_role": 0,
            "preference_role_label": "fixed_loss",
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
