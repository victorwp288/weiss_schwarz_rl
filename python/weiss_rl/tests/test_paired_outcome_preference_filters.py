from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_preference_filters import (
    PairedOutcomePreferenceFilterConfig,
    PairedOutcomePreferenceSpanFilterConfig,
    aligned_preference_pair_row_mask,
    filter_paired_outcome_preference_dataset,
    filter_paired_outcome_preference_spans,
    select_preference_episode_indices,
)
from weiss_rl.experiments.paired_outcome_preference_span_audit import (
    PairedOutcomePreferenceSpanAuditConfig,
    build_paired_outcome_preference_span_audit,
)
from weiss_rl.replay.trajectory_bc import BC_DATASET_FORMAT, ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def test_select_preference_episode_indices_can_exclude_specific_preference_pair() -> None:
    bundles = [
        {"preference_pair_id": 0, "source_pair_index": 205},
        {"preference_pair_id": 0, "source_pair_index": 205},
        {"preference_pair_id": 1, "source_pair_index": 229},
        {"preference_pair_id": 1, "source_pair_index": 229},
        {"preference_pair_id": 2, "source_pair_index": 70},
    ]

    selected = select_preference_episode_indices(
        bundles=bundles,
        exclude_preference_pair_ids=[0],
    )

    assert selected == [2, 3, 4]


def test_select_preference_episode_indices_can_exclude_source_pair_index() -> None:
    bundles = [
        {"preference_pair_id": 0, "source_pair_index": 205},
        {"preference_pair_id": 1, "source_pair_index": 229},
        {"preference_pair_id": 2, "pair_index": 70},
    ]

    selected = select_preference_episode_indices(
        bundles=bundles,
        exclude_source_pair_indices=[205],
    )

    assert selected == [1, 2]


def test_select_preference_episode_indices_include_filters_are_intersections() -> None:
    bundles = [
        {"preference_pair_id": 0, "source_pair_index": 205},
        {"preference_pair_id": 1, "source_pair_index": 229},
        {"preference_pair_id": 2, "source_pair_index": 70},
    ]

    selected = select_preference_episode_indices(
        bundles=bundles,
        include_preference_pair_ids=[1, 2],
        include_source_pair_indices=[70],
    )

    assert selected == [2]


def test_select_preference_episode_indices_can_filter_source_opponents() -> None:
    bundles = [
        {"preference_pair_id": 0, "source_pair_index": 205, "source_opponent_policy_id": "B2 HeuristicPublic"},
        {"preference_pair_id": 0, "source_pair_index": 205, "source_opponent_policy_id": "B2 HeuristicPublic"},
        {"preference_pair_id": 1, "source_pair_index": 205, "source_opponent_policy_id": "policy_000003"},
        {"preference_pair_id": 1, "source_pair_index": 205, "source_opponent_policy_id": "policy_000003"},
        {"preference_pair_id": 2, "source_pair_index": 229, "source_opponent_policy_id": "policy_000004"},
    ]

    selected = select_preference_episode_indices(
        bundles=bundles,
        include_source_pair_indices=[205],
        include_source_opponent_policy_ids=["policy_000003"],
    )

    assert selected == [2, 3]


def test_preference_episode_filter_summary_reports_selected_opponents(tmp_path: Path) -> None:
    dataset_path = tmp_path / "preference.npz"
    save_replay_trajectory_bc_dataset(dataset_path, _tiny_preference_dataset_with_opponents())

    _filtered, summary = filter_paired_outcome_preference_dataset(
        PairedOutcomePreferenceFilterConfig(
            dataset_path=dataset_path,
            output_dataset_path=tmp_path / "filtered.npz",
            output_summary_json=tmp_path / "filtered.json",
            include_source_opponent_policy_ids=("policy_000003",),
        )
    )

    assert summary["output_episode_count"] == 2
    assert summary["output_train_rows"] == 8
    assert summary["selected_preference_pair_ids"] == [1]
    assert summary["selected_source_opponent_policy_ids"] == ["policy_000003"]


def test_aligned_preference_pair_row_mask_keeps_only_action_difference_steps() -> None:
    dataset = _tiny_preference_dataset()

    row_mask, summaries = aligned_preference_pair_row_mask(dataset)

    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[1, 0] = True
    expected[1, 1] = True
    expected[3, 0] = True
    expected[3, 1] = True
    assert row_mask.tolist() == expected.tolist()
    assert summaries[0]["preference_pair_id"] == 0
    assert summaries[0]["aligned_train_steps"] == 4
    assert summaries[0]["aligned_action_difference_steps"] == 2
    assert summaries[0]["output_rows"] == 4
    assert summaries[1]["preference_pair_id"] == 1
    assert summaries[1]["output_rows"] == 0


def test_aligned_preference_pair_row_mask_can_keep_same_action_steps() -> None:
    dataset = _tiny_preference_dataset()

    row_mask, summaries = aligned_preference_pair_row_mask(dataset, require_action_difference=False)

    assert int(np.count_nonzero(row_mask[:, 0])) == 4
    assert int(np.count_nonzero(row_mask[:, 1])) == 4
    assert int(np.count_nonzero(row_mask[:, 2])) == 4
    assert int(np.count_nonzero(row_mask[:, 3])) == 4
    assert summaries[0]["output_rows"] == 8
    assert summaries[1]["output_rows"] == 8


def test_aligned_preference_pair_row_mask_can_exclude_same_family_action_differences() -> None:
    dataset = _tiny_preference_dataset()
    dataset.teacher_family[3, 1] = 1

    row_mask, summaries = aligned_preference_pair_row_mask(
        dataset,
        exclude_same_family_action_differences=True,
    )

    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[3, 0] = True
    expected[3, 1] = True
    assert row_mask.tolist() == expected.tolist()
    assert summaries[0]["aligned_action_difference_same_family_steps"] == 1
    assert summaries[0]["aligned_action_difference_cross_family_steps"] == 1
    assert summaries[0]["excluded_same_family_action_difference_steps"] == 1
    assert summaries[0]["output_rows"] == 2


def test_aligned_preference_pair_row_mask_can_require_same_current_state() -> None:
    dataset = _tiny_preference_dataset()
    dataset.obs[3, 1, 0] = 1.0

    row_mask, summaries = aligned_preference_pair_row_mask(dataset, require_same_current_state=True)

    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[1, 0] = True
    expected[1, 1] = True
    assert row_mask.tolist() == expected.tolist()
    assert summaries[0]["aligned_action_difference_steps"] == 2
    assert summaries[0]["aligned_action_difference_same_current_state_steps"] == 1
    assert summaries[0]["output_rows"] == 2


def test_aligned_preference_pair_row_mask_can_require_same_history() -> None:
    dataset = _tiny_preference_dataset()
    dataset.obs[2, 1, 0] = 1.0

    row_mask, summaries = aligned_preference_pair_row_mask(dataset, require_same_history=True)

    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[1, 0] = True
    expected[1, 1] = True
    assert row_mask.tolist() == expected.tolist()
    assert summaries[0]["aligned_action_difference_same_history_steps"] == 1
    assert summaries[0]["output_rows"] == 2


def test_aligned_preference_pair_row_mask_can_exclude_reverse_label_current_state_conflicts() -> None:
    dataset = _tiny_preference_dataset()
    dataset.actions[1, 2] = 9
    dataset.actions[1, 3] = 2
    dataset.obs[3, :, 0] = 3.0

    row_mask, summaries = aligned_preference_pair_row_mask(dataset, exclude_current_state_conflicts=True)

    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[3, 0] = True
    expected[3, 1] = True
    assert row_mask.tolist() == expected.tolist()
    assert summaries[0]["excluded_current_state_conflict_steps"] == 1
    assert summaries[0]["output_rows"] == 2
    assert summaries[1]["excluded_current_state_conflict_steps"] == 1
    assert summaries[1]["output_rows"] == 0


def test_span_filter_keeps_repeated_compact_span_steps(tmp_path: Path) -> None:
    dataset_path = _write_span_filter_dataset(tmp_path / "preference.npz", repeated_edges=True)
    audit = build_paired_outcome_preference_span_audit(
        PairedOutcomePreferenceSpanAuditConfig(dataset_path=dataset_path, max_compact_span_width=3)
    )
    audit_path = tmp_path / "span_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    _filtered, summary = filter_paired_outcome_preference_spans(
        PairedOutcomePreferenceSpanFilterConfig(
            dataset_path=dataset_path,
            span_audit_json=audit_path,
            output_dataset_path=tmp_path / "span_filtered.npz",
            output_summary_json=tmp_path / "span_filtered_summary.json",
            include_span_modes=("repeated_action_label",),
        )
    )

    assert summary["output_train_rows"] == 8
    assert summary["selected_span_count"] == 2
    assert summary["selected_preference_pair_ids"] == [0, 1]
    assert summary["pair_row_counts"] == {"0": 4, "1": 4}
    assert {tuple(row["edge_step_indices"]) for row in summary["kept_spans"]} == {(1, 2)}


def test_span_filter_rejects_failed_span_audit(tmp_path: Path) -> None:
    dataset_path = _write_span_filter_dataset(tmp_path / "preference.npz", repeated_edges=False)
    audit = build_paired_outcome_preference_span_audit(
        PairedOutcomePreferenceSpanAuditConfig(dataset_path=dataset_path, max_compact_span_width=3)
    )
    audit_path = tmp_path / "span_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="span audit did not pass"):
        filter_paired_outcome_preference_spans(
            PairedOutcomePreferenceSpanFilterConfig(
                dataset_path=dataset_path,
                span_audit_json=audit_path,
                output_dataset_path=tmp_path / "span_filtered.npz",
            )
        )


def _tiny_preference_dataset() -> ReplayTrajectoryDataset:
    time_steps = 4
    episode_count = 4
    row_count = time_steps * episode_count
    legal_offsets = np.arange(row_count + 1, dtype=np.uint32)
    actions = np.asarray(
        [
            [1, 1, 5, 5],
            [2, 9, 6, 6],
            [3, 3, 7, 7],
            [4, 8, 8, 8],
        ],
        dtype=np.int64,
    )
    return ReplayTrajectoryDataset(
        obs=np.zeros((time_steps, episode_count, 3), dtype=np.float32),
        actor=np.zeros((time_steps, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=np.zeros((row_count,), dtype=np.uint32),
        legal_offsets=legal_offsets,
        legal_action_meta=np.zeros((row_count, 4), dtype=np.uint16),
        teacher_family=np.zeros((time_steps, episode_count), dtype=np.int32),
        teacher_slot=np.zeros((time_steps, episode_count), dtype=np.int32),
        teacher_move_source=np.zeros((time_steps, episode_count), dtype=np.int32),
        teacher_attack_type=np.zeros((time_steps, episode_count), dtype=np.int32),
        teacher_action=actions.astype(np.int32),
        teacher_valid=np.ones((time_steps, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((time_steps, episode_count), dtype=np.bool_),
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "train_rows": int(row_count),
            "selected_bundles": [
                {"preference_pair_id": 0, "preference_role": 1, "source_pair_index": 10},
                {"preference_pair_id": 0, "preference_role": 0, "source_pair_index": 10},
                {"preference_pair_id": 1, "preference_role": 1, "source_pair_index": 11},
                {"preference_pair_id": 1, "preference_role": 0, "source_pair_index": 11},
            ],
        },
    )


def _tiny_preference_dataset_with_opponents() -> ReplayTrajectoryDataset:
    dataset = _tiny_preference_dataset()
    dataset.metadata["episode_count"] = 4
    dataset.metadata["selected_bundles"] = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "source_pair_index": 205,
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "source_pair_index": 205,
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
        {
            "preference_pair_id": 1,
            "preference_role": 1,
            "source_pair_index": 205,
            "source_opponent_policy_id": "policy_000003",
        },
        {
            "preference_pair_id": 1,
            "preference_role": 0,
            "source_pair_index": 205,
            "source_opponent_policy_id": "policy_000003",
        },
    ]
    return dataset


def _write_span_filter_dataset(path: Path, *, repeated_edges: bool) -> Path:
    time_steps = 6
    episode_count = 4
    row_count = time_steps * episode_count
    actions = np.zeros((time_steps, episode_count), dtype=np.int64)
    actions[:, 0] = np.asarray([10, 11, 12, 12, 13, 0])
    actions[:, 1] = np.asarray([10, 21, 22, 12, 23, 0])
    actions[:, 2] = np.asarray([10, 11, 12, 12, 13, 0])
    actions[:, 3] = np.asarray([10, 21, 22, 12, 13, 0] if repeated_edges else [10, 31, 32, 12, 13, 0])
    legal_row = np.asarray([0, 10, 11, 12, 13, 21, 22, 23, 31, 32], dtype=np.uint32)
    legal_ids = np.tile(legal_row, row_count)
    legal_offsets = np.arange(0, legal_ids.shape[0] + 1, legal_row.shape[0], dtype=np.uint32)
    save_replay_trajectory_bc_dataset(
        path,
        ReplayTrajectoryDataset(
            obs=np.zeros((time_steps, episode_count, 3), dtype=np.float32),
            actor=np.zeros((time_steps, episode_count), dtype=np.int8),
            to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int8),
            actions=actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=np.zeros((legal_ids.shape[0], 4), dtype=np.uint16),
            teacher_family=np.zeros((time_steps, episode_count), dtype=np.int32),
            teacher_slot=np.zeros((time_steps, episode_count), dtype=np.int32),
            teacher_move_source=np.zeros((time_steps, episode_count), dtype=np.int32),
            teacher_attack_type=np.zeros((time_steps, episode_count), dtype=np.int32),
            teacher_action=actions.astype(np.int32),
            teacher_valid=np.ones((time_steps, episode_count), dtype=np.bool_),
            policy_train_mask=np.ones((time_steps, episode_count), dtype=np.bool_),
            reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
            metadata={
                "format": BC_DATASET_FORMAT,
                "episode_count": episode_count,
                "time_steps": time_steps,
                "row_count": row_count,
                "train_rows": row_count,
                "selected_bundles": [
                    _span_filter_bundle(pair_id=0, role=1, source_pair_index=16),
                    _span_filter_bundle(pair_id=0, role=0, source_pair_index=16),
                    _span_filter_bundle(pair_id=1, role=1, source_pair_index=205),
                    _span_filter_bundle(pair_id=1, role=0, source_pair_index=205),
                ],
            },
        ),
    )
    return path


def _span_filter_bundle(*, pair_id: int, role: int, source_pair_index: int) -> dict[str, object]:
    return {
        "preference_pair_id": pair_id,
        "preference_role": role,
        "preference_role_label": "preferred" if role == 1 else "rejected",
        "source_opponent_policy_id": "B2 HeuristicPublic",
        "source_pair_index": source_pair_index,
        "episode_seed": 100 + pair_id,
        "swap_index": 0,
    }
