from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from weiss_rl.experiments.paired_swing_context_margins import (
    PairedSwingContextMarginConfig,
    paired_swing_margin_rows_from_packed_scores,
)
from weiss_rl.experiments.paired_swing_context_margins_reporting import (
    paired_swing_context_margins_output_payload,
)
from weiss_rl.experiments.paired_swing_context_margins_runtime import (
    paired_swing_context_margins_config_from_args,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def test_paired_swing_context_margins_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_swing_context_margins,
        paired_swing_context_margins_cli,
        paired_swing_context_margins_entrypoint,
        paired_swing_context_margins_runtime,
    )

    assert paired_swing_context_margins_entrypoint._build_parser is (
        paired_swing_context_margins_cli.build_paired_swing_context_margins_parser
    )
    assert paired_swing_context_margins_entrypoint.run_paired_swing_context_margins is (
        paired_swing_context_margins_runtime.run_paired_swing_context_margins
    )
    assert paired_swing_context_margins_entrypoint.PairedSwingContextMarginConfig is (
        paired_swing_context_margins.PairedSwingContextMarginConfig
    )
    assert paired_swing_context_margins_entrypoint.build_paired_swing_context_margin_report is (
        paired_swing_context_margins.build_paired_swing_context_margin_report
    )
    assert paired_swing_context_margins_entrypoint.paired_swing_margin_rows_from_packed_scores is (
        paired_swing_context_margins.paired_swing_margin_rows_from_packed_scores
    )
    assert paired_swing_context_margins_entrypoint.write_paired_swing_context_margin_report is (
        paired_swing_context_margins.write_paired_swing_context_margin_report
    )


def test_paired_swing_context_margins_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_swing_context_margins_cli import (
        build_paired_swing_context_margins_parser,
    )

    args = build_paired_swing_context_margins_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset.npz"),
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-json",
            str(tmp_path / "margins.json"),
        ]
    )

    assert args.dataset == tmp_path / "dataset.npz"
    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.run_dir == tmp_path / "run"
    assert args.checkpoint == tmp_path / "checkpoint.pt"
    assert args.positive_action_source == "actions"
    assert args.negative_action_source == "teacher_action"
    assert args.report_action_id == [104, 124]
    assert args.output_json == tmp_path / "margins.json"


def test_paired_swing_context_margins_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "dataset.npz",
        stack_config=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "checkpoint.pt",
        positive_action_source="teacher_action",
        negative_action_source="actions",
        report_action_id=[104, 124, 257],
    )

    config = paired_swing_context_margins_config_from_args(args)

    assert config == PairedSwingContextMarginConfig(
        dataset_path=tmp_path / "dataset.npz",
        stack_config_path=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        positive_action_source="teacher_action",
        negative_action_source="actions",
        report_action_ids=(104, 124, 257),
    )


def test_paired_swing_context_margins_reporting_preserves_compact_console_payload(tmp_path: Path) -> None:
    report = {
        "row_count": 3,
        "context_episode_count": 2,
        "context_coverage": {"missing_context_episode_count": 1},
        "positive_margin_min": -0.25,
        "positive_margin_mean": 0.125,
    }

    assert paired_swing_context_margins_output_payload(output_json=tmp_path / "margins.json", report=report) == {
        "output_json": (tmp_path / "margins.json").as_posix(),
        "row_count": 3,
        "context_episode_count": 2,
        "missing_context_episode_count": 1,
        "positive_margin_min": -0.25,
        "positive_margin_mean": 0.125,
    }


def test_paired_swing_margin_rows_reports_opponent_context_and_action_logps() -> None:
    dataset = ReplayTrajectoryDataset(
        obs=np.zeros((1, 2, 4), dtype=np.float32),
        actor=np.zeros((1, 2), dtype=np.int8),
        to_play_seat=np.zeros((1, 2), dtype=np.int8),
        actions=np.asarray([[124, 104]], dtype=np.int64),
        legal_ids=np.asarray([104, 124, 104, 124], dtype=np.uint32),
        legal_offsets=np.asarray([0, 2, 4], dtype=np.uint32),
        legal_action_meta=np.zeros((4, 4), dtype=np.uint16),
        teacher_family=np.full((1, 2), -1, dtype=np.int32),
        teacher_slot=np.full((1, 2), -1, dtype=np.int32),
        teacher_move_source=np.full((1, 2), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, 2), -1, dtype=np.int32),
        teacher_action=np.asarray([[104, 124]], dtype=np.int32),
        teacher_valid=np.ones((1, 2), dtype=np.bool_),
        policy_train_mask=np.ones((1, 2), dtype=np.bool_),
        reset_before_step=np.zeros((1, 2), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": 2,
            "episode_count": 2,
            "time_steps": 1,
            "row_count": 2,
            "train_rows": 2,
            "selected_bundles": [
                {
                    "source_dataset_label": "b2",
                    "source_opponent_policy_id": "B2 HeuristicPublic",
                    "source_pair_index": 205,
                },
                {
                    "source_dataset_label": "learned",
                    "source_opponent_policy_id": "seed_policy_000003",
                    "source_pair_indices": [205],
                },
            ],
        },
    )

    rows = paired_swing_margin_rows_from_packed_scores(
        dataset,
        packed_scores=torch.tensor([0.0, 1.0, 2.0, 0.0]),
        positive_action_source="actions",
        negative_action_source="teacher_action",
        opponent_context_indices=np.asarray([3, 7], dtype=np.int64),
        report_action_ids=(104, 124),
    )

    assert [row["opponent_context_index"] for row in rows] == [3, 7]
    assert rows[0]["positive_action"] == 124
    assert rows[0]["positive_minus_negative_logp"] > 0.0
    assert rows[1]["positive_action"] == 104
    assert rows[1]["positive_minus_negative_logp"] > 0.0
    assert rows[0]["top_action"] == 124
    assert rows[1]["top_action"] == 104
    assert rows[0]["positive_rank"] == 1
    assert rows[0]["negative_rank"] == 2
    assert rows[1]["positive_rank"] == 1
    assert rows[1]["negative_rank"] == 2
    assert rows[0]["reported_action_logps"]["124"] > rows[0]["reported_action_logps"]["104"]
    assert rows[1]["reported_action_logps"]["104"] > rows[1]["reported_action_logps"]["124"]
