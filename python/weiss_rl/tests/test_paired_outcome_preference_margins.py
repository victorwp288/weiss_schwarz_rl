from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_preference_margins import (
    PairedOutcomePreferenceMarginConfig,
    preference_margin_rows_from_logps,
)
from weiss_rl.experiments.paired_outcome_preference_margins_reporting import (
    paired_outcome_preference_margins_output_line,
    paired_outcome_preference_margins_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_margins_runtime import (
    paired_outcome_preference_margin_config_from_args,
)


def test_paired_outcome_preference_margins_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_margins,
        paired_outcome_preference_margins_cli,
        paired_outcome_preference_margins_entrypoint,
        paired_outcome_preference_margins_runtime,
    )

    assert paired_outcome_preference_margins_entrypoint._build_parser is (
        paired_outcome_preference_margins_cli.build_paired_outcome_preference_margins_parser
    )
    assert paired_outcome_preference_margins_entrypoint.run_paired_outcome_preference_margins is (
        paired_outcome_preference_margins_runtime.run_paired_outcome_preference_margins
    )
    assert paired_outcome_preference_margins_entrypoint.PairedOutcomePreferenceMarginConfig is (
        paired_outcome_preference_margins.PairedOutcomePreferenceMarginConfig
    )
    assert paired_outcome_preference_margins_entrypoint.build_paired_outcome_preference_margin_report is (
        paired_outcome_preference_margins.build_paired_outcome_preference_margin_report
    )
    assert paired_outcome_preference_margins_entrypoint.write_paired_outcome_preference_margin_report is (
        paired_outcome_preference_margins.write_paired_outcome_preference_margin_report
    )


def test_paired_outcome_preference_margins_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_margins_cli import (
        build_paired_outcome_preference_margins_parser,
    )

    args = build_paired_outcome_preference_margins_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "preference.npz"),
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--reference-checkpoint",
            str(tmp_path / "reference.pt"),
            "--output-json",
            str(tmp_path / "margins.json"),
        ]
    )

    assert args.dataset == tmp_path / "preference.npz"
    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.run_dir == tmp_path / "run"
    assert args.checkpoint == tmp_path / "checkpoint.pt"
    assert args.reference_checkpoint == tmp_path / "reference.pt"
    assert args.aggregation == "mean"
    assert args.output_json == tmp_path / "margins.json"


def test_paired_outcome_preference_margins_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "preference.npz",
        stack_config=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "checkpoint.pt",
        reference_checkpoint=tmp_path / "reference.pt",
        aggregation="sum",
    )

    config = paired_outcome_preference_margin_config_from_args(args)

    assert config == PairedOutcomePreferenceMarginConfig(
        dataset_path=tmp_path / "preference.npz",
        stack_config_path=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        reference_checkpoint_path=tmp_path / "reference.pt",
        aggregation="sum",
    )


def test_paired_outcome_preference_margins_reporting_preserves_compact_console_json(tmp_path: Path) -> None:
    report = {
        "pair_count": 4,
        "train_rows": 12,
        "dpo_margin_mean": 0.25,
        "dpo_margin_min": -0.5,
        "satisfied_fraction": 0.75,
        "current_context_episode_count": 7,
        "reference_context_episode_count": 6,
        "current_context_coverage": {"missing_context_episode_count": 1},
        "reference_context_coverage": {"missing_context_episode_count": 2},
    }

    assert paired_outcome_preference_margins_output_payload(
        output_json=tmp_path / "margins.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "margins.json").as_posix(),
        "pair_count": 4,
        "train_rows": 12,
        "dpo_margin_mean": 0.25,
        "dpo_margin_min": -0.5,
        "satisfied_fraction": 0.75,
        "current_context_episode_count": 7,
        "reference_context_episode_count": 6,
        "current_missing_context_episode_count": 1,
        "reference_missing_context_episode_count": 2,
    }
    assert paired_outcome_preference_margins_output_line(
        output_json=tmp_path / "margins.json",
        report=report,
    ) == (
        '{"current_context_episode_count": 7, "current_missing_context_episode_count": 1, '
        '"dpo_margin_mean": 0.25, "dpo_margin_min": -0.5, '
        f'"output_json": "{(tmp_path / "margins.json").as_posix()}", '
        '"pair_count": 4, "reference_context_episode_count": 6, '
        '"reference_missing_context_episode_count": 2, "satisfied_fraction": 0.75, "train_rows": 12}'
    )


def test_preference_margin_rows_from_logps_groups_complete_pairs() -> None:
    selected_bundles = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "preference_role_label": "preferred",
            "merge_source_dataset_label": "learned_repair",
            "source_opponent_policy_id": "policy_000003",
            "source_pair_index": 205,
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "preference_role_label": "rejected",
            "merge_source_dataset_label": "learned_repair",
            "source_opponent_policy_id": "policy_000003",
            "source_pair_index": 205,
        },
    ]
    current = np.asarray([[-1.0, -3.0], [-2.0, -4.0]], dtype=np.float32)
    reference = np.asarray([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 0], [0, 0]], dtype=np.int64)
    roles = np.asarray([[1, 0], [1, 0]], dtype=np.int64)
    mask = np.ones((2, 2), dtype=np.bool_)

    rows = preference_margin_rows_from_logps(
        selected_bundles=selected_bundles,
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=mask,
        aggregation="mean",
    )

    assert len(rows) == 1
    assert rows[0]["group_label"] == "learned_repair"
    assert rows[0]["preferred_rows"] == 2
    assert rows[0]["rejected_rows"] == 2
    assert rows[0]["current_raw_margin"] == pytest.approx(2.0)
    assert rows[0]["reference_raw_margin"] == pytest.approx(0.0)
    assert rows[0]["dpo_margin"] == pytest.approx(2.0)


def test_preference_margin_rows_from_logps_falls_back_to_source_dataset_label() -> None:
    selected_bundles = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "source_dataset_label": "fixed_protect",
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "source_dataset_label": "fixed_protect",
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
    ]
    current = np.asarray([[-1.0, -2.0]], dtype=np.float32)
    reference = np.asarray([[-1.0, -2.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 0]], dtype=np.int64)
    roles = np.asarray([[1, 0]], dtype=np.int64)

    rows = preference_margin_rows_from_logps(
        selected_bundles=selected_bundles,
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=np.ones((1, 2), dtype=np.bool_),
    )

    assert rows[0]["group_label"] == "fixed_protect"


def test_preference_margin_rows_from_logps_skips_incomplete_pairs() -> None:
    current = np.asarray([[-1.0, -3.0]], dtype=np.float32)
    reference = np.asarray([[-1.0, -3.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 1]], dtype=np.int64)
    roles = np.asarray([[1, 0]], dtype=np.int64)
    mask = np.ones((1, 2), dtype=np.bool_)

    rows = preference_margin_rows_from_logps(
        selected_bundles=[],
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=mask,
    )

    assert rows == []
