from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_preference_edge_margins import (
    PairedOutcomePreferenceEdgeMarginConfig,
    edge_margin_rows_from_logps,
    evaluate_edge_margin_rows,
)
from weiss_rl.experiments.paired_outcome_preference_edge_margins_reporting import (
    paired_outcome_preference_edge_margins_output_line,
    paired_outcome_preference_edge_margins_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_edge_margins_runtime import (
    PairedOutcomePreferenceEdgeMarginRunResult,
    paired_outcome_preference_edge_margin_config_from_args,
)


def _coverage(count: int) -> dict[str, object]:
    return {
        "episode_count": count,
        "context_episode_count": count,
        "missing_context_episode_count": 0,
        "empty_opponent_id_episode_count": 0,
        "missing_context_opponent_policy_ids": [],
        "mapped_opponent_policy_ids": ["B2 HeuristicPublic"],
        "opponents": [],
    }


def test_paired_outcome_preference_edge_margins_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_edge_margins,
        paired_outcome_preference_edge_margins_cli,
        paired_outcome_preference_edge_margins_entrypoint,
        paired_outcome_preference_edge_margins_runtime,
    )

    assert paired_outcome_preference_edge_margins_entrypoint._build_parser is (
        paired_outcome_preference_edge_margins_cli.build_paired_outcome_preference_edge_margins_parser
    )
    assert paired_outcome_preference_edge_margins_entrypoint.run_paired_outcome_preference_edge_margins is (
        paired_outcome_preference_edge_margins_runtime.run_paired_outcome_preference_edge_margins
    )
    assert paired_outcome_preference_edge_margins_entrypoint.PairedOutcomePreferenceEdgeMarginConfig is (
        paired_outcome_preference_edge_margins.PairedOutcomePreferenceEdgeMarginConfig
    )
    assert paired_outcome_preference_edge_margins_entrypoint.build_paired_outcome_preference_edge_margin_report is (
        paired_outcome_preference_edge_margins.build_paired_outcome_preference_edge_margin_report
    )
    assert paired_outcome_preference_edge_margins_entrypoint.write_paired_outcome_preference_edge_margin_report is (
        paired_outcome_preference_edge_margins.write_paired_outcome_preference_edge_margin_report
    )


def test_paired_outcome_preference_edge_margins_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_edge_margins_cli import (
        build_paired_outcome_preference_edge_margins_parser,
    )

    args = build_paired_outcome_preference_edge_margins_parser().parse_args(
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
            str(tmp_path / "edge_margins.json"),
        ]
    )

    assert args.dataset == tmp_path / "preference.npz"
    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.run_dir == tmp_path / "run"
    assert args.checkpoint == tmp_path / "checkpoint.pt"
    assert args.reference_checkpoint == tmp_path / "reference.pt"
    assert args.spec_bundle_json is None
    assert args.include_same_action is False
    assert args.min_mean_delta == 0.0
    assert args.min_min_delta == 0.0
    assert args.min_edge_improved_fraction == 1.0
    assert args.max_edge_worsened_fraction == 0.0
    assert args.min_same_state_mean_delta == 0.0
    assert args.min_required_group_mean_delta == 0.0
    assert args.required_group == []
    assert args.allow_missing_context is False
    assert args.output_json == tmp_path / "edge_margins.json"


def test_paired_outcome_preference_edge_margins_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "preference.npz",
        stack_config=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "checkpoint.pt",
        reference_checkpoint=tmp_path / "reference.pt",
        spec_bundle_json=tmp_path / "spec_bundle.json",
        include_same_action=True,
        min_mean_delta=0.1,
        min_min_delta=-0.2,
        min_edge_improved_fraction=0.75,
        max_edge_worsened_fraction=0.05,
        min_same_state_mean_delta=0.03,
        min_required_group_mean_delta=0.04,
        required_group=["fixed_preserve", "learned_repair"],
        allow_missing_context=True,
    )

    config = paired_outcome_preference_edge_margin_config_from_args(args)

    assert config == PairedOutcomePreferenceEdgeMarginConfig(
        dataset_path=tmp_path / "preference.npz",
        stack_config_path=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        reference_checkpoint_path=tmp_path / "reference.pt",
        spec_bundle_json=tmp_path / "spec_bundle.json",
        include_same_action=True,
        min_mean_delta=0.1,
        min_min_delta=-0.2,
        min_edge_improved_fraction=0.75,
        max_edge_worsened_fraction=0.05,
        min_same_state_mean_delta=0.03,
        min_required_group_mean_delta=0.04,
        required_groups=("fixed_preserve", "learned_repair"),
        require_context=False,
    )


def test_paired_outcome_preference_edge_margins_reporting_preserves_compact_gate_json(tmp_path: Path) -> None:
    report = {
        "passed": False,
        "failures": ["min_delta_below:-0.1<0"],
        "summary": {"mean_delta": 0.2, "min_delta": -0.1},
    }

    assert paired_outcome_preference_edge_margins_output_payload(
        output_json=tmp_path / "edge_margins.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "edge_margins.json").as_posix(),
        "passed": False,
        "failures": ["min_delta_below:-0.1<0"],
        "summary": {"mean_delta": 0.2, "min_delta": -0.1},
    }
    assert paired_outcome_preference_edge_margins_output_line(
        output_json=tmp_path / "edge_margins.json",
        report=report,
    ) == (
        '{"failures": ["min_delta_below:-0.1<0"], '
        f'"output_json": "{(tmp_path / "edge_margins.json").as_posix()}", '
        '"passed": false, "summary": {"mean_delta": 0.2, "min_delta": -0.1}}'
    )


def test_paired_outcome_preference_edge_margins_run_result_exit_code_tracks_gate_status(tmp_path: Path) -> None:
    assert (
        PairedOutcomePreferenceEdgeMarginRunResult(
            output_json=tmp_path / "edge_margins.json",
            report={"passed": True},
        ).exit_code
        == 0
    )
    assert (
        PairedOutcomePreferenceEdgeMarginRunResult(
            output_json=tmp_path / "edge_margins.json",
            report={"passed": False},
        ).exit_code
        == 2
    )


def test_edge_margin_rows_compare_aligned_preferred_and_rejected_actions() -> None:
    time_steps = 2
    episode_count = 2
    actions = np.asarray([[10, 20], [30, 30]], dtype=np.int64)
    obs = np.asarray(
        [
            [[1.0, 2.0], [1.0, 2.0]],
            [[3.0, 4.0], [3.0, 4.0]],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(
        actions=actions,
        policy_train_mask=np.ones((time_steps, episode_count), dtype=np.bool_),
        obs=obs,
        actor=np.zeros((time_steps, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int8),
        legal_ids=np.tile(np.asarray([10, 20, 30], dtype=np.uint32), time_steps * episode_count),
        legal_offsets=np.arange(0, (time_steps * episode_count + 1) * 3, 3, dtype=np.uint32),
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        episode_count=episode_count,
        time_steps=time_steps,
    )
    bundles = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "preference_role_label": "selected_win_preferred",
            "source_opponent_policy_id": "B2 HeuristicPublic",
            "source_pair_index": 205,
            "merge_source_dataset_label": "fixed_preserve",
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "preference_role_label": "candidate_loss_rejected",
            "source_opponent_policy_id": "B2 HeuristicPublic",
            "source_pair_index": 205,
            "merge_source_dataset_label": "fixed_preserve",
        },
    ]
    current_logp = np.asarray([[-1.0, -3.0], [-0.2, -0.2]], dtype=np.float32)
    reference_logp = np.asarray([[-2.0, -2.5], [-0.2, -0.2]], dtype=np.float32)

    rows = edge_margin_rows_from_logps(
        dataset=dataset,
        selected_bundles=bundles,
        current_action_logp=current_logp,
        reference_action_logp=reference_logp,
    )

    assert len(rows) == 1
    assert rows[0]["preference_pair_id"] == 0
    assert rows[0]["group_label"] == "fixed_preserve"
    assert rows[0]["same_current_state"] is True
    assert rows[0]["edge_delta"] == pytest.approx(1.5)


def test_edge_margin_gate_fails_on_any_worsened_edge() -> None:
    rows = [
        {
            "preference_pair_id": 0,
            "group_label": "fixed_preserve",
            "source_opponent_policy_id": "B2 HeuristicPublic",
            "source_pair_index": 205,
            "same_current_state": True,
            "same_history": True,
            "edge_delta": 0.1,
        },
        {
            "preference_pair_id": 1,
            "group_label": "learned_repair",
            "source_opponent_policy_id": "seed_c3aac2f9dc_policy_000005",
            "source_pair_index": 205,
            "same_current_state": False,
            "same_history": False,
            "edge_delta": -0.01,
        },
    ]

    report = evaluate_edge_margin_rows(
        rows,
        config=PairedOutcomePreferenceEdgeMarginConfig(
            dataset_path=Path("dataset.npz"),
            stack_config_path=Path("config.yaml"),
            run_dir=Path("run"),
            checkpoint_path=Path("post.pt"),
            reference_checkpoint_path=Path("pre.pt"),
            required_groups=("fixed_preserve", "learned_repair"),
        ),
        episode_count=2,
        train_rows=2,
        current_context_coverage=_coverage(2),
        reference_context_coverage=_coverage(2),
    )

    assert report["passed"] is False
    assert any(str(failure).startswith("min_delta_below") for failure in report["failures"])
    assert any(str(failure).startswith("edge_worsened_fraction_above") for failure in report["failures"])
    assert report["summary"]["edge_improved_fraction"] == pytest.approx(0.5)
