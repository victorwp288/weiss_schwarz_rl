from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.diagnostics.trajectory_policy_drift import (
    summarize_policy_drift,
    summarize_policy_drift_by_group,
    summarize_policy_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY_NAMES = ("pass", "play", "attack")


def test_trajectory_policy_drift_requires_fixed_pythonhashseed() -> None:
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/trajectory_policy_drift.py",
            "--stack-config",
            "missing.yaml",
            "--dataset",
            "missing.npz",
            "--policy",
            "direct|runs/missing|training/checkpoints/latest.pt",
            "--output-json",
            "diagnostics/missing.json",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_summarize_policy_scores_reports_target_alignment_by_family() -> None:
    summary = summarize_policy_scores(
        label="policy",
        top_actions=np.asarray([1, 2, 3, 4]),
        target_actions=np.asarray([1, 3, 3, 4]),
        target_probabilities=np.asarray([0.8, 0.2, 0.7, 0.6]),
        target_log_probs=np.log(np.asarray([0.8, 0.2, 0.7, 0.6])),
        top_families=np.asarray([1, 1, 2, 2]),
        target_families=np.asarray([1, 2, 2, 2]),
        row_mask=np.asarray([True, True, True, False]),
        family_names=FAMILY_NAMES,
        values=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    assert summary["row_count"] == 3
    assert summary["top_action_matches_target_rate"] == pytest.approx(2 / 3)
    assert summary["top_family_matches_target_rate"] == pytest.approx(2 / 3)
    assert summary["mean_probability_on_target_action"] == pytest.approx((0.8 + 0.2 + 0.7) / 3)
    assert summary["value_percentiles"]["mean"] == pytest.approx(0.2)
    assert summary["target_family_summaries"][0]["family"] == "attack"


def test_summarize_policy_drift_identifies_lost_target_top_actions_and_drops() -> None:
    summary = summarize_policy_drift(
        reference_label="direct",
        candidate_label="update10",
        reference_top_actions=np.asarray([1, 3, 5, 7]),
        candidate_top_actions=np.asarray([2, 3, 4, 7]),
        reference_target_probabilities=np.asarray([0.7, 0.6, 0.8, 0.1]),
        candidate_target_probabilities=np.asarray([0.1, 0.7, 0.3, 0.1]),
        reference_top_families=np.asarray([1, 2, 2, 0]),
        candidate_top_families=np.asarray([1, 2, 1, 0]),
        target_actions=np.asarray([1, 3, 5, 7]),
        target_families=np.asarray([1, 2, 2, 0]),
        row_mask=np.asarray([True, True, True, False]),
        family_names=FAMILY_NAMES,
        candidate_target_log_probs=np.asarray([-1.0, -0.5, -2.0, -0.1]),
        candidate_top_log_probs=np.asarray([-0.9999995, -0.5, -1.99995, -0.1]),
        reference_values=np.asarray([0.6, 0.5, 0.4, 0.3]),
        candidate_values=np.asarray([0.2, 0.6, 0.1, 0.3]),
        row_coordinates=[
            {"row_index": 0, "step_index": 0},
            {"row_index": 1, "step_index": 1},
            {"row_index": 2, "step_index": 2},
            {"row_index": 3, "step_index": 3},
        ],
        max_examples=1,
    )

    assert summary["row_count"] == 3
    assert summary["top_action_changed_rate"] == pytest.approx(2 / 3)
    assert summary["top_family_changed_rate"] == pytest.approx(1 / 3)
    assert summary["lost_target_top_action_rate"] == pytest.approx(2 / 3)
    assert summary["gained_target_top_action_rate"] == pytest.approx(0.0)
    assert summary["mean_target_action_probability_delta"] == pytest.approx((-0.6 + 0.1 - 0.5) / 3)
    assert summary["mean_value_delta"] == pytest.approx((-0.4 + 0.1 - 0.3) / 3)
    assert summary["lost_target_top_action_same_family_rate"] == pytest.approx(0.5)
    assert summary["lost_target_top_action_abs_probability_delta_lte_1e-5_count"] == 0
    margin_summary = summary["lost_target_top_action_candidate_top_over_target_margin"]
    assert margin_summary["count"] == 2
    assert margin_summary["near_tie_thresholds"][0]["count"] == 1
    assert margin_summary["near_tie_thresholds"][2]["count"] == 2
    assert summary["largest_target_probability_drops"][0]["row_index"] == 0
    assert summary["lost_target_top_action_examples"][0]["candidate_top_over_target_logp_margin"] == pytest.approx(5e-5)


def test_summarize_policy_drift_by_group_splits_rows_by_label() -> None:
    summaries = summarize_policy_drift_by_group(
        group_name="role",
        group_labels=np.asarray(["preferred", "preferred", "rejected", ""]),
        reference_label="direct",
        candidate_label="update10",
        reference_top_actions=np.asarray([1, 3, 5, 7]),
        candidate_top_actions=np.asarray([2, 3, 5, 7]),
        reference_target_probabilities=np.asarray([0.7, 0.6, 0.8, 0.1]),
        candidate_target_probabilities=np.asarray([0.1, 0.7, 0.9, 0.1]),
        reference_top_families=np.asarray([1, 2, 2, 0]),
        candidate_top_families=np.asarray([1, 2, 2, 0]),
        target_actions=np.asarray([1, 3, 5, 7]),
        target_families=np.asarray([1, 2, 2, 0]),
        row_mask=np.asarray([True, True, True, True]),
        family_names=FAMILY_NAMES,
    )

    by_role = {str(summary["role"]): summary for summary in summaries}
    assert set(by_role) == {"preferred", "rejected"}
    assert by_role["preferred"]["row_count"] == 2
    assert by_role["preferred"]["lost_target_top_action_rate"] == pytest.approx(0.5)
    assert by_role["rejected"]["row_count"] == 1
    assert by_role["rejected"]["mean_target_action_probability_delta"] == pytest.approx(0.1)


def test_trajectory_policy_drift_reporting_builds_report_with_group_summaries(tmp_path: Path) -> None:
    from weiss_rl.experiments.trajectory_policy_drift_reporting import (
        PolicyScores,
        build_trajectory_policy_drift_report,
        parse_policy_specs,
        print_trajectory_policy_drift_summary,
        source_opponent_policy_ids_by_episode,
        trajectory_row_coordinates,
        trajectory_row_group_labels,
        write_trajectory_policy_drift_report,
    )

    dataset = SimpleNamespace(
        actions=np.asarray([[1, 2], [1, 2]], dtype=np.int64),
        policy_train_mask=np.asarray([[True, True], [True, False]], dtype=np.bool_),
        episode_count=2,
        time_steps=2,
        metadata={
            "bundle_count": 2,
            "train_rows": 3,
            "row_count": 4,
            "unsupported_target_rows": 1,
            "spec_hash256": "s" * 64,
            "selected_bundles": [
                {
                    "pair_index": 0,
                    "swap_index": 0,
                    "focal_seat": 0,
                    "episode_seed": 101,
                    "preference_pair_id": "pair_a",
                    "preference_role": "preferred",
                    "preference_role_label": "winner",
                    "source_opponent_policy_id": "B2",
                },
                {
                    "pair_index": 0,
                    "swap_index": 1,
                    "focal_seat": 1,
                    "episode_seed": 202,
                    "preference_pair_id": "pair_a",
                    "preference_role": "rejected",
                    "preference_role_label": "loser",
                    "source_opponent_policy_id": "B3",
                },
            ],
        },
    )
    policy_specs = parse_policy_specs(
        [
            f"direct|{tmp_path / 'runs' / 'direct'}|training\\checkpoints\\direct.pt",
            f"update10|{tmp_path / 'runs' / 'update10'}|training/checkpoints/update10.pt",
        ]
    )
    scores_by_label = {
        "direct": PolicyScores(
            label="direct",
            top_actions=np.asarray([1, 2, 1, 2]),
            top_log_probs=np.log(np.asarray([0.8, 0.7, 0.6, 0.5])),
            target_log_probs=np.log(np.asarray([0.8, 0.7, 0.6, 0.5])),
            target_probabilities=np.asarray([0.8, 0.7, 0.6, 0.5]),
            top_families=np.asarray([1, 2, 1, 2]),
            values=np.asarray([0.1, 0.2, 0.3, 0.4]),
            opponent_context_episode_count=2,
        ),
        "update10": PolicyScores(
            label="update10",
            top_actions=np.asarray([2, 2, 1, 2]),
            top_log_probs=np.log(np.asarray([0.75, 0.8, 0.55, 0.5])),
            target_log_probs=np.log(np.asarray([0.2, 0.8, 0.55, 0.5])),
            target_probabilities=np.asarray([0.2, 0.8, 0.55, 0.5]),
            top_families=np.asarray([2, 2, 1, 2]),
            values=np.asarray([0.0, 0.25, 0.25, 0.4]),
            opponent_context_episode_count=2,
        ),
    }

    report = build_trajectory_policy_drift_report(
        stack_config=tmp_path / "stack.yaml",
        dataset_path=tmp_path / "dataset.npz",
        dataset=dataset,
        device="cpu",
        python_hash_seed=123,
        torch_threads=1,
        output_json=tmp_path / "out" / "drift.json",
        reference_label="direct",
        policy_specs=policy_specs,
        scores_by_label=scores_by_label,
        family_names=("pass", "play", "attack"),
        family_by_action=np.asarray([0, 1, 2], dtype=np.int64),
        max_examples=2,
    )

    assert policy_specs[0].checkpoint_relpath == "training/checkpoints/direct.pt"
    assert source_opponent_policy_ids_by_episode(dataset) == ["B2", "B3"]
    assert trajectory_row_coordinates(dataset)[1]["source_opponent_policy_id"] == "B3"
    assert trajectory_row_group_labels(dataset)["preference_role_label"].tolist() == [
        "winner",
        "loser",
        "winner",
        "loser",
    ]
    assert report["format"] == "trajectory_policy_drift_v1"
    assert report["dataset_metadata"]["train_rows"] == 3
    assert report["policies"][0]["label"] == "direct"
    assert report["policy_summaries"][0]["opponent_context_episode_count"] == 2
    assert report["drift_summaries"][0]["candidate_label"] == "update10"
    assert {
        item["preference_role_label"] for item in report["drift_summaries"][0]["preference_role_drift_summaries"]
    } == {
        "winner",
        "loser",
    }
    assert {
        item["source_opponent_policy_id"] for item in report["drift_summaries"][0]["source_opponent_drift_summaries"]
    } == {
        "B2",
        "B3",
    }

    output_json = tmp_path / "out" / "drift.json"
    write_trajectory_policy_drift_report(output_json, report)
    assert output_json.is_file()
    print_trajectory_policy_drift_summary(report)


def test_trajectory_policy_drift_entrypoint_facade_reexports_reporting_helpers() -> None:
    from weiss_rl.experiments import (
        trajectory_policy_drift_cli,
        trajectory_policy_drift_entrypoint,
        trajectory_policy_drift_reporting,
        trajectory_policy_drift_scoring,
    )

    assert trajectory_policy_drift_entrypoint.PolicySpec is trajectory_policy_drift_reporting.PolicySpec
    assert trajectory_policy_drift_entrypoint.PolicyScores is trajectory_policy_drift_reporting.PolicyScores
    assert trajectory_policy_drift_entrypoint._build_parser is (
        trajectory_policy_drift_cli.build_trajectory_policy_drift_parser
    )
    assert trajectory_policy_drift_entrypoint._parse_policy_spec is trajectory_policy_drift_reporting.parse_policy_spec
    assert trajectory_policy_drift_entrypoint.parse_policy_specs is trajectory_policy_drift_reporting.parse_policy_specs
    assert trajectory_policy_drift_entrypoint._score_policy is trajectory_policy_drift_scoring.score_policy
    assert trajectory_policy_drift_entrypoint._dense_policy_scores_from_packed_logits is (
        trajectory_policy_drift_scoring.dense_policy_scores_from_packed_logits
    )
    assert trajectory_policy_drift_entrypoint._safe_actions_for_scoring is (
        trajectory_policy_drift_scoring.safe_actions_for_scoring
    )
    assert trajectory_policy_drift_entrypoint._row_coordinates is (
        trajectory_policy_drift_reporting.trajectory_row_coordinates
    )
    assert trajectory_policy_drift_entrypoint._row_group_labels is (
        trajectory_policy_drift_reporting.trajectory_row_group_labels
    )
    assert trajectory_policy_drift_entrypoint._source_opponent_policy_ids_by_episode is (
        trajectory_policy_drift_reporting.source_opponent_policy_ids_by_episode
    )
    assert trajectory_policy_drift_entrypoint._print_summary is (
        trajectory_policy_drift_reporting.print_trajectory_policy_drift_summary
    )


def test_trajectory_policy_drift_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.trajectory_policy_drift_cli import build_trajectory_policy_drift_parser

    parser = build_trajectory_policy_drift_parser()
    args = parser.parse_args(
        [
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--dataset",
            str(tmp_path / "dataset.npz"),
            "--policy",
            f"direct|{tmp_path / 'run'}|training/checkpoints/latest.pt",
            "--output-json",
            str(tmp_path / "drift.json"),
        ]
    )

    assert args.reference_label is None
    assert args.device == "cuda"
    assert args.torch_threads == 1
    assert args.max_examples == 25


def test_dense_policy_scores_from_packed_logits_handles_finite_and_all_infinite_rows() -> None:
    from weiss_rl.experiments.trajectory_policy_drift_scoring import dense_policy_scores_from_packed_logits

    top_actions, target_logp, top_logp = dense_policy_scores_from_packed_logits(
        np.asarray([0.0, 1.0, -np.inf, -np.inf], dtype=np.float64),
        legal_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        legal_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        target_actions=np.asarray([1, 4], dtype=np.int64),
    )

    assert top_actions.tolist() == [2, 3]
    assert target_logp[0] == pytest.approx(-np.log1p(np.e))
    assert top_logp[0] == pytest.approx(1.0 - np.log1p(np.e))
    assert target_logp[1] == pytest.approx(-np.log(2.0))
    assert top_logp[1] == pytest.approx(-np.log(2.0))


def test_safe_actions_for_scoring_replaces_non_train_rows_with_legal_placeholders() -> None:
    from weiss_rl.experiments.trajectory_policy_drift_scoring import safe_actions_for_scoring

    dataset = SimpleNamespace(
        actions=np.asarray([[9, 999]], dtype=np.int64),
        policy_train_mask=np.asarray([[True, False]], dtype=np.bool_),
        legal_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        legal_ids=np.asarray([9, 10, 7, 8], dtype=np.int64),
    )

    assert safe_actions_for_scoring(dataset).tolist() == [[9, 7]]
