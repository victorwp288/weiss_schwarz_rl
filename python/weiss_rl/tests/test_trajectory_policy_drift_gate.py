from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.trajectory_policy_drift_gate import (
    TrajectoryPolicyDriftGateConfig,
    evaluate_trajectory_policy_drift_gate,
)


def test_trajectory_policy_drift_gate_passes_clean_context_covered_report(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "drift.json",
        lost=0.0,
        gained=0.1,
        mean_probability_delta=0.02,
        largest_drop=0.0,
        reference_match=0.7,
        candidate_match=0.8,
    )

    report = evaluate_trajectory_policy_drift_gate(
        TrajectoryPolicyDriftGateConfig(
            drift_report_json=path,
            min_gained_target_top_action_rate=0.05,
            min_gain_minus_loss_rate=0.05,
        )
    )

    assert report["passed"] is True
    assert report["summary"]["gain_minus_loss_rate"] == 0.1


def test_trajectory_policy_drift_gate_fails_target_top_loss_and_context_gap(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "drift.json",
        lost=0.07,
        gained=0.002,
        mean_probability_delta=0.01,
        largest_drop=-0.03,
        reference_match=0.97,
        candidate_match=0.90,
        candidate_context=4,
    )

    report = evaluate_trajectory_policy_drift_gate(
        TrajectoryPolicyDriftGateConfig(
            drift_report_json=path,
            max_target_probability_drop=0.02,
        )
    )

    assert report["passed"] is False
    assert "candidate_context_episodes_below:4<10" in report["failures"]
    assert any(item.startswith("lost_target_top_action_rate_above") for item in report["failures"])
    assert any(item.startswith("gain_minus_loss_rate_below") for item in report["failures"])
    assert any(item.startswith("largest_target_probability_drop_below") for item in report["failures"])
    assert any(item.startswith("top_action_match_delta_below") for item in report["failures"])


def test_trajectory_policy_drift_gate_can_track_near_tie_top_action_flips(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "drift.json",
        lost=0.07,
        gained=0.0,
        mean_probability_delta=0.0,
        largest_drop=-1e-7,
        reference_match=0.97,
        candidate_match=0.90,
        lost_near_tie_count=7,
        top_changed_near_tie_count=10,
    )

    report = evaluate_trajectory_policy_drift_gate(
        TrajectoryPolicyDriftGateConfig(
            drift_report_json=path,
            max_lost_target_top_action_rate=0.08,
            min_gain_minus_loss_rate=-0.08,
            max_top_action_match_drop_rate=0.08,
            max_target_probability_drop=1e-6,
            top_action_near_tie_margin=1e-5,
            max_lost_target_non_near_tie_rate=0.0,
            max_top_action_changed_non_near_tie_rate=0.0,
        )
    )

    assert report["passed"] is True
    assert report["summary"]["lost_target_non_near_tie_rate"] == 0.0
    assert report["summary"]["top_action_changed_non_near_tie_rate"] == 0.0


def test_trajectory_policy_drift_gate_fails_non_near_tie_top_action_flips(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "drift.json",
        lost=0.07,
        gained=0.0,
        mean_probability_delta=0.0,
        largest_drop=-1e-7,
        reference_match=0.97,
        candidate_match=0.90,
        lost_near_tie_count=6,
        top_changed_near_tie_count=9,
    )

    report = evaluate_trajectory_policy_drift_gate(
        TrajectoryPolicyDriftGateConfig(
            drift_report_json=path,
            max_lost_target_top_action_rate=0.08,
            min_gain_minus_loss_rate=-0.08,
            max_top_action_match_drop_rate=0.08,
            max_target_probability_drop=1e-6,
            top_action_near_tie_margin=1e-5,
            max_lost_target_non_near_tie_rate=0.0,
            max_top_action_changed_non_near_tie_rate=0.0,
        )
    )

    assert report["passed"] is False
    assert any(item.startswith("lost_target_non_near_tie_rate_above") for item in report["failures"])
    assert any(item.startswith("top_action_changed_non_near_tie_rate_above") for item in report["failures"])


def _write_report(
    path: Path,
    *,
    lost: float,
    gained: float,
    mean_probability_delta: float,
    largest_drop: float,
    reference_match: float,
    candidate_match: float,
    candidate_context: int = 10,
    lost_near_tie_count: int = 0,
    top_changed_near_tie_count: int = 0,
) -> Path:
    payload = {
        "format": "trajectory_policy_drift_v1",
        "reference_label": "pre",
        "dataset_metadata": {
            "bundle_count": 10,
            "train_rows": 100,
        },
        "policy_summaries": [
            {
                "label": "pre",
                "opponent_context_episode_count": 10,
                "top_action_matches_target_rate": reference_match,
            },
            {
                "label": "post",
                "opponent_context_episode_count": candidate_context,
                "top_action_matches_target_rate": candidate_match,
            },
        ],
        "drift_summaries": [
            {
                "reference_label": "pre",
                "candidate_label": "post",
                "row_count": 100,
                "top_action_changed_rate": 0.1,
                "top_family_changed_rate": 0.0,
                "lost_target_top_action_rate": lost,
                "gained_target_top_action_rate": gained,
                "mean_target_action_probability_delta": mean_probability_delta,
                "lost_target_top_action_candidate_top_over_target_margin": {
                    "near_tie_thresholds": [
                        {
                            "threshold": 1e-5,
                            "count": lost_near_tie_count,
                            "rate": lost_near_tie_count / 7.0 if lost > 0.0 else 0.0,
                        }
                    ]
                },
                "top_action_changed_candidate_top_over_target_margin": {
                    "near_tie_thresholds": [
                        {
                            "threshold": 1e-5,
                            "count": top_changed_near_tie_count,
                            "rate": top_changed_near_tie_count / 10.0,
                        }
                    ]
                },
                "largest_target_probability_drops": [
                    {
                        "probability_delta": largest_drop,
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
