from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.replay.inspector_report import format_replay_inspection_report

from .replay_inspector_test_support import _heuristic_obs_with_stage, _inspect_with_heuristic_public_policy


def test_inspect_replay_bundle_reports_heuristic_public_family_details(tmp_path: Path) -> None:
    report = _inspect_with_heuristic_public_policy(
        tmp_path,
        policy_b=HEURISTIC_PUBLIC_POLICY_ID,
        policy_a_logits={51: 3.0, 472: 0.5, 473: 0.2, 474: 0.0},
        top_actions=3,
        obs=_heuristic_obs_with_stage(include_counts=True),
    )

    assert report["policy_b"]["kind"] == "heuristic_public"
    assert report["top_differences"][0]["policy_a_top_action"]["family"] == "pass"
    assert report["top_differences"][0]["policy_b_top_action"]["family"] == "attack"
    assert report["top_differences"][0]["policy_b_top_action"]["attack_type"] == "direct"
    assert report["top_differences"][0]["policy_a_probability_on_policy_b_top_action"] == pytest.approx(
        0.0417437858,
        rel=1e-6,
    )
    assert report["top_differences"][0]["policy_a_probability_on_policy_b_top_action_family"] == pytest.approx(
        0.1615536310,
        rel=1e-6,
    )
    assert report["summary"]["policy_a_mean_probability_on_policy_b_top_action_family"] == pytest.approx(
        0.1615536310,
        rel=1e-6,
    )
    assert report["summary"]["policy_a_mean_family_probability_masses"][0]["family"] == "pass"
    assert report["top_differences"][0]["policy_a_rank_of_policy_b_top_action"] == 4
    assert report["summary"]["policy_a_matches_policy_b_top_action_rate"] == 0.0
    assert report["summary"]["policy_a_matches_policy_b_top_action_family_rate"] == 0.0
    assert report["summary"]["policy_a_top_action_mismatch_count"] == 1
    assert report["summary"]["policy_a_top_action_family_mismatch_count"] == 1
    assert report["summary"]["top_action_family_confusions"][0] == {
        "policy_b_family": "attack",
        "policy_a_family": "pass",
        "count": 1,
    }


def test_replay_inspection_report_includes_heuristic_public_trajectory_summary(tmp_path: Path) -> None:
    report = _inspect_with_heuristic_public_policy(
        tmp_path,
        policy_b=HEURISTIC_PUBLIC_POLICY_ID,
        policy_a_logits={51: 3.0, 472: 0.5, 473: 0.2, 474: 0.0},
        top_actions=3,
        obs=_heuristic_obs_with_stage(include_counts=True),
    )

    trajectory_summary = report["trajectory_summary"]
    assert trajectory_summary["recorded_family_counts"][0] == {"family": "attack", "count": 1}
    assert trajectory_summary["decision_kind_counts"] == [{"decision_kind": "0", "count": 1}]
    assert trajectory_summary["legal_family_presence_rates"][-2:] == [
        {"family": "attack", "rate": 1.0},
        {"family": "pass", "rate": 1.0},
    ]
    assert trajectory_summary["numeric_summaries"]["self_clock_count"]["mean"] == pytest.approx(6.0)
    assert trajectory_summary["numeric_summaries"]["self_hand_count"]["mean"] == pytest.approx(7.0)
    assert trajectory_summary["numeric_summaries"]["opponent_clock_count"]["mean"] == pytest.approx(4.0)
    assert trajectory_summary["numeric_summaries"]["self_stage_occupied_count"]["mean"] == pytest.approx(1.0)
    assert trajectory_summary["actor_summaries"][0]["actor"] == 0
    assert trajectory_summary["actor_summaries"][0]["recorded_family_counts"][0] == {"family": "attack", "count": 1}

    text_report = format_replay_inspection_report(report)
    assert "trajectory:" in text_report
    assert "attack->pass x1" in text_report
    assert "a474[attack, slot=0, attack_type=direct]" in text_report
    assert "family_match=False" in text_report
