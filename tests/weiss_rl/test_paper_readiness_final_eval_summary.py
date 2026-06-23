from __future__ import annotations

from pathlib import Path

from weiss_rl.eval import build_paper_readiness_summary

from tests.weiss_rl.paper_readiness_test_support import (
    write_final_eval_fixture,
    write_multi_policy_final_eval_fixture,
)


def test_build_paper_readiness_summary_passes_with_balanced_seats_and_strong_b0_winrate(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is True
    assert payload["alarms"] == []
    assert payload["checks"]["truncation_rate"]["rate"] == 0.0
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_source"] == "sole_eligible_non_baseline"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_flags_truncation_seat_bias_and_weak_b0_matchup(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.49, 0.5, 0.52, 0.54],
        seat_results=[
            (2, 0),
            {"seat0_wins": 2, "seat1_wins": 0, "truncations": 1},
            (2, 0),
            (2, 0),
        ],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is False
    assert payload["alarms"] == [
        "truncation_rate",
        "seat_bias_alarm",
        "baseline_win_rate_vs_b0",
    ]
    assert payload["checks"]["truncation_rate"]["rate"] == 1 / 7
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["passed"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 0.0


def test_build_paper_readiness_summary_ignores_reciprocal_matchups_for_guardrail_aggregation(tmp_path: Path) -> None:
    baseline_dir = write_final_eval_fixture(
        tmp_path / "baseline",
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )
    reciprocal_noise_dir = write_final_eval_fixture(
        tmp_path / "reciprocal_noise",
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[
            (1, 1),
            (1, 1),
            {"seat0_wins": 20, "seat1_wins": 0, "truncations": 20},
            (1, 1),
        ],
        games_matrix=[[2, 2], [40, 2]],
        truncation_matrix=[[0, 0], [20, 0]],
    )

    baseline_payload = build_paper_readiness_summary(final_eval_dir=baseline_dir)
    reciprocal_noise_payload = build_paper_readiness_summary(final_eval_dir=reciprocal_noise_dir)

    assert reciprocal_noise_payload["passed"] == baseline_payload["passed"]
    assert reciprocal_noise_payload["alarms"] == baseline_payload["alarms"]
    assert reciprocal_noise_payload["checks"]["truncation_rate"] == baseline_payload["checks"]["truncation_rate"]
    assert (
        reciprocal_noise_payload["checks"]["seat_bias_alarm"]["observed"]
        == baseline_payload["checks"]["seat_bias_alarm"]["observed"]
    )
    assert [
        matchup["diagnostics_path"] for matchup in reciprocal_noise_payload["checks"]["seat_bias_alarm"]["per_matchup"]
    ] == [
        "matchups/00_b0_randomlegal__vs__/00_b0_randomlegal/diagnostics.json",
        "matchups/00_b0_randomlegal__vs__/01_policy_000300/diagnostics.json",
        "matchups/01_policy_000300__vs__/01_policy_000300/diagnostics.json",
    ]


def test_build_paper_readiness_summary_requires_explicit_focal_policy_when_multiple_candidates(tmp_path: Path) -> None:
    final_eval_dir = write_multi_policy_final_eval_fixture(tmp_path)

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is False
    assert payload["alarms"] == ["baseline_win_rate_vs_b0"]
    assert check["focal_policy_id"] is None
    assert check["reason"] == "ambiguous_non_baseline_focal_policy"
    assert check["eligible_non_baseline_policy_ids"] == ["policy_000300", "policy_000400"]
    assert "pass --focal-policy-id" in check["message"]


def test_build_paper_readiness_summary_uses_metadata_named_focal_policy(tmp_path: Path) -> None:
    final_eval_dir = write_multi_policy_final_eval_fixture(
        tmp_path,
        metadata={
            "selection": {"mode": "deterministic_v1"},
            "focal_policy": {"policy_id": "policy_000400"},
        },
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000400"
    assert check["focal_policy_source"] == "metadata"
    assert check["passed"] is True
    assert check["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_uses_recommended_focal_policy_metadata(tmp_path: Path) -> None:
    final_eval_dir = write_multi_policy_final_eval_fixture(
        tmp_path,
        metadata={
            "selection": {"mode": "deterministic_v1"},
            "recommended_focal_policy_id": "policy_000400",
        },
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000400"
    assert check["focal_policy_source"] == "metadata"
    assert check["passed"] is True
    assert check["prob_gt_threshold"] == 1.0
