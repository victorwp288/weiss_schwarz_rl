from __future__ import annotations

from weiss_rl.experiments.opponent_context_coverage import (
    context_coverage_failures_from_report,
    summarize_opponent_context_coverage,
)


def test_summarize_opponent_context_coverage_reports_missing_opponents() -> None:
    summary = summarize_opponent_context_coverage(
        ["B2 HeuristicPublic", "B2 HeuristicPublic", "policy_000004", ""],
        [1, 0, 2, 0],
    )

    assert summary["episode_count"] == 4
    assert summary["context_episode_count"] == 2
    assert summary["missing_context_episode_count"] == 2
    assert summary["empty_opponent_id_episode_count"] == 1
    assert summary["missing_context_opponent_policy_ids"] == ["B2 HeuristicPublic"]
    assert summary["mapped_opponent_policy_ids"] == ["B2 HeuristicPublic", "policy_000004"]
    assert {
        item["opponent_policy_id"]: item["context_indices"]
        for item in summary["opponents"]
        if item["opponent_policy_id"]
    } == {
        "B2 HeuristicPublic": [0, 1],
        "policy_000004": [2],
    }


def test_context_coverage_failures_require_full_episode_coverage() -> None:
    report = {
        "episode_count": 3,
        "current_context_episode_count": 2,
        "current_context_coverage": {
            "episode_count": 3,
            "context_episode_count": 2,
            "empty_opponent_id_episode_count": 0,
            "missing_context_opponent_policy_ids": ["B4 HeuristicPublicControl"],
        },
    }

    failures = context_coverage_failures_from_report(
        report,
        coverage_key="current_context_coverage",
        context_count_key="current_context_episode_count",
        prefix="current",
    )

    assert "current_context_episodes_below:2<3" in failures
    assert "current_coverage_context_episodes_below:2<3" in failures
    assert "current_missing_context_opponents:B4 HeuristicPublicControl" in failures
