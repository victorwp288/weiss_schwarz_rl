from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.diagnostics import trajectory_audit_compare_entrypoint as audit_compare_module


def test_compare_audit_summaries_reports_outcome_role_and_numeric_deltas(tmp_path: Path) -> None:
    module = audit_compare_module
    baseline_path = tmp_path / "baseline" / "summary.json"
    candidate_path = tmp_path / "candidate" / "summary.json"
    baseline_path.parent.mkdir()
    candidate_path.parent.mkdir()
    baseline_episodes = baseline_path.parent / "episodes.jsonl"
    candidate_episodes = candidate_path.parent / "episodes.jsonl"
    baseline_episodes.write_text(
        json.dumps({"outcome": "W", "pass_actions": 2, "pass_with_nonpass_available": 1})
        + "\n"
        + json.dumps({"outcome": "L", "pass_actions": 4, "pass_with_nonpass_available": 3})
        + "\n",
        encoding="utf-8",
    )
    candidate_episodes.write_text(
        json.dumps({"outcome": "W", "pass_actions": 1, "pass_with_nonpass_available": 0})
        + "\n"
        + json.dumps({"outcome": "W", "pass_actions": 3, "pass_with_nonpass_available": 1})
        + "\n",
        encoding="utf-8",
    )
    baseline = _summary_payload(
        episodes_path=baseline_episodes,
        action_match=0.4,
        focal_clock=3.0,
        focal_families=[{"family": "pass", "count": 7}],
        legal_attack_rate=0.5,
    )
    candidate = _summary_payload(
        episodes_path=candidate_episodes,
        action_match=0.6,
        focal_clock=2.0,
        focal_families=[{"family": "pass", "count": 5}, {"family": "attack", "count": 4}],
        legal_attack_rate=0.75,
    )

    comparison = module.compare_audit_summaries(
        baseline=baseline,
        candidate=candidate,
        baseline_summary_path=baseline_path,
        candidate_summary_path=candidate_path,
        baseline_label="failed",
        candidate_label="seed",
    )

    assert comparison["alignment"]["policy_a_matches_policy_b_top_action_rate"]["delta"] == pytest.approx(0.2)
    assert comparison["episode_outcomes"]["baseline"]["mean"] == 0.5
    assert comparison["episode_outcomes"]["candidate"]["mean"] == 1.0
    assert comparison["episode_outcomes"]["delta"]["pass_actions_per_game"] == -1.0
    focal_delta = comparison["roles"]["focal"]["delta"]
    assert focal_delta["numeric_means"]["self_clock_count"] == -1.0
    assert focal_delta["recorded_family_counts"][0] == {
        "family": "attack",
        "baseline": 0,
        "candidate": 4,
        "delta": 4,
    }
    assert focal_delta["legal_family_presence_rates"][0] == {
        "family": "attack",
        "baseline": 0.5,
        "candidate": 0.75,
        "delta": 0.25,
    }


def _summary_payload(
    *,
    episodes_path: Path,
    action_match: float,
    focal_clock: float,
    focal_families: list[dict[str, object]],
    legal_attack_rate: float,
) -> dict[str, object]:
    return {
        "episodes_path": episodes_path.as_posix(),
        "policy_id": "policy",
        "opponent_policy_id": "B4 HeuristicPublicControl",
        "policy_a_matches_policy_b_top_action_rate": action_match,
        "policy_a_matches_policy_b_top_action_family_rate": 0.9,
        "policy_a_mean_probability_on_policy_b_top_action": 0.5,
        "policy_a_mean_probability_on_policy_b_top_action_family": 0.8,
        "trajectory_summary": {
            "role_summaries": [
                {
                    "role": "focal",
                    "compared_steps": 10,
                    "recorded_family_counts": focal_families,
                    "numeric_summaries": {
                        "self_clock_count": {
                            "count": 10,
                            "mean": focal_clock,
                            "p10": focal_clock,
                            "p25": focal_clock,
                            "p50": focal_clock,
                            "p75": focal_clock,
                            "p90": focal_clock,
                        }
                    },
                    "legal_family_presence_rates": [{"family": "attack", "rate": legal_attack_rate}],
                }
            ]
        },
    }
