from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.policy_set import RANDOM_LEGAL_POLICY_ID


def _write_final_eval_fixture(
    tmp_path: Path,
    *,
    candidate_samples: list[float],
    truncation_matrix: list[list[int]],
    seat_results: list[tuple[int, int]],
) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300"]
    candidate_mean = sum(candidate_samples) / len(candidate_samples) if candidate_samples else None
    candidate_ci_low = min(candidate_samples) if candidate_samples else None
    candidate_ci_high = max(candidate_samples) if candidate_samples else None
    matchups = [
        (0, 0, policies[0], policies[0]),
        (0, 1, policies[0], policies[1]),
        (1, 0, policies[1], policies[0]),
        (1, 1, policies[1], policies[1]),
    ]

    for (focal_index, opponent_index, focal_policy_id, opponent_policy_id), (seat0_wins, seat1_wins) in zip(
        matchups,
        seat_results,
        strict=True,
    ):
        diagnostics_path = (
            final_eval_dir
            / "matchups"
            / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
            / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
            / "diagnostics.json"
        )
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        decisive_games = seat0_wins + seat1_wins
        diagnostics_path.write_text(
            json.dumps(
                {
                    "seat_results": {
                        "seat0_wins": seat0_wins,
                        "seat1_wins": seat1_wins,
                        "draws": 0,
                        "truncations": 0,
                        "engine_errors": 0,
                        "decisive_games": decisive_games,
                        "total_games": decisive_games,
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary_payload = {
        "policy_ids": policies,
        "metadata": {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {
                "policy_ids": policies,
                "values": [[2, 2], [2, 2]],
            },
            "truncations": {
                "policy_ids": policies,
                "values": truncation_matrix,
            },
            "mean": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_mean, 0.5]],
            },
            "ci_low": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_ci_low, 0.5]],
            },
            "ci_high": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_ci_high, 0.5]],
            },
            "has_payoff_samples": {
                "policy_ids": policies,
                "values": [[True, True], [bool(candidate_samples), True]],
            },
            "paired_seed_count": {
                "policy_ids": policies,
                "values": [[1, 1], [2 if candidate_samples else 0, 1]],
            },
            "stop_reason": {
                "policy_ids": policies,
                "values": [["precision", "precision"], ["precision", "precision"]],
            },
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": len(candidate_samples),
            "values": [[[], []], [candidate_samples, []]],
        },
        "matchups": [
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "diagnostics_path": (
                    f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__/"
                    f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                ),
            }
            for focal_index, opponent_index, focal_policy_id, opponent_policy_id in matchups
        ],
    }
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_eval_dir


def test_build_paper_readiness_summary_passes_with_balanced_seats_and_strong_b0_winrate(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        truncation_matrix=[[0, 0], [0, 0]],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is True
    assert payload["alarms"] == []
    assert payload["checks"]["truncation_rate"]["rate"] == 0.0
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_flags_truncation_seat_bias_and_weak_b0_matchup(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.49, 0.5, 0.52, 0.54],
        truncation_matrix=[[0, 1], [0, 0]],
        seat_results=[(2, 0), (2, 0), (2, 0), (2, 0)],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is False
    assert payload["alarms"] == [
        "truncation_rate",
        "seat_bias_alarm",
        "baseline_win_rate_vs_b0",
    ]
    assert payload["checks"]["truncation_rate"]["rate"] == 0.125
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["passed"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 0.0
