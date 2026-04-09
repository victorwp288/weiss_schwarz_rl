from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.policy_set import RANDOM_LEGAL_POLICY_ID


SeatResultInput = tuple[int, int] | dict[str, int]


def _normalize_seat_result(value: SeatResultInput) -> dict[str, int]:
    if isinstance(value, tuple):
        seat0_wins, seat1_wins = value
        draws = 0
        truncations = 0
        engine_errors = 0
    else:
        seat0_wins = int(value.get("seat0_wins", 0))
        seat1_wins = int(value.get("seat1_wins", 0))
        draws = int(value.get("draws", 0))
        truncations = int(value.get("truncations", 0))
        engine_errors = int(value.get("engine_errors", 0))
    decisive_games = seat0_wins + seat1_wins
    return {
        "seat0_wins": seat0_wins,
        "seat1_wins": seat1_wins,
        "draws": draws,
        "truncations": truncations,
        "engine_errors": engine_errors,
        "decisive_games": decisive_games,
        "total_games": decisive_games + draws + truncations,
    }


def _write_final_eval_fixture(
    tmp_path: Path,
    *,
    candidate_samples: list[float],
    seat_results: list[SeatResultInput],
    games_matrix: list[list[int]] | None = None,
    truncation_matrix: list[list[int]] | None = None,
    metadata: dict[str, Any] | None = None,
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
    normalized_seat_results = [_normalize_seat_result(value) for value in seat_results]

    for (focal_index, opponent_index, focal_policy_id, opponent_policy_id), seat_result in zip(
        matchups,
        normalized_seat_results,
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
        diagnostics_path.write_text(
            json.dumps(
                {"seat_results": seat_result},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary_payload = {
        "policy_ids": policies,
        "metadata": metadata or {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {
                "policy_ids": policies,
                "values": games_matrix
                or [
                    [normalized_seat_results[0]["total_games"], normalized_seat_results[1]["total_games"]],
                    [normalized_seat_results[2]["total_games"], normalized_seat_results[3]["total_games"]],
                ],
            },
            "truncations": {
                "policy_ids": policies,
                "values": truncation_matrix
                or [
                    [normalized_seat_results[0]["truncations"], normalized_seat_results[1]["truncations"]],
                    [normalized_seat_results[2]["truncations"], normalized_seat_results[3]["truncations"]],
                ],
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
                "focal_policy_index": focal_index,
                "opponent_policy_index": opponent_index,
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


def _write_multi_policy_final_eval_fixture(tmp_path: Path, *, metadata: dict[str, Any] | None = None) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300", "policy_000400"]
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    matchups: list[dict[str, Any]] = []
    for focal_index, focal_policy_id in enumerate(policies):
        for opponent_index, opponent_policy_id in enumerate(policies):
            diagnostics_path = (
                final_eval_dir
                / "matchups"
                / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
                / "diagnostics.json"
            )
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "seat_results": {
                            "seat0_wins": 1,
                            "seat1_wins": 1,
                            "draws": 0,
                            "truncations": 0,
                            "engine_errors": 0,
                            "decisive_games": 2,
                            "total_games": 2,
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            matchups.append(
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__/"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                }
            )

    values_float: list[list[float | None]] = [
        [0.5, 0.0, 0.0],
        [0.91, 0.5, 0.48],
        [0.94, 0.52, 0.5],
    ]
    summary_payload = {
        "policy_ids": policies,
        "metadata": metadata or {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {"policy_ids": policies, "values": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]},
            "truncations": {"policy_ids": policies, "values": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            "mean": {"policy_ids": policies, "values": values_float},
            "ci_low": {"policy_ids": policies, "values": [[0.5, 0.0, 0.0], [0.88, 0.5, 0.45], [0.9, 0.5, 0.5]]},
            "ci_high": {"policy_ids": policies, "values": [[0.5, 0.0, 0.0], [0.95, 0.5, 0.51], [0.97, 0.55, 0.5]]},
            "has_payoff_samples": {
                "policy_ids": policies,
                "values": [[True, True, True], [True, True, True], [True, True, True]],
            },
            "paired_seed_count": {"policy_ids": policies, "values": [[1, 1, 1], [2, 1, 1], [2, 1, 1]]},
            "stop_reason": {
                "policy_ids": policies,
                "values": [
                    ["precision", "precision", "precision"],
                    ["precision", "precision", "precision"],
                    ["precision", "precision", "precision"],
                ],
            },
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": 4,
            "values": [
                [[], [], []],
                [[0.88, 0.9, 0.92, 0.95], [], [0.45, 0.48, 0.5, 0.51]],
                [[0.9, 0.93, 0.95, 0.97], [0.5, 0.51, 0.53, 0.55], []],
            ],
        },
        "matchups": matchups,
    }
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_eval_dir


def test_build_paper_readiness_summary_passes_with_balanced_seats_and_strong_b0_winrate(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(
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
    final_eval_dir = _write_final_eval_fixture(
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
    baseline_dir = _write_final_eval_fixture(
        tmp_path / "baseline",
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )
    reciprocal_noise_dir = _write_final_eval_fixture(
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
    final_eval_dir = _write_multi_policy_final_eval_fixture(tmp_path)

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is False
    assert payload["alarms"] == ["baseline_win_rate_vs_b0"]
    assert check["focal_policy_id"] is None
    assert check["reason"] == "ambiguous_non_baseline_focal_policy"
    assert check["eligible_non_baseline_policy_ids"] == ["policy_000300", "policy_000400"]
    assert "pass --focal-policy-id" in check["message"]



def test_build_paper_readiness_summary_uses_metadata_named_focal_policy(tmp_path: Path) -> None:
    final_eval_dir = _write_multi_policy_final_eval_fixture(
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
