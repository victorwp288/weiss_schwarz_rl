from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID


def test_build_paper_readiness_summary_ignores_baseline_aliases_and_heuristic_variants(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    policies = [
        RANDOM_LEGAL_POLICY_ID,
        "b1_noleague_baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
        "policy_000300",
    ]
    matchups: list[dict[str, Any]] = []
    values: list[list[float]] = []
    ci_low: list[list[float]] = []
    ci_high: list[list[float]] = []
    games: list[list[int]] = []
    truncations: list[list[int]] = []
    for focal_index, focal_policy_id in enumerate(policies):
        row_values: list[float] = []
        row_ci_low: list[float] = []
        row_ci_high: list[float] = []
        row_games: list[int] = []
        row_truncations: list[int] = []
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
                            "seat0_wins": 6,
                            "seat1_wins": 6,
                            "draws": 0,
                            "truncations": 0,
                            "engine_errors": 0,
                            "decisive_games": 12,
                            "total_games": 12,
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
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"/{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                }
            )
            if focal_policy_id == "policy_000300" and opponent_policy_id == RANDOM_LEGAL_POLICY_ID:
                row_values.append(0.9)
                row_ci_low.append(0.82)
                row_ci_high.append(0.96)
            elif focal_policy_id == RANDOM_LEGAL_POLICY_ID and opponent_policy_id == "policy_000300":
                row_values.append(0.1)
                row_ci_low.append(0.04)
                row_ci_high.append(0.18)
            else:
                row_values.append(0.5)
                row_ci_low.append(0.5)
                row_ci_high.append(0.5)
            row_games.append(12)
            row_truncations.append(0)
        values.append(row_values)
        ci_low.append(row_ci_low)
        ci_high.append(row_ci_high)
        games.append(row_games)
        truncations.append(row_truncations)

    summary_payload = {
        "policy_ids": policies,
        "metadata": {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {"policy_ids": policies, "values": games},
            "truncations": {"policy_ids": policies, "values": truncations},
            "mean": {"policy_ids": policies, "values": values},
            "ci_low": {"policy_ids": policies, "values": ci_low},
            "ci_high": {"policy_ids": policies, "values": ci_high},
            "has_payoff_samples": {"policy_ids": policies, "values": [[True] * len(policies) for _ in policies]},
            "paired_seed_count": {"policy_ids": policies, "values": [[12] * len(policies) for _ in policies]},
            "stop_reason": {"policy_ids": policies, "values": [["precision"] * len(policies) for _ in policies]},
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": 4,
            "values": [
                [
                    [0.5, 0.5, 0.5, 0.5]
                    if row == column
                    else ([0.1, 0.08, 0.12, 0.14] if row == 0 and column == len(policies) - 1 else [])
                    for column in range(len(policies))
                ]
                for row in range(len(policies))
            ],
        },
        "matchups": matchups,
    }
    posterior_samples = cast(dict[str, Any], summary_payload["posterior_samples"])
    posterior_values = cast(list[list[list[float]]], posterior_samples["values"])
    posterior_values[-1][0] = [0.9, 0.92, 0.88, 0.94]
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000300"
    assert check["focal_policy_source"] == "sole_eligible_non_baseline"
