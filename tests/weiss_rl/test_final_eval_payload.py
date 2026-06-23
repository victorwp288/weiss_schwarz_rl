from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import final_eval as final_eval_module
from weiss_rl.eval.final.payload import build_final_eval_payload


def test_final_eval_payload_builder_preserves_metadata_paths_and_reverse_cells(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    seed_file = tmp_path / "report_eval_seeds.txt"
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    policy_ids = ["policy_a", "policy_b"]

    def result(
        *,
        focal_index: int,
        opponent_index: int,
        mean: float,
        posterior_samples: tuple[float, ...],
    ) -> dict[str, Any]:
        matchup_dir = output_dir / "matchups" / f"{focal_index:02d}_vs_{opponent_index:02d}"
        return {
            "focal_policy_id": policy_ids[focal_index],
            "opponent_policy_id": policy_ids[opponent_index],
            "focal_index": focal_index,
            "opponent_index": opponent_index,
            "matchup_dir": matchup_dir,
            "episodes_path": matchup_dir / "episodes.jsonl",
            "posterior_samples": posterior_samples,
            "summary": {
                "paired_seeds": 3,
                "observed_paired_seeds": 3,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "precision",
                "should_stop": True,
                "summary": {
                    "games": 6,
                    "wins": int(round(mean * 6)),
                    "losses": 6 - int(round(mean * 6)),
                    "draws": 0,
                    "truncations": 0,
                    "engine_errors": 0,
                },
                "uncertainty": {
                    "mean": mean,
                    "ci_low": max(0.0, mean - 0.1),
                    "ci_high": min(1.0, mean + 0.1),
                    "ci_half_width": 0.1,
                    "prob_gt_half": 0.8,
                    "prob_lt_half": 0.2,
                    "paired_seed_count": 3,
                },
            },
        }

    payload = build_final_eval_payload(
        output_dir=output_dir,
        policy_ids=policy_ids,
        matchup_results=[
            result(focal_index=0, opponent_index=0, mean=0.5, posterior_samples=(0.5,)),
            result(focal_index=0, opponent_index=1, mean=0.75, posterior_samples=(0.2, 0.8)),
            result(focal_index=1, opponent_index=1, mean=0.5, posterior_samples=(0.5,)),
        ],
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        paired_seeds=[11, 22, 33],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        scheme="S0",
        sample_count=16,
        selection_payload={"mode": "explicit", "policy_count": 2},
        metadata={"paper_tag": "final_results_v1"},
        seed_file_path=seed_file,
    )

    assert final_eval_module._build_final_eval_payload is build_final_eval_payload
    assert payload["metadata"]["paper_tag"] == "final_results_v1"
    assert payload["metadata"]["seed_file"]["path"] == seed_file.as_posix()
    assert payload["metadata"]["seed_file"]["sha256"]
    assert payload["matrices"]["mean"]["values"] == [[0.5, 0.75], [0.25, 0.5]]
    assert payload["matrices"]["wins"]["values"] == [[3, 4], [2, 3]]
    assert payload["posterior_samples"]["values"] == [[[0.5], [0.2, 0.8]], [[0.8, 0.19999999999999996], [0.5]]]
    assert payload["matchups"][1]["matchup_dir"] == "matchups/00_vs_01"
    assert payload["matchups"][1]["summary_path"] == "matchups/00_vs_01/matchup_summary.json"
    assert payload["matchups"][1]["matrix_cells"] == [
        {"focal_policy_index": 0, "opponent_policy_index": 1},
        {"focal_policy_index": 1, "opponent_policy_index": 0},
    ]
