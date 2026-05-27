from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.paired_outcome_overlap_report import (
    PairedOutcomeOverlapReportConfig,
    build_paired_outcome_overlap_report,
)


def test_overlap_report_detects_fixed_loss_and_learned_gain_same_pair(tmp_path: Path) -> None:
    compare_json = tmp_path / "compare.json"
    _write_compare(
        compare_json,
        rows=[
            _row(
                "B2 HeuristicPublic",
                delta_wins=-1,
                examples=[
                    _example(
                        pair_index=16,
                        swap_index=0,
                        episode_seed=42,
                        baseline_outcome="W",
                        candidate_outcome="L",
                    )
                ],
            ),
            _row(
                "seed_policy_000004",
                delta_wins=1,
                examples=[
                    _example(
                        pair_index=16,
                        swap_index=0,
                        episode_seed=42,
                        baseline_outcome="L",
                        candidate_outcome="W",
                    )
                ],
            ),
        ],
    )

    report = build_paired_outcome_overlap_report(PairedOutcomeOverlapReportConfig(compare_json_paths=(compare_json,)))

    entry = report["reports"][0]
    assert entry["conflict_key_count"] == 1
    conflict = entry["conflict_keys"][0]
    assert conflict["pair_index"] == 16
    assert conflict["swap_index"] == 0
    assert conflict["episode_seed"] == 42
    assert conflict["conflict_types"] == ["fixed_loss_and_learned_gain"]
    assert conflict["counts"]["fixed_candidate_loss"] == 1
    assert conflict["counts"]["learned_candidate_gain"] == 1


def test_overlap_report_marks_truncated_compare_rows(tmp_path: Path) -> None:
    compare_json = tmp_path / "compare.json"
    _write_compare(
        compare_json,
        rows=[
            _row(
                "B2 HeuristicPublic",
                delta_wins=-3,
                changed_outcome=3,
                examples=[
                    _example(
                        pair_index=1,
                        swap_index=0,
                        episode_seed=10,
                        baseline_outcome="W",
                        candidate_outcome="L",
                    )
                ],
            )
        ],
    )

    report = build_paired_outcome_overlap_report(PairedOutcomeOverlapReportConfig(compare_json_paths=(compare_json,)))

    entry = report["reports"][0]
    assert entry["truncated_row_count"] == 1
    assert entry["truncated_rows"][0]["opponent_policy_id"] == "B2 HeuristicPublic"
    assert entry["truncated_rows"][0]["missing_examples"] == 2


def _write_compare(path: Path, *, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "paired_outcome_compare_v1",
                "baseline": {"label": "selected"},
                "candidate": {"label": "candidate"},
                "fixed_opponents": ["B2 HeuristicPublic"],
                "learned_opponents": ["seed_policy_000004"],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def _row(
    opponent: str,
    *,
    delta_wins: int,
    examples: list[dict],
    changed_outcome: int | None = None,
) -> dict:
    return {
        "status": "ok",
        "opponent_policy_id": opponent,
        "delta_wins": delta_wins,
        "changed_outcome": len(examples) if changed_outcome is None else changed_outcome,
        "examples": examples,
    }


def _example(
    *,
    pair_index: int,
    swap_index: int,
    episode_seed: int,
    baseline_outcome: str,
    candidate_outcome: str,
) -> dict:
    return {
        "pair_index": pair_index,
        "swap_index": swap_index,
        "episode_seed": episode_seed,
        "baseline_outcome": baseline_outcome,
        "candidate_outcome": candidate_outcome,
        "baseline_decision_count": 10,
        "candidate_decision_count": 12,
        "baseline_pass_actions": 3,
        "candidate_pass_actions": 4,
        "baseline_pass_with_nonpass_available": 1,
        "candidate_pass_with_nonpass_available": 2,
    }
