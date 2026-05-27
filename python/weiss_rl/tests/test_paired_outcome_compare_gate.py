from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.paired_outcome_compare_gate import (
    PairedOutcomeCompareGateConfig,
    evaluate_paired_outcome_compare_gate,
)


def test_paired_outcome_compare_gate_passes_no_regression_screen(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=0,
        learned_delta=1,
        rows=[
            ("B2 HeuristicPublic", "fixed", 0),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", "learned", 1),
        ],
    )

    report = evaluate_paired_outcome_compare_gate(
        PairedOutcomeCompareGateConfig(
            compare_jsons=(compare,),
            required_opponents=("B2 HeuristicPublic", "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004"),
        )
    )

    assert report["passed"] is True
    assert report["entries"][0]["group_deltas"]["learned_delta_wins"] == 1


def test_paired_outcome_compare_gate_fails_fixed_and_learned_row_drops(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=-3,
        learned_delta=-1,
        rows=[
            ("B2 HeuristicPublic", "fixed", -3),
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", "learned", -1),
        ],
    )

    report = evaluate_paired_outcome_compare_gate(PairedOutcomeCompareGateConfig(compare_jsons=(compare,)))

    reasons = [failure["reason"] for failure in report["failures"]]
    assert report["passed"] is False
    assert "fixed_aggregate_drop" in reasons
    assert "learned_aggregate_drop" in reasons
    assert "fixed_row_drop" in reasons
    assert "learned_row_drop" in reasons


def test_paired_outcome_compare_gate_requires_expected_opponents(tmp_path: Path) -> None:
    compare = _write_compare(
        tmp_path / "compare.json",
        fixed_delta=0,
        learned_delta=0,
        rows=[("B2 HeuristicPublic", "fixed", 0)],
    )

    report = evaluate_paired_outcome_compare_gate(
        PairedOutcomeCompareGateConfig(
            compare_jsons=(compare,),
            required_opponents=("B2 HeuristicPublic", "policy_000004"),
        )
    )

    assert report["passed"] is False
    assert any(failure["reason"] == "missing_required_opponent" for failure in report["failures"])


def _write_compare(
    path: Path,
    *,
    fixed_delta: int,
    learned_delta: int,
    rows: list[tuple[str, str, int]],
) -> Path:
    fixed_opponents = [opponent for opponent, group, _delta in rows if group == "fixed"]
    learned_opponents = [opponent for opponent, group, _delta in rows if group == "learned"]
    payload_rows = [
        {
            "baseline_opponent_policy_id": opponent,
            "candidate_opponent_policy_id": opponent,
            "delta_wins": delta,
            "baseline_wins": 8,
            "candidate_wins": 8 + delta,
            "shared_games": 16,
        }
        for opponent, _group, delta in rows
    ]
    path.write_text(
        json.dumps(
            {
                "kind": "paired_targeted_outcome_compare_v1",
                "baseline": {"label": "selected"},
                "candidate": {"label": "candidate"},
                "fixed_opponents": fixed_opponents,
                "learned_opponents": learned_opponents,
                "groups": {
                    "all_compared": {"delta_wins": fixed_delta + learned_delta},
                    "fixed_baselines": {"delta_wins": fixed_delta},
                    "learned_opponents": {"delta_wins": learned_delta},
                },
                "rows": payload_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
