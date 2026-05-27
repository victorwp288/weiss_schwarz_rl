from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.main_league_multiobjective_gate import (
    FIXED_THESIS_OPPONENTS,
    MultiObjectiveGateConfig,
    evaluate_main_league_multiobjective_gate,
)


def _write_summary(path: Path, rows: list[tuple[str, int, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "opponent_policy_id": opponent,
                        "wins": wins,
                        "games": games,
                        "mean": wins / games,
                        "summary_path": f"matchups/{opponent}/matchup_summary.json",
                    }
                    for opponent, wins, games in rows
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_multiobjective_gate_passes_fixed_and_improved_learned_panel(tmp_path: Path) -> None:
    learned = ("seed_a", "seed_b")
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 64, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_a", 58, 100),
            ("seed_b", 62, 100),
        ],
    )
    reference = _write_summary(
        tmp_path / "reference.json",
        [
            *[(opponent, 62, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_a", 57, 100),
            ("seed_b", 60, 100),
        ],
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=learned,
            min_fixed_score=0.5,
            max_fixed_reference_drop=0.0,
            min_learned_score=0.5,
            min_learned_mean=0.5,
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is True
    assert gate["failures"] == []
    assert gate["groups"]["fixed_baselines"]["mean"] == pytest.approx(0.64)
    assert gate["groups"]["learned_opponents"]["mean"] == pytest.approx(0.60)
    assert gate["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(0.015)


def test_multiobjective_gate_rejects_fixed_drop_and_learned_panel_regression(tmp_path: Path) -> None:
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 64, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("B4 HeuristicPublicControl", 59, 100),
            ("seed_a", 54, 100),
            ("seed_b", 55, 100),
        ],
    )
    reference = _write_summary(
        tmp_path / "reference.json",
        [
            *[(opponent, 64, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("B4 HeuristicPublicControl", 64, 100),
            ("seed_a", 57, 100),
            ("seed_b", 58, 100),
        ],
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=("seed_a", "seed_b"),
            max_fixed_reference_drop=0.0,
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is False
    reasons = {(failure.get("group"), failure.get("opponent"), failure.get("reason")) for failure in gate["failures"]}
    assert ("fixed_baselines", "B4 HeuristicPublicControl", "below_reference_drop_limit") in reasons
    assert ("learned_opponents", None, "below_min_group_reference_delta") in reasons


def test_multiobjective_gate_rejects_aggregate_learned_regression_despite_fixed_gains(tmp_path: Path) -> None:
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 66, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_a", 54, 100),
            ("seed_b", 55, 100),
        ],
    )
    reference = _write_summary(
        tmp_path / "reference.json",
        [
            *[(opponent, 64, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_a", 57, 100),
            ("seed_b", 58, 100),
        ],
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=("seed_a", "seed_b"),
            max_fixed_reference_drop=0.0,
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is False
    assert gate["groups"]["fixed_baselines"]["mean"] == pytest.approx(0.66)
    assert gate["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(-0.03)
    assert len(gate["failures"]) == 1
    failure = gate["failures"][0]
    assert failure["group"] == "learned_opponents"
    assert failure["reason"] == "below_min_group_reference_delta"
    assert failure["delta"] == pytest.approx(-0.03)
    assert failure["threshold"] == pytest.approx(0.0)


def test_multiobjective_gate_aliases_reference_opponent_ids(tmp_path: Path) -> None:
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 70, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("candidate_seed_a", 56, 100),
        ],
    )
    reference = _write_summary(
        tmp_path / "reference.json",
        [
            ("source_seed_a", 55, 100),
        ],
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=("candidate_seed_a",),
            opponent_aliases={"source_seed_a": "candidate_seed_a"},
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is True
    row = gate["groups"]["learned_opponents"]["rows"][0]
    assert row["reference"] == pytest.approx(0.55)
    assert row["delta"] == pytest.approx(0.01)


def test_multiobjective_gate_resolves_seed_wrapped_suffix_reference_ids(tmp_path: Path) -> None:
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 70, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003", 56, 100),
        ],
    )
    reference = _write_summary(
        tmp_path / "reference.json",
        [
            *[(opponent, 70, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("policy_000003", 55, 100),
        ],
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=("seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",),
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is True
    row = gate["groups"]["learned_opponents"]["rows"][0]
    assert row["candidate_opponent_policy_id"] == "seed_b8c698d26a_seed_c3aac2f9dc_policy_000003"
    assert row["reference_opponent_policy_id"] == "policy_000003"
    assert row["delta"] == pytest.approx(0.01)


def test_multiobjective_gate_loads_simple_anchor_score_reference(tmp_path: Path) -> None:
    candidate = _write_summary(
        tmp_path / "candidate.json",
        [
            *[(opponent, 70, 100) for opponent in FIXED_THESIS_OPPONENTS],
            ("seed_a", 56, 100),
        ],
    )
    reference = tmp_path / "reference_scores.json"
    reference.write_text(
        json.dumps({"anchor_scores": {"seed_a": 55 / 100}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    gate = evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(candidate,),
            reference_summary_jsons=(reference,),
            learned_opponents=("seed_a",),
            min_learned_reference_delta=0.0,
        )
    )

    assert gate["passed"] is True
    assert gate["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(0.01)
