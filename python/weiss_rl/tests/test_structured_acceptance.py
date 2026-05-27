from __future__ import annotations

from pathlib import Path

from weiss_rl.experiments.structured_acceptance import build_structured_baseline_contract


def test_build_structured_baseline_contract_extracts_counts_and_targets() -> None:
    contract = build_structured_baseline_contract(
        baseline_run_dir=Path("/tmp/baseline"),
        baseline_update=300,
        dev_eval_summary={
            "aggregate_score": 0.375,
            "anchor_scores": {
                "B0 RandomLegal": 0.6875,
                "B1 NoLeague baseline": 0.4375,
                "B2 HeuristicPublic": 0.0,
            },
        },
        audit_summary_path=Path("/tmp/audit/summary.json"),
        audit_summary={
            "top_family_pairs": [
                {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 101},
                {"policy_a_family": "main_move", "policy_b_family": "main_play_character", "count": 98},
            ],
            "top_action_label_pairs": [
                {
                    "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                    "policy_b_action_label": "pass",
                    "count": 78,
                },
                {
                    "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                    "policy_b_action_label": "main_play_character(hand_index=0, stage_slot=0)",
                    "count": 41,
                },
            ],
        },
    )

    payload = contract.to_dict()
    assert payload["mismatch_baseline"] == {
        "main_move_to_pass": 101,
        "main_move_to_main_play_character": 98,
        "exact_main_move_0_2_to_pass": 78,
    }
    assert payload["dominant_exact_pair"] == {
        "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
        "policy_b_action_label": "pass",
        "count": 78,
    }
    assert payload["acceptance_targets"]["u120"]["max_main_move_to_pass"] == 40
    assert payload["acceptance_targets"]["u120"]["max_main_move_to_main_play_character"] == 39
    assert payload["acceptance_targets"]["u300"]["min_aggregate_score"] == 0.475
    assert payload["acceptance_targets"]["u300"]["max_anchor_regressions"] == {
        "B0 RandomLegal": 0.6375,
        "B1 NoLeague baseline": 0.3875,
    }


def test_build_structured_baseline_contract_requires_b2_anchor() -> None:
    try:
        build_structured_baseline_contract(
            baseline_run_dir=Path("/tmp/baseline"),
            baseline_update=300,
            dev_eval_summary={"aggregate_score": 0.1, "anchor_scores": {"B0 RandomLegal": 0.6}},
            audit_summary_path=Path("/tmp/audit/summary.json"),
            audit_summary={},
        )
    except ValueError as exc:
        assert "B2 HeuristicPublic" in str(exc)
    else:
        raise AssertionError("expected missing B2 anchor to raise")
