from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.main_league_frontier_audit import (
    MainLeagueFrontierAuditConfig,
    build_main_league_frontier_audit,
    write_main_league_frontier_audit_markdown,
)


def test_frontier_audit_keeps_selected_locked_when_scorecards_stop(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "main_league_frontier_scorecard_candidate_20260521.json",
        {
            "counts": {"stop": 1, "total": 1},
            "entries": [
                {
                    "candidate_label": "candidate_u1_confirm64",
                    "baseline_label": "selected",
                    "panel_kind": "full",
                    "paired_seeds": 64,
                    "escalation": {"decision": "stop", "reason": "full_gate_failed"},
                    "group_deltas": {
                        "fixed_delta_wins": 2,
                        "learned_delta_wins": 0,
                        "all_delta_wins": 2,
                    },
                    "full_gate": {
                        "passed": False,
                        "failures": [{"reason": "full_learned_aggregate_drop"}],
                    },
                    "sentinel_gate": {"passed": True, "failures": []},
                }
            ],
        },
    )
    _write_json(
        diagnostics / "main_league_fast_loop_gate_candidate_20260521.json",
        {
            "kind": "main_league_fast_loop_gate_v1",
            "candidate_label": "candidate_u1_confirm64",
            "failures": [{"reason": "wrong_escalation_decision"}],
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["decision"]["publishable_successor_exists"] is False
    assert report["decision"]["selected_remains_locked"] is True
    assert report["counts"]["candidates_by_next_stage"] == {"stop": 1}
    record = report["candidate_records"][0]
    assert record["candidate_label"] == "candidate_u1"
    assert record["best_scorecard"]["learned_delta_wins"] == 0
    assert record["failed_gate_count"] == 1


def test_frontier_audit_keeps_confirm256_ready_unpublished(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "main_league_frontier_scorecard_candidate_20260521.json",
        {
            "entries": [
                {
                    "candidate_label": "candidate_confirm128",
                    "panel_kind": "full",
                    "paired_seeds": 128,
                    "escalation": {"decision": "run_confirm256", "reason": "confirm128_gate_passed"},
                    "group_deltas": {
                        "fixed_delta_wins": 0,
                        "learned_delta_wins": 3,
                        "all_delta_wins": 3,
                    },
                }
            ]
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["decision"]["publishable_successor_exists"] is False
    assert report["decision"]["selected_remains_locked"] is True
    assert report["counts"]["candidates_by_next_stage"] == {"confirm256": 1}


def test_frontier_audit_collapses_sentinel_family_after_full_confirm_stop(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "main_league_frontier_scorecard_candidate_sentinel_20260521.json",
        {
            "entries": [
                {
                    "candidate_label": "candidate_u1_sentinel16",
                    "panel_kind": "sentinel",
                    "paired_seeds": 16,
                    "escalation": {"decision": "run_full_confirm64", "reason": "sentinel_gate_passed"},
                    "group_deltas": {
                        "fixed_delta_wins": 0,
                        "learned_delta_wins": 1,
                        "all_delta_wins": 1,
                    },
                }
            ]
        },
    )
    _write_json(
        diagnostics / "main_league_frontier_scorecard_candidate_confirm64_20260521.json",
        {
            "entries": [
                {
                    "candidate_label": "candidate_u1_confirm64",
                    "panel_kind": "full",
                    "paired_seeds": 64,
                    "escalation": {"decision": "stop", "reason": "full_gate_failed"},
                    "group_deltas": {
                        "fixed_delta_wins": 2,
                        "learned_delta_wins": 0,
                        "all_delta_wins": 2,
                    },
                }
            ]
        },
    )
    _write_json(
        diagnostics / "main_league_fast_loop_gate_candidate_u1_full_confirm64_blocked_20260521.json",
        {
            "kind": "main_league_fast_loop_gate_v1",
            "candidate_label": "candidate_u1_confirm128_blocked",
            "passed": False,
            "failures": [{"reason": "wrong_escalation_decision"}],
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["counts"]["candidates_by_next_stage"] == {"stop": 1}
    record = report["candidate_records"][0]
    assert record["candidate_label"] == "candidate_u1"
    assert record["evidence_labels"] == [
        "candidate_u1_confirm128_blocked",
        "candidate_u1_confirm64",
        "candidate_u1_sentinel16",
    ]


def test_frontier_audit_marks_gate_only_survivor_as_sentinel(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "main_league_fast_loop_gate_candidate_u1_sentinel_20260521.json",
        {
            "kind": "main_league_fast_loop_gate_v1",
            "candidate_label": "candidate_u1_20260521",
            "passed": True,
            "failures": [],
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["counts"]["candidates_by_next_stage"] == {"sentinel": 1}
    assert report["candidate_records"][0]["candidate_label"] == "candidate_u1"


def test_frontier_audit_collapses_short_gate_aliases(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "paired_outcome_preference_gate_a050_spanfilter_actionlabel_pref_lowret_u1_20260521.json",
        {
            "kind": "paired_outcome_preference_mechanistic_gate_v1",
            "passed": True,
            "failures": [],
        },
    )
    _write_json(
        diagnostics / "trajectory_policy_drift_gate_a050_spanfilter_actionlabel_pref_lowret_u1_20260521.json",
        {
            "kind": "trajectory_policy_drift_gate_v1",
            "candidate_label": "spanpref_lowret_u1",
            "passed": False,
            "failures": ["lost_target_top_action_rate_above:0.1>0"],
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["counts"]["candidates_by_next_stage"] == {"stop": 1}
    record = report["candidate_records"][0]
    assert record["candidate_label"] == "spanpref_lowret_u1"
    assert record["passed_gate_count"] == 1
    assert record["failed_gate_count"] == 1


def test_frontier_audit_treats_paired_compare_gate_as_gate(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "paired_outcome_compare_sentinel16_pairw9x2_e2_vs_selected_20260521.json",
        {
            "kind": "paired_targeted_outcome_compare_v1",
            "candidate": {"label": "pairw9x2_e2"},
            "baseline": {"label": "selected"},
            "groups": {
                "all_compared": {"delta_wins": 0, "shared_games": 224},
                "fixed_baselines": {"delta_wins": 0, "shared_games": 64},
                "learned_opponents": {"delta_wins": 0, "shared_games": 160},
            },
        },
    )
    _write_json(
        diagnostics / "paired_outcome_compare_gate_sentinel16_pairw9x2_e2_improvement_vs_selected_20260521.json",
        {
            "kind": "paired_outcome_compare_gate_v1",
            "passed": False,
            "failures": [{"reason": "learned_aggregate_drop", "delta_wins": 0, "threshold": 1}],
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["counts"]["candidates_by_next_stage"] == {"stop": 1}
    record = report["candidate_records"][0]
    assert record["candidate_label"] == "pairw9x2_e2"
    assert record["compare_count"] == 1
    assert record["failed_gate_count"] == 1


def test_frontier_audit_stops_compare_only_full_depth_without_learned_gain(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _write_json(
        diagnostics / "paired_outcome_compare_selected_vs_candidate_confirm128_20260521.json",
        {
            "kind": "paired_targeted_outcome_compare_v1",
            "candidate": {"label": "candidate_confirm128"},
            "baseline": {"label": "selected"},
            "groups": {
                "all_compared": {"delta_wins": 3, "shared_games": 3328},
                "fixed_baselines": {"delta_wins": 3, "shared_games": 1280},
                "learned_opponents": {"delta_wins": 0, "shared_games": 2048},
            },
        },
    )

    report = build_main_league_frontier_audit(
        MainLeagueFrontierAuditConfig(diagnostics_dir=diagnostics, date_token="20260521")
    )

    assert report["counts"]["candidates_by_next_stage"] == {"stop": 1}
    record = report["candidate_records"][0]
    assert record["candidate_label"] == "candidate"
    assert record["compare_count"] == 1


def test_frontier_audit_writes_markdown(tmp_path: Path) -> None:
    report = {
        "selected_run": "runs/selected",
        "selected_policy_id": "main_selected",
        "candidate_count": 1,
        "scorecard_entry_count": 1,
        "gate_entry_count": 0,
        "decision": {"publishable_successor_exists": False, "reason": "no_confirm256_publishable_successor"},
        "best_non_publishable_signals": [
            {
                "candidate_label": "candidate|one",
                "panel_kind": "full",
                "paired_seeds": 64,
                "fixed_delta_wins": 2,
                "learned_delta_wins": 0,
                "decision": "stop",
                "reason": "full_gate_failed",
            }
        ],
    }
    output = tmp_path / "audit.md"

    write_main_league_frontier_audit_markdown(output, report)

    text = output.read_text(encoding="utf-8")
    assert "Main League Frontier Audit" in text
    assert "candidate\\|one" in text


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
