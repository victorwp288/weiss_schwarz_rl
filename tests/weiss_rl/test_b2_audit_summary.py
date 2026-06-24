from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weiss_rl.diagnostics.b2_audit import b2_audit_aggregation as aggregation
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import b2_audit_plan_payload

from .b2_disagreement_audit_test_support import _numeric_summary, weighted_audit_bundle_summaries


def test_aggregate_audit_summary_ranks_repeated_family_pairs_and_weighted_means(tmp_path: Path) -> None:
    source = SimpleNamespace(
        focal_policy_id="learner",
        opponent_policy_id="B2 HeuristicPublic",
        config_hash256="a" * 64,
        spec_hash256="b" * 64,
        paired_seeds=(42, 7),
    )
    bundle_summaries = weighted_audit_bundle_summaries()

    summary = aggregation.aggregate_audit_summary(
        source=source,
        policy_id="learner",
        opponent_policy_id=source.opponent_policy_id,
        episodes_jsonl=tmp_path / "source.jsonl",
        run_dir=tmp_path / "source_run",
        output_run_dir=tmp_path / "output_run",
        episodes_path=tmp_path / "audit" / "episodes.jsonl",
        game_count=4,
        bundle_summaries=bundle_summaries,
        inspection_errors=[],
    )

    assert summary["status"] == "ok"
    assert summary["audit_plan"] == b2_audit_plan_payload()
    assert [step["step_id"] for step in summary["audit_plan"]] == [
        "reuse_source_seeds",
        "resolve_policies",
        "rerun_matchup",
        "inspect_replays",
        "aggregate_findings",
    ]
    assert summary["opponent_policy_id"] == "B2 HeuristicPublic"
    assert summary["games"] == 4
    assert summary["bundle_count"] == 2
    assert summary["top_family_pairs"][0] == {
        "policy_a_family": "attack",
        "policy_b_family": "pass",
        "count": 4,
    }
    assert summary["top_policy_a_families"][0] == {"family": "attack", "count": 6}
    assert summary["top_policy_b_families"][0] == {"family": "pass", "count": 6}
    assert summary["top_recorded_families"][0] == {"family": "attack", "count": 4}
    assert summary["top_action_label_pairs"][0] == {
        "policy_a_action_label": "attack(slot=0, attack_type=direct)",
        "policy_b_action_label": "pass",
        "count": 4,
    }
    assert summary["top_action_family_confusions"][:3] == [
        {"policy_b_family": "attack", "policy_a_family": "attack", "count": 12},
        {"policy_b_family": "pass", "policy_a_family": "main_move", "count": 12},
        {"policy_b_family": "pass", "policy_a_family": "attack", "count": 6},
    ]
    assert summary["top_policy_a_action_labels"][0] == {
        "action_label": "attack(slot=0, attack_type=direct)",
        "count": 6,
    }
    assert summary["top_policy_b_action_labels"][0] == {"action_label": "pass", "count": 6}
    assert summary["max_total_variation"] == pytest.approx(0.9)
    assert summary["mean_total_variation"] == pytest.approx((0.2 * 10 + 0.4 * 20) / 30)
    assert summary["policy_a_matches_policy_b_top_action_rate"] == pytest.approx((0.1 * 10 + 0.4 * 20) / 30)
    assert summary["policy_a_matches_policy_b_top_action_family_rate"] == pytest.approx((0.2 * 10 + 0.5 * 20) / 30)
    assert summary["policy_a_mean_probability_on_policy_b_top_action"] == pytest.approx((0.3 * 10 + 0.6 * 20) / 30)
    assert summary["policy_a_mean_probability_on_policy_b_top_action_family"] == pytest.approx(
        (0.4 * 10 + 0.7 * 20) / 30
    )
    assert summary["policy_a_weighted_mean_median_rank_of_policy_b_top_action"] == pytest.approx(
        (2.0 * 10 + 4.0 * 20) / 30
    )
    assert summary["policy_a_legal_surface_filter_rate"] == pytest.approx((0.7 * 10 + 0.2 * 20) / 30)
    assert summary["policy_b_legal_surface_filter_rate"] == pytest.approx((0.0 * 10 + 0.1 * 20) / 30)
    assert summary["policy_a_mean_raw_minus_policy_a_legal_action_count"] == pytest.approx((2.0 * 10 + 0.5 * 20) / 30)
    assert summary["policy_b_mean_raw_minus_policy_b_legal_action_count"] == pytest.approx((0.0 * 10 + 0.25 * 20) / 30)
    assert summary["policy_b_top_action_illegal_for_policy_a_rate"] == pytest.approx((0.6 * 10 + 0.15 * 20) / 30)
    assert summary["policy_a_top_action_illegal_for_policy_b_rate"] == pytest.approx((0.0 * 10 + 0.05 * 20) / 30)
    assert summary["policy_a_top_logit_margin_percentiles_bundle_weighted"] == {
        "aggregation": "weighted_mean_of_bundle_percentiles",
        "source_summary_key": "policy_a_top_logit_margin_percentiles",
        "count": 30,
        "mean": pytest.approx((0.2 * 10 + 0.5 * 20) / 30),
        "p10": pytest.approx((0.05 * 10 + 0.1 * 20) / 30),
        "p25": pytest.approx((0.1 * 10 + 0.3 * 20) / 30),
        "p50": pytest.approx((0.2 * 10 + 0.6 * 20) / 30),
        "p75": pytest.approx((0.3 * 10 + 0.8 * 20) / 30),
        "p90": pytest.approx((0.4 * 10 + 1.0 * 20) / 30),
    }
    assert summary["policy_a_probability_on_policy_b_top_action_percentiles_bundle_weighted"]["p50"] == pytest.approx(
        (0.3 * 10 + 0.65 * 20) / 30
    )
    assert summary["policy_a_top_probability_margin_percentiles_bundle_weighted"]["mean"] == pytest.approx(
        (0.05 * 10 + 0.12 * 20) / 30
    )
    assert summary["policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles_bundle_weighted"][
        "p90"
    ] == pytest.approx((1.5 * 10 + 0.7 * 20) / 30)
    assert summary["raw_legal_action_count_percentiles_bundle_weighted"]["mean"] == pytest.approx(
        (6.0 * 10 + 5.0 * 20) / 30
    )
    assert summary["policy_a_legal_action_count_percentiles_bundle_weighted"]["p50"] == pytest.approx(
        (4.0 * 10 + 4.0 * 20) / 30
    )
    assert summary["policy_a_policy_b_top_action_same_family_logit_margin_percentiles_bundle_weighted"]["count"] == 0
    assert summary["policy_b_top_family_summaries"][0]["family"] == "pass"
    assert summary["policy_b_top_family_summaries"][0]["count"] == 18
    assert summary["policy_b_top_family_summaries"][1]["family"] == "attack"
    assert summary["policy_b_top_family_summaries"][1]["count"] == 12
    assert summary["policy_b_top_family_summaries"][1]["policy_a_matches_policy_b_top_action_rate"] == pytest.approx(
        (0.25 * 4 + 0.5 * 8) / 12
    )
    assert summary["policy_b_top_family_summaries"][1][
        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles_bundle_weighted"
    ]["mean"] == pytest.approx((0.1 * 4 + 0.4 * 8) / 12)
    assert summary["policy_b_top_family_summaries"][1]["policy_b_top_action_legal_for_policy_a_rate"] == pytest.approx(
        (0.5 * 4 + 0.875 * 8) / 12
    )
    assert summary["policy_b_top_family_summaries"][1]["policy_a_legal_surface_filter_rate"] == pytest.approx(
        (0.25 * 4 + 0.125 * 8) / 12
    )
    assert summary["policy_a_mean_family_probability_masses"][0] == {
        "family": "pass",
        "mean_probability": pytest.approx((0.4 * 10 + 0.8 * 20) / 30),
    }
    assert summary["top_examples"][0]["example"] == "second"


def test_aggregate_trajectory_summary_merges_counts_and_focal_roles() -> None:
    bundle_summaries = [
        {
            "focal_seat": 0,
            "trajectory_summary": {
                "compared_steps": 2,
                "recorded_family_counts": [{"family": "pass", "count": 1}, {"family": "attack", "count": 1}],
                "phase_counts": [{"phase": "2", "count": 2}],
                "decision_kind_counts": [{"decision_kind": "3", "count": 2}],
                "legal_family_presence_rates": [{"family": "attack", "rate": 0.5}],
                "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=3.0)},
                "actor_summaries": [
                    {
                        "actor": 0,
                        "compared_steps": 1,
                        "recorded_family_counts": [{"family": "pass", "count": 1}],
                        "phase_counts": [{"phase": "2", "count": 1}],
                        "decision_kind_counts": [{"decision_kind": "3", "count": 1}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 0.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=1, mean=2.0)},
                    },
                    {
                        "actor": 1,
                        "compared_steps": 1,
                        "recorded_family_counts": [{"family": "attack", "count": 1}],
                        "phase_counts": [{"phase": "2", "count": 1}],
                        "decision_kind_counts": [{"decision_kind": "3", "count": 1}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=1, mean=5.0)},
                    },
                ],
            },
        },
        {
            "focal_seat": 1,
            "trajectory_summary": {
                "compared_steps": 2,
                "recorded_family_counts": [{"family": "clock_from_hand", "count": 2}],
                "phase_counts": [{"phase": "1", "count": 2}],
                "decision_kind_counts": [{"decision_kind": "4", "count": 2}],
                "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=4.0)},
                "actor_summaries": [
                    {
                        "actor": 1,
                        "compared_steps": 2,
                        "recorded_family_counts": [{"family": "clock_from_hand", "count": 2}],
                        "phase_counts": [{"phase": "1", "count": 2}],
                        "decision_kind_counts": [{"decision_kind": "4", "count": 2}],
                        "legal_family_presence_rates": [{"family": "attack", "rate": 1.0}],
                        "numeric_summaries": {"self_clock_count": _numeric_summary(count=2, mean=4.0)},
                    }
                ],
            },
        },
    ]

    summary = aggregation.aggregate_trajectory_summary(bundle_summaries)

    assert summary["compared_steps"] == 4
    assert summary["recorded_family_counts"] == [
        {"family": "clock_from_hand", "count": 2},
        {"family": "attack", "count": 1},
        {"family": "pass", "count": 1},
    ]
    assert summary["numeric_summaries"]["self_clock_count"]["mean"] == pytest.approx((3.0 * 2 + 4.0 * 2) / 4)
    assert summary["legal_family_presence_rates"] == [{"family": "attack", "rate": 0.75}]
    role_summaries = {item["role"]: item for item in summary["role_summaries"]}
    assert role_summaries["focal"]["compared_steps"] == 3
    assert role_summaries["focal"]["recorded_family_counts"][0] == {"family": "clock_from_hand", "count": 2}
    assert role_summaries["opponent"]["compared_steps"] == 1
    assert role_summaries["opponent"]["recorded_family_counts"][0] == {"family": "attack", "count": 1}
