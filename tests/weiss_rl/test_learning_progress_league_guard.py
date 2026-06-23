from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.diagnostics.learning_progress import build_learning_progress_summary, evaluate_league_guard

from .learning_progress_test_support import _write_json, _write_jsonl


def test_learning_progress_diagnostic_league_guard_fails_on_anchor_and_promotion_health(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "training" / "logs" / "training_metrics.jsonl",
        [{"update_count": 20, "loss": 1.0, "vtrace_rho_p99": 31.0}],
    )
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u10_p2": {
                "update_count": 10,
                "aggregate_score": 0.62,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.55,
                    "B3 HeuristicPublicAggro": 0.52,
                    "B4 HeuristicPublicControl": 0.50,
                },
            },
            "train_u20_p4": {
                "update_count": 20,
                "aggregate_score": 0.50,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.34,
                    "B3 HeuristicPublicAggro": 0.41,
                    "B4 HeuristicPublicControl": 0.47,
                },
            },
        },
    )
    for update in (10, 15, 20):
        _write_json(
            run_dir / "eval" / "promotion_gate" / f"update_{update}" / "promotion_gate.json",
            {
                "focal_policy_id": f"policy_{update:06d}",
                "decision": {"passed": False, "reasons": [{"code": "anchor_loss_guardrail_exceeded"}]},
                "overall_posterior": {"mean": 0.5, "prob_gt_target": 0.1},
            },
        )

    summary = build_learning_progress_summary(run_dir)
    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is False
    failure_codes = {failure["code"] for failure in guard["failures"]}
    assert "latest_anchor_below_threshold" in failure_codes
    assert "latest_periodic_drop_exceeded" in failure_codes
    assert "promotion_gate_no_pass_after_attempts" in failure_codes
    assert "promotion_gate_consecutive_failures_exceeded" in failure_codes
    assert "vtrace_rho_p99_exceeded" in failure_codes
    b2_failure = next(
        failure
        for failure in guard["failures"]
        if failure["code"] == "latest_anchor_below_threshold" and failure["anchor"] == "B2 HeuristicPublic"
    )
    assert b2_failure["observed"] == pytest.approx(0.34)
    assert guard["latest_anchor_scores"]["B4 HeuristicPublicControl"] == pytest.approx(0.47)


def test_learning_progress_diagnostic_league_guard_passes_on_healthy_probe() -> None:
    summary = {
        "periodic_dev_eval": {
            "latest_minus_best": -0.01,
            "records": [
                {
                    "update_count": 20,
                    "aggregate_score": 0.62,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.55,
                        "B3 HeuristicPublicAggro": 0.53,
                        "B4 HeuristicPublicControl": 0.51,
                    },
                }
            ],
        },
        "promotion_gate": {
            "attempt_count": 3,
            "passed_count": 1,
            "consecutive_failure_count": 1,
        },
        "off_policy": {"max_vtrace_rho_p99": 12.0},
    }

    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is True
    assert guard["failures"] == []


def test_learning_progress_diagnostic_league_guard_uses_train_vtrace_tail_when_available() -> None:
    summary = {
        "periodic_dev_eval": {
            "latest_minus_best": -0.01,
            "records": [
                {
                    "update_count": 20,
                    "aggregate_score": 0.62,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.55,
                        "B3 HeuristicPublicAggro": 0.53,
                        "B4 HeuristicPublicControl": 0.51,
                    },
                }
            ],
        },
        "promotion_gate": {
            "attempt_count": 3,
            "passed_count": 1,
            "consecutive_failure_count": 1,
        },
        "off_policy": {
            "max_vtrace_rho_p99": 31.0,
            "max_vtrace_train_rho_p99": 2.0,
        },
    }

    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is True
    assert guard["failures"] == []
    assert guard["vtrace_guard_tail_source"] == "train"
