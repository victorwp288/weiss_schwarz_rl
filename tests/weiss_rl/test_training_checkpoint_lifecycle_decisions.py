from __future__ import annotations

from types import SimpleNamespace

import pytest
import weiss_rl.training.checkpointing.lifecycle as checkpoint_lifecycle


def test_checkpoint_lifecycle_rollback_decision_collects_reason_and_diagnostics() -> None:
    checkpoint_guard = SimpleNamespace(
        min_best_score=0.55,
        rollback_score_margin=0.10,
        rollback_truncation_rate_threshold=0.25,
        rollback_max_prob_lt_half=0.80,
    )

    decision = checkpoint_lifecycle.rollback_to_best_decision(
        checkpoint_guard=checkpoint_guard,
        best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80},
        learner_update_count=120,
        dev_eval_summary={
            "aggregate_score": 0.54,
            "anchors": {
                "B2": {
                    "summary": {"games": 20, "truncations": 1, "no_progress_timeouts": 3, "natural_timeouts": 2},
                    "uncertainty": {"prob_gt_half": 0.40, "prob_lt_half": 0.60, "ci_half_width": 0.15},
                }
            },
        },
    )

    assert decision is not None
    assert decision.current_score == pytest.approx(0.54)
    assert decision.best.score == pytest.approx(0.70)
    assert decision.best.update_count == 80
    assert decision.reasons == ["score_drop"]
    assert decision.confidence["max_prob_lt_half"] == pytest.approx(0.60)
    assert decision.worst_truncation_rate == pytest.approx(0.05)
    assert decision.worst_no_progress_timeout_rate == pytest.approx(0.15)
    assert decision.worst_natural_timeout_rate == pytest.approx(0.10)
    assert decision.worst_stall_rate == pytest.approx(0.15)


def test_checkpoint_lifecycle_finalize_decision_requires_current_score_below_best() -> None:
    best_record = {"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80}
    losing_summary = {
        "aggregate_score": 0.60,
        "anchors": {
            "B2": {
                "summary": {"games": 20, "truncations": 0},
                "uncertainty": {"prob_gt_half": 0.45, "prob_lt_half": 0.55, "ci_half_width": 0.10},
            }
        },
    }

    decision = checkpoint_lifecycle.finalize_to_best_decision(
        best_record=best_record,
        dev_eval_summary=losing_summary,
    )

    assert decision is not None
    assert decision.current_score == pytest.approx(0.60)
    assert decision.best.update_count == 80
    assert decision.confidence["min_prob_gt_half"] == pytest.approx(0.45)
    assert (
        checkpoint_lifecycle.finalize_to_best_decision(
            best_record=best_record,
            dev_eval_summary={"aggregate_score": 0.70},
        )
        is None
    )
