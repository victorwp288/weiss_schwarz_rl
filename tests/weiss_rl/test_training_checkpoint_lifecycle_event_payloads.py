from __future__ import annotations

import pytest
import weiss_rl.training.checkpointing.lifecycle_decisions as checkpoint_lifecycle_decisions


def test_checkpoint_lifecycle_decision_payloads_preserve_decision_diagnostics() -> None:
    rollback_decision = checkpoint_lifecycle_decisions.RollbackToBestDecision(
        current_score=0.42,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.2, "max_prob_lt_half": 0.8, "max_ci_half_width": 0.15},
        reasons=["score_drop", "confidence"],
        worst_truncation_rate=0.05,
        worst_stall_rate=0.2,
        worst_no_progress_timeout_rate=0.1,
        worst_natural_timeout_rate=0.03,
    )

    rollback_payload = checkpoint_lifecycle_decisions.rollback_to_best_event_payload(
        learner_update_count=120,
        policy_version=7,
        decision=rollback_decision,
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        publish_metrics={"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5},
        latest_metrics={"loss": 3.0},
        demoted_champions=["policy_000120"],
    )

    assert rollback_payload["action"] == "rollback_to_best"
    assert rollback_payload["current_score"] == pytest.approx(0.42)
    assert rollback_payload["best_score"] == pytest.approx(0.70)
    assert rollback_payload["best_update_count"] == 80
    assert rollback_payload["worst_stall_rate"] == pytest.approx(0.2)
    assert rollback_payload["reasons"] == ["score_drop", "confidence"]
    assert rollback_payload["rolled_back_checkpoint_path"] == "training/checkpoints/best.pt"
    assert rollback_payload["latest_loss"] == pytest.approx(3.0)
    assert rollback_payload["demoted_champions"] == ["policy_000120"]

    finalize_decision = checkpoint_lifecycle_decisions.FinalizeToBestDecision(
        current_score=0.60,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.45, "max_prob_lt_half": 0.55, "max_ci_half_width": 0.10},
    )
    finalize_payload = checkpoint_lifecycle_decisions.finalize_to_best_event_payload(
        learner_update_count=140,
        policy_version=9,
        decision=finalize_decision,
        latest_metrics=None,
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        demoted_champions=[],
    )

    assert finalize_payload["action"] == "finalize_to_best"
    assert finalize_payload["current_score"] == pytest.approx(0.60)
    assert finalize_payload["best_score"] == pytest.approx(0.70)
    assert finalize_payload["latest_loss"] is None
    assert finalize_payload["demoted_champions"] == []
