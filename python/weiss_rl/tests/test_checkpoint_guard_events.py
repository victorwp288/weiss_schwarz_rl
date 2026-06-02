from __future__ import annotations

from types import SimpleNamespace

import pytest

from weiss_rl.training.checkpointing.guard_events import (
    BestDevEvalCheckpoint,
    best_dev_eval_checkpoint,
    build_finalize_to_best_event_payload,
    build_rollback_to_best_event_payload,
    checkpoint_guard_rollback_reasons,
)


def test_best_dev_eval_checkpoint_requires_prior_update_for_rollback() -> None:
    record = {"metric_kind": "dev_eval_mean", "metric_value": 0.75, "update_count": 12}

    assert best_dev_eval_checkpoint(record) == BestDevEvalCheckpoint(score=0.75, update_count=12)
    assert (
        best_dev_eval_checkpoint(
            record,
            learner_update_count=12,
            require_prior_update=True,
        )
        is None
    )
    assert best_dev_eval_checkpoint(
        record,
        learner_update_count=13,
        require_prior_update=True,
    ) == BestDevEvalCheckpoint(score=0.75, update_count=12)


def test_checkpoint_guard_rollback_reasons_include_boundary_triggers() -> None:
    guard = SimpleNamespace(
        rollback_score_margin=0.10,
        rollback_truncation_rate_threshold=0.30,
        rollback_max_prob_lt_half=0.40,
    )

    assert checkpoint_guard_rollback_reasons(
        checkpoint_guard=guard,
        current_score=0.40,
        best_score=0.50,
        worst_stall_rate=0.30,
        max_prob_lt_half=0.40,
    ) == ["score_drop", "truncation", "confidence"]

    assert (
        checkpoint_guard_rollback_reasons(
            checkpoint_guard=guard,
            current_score=0.401,
            best_score=0.50,
            worst_stall_rate=0.299,
            max_prob_lt_half=0.399,
        )
        == []
    )


def test_build_rollback_event_payload_preserves_checkpoint_guard_contract() -> None:
    payload = build_rollback_to_best_event_payload(
        learner_update_count=20,
        policy_version=8,
        current_score=0.25,
        best=BestDevEvalCheckpoint(score=0.55, update_count=12),
        worst_stall_rate=0.4,
        worst_truncation_rate=0.3,
        worst_no_progress_timeout_rate=0.2,
        worst_natural_timeout_rate=0.1,
        confidence={"min_prob_gt_half": 0.45, "max_prob_lt_half": 0.6, "max_ci_half_width": 0.3},
        reasons=("score_drop", "confidence"),
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        publish_metrics={"snapshot_publish_latency_ms": 12.5, "snapshot_apply_latency_ms": 3.25},
        latest_metrics={"loss": 1.5},
        demoted_champions=("policy_000020",),
    )

    assert payload == {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": 20,
        "policy_version": 8,
        "current_score": pytest.approx(0.25),
        "best_score": pytest.approx(0.55),
        "best_update_count": 12,
        "worst_stall_rate": pytest.approx(0.4),
        "worst_truncation_rate": pytest.approx(0.3),
        "worst_no_progress_timeout_rate": pytest.approx(0.2),
        "worst_natural_timeout_rate": pytest.approx(0.1),
        "min_prob_gt_half": pytest.approx(0.45),
        "max_prob_lt_half": pytest.approx(0.6),
        "max_ci_half_width": pytest.approx(0.3),
        "reasons": ["score_drop", "confidence"],
        "best_checkpoint_path": "training/checkpoints/best.pt",
        "latest_checkpoint_path": "training/checkpoints/latest.pt",
        "rolled_back_checkpoint_path": "training/checkpoints/best.pt",
        "snapshot_publish_latency_ms": pytest.approx(12.5),
        "snapshot_apply_latency_ms": pytest.approx(3.25),
        "latest_loss": pytest.approx(1.5),
        "demoted_champions": ["policy_000020"],
    }


def test_build_finalize_event_payload_excludes_rollback_only_fields() -> None:
    payload = build_finalize_to_best_event_payload(
        learner_update_count=24,
        policy_version=9,
        current_score=0.40,
        best=BestDevEvalCheckpoint(score=0.62, update_count=18),
        confidence={"min_prob_gt_half": 0.52, "max_prob_lt_half": 0.30, "max_ci_half_width": 0.12},
        latest_metrics=None,
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        demoted_champions=(),
    )

    assert payload["action"] == "finalize_to_best"
    assert payload["best_score"] == pytest.approx(0.62)
    assert payload["latest_loss"] is None
    assert "reasons" not in payload
    assert "rolled_back_checkpoint_path" not in payload
    assert "snapshot_publish_latency_ms" not in payload
