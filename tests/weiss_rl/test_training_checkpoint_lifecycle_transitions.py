from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import weiss_rl.training.checkpointing.lifecycle_decisions as checkpoint_lifecycle_decisions
import weiss_rl.training.checkpointing.lifecycle_transitions as checkpoint_lifecycle_transitions

from .training_checkpoint_test_support import _Learner, _TrainingPaths


def test_checkpoint_lifecycle_transitions_apply_effects_and_build_relative_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    learner = _Learner()
    learner.update_count = 120
    learner.policy_version = 7
    runtime = object()
    learner_model = object()
    restore_checkpoint = object()
    calls: list[tuple[str, dict[str, object]]] = []

    rollback_decision = checkpoint_lifecycle_decisions.RollbackToBestDecision(
        current_score=0.42,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.2, "max_prob_lt_half": 0.8, "max_ci_half_width": 0.15},
        reasons=["score_drop"],
        worst_truncation_rate=0.05,
        worst_stall_rate=0.2,
        worst_no_progress_timeout_rate=0.1,
        worst_natural_timeout_rate=0.03,
    )

    def fake_rollback_effects(**kwargs: object) -> SimpleNamespace:
        calls.append(("rollback_effects", dict(kwargs)))
        return SimpleNamespace(
            best_checkpoint_path=paths.best_checkpoint_path,
            demoted_champions=["policy_000120"],
            publish_metrics={"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5},
        )

    monkeypatch.setattr(
        checkpoint_lifecycle_transitions,
        "apply_rollback_to_best_effects",
        fake_rollback_effects,
    )

    rollback_payload = checkpoint_lifecycle_transitions.apply_rollback_decision_to_event_payload(
        training_paths=paths,
        run_dir=tmp_path,
        runtime=runtime,
        learner=learner,
        learner_model=learner_model,
        latest_metrics={"loss": 3.0},
        decision=rollback_decision,
        restore_checkpoint=restore_checkpoint,
    )

    assert rollback_payload["action"] == "rollback_to_best"
    assert rollback_payload["update_count"] == 120
    assert rollback_payload["policy_version"] == 7
    assert rollback_payload["best_checkpoint_path"] == "training/checkpoints/best.pt"
    assert rollback_payload["latest_checkpoint_path"] == "training/checkpoints/latest.pt"
    assert rollback_payload["snapshot_publish_latency_ms"] == pytest.approx(1.25)
    assert rollback_payload["latest_loss"] == pytest.approx(3.0)
    assert rollback_payload["demoted_champions"] == ["policy_000120"]
    assert calls[0] == (
        "rollback_effects",
        {
            "training_paths": paths,
            "runtime": runtime,
            "learner_model": learner_model,
            "learner_update_count": 120,
            "best_update_count": 80,
            "restore_checkpoint": restore_checkpoint,
        },
    )

    finalize_decision = checkpoint_lifecycle_decisions.FinalizeToBestDecision(
        current_score=0.60,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.45, "max_prob_lt_half": 0.55, "max_ci_half_width": 0.10},
    )

    def fake_finalize_effects(**kwargs: object) -> SimpleNamespace:
        calls.append(("finalize_effects", dict(kwargs)))
        return SimpleNamespace(
            best_checkpoint_path=paths.best_checkpoint_path,
            demoted_champions=[],
            publish_metrics={},
        )

    monkeypatch.setattr(
        checkpoint_lifecycle_transitions,
        "apply_finalize_to_best_effects",
        fake_finalize_effects,
    )

    finalize_payload = checkpoint_lifecycle_transitions.apply_finalize_decision_to_event_payload(
        training_paths=paths,
        run_dir=tmp_path,
        runtime=runtime,
        learner=learner,
        latest_metrics=None,
        decision=finalize_decision,
        restore_checkpoint=restore_checkpoint,
    )

    assert finalize_payload["action"] == "finalize_to_best"
    assert finalize_payload["update_count"] == 120
    assert finalize_payload["policy_version"] == 7
    assert finalize_payload["best_checkpoint_path"] == "training/checkpoints/best.pt"
    assert finalize_payload["latest_checkpoint_path"] == "training/checkpoints/latest.pt"
    assert finalize_payload["latest_loss"] is None
    assert finalize_payload["demoted_champions"] == []
    assert calls[1] == (
        "finalize_effects",
        {
            "training_paths": paths,
            "runtime": runtime,
            "best_update_count": 80,
            "restore_checkpoint": restore_checkpoint,
        },
    )
