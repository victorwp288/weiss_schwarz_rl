from __future__ import annotations

import weiss_rl.training.checkpointing.lifecycle as checkpoint_lifecycle
import weiss_rl.training.checkpointing.lifecycle_decisions as checkpoint_lifecycle_decisions
import weiss_rl.training.checkpointing.lifecycle_plans as checkpoint_lifecycle_plans


def test_checkpoints_reexports_canonical_checkpoint_lifecycle_boundary() -> None:
    from weiss_rl.training import checkpoints

    assert checkpoints.append_checkpoint_guard_event is checkpoint_lifecycle.append_checkpoint_guard_event
    assert checkpoints.checkpoint_guard_log_path is checkpoint_lifecycle.checkpoint_guard_log_path
    assert (
        checkpoints.extract_structured_guard_b2_anchor_score
        is checkpoint_lifecycle.extract_structured_guard_b2_anchor_score
    )
    assert checkpoints.maybe_log_structured_mainmove_guard is checkpoint_lifecycle.maybe_log_structured_mainmove_guard
    assert checkpoints.maybe_rollback_to_best_checkpoint is checkpoint_lifecycle.maybe_rollback_to_best_checkpoint
    assert checkpoints.maybe_finalize_from_best_checkpoint is checkpoint_lifecycle.maybe_finalize_from_best_checkpoint
    assert checkpoint_lifecycle.maybe_rollback_to_best_checkpoint.__module__ == (
        "weiss_rl.training.checkpointing.lifecycle"
    )


def test_checkpoint_lifecycle_reexports_canonical_decision_boundary() -> None:
    assert checkpoint_lifecycle.RollbackToBestDecision is checkpoint_lifecycle_decisions.RollbackToBestDecision
    assert checkpoint_lifecycle.FinalizeToBestDecision is checkpoint_lifecycle_decisions.FinalizeToBestDecision
    assert checkpoint_lifecycle.rollback_lifecycle_decision is checkpoint_lifecycle_plans.rollback_lifecycle_decision
    assert checkpoint_lifecycle.finalize_lifecycle_decision is checkpoint_lifecycle_plans.finalize_lifecycle_decision
    assert checkpoint_lifecycle.rollback_to_best_decision is checkpoint_lifecycle_decisions.rollback_to_best_decision
    assert checkpoint_lifecycle.rollback_to_best_event_payload is (
        checkpoint_lifecycle_decisions.rollback_to_best_event_payload
    )
    assert checkpoint_lifecycle.finalize_to_best_decision is checkpoint_lifecycle_decisions.finalize_to_best_decision
    assert checkpoint_lifecycle.finalize_to_best_event_payload is (
        checkpoint_lifecycle_decisions.finalize_to_best_event_payload
    )
    assert checkpoint_lifecycle_decisions.rollback_to_best_decision.__module__ == (
        "weiss_rl.training.checkpointing.lifecycle_decisions"
    )
    assert checkpoint_lifecycle_plans.rollback_lifecycle_decision.__module__ == (
        "weiss_rl.training.checkpointing.lifecycle_plans"
    )
