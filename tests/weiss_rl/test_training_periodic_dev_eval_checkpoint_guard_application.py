from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.checkpointing.guards.periodic_dev_eval_guard import (
    CheckpointGuardApplicationResult,
    apply_periodic_dev_eval_checkpoint_guard,
)

from tests.weiss_rl.training_periodic_dev_eval_guard_test_support import make_periodic_dev_eval_hooks


class FakeTensorBoardLogger:
    def __init__(self, events: list[tuple[str, dict[str, object]]]) -> None:
        self.events = events

    def log_periodic_dev_eval(self, payload: object, *, step: int) -> None:
        self.events.append(("tb_eval", {"payload": payload, "step": step}))

    def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
        self.events.append(("tb_tracker", {"payload": payload, "step": step}))


def test_periodic_dev_eval_checkpoint_guard_helper_keeps_state_without_rollback() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    latest_metrics = {"loss": 1.0}
    effective_summary = {"anchor_scores": {}, "aggregate_score": 0.3}
    tracker_payload = {"best": {"update": 5}}

    result = apply_periodic_dev_eval_checkpoint_guard(
        hooks=make_periodic_dev_eval_hooks(
            ensure_current_checkpoint=lambda **kwargs: events.append(("ensure", kwargs)) or Path("checkpoint.pt"),
            publish_checkpoint_aliases=lambda **kwargs: events.append(("aliases", kwargs)) or tracker_payload,
            maybe_log_structured_mainmove_guard=lambda **kwargs: events.append(("guard", kwargs)),
            maybe_rollback_to_best_checkpoint=lambda **kwargs: events.append(("rollback", kwargs)) or None,
        ),
        stack=SimpleNamespace(config=SimpleNamespace(curriculum=None)),
        learner=SimpleNamespace(update_count=5),
        model=object(),
        artifacts=object(),
        training_paths=object(),
        runtime=object(),
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        latest_metrics=latest_metrics,
        effective_summary=effective_summary,
        last_checkpoint_guard_rollback_update=2,
        run_id256="run-id",
        tensorboard_logger=FakeTensorBoardLogger(events),
        update_count=5,
    )

    assert result == CheckpointGuardApplicationResult(
        tracker_payload=tracker_payload,
        next_rollback_update=2,
        stop_requested=False,
    )
    assert [event[0] for event in events] == ["ensure", "aliases", "guard", "rollback", "tb_eval", "tb_tracker"]
    assert events[1][1]["dev_eval_summary"] is effective_summary
    assert events[3][1]["last_rollback_update"] == 2
    assert "checkpoint_guard_stop_after_rollback" not in latest_metrics
