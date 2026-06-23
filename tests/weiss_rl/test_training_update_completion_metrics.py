from __future__ import annotations

from types import SimpleNamespace

import weiss_rl.training.loop.update_completion as training_update_completion

from .training_update_phase_test_support import RecordingScope, SnapshotPublishingRuntime


def test_training_update_completion_metrics_preserve_merge_precedence() -> None:
    latest_metrics = {"shared": 0.0, "loss": 1.0}
    completion = training_update_completion.TrainingUpdateCompletionMetrics(
        runtime={"shared": 2.0, "runtime_only": 3.0},
        schedule={"shared": 4.0, "schedule_only": 5.0},
        snapshot={"shared": 6.0, "snapshot_only": 7.0},
    )

    merged = completion.apply_to(latest_metrics)

    assert merged is latest_metrics
    assert latest_metrics == {
        "loss": 1.0,
        "runtime_only": 3.0,
        "schedule_only": 5.0,
        "shared": 6.0,
        "snapshot_only": 7.0,
    }


def test_collect_training_update_completion_metrics_publishes_snapshot_after_runtime_and_schedule() -> None:
    events: list[tuple[object, ...]] = []
    learner = SimpleNamespace(update_count=12)
    model = object()

    completion = training_update_completion.collect_training_update_completion_metrics(
        learner=learner,
        model=model,
        runtime=SnapshotPublishingRuntime(events, {"snapshot_metric": 3.0}),
        runtime_metrics={"runtime_metric": 1.0},
        schedule_metrics={"schedule_metric": 2.0},
        profile_timers=True,
        profile_block=lambda enabled, name: events.append(("profile", enabled, name)) or RecordingScope(events, name),
    )

    assert completion == training_update_completion.TrainingUpdateCompletionMetrics(
        runtime={"runtime_metric": 1.0},
        schedule={"schedule_metric": 2.0},
        snapshot={"snapshot_metric": 3.0},
    )
    assert [event[0] for event in events] == ["profile", "enter", "snapshot", "exit"]
    assert events[2][1] == {"learner_model": model, "learner_update_count": 12}
