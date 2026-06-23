from __future__ import annotations

from types import SimpleNamespace

import weiss_rl.training.loop.update_completion as training_update_completion
import weiss_rl.training.loop.update_schedule as training_update_schedule

from .training_update_phase_test_support import RecordingScope, SnapshotPublishingRuntime


def test_training_update_phase_helpers_preserve_schedule_and_completion_side_effects() -> None:
    events: list[tuple[object, ...]] = []
    stack = object()
    model = object()
    training_config = object()

    class FakeLearner:
        update_count = 2

        def __init__(self) -> None:
            self.entropy_coef: float | None = None
            self.logger = SimpleNamespace(merge_latest_custom_metrics=lambda **kwargs: events.append(("merge", kwargs)))

        def set_entropy_coef(self, value: float) -> None:
            events.append(("set_entropy", value))
            self.entropy_coef = value

        def get_policy_version(self) -> int:
            return 11

    learner = FakeLearner()

    def apply_guidance(**kwargs: object) -> dict[str, float]:
        events.append(("guidance", kwargs))
        return {"guidance_metric": 1.0}

    def entropy_coef(config: object, *, update_count: int) -> float:
        events.append(("entropy", config, update_count))
        return 0.125

    schedule = training_update_schedule.apply_training_update_schedule(
        learner=learner,
        model=model,
        stack=stack,
        training_config=training_config,
        init_schedule_offset_updates=7,
        apply_guidance_schedule_for_next_update=apply_guidance,
        entropy_coef_for_next_update=entropy_coef,
    )

    assert schedule.update_count == 10
    assert schedule.metrics == {
        "guidance_metric": 1.0,
        "guidance_schedule_update_count": 10.0,
        "init_schedule_offset_updates": 7.0,
    }
    assert learner.entropy_coef == 0.125

    latest_metrics = {"loss": 3.0}
    completed = training_update_completion.complete_training_update_metrics(
        learner=learner,
        model=model,
        runtime=SnapshotPublishingRuntime(events, {"snapshot_metric": 4.0}),
        latest_metrics=latest_metrics,
        runtime_metrics={"runtime_metric": 2.0},
        schedule_metrics=schedule.metrics,
        profile_timers=True,
        profile_block=lambda enabled, name: events.append(("profile", enabled, name)) or RecordingScope(events, name),
    )

    assert completed is latest_metrics
    assert latest_metrics == {
        "loss": 3.0,
        "runtime_metric": 2.0,
        "guidance_metric": 1.0,
        "guidance_schedule_update_count": 10.0,
        "init_schedule_offset_updates": 7.0,
        "snapshot_metric": 4.0,
    }
    assert [event[0] for event in events] == [
        "guidance",
        "entropy",
        "set_entropy",
        "profile",
        "enter",
        "snapshot",
        "exit",
        "merge",
    ]
    assert events[0][1]["update_count"] == 10
    assert events[5][1] == {"learner_model": model, "learner_update_count": 2}
    assert events[7][1]["metrics"] is latest_metrics
