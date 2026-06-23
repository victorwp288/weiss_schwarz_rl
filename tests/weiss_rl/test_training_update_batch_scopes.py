from __future__ import annotations

from types import SimpleNamespace

import weiss_rl.training.loop.update_batch as training_update_batch

from .training_update_phase_test_support import RecordingScope


def test_training_update_batch_helpers_preserve_profile_and_thread_scopes() -> None:
    events: list[tuple[object, ...]] = []
    runtime = object()
    training_config = object()
    rewards_config = object()
    learner_batch = object()
    runtime_batch = SimpleNamespace(learner_batch=learner_batch, runtime_metrics={"runtime": 1.0})

    class FakeLearner:
        def update(self, received_batch: object) -> dict[str, float]:
            events.append(("learner_update", received_batch))
            return {"loss": 2.0}

    def profile_block(enabled: bool, name: str) -> RecordingScope:
        events.append(("profile", enabled, name))
        return RecordingScope(events, f"profile:{name}")

    def torch_num_threads_scope(thread_count: int | None) -> RecordingScope:
        events.append(("threads", thread_count))
        return RecordingScope(events, f"threads:{thread_count}")

    collected = training_update_batch.collect_runtime_training_batch(
        runtime=runtime,
        algorithm="impala",
        training_config=training_config,
        rewards_config=rewards_config,
        profile_timers=True,
        actor_torch_threads=3,
        collect_training_batch=lambda **kwargs: events.append(("collect", kwargs)) or runtime_batch,
        profile_block=profile_block,
        torch_num_threads_scope=torch_num_threads_scope,
    )
    metrics = training_update_batch.apply_learner_training_batch(
        learner=FakeLearner(),
        learner_batch=learner_batch,
        profile_timers=False,
        learner_torch_threads=5,
        profile_block=profile_block,
        torch_num_threads_scope=torch_num_threads_scope,
    )

    assert collected is runtime_batch
    assert metrics == {"loss": 2.0}
    assert [event[0] for event in events] == [
        "profile",
        "enter",
        "threads",
        "enter",
        "collect",
        "exit",
        "exit",
        "profile",
        "enter",
        "threads",
        "enter",
        "learner_update",
        "exit",
        "exit",
    ]
    assert events[0] == ("profile", True, "collect_update_batch")
    assert events[1] == ("enter", "profile:collect_update_batch")
    assert events[2] == ("threads", 3)
    assert events[4][1] == {
        "runtime": runtime,
        "algorithm": "impala",
        "training_config": training_config,
        "rewards_config": rewards_config,
    }
    assert events[7] == ("profile", False, "learner_update")
    assert events[8] == ("enter", "profile:learner_update")
    assert events[9] == ("threads", 5)
    assert events[11] == ("learner_update", learner_batch)
