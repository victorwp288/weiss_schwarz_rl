from __future__ import annotations

from types import SimpleNamespace

import pytest
import weiss_rl.training.loop.update_step as training_update_step
from weiss_rl.training.loop.update import run_training_update_step
from weiss_rl.training.replay_data.training_replay_states import TrainingReplayStates


def test_training_update_step_preserves_schedule_collect_replay_and_snapshot_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    replay_states = TrainingReplayStates(
        trajectory_bc=None,
        paired_swing=None,
        paired_outcome_preference=None,
    )
    model = object()
    stack = object()
    algorithm = object()
    training_config = object()
    rewards_config = object()
    device = object()

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, *_exc: object) -> None:
            events.append(("exit", self.name))

    class FakeLearner:
        update_count = 4

        def __init__(self) -> None:
            self.entropy_coef: float | None = None
            self.logger = SimpleNamespace(merge_latest_custom_metrics=self._merge_latest_custom_metrics)

        def set_entropy_coef(self, value: float) -> None:
            events.append(("set_entropy", value))
            self.entropy_coef = value

        def update(self, learner_batch: object) -> dict[str, float]:
            events.append(("learner_update", learner_batch))
            self.update_count = 5
            return {"loss": 1.0}

        def get_policy_version(self) -> int:
            return 9

        def _merge_latest_custom_metrics(self, **kwargs: object) -> None:
            events.append(("merge_metrics", kwargs))

    learner = FakeLearner()

    class FakeRuntime:
        def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
            events.append(("snapshot", kwargs))
            return {"snapshot_metric": 5.0}

    runtime = FakeRuntime()

    def fake_apply_guidance(**kwargs: object) -> dict[str, float]:
        events.append(("guidance", kwargs))
        return {"guidance_metric": 2.0}

    def fake_entropy_coef(config: object, *, update_count: int) -> float:
        events.append(("entropy_lookup", config, update_count))
        return 0.25

    def fake_collect_batch(**kwargs: object) -> object:
        events.append(("collect", kwargs))
        return SimpleNamespace(
            learner_batch="learner-batch",
            runtime_metrics={"runtime_metric": 4.0},
        )

    def fake_profile_block(enabled: bool, name: str) -> FakeScope:
        events.append(("profile", enabled, name))
        return FakeScope(f"profile:{name}")

    def fake_thread_scope(thread_count: int | None) -> FakeScope:
        events.append(("threads", thread_count))
        return FakeScope(f"threads:{thread_count}")

    def fake_replay(**kwargs: object) -> None:
        events.append(("replay", kwargs))
        latest_metrics = kwargs["latest_metrics"]
        assert latest_metrics == {"loss": 1.0}
        latest_metrics["replay_metric"] = 3.0

    monkeypatch.setattr(training_update_step, "run_post_update_replay", fake_replay)

    latest_metrics = run_training_update_step(
        learner=learner,
        model=model,
        stack=stack,
        runtime=runtime,
        algorithm=algorithm,
        training_config=training_config,
        rewards_config=rewards_config,
        replay_states=replay_states,
        device=device,
        init_schedule_offset_updates=10,
        profile_timers=True,
        actor_torch_threads=2,
        learner_torch_threads=3,
        apply_guidance_schedule_for_next_update=fake_apply_guidance,
        entropy_coef_for_next_update=fake_entropy_coef,
        collect_training_batch=fake_collect_batch,
        profile_block=fake_profile_block,
        torch_num_threads_scope=fake_thread_scope,
    )

    assert learner.entropy_coef == 0.25
    assert latest_metrics == {
        "loss": 1.0,
        "replay_metric": 3.0,
        "runtime_metric": 4.0,
        "guidance_metric": 2.0,
        "guidance_schedule_update_count": 15.0,
        "init_schedule_offset_updates": 10.0,
        "snapshot_metric": 5.0,
    }
    event_names = [event[0] for event in events]
    assert event_names == [
        "guidance",
        "entropy_lookup",
        "set_entropy",
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
        "replay",
        "profile",
        "enter",
        "snapshot",
        "exit",
        "merge_metrics",
    ]
    assert events[0][1]["update_count"] == 15
    assert events[5] == ("threads", 2)
    assert events[12] == ("threads", 3)
    assert events[17][1]["update_count"] == 5
    assert events[17][1]["latest_metrics"] is latest_metrics
    assert events[20][1] == {"learner_model": model, "learner_update_count": 5}
    merge_kwargs = events[-1][1]
    assert merge_kwargs["update_count"] == 5
    assert merge_kwargs["policy_version"] == 9
    assert merge_kwargs["metrics"] is latest_metrics
