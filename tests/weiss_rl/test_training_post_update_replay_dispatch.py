from __future__ import annotations

import pytest
import weiss_rl.training.replay_data.training_replay_dispatch as training_replay_dispatch
from weiss_rl.training.replay_data.training_replay_states import TrainingReplayStates


def test_post_update_replay_runs_each_auxiliary_path_in_stable_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    replay_states = TrainingReplayStates(
        trajectory_bc=object(),
        paired_swing=object(),
        paired_outcome_preference=object(),
    )
    learner = object()
    training_config = object()
    device = object()
    latest_metrics = {"loss": 0.25}

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            calls.append(("enter", self.name))

        def __exit__(self, *_exc: object) -> None:
            calls.append(("exit", self.name))

    def fake_profile_block(enabled: bool, name: str) -> FakeScope:
        calls.append(("profile", enabled, name))
        return FakeScope(f"profile:{name}")

    def fake_thread_scope(thread_count: int | None) -> FakeScope:
        calls.append(("threads", thread_count))
        return FakeScope(f"threads:{thread_count}")

    def fake_trajectory_bc(**kwargs: object) -> None:
        calls.append(("trajectory", kwargs))

    def fake_paired_swing(**kwargs: object) -> None:
        calls.append(("swing", kwargs))

    def fake_preference(**kwargs: object) -> None:
        calls.append(("preference", kwargs))

    monkeypatch.setattr(training_replay_dispatch, "maybe_run_trajectory_bc_replay", fake_trajectory_bc)
    monkeypatch.setattr(training_replay_dispatch, "maybe_run_paired_swing_replay", fake_paired_swing)
    monkeypatch.setattr(training_replay_dispatch, "maybe_run_paired_outcome_preference_replay", fake_preference)

    training_replay_dispatch.run_post_update_replay(
        replay_states=replay_states,
        learner=learner,
        training_config=training_config,
        device=device,
        update_count=17,
        latest_metrics=latest_metrics,
        profile_timers=True,
        learner_torch_threads=3,
        profile_block=fake_profile_block,
        torch_num_threads_scope=fake_thread_scope,
    )

    replay_calls = [call for call in calls if call[0] in {"trajectory", "swing", "preference"}]
    assert [call[0] for call in replay_calls] == ["trajectory", "swing", "preference"]
    assert replay_calls[0][1] == {
        "state": replay_states.trajectory_bc,
        "learner": learner,
        "training_config": training_config,
        "device": device,
        "update_count": 17,
        "latest_metrics": latest_metrics,
    }
    assert replay_calls[1][1] == {
        "state": replay_states.paired_swing,
        "learner": learner,
        "device": device,
        "update_count": 17,
        "latest_metrics": latest_metrics,
    }
    assert replay_calls[2][1] == {
        "state": replay_states.paired_outcome_preference,
        "learner": learner,
        "device": device,
        "update_count": 17,
        "latest_metrics": latest_metrics,
    }
    assert [call for call in calls if call[0] == "profile"] == [
        ("profile", True, "trajectory_bc_replay"),
        ("profile", True, "paired_swing_replay"),
        ("profile", True, "paired_outcome_preference_replay"),
    ]
    assert [call for call in calls if call[0] == "threads"] == [("threads", 3), ("threads", 3), ("threads", 3)]


def test_post_update_replay_paths_document_stable_dispatch_contract() -> None:
    paths = training_replay_dispatch.post_update_replay_paths()

    assert training_replay_dispatch.post_update_replay_path_specs() == (
        ("trajectory_bc_replay", "trajectory_bc", "maybe_run_trajectory_bc_replay", True),
        ("paired_swing_replay", "paired_swing", "maybe_run_paired_swing_replay", False),
        (
            "paired_outcome_preference_replay",
            "paired_outcome_preference",
            "maybe_run_paired_outcome_preference_replay",
            False,
        ),
    )
    assert [path.runner for path in paths] == [
        training_replay_dispatch.maybe_run_trajectory_bc_replay,
        training_replay_dispatch.maybe_run_paired_swing_replay,
        training_replay_dispatch.maybe_run_paired_outcome_preference_replay,
    ]
