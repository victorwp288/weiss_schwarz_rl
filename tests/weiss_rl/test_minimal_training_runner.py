from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import weiss_rl.training.loop.runner as training_runner
import weiss_rl.training.loop.update_step as training_update_step
from weiss_rl.training.loop.runner import MinimalTrainingRunHooks, run_minimal_training_updates
from weiss_rl.training.minimal.dev_eval import (
    PeriodicDevEvalGuardResult,
    TrainingPeriodicDevEvalHooks,
)
from weiss_rl.training.minimal.finalization import (
    TrainingFinalCheckpointHooks,
)
from weiss_rl.training.minimal.promotion import (
    TrainingCheckpointPromotionHooks,
)


def test_run_minimal_training_updates_threads_core_execution_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    summaries = [
        {"aggregate_score": 1.0, "update_count": 1},
        {"aggregate_score": 2.0, "update_count": 2},
    ]
    training_config = object()
    rewards_config = object()
    training_paths = SimpleNamespace(
        scalars_path=tmp_path / "scalars.jsonl",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    stack = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(system=SimpleNamespace(learner_torch_threads=5)),
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    learner = SimpleNamespace(update_count=0, get_policy_version=lambda: 9)
    model = object()

    class FakeRuntime:
        def close(self) -> None:
            events.append(("close", {}))

    class RecordingProfilerContext:
        def __enter__(self) -> None:
            events.append(("profiler_enter", {}))

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            events.append(("profiler_exit", {"exc_type": exc_type}))

    class FakeProfiler:
        def export_chrome_trace(self, trace_path: str) -> None:
            events.append(("export_trace", {"trace_path": trace_path}))

    runtime = FakeRuntime()
    setup = SimpleNamespace(
        training_config=training_config,
        rewards_config=rewards_config,
        training_paths=training_paths,
        algorithm="ppo",
        model=model,
        learner=learner,
        runtime=runtime,
        latest_metrics={"setup_metric": 1.0},
        init_schedule_offset_updates=4,
        resume_state=None,
        config_hash256="config-hash",
    )

    def fake_replay_states_from_config(received_training_config: object, *, repo_root: Path) -> object:
        events.append(("replay_states", {"training_config": received_training_config, "repo_root": repo_root}))
        return SimpleNamespace()

    def fake_reset_anchor(**kwargs: object) -> None:
        events.append(("reset_anchor", kwargs))

    def fake_update_step_from_context(**kwargs: object) -> dict[str, float]:
        events.append(("update", kwargs))
        inputs = kwargs["inputs"]
        options = kwargs["options"]
        hooks_arg = kwargs["hooks"]
        assert isinstance(inputs, training_update_step.TrainingUpdateStepInputs)
        assert isinstance(options, training_update_step.TrainingUpdateStepOptions)
        assert isinstance(hooks_arg, training_update_step.TrainingUpdateStepHooks)
        assert inputs.init_schedule_offset_updates == 4
        assert inputs.replay_states == SimpleNamespace()
        assert options.actor_torch_threads == 3
        assert options.learner_torch_threads == 5
        assert hooks_arg.collect_training_batch is hooks.collect_training_batch
        learner.update_count += 1
        return {"loss": float(learner.update_count)}

    def fake_checkpoint(**kwargs: object) -> object:
        events.append(("checkpoint", kwargs))
        assert kwargs["config_hash256"] == "config-hash"
        return None

    def fake_dev_eval(**kwargs: object) -> PeriodicDevEvalGuardResult:
        events.append(("dev_eval", kwargs))
        update_count = int(learner.update_count)
        return PeriodicDevEvalGuardResult(
            last_dev_eval_summary=summaries[update_count - 1],
            last_dev_eval_update_count=update_count,
            last_checkpoint_guard_rollback_update=None,
            stop_requested=False,
        )

    def fake_finalize(**kwargs: object) -> dict[str, object]:
        events.append(("finalize", kwargs))
        return {"final": True}

    monkeypatch.setattr(training_runner, "training_replay_states_from_config", fake_replay_states_from_config)
    monkeypatch.setattr(training_runner, "reset_policy_anchor_for_fresh_preference_replay", fake_reset_anchor)
    monkeypatch.setattr(training_runner, "run_training_update_step_from_context", fake_update_step_from_context)
    monkeypatch.setattr(training_runner, "maybe_checkpoint_and_promote_snapshot", fake_checkpoint)
    monkeypatch.setattr(training_runner, "maybe_run_periodic_dev_eval_and_checkpoint_guard", fake_dev_eval)
    monkeypatch.setattr(training_runner, "finalize_training_checkpoint_selection", fake_finalize)

    tensorboard_steps: list[dict[str, object]] = []
    tensorboard_logger = SimpleNamespace(log_training_step=lambda **kwargs: tensorboard_steps.append(kwargs))
    hooks = MinimalTrainingRunHooks(
        central_runtime_actor_torch_threads=lambda received_stack, received_runtime: (
            events.append(("actor_threads", {"stack": received_stack, "runtime": received_runtime})) or 3
        ),
        build_training_profiler=lambda **kwargs: (
            events.append(("build_profiler", kwargs))
            or (FakeProfiler(), RecordingProfilerContext(), tmp_path / "profiler")
        ),
        run_structured_warmstart=lambda **kwargs: events.append(("warmstart", kwargs)) or {"warmstart": 1.0},
        profile_block=lambda _enabled, _name: nullcontext(),
        apply_guidance_schedule_for_next_update=lambda **_kwargs: {},
        entropy_coef_for_next_update=lambda *_args, **_kwargs: 0.0,
        torch_num_threads_scope=lambda _threads: nullcontext(),
        collect_training_batch=lambda **_kwargs: SimpleNamespace(),
        write_scalars_record=lambda **kwargs: events.append(("scalars", kwargs)),
        checkpoint_promotion=TrainingCheckpointPromotionHooks(
            write_checkpoint=lambda **_kwargs: None,
            publish_checkpoint_aliases=lambda **_kwargs: {},
            maybe_log_structured_mainmove_guard=lambda **_kwargs: None,
            persist_snapshot_registry_entry=lambda **_kwargs: "candidate",
            run_snapshot_promotion_gate=lambda **_kwargs: False,
        ),
        periodic_dev_eval=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: False,
            run_periodic_dev_eval=lambda **_kwargs: {},
            slug_policy_id=str,
            load_checkpoint_tracker=lambda _training_paths: {},
            confirmatory_dev_eval_request=lambda **_kwargs: None,
            periodic_dev_eval_schedule=lambda _stack: None,
            expand_periodic_dev_eval_paired_seeds=lambda *_args, **_kwargs: [],
            ensure_current_checkpoint=lambda **_kwargs: tmp_path / "checkpoint.pt",
            publish_checkpoint_aliases=lambda **_kwargs: {},
            maybe_log_structured_mainmove_guard=lambda **_kwargs: None,
            maybe_rollback_to_best_checkpoint=lambda **_kwargs: None,
        ),
        final_checkpoint=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=lambda **_kwargs: tmp_path / "checkpoint.pt",
            publish_checkpoint_aliases=lambda **_kwargs: {},
            maybe_finalize_from_best_checkpoint=lambda **_kwargs: None,
            load_checkpoint_tracker=lambda _training_paths: {},
        ),
    )

    metrics = run_minimal_training_updates(
        stack=stack,
        contract=object(),
        artifacts=artifacts,
        setup=setup,
        max_updates=2,
        profile_timers=True,
        torch_profiler=True,
        device=object(),
        checkpoint_interval_updates=1,
        run_id256="run-id",
        spec_hash256="spec-hash",
        tensorboard_logger=tensorboard_logger,
        hooks=hooks,
    )

    assert metrics == {"loss": 2.0}
    assert [event[0] for event in events] == [
        "actor_threads",
        "replay_states",
        "reset_anchor",
        "build_profiler",
        "profiler_enter",
        "warmstart",
        "update",
        "scalars",
        "checkpoint",
        "dev_eval",
        "update",
        "scalars",
        "checkpoint",
        "dev_eval",
        "close",
        "profiler_exit",
        "export_trace",
        "finalize",
    ]
    assert events[5][1]["actor_torch_threads"] == 3
    assert events[5][1]["learner_torch_threads"] == 5
    assert events[8][1]["last_dev_eval_summary"] is None
    assert events[12][1]["last_dev_eval_summary"] == summaries[0]
    assert events[17][1]["last_dev_eval_summary"] == summaries[1]
    assert events[17][1]["last_dev_eval_update_count"] == 2
    assert tensorboard_steps[0]["update_count"] == 1
    assert tensorboard_steps[0]["metrics"] == {"loss": 1.0}
    assert tensorboard_steps[1]["update_count"] == 2
    assert tensorboard_steps[1]["metrics"] == {"loss": 2.0}
