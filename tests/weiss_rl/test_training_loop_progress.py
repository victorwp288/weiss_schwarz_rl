from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import weiss_rl.training.loop.loop_progress as training_loop_progress
import weiss_rl.training.loop.post_update as training_post_update
from weiss_rl.training.minimal.dev_eval import (
    PeriodicDevEvalGuardResult,
)


def test_training_loop_progress_reexports_canonical_post_update_boundary() -> None:
    assert (
        training_loop_progress.PostUpdateCheckpointDevEvalContext
        is training_post_update.PostUpdateCheckpointDevEvalContext
    )
    assert (
        training_loop_progress.PostUpdateCheckpointDevEvalSchedule
        is training_post_update.PostUpdateCheckpointDevEvalSchedule
    )
    assert (
        training_loop_progress.PostUpdateCheckpointDevEvalHooks is training_post_update.PostUpdateCheckpointDevEvalHooks
    )
    assert training_loop_progress.FinalTrainingCheckpointContext is training_post_update.FinalTrainingCheckpointContext
    assert training_loop_progress.FinalTrainingCheckpointHooks is training_post_update.FinalTrainingCheckpointHooks
    assert (
        training_loop_progress.run_post_update_checkpoint_and_dev_eval_from_context
        is training_post_update.run_post_update_checkpoint_and_dev_eval_from_context
    )
    assert (
        training_loop_progress.finalize_training_loop_progress_from_context
        is training_post_update.finalize_training_loop_progress_from_context
    )
    assert training_post_update.run_post_update_checkpoint_and_dev_eval_from_context.__module__ == (
        "weiss_rl.training.loop.post_update"
    )


def test_post_update_stage_plan_names_checkpoint_before_dev_eval() -> None:
    assert [(stage.name, stage.purpose) for stage in training_post_update.POST_UPDATE_CHECKPOINT_DEV_EVAL_PLAN] == [
        ("checkpoint", "publish the current checkpoint and promotion aliases when scheduled"),
        ("dev_eval", "run periodic dev eval and apply checkpoint guard decisions"),
    ]


def test_training_loop_progress_applies_dev_eval_result_and_stop_flag() -> None:
    progress = training_loop_progress.TrainingLoopProgress(latest_metrics={"loss": 1.0})
    summary = {"aggregate_score": 0.25}
    result = PeriodicDevEvalGuardResult(
        last_dev_eval_summary=summary,
        last_dev_eval_update_count=8,
        last_checkpoint_guard_rollback_update=6,
        stop_requested=True,
    )

    stop_requested = progress.apply_dev_eval_result(result)

    assert stop_requested is True
    assert progress.latest_metrics == {"loss": 1.0}
    assert progress.last_dev_eval_summary is summary
    assert progress.last_dev_eval_update_count == 8
    assert progress.last_checkpoint_guard_rollback_update == 6


def test_training_loop_progress_records_scalars_and_tensorboard_with_latest_metrics() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    progress = training_loop_progress.TrainingLoopProgress(latest_metrics={"loss": 1.0})
    learner = SimpleNamespace(update_count=3, get_policy_version=lambda: 11)
    training_paths = SimpleNamespace(scalars_path=Path("metrics.jsonl"))
    tensorboard_logger = SimpleNamespace(log_training_step=lambda **kwargs: events.append(("tensorboard", kwargs)))

    training_loop_progress.write_training_update_outputs(
        progress=progress,
        learner=learner,
        training_paths=training_paths,
        start_time=100.0,
        tensorboard_logger=tensorboard_logger,
        write_scalars_record=lambda **kwargs: events.append(("scalars", kwargs)),
    )

    assert events[0] == (
        "scalars",
        {
            "scalars_path": Path("metrics.jsonl"),
            "learner": learner,
            "metrics": progress.latest_metrics,
            "start_time": 100.0,
        },
    )
    assert events[1][0] == "tensorboard"
    assert events[1][1]["update_count"] == 3
    assert events[1][1]["policy_version"] == 11
    assert events[1][1]["metrics"] is progress.latest_metrics


def test_training_loop_progress_runs_checkpoint_then_dev_eval_and_updates_state() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    progress = training_loop_progress.TrainingLoopProgress(
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary={"aggregate_score": 0.1},
        last_dev_eval_update_count=2,
        last_checkpoint_guard_rollback_update=1,
    )
    next_summary = {"aggregate_score": 0.4}

    def checkpoint_fn(**kwargs: object) -> None:
        events.append(("checkpoint", kwargs))

    def dev_eval_fn(**kwargs: object) -> PeriodicDevEvalGuardResult:
        events.append(("dev_eval", kwargs))
        return PeriodicDevEvalGuardResult(
            last_dev_eval_summary=next_summary,
            last_dev_eval_update_count=5,
            last_checkpoint_guard_rollback_update=4,
            stop_requested=True,
        )

    stop_requested = training_loop_progress.run_post_update_checkpoint_and_dev_eval(
        progress=progress,
        learner=object(),
        model=object(),
        stack=object(),
        contract=object(),
        artifacts=object(),
        training_paths=object(),
        runtime=object(),
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        checkpoint_interval_updates=3,
        run_id256="run",
        config_hash256="config",
        tensorboard_logger=None,
        checkpoint_hooks=object(),
        periodic_dev_eval_hooks=object(),
        checkpoint_fn=checkpoint_fn,
        dev_eval_fn=dev_eval_fn,
    )

    assert stop_requested is True
    assert [event[0] for event in events] == ["checkpoint", "dev_eval"]
    assert events[0][1]["latest_metrics"] is progress.latest_metrics
    assert events[0][1]["last_dev_eval_summary"] == {"aggregate_score": 0.1}
    assert events[1][1]["last_dev_eval_summary"] == {"aggregate_score": 0.1}
    assert events[1][1]["last_dev_eval_update_count"] == 2
    assert events[1][1]["last_checkpoint_guard_rollback_update"] == 1
    assert progress.last_dev_eval_summary is next_summary
    assert progress.last_dev_eval_update_count == 5
    assert progress.last_checkpoint_guard_rollback_update == 4
