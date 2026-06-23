from __future__ import annotations

import weiss_rl.training.loop.loop_progress as training_loop_progress
import weiss_rl.training.loop.post_update as training_post_update
from weiss_rl.training.minimal.dev_eval import (
    PeriodicDevEvalGuardResult,
)


def test_post_update_context_runner_preserves_checkpoint_dev_eval_and_finalization_payloads() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    progress = training_loop_progress.TrainingLoopProgress(
        latest_metrics={"loss": 2.0},
        last_dev_eval_summary={"aggregate_score": 0.2},
        last_dev_eval_update_count=3,
        last_checkpoint_guard_rollback_update=2,
    )
    context = training_post_update.PostUpdateCheckpointDevEvalContext(
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
        run_id256="run",
        config_hash256="config",
        tensorboard_logger=object(),
    )
    checkpoint_hooks = object()
    periodic_hooks = object()
    final_hooks = object()
    next_summary = {"aggregate_score": 0.6}

    def checkpoint_fn(**kwargs: object) -> None:
        events.append(("checkpoint", kwargs))

    def dev_eval_fn(**kwargs: object) -> PeriodicDevEvalGuardResult:
        events.append(("dev_eval", kwargs))
        return PeriodicDevEvalGuardResult(
            last_dev_eval_summary=next_summary,
            last_dev_eval_update_count=7,
            last_checkpoint_guard_rollback_update=6,
            stop_requested=False,
        )

    def finalize_fn(**kwargs: object) -> dict[str, object]:
        events.append(("finalize", kwargs))
        return {"finalized": True}

    stop_requested = training_post_update.run_post_update_checkpoint_and_dev_eval_from_context(
        progress=progress,
        context=context,
        schedule=training_post_update.PostUpdateCheckpointDevEvalSchedule(checkpoint_interval_updates=5),
        hooks=training_post_update.PostUpdateCheckpointDevEvalHooks(
            checkpoint_hooks=checkpoint_hooks,
            periodic_dev_eval_hooks=periodic_hooks,
            checkpoint_fn=checkpoint_fn,
            dev_eval_fn=dev_eval_fn,
        ),
    )
    final_result = training_post_update.finalize_training_loop_progress_from_context(
        progress=progress,
        context=training_post_update.FinalTrainingCheckpointContext(
            learner=context.learner,
            stack=context.stack,
            artifacts=context.artifacts,
            training_paths=context.training_paths,
            runtime=context.runtime,
            device=context.device,
            spec_hash256=context.spec_hash256,
            algorithm=context.algorithm,
            tensorboard_logger=context.tensorboard_logger,
        ),
        hooks=training_post_update.FinalTrainingCheckpointHooks(
            hooks=final_hooks,
            finalize_fn=finalize_fn,
        ),
    )

    assert stop_requested is False
    assert final_result == {"finalized": True}
    assert [event[0] for event in events] == ["checkpoint", "dev_eval", "finalize"]
    checkpoint_kwargs = events[0][1]
    assert checkpoint_kwargs["learner"] is context.learner
    assert checkpoint_kwargs["latest_metrics"] is progress.latest_metrics
    assert checkpoint_kwargs["last_dev_eval_summary"] == {"aggregate_score": 0.2}
    assert checkpoint_kwargs["checkpoint_interval_updates"] == 5
    assert checkpoint_kwargs["hooks"] is checkpoint_hooks
    dev_eval_kwargs = events[1][1]
    assert dev_eval_kwargs["model"] is context.model
    assert dev_eval_kwargs["last_dev_eval_summary"] == {"aggregate_score": 0.2}
    assert dev_eval_kwargs["last_dev_eval_update_count"] == 3
    assert dev_eval_kwargs["last_checkpoint_guard_rollback_update"] == 2
    assert dev_eval_kwargs["hooks"] is periodic_hooks
    assert progress.last_dev_eval_summary is next_summary
    assert progress.last_dev_eval_update_count == 7
    assert progress.last_checkpoint_guard_rollback_update == 6
    finalize_kwargs = events[2][1]
    assert finalize_kwargs["latest_metrics"] is progress.latest_metrics
    assert finalize_kwargs["last_dev_eval_summary"] is next_summary
    assert finalize_kwargs["last_dev_eval_update_count"] == 7
    assert finalize_kwargs["hooks"] is final_hooks
