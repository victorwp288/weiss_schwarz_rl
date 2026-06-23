from __future__ import annotations

from types import SimpleNamespace

import weiss_rl.training.loop.update_stage_pipeline as training_update_stage_pipeline
import weiss_rl.training.loop.update_step as training_update_step


def test_training_update_stage_pipeline_preserves_stage_order_and_payloads() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=4)
    learner_batch = object()
    runtime_batch = SimpleNamespace(learner_batch=learner_batch, runtime_metrics={"runtime": 1.0})
    schedule_result = SimpleNamespace(metrics={"schedule": 2.0})
    latest_metrics = {"loss": 3.0}
    completed_metrics = {"loss": 3.0, "runtime": 1.0, "schedule": 2.0, "snapshot": 4.0}
    inputs = training_update_step.TrainingUpdateStepInputs(
        learner=learner,
        model=object(),
        stack=object(),
        runtime=object(),
        algorithm="impala",
        training_config=object(),
        rewards_config=object(),
        replay_states=object(),
        device=object(),
        init_schedule_offset_updates=9,
    )
    options = training_update_step.TrainingUpdateStepOptions(
        profile_timers=True,
        actor_torch_threads=2,
        learner_torch_threads=5,
    )
    hooks = training_update_step.TrainingUpdateStepHooks(
        apply_guidance_schedule_for_next_update=object(),
        entropy_coef_for_next_update=object(),
        collect_training_batch=object(),
        profile_block=object(),
        torch_num_threads_scope=object(),
    )

    def apply_schedule(**kwargs: object) -> object:
        events.append(("schedule", kwargs))
        return schedule_result

    def collect_batch(**kwargs: object) -> object:
        events.append(("collect", kwargs))
        return runtime_batch

    def apply_learner(**kwargs: object) -> dict[str, float]:
        events.append(("learner", kwargs))
        return latest_metrics

    def run_replay(**kwargs: object) -> None:
        events.append(("replay", kwargs))

    def complete_metrics(**kwargs: object) -> dict[str, float]:
        events.append(("complete", kwargs))
        return completed_metrics

    result = training_update_stage_pipeline.run_training_update_stage_pipeline(
        inputs=inputs,
        options=options,
        hooks=hooks,
        stage_functions=training_update_stage_pipeline.TrainingUpdateStageFunctions(
            apply_training_update_schedule=apply_schedule,
            collect_runtime_training_batch=collect_batch,
            apply_learner_training_batch=apply_learner,
            run_post_update_replay=run_replay,
            complete_training_update_metrics=complete_metrics,
        ),
    )

    assert result is completed_metrics
    assert [event[0] for event in events] == ["schedule", "collect", "learner", "replay", "complete"]
    assert events[0][1] == {
        "learner": learner,
        "model": inputs.model,
        "stack": inputs.stack,
        "training_config": inputs.training_config,
        "init_schedule_offset_updates": 9,
        "apply_guidance_schedule_for_next_update": hooks.apply_guidance_schedule_for_next_update,
        "entropy_coef_for_next_update": hooks.entropy_coef_for_next_update,
    }
    assert events[1][1] == {
        "runtime": inputs.runtime,
        "algorithm": "impala",
        "training_config": inputs.training_config,
        "rewards_config": inputs.rewards_config,
        "profile_timers": True,
        "actor_torch_threads": 2,
        "collect_training_batch": hooks.collect_training_batch,
        "profile_block": hooks.profile_block,
        "torch_num_threads_scope": hooks.torch_num_threads_scope,
    }
    assert events[2][1] == {
        "learner": learner,
        "learner_batch": learner_batch,
        "profile_timers": True,
        "learner_torch_threads": 5,
        "profile_block": hooks.profile_block,
        "torch_num_threads_scope": hooks.torch_num_threads_scope,
    }
    assert events[3][1] == {
        "replay_states": inputs.replay_states,
        "learner": learner,
        "training_config": inputs.training_config,
        "device": inputs.device,
        "update_count": 4,
        "latest_metrics": latest_metrics,
        "profile_timers": True,
        "learner_torch_threads": 5,
        "profile_block": hooks.profile_block,
        "torch_num_threads_scope": hooks.torch_num_threads_scope,
    }
    assert events[4][1] == {
        "learner": learner,
        "model": inputs.model,
        "runtime": inputs.runtime,
        "latest_metrics": latest_metrics,
        "runtime_metrics": {"runtime": 1.0},
        "schedule_metrics": {"schedule": 2.0},
        "profile_timers": True,
        "profile_block": hooks.profile_block,
    }
