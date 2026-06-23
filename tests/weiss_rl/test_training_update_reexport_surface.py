from __future__ import annotations

import weiss_rl.training.loop.run_contexts as training_run_contexts
import weiss_rl.training.loop.runner as training_runner
import weiss_rl.training.loop.update as training_update
import weiss_rl.training.loop.update_batch as training_update_batch
import weiss_rl.training.loop.update_completion as training_update_completion
import weiss_rl.training.loop.update_phases as training_update_phases
import weiss_rl.training.loop.update_schedule as training_update_schedule
import weiss_rl.training.loop.update_stage_pipeline as training_update_stage_pipeline
import weiss_rl.training.loop.update_step as training_update_step
import weiss_rl.training.replay_data.training_replay_dispatch as training_replay_dispatch
import weiss_rl.training.replay_data.training_replay_states as training_replay_states


def test_training_update_reexports_canonical_replay_dispatch_helpers() -> None:
    assert training_update.TrainingReplayStates is training_replay_dispatch.TrainingReplayStates
    assert training_update.training_replay_states_from_config is (
        training_replay_dispatch.training_replay_states_from_config
    )
    assert training_update.reset_policy_anchor_for_fresh_preference_replay is (
        training_replay_dispatch.reset_policy_anchor_for_fresh_preference_replay
    )
    assert training_update.reset_policy_anchor_to_current_model is (
        training_replay_dispatch.reset_policy_anchor_to_current_model
    )
    assert training_update.run_post_update_replay is training_replay_dispatch.run_post_update_replay


def test_training_replay_dispatch_reexports_split_state_helpers() -> None:
    assert training_replay_dispatch.TrainingReplayStates is training_replay_states.TrainingReplayStates
    assert training_replay_dispatch.training_replay_states_from_config is (
        training_replay_states.training_replay_states_from_config
    )
    assert training_replay_dispatch.reset_policy_anchor_for_fresh_preference_replay is (
        training_replay_states.reset_policy_anchor_for_fresh_preference_replay
    )
    assert (
        training_replay_dispatch.reset_policy_anchor_to_current_model
        is training_replay_states.reset_policy_anchor_to_current_model
    )


def test_training_update_reexports_canonical_phase_helpers() -> None:
    assert (
        training_update.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
        is training_update_phases.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
    )
    assert training_update.TrainingUpdateScheduleResult is training_update_phases.TrainingUpdateScheduleResult
    assert (
        training_update.schedule_update_count_for_next_update
        is training_update_phases.schedule_update_count_for_next_update
    )
    assert training_update.apply_training_update_schedule is training_update_phases.apply_training_update_schedule
    assert training_update.collect_runtime_training_batch is training_update_phases.collect_runtime_training_batch
    assert training_update.apply_learner_training_batch is training_update_phases.apply_learner_training_batch
    assert (
        training_update.collect_training_update_completion_metrics
        is training_update_phases.collect_training_update_completion_metrics
    )
    assert training_update.complete_training_update_metrics is training_update_phases.complete_training_update_metrics


def test_training_update_phases_reexports_split_phase_modules() -> None:
    assert (
        training_update_phases.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
        is training_update_completion.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
    )
    assert training_update_phases.TrainingUpdateScheduleResult is training_update_schedule.TrainingUpdateScheduleResult
    assert (
        training_update_phases.schedule_update_count_for_next_update
        is training_update_schedule.schedule_update_count_for_next_update
    )
    assert (
        training_update_phases.apply_training_update_schedule is training_update_schedule.apply_training_update_schedule
    )
    assert training_update_phases.RuntimeTrainingBatchResult is training_update_batch.RuntimeTrainingBatchResult
    assert training_update_phases.collect_runtime_training_batch is training_update_batch.collect_runtime_training_batch
    assert training_update_phases.apply_learner_training_batch is training_update_batch.apply_learner_training_batch
    assert (
        training_update_phases.TrainingUpdateCompletionMetrics
        is training_update_completion.TrainingUpdateCompletionMetrics
    )
    assert (
        training_update_phases.publish_runtime_snapshot_after_update
        is training_update_completion.publish_runtime_snapshot_after_update
    )
    assert (
        training_update_phases.collect_training_update_completion_metrics
        is training_update_completion.collect_training_update_completion_metrics
    )
    assert (
        training_update_phases.merge_post_update_auxiliary_metrics_into_training_log
        is training_update_completion.merge_post_update_auxiliary_metrics_into_training_log
    )
    assert (
        training_update_phases.complete_training_update_metrics
        is training_update_completion.complete_training_update_metrics
    )


def test_training_update_reexports_canonical_step_context_helpers() -> None:
    assert training_update.TrainingUpdateStepInputs is training_update_step.TrainingUpdateStepInputs
    assert training_update.TrainingUpdateStepOptions is training_update_step.TrainingUpdateStepOptions
    assert training_update.TrainingUpdateStepHooks is training_update_step.TrainingUpdateStepHooks
    assert (
        training_update.run_training_update_step_from_context
        is training_update_step.run_training_update_step_from_context
    )
    assert training_update_step.run_training_update_step_from_context.__module__ == (
        "weiss_rl.training.loop.update_step"
    )


def test_training_update_step_uses_canonical_stage_pipeline_boundary() -> None:
    assert (
        training_update_step.TrainingUpdateStageFunctions is training_update_stage_pipeline.TrainingUpdateStageFunctions
    )
    assert (
        training_update_step.run_training_update_stage_pipeline
        is training_update_stage_pipeline.run_training_update_stage_pipeline
    )
    assert training_update_stage_pipeline.run_training_update_stage_pipeline.__module__ == (
        "weiss_rl.training.loop.update_stage_pipeline"
    )


def test_training_runner_uses_canonical_context_builder_boundary() -> None:
    assert training_runner.build_training_run_contexts is training_run_contexts.build_training_run_contexts
    assert training_run_contexts.build_training_run_contexts.__module__ == "weiss_rl.training.loop.run_contexts"
