from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from weiss_rl.training import (
    checkpoint_finalization,
    checkpoint_periodic_dev_eval,
    checkpoint_periodic_dev_eval_confirmatory,
    checkpoint_periodic_dev_eval_guard,
    checkpoint_snapshot_promotion,
    training_loop_progress,
    training_post_update,
    training_replay_dispatch,
    training_replay_paths,
    training_replay_states,
    training_run_contexts,
    training_runner,
    training_setup,
    training_update,
    training_update_batch,
    training_update_completion,
    training_update_phases,
    training_update_schedule,
    training_update_stage_pipeline,
    training_update_step,
)
from weiss_rl.training.minimal_dev_eval import (
    PeriodicDevEvalGuardResult,
    TrainingPeriodicDevEvalHooks,
    _maybe_run_periodic_dev_eval_and_checkpoint_guard,
)
from weiss_rl.training.minimal_finalization import (
    TrainingFinalCheckpointHooks,
    _final_dev_eval_summary_for_update,
    _finalize_training_checkpoint_selection,
)
from weiss_rl.training.minimal_hook_groups import minimal_training_hook_groups
from weiss_rl.training.minimal_initialization import (
    _effective_init_schedule_offset_from_checkpoint,
    _infer_init_schedule_offset_from_scalars,
    _publish_initial_runtime_snapshot_after_resume,
)
from weiss_rl.training.minimal_promotion import (
    TrainingCheckpointPromotionHooks,
    _league_reference_update_from_metrics,
    _maybe_checkpoint_and_promote_snapshot,
)
from weiss_rl.training.minimal_setup import (
    MinimalTrainingSetupHooks,
    _require_training_stack_components,
    build_minimal_training_setup,
)
from weiss_rl.training.minimal_update import (
    _POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    TrainingReplayStates,
    _merge_post_update_auxiliary_metrics_into_training_log,
    _reset_policy_anchor_for_fresh_preference_replay,
    _run_post_update_replay,
    _run_training_update_step,
    _schedule_update_count_for_next_update,
)
from weiss_rl.training.train_entrypoint_main import _require_explicit_resume_geometry
from weiss_rl.training.train_entrypoint_phases import (
    TrainCliState,
    TrainStartupState,
    execute_train_run,
    prepare_train_manifest_state,
    prepare_train_startup_state,
    resolve_train_cli_state,
)
from weiss_rl.training.training_runner import MinimalTrainingRunHooks, run_minimal_training_updates


def test_train_entrypoint_phases_facade_reexports_split_phase_modules() -> None:
    from weiss_rl.training import (
        train_entrypoint_cli_phase,
        train_entrypoint_execution_phase,
        train_entrypoint_manifest_phase,
        train_entrypoint_phases,
        train_entrypoint_startup_phase,
        train_entrypoint_state,
    )

    assert train_entrypoint_phases.TrainCliState is train_entrypoint_state.TrainCliState
    assert train_entrypoint_phases.TrainStartupState is train_entrypoint_state.TrainStartupState
    assert train_entrypoint_phases.TrainManifestState is train_entrypoint_state.TrainManifestState
    assert train_entrypoint_phases.require_explicit_resume_geometry is (
        train_entrypoint_cli_phase.require_explicit_resume_geometry
    )
    assert train_entrypoint_phases.resolve_train_cli_state is train_entrypoint_cli_phase.resolve_train_cli_state
    assert train_entrypoint_phases.prepare_train_startup_state is (
        train_entrypoint_startup_phase.prepare_train_startup_state
    )
    assert train_entrypoint_phases.prepare_train_manifest_state is (
        train_entrypoint_manifest_phase.prepare_train_manifest_state
    )
    assert train_entrypoint_phases.execute_train_run is train_entrypoint_execution_phase.execute_train_run


def test_minimal_promotion_reexports_checkpoint_snapshot_promotion_boundary() -> None:
    import weiss_rl.training.minimal_promotion as minimal_promotion

    assert (
        minimal_promotion.TrainingCheckpointPromotionHooks
        is checkpoint_snapshot_promotion.TrainingCheckpointPromotionHooks
    )
    assert (
        minimal_promotion._league_reference_update_from_metrics
        is checkpoint_snapshot_promotion.league_reference_update_from_metrics
    )
    assert (
        minimal_promotion._maybe_checkpoint_and_promote_snapshot
        is checkpoint_snapshot_promotion.maybe_checkpoint_and_promote_snapshot
    )
    assert checkpoint_snapshot_promotion.maybe_checkpoint_and_promote_snapshot.__module__ == (
        "weiss_rl.training.checkpoint_snapshot_promotion"
    )


def test_minimal_dev_eval_reexports_checkpoint_periodic_dev_eval_boundary() -> None:
    import weiss_rl.training.minimal_dev_eval as minimal_dev_eval

    assert minimal_dev_eval.PeriodicDevEvalGuardResult is checkpoint_periodic_dev_eval.PeriodicDevEvalGuardResult
    assert minimal_dev_eval.TrainingPeriodicDevEvalHooks is checkpoint_periodic_dev_eval.TrainingPeriodicDevEvalHooks
    assert (
        minimal_dev_eval._maybe_run_periodic_dev_eval_and_checkpoint_guard
        is checkpoint_periodic_dev_eval.maybe_run_periodic_dev_eval_and_checkpoint_guard
    )
    assert checkpoint_periodic_dev_eval.maybe_run_periodic_dev_eval_and_checkpoint_guard.__module__ == (
        "weiss_rl.training.checkpoint_periodic_dev_eval"
    )


def test_checkpoint_periodic_dev_eval_reexports_guard_application_boundary() -> None:
    assert (
        checkpoint_periodic_dev_eval.PeriodicDevEvalEffectiveSummary
        is checkpoint_periodic_dev_eval_confirmatory.PeriodicDevEvalEffectiveSummary
    )
    assert (
        checkpoint_periodic_dev_eval.checkpoint_tracker_best_record
        is checkpoint_periodic_dev_eval_confirmatory.checkpoint_tracker_best_record
    )
    assert (
        checkpoint_periodic_dev_eval.maybe_run_confirmatory_dev_eval
        is checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval
    )
    assert (
        checkpoint_periodic_dev_eval.CheckpointGuardApplicationResult
        is checkpoint_periodic_dev_eval_guard.CheckpointGuardApplicationResult
    )
    assert (
        checkpoint_periodic_dev_eval.apply_periodic_dev_eval_checkpoint_guard
        is checkpoint_periodic_dev_eval_guard.apply_periodic_dev_eval_checkpoint_guard
    )
    assert checkpoint_periodic_dev_eval_guard.apply_periodic_dev_eval_checkpoint_guard.__module__ == (
        "weiss_rl.training.checkpoint_periodic_dev_eval_guard"
    )
    assert checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval.__module__ == (
        "weiss_rl.training.checkpoint_periodic_dev_eval_confirmatory"
    )


def test_minimal_finalization_reexports_checkpoint_finalization_boundary() -> None:
    import weiss_rl.training.minimal_finalization as minimal_finalization

    assert minimal_finalization.TrainingFinalCheckpointHooks is checkpoint_finalization.TrainingFinalCheckpointHooks
    assert (
        minimal_finalization._final_dev_eval_summary_for_update
        is checkpoint_finalization.final_dev_eval_summary_for_update
    )
    assert (
        minimal_finalization._finalize_training_checkpoint_selection
        is checkpoint_finalization.finalize_training_checkpoint_selection
    )
    assert checkpoint_finalization.finalize_training_checkpoint_selection.__module__ == (
        "weiss_rl.training.checkpoint_finalization"
    )


def test_minimal_setup_reexports_canonical_training_setup_boundary() -> None:
    import weiss_rl.training.minimal_initialization as minimal_initialization
    import weiss_rl.training.minimal_setup as minimal_setup

    assert minimal_setup.MinimalTrainingSetup is training_setup.MinimalTrainingSetup
    assert minimal_setup.MinimalTrainingSetupHooks is training_setup.MinimalTrainingSetupHooks
    assert minimal_setup.build_minimal_training_setup is training_setup.build_minimal_training_setup
    assert minimal_setup._require_training_stack_components is training_setup.require_training_stack_components
    assert (
        minimal_initialization._publish_initial_runtime_snapshot_after_resume
        is training_setup.publish_initial_runtime_snapshot_after_resume
    )
    assert (
        minimal_initialization._effective_init_schedule_offset_from_checkpoint
        is training_setup.effective_init_schedule_offset_from_checkpoint
    )
    assert minimal_initialization._infer_init_schedule_offset_from_scalars is (
        training_setup.infer_init_schedule_offset_from_scalars
    )
    assert training_setup.build_minimal_training_setup.__module__ == "weiss_rl.training.training_setup"


def test_minimal_update_reexports_canonical_training_update_boundary() -> None:
    import weiss_rl.training.minimal_update as minimal_update

    assert minimal_update.TrainingReplayStates is training_update.TrainingReplayStates
    assert (
        minimal_update._POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
        is training_update.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES
    )
    assert minimal_update.RuntimeTrainingBatchResult is training_update.RuntimeTrainingBatchResult
    assert minimal_update.TrainingUpdateCompletionMetrics is training_update.TrainingUpdateCompletionMetrics
    assert minimal_update.training_replay_states_from_config is training_update.training_replay_states_from_config
    assert (
        minimal_update._schedule_update_count_for_next_update is training_update.schedule_update_count_for_next_update
    )
    assert (
        minimal_update._merge_post_update_auxiliary_metrics_into_training_log
        is training_update.merge_post_update_auxiliary_metrics_into_training_log
    )
    assert (
        minimal_update._reset_policy_anchor_for_fresh_preference_replay
        is training_update.reset_policy_anchor_for_fresh_preference_replay
    )
    assert minimal_update._run_post_update_replay is training_update.run_post_update_replay
    assert minimal_update._run_training_update_step is training_update.run_training_update_step
    assert training_update.run_training_update_step.__module__ == "weiss_rl.training.training_update"


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


def test_training_replay_path_builder_preserves_order_and_injected_runners() -> None:
    def trajectory_bc_runner(**_kwargs: object) -> None:
        return None

    def paired_swing_runner(**_kwargs: object) -> None:
        return None

    def paired_outcome_preference_runner(**_kwargs: object) -> None:
        return None

    paths = training_replay_paths.build_post_update_replay_paths(
        trajectory_bc_runner=trajectory_bc_runner,
        paired_swing_runner=paired_swing_runner,
        paired_outcome_preference_runner=paired_outcome_preference_runner,
    )

    assert [path.runner for path in paths] == [
        trajectory_bc_runner,
        paired_swing_runner,
        paired_outcome_preference_runner,
    ]
    assert training_replay_paths.post_update_replay_path_specs(paths) == (
        ("trajectory_bc_replay", "trajectory_bc", "maybe_run_trajectory_bc_replay", True),
        ("paired_swing_replay", "paired_swing", "maybe_run_paired_swing_replay", False),
        (
            "paired_outcome_preference_replay",
            "paired_outcome_preference",
            "maybe_run_paired_outcome_preference_replay",
            False,
        ),
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
        "weiss_rl.training.training_update_step"
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
        "weiss_rl.training.training_update_stage_pipeline"
    )


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


def test_minimal_runner_reexports_canonical_training_runner_boundary() -> None:
    import weiss_rl.training.minimal_runner as minimal_runner

    assert minimal_runner.MinimalTrainingRunHooks is training_runner.MinimalTrainingRunHooks
    assert minimal_runner.run_minimal_training_updates is training_runner.run_minimal_training_updates
    assert training_runner.run_minimal_training_updates.__module__ == "weiss_rl.training.training_runner"


def test_training_runner_uses_canonical_context_builder_boundary() -> None:
    assert training_runner.build_training_run_contexts is training_run_contexts.build_training_run_contexts
    assert training_run_contexts.build_training_run_contexts.__module__ == "weiss_rl.training.training_run_contexts"


def test_training_run_context_builder_preserves_update_and_checkpoint_payloads(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=0)
    model = object()
    runtime = object()
    training_config = object()
    rewards_config = object()
    training_paths = object()
    replay_states = object()
    checkpoint_promotion_hooks = object()
    periodic_dev_eval_hooks = object()
    final_checkpoint_hooks = object()

    stack = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(system=SimpleNamespace(learner_torch_threads=7)),
    )
    setup = SimpleNamespace(
        learner=learner,
        model=model,
        runtime=runtime,
        training_config=training_config,
        rewards_config=rewards_config,
        training_paths=training_paths,
        algorithm="impala",
        latest_metrics={"setup": 1.0},
        init_schedule_offset_updates=11,
        resume_state={"checkpoint": "resume"},
        config_hash256="config-hash",
    )
    hooks = SimpleNamespace(
        central_runtime_actor_torch_threads=lambda received_stack, received_runtime: (
            events.append(("actor_threads", {"stack": received_stack, "runtime": received_runtime})) or 3
        ),
        apply_guidance_schedule_for_next_update=object(),
        entropy_coef_for_next_update=object(),
        collect_training_batch=object(),
        profile_block=object(),
        torch_num_threads_scope=object(),
        checkpoint_promotion=checkpoint_promotion_hooks,
        periodic_dev_eval=periodic_dev_eval_hooks,
        final_checkpoint=final_checkpoint_hooks,
    )

    def replay_states_from_config(received_training_config: object, *, repo_root: Path) -> object:
        events.append(("replay_states", {"training_config": received_training_config, "repo_root": repo_root}))
        return replay_states

    def reset_policy_anchor(**kwargs: object) -> None:
        events.append(("reset_anchor", kwargs))

    checkpoint_fn = object()
    dev_eval_fn = object()
    finalize_fn = object()
    contexts = training_run_contexts.build_training_run_contexts(
        stack=stack,
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        setup=setup,
        profile_timers=True,
        device=object(),
        checkpoint_interval_updates=5,
        run_id256="run-id",
        spec_hash256="spec-hash",
        tensorboard_logger=None,
        hooks=hooks,
        replay_states_from_config=replay_states_from_config,
        reset_policy_anchor=reset_policy_anchor,
        checkpoint_fn=checkpoint_fn,
        dev_eval_fn=dev_eval_fn,
        finalize_fn=finalize_fn,
    )

    assert [event[0] for event in events] == ["actor_threads", "replay_states", "reset_anchor"]
    assert events[0][1] == {"stack": stack, "runtime": runtime}
    assert events[1][1] == {"training_config": training_config, "repo_root": tmp_path}
    assert events[2][1] == {
        "learner": learner,
        "replay_states": replay_states,
        "resume_state": {"checkpoint": "resume"},
    }
    assert contexts.progress.latest_metrics is setup.latest_metrics
    assert contexts.update_inputs.learner is learner
    assert contexts.update_inputs.model is model
    assert contexts.update_inputs.runtime is runtime
    assert contexts.update_inputs.algorithm == "impala"
    assert contexts.update_inputs.training_config is training_config
    assert contexts.update_inputs.rewards_config is rewards_config
    assert contexts.update_inputs.replay_states is replay_states
    assert contexts.update_inputs.init_schedule_offset_updates == 11
    assert contexts.update_options.profile_timers is True
    assert contexts.update_options.actor_torch_threads == 3
    assert contexts.update_options.learner_torch_threads == 7
    assert contexts.update_hooks.collect_training_batch is hooks.collect_training_batch
    assert contexts.post_update_context.training_paths is training_paths
    assert contexts.post_update_context.config_hash256 == "config-hash"
    assert contexts.post_update_schedule.checkpoint_interval_updates == 5
    assert contexts.post_update_hooks.checkpoint_hooks is checkpoint_promotion_hooks
    assert contexts.post_update_hooks.periodic_dev_eval_hooks is periodic_dev_eval_hooks
    assert contexts.post_update_hooks.checkpoint_fn is checkpoint_fn
    assert contexts.post_update_hooks.dev_eval_fn is dev_eval_fn
    assert contexts.final_checkpoint_context.spec_hash256 == "spec-hash"
    assert contexts.final_checkpoint_hooks.hooks is final_checkpoint_hooks
    assert contexts.final_checkpoint_hooks.finalize_fn is finalize_fn
    assert contexts.actor_torch_threads == 3
    assert contexts.learner_torch_threads == 7


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
        "weiss_rl.training.training_post_update"
    )


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


def test_require_training_stack_components_rejects_missing_core_config() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            training=object(),
            model=None,
            environment=object(),
            rewards=object(),
        )
    )

    with pytest.raises(RuntimeError, match="missing training, model, environment, or rewards"):
        _require_training_stack_components(stack)


def test_minimal_training_hook_groups_thread_canonical_training_dependencies() -> None:
    names = (
        "spec_dimensions",
        "training_paths",
        "validate_algorithm_model_contract",
        "build_policy_value_model",
        "maybe_compile_learner_model",
        "build_training_learner",
        "restore_learner_from_checkpoint",
        "initialize_learner_from_checkpoint",
        "compute_config_hash256",
        "ensure_noleague_baseline_anchor",
        "import_seed_snapshot_pool",
        "canonical_config_dict",
        "build_runtime_config",
        "queue_runtime_cls",
        "central_runtime_actor_torch_threads",
        "build_training_profiler",
        "run_structured_warmstart",
        "profile_block",
        "apply_guidance_schedule_for_next_update",
        "entropy_coef_for_next_update",
        "torch_num_threads_scope",
        "collect_training_batch",
        "write_scalars_record",
        "write_checkpoint",
        "publish_checkpoint_aliases",
        "maybe_log_structured_mainmove_guard",
        "persist_snapshot_registry_entry",
        "run_snapshot_promotion_gate",
        "should_run_periodic_dev_eval",
        "run_periodic_dev_eval",
        "slug_policy_id",
        "load_checkpoint_tracker",
        "confirmatory_dev_eval_request",
        "periodic_dev_eval_schedule",
        "expand_periodic_dev_eval_paired_seeds",
        "ensure_current_checkpoint",
        "maybe_rollback_to_best_checkpoint",
        "maybe_finalize_from_best_checkpoint",
    )
    values = {name: object() for name in names}
    groups = minimal_training_hook_groups(SimpleNamespace(**values))

    assert groups.setup.spec_dimensions is values["spec_dimensions"]
    assert groups.setup.training_paths is values["training_paths"]
    assert groups.setup.queue_runtime_cls is values["queue_runtime_cls"]
    assert groups.setup.import_seed_snapshot_pool is values["import_seed_snapshot_pool"]
    assert groups.checkpoint_promotion.write_checkpoint is values["write_checkpoint"]
    assert groups.checkpoint_promotion.publish_checkpoint_aliases is values["publish_checkpoint_aliases"]
    assert groups.checkpoint_promotion.run_snapshot_promotion_gate is values["run_snapshot_promotion_gate"]
    assert groups.periodic_dev_eval.should_run_periodic_dev_eval is values["should_run_periodic_dev_eval"]
    assert groups.periodic_dev_eval.run_periodic_dev_eval is values["run_periodic_dev_eval"]
    assert groups.periodic_dev_eval.maybe_rollback_to_best_checkpoint is values["maybe_rollback_to_best_checkpoint"]
    assert groups.final_checkpoint.ensure_current_checkpoint is values["ensure_current_checkpoint"]
    assert groups.final_checkpoint.maybe_finalize_from_best_checkpoint is values["maybe_finalize_from_best_checkpoint"]
    assert groups.run.build_training_profiler is values["build_training_profiler"]
    assert groups.run.collect_training_batch is values["collect_training_batch"]
    assert groups.run.write_scalars_record is values["write_scalars_record"]
    assert groups.run.checkpoint_promotion is groups.checkpoint_promotion
    assert groups.run.periodic_dev_eval is groups.periodic_dev_eval
    assert groups.run.final_checkpoint is groups.final_checkpoint


def test_build_minimal_training_setup_restores_offset_and_builds_runtime(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    scalars_path = tmp_path / "training_metrics.jsonl"
    scalars_path.write_text(
        '{"update_count": 3, "init_schedule_offset_updates": 17}\n',
        encoding="utf-8",
    )
    training_paths = SimpleNamespace(
        scalars_path=scalars_path,
        performance_log_path=tmp_path / "performance.jsonl",
    )
    training_config = SimpleNamespace(algorithm=" ppo ")
    model_config = SimpleNamespace(recurrent_core="lstm", encoder_kind="typed")
    rewards_config = object()
    stack = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(
            training=training_config,
            model=model_config,
            environment=object(),
            rewards=rewards_config,
        ),
    )
    contract = SimpleNamespace(
        spec_bundle={
            "action": {"pass_action_id": 8},
            "observation": {"shape": [3]},
        }
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    runtime_mode = object()
    b1_baseline_run_dir = tmp_path / "b1"
    seed_snapshot_run_dir = tmp_path / "seed_source"
    device = object()

    class FakeModel:
        def __init__(self) -> None:
            self.device: object | None = None

        def to(self, target_device: object) -> FakeModel:
            calls.append(("model_to", {"device": target_device}))
            self.device = target_device
            return self

        def state_dict(self) -> dict[str, object]:
            calls.append(("state_dict", {}))
            return {"weight": 1}

    class FakeLearner:
        def __init__(self, model: FakeModel) -> None:
            self.model = model
            self.update_count = 0
            self.init_schedule_offset_updates = -1

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("runtime", kwargs))
            self.kwargs = kwargs

        def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
            calls.append(("snapshot", kwargs))
            return {"snapshot_publish_latency_ms": 1.0}

    model = FakeModel()
    learner = FakeLearner(model)
    compiled_model = object()
    runtime_config = object()
    resume_state = SimpleNamespace(
        checkpoint_path=tmp_path / "resume.pt",
        update_count=5,
        policy_version=7,
        init_schedule_offset_updates=0,
    )

    def spec_dimensions(received_contract: object) -> tuple[int, int]:
        calls.append(("spec_dimensions", {"contract": received_contract}))
        return 3, 9

    def training_paths_fn(run_dir: Path) -> object:
        calls.append(("training_paths", {"run_dir": run_dir}))
        return training_paths

    def validate_algorithm_model_contract(**kwargs: object) -> None:
        calls.append(("validate", kwargs))

    def build_policy_value_model(**kwargs: object) -> FakeModel:
        calls.append(("build_model", kwargs))
        return model

    def maybe_compile_learner_model(**kwargs: object) -> object:
        calls.append(("compile", kwargs))
        return compiled_model

    def build_training_learner(**kwargs: object) -> FakeLearner:
        calls.append(("build_learner", kwargs))
        return learner

    def restore_learner_from_checkpoint(**kwargs: object) -> object:
        calls.append(("restore", kwargs))
        learner.update_count = int(resume_state.update_count)
        return resume_state

    def fail_initialize(**_kwargs: object) -> object:
        raise AssertionError("init-from-checkpoint should not run in this setup")

    def compute_config_hash256(received_stack: object) -> str:
        calls.append(("config_hash", {"stack": received_stack}))
        return "config-hash"

    def ensure_noleague_baseline_anchor(**kwargs: object) -> None:
        calls.append(("baseline", kwargs))

    def import_seed_snapshot_pool(**kwargs: object) -> None:
        calls.append(("seed_snapshot", kwargs))

    def canonical_config_dict(received_stack: object) -> dict[str, object]:
        calls.append(("canonical", {"stack": received_stack}))
        return {"config": "canonical"}

    def build_runtime_config(**kwargs: object) -> object:
        calls.append(("runtime_config", kwargs))
        return runtime_config

    setup = build_minimal_training_setup(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        num_envs=4,
        unroll_length=8,
        profile="fast",
        device=device,
        seed=99,
        checkpoint_interval_updates=5,
        spec_hash256="spec-hash",
        runtime_mode=runtime_mode,
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
        hooks=MinimalTrainingSetupHooks(
            spec_dimensions=spec_dimensions,
            training_paths=training_paths_fn,
            validate_algorithm_model_contract=validate_algorithm_model_contract,
            build_policy_value_model=build_policy_value_model,
            maybe_compile_learner_model=maybe_compile_learner_model,
            build_training_learner=build_training_learner,
            restore_learner_from_checkpoint=restore_learner_from_checkpoint,
            initialize_learner_from_checkpoint=fail_initialize,
            compute_config_hash256=compute_config_hash256,
            ensure_noleague_baseline_anchor=ensure_noleague_baseline_anchor,
            import_seed_snapshot_pool=import_seed_snapshot_pool,
            canonical_config_dict=canonical_config_dict,
            build_runtime_config=build_runtime_config,
            queue_runtime_cls=FakeRuntime,
        ),
    )

    assert setup.observation_dim == 3
    assert setup.action_dim == 9
    assert setup.training_config is training_config
    assert setup.rewards_config is rewards_config
    assert setup.training_paths is training_paths
    assert setup.pass_action_id == 8
    assert setup.algorithm == "ppo"
    assert setup.model is model
    assert setup.learner is learner
    assert isinstance(setup.runtime, FakeRuntime)
    assert setup.latest_metrics == {"snapshot_publish_latency_ms": 1.0}
    assert setup.init_schedule_offset_updates == 17
    assert setup.resume_state is resume_state
    assert setup.config_hash256 == "config-hash"
    assert learner.init_schedule_offset_updates == 17
    assert [call[0] for call in calls] == [
        "spec_dimensions",
        "training_paths",
        "validate",
        "build_model",
        "model_to",
        "compile",
        "build_learner",
        "restore",
        "config_hash",
        "baseline",
        "state_dict",
        "canonical",
        "seed_snapshot",
        "runtime_config",
        "runtime",
        "snapshot",
    ]
    assert calls[2][1] == {"algorithm": "ppo", "recurrent_core": "lstm", "encoder_kind": "typed"}
    assert calls[3][1]["observation_dim"] == 3
    assert calls[3][1]["observation_spec"] == {"shape": [3]}
    assert calls[6][1]["compiled_model"] is compiled_model
    assert calls[6][1]["pass_action_id"] == 8
    assert calls[7][1]["expected_spec_hash256"] == "spec-hash"
    assert calls[9][1]["baseline_run_dir"] == b1_baseline_run_dir
    assert calls[12][1]["expected_model_state_dict"] == {"weight": 1}
    assert calls[12][1]["expected_config_canonical"] == {"config": "canonical"}
    assert calls[13][1] == {
        "stack": stack,
        "num_envs": 4,
        "unroll_length": 8,
        "profile": "fast",
        "seed": 99,
        "pass_action_id": 8,
        "runtime_mode": runtime_mode,
    }
    assert calls[14][1]["config"] is runtime_config
    assert calls[14][1]["performance_log_path"] == training_paths.performance_log_path
    assert calls[15][1] == {"learner_model": model, "learner_update_count": 5, "force": True}


def test_build_minimal_training_setup_init_checkpoint_override_sets_schedule_offset(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    training_paths = SimpleNamespace(
        scalars_path=tmp_path / "training_metrics.jsonl",
        performance_log_path=tmp_path / "performance.jsonl",
    )
    model = SimpleNamespace(to=lambda _device: model, state_dict=lambda: {"weight": 1})
    learner = SimpleNamespace(model=model, update_count=0, init_schedule_offset_updates=-1)
    stack = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(
            training=SimpleNamespace(algorithm="ppo"),
            model=SimpleNamespace(recurrent_core="none", encoder_kind="flat"),
            environment=object(),
            rewards=object(),
        ),
    )
    contract = SimpleNamespace(spec_bundle={"action": {"pass_action_id": 8}})
    init_state = SimpleNamespace(
        checkpoint_path=tmp_path / "init.pt",
        update_count=20,
        init_schedule_offset_updates=30,
        policy_version=4,
    )

    class FakeRuntime:
        def __init__(self, **_kwargs: object) -> None:
            pass

    setup = build_minimal_training_setup(
        stack=stack,
        contract=contract,
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        num_envs=1,
        unroll_length=1,
        profile="default",
        device=object(),
        seed=1,
        checkpoint_interval_updates=2,
        spec_hash256="spec",
        runtime_mode=object(),
        b1_baseline_run_dir=None,
        seed_snapshot_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=tmp_path / "init.pt",
        init_schedule_offset_override_updates=12,
        hooks=MinimalTrainingSetupHooks(
            spec_dimensions=lambda _contract: (3, 9),
            training_paths=lambda _run_dir: training_paths,
            validate_algorithm_model_contract=lambda **kwargs: calls.append(("validate", kwargs)),
            build_policy_value_model=lambda **kwargs: model,
            maybe_compile_learner_model=lambda **kwargs: object(),
            build_training_learner=lambda **kwargs: learner,
            restore_learner_from_checkpoint=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("resume restore should not run")
            ),
            initialize_learner_from_checkpoint=lambda **kwargs: calls.append(("init", kwargs)) or init_state,
            compute_config_hash256=lambda _stack: "config",
            ensure_noleague_baseline_anchor=lambda **kwargs: calls.append(("baseline", kwargs)),
            import_seed_snapshot_pool=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("seed snapshot import should not run")
            ),
            canonical_config_dict=lambda _stack: {},
            build_runtime_config=lambda **kwargs: object(),
            queue_runtime_cls=FakeRuntime,
        ),
    )

    assert setup.init_schedule_offset_updates == 12
    assert learner.init_schedule_offset_updates == 12
    assert calls[1][0] == "init"
    assert calls[1][1]["checkpoint_path"] == tmp_path / "init.pt"


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


def test_publish_initial_runtime_snapshot_after_resume_forces_current_update() -> None:
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        calls.append(dict(kwargs))
        return {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}

    runtime.maybe_publish_snapshot = _publish_snapshot
    model = object()

    metrics = _publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=model, update_count=25)

    assert metrics == {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}
    assert calls == [{"learner_model": model, "learner_update_count": 25, "force": True}]


def test_publish_initial_runtime_snapshot_after_resume_skips_zero_update() -> None:
    runtime = SimpleNamespace()

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        raise AssertionError("zero-update fresh runs must not publish a resume snapshot")

    runtime.maybe_publish_snapshot = _publish_snapshot

    metrics = _publish_initial_runtime_snapshot_after_resume(runtime=runtime, model=object(), update_count=0)

    assert metrics == {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}


def test_schedule_update_count_for_next_update_preserves_source_checkpoint_time() -> None:
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=0,
            init_schedule_offset_updates=90,
        )
        == 91
    )
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        _schedule_update_count_for_next_update(
            learner_update_count=24,
            init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_preserves_nested_warmstart_time() -> None:
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
        )
        == 115
    )
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=0,
        )
        == 25
    )


def test_effective_init_schedule_offset_from_checkpoint_allows_explicit_override() -> None:
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
            override_updates=0,
        )
        == 0
    )
    assert (
        _effective_init_schedule_offset_from_checkpoint(
            source_update_count=25,
            source_init_schedule_offset_updates=90,
            override_updates=12,
        )
        == 12
    )


def test_infer_init_schedule_offset_from_scalars_recovers_latest_offset(tmp_path: Path) -> None:
    scalars_path = tmp_path / "scalars.jsonl"
    scalars_path.write_text(
        "\n".join(
            (
                '{"update_count": 1, "init_schedule_offset_updates": 90}',
                '{"update_count": 2}',
                '{"update_count": 3, "init_schedule_offset_updates": 90.0}',
            )
        ),
        encoding="utf-8",
    )

    assert _infer_init_schedule_offset_from_scalars(scalars_path) == 90


def test_merge_post_update_auxiliary_metrics_into_training_log_uses_latest_learner_record() -> None:
    calls: list[dict[str, object]] = []
    logger = SimpleNamespace(
        merge_latest_custom_metrics=lambda **kwargs: calls.append(dict(kwargs)),
    )
    learner = SimpleNamespace(
        logger=logger,
        update_count=7,
        get_policy_version=lambda: 3,
    )
    metrics = {"paired_swing_replay_loss": 0.25}

    _merge_post_update_auxiliary_metrics_into_training_log(learner=learner, metrics=metrics)

    assert calls == [
        {
            "update_count": 7,
            "policy_version": 3,
            "metrics": metrics,
            "prefixes": _POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
        }
    ]
    assert "pfsp_" in calls[0]["prefixes"]
    assert "collector_pfsp_" in calls[0]["prefixes"]


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

    schedule = training_update_phases.apply_training_update_schedule(
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

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, *_exc: object) -> None:
            events.append(("exit", self.name))

    class FakeRuntime:
        def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
            events.append(("snapshot", kwargs))
            return {"snapshot_metric": 4.0}

    latest_metrics = {"loss": 3.0}
    completed = training_update_phases.complete_training_update_metrics(
        learner=learner,
        model=model,
        runtime=FakeRuntime(),
        latest_metrics=latest_metrics,
        runtime_metrics={"runtime_metric": 2.0},
        schedule_metrics=schedule.metrics,
        profile_timers=True,
        profile_block=lambda enabled, name: events.append(("profile", enabled, name)) or FakeScope(name),
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


def test_training_update_batch_helpers_preserve_profile_and_thread_scopes() -> None:
    events: list[tuple[object, ...]] = []
    runtime = object()
    training_config = object()
    rewards_config = object()
    learner_batch = object()
    runtime_batch = SimpleNamespace(learner_batch=learner_batch, runtime_metrics={"runtime": 1.0})

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, *_exc: object) -> None:
            events.append(("exit", self.name))

    class FakeLearner:
        def update(self, received_batch: object) -> dict[str, float]:
            events.append(("learner_update", received_batch))
            return {"loss": 2.0}

    def profile_block(enabled: bool, name: str) -> FakeScope:
        events.append(("profile", enabled, name))
        return FakeScope(f"profile:{name}")

    def torch_num_threads_scope(thread_count: int | None) -> FakeScope:
        events.append(("threads", thread_count))
        return FakeScope(f"threads:{thread_count}")

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


def test_training_update_completion_metrics_preserve_merge_precedence() -> None:
    latest_metrics = {"shared": 0.0, "loss": 1.0}
    completion = training_update_phases.TrainingUpdateCompletionMetrics(
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

    class FakeScope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, *_exc: object) -> None:
            events.append(("exit", self.name))

    class FakeRuntime:
        def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
            events.append(("snapshot", kwargs))
            return {"snapshot_metric": 3.0}

    completion = training_update_phases.collect_training_update_completion_metrics(
        learner=learner,
        model=model,
        runtime=FakeRuntime(),
        runtime_metrics={"runtime_metric": 1.0},
        schedule_metrics={"schedule_metric": 2.0},
        profile_timers=True,
        profile_block=lambda enabled, name: events.append(("profile", enabled, name)) or FakeScope(name),
    )

    assert completion == training_update_phases.TrainingUpdateCompletionMetrics(
        runtime={"runtime_metric": 1.0},
        schedule={"schedule_metric": 2.0},
        snapshot={"snapshot_metric": 3.0},
    )
    assert [event[0] for event in events] == ["profile", "enter", "snapshot", "exit"]
    assert events[2][1] == {"learner_model": model, "learner_update_count": 12}


def test_fresh_preference_replay_resets_policy_anchor_once() -> None:
    calls: list[dict[str, object]] = []
    learner = SimpleNamespace(reset_policy_anchor_to_current_model=lambda **kwargs: calls.append(dict(kwargs)))
    replay_states = TrainingReplayStates(
        trajectory_bc=None,
        paired_swing=None,
        paired_outcome_preference=object(),
    )

    _reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=replay_states,
        resume_state=None,
    )

    assert calls == [{"force": True}]


def test_preference_replay_anchor_reset_skips_resume_and_disabled_replay() -> None:
    learner = SimpleNamespace(
        reset_policy_anchor_to_current_model=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("policy anchor should only reset for fresh preference replay")
        )
    )

    _reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=TrainingReplayStates(
            trajectory_bc=None,
            paired_swing=None,
            paired_outcome_preference=None,
        ),
        resume_state=None,
    )
    _reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=TrainingReplayStates(
            trajectory_bc=None,
            paired_swing=None,
            paired_outcome_preference=object(),
        ),
        resume_state={},
    )


def test_fresh_preference_replay_requires_policy_anchor_support() -> None:
    with pytest.raises(ValueError, match="policy-anchor support"):
        _reset_policy_anchor_for_fresh_preference_replay(
            learner=object(),
            replay_states=TrainingReplayStates(
                trajectory_bc=None,
                paired_swing=None,
                paired_outcome_preference=object(),
            ),
            resume_state=None,
        )


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

    _run_post_update_replay(
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

    latest_metrics = _run_training_update_step(
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


def test_league_reference_update_from_metrics_uses_effective_update_when_present() -> None:
    assert _league_reference_update_from_metrics({}) is None
    assert _league_reference_update_from_metrics({"league_effective_update": 42.0}) == 42


def test_checkpoint_promotion_skips_non_interval_update(tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_hook(**_kwargs: object) -> object:
        calls.append("unexpected")
        raise AssertionError("checkpoint hooks must not run outside the interval")

    runtime = SimpleNamespace(refresh_opponent_pool=fail_hook)
    learner = SimpleNamespace(update_count=5, model=object(), get_policy_version=lambda: 9)
    hooks = TrainingCheckpointPromotionHooks(
        write_checkpoint=fail_hook,
        publish_checkpoint_aliases=fail_hook,
        maybe_log_structured_mainmove_guard=fail_hook,
        persist_snapshot_registry_entry=fail_hook,
        run_snapshot_promotion_gate=fail_hook,
    )

    tracker_payload = _maybe_checkpoint_and_promote_snapshot(
        learner=learner,
        stack=object(),
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
        runtime=runtime,
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=None,
        checkpoint_interval_updates=3,
        run_id256="run-id",
        config_hash256="config",
        tensorboard_logger=None,
        hooks=hooks,
    )

    assert tracker_payload is None
    assert calls == []


def test_checkpoint_promotion_writes_aliases_registry_and_refreshes_on_promotion(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    checkpoint_dir = tmp_path / "checkpoints"
    training_paths = SimpleNamespace(checkpoints_dir=checkpoint_dir)
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    stack = object()
    contract = object()
    device = object()
    algorithm = object()
    latest_metrics = {"loss": 1.0, "league_effective_update": 42.0}
    dev_eval_summary = {"aggregate_score": 0.75}

    class FakeModel:
        def state_dict(self) -> dict[str, object]:
            calls.append(("state_dict", {}))
            return {"weight": 1}

    learner = SimpleNamespace(update_count=6, model=FakeModel(), get_policy_version=lambda: 11)

    class FakeRuntime:
        def refresh_opponent_pool(self) -> None:
            calls.append(("refresh", {}))

    class FakeTensorBoardLogger:
        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            calls.append(("tensorboard", {"payload": payload, "step": step}))

    def write_checkpoint(**kwargs: object) -> None:
        calls.append(("write", kwargs))

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        calls.append(("aliases", kwargs))
        return {"best": {"update": 6}}

    def maybe_log_structured_mainmove_guard(**kwargs: object) -> None:
        calls.append(("guard", kwargs))

    def persist_snapshot_registry_entry(**kwargs: object) -> str:
        calls.append(("persist", kwargs))
        return "candidate_policy"

    def run_snapshot_promotion_gate(**kwargs: object) -> bool:
        calls.append(("promotion", kwargs))
        return True

    tracker_payload = _maybe_checkpoint_and_promote_snapshot(
        learner=learner,
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=FakeRuntime(),
        device=device,
        spec_hash256="spec-hash",
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        last_dev_eval_summary=dev_eval_summary,
        checkpoint_interval_updates=3,
        run_id256="run-id",
        config_hash256="config-hash",
        tensorboard_logger=FakeTensorBoardLogger(),
        hooks=TrainingCheckpointPromotionHooks(
            write_checkpoint=write_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_log_structured_mainmove_guard=maybe_log_structured_mainmove_guard,
            persist_snapshot_registry_entry=persist_snapshot_registry_entry,
            run_snapshot_promotion_gate=run_snapshot_promotion_gate,
        ),
    )

    assert tracker_payload == {"best": {"update": 6}}
    assert [call[0] for call in calls] == [
        "write",
        "aliases",
        "guard",
        "tensorboard",
        "state_dict",
        "persist",
        "refresh",
        "promotion",
        "refresh",
    ]
    checkpoint_path = checkpoint_dir / "checkpoint_6.pt"
    assert calls[0][1] == {
        "checkpoint_path": checkpoint_path,
        "learner": learner,
        "stack": stack,
        "device": device,
        "spec_hash256": "spec-hash",
        "algorithm": algorithm,
    }
    assert calls[1][1]["checkpoint_path"] == checkpoint_path
    assert calls[1][1]["latest_metrics"] is latest_metrics
    assert calls[2][1]["dev_eval_summary"] is dev_eval_summary
    assert calls[3][1] == {"payload": tracker_payload, "step": 6}
    assert calls[5][1]["model_state_dict"] == {"weight": 1}
    assert calls[5][1]["policy_version"] == 11
    assert calls[7][1]["candidate_policy_id"] == "candidate_policy"
    assert calls[7][1]["league_reference_update"] == 42
    assert calls[7][1]["policy_version"] == 11
    assert calls[7][1]["run_id256"] == "run-id"
    assert calls[7][1]["config_hash256"] == "config-hash"
    assert calls[7][1]["spec_hash256"] == "spec-hash"


def test_checkpoint_promotion_refreshes_once_when_gate_does_not_promote(tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeModel:
        def state_dict(self) -> dict[str, object]:
            calls.append("state_dict")
            return {"weight": 1}

    def publish_checkpoint_aliases(**_kwargs: object) -> dict[str, object]:
        calls.append("aliases")
        return {"latest": {"update": 6}}

    def persist_snapshot_registry_entry(**_kwargs: object) -> str:
        calls.append("persist")
        return "candidate_policy"

    def run_snapshot_promotion_gate(**_kwargs: object) -> bool:
        calls.append("promotion")
        return False

    class FakeRuntime:
        def refresh_opponent_pool(self) -> None:
            calls.append("refresh")

    tracker_payload = _maybe_checkpoint_and_promote_snapshot(
        learner=SimpleNamespace(update_count=6, model=FakeModel(), get_policy_version=lambda: 11),
        stack=object(),
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
        runtime=FakeRuntime(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=None,
        checkpoint_interval_updates=3,
        run_id256="run-id",
        config_hash256="config-hash",
        tensorboard_logger=None,
        hooks=TrainingCheckpointPromotionHooks(
            write_checkpoint=lambda **_kwargs: calls.append("write"),
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_log_structured_mainmove_guard=lambda **_kwargs: calls.append("guard"),
            persist_snapshot_registry_entry=persist_snapshot_registry_entry,
            run_snapshot_promotion_gate=run_snapshot_promotion_gate,
        ),
    )

    assert tracker_payload == {"latest": {"update": 6}}
    assert calls == [
        "write",
        "aliases",
        "guard",
        "state_dict",
        "persist",
        "refresh",
        "promotion",
    ]


def test_checkpoint_promotion_missing_model_fails_after_tracker_logging(tmp_path: Path) -> None:
    calls: list[str] = []
    tracker_payload = {"latest": {"update": 6}}

    class FakeTensorBoardLogger:
        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            assert payload is tracker_payload
            assert step == 6
            calls.append("tensorboard")

    with pytest.raises(RuntimeError, match="without a learner model"):
        _maybe_checkpoint_and_promote_snapshot(
            learner=SimpleNamespace(update_count=6, model=None, get_policy_version=lambda: 11),
            stack=object(),
            contract=object(),
            artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
            training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
            runtime=SimpleNamespace(refresh_opponent_pool=lambda: calls.append("refresh")),
            device=object(),
            spec_hash256="spec-hash",
            algorithm=object(),
            latest_metrics={"loss": 1.0},
            last_dev_eval_summary=None,
            checkpoint_interval_updates=3,
            run_id256="run-id",
            config_hash256="config-hash",
            tensorboard_logger=FakeTensorBoardLogger(),
            hooks=TrainingCheckpointPromotionHooks(
                write_checkpoint=lambda **_kwargs: calls.append("write"),
                publish_checkpoint_aliases=lambda **_kwargs: calls.append("aliases") or tracker_payload,
                maybe_log_structured_mainmove_guard=lambda **_kwargs: calls.append("guard"),
                persist_snapshot_registry_entry=lambda **_kwargs: calls.append("persist"),
                run_snapshot_promotion_gate=lambda **_kwargs: calls.append("promotion"),
            ),
        )

    assert calls == ["write", "aliases", "guard", "tensorboard"]


def test_periodic_dev_eval_guard_skips_when_schedule_says_no(tmp_path: Path) -> None:
    def fail_hook(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("periodic dev eval hooks must not run when the schedule is disabled")

    previous_summary = {"aggregate_score": 0.1}
    result = _maybe_run_periodic_dev_eval_and_checkpoint_guard(
        learner=SimpleNamespace(update_count=7),
        model=object(),
        stack=object(),
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
        runtime=object(),
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=previous_summary,
        last_dev_eval_update_count=3,
        last_checkpoint_guard_rollback_update=2,
        run_id256="run-id",
        config_hash256="config",
        tensorboard_logger=None,
        hooks=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: False,
            run_periodic_dev_eval=fail_hook,
            slug_policy_id=fail_hook,
            load_checkpoint_tracker=fail_hook,
            confirmatory_dev_eval_request=fail_hook,
            periodic_dev_eval_schedule=fail_hook,
            expand_periodic_dev_eval_paired_seeds=fail_hook,
            ensure_current_checkpoint=fail_hook,
            publish_checkpoint_aliases=fail_hook,
            maybe_log_structured_mainmove_guard=fail_hook,
            maybe_rollback_to_best_checkpoint=fail_hook,
        ),
    )

    assert result == PeriodicDevEvalGuardResult(
        last_dev_eval_summary=previous_summary,
        last_dev_eval_update_count=3,
        last_checkpoint_guard_rollback_update=2,
        stop_requested=False,
    )


def test_periodic_dev_eval_confirmatory_helper_skips_without_request() -> None:
    summary = {"anchor_scores": {}, "aggregate_score": 0.25}

    result = checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval(
        hooks=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: True,
            run_periodic_dev_eval=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("confirmatory eval must not run without a request")
            ),
            slug_policy_id=str,
            load_checkpoint_tracker=lambda _paths: {"best": "not-a-record"},
            confirmatory_dev_eval_request=lambda **kwargs: None,
            periodic_dev_eval_schedule=lambda _stack: (_ for _ in ()).throw(
                AssertionError("schedule should not be loaded without a request")
            ),
            expand_periodic_dev_eval_paired_seeds=lambda *_args, **_kwargs: [],
            ensure_current_checkpoint=lambda **_kwargs: Path("checkpoint.pt"),
            publish_checkpoint_aliases=lambda **_kwargs: {},
            maybe_log_structured_mainmove_guard=lambda **_kwargs: None,
            maybe_rollback_to_best_checkpoint=lambda **_kwargs: None,
        ),
        stack=object(),
        learner=SimpleNamespace(update_count=4, get_policy_version=lambda: 9),
        summary_payload=summary,
        contract=object(),
        artifacts=object(),
        training_paths=object(),
        device=object(),
        run_id256="run-id",
        config_hash256="config",
        spec_hash256="spec",
        update_count=4,
    )

    assert result == checkpoint_periodic_dev_eval_confirmatory.PeriodicDevEvalEffectiveSummary(summary=summary)
    assert checkpoint_periodic_dev_eval_confirmatory.checkpoint_tracker_best_record({"best": "not-a-record"}) is None


def test_periodic_dev_eval_confirmatory_helper_expands_and_runs_override_pairs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    summary = {"anchor_scores": {"B2 HeuristicPublic": 0.25}, "aggregate_score": 0.25}
    effective_summary = {"anchor_scores": {"B2 HeuristicPublic": 0.5}, "aggregate_score": 0.5}
    best_record = {"policy_id": "best"}
    learner = SimpleNamespace(update_count=4, get_policy_version=lambda: 9)

    def confirmatory_request(**kwargs: object) -> dict[str, object]:
        events.append(("request", kwargs))
        return {"target_pairs": 3, "reasons": ["wide_ci"]}

    def run_eval(**kwargs: object) -> dict[str, object]:
        events.append(("run", kwargs))
        return effective_summary

    result = checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval(
        hooks=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: True,
            run_periodic_dev_eval=run_eval,
            slug_policy_id=str,
            load_checkpoint_tracker=lambda _paths: {"best": best_record},
            confirmatory_dev_eval_request=confirmatory_request,
            periodic_dev_eval_schedule=lambda _stack: (SimpleNamespace(name="dev_seeds.txt"), {}, [10, 20], "seed-sha"),
            expand_periodic_dev_eval_paired_seeds=lambda *args, **kwargs: (
                events.append(("expand", {"args": args, "kwargs": kwargs})) or ["pair-a", "pair-b", "pair-c"]
            ),
            ensure_current_checkpoint=lambda **_kwargs: Path("checkpoint.pt"),
            publish_checkpoint_aliases=lambda **_kwargs: {},
            maybe_log_structured_mainmove_guard=lambda **_kwargs: None,
            maybe_rollback_to_best_checkpoint=lambda **_kwargs: None,
        ),
        stack=object(),
        learner=learner,
        summary_payload=summary,
        contract=object(),
        artifacts=object(),
        training_paths=object(),
        device=object(),
        run_id256="run-id",
        config_hash256="config",
        spec_hash256="spec",
        update_count=4,
    )

    assert result.summary is effective_summary
    assert result.confirmatory_request == {"target_pairs": 3, "reasons": ["wide_ci"]}
    assert result.confirmatory_pair_count == 3
    assert events[0][1]["existing_best_record"] is best_record
    assert events[1][1]["kwargs"] == {
        "requested_pairs": 3,
        "seed_file_sha256": "seed-sha",
        "update_count": 4,
        "policy_version": 9,
        "scope": "periodic_dev_eval_confirmatory",
    }
    assert events[2][1]["artifact_scope"] == "periodic_dev_eval_confirmatory"
    assert events[2][1]["paired_seeds_override"] == ["pair-a", "pair-b", "pair-c"]
    stdout = capsys.readouterr().out
    assert "Confirmatory dev eval: update=4 paired_seeds=3 aggregate=0.5000 reasons=wide_ci" in stdout


def test_periodic_dev_eval_checkpoint_guard_helper_keeps_state_without_rollback() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    latest_metrics = {"loss": 1.0}
    effective_summary = {"anchor_scores": {}, "aggregate_score": 0.3}
    tracker_payload = {"best": {"update": 5}}

    class FakeTensorBoardLogger:
        def log_periodic_dev_eval(self, payload: object, *, step: int) -> None:
            events.append(("tb_eval", {"payload": payload, "step": step}))

        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            events.append(("tb_tracker", {"payload": payload, "step": step}))

    result = checkpoint_periodic_dev_eval.apply_periodic_dev_eval_checkpoint_guard(
        hooks=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: True,
            run_periodic_dev_eval=lambda **_kwargs: {},
            slug_policy_id=str,
            load_checkpoint_tracker=lambda _paths: {},
            confirmatory_dev_eval_request=lambda **_kwargs: None,
            periodic_dev_eval_schedule=lambda _stack: None,
            expand_periodic_dev_eval_paired_seeds=lambda *_args, **_kwargs: [],
            ensure_current_checkpoint=lambda **kwargs: events.append(("ensure", kwargs)) or Path("checkpoint.pt"),
            publish_checkpoint_aliases=lambda **kwargs: events.append(("aliases", kwargs)) or tracker_payload,
            maybe_log_structured_mainmove_guard=lambda **kwargs: events.append(("guard", kwargs)),
            maybe_rollback_to_best_checkpoint=lambda **kwargs: events.append(("rollback", kwargs)) or None,
        ),
        stack=SimpleNamespace(config=SimpleNamespace(curriculum=None)),
        learner=SimpleNamespace(update_count=5),
        model=object(),
        artifacts=object(),
        training_paths=object(),
        runtime=object(),
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        latest_metrics=latest_metrics,
        effective_summary=effective_summary,
        last_checkpoint_guard_rollback_update=2,
        run_id256="run-id",
        tensorboard_logger=FakeTensorBoardLogger(),
        update_count=5,
    )

    assert result == checkpoint_periodic_dev_eval.CheckpointGuardApplicationResult(
        tracker_payload=tracker_payload,
        next_rollback_update=2,
        stop_requested=False,
    )
    assert [event[0] for event in events] == ["ensure", "aliases", "guard", "rollback", "tb_eval", "tb_tracker"]
    assert events[1][1]["dev_eval_summary"] is effective_summary
    assert events[3][1]["last_rollback_update"] == 2
    assert "checkpoint_guard_stop_after_rollback" not in latest_metrics


def test_periodic_dev_eval_guard_runs_confirmatory_eval_and_stop_after_rollback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    latest_metrics = {"loss": 1.0}
    learner = SimpleNamespace(update_count=8, get_policy_version=lambda: 13)
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(checkpoint_guard=SimpleNamespace(stop_after_rollback=True)),
        )
    )
    contract = object()
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    runtime = object()
    model = object()
    device = object()
    algorithm = object()
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_8.pt"
    summary_payload = {"anchor_scores": {"B2 HeuristicPublic": 0.1}, "aggregate_score": 0.25}
    effective_summary = {"anchor_scores": {"B2 HeuristicPublic": 0.2}, "aggregate_score": 0.5}

    class FakeTensorBoardLogger:
        def log_periodic_dev_eval(self, payload: object, *, step: int) -> None:
            events.append(("tb_eval", {"payload": payload, "step": step}))

        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            events.append(("tb_tracker", {"payload": payload, "step": step}))

    def run_periodic_dev_eval(**kwargs: object) -> dict[str, object]:
        events.append(("run_eval", kwargs))
        if kwargs.get("artifact_scope") == "periodic_dev_eval_confirmatory":
            return effective_summary
        return summary_payload

    def slug_policy_id(policy_id: str) -> str:
        events.append(("slug", {"policy_id": policy_id}))
        return "b2"

    def load_checkpoint_tracker(paths: object) -> dict[str, object]:
        events.append(("load_tracker", {"paths": paths}))
        return {"best": {"policy_id": "best"}}

    def confirmatory_dev_eval_request(**kwargs: object) -> dict[str, object]:
        events.append(("confirm_request", kwargs))
        return {"target_pairs": 3, "reasons": ["wide_ci", "new_best"]}

    def periodic_dev_eval_schedule(config_stack: object) -> tuple[object, list[object], list[int], str]:
        events.append(("schedule", {"stack": config_stack}))
        return SimpleNamespace(name="dev_seeds.txt"), [], [10, 20], "seed-sha"

    def expand_periodic_dev_eval_paired_seeds(*args: object, **kwargs: object) -> list[str]:
        events.append(("expand", {"args": args, "kwargs": kwargs}))
        return ["pair-a", "pair-b", "pair-c"]

    def ensure_current_checkpoint(**kwargs: object) -> Path:
        events.append(("ensure_checkpoint", kwargs))
        return checkpoint_path

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        events.append(("aliases", kwargs))
        return {"best": {"update": 8}}

    def maybe_log_structured_mainmove_guard(**kwargs: object) -> None:
        events.append(("guard_log", kwargs))

    def maybe_rollback_to_best_checkpoint(**kwargs: object) -> dict[str, object]:
        events.append(("rollback", kwargs))
        return {
            "update_count": 8,
            "best_update_count": 6,
            "current_score": 0.25,
            "best_score": 0.5,
            "reasons": ["score_regressed"],
        }

    result = _maybe_run_periodic_dev_eval_and_checkpoint_guard(
        learner=learner,
        model=model,
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=runtime,
        device=device,
        spec_hash256="spec-hash",
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        last_dev_eval_summary=None,
        last_dev_eval_update_count=None,
        last_checkpoint_guard_rollback_update=4,
        run_id256="run-id",
        config_hash256="config-hash",
        tensorboard_logger=FakeTensorBoardLogger(),
        hooks=TrainingPeriodicDevEvalHooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: True,
            run_periodic_dev_eval=run_periodic_dev_eval,
            slug_policy_id=slug_policy_id,
            load_checkpoint_tracker=load_checkpoint_tracker,
            confirmatory_dev_eval_request=confirmatory_dev_eval_request,
            periodic_dev_eval_schedule=periodic_dev_eval_schedule,
            expand_periodic_dev_eval_paired_seeds=expand_periodic_dev_eval_paired_seeds,
            ensure_current_checkpoint=ensure_current_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_log_structured_mainmove_guard=maybe_log_structured_mainmove_guard,
            maybe_rollback_to_best_checkpoint=maybe_rollback_to_best_checkpoint,
        ),
    )

    assert result == PeriodicDevEvalGuardResult(
        last_dev_eval_summary=effective_summary,
        last_dev_eval_update_count=8,
        last_checkpoint_guard_rollback_update=8,
        stop_requested=True,
    )
    assert latest_metrics["checkpoint_guard_stop_after_rollback"] == 1.0
    assert [event[0] for event in events] == [
        "run_eval",
        "slug",
        "load_tracker",
        "confirm_request",
        "schedule",
        "expand",
        "run_eval",
        "ensure_checkpoint",
        "aliases",
        "guard_log",
        "rollback",
        "tb_eval",
        "tb_tracker",
    ]
    assert events[0][1]["run_id256"] == "run-id"
    assert events[0][1]["config_hash256"] == "config-hash"
    assert events[0][1]["spec_hash256"] == "spec-hash"
    assert events[3][1]["existing_best_record"] == {"policy_id": "best"}
    assert events[3][1]["dev_eval_summary"] is summary_payload
    assert events[5][1]["args"] == ([10, 20],)
    assert events[5][1]["kwargs"] == {
        "requested_pairs": 3,
        "seed_file_sha256": "seed-sha",
        "update_count": 8,
        "policy_version": 13,
        "scope": "periodic_dev_eval_confirmatory",
    }
    assert events[6][1]["artifact_dir_name"] == "dev_eval_confirmatory"
    assert events[6][1]["paired_seeds_override"] == ["pair-a", "pair-b", "pair-c"]
    assert events[6][1]["persist_summary"] is False
    assert events[6][1]["update_stall_monitor"] is False
    assert events[8][1]["checkpoint_path"] == checkpoint_path
    assert events[8][1]["dev_eval_summary"] is effective_summary
    assert events[9][1]["dev_eval_summary"] is effective_summary
    assert events[10][1]["last_rollback_update"] == 4
    assert events[10][1]["dev_eval_summary"] is effective_summary
    assert events[11][1] == {"payload": effective_summary, "step": 8}
    assert events[12][1] == {"payload": {"best": {"update": 8}}, "step": 8}
    stdout = capsys.readouterr().out
    assert "Periodic dev eval: update=8 opponent=b2 aggregate=0.2500 anchors=B2 HeuristicPublic" in stdout
    assert "Confirmatory dev eval: update=8 paired_seeds=3 aggregate=0.5000" in stdout
    assert "Checkpoint guard rollback: update=8 best_update=6 current_score=0.2500 best_score=0.5000" in stdout
    assert "Checkpoint guard early stop after rollback: update=8 best_update=6" in stdout


def test_final_dev_eval_summary_for_update_uses_only_current_update_summary() -> None:
    summary = {"aggregate_score": 0.75}

    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=summary,
            last_dev_eval_update_count=12,
            learner_update_count=12,
        )
        is summary
    )
    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=summary,
            last_dev_eval_update_count=11,
            learner_update_count=12,
        )
        is None
    )
    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=None,
            last_dev_eval_update_count=12,
            learner_update_count=12,
        )
        is None
    )


def test_finalize_training_checkpoint_selection_publishes_current_checkpoint_without_guard_reload(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=9)
    stack = object()
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    runtime = object()
    device = object()
    algorithm = object()
    latest_metrics = {"loss": 1.0}
    dev_eval_summary = {"aggregate_score": 0.5}
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_9.pt"

    class FakeTensorBoardLogger:
        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            events.append(("tensorboard", {"payload": payload, "step": step}))

    def ensure_current_checkpoint(**kwargs: object) -> Path:
        events.append(("ensure", kwargs))
        return checkpoint_path

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        events.append(("aliases", kwargs))
        return {"current": {"update": 9}}

    def maybe_finalize_from_best_checkpoint(**kwargs: object) -> None:
        events.append(("finalize", kwargs))
        return None

    def load_checkpoint_tracker(_training_paths: object) -> object:
        raise AssertionError("tracker should not reload when final guard does not change selection")

    tracker_payload = _finalize_training_checkpoint_selection(
        learner=learner,
        stack=stack,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=runtime,
        device=device,
        spec_hash256="spec-hash",
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        last_dev_eval_summary=dev_eval_summary,
        last_dev_eval_update_count=9,
        tensorboard_logger=FakeTensorBoardLogger(),
        hooks=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=ensure_current_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_finalize_from_best_checkpoint=maybe_finalize_from_best_checkpoint,
            load_checkpoint_tracker=load_checkpoint_tracker,
        ),
    )

    assert tracker_payload == {"current": {"update": 9}}
    assert [event[0] for event in events] == ["ensure", "aliases", "finalize", "tensorboard"]
    assert events[0][1]["learner"] is learner
    assert events[0][1]["spec_hash256"] == "spec-hash"
    assert events[1][1]["checkpoint_path"] == checkpoint_path
    assert events[1][1]["latest_metrics"] is latest_metrics
    assert events[1][1]["dev_eval_summary"] is dev_eval_summary
    assert events[2][1]["dev_eval_summary"] is dev_eval_summary
    assert events[3][1] == {"payload": tracker_payload, "step": 9}


def test_finalize_training_checkpoint_selection_reloads_tracker_after_best_finalization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=10)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_10.pt"
    reloaded_tracker = {"best": {"update": 7}}

    class FakeTensorBoardLogger:
        def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
            events.append(("tensorboard", {"payload": payload, "step": step}))

    def ensure_current_checkpoint(**kwargs: object) -> Path:
        events.append(("ensure", kwargs))
        return checkpoint_path

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        events.append(("aliases", kwargs))
        return {"current": {"update": 10}}

    def maybe_finalize_from_best_checkpoint(**kwargs: object) -> dict[str, object]:
        events.append(("finalize", kwargs))
        return {
            "update_count": 10,
            "best_update_count": 7,
            "current_score": 0.2,
            "best_score": 0.6,
        }

    def load_checkpoint_tracker(paths: object) -> dict[str, object]:
        events.append(("load_tracker", {"paths": paths}))
        return reloaded_tracker

    tracker_payload = _finalize_training_checkpoint_selection(
        learner=learner,
        stack=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary={"aggregate_score": 0.5},
        last_dev_eval_update_count=9,
        tensorboard_logger=FakeTensorBoardLogger(),
        hooks=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=ensure_current_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_finalize_from_best_checkpoint=maybe_finalize_from_best_checkpoint,
            load_checkpoint_tracker=load_checkpoint_tracker,
        ),
    )

    assert tracker_payload is reloaded_tracker
    assert [event[0] for event in events] == ["ensure", "aliases", "finalize", "load_tracker", "tensorboard"]
    assert events[1][1]["dev_eval_summary"] is None
    assert events[2][1]["dev_eval_summary"] is None
    assert events[3][1] == {"paths": training_paths}
    assert events[4][1] == {"payload": reloaded_tracker, "step": 10}
    stdout = capsys.readouterr().out
    assert "Checkpoint guard final selection: update=10 best_update=7 current_score=0.2000 best_score=0.6000" in stdout


def test_final_checkpoint_publication_helper_uses_only_current_dev_eval_summary(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=12)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_12.pt"
    current_summary = {"aggregate_score": 0.8}
    stale_summary = {"aggregate_score": 0.2}

    def ensure_current_checkpoint(**kwargs: object) -> Path:
        events.append(("ensure", kwargs))
        return checkpoint_path

    def publish_checkpoint_aliases(**kwargs: object) -> dict[str, object]:
        events.append(("aliases", kwargs))
        return {"latest": {"update": 12}}

    publication = checkpoint_finalization.publish_final_checkpoint_aliases(
        hooks=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=ensure_current_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_finalize_from_best_checkpoint=lambda **_kwargs: None,
            load_checkpoint_tracker=lambda _paths: {},
        ),
        learner=learner,
        stack=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=current_summary,
        last_dev_eval_update_count=12,
        update_count=12,
    )

    assert publication == checkpoint_finalization.FinalCheckpointPublication(
        checkpoint_path=checkpoint_path,
        dev_eval_summary=current_summary,
        tracker_payload={"latest": {"update": 12}},
    )
    assert [event[0] for event in events] == ["ensure", "aliases"]
    assert events[1][1]["dev_eval_summary"] is current_summary

    stale_publication = checkpoint_finalization.publish_final_checkpoint_aliases(
        hooks=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=ensure_current_checkpoint,
            publish_checkpoint_aliases=publish_checkpoint_aliases,
            maybe_finalize_from_best_checkpoint=lambda **_kwargs: None,
            load_checkpoint_tracker=lambda _paths: {},
        ),
        learner=learner,
        stack=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=stale_summary,
        last_dev_eval_update_count=11,
        update_count=12,
    )

    assert stale_publication.dev_eval_summary is None
    assert events[-1][1]["dev_eval_summary"] is None


def test_final_checkpoint_selection_helper_reloads_tracker_only_after_guard_event(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=12)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    publication = checkpoint_finalization.FinalCheckpointPublication(
        checkpoint_path=tmp_path / "checkpoints" / "checkpoint_12.pt",
        dev_eval_summary={"aggregate_score": 0.3},
        tracker_payload={"latest": {"update": 12}},
    )
    reloaded_tracker = {"best": {"update": 9}}

    def maybe_finalize_from_best_checkpoint(**kwargs: object) -> dict[str, object]:
        events.append(("finalize", kwargs))
        return {
            "update_count": 12,
            "best_update_count": 9,
            "current_score": 0.3,
            "best_score": 0.7,
        }

    def load_checkpoint_tracker(paths: object) -> dict[str, object]:
        events.append(("load", {"paths": paths}))
        return reloaded_tracker

    selection = checkpoint_finalization.select_final_checkpoint_tracker_payload(
        hooks=TrainingFinalCheckpointHooks(
            ensure_current_checkpoint=lambda **_kwargs: publication.checkpoint_path,
            publish_checkpoint_aliases=lambda **_kwargs: publication.tracker_payload,
            maybe_finalize_from_best_checkpoint=maybe_finalize_from_best_checkpoint,
            load_checkpoint_tracker=load_checkpoint_tracker,
        ),
        learner=learner,
        stack=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        publication=publication,
    )

    assert selection == checkpoint_finalization.FinalCheckpointSelection(
        tracker_payload=reloaded_tracker,
        guard_event={
            "update_count": 12,
            "best_update_count": 9,
            "current_score": 0.3,
            "best_score": 0.7,
        },
    )
    assert [event[0] for event in events] == ["finalize", "load"]
    assert events[0][1]["dev_eval_summary"] is publication.dev_eval_summary
    assert events[1][1] == {"paths": training_paths}
    stdout = capsys.readouterr().out
    assert "Checkpoint guard final selection: update=12 best_update=9 current_score=0.3000 best_score=0.7000" in stdout


def test_resume_requires_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=None,
        unroll_length=None,
        runtime_mode=None,
        profile=None,
    )

    try:
        _require_explicit_resume_geometry(parser, args)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("resume without explicit geometry should fail")

    assert "--num-envs" in message
    assert "--unroll-length" in message
    assert "--runtime-mode" in message
    assert "--profile" in message


def test_resume_accepts_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=32,
        unroll_length=16,
        runtime_mode="train_async_fast",
        profile="fast",
    )

    _require_explicit_resume_geometry(parser, args)


def test_train_cli_state_rejects_checkpoint_init_with_resume(tmp_path: Path) -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    stack = SimpleNamespace(
        config=SimpleNamespace(training=object()),
    )
    args = SimpleNamespace(
        run_label="run_a",
        run_id_alias="",
        resume_run_dir=tmp_path / "runs" / "source",
        resume_from="",
        num_envs=2,
        unroll_length=4,
        runtime_mode="train_ordered",
        profile="default",
        max_updates=1,
        stack_config=tmp_path / "stack.yaml",
        config_override=(),
        profile_timers=False,
        torch_profiler=False,
        init_from_checkpoint=tmp_path / "checkpoint.pt",
        init_schedule_offset_updates=None,
        public_demo=False,
    )
    api = SimpleNamespace(
        QueueRuntimeMode=str,
        _resolve_run_label=lambda parser, run_label, run_id_alias: run_label,
        _require_positive_int=lambda _flag, value: int(value),
        load_stack_config=lambda _path: stack,
        apply_stack_overrides=lambda loaded_stack, _overrides: loaded_stack,
        parse_override_tokens=lambda _tokens: (),
        _apply_training_flag_overrides=lambda loaded_stack, **_kwargs: loaded_stack,
        _manifest_scaffold_only_reason=lambda _stack: None,
        _resolve_resume_checkpoint_path=lambda *, resume_from, resume_run_dir: None,
    )

    try:
        resolve_train_cli_state(parser=parser, args=args, api=api)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("checkpoint init combined with resume should fail")

    assert "--init-from-checkpoint starts a fresh run" in message


def test_train_startup_state_uses_public_demo_contract_without_runtime_load(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    stack = SimpleNamespace(root=tmp_path, components=[object(), object()])
    cli = TrainCliState(
        run_label="toy_demo",
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_ordered",
        stack=stack,
        training_config=object(),
        manifest_only_reason=None,
        public_demo_enabled=True,
        resume_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
    )
    args = SimpleNamespace(spec_hash="", config_hash="")
    run_identity = SimpleNamespace(run_id256="a" * 64, run_id64="a" * 16, run_dir_name="toy_demo")

    def fake_banner(
        spec_hash256: str,
        config_hash256: str,
        *,
        run_id64: str,
        run_id256: str,
        run_label: str,
        run_dir_name: str,
        spec_mismatch_policy: str,
    ) -> None:
        calls["banner"] = (
            spec_hash256,
            config_hash256,
            run_id64,
            run_id256,
            run_label,
            run_dir_name,
            spec_mismatch_policy,
        )

    def fake_new_run_identity(**kwargs: object) -> object:
        calls["identity"] = kwargs
        return run_identity

    api = SimpleNamespace(
        public_demo_spec_bundle=lambda: {"action": {"pass_action_id": 8}, "spec_hash": "toy"},
        assert_spec_bundle_contract=lambda expected, bundle: calls.setdefault("spec", (expected, bundle)),
        public_demo_spec_hash256=lambda: "b" * 64,
        public_demo_simulator_info=lambda: {"compatibility_hash": "public_demo"},
        load_verified_simulator_contract=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public demo startup must not load the runtime simulator contract")
        ),
        compute_config_hash256=lambda loaded_stack: "c" * 64,
        _expected_sha256=lambda value, *, flag_name: "",
        _require_matching_hash=lambda **kwargs: calls.setdefault("hash", kwargs),
        _git_commit=lambda: "d" * 40,
        _start_nonce=lambda: "nonce",
        new_run_identity=fake_new_run_identity,
        _run_artifacts_from_existing_run_dir=lambda _path: (_ for _ in ()).throw(
            AssertionError("fresh public demo startup should not resolve resume artifacts")
        ),
        resume_run_identity=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh public demo startup should not resume identity")
        ),
        _load_json_object=lambda *_args, **_kwargs: {},
        print_startup_banner=fake_banner,
        _spec_mismatch_policy=lambda loaded_stack: "hard_fail",
    )

    startup = prepare_train_startup_state(parser=SimpleNamespace(), args=args, api=api, cli=cli)

    assert startup.simulator_contract is None
    assert startup.spec_hash256 == "b" * 64
    assert startup.config_hash256 == "c" * 64
    assert startup.run_id256 == "a" * 64
    assert startup.run_id64 == "a" * 16
    assert startup.resume_artifacts is None
    assert calls["spec"] == ("", {"action": {"pass_action_id": 8}, "spec_hash": "toy"})
    assert calls["hash"] == {"flag_name": "--config-hash", "expected": "", "actual": "c" * 64}
    assert calls["identity"]["run_label"] == "toy_demo"
    assert calls["banner"] == ("b" * 64, "c" * 64, "a" * 16, "a" * 64, "toy_demo", "toy_demo", "hard_fail")


def test_train_manifest_state_writes_reports_and_logs_tensorboard_context(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    writes: list[tuple[Path, dict[str, object]]] = []
    training_config = object()
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"report_eval": tmp_path / "seeds.txt"},
        config=SimpleNamespace(
            training=training_config,
            reproducibility=None,
            system=SimpleNamespace(actor_device="cuda:1"),
        ),
    )
    cli = TrainCliState(
        run_label="manifest_run",
        num_envs=4,
        unroll_length=8,
        max_updates=3,
        runtime_mode="train_ordered",
        stack=stack,
        training_config=training_config,
        manifest_only_reason=None,
        public_demo_enabled=False,
        resume_run_dir=None,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=tmp_path / "init.pt",
        init_schedule_offset_override_updates=12,
    )
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=object(),
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "compat"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="manifest_run",
        resume_artifacts=None,
    )
    layout = SimpleNamespace(tensorboard_dir=tmp_path / "tb")
    artifacts = SimpleNamespace(
        run_dir=tmp_path / "runs" / "manifest_run",
        run_dir_name="manifest_run",
        layout=layout,
        manifest_path=tmp_path / "runs" / "manifest_run" / "manifest.json",
        run_summary_path=tmp_path / "runs" / "manifest_run" / "run_summary.json",
        determinism_report_path=tmp_path / "runs" / "manifest_run" / "determinism.json",
        environment_path=tmp_path / "runs" / "manifest_run" / "environment.json",
    )

    class FakeManifest:
        hardware = {"device": "cuda:0"}

        def __init__(self, **kwargs: object) -> None:
            calls["manifest_kwargs"] = kwargs

        def to_dict(self) -> dict[str, object]:
            return {"manifest": "payload"}

    class FakeTensorBoardLogger:
        enabled = True

        def __init__(self, log_dir: Path) -> None:
            calls["tensorboard_dir"] = log_dir

        def log_run_context(self, **kwargs: object) -> None:
            calls["tensorboard_context"] = kwargs

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        writes.append((path, dict(payload)))

    def fake_write_run_artifacts(runs_dir: Path, manifest: object, *, run_label: str | None) -> object:
        calls["write_run_artifacts"] = (runs_dir, manifest, run_label)
        return artifacts

    api = SimpleNamespace(
        _resolve_device=lambda loaded_stack, _device_arg: "cuda:0",
        _resolve_runtime_profile=lambda loaded_stack, _profile_arg: "fast",
        _resolve_seed=lambda loaded_stack, _seed_arg: 99,
        _manifest_actor_device_layout=lambda **kwargs: calls.setdefault("actor_layout", kwargs) or ("cuda:1",),
        _resolve_policy_set_selection=lambda loaded_stack, **kwargs: (
            ["B0 RandomLegal"],
            {"status": "resolved", "mode": "deterministic_v1"},
        ),
        RunManifest=FakeManifest,
        _git_dirty=lambda: False,
        canonical_config_dict=lambda loaded_stack: {"config": "canonical"},
        build_seed_file_manifest=lambda seed_sets, *, root: {"report_eval": {"path": "seeds.txt"}},
        _hardware_summary=lambda device, *, actor_device, actor_device_layout: {
            "device": device,
            "actor_device": actor_device,
            "actor_device_layout": actor_device_layout,
        },
        _evaluation_pinning=lambda loaded_stack: {"eval_device": "cpu"},
        write_run_artifacts=fake_write_run_artifacts,
        _load_json_object=lambda path, *, label: {"label": label},
        augment_run_summary_payload=lambda payload, **kwargs: payload.update({"run_summary": kwargs}),
        augment_determinism_payload=lambda payload, **kwargs: payload.update({"determinism": kwargs}),
        augment_environment_payload=lambda payload, **kwargs: payload.update({"environment": kwargs}),
        _write_json=fake_write_json,
        TensorBoardLogger=FakeTensorBoardLogger,
        tensorboard_unavailable_reason=lambda: None,
        sys=SimpleNamespace(argv=["python", "train.py"], stderr=SimpleNamespace(write=lambda text: None)),
    )
    args = SimpleNamespace(
        device="cuda",
        profile="fast",
        seed=99,
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        seed_snapshot_run_dir=tmp_path / "seed_source",
    )

    manifest_state = prepare_train_manifest_state(args=args, api=api, startup=startup)

    assert manifest_state.artifacts is artifacts
    assert manifest_state.device == "cuda:0"
    assert manifest_state.profile == "fast"
    assert manifest_state.seed == 99
    assert manifest_state.policy_set_selection_details == {"status": "resolved", "mode": "deterministic_v1"}
    assert calls["write_run_artifacts"][0] == tmp_path / "runs"
    assert calls["write_run_artifacts"][2] == "manifest_run"
    assert calls["manifest_kwargs"]["run_id256"] == "a" * 64
    assert calls["manifest_kwargs"]["seed_derivation"]["effective_base_seed64"] == 99
    assert calls["manifest_kwargs"]["seed_derivation"]["cli_seed_override"] is True
    assert calls["manifest_kwargs"]["hardware"]["actor_device"] == "cuda:1"
    assert calls["tensorboard_dir"] == tmp_path / "tb"
    assert calls["tensorboard_context"]["manifest"] == {"manifest": "payload"}
    assert writes == [
        (artifacts.run_summary_path, manifest_state.run_summary_payload),
        (artifacts.determinism_report_path, manifest_state.determinism_payload),
        (artifacts.environment_path, manifest_state.environment_payload),
    ]
    assert manifest_state.run_summary_payload["run_summary"]["init_from_checkpoint_path"] == tmp_path / "init.pt"
    assert manifest_state.determinism_payload["determinism"]["resume_checkpoint_path"] == tmp_path / "resume.pt"
    assert manifest_state.environment_payload["environment"]["argv"] == ["python", "train.py"]


def test_train_execution_dispatch_stages_public_demo_without_training(tmp_path: Path, capsys) -> None:
    calls: dict[str, object] = {}
    cli = TrainCliState(
        run_label="toy_demo",
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_ordered",
        stack=SimpleNamespace(root=tmp_path),
        training_config=object(),
        manifest_only_reason=None,
        public_demo_enabled=True,
        resume_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
    )
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=None,
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "public_demo"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="toy_demo",
        resume_artifacts=None,
    )
    manifest_state = SimpleNamespace(
        artifacts=SimpleNamespace(run_dir=tmp_path / "runs" / "toy_demo"),
        device="cpu",
        profile="default",
        seed=7,
        tensorboard_logger=SimpleNamespace(),
    )

    def fake_stage_public_demo_run(run_dir: Path) -> object:
        calls["stage"] = run_dir
        return SimpleNamespace(policy_ids=["B0 RandomLegal"], catalog_path=run_dir / "public_demo" / "catalog.json")

    api = SimpleNamespace(
        PUBLIC_DEMO_MODE="public_demo",
        stage_public_demo_run=fake_stage_public_demo_run,
        _run_minimal_training=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("public demo must not execute simulator training")
        ),
    )

    execute_train_run(args=SimpleNamespace(), api=api, startup=startup, manifest_state=manifest_state)

    assert calls["stage"] == tmp_path / "runs" / "toy_demo"
    stdout = capsys.readouterr().out
    assert "Staged public-demo toy catalog and policy bundle" in stdout
    assert "demo-only" in stdout


def test_train_execution_dispatch_runs_minimal_training_with_resolved_settings(tmp_path: Path, capsys) -> None:
    calls: dict[str, object] = {}
    training_config = object()
    tensorboard_logger = SimpleNamespace()
    cli = TrainCliState(
        run_label="train_run",
        num_envs=4,
        unroll_length=8,
        max_updates=3,
        runtime_mode="train_async_fast",
        stack=SimpleNamespace(root=tmp_path),
        training_config=training_config,
        manifest_only_reason=None,
        public_demo_enabled=False,
        resume_run_dir=None,
        resume_checkpoint_path=tmp_path / "resume.pt",
        init_from_checkpoint_path=tmp_path / "init_cli.pt",
        init_schedule_offset_override_updates=12,
    )
    contract = object()
    startup = TrainStartupState(
        cli=cli,
        simulator_contract=contract,
        spec_bundle={"action": {"pass_action_id": 8}},
        spec_hash256="b" * 64,
        simulator_info={"compatibility_hash": "runtime"},
        config_hash256="c" * 64,
        git_commit="d" * 40,
        start_nonce="nonce",
        run_id256="a" * 64,
        run_id64="a" * 16,
        run_dir_name="train_run",
        resume_artifacts=None,
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "runs" / "train_run")
    manifest_state = SimpleNamespace(
        artifacts=artifacts,
        device="cuda:0",
        profile="fast",
        seed=99,
        tensorboard_logger=tensorboard_logger,
    )
    execution_settings = SimpleNamespace(
        checkpoint_interval_updates=5,
        b1_baseline_run_dir=tmp_path / "b1",
        seed_snapshot_run_dir=tmp_path / "seed_source",
        init_from_checkpoint_path=tmp_path / "init_resolved.pt",
        profile_timers=True,
        torch_profiler=False,
    )

    def fake_minimal_training(**kwargs: object) -> dict[str, float]:
        calls["minimal"] = kwargs
        return {"loss": 1.25, "policy_loss": 0.5, "value_loss": 0.75, "entropy": 0.125}

    def fake_resolve_training_execution_settings(**kwargs: object) -> object:
        calls["settings"] = kwargs
        return execution_settings

    api = SimpleNamespace(
        _runtime_training_prerequisite_failure=lambda stack: None,
        _raise_runtime_prerequisite_failure=lambda reason: (_ for _ in ()).throw(RuntimeError(reason)),
        _noleague_training_prerequisite_failure=lambda stack: None,
        _raise_noleague_training_prerequisite_failure=lambda reason: (_ for _ in ()).throw(RuntimeError(reason)),
        resolve_training_execution_settings=fake_resolve_training_execution_settings,
        profiling_enabled_message=lambda config: "Profiling enabled",
        _run_minimal_training=fake_minimal_training,
    )
    args = SimpleNamespace(
        checkpoint_interval_updates=5,
        b1_baseline_run_dir=tmp_path / "b1_cli",
        seed_snapshot_run_dir=tmp_path / "seed_cli",
        init_from_checkpoint=tmp_path / "init_cli.pt",
    )

    execute_train_run(args=args, api=api, startup=startup, manifest_state=manifest_state)

    assert calls["settings"] == {
        "training_config": training_config,
        "checkpoint_interval_override": 5,
        "b1_baseline_run_dir": tmp_path / "b1_cli",
        "seed_snapshot_run_dir": tmp_path / "seed_cli",
        "init_from_checkpoint": tmp_path / "init_cli.pt",
    }
    minimal = calls["minimal"]
    assert minimal["stack"] is cli.stack
    assert minimal["contract"] is contract
    assert minimal["artifacts"] is artifacts
    assert minimal["num_envs"] == 4
    assert minimal["unroll_length"] == 8
    assert minimal["max_updates"] == 3
    assert minimal["profile"] == "fast"
    assert minimal["device"] == "cuda:0"
    assert minimal["seed"] == 99
    assert minimal["checkpoint_interval_updates"] == 5
    assert minimal["run_id256"] == "a" * 64
    assert minimal["config_hash256"] == "c" * 64
    assert minimal["spec_hash256"] == "b" * 64
    assert minimal["runtime_mode"] == "train_async_fast"
    assert minimal["b1_baseline_run_dir"] == tmp_path / "b1"
    assert minimal["seed_snapshot_run_dir"] == tmp_path / "seed_source"
    assert minimal["profile_timers"] is True
    assert minimal["torch_profiler"] is False
    assert minimal["resume_checkpoint_path"] == tmp_path / "resume.pt"
    assert minimal["init_from_checkpoint_path"] == tmp_path / "init_resolved.pt"
    assert minimal["init_schedule_offset_override_updates"] == 12
    assert minimal["tensorboard_logger"] is tensorboard_logger
    stdout = capsys.readouterr().out
    assert "Profiling enabled" in stdout
    assert "Completed canonical single-node training run: loss=1.250000" in stdout
