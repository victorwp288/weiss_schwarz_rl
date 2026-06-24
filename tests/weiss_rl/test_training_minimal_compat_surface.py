from __future__ import annotations

import weiss_rl.training.checkpointing.guards.periodic_dev_eval as checkpoint_periodic_dev_eval
import weiss_rl.training.checkpointing.guards.snapshot_promotion as checkpoint_snapshot_promotion
import weiss_rl.training.checkpointing.lifecycle.finalization as checkpoint_finalization
import weiss_rl.training.loop.runner as training_runner
import weiss_rl.training.loop.setup as training_setup
import weiss_rl.training.loop.update as training_update


def test_minimal_promotion_reexports_checkpoint_snapshot_promotion_boundary() -> None:
    import weiss_rl.training.minimal.promotion as minimal_promotion

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
        "weiss_rl.training.checkpointing.guards.snapshot_promotion"
    )


def test_minimal_dev_eval_reexports_checkpoint_periodic_dev_eval_boundary() -> None:
    import weiss_rl.training.minimal.dev_eval as minimal_dev_eval

    assert minimal_dev_eval.PeriodicDevEvalGuardResult is checkpoint_periodic_dev_eval.PeriodicDevEvalGuardResult
    assert minimal_dev_eval.TrainingPeriodicDevEvalHooks is checkpoint_periodic_dev_eval.TrainingPeriodicDevEvalHooks
    assert (
        minimal_dev_eval._maybe_run_periodic_dev_eval_and_checkpoint_guard
        is checkpoint_periodic_dev_eval.maybe_run_periodic_dev_eval_and_checkpoint_guard
    )
    assert checkpoint_periodic_dev_eval.maybe_run_periodic_dev_eval_and_checkpoint_guard.__module__ == (
        "weiss_rl.training.checkpointing.guards.periodic_dev_eval"
    )


def test_minimal_finalization_reexports_checkpoint_finalization_boundary() -> None:
    import weiss_rl.training.minimal.finalization as minimal_finalization

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
        "weiss_rl.training.checkpointing.lifecycle.finalization"
    )


def test_minimal_setup_reexports_canonical_training_setup_boundary() -> None:
    import weiss_rl.training.minimal.initialization as minimal_initialization
    import weiss_rl.training.minimal.setup as minimal_setup

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
    assert training_setup.build_minimal_training_setup.__module__ == "weiss_rl.training.loop.setup"


def test_minimal_update_reexports_canonical_training_update_boundary() -> None:
    import weiss_rl.training.minimal.update as minimal_update

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
    assert training_update.run_training_update_step.__module__ == "weiss_rl.training.loop.update"


def test_minimal_runner_reexports_canonical_training_runner_boundary() -> None:
    import weiss_rl.training.minimal.runner as minimal_runner

    assert minimal_runner.MinimalTrainingRunHooks is training_runner.MinimalTrainingRunHooks
    assert minimal_runner.run_minimal_training_updates is training_runner.run_minimal_training_updates
    assert training_runner.run_minimal_training_updates.__module__ == "weiss_rl.training.loop.runner"
