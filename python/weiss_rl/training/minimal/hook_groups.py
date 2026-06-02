"""Hook grouping for the canonical minimal training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.training.checkpointing.finalization import TrainingFinalCheckpointHooks
from weiss_rl.training.checkpointing.periodic_dev_eval import TrainingPeriodicDevEvalHooks
from weiss_rl.training.checkpointing.snapshot_promotion import TrainingCheckpointPromotionHooks
from weiss_rl.training.loop.runner import MinimalTrainingRunHooks
from weiss_rl.training.loop.setup import MinimalTrainingSetupHooks


@dataclass(frozen=True, slots=True)
class MinimalTrainingHookGroups:
    setup: MinimalTrainingSetupHooks
    checkpoint_promotion: TrainingCheckpointPromotionHooks
    periodic_dev_eval: TrainingPeriodicDevEvalHooks
    final_checkpoint: TrainingFinalCheckpointHooks
    run: MinimalTrainingRunHooks


def minimal_training_hook_groups(hooks: Any) -> MinimalTrainingHookGroups:
    setup_hooks = MinimalTrainingSetupHooks(
        spec_dimensions=hooks.spec_dimensions,
        training_paths=hooks.training_paths,
        validate_algorithm_model_contract=hooks.validate_algorithm_model_contract,
        build_policy_value_model=hooks.build_policy_value_model,
        maybe_compile_learner_model=hooks.maybe_compile_learner_model,
        build_training_learner=hooks.build_training_learner,
        restore_learner_from_checkpoint=hooks.restore_learner_from_checkpoint,
        initialize_learner_from_checkpoint=hooks.initialize_learner_from_checkpoint,
        compute_config_hash256=hooks.compute_config_hash256,
        ensure_noleague_baseline_anchor=hooks.ensure_noleague_baseline_anchor,
        import_seed_snapshot_pool=hooks.import_seed_snapshot_pool,
        canonical_config_dict=hooks.canonical_config_dict,
        build_runtime_config=hooks.build_runtime_config,
        queue_runtime_cls=hooks.queue_runtime_cls,
    )
    checkpoint_promotion_hooks = TrainingCheckpointPromotionHooks(
        write_checkpoint=hooks.write_checkpoint,
        publish_checkpoint_aliases=hooks.publish_checkpoint_aliases,
        maybe_log_structured_mainmove_guard=hooks.maybe_log_structured_mainmove_guard,
        persist_snapshot_registry_entry=hooks.persist_snapshot_registry_entry,
        run_snapshot_promotion_gate=hooks.run_snapshot_promotion_gate,
    )
    periodic_dev_eval_hooks = TrainingPeriodicDevEvalHooks(
        should_run_periodic_dev_eval=hooks.should_run_periodic_dev_eval,
        run_periodic_dev_eval=hooks.run_periodic_dev_eval,
        slug_policy_id=hooks.slug_policy_id,
        load_checkpoint_tracker=hooks.load_checkpoint_tracker,
        confirmatory_dev_eval_request=hooks.confirmatory_dev_eval_request,
        periodic_dev_eval_schedule=hooks.periodic_dev_eval_schedule,
        expand_periodic_dev_eval_paired_seeds=hooks.expand_periodic_dev_eval_paired_seeds,
        ensure_current_checkpoint=hooks.ensure_current_checkpoint,
        publish_checkpoint_aliases=hooks.publish_checkpoint_aliases,
        maybe_log_structured_mainmove_guard=hooks.maybe_log_structured_mainmove_guard,
        maybe_rollback_to_best_checkpoint=hooks.maybe_rollback_to_best_checkpoint,
    )
    final_checkpoint_hooks = TrainingFinalCheckpointHooks(
        ensure_current_checkpoint=hooks.ensure_current_checkpoint,
        publish_checkpoint_aliases=hooks.publish_checkpoint_aliases,
        maybe_finalize_from_best_checkpoint=hooks.maybe_finalize_from_best_checkpoint,
        load_checkpoint_tracker=hooks.load_checkpoint_tracker,
    )
    run_hooks = MinimalTrainingRunHooks(
        central_runtime_actor_torch_threads=hooks.central_runtime_actor_torch_threads,
        build_training_profiler=hooks.build_training_profiler,
        run_structured_warmstart=hooks.run_structured_warmstart,
        profile_block=hooks.profile_block,
        apply_guidance_schedule_for_next_update=hooks.apply_guidance_schedule_for_next_update,
        entropy_coef_for_next_update=hooks.entropy_coef_for_next_update,
        torch_num_threads_scope=hooks.torch_num_threads_scope,
        collect_training_batch=hooks.collect_training_batch,
        write_scalars_record=hooks.write_scalars_record,
        checkpoint_promotion=checkpoint_promotion_hooks,
        periodic_dev_eval=periodic_dev_eval_hooks,
        final_checkpoint=final_checkpoint_hooks,
    )
    return MinimalTrainingHookGroups(
        setup=setup_hooks,
        checkpoint_promotion=checkpoint_promotion_hooks,
        periodic_dev_eval=periodic_dev_eval_hooks,
        final_checkpoint=final_checkpoint_hooks,
        run=run_hooks,
    )
