from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.training.minimal.hook_groups import minimal_training_hook_groups
from weiss_rl.training.minimal.setup import (
    _require_training_stack_components,
)


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
