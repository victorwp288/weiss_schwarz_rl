from __future__ import annotations

from weiss_rl.training.train_entrypoint.cli import (
    TrainCliState,
    TrainStartupState,
    _require_explicit_resume_geometry,
    execute_train_run,
    prepare_train_manifest_state,
    prepare_train_startup_state,
    resolve_train_cli_state,
)


def test_implementation_modules_are_not_training_root_aliases() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "__getattr__")
    assert training.__all__ == ()
    for name in (
        "build_training_learner",
        "run_structured_warmstart",
        "validate_algorithm_model_contract",
        "minimal_dev_eval",
        "dev_eval_opponents",
        "dev_eval_runner",
        "checkpoint_alias_candidates",
        "checkpoint_alias_mutation",
        "checkpoint_alias_publication",
        "checkpoint_aliases",
        "checkpoint_finalization",
        "checkpoint_guard",
        "checkpoint_guard_events",
        "checkpoint_io",
        "checkpoint_lifecycle",
        "checkpoint_lifecycle_decisions",
        "checkpoint_lifecycle_effects",
        "checkpoint_lifecycle_plans",
        "checkpoint_lifecycle_transitions",
        "checkpoint_load",
        "checkpoint_periodic_dev_eval",
        "checkpoint_periodic_dev_eval_confirmatory",
        "checkpoint_periodic_dev_eval_guard",
        "checkpoint_resolution",
        "checkpoint_restore",
        "checkpoint_restore_state",
        "checkpoint_snapshot_promotion",
        "checkpoint_structured_guard",
        "checkpoint_tracker",
        "checkpoint_write",
        "minimal_entrypoint_hooks",
        "minimal_finalization",
        "minimal_hook_groups",
        "minimal_initialization",
        "minimal_loop",
        "minimal_promotion",
        "minimal_runner",
        "minimal_setup",
        "minimal_update",
        "train_entrypoint_checkpoints",
        "train_entrypoint_cli",
        "train_entrypoint_compat",
        "train_entrypoint_core_exports",
        "train_entrypoint_eval_exports",
        "train_entrypoint_lifecycle",
        "train_entrypoint_metadata_hooks",
        "train_entrypoint_metadata_wrappers",
        "train_entrypoint_runner_hooks",
        "train_entrypoint_runner_wrappers",
        "train_entrypoint_snapshots",
        "train_entrypoint_training_exports",
        "train_entrypoint_wrappers",
        "training_loop_progress",
        "training_post_update",
        "training_replay_dispatch",
        "training_replay_paths",
        "training_replay_states",
        "training_run_contexts",
        "training_runner",
        "training_setup",
        "training_update",
        "training_update_batch",
        "training_update_completion",
        "training_update_phases",
        "training_update_schedule",
        "training_update_stage_pipeline",
        "training_update_step",
    ):
        assert not hasattr(training, name), name


def test_train_entrypoint_cli_module_owns_lifecycle_helpers() -> None:
    import weiss_rl.training.train_entrypoint.cli as train_entrypoint_cli

    assert train_entrypoint_cli.TrainCliState is TrainCliState
    assert train_entrypoint_cli.TrainStartupState is TrainStartupState
    assert train_entrypoint_cli._require_explicit_resume_geometry is _require_explicit_resume_geometry
    assert train_entrypoint_cli.resolve_train_cli_state is resolve_train_cli_state
    assert train_entrypoint_cli.prepare_train_startup_state is prepare_train_startup_state
    assert train_entrypoint_cli.prepare_train_manifest_state is prepare_train_manifest_state
    assert train_entrypoint_cli.execute_train_run is execute_train_run
