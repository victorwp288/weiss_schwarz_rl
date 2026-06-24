"""Execute the selected training mode after setup and metadata are ready."""

from __future__ import annotations

from typing import Any

from weiss_rl.training.train_entrypoint.state import TrainManifestState, TrainStartupState
from weiss_rl.training.train_entrypoint.training_run_preflight import (
    resolve_entrypoint_execution_settings,
    verify_training_run_prerequisites,
)


def execute_train_run(*, args: Any, api: Any, startup: TrainStartupState, manifest_state: TrainManifestState) -> None:
    cli = startup.cli
    stack = cli.stack
    artifacts = manifest_state.artifacts

    if cli.public_demo_enabled:
        staged = api.stage_public_demo_run(artifacts.run_dir)
        print(
            "Staged public-demo toy catalog and policy bundle: "
            f"mode={api.PUBLIC_DEMO_MODE} policy_count={len(staged.policy_ids)} "
            f"catalog={staged.catalog_path}"
        )
        print(
            "Public demo mode is intentionally synthetic and demo-only. "
            "It does not execute simulator training or claim thesis-grade results."
        )
        return

    if cli.manifest_only_reason is not None:
        api._print_manifest_only_message(cli.manifest_only_reason)
        return

    assert cli.training_config is not None
    verify_training_run_prerequisites(api=api, stack=stack)
    execution_settings = resolve_entrypoint_execution_settings(
        args=args,
        api=api,
        training_config=cli.training_config,
    )

    profiling_message = api.profiling_enabled_message(cli.training_config)
    if profiling_message is not None:
        print(profiling_message)

    assert startup.simulator_contract is not None
    metrics = api._run_minimal_training(
        stack=stack,
        contract=startup.simulator_contract,
        artifacts=artifacts,
        num_envs=cli.num_envs,
        unroll_length=cli.unroll_length,
        max_updates=cli.max_updates,
        profile=manifest_state.profile,
        device=manifest_state.device,
        seed=manifest_state.seed,
        checkpoint_interval_updates=execution_settings.checkpoint_interval_updates,
        run_id256=startup.run_id256,
        config_hash256=startup.config_hash256,
        spec_hash256=startup.spec_hash256,
        runtime_mode=cli.runtime_mode,
        b1_baseline_run_dir=execution_settings.b1_baseline_run_dir,
        seed_snapshot_run_dir=execution_settings.seed_snapshot_run_dir,
        profile_timers=execution_settings.profile_timers,
        torch_profiler=execution_settings.torch_profiler,
        resume_checkpoint_path=cli.resume_checkpoint_path,
        init_from_checkpoint_path=execution_settings.init_from_checkpoint_path,
        init_schedule_offset_override_updates=cli.init_schedule_offset_override_updates,
        tensorboard_logger=manifest_state.tensorboard_logger,
    )
    print(
        "Completed canonical single-node training run: "
        f"loss={metrics.get('loss', 0.0):.6f} "
        f"policy_loss={metrics.get('policy_loss', 0.0):.6f} "
        f"value_loss={metrics.get('value_loss', 0.0):.6f} "
        f"entropy={metrics.get('entropy', 0.0):.6f}"
    )


__all__ = ["execute_train_run"]
