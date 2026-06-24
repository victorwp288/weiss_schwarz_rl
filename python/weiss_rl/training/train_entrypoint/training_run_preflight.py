"""Preflight checks and settings resolution for a training run."""

from __future__ import annotations

from typing import Any


def verify_training_run_prerequisites(*, api: Any, stack: Any) -> None:
    """Raise the existing entrypoint errors when runtime prerequisites are missing."""

    runtime_prerequisite_failure = api._runtime_training_prerequisite_failure(stack)
    if runtime_prerequisite_failure is not None:
        api._raise_runtime_prerequisite_failure(runtime_prerequisite_failure)
    noleague_prerequisite_failure = api._noleague_training_prerequisite_failure(stack)
    if noleague_prerequisite_failure is not None:
        api._raise_noleague_training_prerequisite_failure(noleague_prerequisite_failure)


def resolve_entrypoint_execution_settings(*, args: Any, api: Any, training_config: Any) -> Any:
    """Resolve the CLI/config settings consumed by the minimal trainer."""

    return api.resolve_training_execution_settings(
        training_config=training_config,
        checkpoint_interval_override=args.checkpoint_interval_updates,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint=args.init_from_checkpoint,
    )


__all__ = ["resolve_entrypoint_execution_settings", "verify_training_run_prerequisites"]
