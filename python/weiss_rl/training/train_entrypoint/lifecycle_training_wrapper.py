"""Minimal training runner wrapper for the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.minimal.entrypoint_hooks import (
    MinimalTrainingEntryRequest,
    run_minimal_training_with_script_hooks,
)


def install_minimal_training_wrapper(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    """Install the concrete training runner used by the script facade."""

    def _run_minimal_training(
        *,
        stack: Any,
        contract: Any,
        artifacts: Any,
        num_envs: int,
        unroll_length: int,
        max_updates: int,
        profile: str,
        device: Any,
        seed: int,
        checkpoint_interval_updates: int,
        run_id256: str,
        config_hash256: str,
        spec_hash256: str,
        runtime_mode: Any,
        b1_baseline_run_dir: Path | None,
        seed_snapshot_run_dir: Path | None = None,
        profile_timers: bool = False,
        torch_profiler: bool = False,
        resume_checkpoint_path: Path | None = None,
        init_from_checkpoint_path: Path | None = None,
        init_schedule_offset_override_updates: int | None = None,
        tensorboard_logger: Any | None = None,
    ) -> dict[str, float]:
        return run_minimal_training_with_script_hooks(
            entrypoint_api(),
            MinimalTrainingEntryRequest(
                stack=stack,
                contract=contract,
                artifacts=artifacts,
                num_envs=num_envs,
                unroll_length=unroll_length,
                max_updates=max_updates,
                profile=profile,
                device=device,
                seed=seed,
                checkpoint_interval_updates=checkpoint_interval_updates,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                runtime_mode=runtime_mode,
                b1_baseline_run_dir=b1_baseline_run_dir,
                seed_snapshot_run_dir=seed_snapshot_run_dir,
                profile_timers=profile_timers,
                torch_profiler=torch_profiler,
                resume_checkpoint_path=resume_checkpoint_path,
                init_from_checkpoint_path=init_from_checkpoint_path,
                init_schedule_offset_override_updates=init_schedule_offset_override_updates,
                tensorboard_logger=tensorboard_logger,
            ),
        )

    namespace["_run_minimal_training"] = _run_minimal_training


__all__ = ["install_minimal_training_wrapper"]
