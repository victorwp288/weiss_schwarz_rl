from __future__ import annotations

from pathlib import Path

from weiss_rl.workflows.entrypoint_command_builders import build_train_entrypoint_command
from weiss_rl.workflows.training_workflow.profiles import TrainProfile


def _train_command(
    *,
    python_exe: str,
    stack_config: Path,
    run_label: str,
    profile: TrainProfile,
    b1_baseline_run_dir: Path | None = None,
    seed_snapshot_run_dir: Path | None = None,
    init_from_checkpoint: Path | None = None,
) -> list[str]:
    return build_train_entrypoint_command(
        python_exe=python_exe,
        stack_config=stack_config,
        run_label=run_label,
        num_envs=profile.num_envs,
        unroll_length=profile.unroll_length,
        max_updates=profile.max_updates,
        runtime_mode=profile.runtime_mode,
        simulator_profile=profile.simulator_profile,
        device=profile.device,
        path_style="posix",
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        init_from_checkpoint=init_from_checkpoint,
        checkpoint_interval_updates=profile.checkpoint_interval_updates,
        overrides=profile.overrides,
    )
