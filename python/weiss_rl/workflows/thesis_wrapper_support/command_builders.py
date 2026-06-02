from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.entrypoint_command_builders import build_eval_entrypoint_command, build_train_entrypoint_command


def build_thesis_train_command(
    *,
    python_exe: str,
    stack_config: Path,
    run_label: str,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    runtime_mode: str,
    profile: str,
    device: str,
    seed: int | None,
    resume_run_dir: Path | None,
    resume_from: str,
    b1_baseline_run_dir: Path | None,
    train_args: Sequence[str],
) -> list[str]:
    return build_train_entrypoint_command(
        python_exe=python_exe,
        stack_config=stack_config,
        run_label=run_label,
        num_envs=num_envs,
        unroll_length=unroll_length,
        max_updates=max_updates,
        runtime_mode=runtime_mode,
        simulator_profile=profile,
        device=device,
        path_style="native",
        seed=seed,
        resume_run_dir=resume_run_dir,
        resume_from=resume_from,
        b1_baseline_run_dir=b1_baseline_run_dir,
        extra_args=train_args,
    )


def build_thesis_eval_command(
    *,
    python_exe: str,
    eval_stack_config: Path,
    run_dir: Path,
    b1_baseline_run_dir: Path | None,
    eval_args: Sequence[str],
) -> list[str]:
    return build_eval_entrypoint_command(
        python_exe=python_exe,
        stack_config=eval_stack_config,
        run_dir=run_dir,
        path_style="native",
        b1_baseline_run_dir=b1_baseline_run_dir,
        extra_args=eval_args,
    )


def build_thesis_compare_command(
    *,
    python_exe: str,
    run_dir: Path,
    compare_run_dirs: Sequence[str],
    compare_launch_group_summary: Path | None,
    compare_out_dir: Path | None,
    compare_args: Sequence[str],
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "weiss_rl.workflows.compare_runs.compare_runs_entrypoint",
        "--run-dir",
        str(run_dir),
    ]
    for baseline_run_dir in compare_run_dirs:
        command.extend(["--run-dir", str(baseline_run_dir)])
    if compare_launch_group_summary is not None:
        command.extend(["--launch-group-summary", str(compare_launch_group_summary)])
    if compare_out_dir is not None:
        command.extend(["--out-dir", str(compare_out_dir)])
    command.extend(str(extra) for extra in compare_args)
    return command
