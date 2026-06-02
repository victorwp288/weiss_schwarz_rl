from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

PathStyle = Literal["native", "posix"]


def _path_arg(path: Path, *, path_style: PathStyle) -> str:
    return path.as_posix() if path_style == "posix" else str(path)


def build_train_entrypoint_command(
    *,
    python_exe: str,
    stack_config: Path,
    run_label: str,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    runtime_mode: str,
    simulator_profile: str,
    device: str,
    path_style: PathStyle,
    seed: int | None = None,
    resume_run_dir: Path | None = None,
    resume_from: str = "",
    b1_baseline_run_dir: Path | None = None,
    seed_snapshot_run_dir: Path | None = None,
    init_from_checkpoint: Path | None = None,
    checkpoint_interval_updates: int | None = None,
    overrides: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "weiss_rl.training.train_entrypoint",
        "--stack-config",
        _path_arg(stack_config, path_style=path_style),
        "--run-label",
        run_label,
        "--num-envs",
        str(int(num_envs)),
        "--unroll-length",
        str(int(unroll_length)),
        "--max-updates",
        str(int(max_updates)),
        "--runtime-mode",
        str(runtime_mode),
    ]
    if simulator_profile:
        command.extend(["--profile", str(simulator_profile)])
    if device:
        command.extend(["--device", str(device)])
    for override in overrides:
        command.extend(["--override", str(override)])
    if checkpoint_interval_updates is not None:
        command.extend(["--checkpoint-interval-updates", str(int(checkpoint_interval_updates))])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if resume_run_dir is not None:
        command.extend(["--resume-run-dir", _path_arg(resume_run_dir, path_style=path_style)])
    if resume_from:
        command.extend(["--resume-from", str(resume_from)])
    if b1_baseline_run_dir is not None:
        command.extend(["--b1-baseline-run-dir", _path_arg(b1_baseline_run_dir, path_style=path_style)])
    if seed_snapshot_run_dir is not None:
        command.extend(["--seed-snapshot-run-dir", _path_arg(seed_snapshot_run_dir, path_style=path_style)])
    if init_from_checkpoint is not None:
        command.extend(["--init-from-checkpoint", _path_arg(init_from_checkpoint, path_style=path_style)])
    command.extend(str(extra) for extra in extra_args)
    return command


def build_eval_entrypoint_command(
    *,
    python_exe: str,
    stack_config: Path,
    run_dir: Path,
    path_style: PathStyle,
    b1_baseline_run_dir: Path | None = None,
    policy_ids: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "weiss_rl.workflows.eval_entrypoint",
        "--stack-config",
        _path_arg(stack_config, path_style=path_style),
        "--run-dir",
        _path_arg(run_dir, path_style=path_style),
    ]
    if b1_baseline_run_dir is not None:
        command.extend(["--b1-baseline-run-dir", _path_arg(b1_baseline_run_dir, path_style=path_style)])
    for policy_id in policy_ids:
        command.extend(["--policy-id", str(policy_id)])
    command.extend(str(extra) for extra in extra_args)
    return command
