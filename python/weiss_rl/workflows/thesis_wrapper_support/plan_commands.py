from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weiss_rl.workflows.thesis_wrapper_support.command_builders import (
    build_thesis_compare_command,
    build_thesis_eval_command,
    build_thesis_train_command,
)
from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs
from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperRequest


@dataclass(frozen=True, slots=True)
class ThesisWrapperCommands:
    train_command: list[str]
    eval_command: list[str] | None
    compare_command: list[str] | None


def build_thesis_wrapper_commands(
    *,
    inputs: ThesisWrapperInputs,
    python_exe: str,
    run_dir: Path,
    stack_config: Path,
    eval_stack_config: Path,
) -> ThesisWrapperCommands:
    train_command = build_thesis_train_command(
        python_exe=python_exe,
        stack_config=stack_config,
        run_label=inputs.run_label,
        num_envs=inputs.num_envs,
        unroll_length=inputs.unroll_length,
        max_updates=inputs.max_updates,
        runtime_mode=inputs.runtime_mode,
        profile=inputs.profile,
        device=inputs.device,
        seed=inputs.seed,
        resume_run_dir=inputs.resume_run_dir,
        resume_from=inputs.resume_from,
        b1_baseline_run_dir=inputs.b1_baseline_run_dir,
        train_args=inputs.train_args,
    )

    eval_command = None
    if not inputs.skip_eval:
        eval_command = build_thesis_eval_command(
            python_exe=python_exe,
            eval_stack_config=eval_stack_config,
            run_dir=run_dir,
            b1_baseline_run_dir=inputs.b1_baseline_run_dir,
            eval_args=inputs.eval_args,
        )

    compare_command = None
    if not inputs.skip_compare:
        compare_command = build_thesis_compare_command(
            python_exe=python_exe,
            run_dir=run_dir,
            compare_run_dirs=inputs.compare_run_dirs,
            compare_launch_group_summary=inputs.compare_launch_group_summary,
            compare_out_dir=inputs.compare_out_dir,
            compare_args=inputs.compare_args,
        )

    return ThesisWrapperCommands(
        train_command=train_command,
        eval_command=eval_command,
        compare_command=compare_command,
    )


def build_thesis_wrapper_commands_for_request(request: ThesisWrapperRequest) -> ThesisWrapperCommands:
    return build_thesis_wrapper_commands(
        inputs=request.inputs,
        python_exe=request.python_exe,
        run_dir=request.run_dir,
        stack_config=request.stack_config,
        eval_stack_config=request.eval_stack_config,
    )


__all__ = [
    "ThesisWrapperCommands",
    "build_thesis_wrapper_commands",
    "build_thesis_wrapper_commands_for_request",
]
