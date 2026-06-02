from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.thesis_wrapper_support.execution import _run_step
from weiss_rl.workflows.thesis_wrapper_support.plan_commands import (
    ThesisWrapperCommands,
    build_thesis_wrapper_commands,
    build_thesis_wrapper_commands_for_request,
)
from weiss_rl.workflows.thesis_wrapper_support.plan_execution import (
    run_thesis_wrapper_commands,
    thesis_wrapper_commands_for_plan,
)
from weiss_rl.workflows.thesis_wrapper_support.request import thesis_wrapper_request
from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperPlan, ThesisWrapperRequest, ThesisWrapperResult
from weiss_rl.workflows.thesis_wrapper_support.summary import (
    thesis_wrapper_summary_payload,
    write_thesis_wrapper_summary,
)


def build_thesis_wrapper_plan(*, args: Any, repo_root: Path, python_exe: str) -> ThesisWrapperPlan:
    return build_thesis_wrapper_plan_for_request(
        thesis_wrapper_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


def build_thesis_wrapper_plan_for_request(request: ThesisWrapperRequest) -> ThesisWrapperPlan:
    commands = build_thesis_wrapper_commands_for_request(request)
    return ThesisWrapperPlan(
        repo_root=request.repo_root,
        python_exe=request.python_exe,
        run_label=request.inputs.run_label,
        run_dir=request.run_dir,
        stack_config=request.stack_config,
        eval_stack_config=request.eval_stack_config,
        preset=request.inputs.preset,
        eval_preset=request.eval_preset,
        train_command=commands.train_command,
        eval_command=commands.eval_command,
        compare_command=commands.compare_command,
        b1_baseline_run_dir=request.inputs.b1_baseline_run_dir,
        dry_run=request.inputs.dry_run,
    )


def run_thesis_wrapper_plan(plan: ThesisWrapperPlan) -> ThesisWrapperResult:
    return run_thesis_wrapper_commands(plan=plan, run_step_fn=_run_step)


__all__ = [
    "ThesisWrapperPlan",
    "ThesisWrapperRequest",
    "ThesisWrapperResult",
    "ThesisWrapperCommands",
    "build_thesis_wrapper_plan",
    "build_thesis_wrapper_plan_for_request",
    "build_thesis_wrapper_commands",
    "build_thesis_wrapper_commands_for_request",
    "run_thesis_wrapper_plan",
    "run_thesis_wrapper_commands",
    "thesis_wrapper_request",
    "thesis_wrapper_summary_payload",
    "thesis_wrapper_commands_for_plan",
    "write_thesis_wrapper_summary",
]
