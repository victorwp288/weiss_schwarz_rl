from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperPlan, ThesisWrapperResult

RunStepFn = Callable[..., dict[str, Any]]


def thesis_wrapper_commands_for_plan(plan: ThesisWrapperPlan) -> list[list[str]]:
    commands = [plan.train_command]
    if plan.eval_command is not None:
        commands.append(plan.eval_command)
    if plan.compare_command is not None:
        commands.append(plan.compare_command)
    return commands


def run_thesis_wrapper_commands(*, plan: ThesisWrapperPlan, run_step_fn: RunStepFn) -> ThesisWrapperResult:
    steps: list[dict[str, Any]] = []
    failed = False

    try:
        for command in thesis_wrapper_commands_for_plan(plan):
            steps.append(run_step_fn(command=command, cwd=plan.repo_root, dry_run=plan.dry_run))
    except subprocess.CalledProcessError:
        failed = True
    return ThesisWrapperResult(plan=plan, steps=steps, failed=failed)


__all__ = ["RunStepFn", "run_thesis_wrapper_commands", "thesis_wrapper_commands_for_plan"]
