from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_surface import EVAL_FINAL_COMMAND, SMOKE_EVAL_COMMAND
from weiss_rl.workflows.evaluation_workflow.eval_commands import _eval_command
from weiss_rl.workflows.evaluation_workflow.payloads import evaluation_workflow_payload
from weiss_rl.workflows.evaluation_workflow.plan_state import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    evaluation_workflow_request,
)


def build_eval_workflow_plan_for_request(request: EvaluationWorkflowRequest) -> EvaluationWorkflowPlan:
    args = request.args
    command = SMOKE_EVAL_COMMAND if args.command == SMOKE_EVAL_COMMAND.name else EVAL_FINAL_COMMAND
    if args.command not in {SMOKE_EVAL_COMMAND.name, EVAL_FINAL_COMMAND.name}:
        raise ValueError(f"unsupported eval workflow command: {args.command!r}")

    run_dir = Path(args.run_dir)
    is_smoke_eval = args.command == SMOKE_EVAL_COMMAND.name
    return EvaluationWorkflowPlan(
        plan_name=f"{run_dir.name}_{args.command}",
        command=_eval_command(
            python_exe=request.python_exe,
            run_dir=run_dir,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            smoke=is_smoke_eval,
        ),
        payload=evaluation_workflow_payload(
            command=command,
            run_dir=run_dir,
            smoke=is_smoke_eval,
            b1_baseline_run_dir=None
            if args.b1_baseline_run_dir is None
            else Path(args.b1_baseline_run_dir).as_posix(),
        ),
    )


def build_eval_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> EvaluationWorkflowPlan:
    return build_eval_workflow_plan_for_request(
        evaluation_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = ["build_eval_workflow_plan", "build_eval_workflow_plan_for_request"]
