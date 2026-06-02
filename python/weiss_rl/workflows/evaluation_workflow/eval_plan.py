from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.eval_commands import _eval_command
from weiss_rl.workflows.evaluation_workflow.plan_state import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    evaluation_workflow_request,
)


def build_eval_workflow_plan_for_request(request: EvaluationWorkflowRequest) -> EvaluationWorkflowPlan:
    args = request.args
    if args.command not in {"smoke-eval", "eval-final"}:
        raise ValueError(f"unsupported eval workflow command: {args.command!r}")

    run_dir = Path(args.run_dir)
    return EvaluationWorkflowPlan(
        plan_name=f"{run_dir.name}_{args.command}",
        command=_eval_command(
            python_exe=request.python_exe,
            run_dir=run_dir,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            smoke=args.command == "smoke-eval",
        ),
        payload={"workflow": str(args.command)},
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
