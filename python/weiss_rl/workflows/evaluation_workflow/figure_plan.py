from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_surface import FIGURES_COMMAND
from weiss_rl.workflows.evaluation_workflow.figure_commands import _figures_command
from weiss_rl.workflows.evaluation_workflow.payloads import evaluation_workflow_payload
from weiss_rl.workflows.evaluation_workflow.plan_state import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    evaluation_workflow_request,
)


def build_figures_workflow_plan_for_request(request: EvaluationWorkflowRequest) -> EvaluationWorkflowPlan:
    args = request.args
    if args.command != FIGURES_COMMAND.name:
        raise ValueError(f"unsupported figures workflow command: {args.command!r}")

    run_dir = Path(args.run_dir)
    return EvaluationWorkflowPlan(
        plan_name=f"{run_dir.name}_{FIGURES_COMMAND.name}",
        command=_figures_command(
            python_exe=request.python_exe,
            run_dir=run_dir,
            fig_id=str(args.fig_id),
            formats=tuple(str(fmt) for fmt in args.formats or ()),
        ),
        payload=evaluation_workflow_payload(
            command=FIGURES_COMMAND,
            run_dir=run_dir,
            fig_id=str(args.fig_id),
            formats=tuple(str(fmt) for fmt in args.formats or ()),
        ),
    )


def build_figures_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> EvaluationWorkflowPlan:
    return build_figures_workflow_plan_for_request(
        evaluation_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = ["build_figures_workflow_plan", "build_figures_workflow_plan_for_request"]
