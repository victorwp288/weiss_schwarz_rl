from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.execution import _run_evaluation_workflow_plan
from weiss_rl.workflows.evaluation_workflow.plan import build_evaluation_workflow_plan_for_request
from weiss_rl.workflows.evaluation_workflow.plan_state import EvaluationWorkflowRequest, evaluation_workflow_request


def dispatch_evaluation_request(request: EvaluationWorkflowRequest) -> bool:
    plan = build_evaluation_workflow_plan_for_request(request)
    if plan is None:
        return False
    _run_evaluation_workflow_plan(plan=plan, repo_root=request.repo_root, dry_run=request.dry_run)
    return True


def dispatch_evaluation_command(*, args: argparse.Namespace, repo_root: Path, python_exe: str) -> bool:
    return dispatch_evaluation_request(
        evaluation_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


__all__ = ["dispatch_evaluation_command", "dispatch_evaluation_request"]
