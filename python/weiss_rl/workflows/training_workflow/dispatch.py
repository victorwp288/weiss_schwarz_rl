from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.training_workflow.execution import _run_training_workflow_plan
from weiss_rl.workflows.training_workflow.plan import build_training_workflow_plan_for_request
from weiss_rl.workflows.training_workflow.plan_state import TrainingWorkflowRequest, training_workflow_request


def dispatch_training_request(request: TrainingWorkflowRequest) -> bool:
    plan = build_training_workflow_plan_for_request(request)
    if plan is None:
        return False
    _run_training_workflow_plan(plan=plan, repo_root=request.repo_root, dry_run=request.dry_run)
    return True


def dispatch_training_command(*, args: argparse.Namespace, repo_root: Path, python_exe: str) -> bool:
    return dispatch_training_request(training_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe))


__all__ = ["dispatch_training_command", "dispatch_training_request"]
