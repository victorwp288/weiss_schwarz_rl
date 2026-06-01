from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.controller_execution import _run_controller_workflow_plan
from weiss_rl.workflows.controller_plan import build_controller_workflow_plan_for_request
from weiss_rl.workflows.controller_plan_state import ControllerWorkflowRequest, controller_workflow_request


def dispatch_controller_request(request: ControllerWorkflowRequest) -> bool:
    plan = build_controller_workflow_plan_for_request(request)
    if plan is None:
        return False
    _run_controller_workflow_plan(plan=plan, repo_root=request.repo_root, dry_run=request.dry_run)
    return True


def dispatch_controller_command(*, args: argparse.Namespace, repo_root: Path, python_exe: str) -> bool:
    return dispatch_controller_request(
        controller_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


__all__ = ["dispatch_controller_command", "dispatch_controller_request"]
