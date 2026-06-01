from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.controller_guard_plan import (
    DEFAULT_GUARD_REQUIRED_ANCHORS,
    build_guard_run_workflow_plan,
    build_guard_run_workflow_plan_for_request,
)
from weiss_rl.workflows.controller_guarded_league_plan import (
    _validate_guarded_league_args,
    build_guarded_league_bootstrap_workflow_plan,
    build_guarded_league_bootstrap_workflow_plan_for_request,
)
from weiss_rl.workflows.controller_guided_plan import (
    build_guided_bootstrap_loop_workflow_plan,
    build_guided_bootstrap_loop_workflow_plan_for_request,
)
from weiss_rl.workflows.controller_plan_state import (
    ControllerWorkflowPlan,
    ControllerWorkflowRequest,
    controller_workflow_request,
)

__all__ = [
    "ControllerWorkflowPlan",
    "ControllerWorkflowRequest",
    "DEFAULT_GUARD_REQUIRED_ANCHORS",
    "_validate_guarded_league_args",
    "build_controller_workflow_plan",
    "build_controller_workflow_plan_for_request",
    "build_guard_run_workflow_plan",
    "build_guard_run_workflow_plan_for_request",
    "build_guarded_league_bootstrap_workflow_plan",
    "build_guarded_league_bootstrap_workflow_plan_for_request",
    "build_guided_bootstrap_loop_workflow_plan",
    "build_guided_bootstrap_loop_workflow_plan_for_request",
    "controller_workflow_request",
]


def build_controller_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> ControllerWorkflowPlan | None:
    return build_controller_workflow_plan_for_request(
        controller_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


def build_controller_workflow_plan_for_request(request: ControllerWorkflowRequest) -> ControllerWorkflowPlan | None:
    if request.command == "guard-run":
        return build_guard_run_workflow_plan_for_request(request)

    if request.command == "guided-bootstrap-loop":
        return build_guided_bootstrap_loop_workflow_plan_for_request(request)

    if request.command == "guarded-league-bootstrap":
        return build_guarded_league_bootstrap_workflow_plan_for_request(request)

    return None
