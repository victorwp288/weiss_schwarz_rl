from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.controller_guard_commands import _guard_run_command
from weiss_rl.workflows.controller_plan_state import (
    ControllerWorkflowPlan,
    ControllerWorkflowRequest,
    controller_workflow_request,
)

DEFAULT_GUARD_REQUIRED_ANCHORS = (
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)


def build_guard_run_workflow_plan_for_request(request: ControllerWorkflowRequest) -> ControllerWorkflowPlan:
    args = request.args
    if args.command != "guard-run":
        raise ValueError(f"unsupported guard-run workflow command: {args.command!r}")

    run_dir = Path(args.run_dir)
    return ControllerWorkflowPlan(
        plan_name=f"{run_dir.name}_guard-run",
        command=_guard_run_command(
            python_exe=request.python_exe,
            run_dir=run_dir,
            required_anchors=tuple(args.required_anchor or DEFAULT_GUARD_REQUIRED_ANCHORS),
            min_latest_anchor_score=float(args.min_latest_anchor_score),
            max_latest_drop=float(args.max_latest_drop),
            require_promotion_pass_after_attempts=int(args.require_promotion_pass_after_attempts),
            max_consecutive_promotion_failures=int(args.max_consecutive_promotion_failures),
            max_vtrace_rho_p99=args.max_vtrace_rho_p99,
        ),
        payload={"workflow": "guard-run"},
    )


def build_guard_run_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> ControllerWorkflowPlan:
    return build_guard_run_workflow_plan_for_request(
        controller_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = [
    "DEFAULT_GUARD_REQUIRED_ANCHORS",
    "build_guard_run_workflow_plan",
    "build_guard_run_workflow_plan_for_request",
]
