from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.controller_guided_commands import _guided_bootstrap_loop_command
from weiss_rl.workflows.controller_plan_state import (
    ControllerWorkflowPlan,
    ControllerWorkflowRequest,
    controller_workflow_request,
)


def build_guided_bootstrap_loop_workflow_plan_for_request(
    request: ControllerWorkflowRequest,
) -> ControllerWorkflowPlan:
    args = request.args
    if args.command != "guided-bootstrap-loop":
        raise ValueError(f"unsupported guided-bootstrap workflow command: {args.command!r}")

    return ControllerWorkflowPlan(
        plan_name=f"{args.run_prefix}_guided-bootstrap-loop",
        command=_guided_bootstrap_loop_command(
            python_exe=request.python_exe,
            initial_run_dir=Path(args.initial_run_dir),
            initial_policy_id=str(args.initial_policy_id),
            seed_run_dir=args.seed_run_dir,
            run_prefix=str(args.run_prefix),
            stack_config=Path(args.stack_config),
            alias_policy_id=str(args.alias_policy_id),
            segments=int(args.segments),
            segment_updates=int(args.segment_updates),
            confirm_paired_seeds=int(args.confirm_paired_seeds),
            stop_on_latest_falloff=bool(args.stop_on_latest_falloff),
        ),
        payload={
            "workflow": "guided-bootstrap-loop",
            "initial_policy_id": str(args.initial_policy_id),
            "segments": int(args.segments),
            "segment_updates": int(args.segment_updates),
            "confirm_paired_seeds": int(args.confirm_paired_seeds),
        },
    )


def build_guided_bootstrap_loop_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> ControllerWorkflowPlan:
    return build_guided_bootstrap_loop_workflow_plan_for_request(
        controller_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = [
    "build_guided_bootstrap_loop_workflow_plan",
    "build_guided_bootstrap_loop_workflow_plan_for_request",
]
