from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_surface import B2_AUDIT_COMMAND
from weiss_rl.workflows.evaluation_workflow.audit_commands import _b2_audit_command
from weiss_rl.workflows.evaluation_workflow.payloads import evaluation_workflow_payload
from weiss_rl.workflows.evaluation_workflow.plan_state import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    evaluation_workflow_request,
)


def build_b2_audit_workflow_plan_for_request(request: EvaluationWorkflowRequest) -> EvaluationWorkflowPlan:
    args = request.args
    if args.command != B2_AUDIT_COMMAND.name:
        raise ValueError(f"unsupported B2 audit workflow command: {args.command!r}")

    run_dir = Path(args.run_dir)
    return EvaluationWorkflowPlan(
        plan_name=f"{run_dir.name}_{B2_AUDIT_COMMAND.name}",
        command=_b2_audit_command(
            python_exe=request.python_exe,
            run_dir=run_dir,
            episodes_jsonl=Path(args.episodes_jsonl),
            policy_id=str(args.policy_id),
            output_run_dir=args.output_run_dir,
            snapshot_registry_json=args.snapshot_registry_json,
            summary_json=args.summary_json,
            top_k=int(args.top_k),
            top_actions=int(args.top_actions),
            allow_policy_id_mismatch=bool(args.allow_policy_id_mismatch),
            accepted_snapshot_config_hashes=tuple(str(value) for value in args.accept_snapshot_config_hash),
        ),
        payload=evaluation_workflow_payload(
            command=B2_AUDIT_COMMAND,
            run_dir=run_dir,
            episodes_jsonl=Path(args.episodes_jsonl).as_posix(),
            policy_id=str(args.policy_id),
            top_k=int(args.top_k),
            top_actions=int(args.top_actions),
            allow_policy_id_mismatch=bool(args.allow_policy_id_mismatch),
        ),
    )


def build_b2_audit_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> EvaluationWorkflowPlan:
    return build_b2_audit_workflow_plan_for_request(
        evaluation_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = ["build_b2_audit_workflow_plan", "build_b2_audit_workflow_plan_for_request"]
