from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.audit_plan import (
    build_b2_audit_workflow_plan,
    build_b2_audit_workflow_plan_for_request,
)
from weiss_rl.workflows.evaluation_workflow.eval_plan import (
    build_eval_workflow_plan,
    build_eval_workflow_plan_for_request,
)
from weiss_rl.workflows.evaluation_workflow.figure_plan import (
    build_figures_workflow_plan,
    build_figures_workflow_plan_for_request,
)
from weiss_rl.workflows.evaluation_workflow.plan_state import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    evaluation_workflow_request,
)

__all__ = [
    "EvaluationWorkflowPlan",
    "EvaluationWorkflowRequest",
    "build_b2_audit_workflow_plan",
    "build_b2_audit_workflow_plan_for_request",
    "build_eval_workflow_plan",
    "build_eval_workflow_plan_for_request",
    "build_evaluation_workflow_plan",
    "build_evaluation_workflow_plan_for_request",
    "build_figures_workflow_plan",
    "build_figures_workflow_plan_for_request",
    "evaluation_workflow_request",
]


def build_evaluation_workflow_plan(
    *,
    args: argparse.Namespace,
    python_exe: str,
) -> EvaluationWorkflowPlan | None:
    return build_evaluation_workflow_plan_for_request(
        evaluation_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


def build_evaluation_workflow_plan_for_request(
    request: EvaluationWorkflowRequest,
) -> EvaluationWorkflowPlan | None:
    if request.command in {"smoke-eval", "eval-final"}:
        return build_eval_workflow_plan_for_request(request)

    if request.command == "figures":
        return build_figures_workflow_plan_for_request(request)

    if request.command == "b2-audit":
        return build_b2_audit_workflow_plan_for_request(request)

    return None
