from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.command_surface import B2_AUDIT_COMMAND, EVAL_FINAL_COMMAND, FIGURES_COMMAND, SMOKE_EVAL_COMMAND
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

EvaluationPlanBuilder = Callable[[EvaluationWorkflowRequest], EvaluationWorkflowPlan]

EVALUATION_PLAN_BUILDERS: dict[str, EvaluationPlanBuilder] = {
    SMOKE_EVAL_COMMAND.name: build_eval_workflow_plan_for_request,
    EVAL_FINAL_COMMAND.name: build_eval_workflow_plan_for_request,
    FIGURES_COMMAND.name: build_figures_workflow_plan_for_request,
    B2_AUDIT_COMMAND.name: build_b2_audit_workflow_plan_for_request,
}


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
    builder = EVALUATION_PLAN_BUILDERS.get(request.command)
    return None if builder is None else builder(request)
