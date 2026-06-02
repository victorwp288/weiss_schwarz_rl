# ruff: noqa: F401,I001

from __future__ import annotations

from collections.abc import MutableMapping

from weiss_rl.workflows.evaluation_workflow.commands import (
    EVAL_STACK_CONFIG,
    _b2_audit_command,
    _eval_command,
    _figures_command,
)
from weiss_rl.workflows.evaluation_workflow.plan import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    build_evaluation_workflow_plan,
    evaluation_workflow_request,
)
from weiss_rl.workflows.planning import _display, _repo_root, _run_or_plan, _write_plan
from weiss_rl.workflows.training_workflow.commands import (
    B1_STACK_CONFIG,
    MAIN_STACK_CONFIG,
    TRAIN_PROFILES,
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    TrainProfile,
    _resolve_b1_seed_checkpoint_path,
    _run_relative_path,
    _train_command,
    build_training_workflow_plan,
    training_workflow_request,
)
from weiss_rl.workflows.workflow_dispatch import (
    WorkflowDispatchRequest,
    dispatch_workflow_command,
    dispatch_workflow_request,
    workflow_dispatch_request,
)

PUBLIC_WORKFLOW_EXPORTS = (
    "B1_STACK_CONFIG",
    "EVAL_STACK_CONFIG",
    "EvaluationWorkflowPlan",
    "EvaluationWorkflowRequest",
    "MAIN_STACK_CONFIG",
    "TRAIN_PROFILES",
    "TrainProfile",
    "TrainingWorkflowPlan",
    "TrainingWorkflowRequest",
    "WorkflowDispatchRequest",
    "_b2_audit_command",
    "_display",
    "_eval_command",
    "_figures_command",
    "_repo_root",
    "_resolve_b1_seed_checkpoint_path",
    "_run_or_plan",
    "_run_relative_path",
    "_train_command",
    "_write_plan",
    "build_evaluation_workflow_plan",
    "build_training_workflow_plan",
    "dispatch_workflow_command",
    "dispatch_workflow_request",
    "evaluation_workflow_request",
    "training_workflow_request",
    "workflow_dispatch_request",
)

__all__ = [*PUBLIC_WORKFLOW_EXPORTS, "export_public_workflow_symbols"]


def export_public_workflow_symbols(target_globals: MutableMapping[str, object]) -> None:
    for name in PUBLIC_WORKFLOW_EXPORTS:
        target_globals[name] = globals()[name]
