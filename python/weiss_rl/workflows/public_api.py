# ruff: noqa: F401,I001

from __future__ import annotations

from collections.abc import MutableMapping

from weiss_rl.workflows.controller_commands import (
    _guard_run_command,
    _guarded_league_bootstrap_command,
    _guided_bootstrap_loop_command,
)
from weiss_rl.workflows.controller_plan import (
    ControllerWorkflowPlan,
    ControllerWorkflowRequest,
    build_controller_workflow_plan,
    controller_workflow_request,
)
from weiss_rl.workflows.evaluation_commands import (
    EVAL_STACK_CONFIG,
    _b2_audit_command,
    _eval_command,
    _figures_command,
)
from weiss_rl.workflows.evaluation_plan import (
    EvaluationWorkflowPlan,
    EvaluationWorkflowRequest,
    build_evaluation_workflow_plan,
    evaluation_workflow_request,
)
from weiss_rl.workflows.planning import _display, _repo_root, _run_or_plan, _write_plan
from weiss_rl.workflows.training_commands import (
    B1_GUIDED_SEED_STACK_CONFIG,
    B1_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG,
    MAIN_STACK_CONFIG,
    TRAIN_PROFILES,
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    TrainProfile,
    _guided_bootstrap_stack_config,
    _resolve_b1_seed_checkpoint_path,
    _resolve_snapshot_checkpoint_path,
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
    "B1_GUIDED_SEED_STACK_CONFIG",
    "B1_STACK_CONFIG",
    "ControllerWorkflowPlan",
    "ControllerWorkflowRequest",
    "EVAL_STACK_CONFIG",
    "EvaluationWorkflowPlan",
    "EvaluationWorkflowRequest",
    "MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG",
    "MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG",
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
    "_guard_run_command",
    "_guarded_league_bootstrap_command",
    "_guided_bootstrap_loop_command",
    "_guided_bootstrap_stack_config",
    "_repo_root",
    "_resolve_b1_seed_checkpoint_path",
    "_resolve_snapshot_checkpoint_path",
    "_run_or_plan",
    "_run_relative_path",
    "_train_command",
    "_write_plan",
    "build_controller_workflow_plan",
    "build_evaluation_workflow_plan",
    "build_training_workflow_plan",
    "controller_workflow_request",
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
