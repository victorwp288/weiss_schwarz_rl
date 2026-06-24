from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.command_surface import TRAIN_B1_COMMAND, TRAIN_MAIN_COMMAND
from weiss_rl.workflows.training_workflow.baseline_plan import (
    build_b1_training_workflow_plan,
    build_b1_training_workflow_plan_for_request,
)
from weiss_rl.workflows.training_workflow.main_plan import (
    build_main_training_workflow_plan,
    build_main_training_workflow_plan_for_request,
)
from weiss_rl.workflows.training_workflow.plan_state import (
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    training_workflow_request,
)

__all__ = [
    "TrainingWorkflowPlan",
    "TrainingWorkflowRequest",
    "build_b1_training_workflow_plan",
    "build_b1_training_workflow_plan_for_request",
    "build_main_training_workflow_plan",
    "build_main_training_workflow_plan_for_request",
    "build_training_workflow_plan",
    "build_training_workflow_plan_for_request",
    "training_workflow_request",
]

TrainingPlanBuilder = Callable[[TrainingWorkflowRequest], TrainingWorkflowPlan]

TRAINING_PLAN_BUILDERS: dict[str, TrainingPlanBuilder] = {
    TRAIN_B1_COMMAND.name: build_b1_training_workflow_plan_for_request,
    TRAIN_MAIN_COMMAND.name: build_main_training_workflow_plan_for_request,
}


def build_training_workflow_plan(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> TrainingWorkflowPlan | None:
    return build_training_workflow_plan_for_request(
        training_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


def build_training_workflow_plan_for_request(request: TrainingWorkflowRequest) -> TrainingWorkflowPlan | None:
    builder = TRAINING_PLAN_BUILDERS.get(request.command)
    return None if builder is None else builder(request)
