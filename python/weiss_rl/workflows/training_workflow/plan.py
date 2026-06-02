from __future__ import annotations

import argparse
from pathlib import Path

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
    if request.command == "train-b1":
        return build_b1_training_workflow_plan_for_request(request)

    if request.command == "train-main":
        return build_main_training_workflow_plan_for_request(request)

    return None
