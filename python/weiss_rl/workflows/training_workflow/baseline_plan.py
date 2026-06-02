from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.training_workflow.command_builders import _train_command
from weiss_rl.workflows.training_workflow.plan_state import (
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    training_workflow_request,
)
from weiss_rl.workflows.training_workflow.profiles import B1_STACK_CONFIG, TRAIN_PROFILES


def build_b1_training_workflow_plan_for_request(request: TrainingWorkflowRequest) -> TrainingWorkflowPlan:
    args = request.args
    profile = TRAIN_PROFILES[str(args.profile)]
    return TrainingWorkflowPlan(
        plan_name=str(args.run_label),
        command=_train_command(
            python_exe=request.python_exe,
            stack_config=B1_STACK_CONFIG,
            run_label=str(args.run_label),
            profile=profile,
        ),
        payload={"workflow": "train-b1", "profile": str(args.profile)},
    )


def build_b1_training_workflow_plan(*, args: argparse.Namespace, python_exe: str) -> TrainingWorkflowPlan:
    return build_b1_training_workflow_plan_for_request(
        training_workflow_request(args=args, repo_root=Path(), python_exe=python_exe)
    )


__all__ = [
    "build_b1_training_workflow_plan",
    "build_b1_training_workflow_plan_for_request",
]
