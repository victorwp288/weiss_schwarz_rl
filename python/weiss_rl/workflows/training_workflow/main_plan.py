from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.training_workflow.command_builders import _train_command
from weiss_rl.workflows.training_workflow.plan_state import (
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    training_workflow_request,
)
from weiss_rl.workflows.training_workflow.profiles import (
    MAIN_STACK_CONFIG,
    TRAIN_PROFILES,
)
from weiss_rl.workflows.training_workflow.snapshot_resolution import (
    _resolve_b1_seed_checkpoint_path,
)


def build_main_training_workflow_plan(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> TrainingWorkflowPlan:
    return build_main_training_workflow_plan_for_request(
        training_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


def build_main_training_workflow_plan_for_request(request: TrainingWorkflowRequest) -> TrainingWorkflowPlan:
    args = request.args
    profile = TRAIN_PROFILES[str(args.profile)]
    init_from_checkpoint, resolved_init_policy_id = _resolve_b1_seed_checkpoint_path(
        repo_root=request.repo_root,
        run_dir=Path(args.b1_baseline_run_dir),
        init_policy_id=str(args.init_policy_id),
    )
    return TrainingWorkflowPlan(
        plan_name=str(args.run_label),
        command=_train_command(
            python_exe=request.python_exe,
            stack_config=MAIN_STACK_CONFIG,
            run_label=str(args.run_label),
            profile=profile,
            b1_baseline_run_dir=Path(args.b1_baseline_run_dir),
            seed_snapshot_run_dir=args.seed_snapshot_run_dir,
            init_from_checkpoint=init_from_checkpoint,
        ),
        payload={
            "workflow": "train-main",
            "profile": str(args.profile),
            "init_policy_id": resolved_init_policy_id,
        },
    )


__all__ = [
    "build_main_training_workflow_plan",
    "build_main_training_workflow_plan_for_request",
]
