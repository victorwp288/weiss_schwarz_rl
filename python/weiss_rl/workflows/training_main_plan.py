from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.training_command_builders import _train_command
from weiss_rl.workflows.training_plan_state import (
    TrainingWorkflowPlan,
    TrainingWorkflowRequest,
    training_workflow_request,
)
from weiss_rl.workflows.training_profiles import MAIN_STACK_CONFIG, TRAIN_PROFILES, _guided_bootstrap_stack_config
from weiss_rl.workflows.training_snapshot_resolution import (
    _resolve_b1_seed_checkpoint_path,
    _resolve_snapshot_checkpoint_path,
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


def build_main_guided_bootstrap_training_workflow_plan(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> TrainingWorkflowPlan:
    return build_main_guided_bootstrap_training_workflow_plan_for_request(
        training_workflow_request(args=args, repo_root=repo_root, python_exe=python_exe)
    )


def build_main_guided_bootstrap_training_workflow_plan_for_request(
    request: TrainingWorkflowRequest,
) -> TrainingWorkflowPlan:
    args = request.args
    profile = TRAIN_PROFILES[str(args.profile)]
    init_from_checkpoint = args.init_from_checkpoint
    if init_from_checkpoint is None:
        if args.init_from_run_dir is None or not str(args.init_policy_id).strip():
            raise SystemExit(
                "train-main-guided-bootstrap requires either --init-from-checkpoint or "
                "--init-from-run-dir plus --init-policy-id"
            )
        init_from_checkpoint = _resolve_snapshot_checkpoint_path(
            repo_root=request.repo_root,
            run_dir=Path(args.init_from_run_dir),
            policy_id=str(args.init_policy_id),
        )
    elif args.init_from_run_dir is not None or str(args.init_policy_id).strip():
        raise SystemExit("--init-from-checkpoint cannot be combined with --init-from-run-dir/--init-policy-id")
    return TrainingWorkflowPlan(
        plan_name=str(args.run_label),
        command=_train_command(
            python_exe=request.python_exe,
            stack_config=_guided_bootstrap_stack_config(
                vtrace_clamp=bool(args.vtrace_clamp),
                seed_champions=bool(args.seed_champions),
                selected_seed_champion=bool(args.selected_seed_champion),
            ),
            run_label=str(args.run_label),
            profile=profile,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
            init_from_checkpoint=Path(init_from_checkpoint),
        ),
        payload={
            "workflow": "train-main-guided-bootstrap",
            "profile": str(args.profile),
            "vtrace_clamp": bool(args.vtrace_clamp),
            "seed_champions": bool(args.seed_champions),
            "selected_seed_champion": bool(args.selected_seed_champion),
            "init_policy_id": str(args.init_policy_id).strip() or None,
        },
    )


__all__ = [
    "build_main_guided_bootstrap_training_workflow_plan",
    "build_main_guided_bootstrap_training_workflow_plan_for_request",
    "build_main_training_workflow_plan",
    "build_main_training_workflow_plan_for_request",
]
