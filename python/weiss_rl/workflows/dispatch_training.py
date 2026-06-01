"""Training workflow handlers for the package CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_builders import build_train_command
from weiss_rl.workflows.plans import run_or_write_workflow_plan
from weiss_rl.workflows.profiles import (
    B1_GUIDED_SEED_STACK_CONFIG,
    B1_STACK_CONFIG,
    MAIN_STACK_CONFIG,
    TRAIN_PROFILES,
    guided_bootstrap_stack_config,
)
from weiss_rl.workflows.snapshots import (
    resolve_b1_seed_checkpoint_path,
    resolve_single_snapshot_checkpoint_path,
    resolve_snapshot_checkpoint_path,
)

__all__ = [
    "dispatch_train_b1",
    "dispatch_train_b1_guided_seed",
    "dispatch_train_main",
    "dispatch_train_main_guided_bootstrap",
]


def _dispatch_simple_training_workflow(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    python_exe: str,
    workflow: str,
    stack_config: Path,
) -> None:
    profile = TRAIN_PROFILES[str(args.profile)]
    command = build_train_command(
        python_exe=python_exe,
        stack_config=stack_config,
        run_label=str(args.run_label),
        profile=profile,
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=str(args.run_label),
        command=command,
        dry_run=bool(args.dry_run),
        workflow=workflow,
        payload={"profile": str(args.profile)},
    )


def dispatch_train_b1(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    _dispatch_simple_training_workflow(
        args,
        repo_root=repo_root,
        python_exe=python_exe,
        workflow="train-b1",
        stack_config=B1_STACK_CONFIG,
    )


def dispatch_train_b1_guided_seed(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    _dispatch_simple_training_workflow(
        args,
        repo_root=repo_root,
        python_exe=python_exe,
        workflow="train-b1-guided-seed",
        stack_config=B1_GUIDED_SEED_STACK_CONFIG,
    )


def _resolve_train_main_init_checkpoint(
    *,
    repo_root: Path,
    b1_baseline_run_dir: Path,
    init_policy_id: str,
    profile_name: str,
) -> tuple[Path, str]:
    try:
        return resolve_b1_seed_checkpoint_path(
            repo_root=repo_root,
            run_dir=b1_baseline_run_dir,
            init_policy_id=init_policy_id,
        )
    except SystemExit:
        if profile_name != "smoke" or init_policy_id.strip() not in {"", "auto"}:
            raise
        return resolve_single_snapshot_checkpoint_path(
            repo_root=repo_root,
            run_dir=b1_baseline_run_dir,
        )


def dispatch_train_main(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    profile_name = str(args.profile)
    profile = TRAIN_PROFILES[profile_name]
    b1_baseline_run_dir = Path(args.b1_baseline_run_dir)
    init_from_checkpoint, resolved_init_policy_id = _resolve_train_main_init_checkpoint(
        repo_root=repo_root,
        b1_baseline_run_dir=b1_baseline_run_dir,
        init_policy_id=str(args.init_policy_id),
        profile_name=profile_name,
    )
    command = build_train_command(
        python_exe=python_exe,
        stack_config=MAIN_STACK_CONFIG,
        run_label=str(args.run_label),
        profile=profile,
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint=init_from_checkpoint,
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=str(args.run_label),
        command=command,
        dry_run=bool(args.dry_run),
        workflow="train-main",
        payload={
            "profile": str(args.profile),
            "init_policy_id": resolved_init_policy_id,
        },
    )


def _resolve_guided_bootstrap_init_checkpoint(
    *,
    repo_root: Path,
    init_from_checkpoint: Path | None,
    init_from_run_dir: Path | None,
    init_policy_id: str,
) -> Path:
    resolved_init_policy_id = init_policy_id.strip()
    if init_from_checkpoint is None:
        if init_from_run_dir is None or not resolved_init_policy_id:
            raise SystemExit(
                "train-main-guided-bootstrap requires either --init-from-checkpoint or "
                "--init-from-run-dir plus --init-policy-id"
            )
        return resolve_snapshot_checkpoint_path(
            repo_root=repo_root,
            run_dir=Path(init_from_run_dir),
            policy_id=resolved_init_policy_id,
        )
    if init_from_run_dir is not None or resolved_init_policy_id:
        raise SystemExit("--init-from-checkpoint cannot be combined with --init-from-run-dir/--init-policy-id")
    return Path(init_from_checkpoint)


def dispatch_train_main_guided_bootstrap(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    profile = TRAIN_PROFILES[str(args.profile)]
    init_from_checkpoint = _resolve_guided_bootstrap_init_checkpoint(
        repo_root=repo_root,
        init_from_checkpoint=args.init_from_checkpoint,
        init_from_run_dir=args.init_from_run_dir,
        init_policy_id=str(args.init_policy_id),
    )
    command = build_train_command(
        python_exe=python_exe,
        stack_config=guided_bootstrap_stack_config(
            vtrace_clamp=bool(args.vtrace_clamp),
            seed_champions=bool(args.seed_champions),
            selected_seed_champion=bool(args.selected_seed_champion),
        ),
        run_label=str(args.run_label),
        profile=profile,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
        init_from_checkpoint=init_from_checkpoint,
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=str(args.run_label),
        command=command,
        dry_run=bool(args.dry_run),
        workflow="train-main-guided-bootstrap",
        payload={
            "profile": str(args.profile),
            "vtrace_clamp": bool(args.vtrace_clamp),
            "seed_champions": bool(args.seed_champions),
            "selected_seed_champion": bool(args.selected_seed_champion),
            "init_policy_id": str(args.init_policy_id).strip() or None,
        },
    )
