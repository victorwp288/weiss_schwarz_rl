"""Evaluation, figure, and diagnostic workflow handlers."""

from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_builders import (
    build_b2_audit_command,
    build_eval_command,
    build_figures_command,
    build_guard_run_command,
)
from weiss_rl.workflows.plans import run_or_write_workflow_plan

__all__ = [
    "dispatch_b2_audit",
    "dispatch_eval",
    "dispatch_figures",
    "dispatch_guard_run",
]


def dispatch_eval(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    run_dir = Path(args.run_dir)
    command = build_eval_command(
        python_exe=python_exe,
        run_dir=run_dir,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        smoke=args.command == "smoke-eval",
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{run_dir.name}_{args.command}",
        command=command,
        dry_run=bool(args.dry_run),
        workflow=str(args.command),
    )


def dispatch_figures(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    run_dir = Path(args.run_dir)
    command = build_figures_command(
        python_exe=python_exe,
        run_dir=run_dir,
        fig_id=str(args.fig_id),
        formats=tuple(args.formats or ()),
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{run_dir.name}_figures",
        command=command,
        dry_run=bool(args.dry_run),
        workflow="figures",
    )


def dispatch_b2_audit(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    run_dir = Path(args.run_dir)
    command = build_b2_audit_command(
        python_exe=python_exe,
        run_dir=run_dir,
        episodes_jsonl=Path(args.episodes_jsonl),
        policy_id=str(args.policy_id),
        output_run_dir=args.output_run_dir,
        snapshot_registry_json=args.snapshot_registry_json,
        summary_json=args.summary_json,
        top_k=int(args.top_k),
        top_actions=int(args.top_actions),
        allow_policy_id_mismatch=bool(args.allow_policy_id_mismatch),
        accepted_snapshot_config_hashes=tuple(str(value) for value in args.accept_snapshot_config_hash),
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{run_dir.name}_b2-audit",
        command=command,
        dry_run=bool(args.dry_run),
        workflow="b2-audit",
    )


def dispatch_guard_run(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    run_dir = Path(args.run_dir)
    command = build_guard_run_command(
        python_exe=python_exe,
        run_dir=run_dir,
        required_anchors=args.required_anchor,
        min_latest_anchor_score=float(args.min_latest_anchor_score),
        max_latest_drop=float(args.max_latest_drop),
        require_promotion_pass_after_attempts=int(args.require_promotion_pass_after_attempts),
        max_consecutive_promotion_failures=int(args.max_consecutive_promotion_failures),
        max_vtrace_rho_p99=args.max_vtrace_rho_p99,
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{run_dir.name}_guard-run",
        command=command,
        dry_run=bool(args.dry_run),
        workflow="guard-run",
    )
