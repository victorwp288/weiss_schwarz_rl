"""Guided bootstrap workflow handlers for the package CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.workflows.command_builders import (
    build_guarded_league_bootstrap_command,
    build_guided_bootstrap_loop_command,
)
from weiss_rl.workflows.plans import run_or_write_workflow_plan

__all__ = [
    "dispatch_guarded_league_bootstrap",
    "dispatch_guided_bootstrap_loop",
]


def dispatch_guided_bootstrap_loop(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    command = build_guided_bootstrap_loop_command(
        python_exe=python_exe,
        initial_run_dir=Path(args.initial_run_dir),
        initial_policy_id=str(args.initial_policy_id),
        seed_run_dir=args.seed_run_dir,
        run_prefix=str(args.run_prefix),
        stack_config=Path(args.stack_config),
        alias_policy_id=str(args.alias_policy_id),
        segments=int(args.segments),
        segment_updates=int(args.segment_updates),
        confirm_paired_seeds=int(args.confirm_paired_seeds),
        stop_on_latest_falloff=bool(args.stop_on_latest_falloff),
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{args.run_prefix}_guided-bootstrap-loop",
        command=command,
        dry_run=bool(args.dry_run),
        workflow="guided-bootstrap-loop",
        payload={
            "initial_policy_id": str(args.initial_policy_id),
            "segments": int(args.segments),
            "segment_updates": int(args.segment_updates),
            "confirm_paired_seeds": int(args.confirm_paired_seeds),
        },
    )


def dispatch_guarded_league_bootstrap(args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None:
    if args.first_init_schedule_offset_updates is not None and int(args.first_init_schedule_offset_updates) < 0:
        raise SystemExit("--first-init-schedule-offset-updates must be >= 0")
    if int(args.publish_min_confirm_paired_seeds) < 1:
        raise SystemExit("--publish-min-confirm-paired-seeds must be >= 1")
    if int(args.confirm_recent_candidate_count) < 1:
        raise SystemExit("--confirm-recent-candidate-count must be >= 1")
    command = build_guarded_league_bootstrap_command(
        python_exe=python_exe,
        init_from_checkpoint=Path(args.init_from_checkpoint),
        seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
        run_prefix=str(args.run_prefix),
        stack_config=Path(args.stack_config),
        segments=int(args.segments),
        segment_updates=int(args.segment_updates),
        first_init_schedule_offset_updates=args.first_init_schedule_offset_updates,
        confirm_paired_seeds=int(args.confirm_paired_seeds),
        publish_min_confirm_paired_seeds=int(args.publish_min_confirm_paired_seeds),
        confirm_recent_candidate_count=int(args.confirm_recent_candidate_count),
        reference_summary_json=args.reference_summary_json,
        multiobjective_reference_summary_jsons=tuple(args.multiobjective_reference_summary_json or ()),
        multiobjective_fixed_opponents=tuple(args.multiobjective_fixed_opponent or ()),
        learned_guard_opponents=tuple(args.learned_guard_opponent or ()),
        min_learned_guard_mean=float(args.min_learned_guard_mean),
        min_learned_guard_reference_delta=float(args.min_learned_guard_reference_delta),
        reference_label=str(args.reference_label),
        min_required_anchor_score=float(args.min_required_anchor_score),
        max_reference_drop=float(args.max_reference_drop),
        selected_alias_policy_id=str(args.selected_alias_policy_id),
    )
    run_or_write_workflow_plan(
        repo_root=repo_root,
        plan_name=f"{args.run_prefix}_guarded-league-bootstrap",
        command=command,
        dry_run=bool(args.dry_run),
        workflow="guarded-league-bootstrap",
        payload={
            "segments": int(args.segments),
            "segment_updates": int(args.segment_updates),
            "first_init_schedule_offset_updates": args.first_init_schedule_offset_updates,
            "confirm_paired_seeds": int(args.confirm_paired_seeds),
            "publish_min_confirm_paired_seeds": int(args.publish_min_confirm_paired_seeds),
            "confirm_recent_candidate_count": int(args.confirm_recent_candidate_count),
            "reference_label": str(args.reference_label) if args.reference_summary_json is not None else None,
            "multiobjective_reference_summary_jsons": [
                path.as_posix() for path in tuple(args.multiobjective_reference_summary_json or ())
            ],
            "learned_guard_opponents": list(args.learned_guard_opponent or ()),
            "selected_alias_policy_id": str(args.selected_alias_policy_id),
        },
    )
