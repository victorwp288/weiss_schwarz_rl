from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.experiment_launcher import LaunchPlan
from weiss_rl.experiments.sweeps import list_sweep_presets


class ResolveDevicesFn(Protocol):
    def __call__(
        self,
        *,
        requested_devices: list[str] | None,
        cuda_available: bool,
        cuda_count: int,
    ) -> tuple[str, ...]: ...


class BuildSweepLaunchPlanFn(Protocol):
    def __call__(
        self,
        *,
        preset_id: str,
        repo_root: Path,
        group_label: str,
        seeds: list[int],
        devices: tuple[str, ...],
        train_args: list[str] | None,
    ) -> tuple[LaunchPlan, dict[str, Any]]: ...


class ExecuteLaunchPlanFn(Protocol):
    def __call__(self, *, repo_root: Path, plan: LaunchPlan, dry_run: bool) -> dict[str, Any]: ...


def build_sweep_experiments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact reproducible hyperparameter sweep on one node")
    parser.add_argument("--preset", required=True, choices=list_sweep_presets())
    parser.add_argument("--group-label", required=True, help="Stable run-group label for the sweep")
    parser.add_argument("--seed", action="append", type=int, required=True, help="Repeatable seed list for the sweep")
    parser.add_argument(
        "--device",
        action="append",
        default=None,
        help="Optional device list such as cuda:0, cuda:1, or cpu. Defaults to all visible local CUDA devices or cpu.",
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=None,
        help="Extra raw argument token to forward to train.py for every sweep run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the sweep plan without spawning training runs")
    return parser


def run_sweep_experiments_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    cuda_available: bool,
    cuda_count: int,
    resolve_devices_fn: ResolveDevicesFn,
    build_sweep_launch_plan_fn: BuildSweepLaunchPlanFn,
    execute_launch_plan_fn: ExecuteLaunchPlanFn,
) -> tuple[dict[str, Any], Path]:
    devices = resolve_devices_fn(
        requested_devices=args.device,
        cuda_available=cuda_available,
        cuda_count=cuda_count,
    )
    plan, sweep_payload = build_sweep_launch_plan_fn(
        preset_id=args.preset,
        repo_root=repo_root,
        group_label=args.group_label,
        seeds=[int(seed) for seed in args.seed],
        devices=devices,
        train_args=args.train_arg,
    )
    summary = execute_launch_plan_fn(repo_root=repo_root, plan=plan, dry_run=bool(args.dry_run))
    plan_path = write_sweep_plan(repo_root=repo_root, group_label=str(args.group_label), sweep_payload=sweep_payload)
    return summary, plan_path


def write_sweep_plan(*, repo_root: Path, group_label: str, sweep_payload: dict[str, Any]) -> Path:
    plan_path = repo_root / "runs" / "launch_groups" / group_label / "sweep_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(sweep_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


def sweep_summary_line(*, preset: str, summary: dict[str, Any], plan_path: Path) -> str:
    return (
        "Sweep summary: "
        f"preset={preset} jobs={len(summary['jobs'])} "
        f"max_parallel_jobs={summary['max_parallel_jobs']} plan={plan_path}"
    )
