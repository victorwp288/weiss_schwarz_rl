from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.experiment_launcher import LaunchPlan


class ResolveDevicesFn(Protocol):
    def __call__(
        self,
        *,
        requested_devices: list[str] | None,
        cuda_available: bool,
        cuda_count: int,
    ) -> tuple[str, ...]: ...


class BuildLaunchPlanFn(Protocol):
    def __call__(
        self,
        *,
        group_label: str,
        stack_configs: list[str],
        seeds: list[int],
        devices: tuple[str, ...],
        run_label_prefix: str | None,
        extra_args: list[str] | None,
    ) -> LaunchPlan: ...


class ExecuteLaunchPlanFn(Protocol):
    def __call__(self, *, repo_root: Path, plan: LaunchPlan, dry_run: bool) -> dict[str, Any]: ...


def build_launch_experiments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a single-node run group across CPU or multiple local GPUs")
    parser.add_argument("--group-label", required=True, help="Stable label for the run group summary artifact")
    parser.add_argument(
        "--stack-config",
        action="append",
        required=True,
        help="Stack config to train. Repeat to launch multiple stacks in one group.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        required=True,
        help="Seed to launch. Repeat to launch multiple seeds.",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=None,
        help="Optional device list such as cuda:0, cuda:1, or cpu. Defaults to all local CUDA devices or cpu.",
    )
    parser.add_argument(
        "--run-label-prefix",
        default="",
        help="Optional prefix for child run labels. Defaults to the group label.",
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=None,
        help="Extra raw argument to forward to train.py. Repeat per token.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the run-group plan without spawning train.py")
    return parser


def run_launch_experiments_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    cuda_available: bool,
    cuda_count: int,
    resolve_devices_fn: ResolveDevicesFn,
    build_launch_plan_fn: BuildLaunchPlanFn,
    execute_launch_plan_fn: ExecuteLaunchPlanFn,
) -> dict[str, Any]:
    devices = resolve_devices_fn(
        requested_devices=args.device,
        cuda_available=cuda_available,
        cuda_count=cuda_count,
    )
    plan = build_launch_plan_fn(
        group_label=args.group_label,
        stack_configs=[str(Path(path).resolve()) for path in args.stack_config],
        seeds=[int(seed) for seed in args.seed],
        devices=devices,
        run_label_prefix=args.run_label_prefix or None,
        extra_args=args.train_arg,
    )
    return execute_launch_plan_fn(repo_root=repo_root, plan=plan, dry_run=bool(args.dry_run))


def launch_summary_line(summary: dict[str, Any]) -> str:
    return (
        "Launch group summary: "
        f"group={summary['group_label']} jobs={len(summary['jobs'])} "
        f"max_parallel_jobs={summary['max_parallel_jobs']} dry_run={summary['dry_run']}"
    )
