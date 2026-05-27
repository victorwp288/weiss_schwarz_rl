from __future__ import annotations

import argparse
from pathlib import Path

import torch
from weiss_rl.experiments.experiment_launcher import build_launch_plan, execute_launch_plan, resolve_devices


def main() -> None:
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2].parent
    devices = resolve_devices(
        requested_devices=args.device,
        cuda_available=torch.cuda.is_available(),
        cuda_count=torch.cuda.device_count(),
    )
    plan = build_launch_plan(
        group_label=args.group_label,
        stack_configs=[str(Path(path).resolve()) for path in args.stack_config],
        seeds=[int(seed) for seed in args.seed],
        devices=devices,
        run_label_prefix=args.run_label_prefix or None,
        extra_args=args.train_arg,
    )
    summary = execute_launch_plan(repo_root=repo_root, plan=plan, dry_run=bool(args.dry_run))
    print(
        "Launch group summary: "
        f"group={summary['group_label']} jobs={len(summary['jobs'])} "
        f"max_parallel_jobs={summary['max_parallel_jobs']} dry_run={summary['dry_run']}"
    )


if __name__ == "__main__":
    main()
