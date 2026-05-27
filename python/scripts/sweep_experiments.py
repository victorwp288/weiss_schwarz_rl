from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from weiss_rl.experiments.experiment_launcher import execute_launch_plan, resolve_devices
from weiss_rl.experiments.sweeps import build_sweep_launch_plan, list_sweep_presets


def main() -> None:
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2].parent
    devices = resolve_devices(
        requested_devices=args.device,
        cuda_available=torch.cuda.is_available(),
        cuda_count=torch.cuda.device_count(),
    )
    plan, sweep_payload = build_sweep_launch_plan(
        preset_id=args.preset,
        repo_root=repo_root,
        group_label=args.group_label,
        seeds=[int(seed) for seed in args.seed],
        devices=devices,
        train_args=args.train_arg,
    )
    summary = execute_launch_plan(repo_root=repo_root, plan=plan, dry_run=bool(args.dry_run))
    group_dir = repo_root / "runs" / "launch_groups" / args.group_label
    plan_path = group_dir / "sweep_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(sweep_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Sweep summary: "
        f"preset={args.preset} jobs={len(summary['jobs'])} "
        f"max_parallel_jobs={summary['max_parallel_jobs']} plan={plan_path}"
    )


if __name__ == "__main__":
    main()
