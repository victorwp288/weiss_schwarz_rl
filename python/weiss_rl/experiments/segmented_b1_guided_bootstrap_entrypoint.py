from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.segmented_b1_guided_bootstrap import (
    DEFAULT_ALIAS_POLICY_ID,
    DEFAULT_RUN_LABEL,
    DEFAULT_STACK_CONFIG,
    SegmentedBootstrapConfig,
    SegmentRuntime,
    run_segmented_b1_guided_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmented guided-bootstrap B1 continuation controller")
    parser.add_argument("--initial-run-dir", type=Path, required=True)
    parser.add_argument("--initial-policy-id", default="guided_bootstrap_floor_selected")
    parser.add_argument("--seed-run-dir", type=Path, default=None)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--alias-policy-id", default=DEFAULT_ALIAS_POLICY_ID)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment-updates", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=288)
    parser.add_argument("--unroll-length", type=int, default=64)
    parser.add_argument("--runtime-mode", default="train_async_fast")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-interval-updates", type=int, default=5)
    parser.add_argument("--collection-backend", default="process")
    parser.add_argument("--no-profile-timers", action="store_true")
    parser.add_argument("--confirm-paired-seeds", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-required-anchor-score", type=float, default=0.5)
    parser.add_argument("--max-selected-drop", type=float, default=0.02)
    parser.add_argument("--max-latest-drop", type=float, default=0.05)
    parser.add_argument(
        "--stop-on-latest-falloff",
        action="store_true",
        help="Stop after a segment if latest fell behind best instead of reanchoring to the selected alias.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = SegmentRuntime(
        num_envs=int(args.num_envs),
        unroll_length=int(args.unroll_length),
        segment_updates=int(args.segment_updates),
        runtime_mode=str(args.runtime_mode),
        simulator_profile=str(args.profile),
        device=str(args.device),
        checkpoint_interval_updates=int(args.checkpoint_interval_updates),
        collection_backend=str(args.collection_backend),
        profile_timers=not bool(args.no_profile_timers),
    )
    config = SegmentedBootstrapConfig(
        repo_root=Path.cwd(),
        initial_run_dir=Path(args.initial_run_dir),
        initial_policy_id=str(args.initial_policy_id),
        seed_run_dir=args.seed_run_dir,
        run_prefix=str(args.run_prefix),
        stack_config=Path(args.stack_config),
        alias_policy_id=str(args.alias_policy_id),
        segments=int(args.segments),
        runtime=runtime,
        confirm_paired_seeds=int(args.confirm_paired_seeds),
        bootstrap_samples=int(args.bootstrap_samples),
        min_required_anchor_score=float(args.min_required_anchor_score),
        max_selected_drop=float(args.max_selected_drop),
        max_latest_drop=float(args.max_latest_drop),
        stop_on_latest_falloff=bool(args.stop_on_latest_falloff),
        dry_run=bool(args.dry_run),
    )
    summary = run_segmented_b1_guided_bootstrap(config)
    print(json.dumps({"status": summary["status"], "summary_path": summary["summary_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
