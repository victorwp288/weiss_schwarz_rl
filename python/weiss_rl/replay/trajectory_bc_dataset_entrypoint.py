from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.replay.trajectory_bc import (
    build_replay_trajectory_bc_dataset,
    load_teacher_action_overrides_jsonl,
    save_replay_trajectory_bc_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a targeted replay trajectory behavior-cloning dataset from captured replay bundles"
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Training stack config used for action guards")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Source run directory containing spec_bundle.json for action/observation decoding",
    )
    parser.add_argument(
        "--audit-run-dir",
        type=Path,
        default=None,
        help="Audit run directory containing audit/replay_bundles and audit/episodes.jsonl",
    )
    parser.add_argument(
        "--replay-bundle",
        type=Path,
        action="append",
        default=[],
        help="Replay bundle zip to include. May be repeated; used in addition to --audit-run-dir bundles.",
    )
    parser.add_argument(
        "--bundle-glob",
        default="*.zip",
        help="Glob under <audit-run-dir>/audit/replay_bundles when --audit-run-dir is provided",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=None,
        help="Seat-swapped episodes.jsonl used to infer focal seat/outcome. Defaults to <audit-run-dir>/audit/episodes.jsonl.",
    )
    parser.add_argument(
        "--include-outcome",
        action="append",
        default=None,
        help="Focal outcome token to include. Repeat for multiple; default is W. Pass ALL to disable outcome filtering.",
    )
    parser.add_argument(
        "--focal-seat",
        type=int,
        default=None,
        choices=(0, 1),
        help="Explicit focal seat when episodes_jsonl is unavailable",
    )
    parser.add_argument("--max-bundles", type=int, default=None, help="Optional cap on selected replay bundles")
    parser.add_argument(
        "--teacher-action-overrides-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL mapping bundle_path/bundle_name + step_index to a teacher_action. "
            "When provided, only override rows are trainable; non-overridden rows preserve recurrent continuity."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npz dataset path")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to output path with .summary.json suffix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bundle_paths = list(args.replay_bundle)
    episodes_jsonl = args.episodes_jsonl
    if args.audit_run_dir is not None:
        bundle_dir = args.audit_run_dir / "audit" / "replay_bundles"
        if not bundle_dir.is_dir():
            parser.error(f"replay bundle directory not found: {bundle_dir}")
        bundle_paths.extend(sorted(bundle_dir.glob(str(args.bundle_glob))))
        if episodes_jsonl is None:
            episodes_jsonl = args.audit_run_dir / "audit" / "episodes.jsonl"
    if not bundle_paths:
        parser.error("provide --audit-run-dir or at least one --replay-bundle")
    if episodes_jsonl is not None and not episodes_jsonl.is_file():
        parser.error(f"episodes_jsonl not found: {episodes_jsonl}")

    include_outcomes = tuple(args.include_outcome or ("W",))
    if any(str(item).strip().upper() == "ALL" for item in include_outcomes):
        include_outcomes = ()
    teacher_action_overrides = None
    if args.teacher_action_overrides_jsonl is not None:
        if not args.teacher_action_overrides_jsonl.is_file():
            parser.error(f"teacher_action_overrides_jsonl not found: {args.teacher_action_overrides_jsonl}")
        teacher_action_overrides = load_teacher_action_overrides_jsonl(args.teacher_action_overrides_jsonl)

    dataset = build_replay_trajectory_bc_dataset(
        bundle_paths=bundle_paths,
        run_dir=args.run_dir,
        stack=args.stack_config,
        episodes_jsonl=episodes_jsonl,
        include_outcomes=include_outcomes,
        focal_seat=args.focal_seat,
        max_bundles=args.max_bundles,
        teacher_action_overrides=teacher_action_overrides,
    )
    save_replay_trajectory_bc_dataset(args.output, dataset)
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dataset.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Replay trajectory BC dataset written to "
        f"{args.output} with {dataset.metadata['train_rows']} train rows across "
        f"{dataset.metadata['bundle_count']} bundles; summary written to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
