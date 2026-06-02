from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.replay.trajectory_bc import (
    load_replay_trajectory_bc_dataset,
    merge_replay_trajectory_bc_datasets,
    save_replay_trajectory_bc_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge replay trajectory behavior-cloning datasets along the episode axis"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Source replay trajectory BC .npz dataset. May be repeated.",
    )
    parser.add_argument(
        "--source-label",
        action="append",
        default=None,
        help="Optional label for each --dataset, repeated in the same order.",
    )
    parser.add_argument(
        "--preserve-source-bundle-labels",
        action="store_true",
        help=(
            "Keep existing per-episode source_dataset_label values from already-merged inputs "
            "instead of flattening them under the top-level --source-label."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output merged .npz dataset path")
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
    dataset_paths = [Path(path) for path in args.dataset]
    if len(dataset_paths) < 2:
        parser.error("provide at least two --dataset paths")
    source_labels = tuple(args.source_label or ())
    if source_labels and len(source_labels) != len(dataset_paths):
        parser.error("--source-label count must match --dataset count when labels are provided")
    datasets = []
    for path in dataset_paths:
        if not path.is_file():
            parser.error(f"dataset not found: {path}")
        dataset = load_replay_trajectory_bc_dataset(path)
        dataset.metadata["dataset_path"] = path.resolve().as_posix()
        datasets.append(dataset)
    merged = merge_replay_trajectory_bc_datasets(
        datasets,
        source_labels=source_labels or None,
        preserve_source_bundle_labels=bool(args.preserve_source_bundle_labels),
    )
    save_replay_trajectory_bc_dataset(args.output, merged)
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(merged.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "Merged replay trajectory BC dataset written to "
        f"{args.output} with {merged.metadata['train_rows']} train rows across "
        f"{merged.metadata['bundle_count']} bundles; summary written to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
