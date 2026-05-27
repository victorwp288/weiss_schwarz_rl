#!/usr/bin/env python3
"""Filter paired-swing replay episodes by source metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_swing_filters import (
    PairedSwingEpisodeFilterConfig,
    filter_paired_swing_dataset,
    write_paired_swing_filter_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-pair-index", action="append", default=[], type=int)
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument("--positive-action-source", default="actions")
    parser.add_argument("--negative-action-source", default="teacher_action")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--allow-zero-distinct-rows", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _dataset, summary = filter_paired_swing_dataset(
        PairedSwingEpisodeFilterConfig(
            dataset_path=args.dataset,
            output_dataset_path=args.output,
            source_pair_indices=tuple(int(index) for index in args.source_pair_index),
            source_labels=tuple(str(label) for label in args.source_label),
            positive_action_source=str(args.positive_action_source),
            negative_action_source=str(args.negative_action_source),
            require_distinct_train_rows=not bool(args.allow_zero_distinct_rows),
        )
    )
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    write_paired_swing_filter_summary(summary_path, summary)
    print(json.dumps({"summary_json": summary_path.as_posix(), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
