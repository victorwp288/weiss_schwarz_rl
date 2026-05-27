#!/usr/bin/env python3
"""Detect contradictory paired-swing preferences across replay datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_swing_conflicts import (
    PairedSwingConflictConfig,
    build_paired_swing_conflict_report,
    write_paired_swing_conflict_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--positive-action-source", default="actions")
    parser.add_argument("--negative-action-source", default="teacher_action")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_swing_conflict_report(
        PairedSwingConflictConfig(
            dataset_paths=tuple(Path(path) for path in args.dataset),
            positive_action_source=str(args.positive_action_source),
            negative_action_source=str(args.negative_action_source),
            max_examples=int(args.max_examples),
        )
    )
    write_paired_swing_conflict_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "preference_row_count": report["preference_row_count"],
                "current_state_conflict_count": report["current_state_conflict_count"],
                "history_conflict_count": report["history_conflict_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
