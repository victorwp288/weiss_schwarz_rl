#!/usr/bin/env python3
"""Report paired-seed overlaps between fixed and learned outcome flips."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_overlap_report import (
    PairedOutcomeOverlapReportConfig,
    build_paired_outcome_overlap_report,
    write_paired_outcome_overlap_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--max-examples-per-key", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_overlap_report(
        PairedOutcomeOverlapReportConfig(
            compare_json_paths=tuple(Path(path) for path in args.compare_json),
            max_examples_per_key=int(args.max_examples_per_key),
        )
    )
    write_paired_outcome_overlap_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "report_count": report["report_count"],
                "total_conflict_key_count": report["total_conflict_key_count"],
                "total_truncated_rows": report["total_truncated_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
