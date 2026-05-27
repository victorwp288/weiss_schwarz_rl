#!/usr/bin/env python3
"""Summarize paired outcome swings and repair targets from compare artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--opponent-pool-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument("--learned-opponent", action="append", default=[])
    parser.add_argument("--max-examples-per-bucket", type=int, default=24)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str((Path.cwd() / "python").resolve()))
    from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
    from weiss_rl.experiments.paired_swing_report import (
        PairedSwingReportConfig,
        build_paired_swing_report,
        write_paired_swing_report,
    )

    report = build_paired_swing_report(
        PairedSwingReportConfig(
            compare_jsons=tuple(path.resolve() for path in args.compare_json),
            opponent_pool_jsonls=tuple(path.resolve() for path in args.opponent_pool_jsonl),
            fixed_opponents=tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS),
            learned_opponents=tuple(str(item) for item in args.learned_opponent),
            max_examples_per_bucket=int(args.max_examples_per_bucket),
            notes=str(args.notes),
        )
    )
    write_paired_swing_report(report, args.output_json)
    aggregate = report["aggregate"]["groups"]
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "all_delta_wins": aggregate["all"]["delta_wins"],
                "fixed_delta_wins": aggregate["fixed"]["delta_wins"],
                "learned_delta_wins": aggregate["learned"]["delta_wins"],
                "hard_negative_delta_wins": aggregate["hard_negative"]["delta_wins"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
