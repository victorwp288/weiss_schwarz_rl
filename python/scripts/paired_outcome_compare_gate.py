#!/usr/bin/env python3
"""Gate paired-outcome compare screens before larger main-league evals."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_compare_gate import (
    PairedOutcomeCompareGateConfig,
    evaluate_paired_outcome_compare_gate,
    write_paired_outcome_compare_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-all-delta-wins", type=int, default=0)
    parser.add_argument("--min-fixed-delta-wins", type=int, default=0)
    parser.add_argument("--min-learned-delta-wins", type=int, default=0)
    parser.add_argument("--max-fixed-row-drop-wins", type=int, default=0)
    parser.add_argument("--max-learned-row-drop-wins", type=int, default=0)
    parser.add_argument("--required-opponent", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_paired_outcome_compare_gate(
        PairedOutcomeCompareGateConfig(
            compare_jsons=tuple(Path(path) for path in args.compare_json),
            min_all_delta_wins=int(args.min_all_delta_wins),
            min_fixed_delta_wins=int(args.min_fixed_delta_wins),
            min_learned_delta_wins=int(args.min_learned_delta_wins),
            max_fixed_row_drop_wins=int(args.max_fixed_row_drop_wins),
            max_learned_row_drop_wins=int(args.max_learned_row_drop_wins),
            required_opponents=tuple(str(item) for item in args.required_opponent),
        )
    )
    write_paired_outcome_compare_gate(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": bool(report["passed"]),
                "failures": report["failures"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
