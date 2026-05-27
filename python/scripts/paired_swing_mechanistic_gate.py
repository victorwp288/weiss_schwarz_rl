#!/usr/bin/env python3
"""Gate paired-swing pre/post margin diagnostics before game eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_swing_mechanistic_gate import (
    PairedSwingMechanisticGateConfig,
    evaluate_paired_swing_mechanistic_gate,
    write_paired_swing_mechanistic_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-report-json", required=True, type=Path)
    parser.add_argument("--post-report-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-min-delta", type=float, default=0.0)
    parser.add_argument("--min-row-improved-fraction", type=float, default=0.60)
    parser.add_argument("--max-row-worsened-fraction", type=float, default=0.15)
    parser.add_argument("--min-top-positive-delta", type=int, default=0)
    parser.add_argument("--max-positive-rank-worsened-fraction", type=float, default=0.05)
    parser.add_argument("--min-protected-label-mean-delta", type=float, default=0.0)
    parser.add_argument("--protected-label-contains", action="append", default=["preserve"])
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_paired_swing_mechanistic_gate(
        PairedSwingMechanisticGateConfig(
            pre_report_json=args.pre_report_json,
            post_report_json=args.post_report_json,
            min_mean_delta=float(args.min_mean_delta),
            min_min_delta=float(args.min_min_delta),
            min_row_improved_fraction=float(args.min_row_improved_fraction),
            max_row_worsened_fraction=float(args.max_row_worsened_fraction),
            min_top_positive_delta=int(args.min_top_positive_delta),
            max_positive_rank_worsened_fraction=float(args.max_positive_rank_worsened_fraction),
            min_protected_label_mean_delta=float(args.min_protected_label_mean_delta),
            protected_label_contains=tuple(str(item) for item in args.protected_label_contains),
            require_context=not bool(args.allow_missing_context),
        )
    )
    write_paired_swing_mechanistic_gate(args.output_json, report)
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
