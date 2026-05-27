#!/usr/bin/env python3
"""Gate paired-outcome preference margin diagnostics before game eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate import (
    PairedOutcomePreferenceMechanisticGateConfig,
    evaluate_paired_outcome_preference_mechanistic_gate,
    write_paired_outcome_preference_mechanistic_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-report-json", required=True, type=Path)
    parser.add_argument("--post-report-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-min-delta", type=float, default=0.0)
    parser.add_argument("--min-pair-improved-fraction", type=float, default=1.0)
    parser.add_argument("--max-pair-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--min-group-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-required-group-mean-delta", type=float, default=0.0)
    parser.add_argument("--required-group", action="append", default=[])
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_paired_outcome_preference_mechanistic_gate(
        PairedOutcomePreferenceMechanisticGateConfig(
            pre_report_json=args.pre_report_json,
            post_report_json=args.post_report_json,
            min_mean_delta=float(args.min_mean_delta),
            min_min_delta=float(args.min_min_delta),
            min_pair_improved_fraction=float(args.min_pair_improved_fraction),
            max_pair_worsened_fraction=float(args.max_pair_worsened_fraction),
            min_group_mean_delta=float(args.min_group_mean_delta),
            min_required_group_mean_delta=float(args.min_required_group_mean_delta),
            required_groups=tuple(str(item) for item in args.required_group),
            require_context=not bool(args.allow_missing_context),
        )
    )
    write_paired_outcome_preference_mechanistic_gate(args.output_json, report)
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
