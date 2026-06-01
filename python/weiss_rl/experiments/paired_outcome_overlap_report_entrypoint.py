#!/usr/bin/env python3
"""Report paired-seed overlaps between fixed and learned outcome flips."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_overlap_report as _overlap_report
from weiss_rl.experiments.paired_outcome_overlap_report_cli import (
    build_paired_outcome_overlap_report_parser,
    parse_paired_outcome_overlap_report_args,
)
from weiss_rl.experiments.paired_outcome_overlap_report_reporting import (
    paired_outcome_overlap_report_output_line,
)
from weiss_rl.experiments.paired_outcome_overlap_report_runtime import (
    run_paired_outcome_overlap_report,
)

PairedOutcomeOverlapReportConfig = _overlap_report.PairedOutcomeOverlapReportConfig
build_paired_outcome_overlap_report = _overlap_report.build_paired_outcome_overlap_report
write_paired_outcome_overlap_report = _overlap_report.write_paired_outcome_overlap_report
_build_parser = build_paired_outcome_overlap_report_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_overlap_report_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_overlap_report(parse_args(argv))
    print(paired_outcome_overlap_report_output_line(output_json=result.output_json, report=result.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
