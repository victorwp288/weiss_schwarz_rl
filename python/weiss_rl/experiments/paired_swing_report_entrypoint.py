#!/usr/bin/env python3
"""Summarize paired outcome swings and repair targets from compare artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_swing_report as _paired_swing_report
from weiss_rl.experiments.paired_swing_report_cli import (
    build_paired_swing_report_parser,
    parse_paired_swing_report_args,
)
from weiss_rl.experiments.paired_swing_report_reporting import paired_swing_report_output_line
from weiss_rl.experiments.paired_swing_report_runtime import run_paired_swing_report

PairedSwingReportConfig = _paired_swing_report.PairedSwingReportConfig
build_paired_swing_report = _paired_swing_report.build_paired_swing_report
write_paired_swing_report = _paired_swing_report.write_paired_swing_report
_build_parser = build_paired_swing_report_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_swing_report_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_paired_swing_report(parse_args(argv))
    print(paired_swing_report_output_line(output_json=result.output_json, report=result.report))


if __name__ == "__main__":
    main()
