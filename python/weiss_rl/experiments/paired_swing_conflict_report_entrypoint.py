#!/usr/bin/env python3
"""Detect contradictory paired-swing preferences across replay datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_swing_conflicts as _conflicts
from weiss_rl.experiments.paired_swing_conflict_report_cli import (
    build_paired_swing_conflict_report_parser,
    parse_paired_swing_conflict_report_args,
)
from weiss_rl.experiments.paired_swing_conflict_report_reporting import (
    paired_swing_conflict_report_output_line,
)
from weiss_rl.experiments.paired_swing_conflict_report_runtime import (
    run_paired_swing_conflict_report,
)

PairedSwingConflictConfig = _conflicts.PairedSwingConflictConfig
build_paired_swing_conflict_report = _conflicts.build_paired_swing_conflict_report
write_paired_swing_conflict_report = _conflicts.write_paired_swing_conflict_report
_build_parser = build_paired_swing_conflict_report_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_swing_conflict_report_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_swing_conflict_report(parse_args(argv))
    print(paired_swing_conflict_report_output_line(output_json=result.output_json, report=result.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
