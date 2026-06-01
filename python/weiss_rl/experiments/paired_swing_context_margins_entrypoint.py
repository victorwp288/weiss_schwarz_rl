#!/usr/bin/env python3
"""Emit row-level opponent-context log-prob margins for paired-swing replay rows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_swing_context_margins as _context_margins
from weiss_rl.experiments.paired_swing_context_margins_cli import (
    build_paired_swing_context_margins_parser,
    parse_paired_swing_context_margins_args,
)
from weiss_rl.experiments.paired_swing_context_margins_reporting import (
    paired_swing_context_margins_output_line,
)
from weiss_rl.experiments.paired_swing_context_margins_runtime import (
    run_paired_swing_context_margins,
)

PairedSwingContextMarginConfig = _context_margins.PairedSwingContextMarginConfig
build_paired_swing_context_margin_report = _context_margins.build_paired_swing_context_margin_report
paired_swing_margin_rows_from_packed_scores = _context_margins.paired_swing_margin_rows_from_packed_scores
write_paired_swing_context_margin_report = _context_margins.write_paired_swing_context_margin_report
_build_parser = build_paired_swing_context_margins_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_swing_context_margins_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_swing_context_margins(parse_args(argv))
    print(paired_swing_context_margins_output_line(output_json=result.output_json, report=result.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
