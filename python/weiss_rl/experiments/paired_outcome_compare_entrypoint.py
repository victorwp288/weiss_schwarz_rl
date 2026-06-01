#!/usr/bin/env python3
"""Compare paired outcomes between two targeted-confirm summaries."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_compare as _compare
from weiss_rl.experiments.paired_outcome_compare_cli import (
    build_paired_outcome_compare_parser,
    parse_paired_outcome_compare_args,
)
from weiss_rl.experiments.paired_outcome_compare_reporting import paired_outcome_compare_output_line
from weiss_rl.experiments.paired_outcome_compare_runtime import run_paired_outcome_compare

PairedOutcomeCompareConfig = _compare.PairedOutcomeCompareConfig
compare_paired_targeted_outcomes = _compare.compare_paired_targeted_outcomes
_build_parser = build_paired_outcome_compare_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_compare_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_paired_outcome_compare(parse_args(argv))
    print(paired_outcome_compare_output_line(report=result.report, output_json=result.output_json))


if __name__ == "__main__":
    main()
