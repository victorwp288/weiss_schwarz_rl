"""CLI parser for paired-outcome overlap reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_overlap_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report paired-seed overlaps between fixed and learned outcome flips.")
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--max-examples-per-key", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_overlap_report_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_overlap_report_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_overlap_report_parser",
    "parse_paired_outcome_overlap_report_args",
]
