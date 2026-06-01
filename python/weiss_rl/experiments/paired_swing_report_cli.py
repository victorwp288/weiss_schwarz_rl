"""CLI parser for paired-swing repair-target reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_swing_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize paired outcome swings and repair targets from compare artifacts."
    )
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--opponent-pool-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument("--learned-opponent", action="append", default=[])
    parser.add_argument("--max-examples-per-bucket", type=int, default=24)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_swing_report_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_swing_report_parser().parse_args(argv)


__all__ = [
    "build_paired_swing_report_parser",
    "parse_paired_swing_report_args",
]
