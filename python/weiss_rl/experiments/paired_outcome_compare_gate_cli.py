"""CLI parser for paired-outcome compare gates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_compare_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate paired-outcome compare screens before larger main-league evals.")
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-all-delta-wins", type=int, default=0)
    parser.add_argument("--min-fixed-delta-wins", type=int, default=0)
    parser.add_argument("--min-learned-delta-wins", type=int, default=0)
    parser.add_argument("--max-fixed-row-drop-wins", type=int, default=0)
    parser.add_argument("--max-learned-row-drop-wins", type=int, default=0)
    parser.add_argument("--required-opponent", action="append", default=[])
    return parser


def parse_paired_outcome_compare_gate_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_compare_gate_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_compare_gate_parser",
    "parse_paired_outcome_compare_gate_args",
]
