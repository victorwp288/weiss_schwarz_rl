"""CLI parser for paired targeted-outcome comparison reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare paired outcomes between two targeted-confirm summaries.")
    parser.add_argument("--baseline-summary-json", required=True, type=Path)
    parser.add_argument("--candidate-summary-json", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument(
        "--learned-opponent",
        action="append",
        default=[],
        help="Learned/champion/hard-negative opponent to group separately. When omitted, shared non-fixed rows are inferred.",
    )
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument(
        "--pair-index-split",
        type=int,
        default=None,
        help="Optional pair-index boundary for first/extension split summaries, e.g. 64 for confirm128.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_compare_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_compare_parser().parse_args(argv)


__all__ = ["build_paired_outcome_compare_parser", "parse_paired_outcome_compare_args"]
