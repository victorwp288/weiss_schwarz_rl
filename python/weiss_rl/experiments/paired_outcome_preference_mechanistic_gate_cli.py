"""CLI parser for paired-outcome preference mechanistic gates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_mechanistic_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate paired-outcome preference margin diagnostics before game eval.")
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


def parse_paired_outcome_preference_mechanistic_gate_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_paired_outcome_preference_mechanistic_gate_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_mechanistic_gate_parser",
    "parse_paired_outcome_preference_mechanistic_gate_args",
]
