"""CLI parser for paired-swing mechanistic gates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_swing_mechanistic_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate paired-swing pre/post margin diagnostics before game eval.")
    parser.add_argument("--pre-report-json", required=True, type=Path)
    parser.add_argument("--post-report-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-min-delta", type=float, default=0.0)
    parser.add_argument("--min-row-improved-fraction", type=float, default=0.60)
    parser.add_argument("--max-row-worsened-fraction", type=float, default=0.15)
    parser.add_argument("--min-top-positive-delta", type=int, default=0)
    parser.add_argument("--max-positive-rank-worsened-fraction", type=float, default=0.05)
    parser.add_argument("--min-protected-label-mean-delta", type=float, default=0.0)
    parser.add_argument("--protected-label-contains", action="append", default=["preserve"])
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def parse_paired_swing_mechanistic_gate_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_swing_mechanistic_gate_parser().parse_args(argv)


__all__ = [
    "build_paired_swing_mechanistic_gate_parser",
    "parse_paired_swing_mechanistic_gate_args",
]
