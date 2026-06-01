"""CLI parser for paired-outcome preference row guards."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_row_guard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate row-level target-action drift on paired outcome preference replay."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--protected-group", action="append", default=[])
    parser.add_argument("--required-group", action="append", default=[])
    parser.add_argument("--min-required-group-mean-logp-delta", type=float, default=0.0)
    parser.add_argument("--min-protected-mean-logp-delta", type=float, default=0.0)
    parser.add_argument("--max-protected-row-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--max-protected-rank-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--max-protected-top-family-changed-rate", type=float, default=0.0)
    parser.add_argument("--top-action-near-tie-margin", type=float, default=1e-5)
    parser.add_argument("--max-protected-lost-target-non-near-tie-rate", type=float, default=0.0)
    parser.add_argument("--allow-missing-context", action="store_true")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_row_guard_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_row_guard_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_row_guard_parser",
    "parse_paired_outcome_preference_row_guard_args",
]
