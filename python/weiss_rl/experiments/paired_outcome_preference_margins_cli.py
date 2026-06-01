"""CLI parser for paired-outcome preference margin reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_margins_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit mechanistic DPO-style margins for paired outcome preference replay data."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--aggregation", choices=("mean", "sum"), default="mean")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_margins_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_margins_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_margins_parser",
    "parse_paired_outcome_preference_margins_args",
]
