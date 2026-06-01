"""CLI parser for paired-outcome preference decision reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_decisions_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize paired-outcome preference decisions and same-state conflicts."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--spec-bundle-json",
        type=Path,
        default=None,
        help="Optional spec_bundle.json used to decode action ids into families and slots.",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--top-action-edges", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_decisions_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_decisions_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_decisions_parser",
    "parse_paired_outcome_preference_decisions_args",
]
