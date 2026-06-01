"""CLI parser for paired-outcome preference span audits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_span_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report compact trajectory spans in paired-outcome preference replay datasets."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--spec-bundle-json", default=None, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--max-compact-span-width", type=int, default=8)
    parser.add_argument("--min-repeated-pair-count", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def parse_paired_outcome_preference_span_audit_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_span_audit_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_span_audit_parser",
    "parse_paired_outcome_preference_span_audit_args",
]
