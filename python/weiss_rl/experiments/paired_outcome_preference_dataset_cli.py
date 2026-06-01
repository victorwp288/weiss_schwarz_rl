"""CLI parser for paired-outcome preference dataset generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_opponent_match_aliases(values: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--opponent-match-alias must be FROM=TO, got: {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target:
            raise SystemExit(f"--opponent-match-alias must have non-empty FROM and TO, got: {value!r}")
        aliases[source] = target
    return aliases


def build_paired_outcome_preference_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge preferred/rejected trajectory datasets into explicit preference replay data."
    )
    parser.add_argument("--preferred-dataset", required=True, type=Path)
    parser.add_argument("--rejected-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--max-pairs", default=None, type=int)
    parser.add_argument("--preferred-label", default="preferred")
    parser.add_argument("--rejected-label", default="rejected")
    parser.add_argument(
        "--opponent-match-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Canonicalize opponent IDs only for preferred/rejected episode matching. "
            "The original source_opponent_policy_id metadata is preserved for context and audit."
        ),
    )
    return parser


def parse_paired_outcome_preference_dataset_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_dataset_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_dataset_parser",
    "parse_opponent_match_aliases",
    "parse_paired_outcome_preference_dataset_args",
]
