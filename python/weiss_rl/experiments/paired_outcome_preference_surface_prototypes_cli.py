"""CLI parser for paired-outcome preference surface-prototype diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_surface_prototype_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report exact prototype-key coverage across paired-outcome preference datasets."
    )
    parser.add_argument("--prototype-dataset", required=True, type=Path)
    parser.add_argument("--probe-dataset", action="append", required=True, type=Path)
    parser.add_argument(
        "--probe-label",
        action="append",
        default=[],
        help="Optional label for each --probe-dataset, in the same order.",
    )
    parser.add_argument(
        "--key-mode",
        choices=("current", "current_history", "current_history_opponent"),
        default="current_history_opponent",
    )
    parser.add_argument(
        "--opponent-key-mode",
        choices=("raw_policy_id", "context_index"),
        default="raw_policy_id",
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        help="Optional stack YAML used to resolve model opponent context indices.",
    )
    parser.add_argument(
        "--opponent-context-policy-id",
        action="append",
        default=[],
        help="Additional opponent context policy id used when --opponent-key-mode=context_index.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_surface_prototype_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_paired_outcome_preference_surface_prototype_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_surface_prototype_parser",
    "parse_paired_outcome_preference_surface_prototype_args",
]
