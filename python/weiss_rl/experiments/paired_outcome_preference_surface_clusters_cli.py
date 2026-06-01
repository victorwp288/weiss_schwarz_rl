"""CLI parser for paired-outcome preference surface-cluster diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_surface_cluster_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify paired-outcome preference rows by public-surface separability."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--spec-bundle-json",
        type=Path,
        default=None,
        help="Optional spec_bundle.json used to decode action ids into families and slots.",
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        help="Optional training config used to read model.opponent_context_policy_ids.",
    )
    parser.add_argument(
        "--opponent-context-policy-id",
        action="append",
        default=[],
        help="Additional policy id that maps to a nonzero opponent-context index.",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_surface_cluster_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_paired_outcome_preference_surface_cluster_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_surface_cluster_parser",
    "parse_paired_outcome_preference_surface_cluster_args",
]
