"""CLI parser for paired-outcome preference edge-margin gates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_outcome_preference_edge_margins_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate row-level paired-outcome preference edge movement before game eval."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--spec-bundle-json", default=None, type=Path)
    parser.add_argument("--include-same-action", action="store_true")
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-min-delta", type=float, default=0.0)
    parser.add_argument("--min-edge-improved-fraction", type=float, default=1.0)
    parser.add_argument("--max-edge-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--min-same-state-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-required-group-mean-delta", type=float, default=0.0)
    parser.add_argument("--required-group", action="append", default=[])
    parser.add_argument("--allow-missing-context", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_outcome_preference_edge_margins_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_outcome_preference_edge_margins_parser().parse_args(argv)


__all__ = [
    "build_paired_outcome_preference_edge_margins_parser",
    "parse_paired_outcome_preference_edge_margins_args",
]
