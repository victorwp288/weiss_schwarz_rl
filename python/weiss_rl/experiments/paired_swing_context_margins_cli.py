"""CLI parser for paired-swing opponent-context margin reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_swing_context_margins_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit row-level opponent-context log-prob margins for paired-swing replay rows."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--positive-action-source", default="actions")
    parser.add_argument("--negative-action-source", default="teacher_action")
    parser.add_argument("--report-action-id", action="append", default=[104, 124], type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_swing_context_margins_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_swing_context_margins_parser().parse_args(argv)


__all__ = [
    "build_paired_swing_context_margins_parser",
    "parse_paired_swing_context_margins_args",
]
