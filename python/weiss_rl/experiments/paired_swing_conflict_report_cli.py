"""CLI parser for paired-swing preference conflict reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_swing_conflict_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect contradictory paired-swing preferences across replay datasets."
    )
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--positive-action-source", default="actions")
    parser.add_argument("--negative-action-source", default="teacher_action")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def parse_paired_swing_conflict_report_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_paired_swing_conflict_report_parser().parse_args(argv)


__all__ = [
    "build_paired_swing_conflict_report_parser",
    "parse_paired_swing_conflict_report_args",
]
