from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.parser import add_evaluation_parsers
from weiss_rl.workflows.training_workflow.parser import add_training_parsers


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print and save the command without executing it")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small thesis workflow command surface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_training_parsers(subparsers, _add_common)
    add_evaluation_parsers(subparsers, _add_common)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
