from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.command_surface import (
    B2_AUDIT_COMMAND,
    EVAL_FINAL_COMMAND,
    FIGURES_COMMAND,
    SMOKE_EVAL_COMMAND,
)
from weiss_rl.workflows.parser_argument_helpers import (
    add_b1_anchor_argument,
    add_public_workflow_parser,
    add_run_dir_argument,
)


def add_smoke_eval_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, SMOKE_EVAL_COMMAND)
    add_common(parser)
    add_run_dir_argument(parser)
    add_b1_anchor_argument(parser, required=False)
    return parser


def add_eval_final_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, EVAL_FINAL_COMMAND)
    add_common(parser)
    add_run_dir_argument(parser)
    add_b1_anchor_argument(parser, required=True)
    return parser


def add_figures_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, FIGURES_COMMAND)
    add_common(parser)
    add_run_dir_argument(parser)
    parser.add_argument("--fig-id", type=str, default="")
    parser.add_argument("--format", dest="formats", action="append", default=None)
    return parser


def add_b2_audit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, B2_AUDIT_COMMAND)
    add_common(parser)
    add_run_dir_argument(parser)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--output-run-dir", type=Path, default=None)
    parser.add_argument("--snapshot-registry-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--top-actions", type=int, default=5)
    parser.add_argument("--allow-policy-id-mismatch", action="store_true")
    parser.add_argument("--accept-snapshot-config-hash", action="append", default=[])
    return parser


__all__ = [
    "add_b2_audit_parser",
    "add_eval_final_parser",
    "add_figures_parser",
    "add_smoke_eval_parser",
]
