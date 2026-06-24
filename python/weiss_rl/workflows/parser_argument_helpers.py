"""Shared argparse helpers for the public thesis workflow commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.command_help import public_workflow_command_epilog
from weiss_rl.workflows.command_surface import PublicWorkflowCommand


def add_public_workflow_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    command: PublicWorkflowCommand,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        command.name,
        help=command.help,
        description=command.description,
        epilog=public_workflow_command_epilog(command),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_run_label_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-label", required=True, help="Human-readable run label used for the output run")


def add_run_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory used by this workflow command")


def add_b1_anchor_argument(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--b1-run",
        "--b1-baseline-run-dir",
        dest="b1_baseline_run_dir",
        type=Path,
        required=required,
        default=None if not required else argparse.SUPPRESS,
        help="Selected B1 no-league baseline run directory",
    )


def add_training_profile_argument(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str],
    default: str = "smoke",
) -> None:
    parser.add_argument(
        "--profile",
        choices=tuple(choices),
        default=default,
        help="Named workflow profile controlling config, runtime size, and command defaults",
    )


__all__ = [
    "add_public_workflow_parser",
    "add_b1_anchor_argument",
    "add_run_dir_argument",
    "add_run_label_argument",
    "add_training_profile_argument",
]
