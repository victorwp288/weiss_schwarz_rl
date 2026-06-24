from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.command_surface import TRAIN_B1_COMMAND, TRAIN_MAIN_COMMAND
from weiss_rl.workflows.parser_argument_helpers import (
    add_b1_anchor_argument,
    add_public_workflow_parser,
    add_run_label_argument,
    add_training_profile_argument,
)
from weiss_rl.workflows.training_workflow.commands import TRAIN_PROFILES


def add_train_b1_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, TRAIN_B1_COMMAND)
    add_common(parser)
    add_run_label_argument(parser)
    add_training_profile_argument(parser, choices=tuple(TRAIN_PROFILES))
    return parser


def add_train_main_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = add_public_workflow_parser(subparsers, TRAIN_MAIN_COMMAND)
    add_common(parser)
    add_run_label_argument(parser)
    add_b1_anchor_argument(parser, required=True)
    parser.add_argument("--seed-run", "--seed-snapshot-run-dir", dest="seed_snapshot_run_dir", type=Path)
    parser.add_argument(
        "--init-policy-id",
        default="auto",
        help=(
            "B1 snapshot policy id used to initialize the main learner. "
            "Default auto tries selected_candidate, then canonical B1 aliases."
        ),
    )
    add_training_profile_argument(parser, choices=tuple(TRAIN_PROFILES))
    return parser


__all__ = [
    "add_train_b1_parser",
    "add_train_main_parser",
]
