from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.training_workflow.commands import TRAIN_PROFILES


def add_train_b1_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train-b1", help="Train the B1 NoLeague baseline")
    add_common(parser)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")
    return parser


def add_train_main_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train-main", help="Train the main league thesis model")
    add_common(parser)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)
    parser.add_argument("--seed-run", "--seed-snapshot-run-dir", dest="seed_snapshot_run_dir", type=Path)
    parser.add_argument(
        "--init-policy-id",
        default="auto",
        help=(
            "B1 snapshot policy id used to initialize the main learner. "
            "Default auto tries selected_candidate, then canonical B1 aliases."
        ),
    )
    parser.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")
    return parser


__all__ = [
    "add_train_b1_parser",
    "add_train_main_parser",
]
