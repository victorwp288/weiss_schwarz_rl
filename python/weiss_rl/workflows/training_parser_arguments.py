from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.training_commands import TRAIN_PROFILES


def add_train_b1_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train-b1", help="Train the B1 NoLeague baseline")
    add_common(parser)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")
    return parser


def add_train_b1_guided_seed_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "train-b1-guided-seed",
        help="Train the guided B1-derived seed policy used for league bootstrap ablations",
    )
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


def add_train_main_guided_bootstrap_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "train-main-guided-bootstrap",
        help="Train the guarded guided-bootstrap league path from a confirmed seed checkpoint",
    )
    add_common(parser)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--init-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-from-run-dir",
        type=Path,
        default=None,
        help="Run directory whose snapshot registry should be used to resolve --init-policy-id.",
    )
    parser.add_argument(
        "--init-policy-id",
        type=str,
        default="",
        help="Snapshot policy id to initialize from, resolved to training/checkpoints/checkpoint_<update>.pt.",
    )
    parser.add_argument(
        "--seed-run",
        "--seed-snapshot-run-dir",
        dest="seed_snapshot_run_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--b1-run",
        "--b1-baseline-run-dir",
        dest="b1_baseline_run_dir",
        type=Path,
        default=None,
        help="Optional strict B1 anchor; omitted for the current guided-bootstrap path.",
    )
    parser.add_argument(
        "--vtrace-clamp",
        action="store_true",
        help="Use the conservative V-trace-clipped guided-bootstrap stack.",
    )
    parser.add_argument(
        "--seed-champions",
        action="store_true",
        help=(
            "Treat imported seed snapshots as training-pool champions. This does not mark the run as thesis-promoted."
        ),
    )
    parser.add_argument(
        "--selected-seed-champion",
        action="store_true",
        help=(
            "Use the selected guided-bootstrap stack, where only pinned snapshots in --seed-run "
            "are imported as training-pool champions."
        ),
    )
    parser.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")
    return parser


__all__ = [
    "add_train_b1_guided_seed_parser",
    "add_train_b1_parser",
    "add_train_main_guided_bootstrap_parser",
    "add_train_main_parser",
]
