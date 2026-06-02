"""CLI parser and validation for trajectory behavior-cloning warmstarts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_trajectory_bc_warmstart_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply targeted replay trajectory behavior cloning as an auxiliary warmstart checkpoint"
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Training stack config")
    parser.add_argument("--dataset", type=Path, required=True, help="Replay trajectory BC .npz dataset")
    parser.add_argument("--init-from-checkpoint", type=Path, required=True, help="Checkpoint to warmstart from")
    parser.add_argument(
        "--output-run-dir", type=Path, required=True, help="Run directory for the warmstarted checkpoint"
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--epochs", type=int, default=2, help="Dataset passes")
    parser.add_argument("--batch-episodes", type=int, default=8, help="Episode columns per auxiliary update")
    parser.add_argument("--seed", type=int, default=20260516, help="Dataset shuffle seed")
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Force mixed precision for the auxiliary updates"
    )
    parser.add_argument("--teacher-family-coef", type=float, default=0.05)
    parser.add_argument("--teacher-slot-coef", type=float, default=0.05)
    parser.add_argument("--teacher-move-source-coef", type=float, default=0.02)
    parser.add_argument("--teacher-attack-type-coef", type=float, default=0.02)
    parser.add_argument("--teacher-action-coef", type=float, default=0.20)
    parser.add_argument("--teacher-same-family-action-coef", type=float, default=0.60)
    parser.add_argument("--teacher-action-margin-coef", type=float, default=0.0)
    parser.add_argument("--teacher-action-margin", type=float, default=0.5)
    parser.add_argument("--teacher-same-family-action-margin-coef", type=float, default=0.10)
    parser.add_argument("--teacher-same-family-action-margin", type=float, default=0.5)
    parser.add_argument(
        "--exact-action-family",
        action="append",
        default=None,
        help="Restrict exact-action auxiliary losses to this family. Repeat for multiple.",
    )
    return parser


def validate_trajectory_bc_warmstart_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.batch_episodes) <= 0:
        parser.error("--batch-episodes must be positive")


def parse_trajectory_bc_warmstart_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_trajectory_bc_warmstart_parser()
    args = parser.parse_args(argv)
    validate_trajectory_bc_warmstart_args(parser, args)
    return args


__all__ = [
    "build_trajectory_bc_warmstart_parser",
    "parse_trajectory_bc_warmstart_args",
    "validate_trajectory_bc_warmstart_args",
]
