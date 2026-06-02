"""CLI parser and validation for paired-swing warmstarts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_paired_swing_warmstart_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply paired-swing replay as an auxiliary-only warmstart checkpoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--coef", type=float, default=0.08)
    parser.add_argument("--positive-action-source", choices=("actions", "teacher_action"), default="actions")
    parser.add_argument("--negative-action-source", choices=("actions", "teacher_action"), default="teacher_action")
    parser.add_argument("--loss-scope", choices=("row", "episode_mean", "label_mean"), default="row")
    parser.add_argument("--compare-to", choices=("negative", "top_other"), default="negative")
    parser.add_argument("--margin-retention-coef", type=float, default=0.0)
    parser.add_argument("--margin-retention-margin", type=float, default=0.0)
    parser.add_argument("--top-action-retention-coef", type=float, default=0.0)
    parser.add_argument("--top-action-retention-margin", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-dataset", type=Path, default=None)
    parser.add_argument("--full-surface-retention-coef", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-margin", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-batch-episodes", type=int, default=0)
    parser.add_argument(
        "--full-surface-retention-mode",
        choices=("reference_top", "target_action"),
        default="reference_top",
    )
    parser.add_argument("--conflict-filter", choices=("none", "current_state", "history"), default="none")
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def validate_paired_swing_warmstart_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.batch_episodes) <= 0:
        parser.error("--batch-episodes must be positive")
    if float(args.margin) < 0.0:
        parser.error("--margin must be nonnegative")
    if float(args.coef) < 0.0:
        parser.error("--coef must be nonnegative")
    if float(args.margin_retention_coef) < 0.0:
        parser.error("--margin-retention-coef must be nonnegative")
    if float(args.margin_retention_margin) < 0.0:
        parser.error("--margin-retention-margin must be nonnegative")
    if float(args.top_action_retention_coef) < 0.0:
        parser.error("--top-action-retention-coef must be nonnegative")
    if float(args.top_action_retention_margin) < 0.0:
        parser.error("--top-action-retention-margin must be nonnegative")
    if float(args.full_surface_retention_coef) < 0.0:
        parser.error("--full-surface-retention-coef must be nonnegative")
    if float(args.full_surface_retention_margin) < 0.0:
        parser.error("--full-surface-retention-margin must be nonnegative")
    if int(args.full_surface_retention_batch_episodes) < 0:
        parser.error("--full-surface-retention-batch-episodes must be nonnegative")
    if float(args.full_surface_retention_coef) != 0.0 and args.full_surface_retention_dataset is None:
        parser.error("--full-surface-retention-dataset is required when --full-surface-retention-coef is nonzero")
    if str(args.positive_action_source) == str(args.negative_action_source):
        parser.error("--positive-action-source and --negative-action-source must differ")


def parse_paired_swing_warmstart_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_paired_swing_warmstart_parser()
    args = parser.parse_args(argv)
    validate_paired_swing_warmstart_args(parser, args)
    return args


__all__ = [
    "build_paired_swing_warmstart_parser",
    "parse_paired_swing_warmstart_args",
    "validate_paired_swing_warmstart_args",
]
