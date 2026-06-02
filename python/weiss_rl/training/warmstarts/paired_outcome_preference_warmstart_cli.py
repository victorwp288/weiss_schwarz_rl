"""CLI parser and validation for paired-outcome preference warmstarts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_support import (
    _parse_pair_role_selectors,
    _parse_pair_weights,
)


def build_paired_outcome_preference_warmstart_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply paired-outcome preference replay as an auxiliary-only warmstart checkpoint"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--coef", type=float, default=0.08)
    parser.add_argument(
        "--optimizer-lr-scale",
        type=float,
        default=1.0,
        help="Multiply optimizer learning rates for this auxiliary-only warmstart run.",
    )
    parser.add_argument("--aggregation", choices=("mean", "sum", "edge_mean"), default="mean")
    parser.add_argument("--group-balance", action="store_true")
    parser.add_argument(
        "--pair-weight",
        action="append",
        default=[],
        metavar="PAIR_ID=WEIGHT",
        help=(
            "Upweight a specific preference pair id during auxiliary replay. "
            "May be repeated, for example --pair-weight 9=8.0."
        ),
    )
    parser.add_argument("--target-logp-retention-coef", type=float, default=0.0)
    parser.add_argument("--target-logp-retention-margin", type=float, default=0.0)
    parser.add_argument(
        "--target-logp-retention-role",
        choices=("all", "preferred", "rejected"),
        default="preferred",
    )
    parser.add_argument(
        "--target-logp-retention-reference-top-only",
        action="store_true",
        help="Apply target-logp retention only on rows where the reference policy ranked the replay action top.",
    )
    parser.add_argument(
        "--target-logp-retention-pair-role",
        action="append",
        default=[],
        metavar="PAIR_ID:ROLE",
        help=(
            "Scope target-logp retention to a preference pair and role. "
            "ROLE is preferred, rejected, or all. May be repeated."
        ),
    )
    parser.add_argument("--top-action-retention-coef", type=float, default=0.0)
    parser.add_argument("--top-action-retention-margin", type=float, default=0.0)
    parser.add_argument(
        "--top-action-retention-role",
        choices=("all", "preferred", "rejected"),
        default="all",
    )
    parser.add_argument(
        "--top-action-retention-reference-top-only",
        action="store_true",
        help="Apply top-action retention only on rows where the reference policy ranked the replay action top.",
    )
    parser.add_argument(
        "--top-action-retention-pair-role",
        action="append",
        default=[],
        metavar="PAIR_ID:ROLE",
        help=(
            "Scope top-action retention to a preference pair and role. "
            "ROLE is preferred, rejected, or all. May be repeated."
        ),
    )
    return parser


def validate_paired_outcome_preference_warmstart_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.batch_episodes) <= 0:
        parser.error("--batch-episodes must be positive")
    if float(args.beta) <= 0.0:
        parser.error("--beta must be positive")
    if float(args.coef) < 0.0:
        parser.error("--coef must be nonnegative")
    if float(args.optimizer_lr_scale) <= 0.0:
        parser.error("--optimizer-lr-scale must be positive")
    if float(args.target_logp_retention_coef) < 0.0:
        parser.error("--target-logp-retention-coef must be nonnegative")
    if float(args.target_logp_retention_margin) < 0.0:
        parser.error("--target-logp-retention-margin must be nonnegative")
    if float(args.top_action_retention_coef) < 0.0:
        parser.error("--top-action-retention-coef must be nonnegative")
    if float(args.top_action_retention_margin) < 0.0:
        parser.error("--top-action-retention-margin must be nonnegative")
    try:
        _parse_pair_weights(args.pair_weight)
        _parse_pair_role_selectors(args.target_logp_retention_pair_role)
        _parse_pair_role_selectors(args.top_action_retention_pair_role)
    except ValueError as exc:
        parser.error(str(exc))


def parse_paired_outcome_preference_warmstart_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_paired_outcome_preference_warmstart_parser()
    args = parser.parse_args(argv)
    validate_paired_outcome_preference_warmstart_args(parser, args)
    return args


__all__ = [
    "build_paired_outcome_preference_warmstart_parser",
    "parse_paired_outcome_preference_warmstart_args",
    "validate_paired_outcome_preference_warmstart_args",
]
