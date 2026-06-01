from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.workflows.training_commands import (
    MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG,
)


def add_guard_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("guard-run", help="Fail fast on unhealthy B1/main league probe artifacts")
    add_common(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--required-anchor",
        action="append",
        default=None,
        help="Anchor that must remain above --min-latest-anchor-score; defaults to B2/B3/B4.",
    )
    parser.add_argument("--min-latest-anchor-score", type=float, default=0.45)
    parser.add_argument("--max-latest-drop", type=float, default=0.05)
    parser.add_argument("--require-promotion-pass-after-attempts", type=int, default=3)
    parser.add_argument("--max-consecutive-promotion-failures", type=int, default=3)
    parser.add_argument("--max-vtrace-rho-p99", type=float, default=None)
    return parser


def add_guided_bootstrap_loop_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "guided-bootstrap-loop",
        help="Run segmented guided-bootstrap continuation with automatic confirm/select/reanchor decisions",
    )
    add_common(parser)
    parser.add_argument("--initial-run-dir", type=Path, required=True)
    parser.add_argument("--initial-policy-id", default="guided_bootstrap_floor_selected")
    parser.add_argument("--seed-run-dir", type=Path, default=None)
    parser.add_argument("--run-prefix", default="b1_guided_floor_segmented")
    parser.add_argument("--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG)
    parser.add_argument("--alias-policy-id", default="guided_bootstrap_floor_segmented_selected")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment-updates", type=int, default=25)
    parser.add_argument("--confirm-paired-seeds", type=int, default=64)
    parser.add_argument("--stop-on-latest-falloff", action="store_true")
    return parser


def add_guarded_league_bootstrap_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "guarded-league-bootstrap",
        help="Run short guided-league segments, confirming B2/B3/B4 before advancing the selected checkpoint",
    )
    add_common(parser)
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-snapshot-run-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="guarded_league_bootstrap")
    parser.add_argument("--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment-updates", type=int, default=10)
    parser.add_argument("--first-init-schedule-offset-updates", type=int, default=None)
    parser.add_argument("--confirm-paired-seeds", type=int, default=64)
    parser.add_argument("--publish-min-confirm-paired-seeds", type=int, default=256)
    parser.add_argument(
        "--confirm-recent-candidate-count",
        type=int,
        default=1,
        help="Confirm this many recent train snapshots per segment before selecting the best confirmed checkpoint.",
    )
    parser.add_argument("--reference-summary-json", type=Path, default=None)
    parser.add_argument("--multiobjective-reference-summary-json", action="append", type=Path, default=[])
    parser.add_argument("--multiobjective-fixed-opponent", action="append", default=[])
    parser.add_argument("--learned-guard-opponent", action="append", default=[])
    parser.add_argument("--min-learned-guard-mean", type=float, default=0.5)
    parser.add_argument("--min-learned-guard-reference-delta", type=float, default=0.0)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--min-required-anchor-score", type=float, default=0.5)
    parser.add_argument("--max-reference-drop", type=float, default=0.04)
    parser.add_argument("--selected-alias-policy-id", default="main_league_selected")
    return parser


__all__ = [
    "add_guard_run_parser",
    "add_guarded_league_bootstrap_parser",
    "add_guided_bootstrap_loop_parser",
]
