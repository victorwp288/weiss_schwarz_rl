from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.profiles import (
    MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG,
    MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG,
    TRAIN_PROFILES,
)

__all__ = ["build_workflow_parser", "parse_workflow_args"]


def _add_common_workflow_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print and save the command without executing it")


def build_workflow_parser() -> argparse.ArgumentParser:
    """Build the canonical thesis workflow command parser."""

    parser = argparse.ArgumentParser(description="Small thesis workflow command surface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_b1 = subparsers.add_parser("train-b1", help="Train the B1 NoLeague baseline")
    _add_common_workflow_options(train_b1)
    train_b1.add_argument("--run-label", required=True)
    train_b1.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_b1_guided = subparsers.add_parser(
        "train-b1-guided-seed",
        help="Train the guided B1-derived seed policy used for league bootstrap ablations",
    )
    _add_common_workflow_options(train_b1_guided)
    train_b1_guided.add_argument("--run-label", required=True)
    train_b1_guided.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_main = subparsers.add_parser("train-main", help="Train the main league thesis model")
    _add_common_workflow_options(train_main)
    train_main.add_argument("--run-label", required=True)
    train_main.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)
    train_main.add_argument("--seed-run", "--seed-snapshot-run-dir", dest="seed_snapshot_run_dir", type=Path)
    train_main.add_argument(
        "--init-policy-id",
        default="auto",
        help=(
            "B1 snapshot policy id used to initialize the main learner. "
            "Default auto tries canonical B1 aliases, then selected_candidate."
        ),
    )
    train_main.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_main_guided = subparsers.add_parser(
        "train-main-guided-bootstrap",
        help="Train the guarded guided-bootstrap league path from a confirmed seed checkpoint",
    )
    _add_common_workflow_options(train_main_guided)
    train_main_guided.add_argument("--run-label", required=True)
    train_main_guided.add_argument("--init-from-checkpoint", type=Path, default=None)
    train_main_guided.add_argument(
        "--init-from-run-dir",
        type=Path,
        default=None,
        help="Run directory whose snapshot registry should be used to resolve --init-policy-id.",
    )
    train_main_guided.add_argument(
        "--init-policy-id",
        type=str,
        default="",
        help="Snapshot policy id to initialize from, resolved to training/checkpoints/checkpoint_<update>.pt.",
    )
    train_main_guided.add_argument(
        "--seed-run",
        "--seed-snapshot-run-dir",
        dest="seed_snapshot_run_dir",
        type=Path,
        required=True,
    )
    train_main_guided.add_argument(
        "--b1-run",
        "--b1-baseline-run-dir",
        dest="b1_baseline_run_dir",
        type=Path,
        default=None,
        help="Optional strict B1 anchor; omitted for the current guided-bootstrap path.",
    )
    train_main_guided.add_argument(
        "--vtrace-clamp",
        action="store_true",
        help="Use the conservative V-trace-clipped guided-bootstrap stack.",
    )
    train_main_guided.add_argument(
        "--seed-champions",
        action="store_true",
        help="Treat imported seed snapshots as training-pool champions. This does not mark the run as thesis-promoted.",
    )
    train_main_guided.add_argument(
        "--selected-seed-champion",
        action="store_true",
        help=(
            "Use the selected guided-bootstrap stack, where only pinned snapshots in --seed-run "
            "are imported as training-pool champions."
        ),
    )
    train_main_guided.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    smoke_eval = subparsers.add_parser("smoke-eval", help="Run a tiny deterministic eval on a run directory")
    _add_common_workflow_options(smoke_eval)
    smoke_eval.add_argument("--run-dir", type=Path, required=True)
    smoke_eval.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, default=None)

    thesis_eval = subparsers.add_parser("eval-final", help="Run the thesis-grade final evaluation")
    _add_common_workflow_options(thesis_eval)
    thesis_eval.add_argument("--run-dir", type=Path, required=True)
    thesis_eval.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)

    figures = subparsers.add_parser("figures", help="Export paper figures and tables for a run")
    _add_common_workflow_options(figures)
    figures.add_argument("--run-dir", type=Path, required=True)
    figures.add_argument("--fig-id", type=str, default="")
    figures.add_argument("--format", dest="formats", action="append", default=None)

    b2_audit = subparsers.add_parser("b2-audit", help="Run the standard learner-vs-B2 disagreement audit")
    _add_common_workflow_options(b2_audit)
    b2_audit.add_argument("--run-dir", type=Path, required=True)
    b2_audit.add_argument("--episodes-jsonl", type=Path, required=True)
    b2_audit.add_argument("--policy-id", required=True)
    b2_audit.add_argument("--output-run-dir", type=Path, default=None)
    b2_audit.add_argument("--snapshot-registry-json", type=Path, default=None)
    b2_audit.add_argument("--summary-json", type=Path, default=None)
    b2_audit.add_argument("--top-k", type=int, default=25)
    b2_audit.add_argument("--top-actions", type=int, default=5)
    b2_audit.add_argument("--allow-policy-id-mismatch", action="store_true")
    b2_audit.add_argument("--accept-snapshot-config-hash", action="append", default=[])

    guard_run = subparsers.add_parser("guard-run", help="Fail fast on unhealthy B1/main league probe artifacts")
    _add_common_workflow_options(guard_run)
    guard_run.add_argument("--run-dir", type=Path, required=True)
    guard_run.add_argument(
        "--required-anchor",
        action="append",
        default=None,
        help="Anchor that must remain above --min-latest-anchor-score; defaults to B2/B3/B4.",
    )
    guard_run.add_argument("--min-latest-anchor-score", type=float, default=0.45)
    guard_run.add_argument("--max-latest-drop", type=float, default=0.05)
    guard_run.add_argument("--require-promotion-pass-after-attempts", type=int, default=3)
    guard_run.add_argument("--max-consecutive-promotion-failures", type=int, default=3)
    guard_run.add_argument("--max-vtrace-rho-p99", type=float, default=None)

    guided_loop = subparsers.add_parser(
        "guided-bootstrap-loop",
        help="Run segmented guided-bootstrap continuation with automatic confirm/select/reanchor decisions",
    )
    _add_common_workflow_options(guided_loop)
    guided_loop.add_argument("--initial-run-dir", type=Path, required=True)
    guided_loop.add_argument("--initial-policy-id", default="guided_bootstrap_floor_selected")
    guided_loop.add_argument("--seed-run-dir", type=Path, default=None)
    guided_loop.add_argument("--run-prefix", default="b1_guided_floor_segmented")
    guided_loop.add_argument(
        "--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG
    )
    guided_loop.add_argument("--alias-policy-id", default="guided_bootstrap_floor_segmented_selected")
    guided_loop.add_argument("--segments", type=int, default=4)
    guided_loop.add_argument("--segment-updates", type=int, default=25)
    guided_loop.add_argument("--confirm-paired-seeds", type=int, default=64)
    guided_loop.add_argument("--stop-on-latest-falloff", action="store_true")

    guarded_league = subparsers.add_parser(
        "guarded-league-bootstrap",
        help="Run short guided-league segments, confirming B2/B3/B4 before advancing the selected checkpoint",
    )
    _add_common_workflow_options(guarded_league)
    guarded_league.add_argument("--init-from-checkpoint", type=Path, required=True)
    guarded_league.add_argument("--seed-snapshot-run-dir", type=Path, required=True)
    guarded_league.add_argument("--run-prefix", default="guarded_league_bootstrap")
    guarded_league.add_argument("--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG)
    guarded_league.add_argument("--segments", type=int, default=4)
    guarded_league.add_argument("--segment-updates", type=int, default=10)
    guarded_league.add_argument("--first-init-schedule-offset-updates", type=int, default=None)
    guarded_league.add_argument("--confirm-paired-seeds", type=int, default=64)
    guarded_league.add_argument("--publish-min-confirm-paired-seeds", type=int, default=256)
    guarded_league.add_argument(
        "--confirm-recent-candidate-count",
        type=int,
        default=1,
        help="Confirm this many recent train snapshots per segment before selecting the best confirmed checkpoint.",
    )
    guarded_league.add_argument("--reference-summary-json", type=Path, default=None)
    guarded_league.add_argument("--multiobjective-reference-summary-json", action="append", type=Path, default=[])
    guarded_league.add_argument("--multiobjective-fixed-opponent", action="append", default=[])
    guarded_league.add_argument("--learned-guard-opponent", action="append", default=[])
    guarded_league.add_argument("--min-learned-guard-mean", type=float, default=0.5)
    guarded_league.add_argument("--min-learned-guard-reference-delta", type=float, default=0.0)
    guarded_league.add_argument("--reference-label", default="reference")
    guarded_league.add_argument("--min-required-anchor-score", type=float, default=0.5)
    guarded_league.add_argument("--max-reference-drop", type=float, default=0.04)
    guarded_league.add_argument("--selected-alias-policy-id", default="main_league_selected")

    return parser


def parse_workflow_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse package CLI arguments for the canonical workflow entrypoint."""

    return build_workflow_parser().parse_args(argv)
