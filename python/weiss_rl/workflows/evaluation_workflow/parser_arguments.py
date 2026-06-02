from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


def add_smoke_eval_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("smoke-eval", help="Run a tiny deterministic eval on a run directory")
    add_common(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, default=None)
    return parser


def add_eval_final_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("eval-final", help="Run the thesis-grade final evaluation")
    add_common(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)
    return parser


def add_figures_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("figures", help="Export paper figures and tables for a run")
    add_common(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fig-id", type=str, default="")
    parser.add_argument("--format", dest="formats", action="append", default=None)
    return parser


def add_b2_audit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("b2-audit", help="Run the standard learner-vs-B2 disagreement audit")
    add_common(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
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
