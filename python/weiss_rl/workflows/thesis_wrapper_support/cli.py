from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.thesis_wrapper_support.plan import (
    build_thesis_wrapper_plan_for_request,
    run_thesis_wrapper_plan,
    write_thesis_wrapper_summary,
)
from weiss_rl.workflows.thesis_wrapper_support.presets import _PRESET_PATHS
from weiss_rl.workflows.thesis_wrapper_support.request import thesis_wrapper_request


def build_thesis_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin wrapper for canonical thesis train/eval/compare runs")
    parser.add_argument("--stack-config", type=Path, default=None)
    parser.add_argument("--eval-stack-config", type=Path, default=None)
    parser.add_argument("--preset", choices=tuple(_PRESET_PATHS), default="standard")
    parser.add_argument("--eval-preset", choices=tuple(_PRESET_PATHS), default="")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--unroll-length", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--runtime-mode", type=str, default="train_ordered")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--compare-run-dir", action="append", default=None)
    parser.add_argument("--compare-launch-group-summary", type=Path, default=None)
    parser.add_argument("--compare-out-dir", type=Path, default=None)
    parser.add_argument("--train-arg", action="append", default=None)
    parser.add_argument("--eval-arg", action="append", default=None)
    parser.add_argument("--compare-arg", action="append", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def thesis_wrapper_repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root is None:
        return Path(__file__).resolve().parents[4]
    return Path(args.repo_root).resolve()


def print_thesis_wrapper_presets(*, repo_root: Path) -> None:
    for name, path in _PRESET_PATHS.items():
        print(f"{name}: {(repo_root / path).as_posix()}")


def run_thesis_wrapper_cli(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    repo_root: Path,
    python_exe: str,
) -> int:
    if args.list_presets:
        print_thesis_wrapper_presets(repo_root=repo_root)
        return 0
    if not str(args.run_label).strip():
        parser.error("--run-label is required unless --list-presets is used")
    request = thesis_wrapper_request(args=args, repo_root=repo_root, python_exe=python_exe)
    plan = build_thesis_wrapper_plan_for_request(request)
    result = run_thesis_wrapper_plan(plan)
    summary_path = write_thesis_wrapper_summary(result)
    print(f"Wrote thesis wrapper summary: {summary_path}")
    return 1 if result.failed else 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_thesis_wrapper_parser()
    args = parser.parse_args(argv)
    status = run_thesis_wrapper_cli(
        args=args,
        parser=parser,
        repo_root=thesis_wrapper_repo_root(args),
        python_exe=sys.executable,
    )
    if status:
        raise SystemExit(status)


__all__ = [
    "build_thesis_wrapper_parser",
    "main",
    "print_thesis_wrapper_presets",
    "run_thesis_wrapper_cli",
    "thesis_wrapper_repo_root",
]
