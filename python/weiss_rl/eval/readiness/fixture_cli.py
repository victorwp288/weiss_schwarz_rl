"""Parser and runtime helpers for the paper-readiness fixture CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from weiss_rl.eval.readiness.fixture_writer import write_paper_readiness_run_fixture

WritePaperReadinessRunFixtureFn = Callable[[Path], Path]


@dataclass(frozen=True, slots=True)
class PaperReadinessFixtureResult:
    run_dir: Path


def build_paper_readiness_fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a minimal thesis-grade run tree for readiness checks")
    parser.add_argument("--run-dir", type=Path, required=True, help="Destination run directory")
    return parser


def run_paper_readiness_fixture_command(
    args: argparse.Namespace,
    *,
    write_paper_readiness_run_fixture_fn: WritePaperReadinessRunFixtureFn = write_paper_readiness_run_fixture,
) -> PaperReadinessFixtureResult:
    run_dir = write_paper_readiness_run_fixture_fn(args.run_dir)
    return PaperReadinessFixtureResult(run_dir=run_dir)
