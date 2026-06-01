"""Runtime orchestration for paired-swing preference conflict reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_swing_conflicts import (
    PairedSwingConflictConfig,
    build_paired_swing_conflict_report,
    write_paired_swing_conflict_report,
)


@dataclass(frozen=True, slots=True)
class PairedSwingConflictReportRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_swing_conflict_report(args: argparse.Namespace) -> PairedSwingConflictReportRunResult:
    report = build_paired_swing_conflict_report(paired_swing_conflict_report_config_from_args(args))
    write_paired_swing_conflict_report(args.output_json, report)
    return PairedSwingConflictReportRunResult(output_json=args.output_json, report=report)


def paired_swing_conflict_report_config_from_args(args: argparse.Namespace) -> PairedSwingConflictConfig:
    return PairedSwingConflictConfig(
        dataset_paths=tuple(Path(path) for path in args.dataset),
        positive_action_source=str(args.positive_action_source),
        negative_action_source=str(args.negative_action_source),
        max_examples=int(args.max_examples),
    )


__all__ = [
    "PairedSwingConflictReportRunResult",
    "paired_swing_conflict_report_config_from_args",
    "run_paired_swing_conflict_report",
]
