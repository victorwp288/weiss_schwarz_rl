"""Runtime orchestration for paired-outcome overlap reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_overlap_report import (
    PairedOutcomeOverlapReportConfig,
    build_paired_outcome_overlap_report,
    write_paired_outcome_overlap_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomeOverlapReportRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_overlap_report(args: argparse.Namespace) -> PairedOutcomeOverlapReportRunResult:
    report = build_paired_outcome_overlap_report(paired_outcome_overlap_report_config_from_args(args))
    write_paired_outcome_overlap_report(args.output_json, report)
    return PairedOutcomeOverlapReportRunResult(output_json=args.output_json, report=report)


def paired_outcome_overlap_report_config_from_args(args: argparse.Namespace) -> PairedOutcomeOverlapReportConfig:
    return PairedOutcomeOverlapReportConfig(
        compare_json_paths=tuple(Path(path) for path in args.compare_json),
        max_examples_per_key=int(args.max_examples_per_key),
    )


__all__ = [
    "PairedOutcomeOverlapReportRunResult",
    "paired_outcome_overlap_report_config_from_args",
    "run_paired_outcome_overlap_report",
]
